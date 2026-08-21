from __future__ import annotations

import argparse
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .adb import (
    adb_capture_bgr, adb_screencap, adb_tap, save_bgr, shot_stamp,
)
from .board import (
    ctag, initial_grid, learn_palette, reachable_summary, update_grid,
)
from .config import FRONT_Y_N
from .display import ClickMark, SolverDisplay
from .models import AnalysisResult, Candidate, FrontNumberCacheEntry, TwoStepPlan
from .monitor import wait_for_parking_idle
from . import ocr as _ocr
from .game_ocr import (
    install_game_digit_ocr,
    read_number_detailed_at,
)
install_game_digit_ocr(_ocr)
from .ocr import read_number_at
from .state import load_state, save_state
from .strategy import (
    best_valid_candidate, choose_two_step_plan, evaluate_candidates,
    format_report, format_two_step_plan,
)
from .unlock import unlock_sixth_slot_at_game_start
from . import vehicles as _vehicles
install_game_digit_ocr(_ocr, _vehicles)
from .vehicles import (
    _front_number_fingerprint, car_color_at, detect_front_and_next,
    detect_front_centers, detect_parked, extend_palette_from_front_numbers,
    parking_roi, read_front_numbers_at_centers,
)


def _resolve_decision_log_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "decision_log", None)
    if path is not None:
        return Path(path)
    return args.shots_dir / "decision_log.txt"


def _append_decision_log(
    log_path: Path,
    *,
    screenshot: Path,
    result: AnalysisResult,
    step_label: str,
    execution: Optional[str] = None,
) -> None:
    """
    追加一条可追踪的决策记录。

    每一步都绑定：
      - 分析截图文件名/路径
      - 当前 turn / step
      - 完整候选与策略依据（result.report）
      - 最终计划
      - 可选的实际执行结果

    使用 append 而不是覆盖，便于同一日志连续追踪多局；每个自动运行会写 SESSION
    分隔，--reset 时额外标记 NEW GAME。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "",
        "=" * 88,
        f"[DECISION] time={timestamp}",
        f"step={step_label}",
        f"turn={result.turn}",
        f"screenshot={screenshot.name}",
        f"screenshot_path={screenshot}",
        f"parking={result.occupied_slots}",
        "-" * 88,
        result.report,
    ]
    if execution:
        lines.extend([
            "-" * 88,
            f"execution={execution}",
        ])
    lines.append("=" * 88)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_execution_update(
    log_path: Path,
    *,
    screenshot: Path,
    step_label: str,
    execution: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[EXECUTION] time={timestamp} step={step_label} "
            f"screenshot={screenshot.name} {execution}\n"
        )


def _append_session_marker(
    log_path: Path,
    *,
    reset: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "#" * 88 + "\n")
        f.write(f"[SESSION] start={timestamp}\n")
        if reset:
            f.write("[NEW GAME] --reset 已启用，本次从新局状态开始\n")
        f.write("#" * 88 + "\n")



def _stable_state_conflicts(
    grid: np.ndarray,
    parked,
) -> Dict[int, Tuple[int, Tuple[int, ...]]]:
    """
    稳定截图中的硬一致性断言。

    游戏规则已经确认：
      - 车辆到停车位后才开始吸收；
      - 停车车会持续自动吸收当前可达的同色块；
      - 监控结束后才进入下一轮分析。

    因此在一张“稳定分析截图”里，如果程序仍认为某停车颜色有
    reachable > 0，则说明至少有一项状态模型不可信：
      - 棋盘拓扑/持久化 grid；
      - 颜色归类；
      - 停车车颜色；
      - reachable 定义。

    这种矛盾不能继续被 flow closure 当成 guaranteed safety。
    """
    reachable, _neighbors = reachable_summary(grid)

    remains_by_color: Dict[int, list[int]] = {}
    for car in parked:
        if car.color is None or car.remain is None:
            continue
        remains_by_color.setdefault(int(car.color), []).append(int(car.remain))

    conflicts: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
    for color, remains in remains_by_color.items():
        supply = int(reachable.get(color, 0))
        if supply > 0:
            conflicts[color] = (
                supply,
                tuple(sorted(remains)),
            )
    return conflicts


def _format_palette_diagnostics(palette: np.ndarray) -> str:
    """
    记录最接近的 palette 色对，后续可直接追踪“相近色是否被错误合并”。

    这里只做诊断，不擅自改变分类阈值。
    """
    if palette is None or len(palette) < 2:
        return ""

    pairs = []
    for i in range(len(palette)):
        for j in range(i + 1, len(palette)):
            dist = float(np.linalg.norm(palette[i] - palette[j]))
            pairs.append((dist, i + 1, j + 1))

    pairs.sort()
    nearest = pairs[: min(6, len(pairs))]
    return "；".join(
        f"{ctag(a)}-{ctag(b)} RGB距离={dist:.1f}"
        for dist, a, b in nearest
    )

def analyze_image(
    image_path: Path,
    state_path: Path,
    reset: bool,
    slots: int,
    front_number_cache: Optional[Dict[int, FrontNumberCacheEntry]] = None,
) -> AnalysisResult:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    front_centers = detect_front_centers(image_bgr)

    # v5.4：先关闭跨轮数字复用。
    #
    # 已经在真实截图中确认过：
    #   红29 离开 -> 绿30 顶上
    # 旧缓存有机会因为数字区域 fingerprint 变化不足而继续复用 29。
    #
    # 每轮只有 3~5 辆第一排车，完整 OCR 的成本远低于错误容量带来的策略风险；
    # 等“车辆身份 + 颜色签名”缓存重新设计完成后再考虑恢复。
    del front_number_cache
    front_numbers = read_front_numbers_at_centers(
        image_bgr,
        front_centers,
    )
    new_front_cache = {
        i: FrontNumberCacheEntry(
            front_numbers.get(i),
            _front_number_fingerprint(
                image_bgr,
                float(cx),
                FRONT_Y_N * image_h,
            ),
        )
        for i, cx in enumerate(front_centers, 1)
    }
    front_ocr_reads = len(front_centers)

    removed_since_last: Optional[int]
    new_state = reset or not state_path.exists()

    if new_state:
        palette = learn_palette(image_rgb)
        palette, new_colors_added = extend_palette_from_front_numbers(
            image_rgb, palette, front_numbers, front_centers
        )
        grid = initial_grid(image_rgb, palette)
        turn = 0
        removed_since_last = None
        parking_empty_ref = parking_roi(image_bgr)
    else:
        palette, prev_grid, turn, saved_size, parking_empty_ref = load_state(
            state_path
        )
        if saved_size != (image_w, image_h):
            raise RuntimeError(
                f"截图尺寸从 {saved_size[0]}x{saved_size[1]} 变成 "
                f"{image_w}x{image_h}，请使用 --reset。"
            )
        palette, new_colors_added = extend_palette_from_front_numbers(
            image_rgb, palette, front_numbers, front_centers
        )
        grid, removed_since_last = update_grid(prev_grid, image_rgb, palette)

    front, nxt = detect_front_and_next(
        image_rgb, image_bgr, palette, front_centers, front_numbers
    )

    parked = detect_parked(
        image_rgb, image_bgr, palette, read_numbers=True
    )
    occupied_slots = len(parked)

    if occupied_slots > slots:
        raise RuntimeError(
            f"停车数字锚点检测到 {occupied_slots} 辆，超过设定停车位 {slots}，"
            "为避免误点击已停止。"
        )

    incomplete_parked = [
        c for c in parked if c.color is None or c.remain is None
    ]
    if incomplete_parked:
        raise RuntimeError(
            "停车数字已检测到，但有停车车的颜色/剩余数字无法可靠识别，"
            "自动流程已停止，避免低估同色分流风险。"
        )

    if front and all(c.remain is None for c in front):
        raise RuntimeError(
            "第一排车辆数字全部识别失败。为避免错误容量导致误点击，自动流程已停止。"
        )

    stable_conflicts = _stable_state_conflicts(grid, parked)

    candidates = evaluate_candidates(
        grid, front, nxt, parked, slots, occupied_slots
    )

    two_step_plan = choose_two_step_plan(
        grid,
        front,
        nxt,
        parked,
        slots,
        occupied_slots,
        candidates,
    )

    report = format_report(
        grid, front, nxt, parked, candidates, removed_since_last,
        slots, len(palette), occupied_slots, new_colors_added,
    )

    palette_diag = _format_palette_diagnostics(palette)
    if palette_diag:
        report += "\n\n颜色类别诊断（最近 palette 色对）:\n" + palette_diag

    if stable_conflicts:
        conflict_text = "；".join(
            (
                f"{ctag(color)}: 稳定截图仍判定 reachable={supply}, "
                f"停车剩余={list(remains)}"
            )
            for color, (supply, remains) in sorted(stable_conflicts.items())
        )
        report += (
            "\n\n!!! MODEL_INCONSISTENT / 硬安全暂停 !!!\n"
            "当前截图已经经过停车分流稳定监控，但棋盘模型仍认为停车车存在"
            "可立即吸收的同色 reachable。按已确认游戏规则，这两件事不能同时成立。\n"
            f"矛盾项: {conflict_text}\n"
            "本轮禁止把这些 reachable 用于 guaranteed completion，"
            "也不执行任何自动点击；请优先检查棋盘拓扑/颜色识别。"
        )
        two_step_plan = None
        best = None
    elif two_step_plan is not None:
        report += "\n\n" + format_two_step_plan(two_step_plan)
        best = two_step_plan.first
    else:
        best = best_valid_candidate(candidates)

    turn += 1
    save_state(
        state_path, palette, grid, turn,
        image_w, image_h, parking_empty_ref
    )

    return AnalysisResult(
        report=report,
        palette=palette,
        grid=grid,
        turn=turn,
        best=best,
        image_w=image_w,
        image_h=image_h,
        front=front,
        nxt=nxt,
        parked=parked,
        parking_empty_ref=parking_empty_ref,
        occupied_slots=occupied_slots,
        new_colors_added=new_colors_added,
        front_number_cache=new_front_cache,
        front_ocr_reads=front_ocr_reads,
        two_step_plan=two_step_plan,
    )


def queue_empty_on_image(
    image_bgr: np.ndarray,
    palette: np.ndarray,
) -> Tuple[bool, int]:
    centers = detect_front_centers(image_bgr)
    if not centers:
        return True, 0

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    nums = read_front_numbers_at_centers(image_bgr, centers)
    palette2, added = extend_palette_from_front_numbers(
        image_rgb, palette, nums, centers
    )
    front, _nxt = detect_front_and_next(
        image_rgb, image_bgr, palette2, centers, nums,
        read_next_numbers=False,
    )
    return (len(front) == 0), added


def wait_for_promoted_next_car(
    result: AnalysisResult,
    plan: TwoStepPlan,
    *,
    serial: Optional[str],
    timeout: float,
    poll_interval: float,
    initial_delay: float,
    display: Optional[SolverDisplay] = None,
) -> Tuple[bool, Optional[np.ndarray], int, int]:
    next_car = next(
        (
            c for c in result.nxt
            if c.column == plan.second.column
        ),
        None,
    )
    first_car = next(
        (
            c for c in result.front
            if c.column == plan.first.column
        ),
        None,
    )
    if next_car is None or first_car is None:
        return False, None, 0, 0

    x = int(round(next_car.x))
    y = int(round(FRONT_Y_N * result.image_h))

    if initial_delay > 0:
        time.sleep(initial_delay)

    deadline = time.monotonic() + max(0.0, timeout)
    last_frame: Optional[np.ndarray] = None
    last_num: Optional[int] = None
    last_num_conf = 0.0
    last_num_votes = 0
    last_color: Optional[int] = None
    color_match_streak = 0

    # 若 A/B 颜色不同，A 离开后在同一列连续两帧看到 B 的目标颜色，
    # 已经足以证明 B 补位。此时数字只用于交叉校验，避免旧 OCR 的 0->9 / 1->7
    # 让正确第二步被取消。
    distinct_color_transition = (
        first_car.color is not None
        and plan.second.color is not None
        and first_car.color != plan.second.color
    )

    while True:
        frame = adb_capture_bgr(serial)
        last_frame = frame

        last_num, last_num_conf, last_num_votes = read_number_detailed_at(
            frame,
            float(x),
            float(y),
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        last_color = car_color_at(
            rgb,
            float(x),
            float(y),
            result.palette,
        )

        color_match = last_color == plan.second.color
        number_match = last_num == plan.second.capacity

        if color_match:
            color_match_streak += 1
        else:
            color_match_streak = 0

        exact_match = color_match and number_match
        color_proven_match = (
            distinct_color_transition
            and color_match_streak >= 2
        )

        if exact_match or color_proven_match:
            if exact_match:
                confirm_reason = (
                    "颜色+真实游戏字形数字均与预测一致"
                )
            else:
                confirm_reason = (
                    "A/B 颜色不同，目标颜色连续两帧确认；"
                    f"数字OCR={last_num if last_num is not None else 'UNKNOWN'}"
                    f"（conf={last_num_conf:.2f}, votes={last_num_votes}）仅作交叉校验"
                )

            if display is not None:
                display.show(
                    frame,
                    stage="连续两步 · 同列补位确认",
                    hint=(
                        f"第{plan.second.column}列第二排已顶上："
                        f"{ctag(plan.second.color)}×{plan.second.capacity}；"
                        f"{confirm_reason}，执行第二次点击。"
                    ),
                    marks=(ClickMark(x, y, "2"),),
                )
            return True, frame, x, y

        if time.monotonic() >= deadline:
            if display is not None and last_frame is not None:
                display.show(
                    last_frame,
                    stage="连续两步 · 第二步取消",
                    hint=(
                        f"补位确认超时：期望 {ctag(plan.second.color)}×"
                        f"{plan.second.capacity}，实际颜色="
                        f"{ctag(last_color) if last_color is not None else 'UNKNOWN'}，"
                        f"数字={last_num if last_num is not None else 'UNKNOWN'} "
                        f"(conf={last_num_conf:.2f}, votes={last_num_votes})。"
                        "只保留第一步；第一步自身已通过稳定状态安全检查。"
                    ),
                )
            return False, last_frame, x, y

        if display is not None:
            display.show(
                frame,
                stage="连续两步 · 等待同列补位",
                hint=(
                    f"等待第{plan.second.column}列第二排 "
                    f"{ctag(plan.second.color)}×{plan.second.capacity} "
                    "进入第一排；不同色补位采用连续两帧颜色确认，"
                    "同色补位仍要求数字一致。"
                ),
            )

        time.sleep(max(0.02, poll_interval))

def tap_candidate_from_result(
    result: AnalysisResult,
    *,
    serial: Optional[str],
    candidate: Optional[Candidate] = None,
) -> Tuple[int, int]:
    target = candidate if candidate is not None else result.best
    if target is None:
        raise ValueError("当前没有候选动作")

    car = next(
        (
            c for c in result.front
            if c.column == target.column
        ),
        None,
    )
    if car is None:
        raise RuntimeError(
            f"找不到候选第 {target.column} 列对应的第一排车辆"
        )

    x = int(round(car.x))
    y = int(round(car.y))
    adb_tap(x, y, serial)
    return x, y


def run_manual_step_mode(args: argparse.Namespace) -> int:
    first = True
    reset_next = args.reset
    log_path = _resolve_decision_log_path(args)
    _append_session_marker(log_path, reset=args.reset)

    print("人工逐步调试模式。")
    print(f"决策日志: {log_path}")

    manual_no = 0
    while True:
        cmd = input(
            "\nEnter截图分析；q退出；r将下一张作为新局重建状态: "
        ).strip().lower()
        if cmd == "q":
            return 0
        if cmd == "r":
            reset_next = True

        manual_no += 1
        shot = args.shots_dir / f"manual_{shot_stamp()}.png"
        adb_screencap(shot, args.serial)
        print(f"截图: {shot}")

        result = analyze_image(
            shot, args.state,
            reset=(reset_next or (first and not args.state.exists())),
            slots=args.slots,
        )
        print(result.report)

        step_label = f"manual-{manual_no}"
        _append_decision_log(
            log_path,
            screenshot=shot,
            result=result,
            step_label=step_label,
        )

        if not args.no_auto_tap and result.best is not None:
            if args.tap_delay > 0:
                time.sleep(args.tap_delay)
            x, y = tap_candidate_from_result(result, serial=args.serial)
            text = (
                f"click column={result.best.column} "
                f"car={ctag(result.best.color)}x{result.best.capacity} "
                f"xy=({x},{y})"
            )
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=text,
            )
            print(
                f"自动点击完成: 第一排第 {result.best.column} 列 "
                f"{ctag(result.best.color)}×{result.best.capacity}，坐标=({x}, {y})"
            )
        elif result.best is None:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution="no-click: no safe candidate",
            )
            print("未执行点击：没有安全候选。")

        first = False
        reset_next = False


def run_auto_flow_mode(args: argparse.Namespace) -> int:
    display = SolverDisplay(
        enabled=not args.no_display,
        max_width=args.display_width,
        max_height=args.display_height,
    )
    display.start()

    try:
        return _run_auto_flow_mode_impl(args, display)
    finally:
        display.close()


def _run_auto_flow_mode_impl(
    args: argparse.Namespace,
    display: SolverDisplay,
) -> int:
    reset_current = args.reset
    round_no = 0
    front_number_cache: Optional[Dict[int, FrontNumberCacheEntry]] = None
    pending_prediction: Optional[Tuple[int, str, str]] = None
    log_path = _resolve_decision_log_path(args)
    _append_session_marker(log_path, reset=args.reset)

    print("完全自动模式已启动。按 Ctrl+C 可随时停止。")
    print(f"决策日志: {log_path}")

    if not args.skip_sixth_slot_unlock:
        did_unlock = unlock_sixth_slot_at_game_start(
            serial=args.serial,
            shots_dir=args.shots_dir,
            min_watch_seconds=args.unlock_ad_wait,
            display=display,
        )
        if did_unlock and args.unlock_return_settle_delay > 0:
            time.sleep(args.unlock_return_settle_delay)
    else:
        print("--skip-sixth-slot-unlock：跳过第6停车位自动解锁检查。")

    print(
        f"分流判定: 点击后 {args.flow_start_delay:.1f}s 建基准；"
        f"每 {args.parking_check_interval:.1f}s 检查；"
        f"连续 {args.parking_idle_timeout:.1f}s 无停车数字像素变化才进入下一步。"
    )
    print(
        "策略模型: 多车并发吸收 + 自动分流闭包；"
        "稳定后停车占用最坏上界必须 < 总停车位。"
    )

    while True:
        round_no += 1
        step_label = f"auto-{round_no}"
        print("\n" + "#" * 72)
        print(f"自动轮次 {round_no}")
        print("#" * 72)

        shot = args.shots_dir / f"analysis_{shot_stamp()}.png"
        adb_screencap(shot, args.serial)
        print(f"分析截图: {shot}")

        analysis_bgr = cv2.imread(str(shot), cv2.IMREAD_COLOR)
        if analysis_bgr is not None:
            display.show(
                analysis_bgr,
                stage=f"自动轮次 {round_no} · 正在分析",
                hint="正在识别棋盘、队列与停车车辆，并执行并发分流闭包策略。",
            )

        result = analyze_image(
            shot, args.state,
            reset=(reset_current or not args.state.exists()),
            slots=args.slots,
            front_number_cache=front_number_cache,
        )
        reset_current = False
        front_number_cache = result.front_number_cache

        if pending_prediction is not None:
            predicted_upper, predicted_step, predicted_basis = pending_prediction
            if result.occupied_slots > predicted_upper:
                result.report += (
                    "\n\n!!! GUARANTEE_BROKEN / 硬安全暂停 !!!\n"
                    f"上一轮 {predicted_step} 的模型保证“稳定后停车占用上界 <= "
                    f"{predicted_upper}”，但本轮稳定截图实际检测到 "
                    f"{result.occupied_slots}/{args.slots}。\n"
                    f"上一轮依据: {predicted_basis}\n"
                    "guaranteed 已被真实结果反证；本轮禁止继续自动点击，"
                    "避免同一错误假设连续传播。"
                )
                result.best = None
                result.two_step_plan = None
            pending_prediction = None

        print(result.report)
        print(
            f"OCR信息: 本轮第一排完整 OCR {result.front_ocr_reads}/"
            f"{len(result.front_number_cache)} 列；安全模式下暂不跨车辆复用数字缓存。"
        )

        # 每一步先落盘完整决策依据，确保即使后续点击/监控异常也能追溯。
        _append_decision_log(
            log_path,
            screenshot=shot,
            result=result,
            step_label=step_label,
        )

        if len(result.front) == 0 and len(result.nxt) == 0:
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="胜利确认",
                    hint=(
                        "排队区第一次检测为空；"
                        f"等待 {args.queue_empty_confirm_delay:.1f}s 再确认一次。"
                    ),
                )
            time.sleep(args.queue_empty_confirm_delay)
            confirm_bgr = adb_capture_bgr(args.serial)
            empty2, _added = queue_empty_on_image(confirm_bgr, result.palette)
            confirm_path = args.shots_dir / f"queue_confirm_{shot_stamp()}.png"
            save_bgr(confirm_path, confirm_bgr)

            if empty2:
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=f"game-complete confirm={confirm_path.name}",
                )
                display.show(
                    confirm_bgr,
                    stage="本局完成",
                    hint="排队区连续两次无车辆；判定本局结束。",
                )
                print(
                    f"排队区连续两次无车辆，判定本局结束。确认截图: {confirm_path}"
                )
                return 0

            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    f"queue-empty-cancelled confirm={confirm_path.name}; "
                    "second check found vehicles"
                ),
            )
            display.show(
                confirm_bgr,
                stage="胜利确认取消",
                hint="第二次重新检测到排队车辆；刚才只是补位动画，重新分析。",
            )
            if args.analysis_settle_delay > 0:
                time.sleep(args.analysis_settle_delay)
            continue

        if args.no_auto_tap:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution="no-auto-tap: analysis only",
            )
            print("已启用 --no-auto-tap：完成一次分析后退出。")
            return 0

        if result.best is None:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution="safety-stop: no stable-safe candidate",
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="安全暂停",
                    hint="当前没有通过稳定状态硬安全约束的候选动作；未执行点击。",
                )
            print("自动流程暂停：当前没有通过稳定状态硬安全约束的候选动作。")
            return 0

        if args.tap_delay > 0:
            time.sleep(args.tap_delay)

        plan = result.two_step_plan
        execution_parts = []
        predicted_occupied_upper = result.best.flow_final_occupied_upper
        predicted_basis = (
            f"single {ctag(result.best.color)}x{result.best.capacity} "
            f"flow_final_occupied_upper={result.best.flow_final_occupied_upper}"
        )

        # v5.3：双步最坏两辆都留下时只要求不超过 6，因此空位>=2 即可。
        if plan is not None and (args.slots - result.occupied_slots) >= 2:
            first_car = next(
                c for c in result.front
                if c.column == plan.first.column
            )

            if plan.second_source == "next":
                second_visual_car = next(
                    c for c in result.nxt
                    if c.column == plan.second.column
                )
                second_label = "2 NEXT"
                second_hint = (
                    f"第{plan.second.column}列第二排 "
                    f"{ctag(plan.second.color)}×{plan.second.capacity}；"
                    "第一步离开后立即顶上，快速确认再点击。"
                )
            else:
                second_visual_car = next(
                    c for c in result.front
                    if c.column == plan.second.column
                )
                second_label = "2"
                second_hint = (
                    f"当前第一排第{plan.second.column}列 "
                    f"{ctag(plan.second.color)}×{plan.second.capacity}。"
                )

            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="执行连续两步",
                    hint=(
                        f"第1步：第{plan.first.column}列 "
                        f"{ctag(plan.first.color)}×{plan.first.capacity}；"
                        f"第2步：{second_hint} "
                        "A+B 已按并发联合动作闭包证明稳定安全。"
                    ),
                    marks=(
                        ClickMark(
                            int(round(first_car.x)),
                            int(round(first_car.y)),
                            "1",
                        ),
                        ClickMark(
                            int(round(second_visual_car.x)),
                            int(round(second_visual_car.y)),
                            second_label,
                        ),
                    ),
                )

            x1, y1 = tap_candidate_from_result(
                result,
                serial=args.serial,
                candidate=plan.first,
            )
            execution_parts.append(
                f"step1 col={plan.first.column} "
                f"{ctag(plan.first.color)}x{plan.first.capacity} "
                f"xy=({x1},{y1})"
            )
            print(
                f"连续两步 1/2: 第 {plan.first.column} 列 "
                f"{ctag(plan.first.color)}×{plan.first.capacity}，"
                f"坐标=({x1}, {y1})"
            )

            second_executed = False

            if plan.second_source == "next":
                ok, _confirm_frame, x2, y2 = wait_for_promoted_next_car(
                    result,
                    plan,
                    serial=args.serial,
                    timeout=args.queue_promote_timeout,
                    poll_interval=args.queue_promote_poll_interval,
                    initial_delay=args.double_step_gap,
                    display=display,
                )
                if ok:
                    adb_tap(x2, y2, args.serial)
                    second_executed = True
                    execution_parts.append(
                        f"step2-next confirmed col={plan.second.column} "
                        f"{ctag(plan.second.color)}x{plan.second.capacity} "
                        f"xy=({x2},{y2})"
                    )
                    print(
                        f"连续两步 2/2: 同列补位确认通过，第 {plan.second.column} 列 "
                        f"{ctag(plan.second.color)}×{plan.second.capacity}，"
                        f"坐标=({x2}, {y2})"
                    )
                else:
                    execution_parts.append(
                        "step2-next cancelled: promote confirmation failed; "
                        "first step remains stable-safe"
                    )
                    print(
                        "连续两步第二步取消：补位未通过颜色+数字确认；"
                        "第一步单独也已通过稳定状态安全检查。"
                    )
            else:
                if args.double_step_gap > 0:
                    time.sleep(args.double_step_gap)
                x2, y2 = tap_candidate_from_result(
                    result,
                    serial=args.serial,
                    candidate=plan.second,
                )
                second_executed = True
                execution_parts.append(
                    f"step2-front col={plan.second.column} "
                    f"{ctag(plan.second.color)}x{plan.second.capacity} "
                    f"xy=({x2},{y2})"
                )
                print(
                    f"连续两步 2/2: 第 {plan.second.column} 列 "
                    f"{ctag(plan.second.color)}×{plan.second.capacity}，"
                    f"坐标=({x2}, {y2})"
                )

            if second_executed:
                predicted_occupied_upper = plan.final_occupied_upper
                predicted_basis = (
                    f"two-step score={plan.score:.1f}, "
                    f"flow_final_occupied_upper={plan.final_occupied_upper}, "
                    f"guaranteed_completions={plan.guaranteed_completions}"
                )
                print("两步已连续执行；现在只做一次停车数字分流监控。")
            else:
                # 同列第二步未执行时，只验证第一步自己的保证。
                predicted_occupied_upper = plan.first.flow_final_occupied_upper
                predicted_basis = (
                    f"two-step fallback to first only; "
                    f"flow_final_occupied_upper="
                    f"{plan.first.flow_final_occupied_upper}"
                )
        else:
            target_car = next(
                c for c in result.front
                if c.column == result.best.column
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="执行点击",
                    hint=(
                        f"点击第{result.best.column}列 "
                        f"{ctag(result.best.color)}×{result.best.capacity}"
                    ),
                    marks=(
                        ClickMark(
                            int(round(target_car.x)),
                            int(round(target_car.y)),
                            "CLICK",
                        ),
                    ),
                )

            x, y = tap_candidate_from_result(
                result,
                serial=args.serial,
            )
            execution_parts.append(
                f"single col={result.best.column} "
                f"{ctag(result.best.color)}x{result.best.capacity} "
                f"xy=({x},{y})"
            )
            print(
                f"自动点击完成: 第一排第 {result.best.column} 列 "
                f"{ctag(result.best.color)}×{result.best.capacity}，"
                f"坐标=({x}, {y})"
            )

        _append_execution_update(
            log_path,
            screenshot=shot,
            step_label=step_label,
            execution="; ".join(execution_parts),
        )

        if analysis_bgr is not None:
            display.show(
                analysis_bgr,
                stage="等待分流启动",
                hint=(
                    f"点击已完成；等待 {args.flow_start_delay:.1f}s "
                    "后建立停车数字监控基准。"
                ),
            )

        monitor_end = wait_for_parking_idle(
            shots_dir=args.shots_dir,
            serial=args.serial,
            start_delay=args.flow_start_delay,
            check_interval=args.parking_check_interval,
            idle_timeout=args.parking_idle_timeout,
            max_failures=args.monitor_max_failures,
            empty_settle_delay=args.empty_settle_delay,
            display=display,
        )
        print(f"本轮停车监控结束: {monitor_end}")
        _append_execution_update(
            log_path,
            screenshot=shot,
            step_label=step_label,
            execution=f"monitor_end={Path(monitor_end).name if monitor_end else monitor_end}",
        )

        pending_prediction = (
            int(predicted_occupied_upper),
            step_label,
            predicted_basis,
        )
        _append_execution_update(
            log_path,
            screenshot=shot,
            step_label=step_label,
            execution=(
                "prediction_checkpoint "
                f"next_stable_parking_upper={predicted_occupied_upper}; "
                f"basis={predicted_basis}"
            ),
        )

        if args.analysis_settle_delay > 0:
            time.sleep(args.analysis_settle_delay)