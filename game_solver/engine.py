from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .adb import (
    adb_capture_bgr, adb_screencap, adb_tap, save_bgr, shot_stamp,
)
from .board import ctag, initial_grid, learn_palette, update_grid
from .config import FRONT_X_N, FRONT_Y_N
from .models import AnalysisResult, Candidate, FrontNumberCacheEntry
from .monitor import wait_for_parking_idle
from .state import load_state, save_state
from .strategy import (
    best_valid_candidate, choose_two_step_plan, evaluate_candidates,
    format_report, format_two_step_plan,
)
from .unlock import unlock_sixth_slot_at_game_start
from .vehicles import (
    _front_number_fingerprint, detect_front_and_next, detect_front_centers,
    detect_parked, extend_palette_from_front_numbers, parking_roi,
    read_front_numbers_at_centers, read_front_numbers_cached,
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

    if front_number_cache is None:
        front_numbers = read_front_numbers_at_centers(image_bgr, front_centers)
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
    else:
        front_numbers, new_front_cache, front_ocr_reads = read_front_numbers_cached(
            image_bgr, front_centers, front_number_cache
        )

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
        # 保留字段以兼容 v3 状态文件结构；v4 不再用它做停车占用硬判断。
        parking_empty_ref = parking_roi(image_bgr)
    else:
        palette, prev_grid, turn, saved_size, parking_empty_ref = load_state(state_path)
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

    # 停车车使用数字锚点：一组 1/2 位数字 = 一辆车。
    parked = detect_parked(
        image_rgb, image_bgr, palette, read_numbers=True
    )
    occupied_slots = len(parked)

    if occupied_slots > slots:
        raise RuntimeError(
            f"停车数字锚点检测到 {occupied_slots} 辆，超过设定停车位 {slots}，"
            "为避免误点击已停止。"
        )

    # 停车车若数字已找到但颜色无法可靠匹配，也不能参与安全证明。
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

    if two_step_plan is not None:
        report += "\n\n" + format_two_step_plan(two_step_plan)
        # 双步模式下第一步选择应服从“整对动作”的预测，而不是单步最高分。
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


def queue_empty_on_image(image_bgr: np.ndarray, palette: np.ndarray) -> Tuple[bool, int]:
    """
    轻量确认排队区是否无车。

    胜利判定直接看当前第一排数字组是否存在。
    列数动态，因此不会再因“本关是5列而代码只看4列”误判为空。
    """
    centers = detect_front_centers(image_bgr)
    if not centers:
        return True, 0

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    nums = read_front_numbers_at_centers(image_bgr, centers)
    palette2, added = extend_palette_from_front_numbers(
        image_rgb, palette, nums, centers
    )
    front, _nxt = detect_front_and_next(
        image_rgb, image_bgr, palette2, centers, nums
    )
    return (len(front) == 0), added



def tap_candidate_from_result(
    result: AnalysisResult,
    *,
    serial: Optional[str],
    candidate: Optional[Candidate] = None,
) -> Tuple[int, int]:
    """
    按当前分析帧中的真实车辆坐标点击。

    candidate=None 时点击 result.best；
    双步模式第二次点击会明确传入 result.two_step_plan.second。
    """
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
    print("人工逐步调试模式。")
    while True:
        cmd = input("\nEnter截图分析；q退出；r将下一张作为新局重建状态: ").strip().lower()
        if cmd == "q":
            return 0
        if cmd == "r":
            reset_next = True

        shot = args.shots_dir / f"manual_{shot_stamp()}.png"
        adb_screencap(shot, args.serial)
        print(f"截图: {shot}")
        result = analyze_image(
            shot, args.state,
            reset=(reset_next or (first and not args.state.exists())),
            slots=args.slots,
        )
        print(result.report)

        if not args.no_auto_tap and result.best is not None:
            if args.tap_delay > 0:
                time.sleep(args.tap_delay)
            x, y = tap_candidate_from_result(result, serial=args.serial)
            print(f"自动点击完成: 第一排第 {result.best.column} 列 {ctag(result.best.color)}×{result.best.capacity}，坐标=({x}, {y})")
        elif result.best is None:
            print("未执行点击：没有安全候选。")

        first = False
        reset_next = False


def run_auto_flow_mode(args: argparse.Namespace) -> int:
    reset_current = args.reset
    round_no = 0
    front_number_cache: Optional[Dict[int, FrontNumberCacheEntry]] = None

    print("完全自动模式已启动。按 Ctrl+C 可随时停止。")

    # 每次自动模式启动都先“看一眼”解锁按钮，而不是再用 state 文件猜是不是新局。
    # - 按钮存在：自动点击并观看广告；
    # - 按钮不存在：立即跳过；
    # 因此即使用户忘记 --reset 或保留了旧 solver_state.npz，也不会漏掉新局解锁。
    if not args.skip_sixth_slot_unlock:
        did_unlock = unlock_sixth_slot_at_game_start(
            serial=args.serial,
            shots_dir=args.shots_dir,
            min_watch_seconds=args.unlock_ad_wait,
        )
        if did_unlock and args.unlock_return_settle_delay > 0:
            time.sleep(args.unlock_return_settle_delay)
    else:
        print("已启用 --skip-sixth-slot-unlock：跳过第6停车位自动解锁检查。")
    print(
        f"分流判定: 点击后 {args.flow_start_delay:.1f}s 建基准；"
        f"每 {args.parking_check_interval:.1f}s 检查；"
        f"连续 {args.parking_idle_timeout:.1f}s 无停车数字像素变化才进入下一步。"
    )
    print("监控截图与分析截图已完全分离：每个自动轮次都会重新 ADB 截取 analysis_*.png。")

    while True:
        round_no += 1
        print("\n" + "#" * 72)
        print(f"自动轮次 {round_no}")
        print("#" * 72)

        # 关键修复：无论上一轮监控返回什么图片，本轮都重新截图。
        # monitor_end_*.png 只用于诊断，永远不会进入 analyze_image。
        shot = args.shots_dir / f"analysis_{shot_stamp()}.png"
        adb_screencap(shot, args.serial)
        print(f"分析截图: {shot}")

        result = analyze_image(
            shot, args.state,
            reset=(reset_current or not args.state.exists()),
            slots=args.slots,
            front_number_cache=front_number_cache,
        )
        reset_current = False
        front_number_cache = result.front_number_cache
        print(result.report)
        print(
            f"速度信息: 本轮第一排实际 OCR {result.front_ocr_reads}/"
            f"{len(result.front_number_cache)} 列，其余列沿用数字视觉缓存。"
        )

        # 胜利判定按用户规则：排队区没有车辆。为防补位动画中间的空帧，连续确认两次。
        if len(result.front) == 0 and len(result.nxt) == 0:
            print(f"排队区第一次检测为空，等待 {args.queue_empty_confirm_delay:.1f}s 再确认一次...")
            time.sleep(args.queue_empty_confirm_delay)
            confirm_bgr = adb_capture_bgr(args.serial)
            empty2, _added = queue_empty_on_image(confirm_bgr, result.palette)
            confirm_path = args.shots_dir / f"queue_confirm_{shot_stamp()}.png"
            save_bgr(confirm_path, confirm_bgr)
            if empty2:
                print(f"排队区连续两次无车辆，判定本局结束。确认截图: {confirm_path}")
                return 0
            print("第二次检测到排队车辆，说明刚才处于补位动画；旧建议作废，下一轮重新截图分析。")
            if args.analysis_settle_delay > 0:
                time.sleep(args.analysis_settle_delay)
            continue

        if args.no_auto_tap:
            print("已启用 --no-auto-tap：完成一次分析后退出。")
            return 0

        if result.best is None:
            print("自动流程暂停：当前没有通过硬安全约束的候选动作。")
            return 0

        if args.tap_delay > 0:
            time.sleep(args.tap_delay)

        plan = result.two_step_plan

        if plan is not None and (args.slots - result.occupied_slots) >= 3:
            x1, y1 = tap_candidate_from_result(
                result,
                serial=args.serial,
                candidate=plan.first,
            )
            print(
                f"连续两步 1/2: 第 {plan.first.column} 列 "
                f"{ctag(plan.first.color)}×{plan.first.capacity}，"
                f"坐标=({x1}, {y1})"
            )

            if args.double_step_gap > 0:
                time.sleep(args.double_step_gap)

            x2, y2 = tap_candidate_from_result(
                result,
                serial=args.serial,
                candidate=plan.second,
            )
            print(
                f"连续两步 2/2: 第 {plan.second.column} 列 "
                f"{ctag(plan.second.color)}×{plan.second.capacity}，"
                f"坐标=({x2}, {y2})"
            )
            print(
                "两步已连续执行；现在只做一次停车数字分流监控。"
            )
        else:
            x, y = tap_candidate_from_result(
                result,
                serial=args.serial,
            )
            print(
                f"自动点击完成: 第一排第 {result.best.column} 列 "
                f"{ctag(result.best.color)}×{result.best.capacity}，"
                f"坐标=({x}, {y})"
            )

        monitor_end = wait_for_parking_idle(
            shots_dir=args.shots_dir,
            serial=args.serial,
            start_delay=args.flow_start_delay,
            check_interval=args.parking_check_interval,
            idle_timeout=args.parking_idle_timeout,
            max_failures=args.monitor_max_failures,
            empty_settle_delay=args.empty_settle_delay,
        )
        print(f"本轮停车监控结束: {monitor_end}")

        # 再给 UI/队列补位一个很短的收尾时间，然后进入下一轮；
        # 下一轮仍会重新 ADB 截 analysis_*.png，而不是复用 monitor_end。
        if args.analysis_settle_delay > 0:
            time.sleep(args.analysis_settle_delay)