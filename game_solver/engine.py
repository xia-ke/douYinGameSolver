from __future__ import annotations

import argparse
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .adb import (
    adb_capture_bgr, adb_screencap, adb_tap, save_bgr, shot_stamp,
)
from .board import (
    ObservedBoard, ctag, learn_palette, observe_board, reachable_summary,
)
from .config import FRONT_Y_N, UNKNOWN
from .display import ClickMark, SolverDisplay
from .models import AnalysisResult, Candidate, TwoStepPlan
from .monitor import wait_for_parking_idle
from .ocr import read_number_at, read_number_detailed_at
from .state import TrustedSessionState, load_state, save_state
from .strategy import (
    best_valid_candidate, choose_two_step_plan, evaluate_candidates,
    format_report, format_two_step_plan,
)
from .unlock import unlock_sixth_slot_at_game_start
from .vehicles import (
    car_color_at, detect_front_and_next,
    detect_front_centers, detect_parked, extend_palette_from_front_numbers,
    parking_roi, read_front_numbers_at_centers,
)



def _resolve_decision_log_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "decision_log", None)
    if path is not None:
        return Path(path)
    return args.shots_dir / "decision_log.txt"


def _resolve_color_log_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "color_log", None)
    if path is not None:
        return Path(path)
    return args.shots_dir / "color_log.txt"


def _resolve_number_log_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "number_log", None)
    if path is not None:
        return Path(path)
    return args.shots_dir / "number_log.txt"


def _diagnostic_color_tag(color: Optional[int]) -> str:
    if color is None:
        return "UNKNOWN"
    value = int(color)
    if value <= 0:
        return "UNKNOWN"
    return ctag(value)


def _diagnostic_number(value: Optional[int]) -> str:
    return "UNKNOWN" if value is None else str(int(value))


def _palette_rgb_hex(rgb: np.ndarray) -> Tuple[int, int, int, str]:
    vals = tuple(
        int(round(max(0.0, min(255.0, float(v)))))
        for v in np.asarray(rgb).reshape(-1)[:3]
    )
    if len(vals) != 3:
        return 0, 0, 0, "#000000"
    r, g, b = vals
    return r, g, b, f"#{r:02X}{g:02X}{b:02X}"


def _append_color_observation_log(
    log_path: Path,
    *,
    screenshot: Path,
    result: AnalysisResult,
    step_label: str,
) -> None:
    """
    每次最终稳定识别后记录一份纯识别色彩快照。

    内容：
      - 当前动态 palette：C01..Cx -> RGB / HEX；
      - 当前持久棋盘的 52x38 颜色矩阵；
      - 每种颜色在棋盘中的格数；
      - 排队区第一排 / 第二排颜色；
      - 停车区颜色（按检测到的 x 坐标从左到右编号）。

    只读取 AnalysisResult，不重复执行颜色识别，不影响决策流程。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    palette = np.asarray(result.palette, dtype=np.float32)
    grid = np.asarray(result.grid)

    health = result.observation_health
    lines = [
        "",
        "=" * 120,
        f"[COLOR_SNAPSHOT] time={timestamp}",
        f"step={step_label}",
        f"turn={result.turn}",
        f"screenshot={screenshot.name}",
        f"screenshot_path={screenshot}",
        f"grid_rows={grid.shape[0]} grid_cols={grid.shape[1]}",
        f"palette_count={len(palette)}",
        f"observation_trusted={'yes' if health.trusted else 'no'}",
        f"observation_reasons={'; '.join(health.reasons) or 'none'}",
        "-" * 120,
        "[PALETTE]",
        "color | RGB             | HEX     | board_cells",
        "------|-----------------|---------|------------",
    ]

    for idx, rgb in enumerate(palette, 1):
        r, g, b, hex_color = _palette_rgb_hex(rgb)
        board_cells = int(np.count_nonzero(grid == idx))
        lines.append(
            f"{ctag(idx):<5} | ({r:3d},{g:3d},{b:3d}) | "
            f"{hex_color:<7} | {board_cells}"
        )

    lines.extend([
        "",
        "[BOARD_COLOR_COUNTS]",
        f"EMPTY={int(np.count_nonzero(grid == 0))}",
        f"UNKNOWN={int(np.count_nonzero(grid == UNKNOWN))}",
    ])
    for idx in range(1, len(palette) + 1):
        lines.append(f"{ctag(idx)}={int(np.count_nonzero(grid == idx))}")

    front_by_col = {
        int(car.column): car
        for car in result.front
        if car.column is not None
    }
    next_by_col = {
        int(car.column): car
        for car in result.nxt
        if car.column is not None
    }
    queue_columns = sorted(set(front_by_col) | set(next_by_col))

    lines.extend([
        "",
        "[QUEUE_COLORS]",
        "column | front_color | next_color",
        "-------|-------------|-----------",
    ])
    if queue_columns:
        for column in queue_columns:
            front_car = front_by_col.get(column)
            next_car = next_by_col.get(column)
            lines.append(
                f"{column:>6d} | "
                f"{_diagnostic_color_tag(front_car.color if front_car else None):<11} | "
                f"{_diagnostic_color_tag(next_car.color if next_car else None)}"
            )
    else:
        lines.append("(empty)")

    parked = sorted(result.parked, key=lambda car: float(car.x))
    lines.extend([
        "",
        "[PARKING_COLORS]",
        "order_left_to_right | x_px    | color",
        "--------------------|---------|--------",
    ])
    if parked:
        for order, car in enumerate(parked, 1):
            lines.append(
                f"{order:>19d} | {float(car.x):>7.1f} | "
                f"{_diagnostic_color_tag(car.color)}"
            )
    else:
        lines.append("(empty)")

    lines.extend([
        "",
        "[BOARD_GRID]",
        "legend: ----=EMPTY, ????=UNKNOWN, C01..Cx=recognized color",
    ])

    if grid.ndim == 2:
        cols = int(grid.shape[1])
        header = "row\\col | " + " ".join(
            f"{c + 1:>4d}" for c in range(cols)
        )
        lines.append(header)
        lines.append("-" * len(header))

        for r in range(int(grid.shape[0])):
            values = []
            for c in range(cols):
                value = int(grid[r, c])
                if value == 0:
                    tag = "----"
                elif value == UNKNOWN or value < 0:
                    tag = "????"
                else:
                    tag = ctag(value)
                values.append(f"{tag:>4}")
            lines.append(f"R{r + 1:02d}     | " + " ".join(values))
    else:
        lines.append(f"(unexpected grid shape: {grid.shape})")

    lines.append("=" * 120)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_number_observation_log(
    log_path: Path,
    *,
    screenshot: Path,
    result: AnalysisResult,
    step_label: str,
) -> None:
    """
    每次最终稳定识别后记录排队区和停车区数字。

    颜色只作为交叉定位列；数字字段完全来自当前 AnalysisResult，
    不在日志阶段再次执行 OCR。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    front_by_col = {
        int(car.column): car
        for car in result.front
        if car.column is not None
    }
    next_by_col = {
        int(car.column): car
        for car in result.nxt
        if car.column is not None
    }
    queue_columns = sorted(set(front_by_col) | set(next_by_col))

    health = result.observation_health
    lines = [
        "",
        "=" * 104,
        f"[NUMBER_SNAPSHOT] time={timestamp}",
        f"step={step_label}",
        f"turn={result.turn}",
        f"screenshot={screenshot.name}",
        f"screenshot_path={screenshot}",
        f"observation_trusted={'yes' if health.trusted else 'no'}",
        f"observation_reasons={'; '.join(health.reasons) or 'none'}",
        "-" * 104,
        "[QUEUE_NUMBERS]",
        "column | front_color | front_number | next_color | next_number",
        "-------|-------------|--------------|------------|------------",
    ]

    if queue_columns:
        for column in queue_columns:
            front_car = front_by_col.get(column)
            next_car = next_by_col.get(column)
            lines.append(
                f"{column:>6d} | "
                f"{_diagnostic_color_tag(front_car.color if front_car else None):<11} | "
                f"{_diagnostic_number(front_car.remain if front_car else None):<12} | "
                f"{_diagnostic_color_tag(next_car.color if next_car else None):<10} | "
                f"{_diagnostic_number(next_car.remain if next_car else None)}"
            )
    else:
        lines.append("(empty)")

    parked = sorted(result.parked, key=lambda car: float(car.x))
    lines.extend([
        "",
        "[PARKING_NUMBERS]",
        "order_left_to_right | x_px    | color   | remain_number",
        "--------------------|---------|---------|--------------",
    ])
    if parked:
        for order, car in enumerate(parked, 1):
            lines.append(
                f"{order:>19d} | {float(car.x):>7.1f} | "
                f"{_diagnostic_color_tag(car.color):<7} | "
                f"{_diagnostic_number(car.remain)}"
            )
    else:
        lines.append("(empty)")

    lines.extend([
        "",
        "[SUMMARY]",
        f"front_detected={len(result.front)}",
        f"next_detected={len(result.nxt)}",
        f"parked_detected={len(result.parked)}",
        f"occupied_slots={result.occupied_slots}",
        "=" * 104,
    ])

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _append_observation_table_logs(
    color_log_path: Path,
    number_log_path: Path,
    *,
    screenshot: Path,
    result: AnalysisResult,
    step_label: str,
) -> None:
    """Write diagnosis-only tables for the selected stable observation."""
    _append_color_observation_log(
        color_log_path,
        screenshot=screenshot,
        result=result,
        step_label=step_label,
    )
    _append_number_observation_log(
        number_log_path,
        screenshot=screenshot,
        result=result,
        step_label=step_label,
    )


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


def _parking_remaining_by_color(parked) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for car in parked:
        if car.color is None or car.remain is None:
            continue
        color = int(car.color)
        out[color] = out.get(color, 0) + int(car.remain)
    return out


def _capacity_before_flow_by_color(
    parked,
    executed_actions,
) -> Dict[int, int]:
    """
    本轮分流开始时，每种颜色所有活跃车辆的总剩余容量。

    = 点击前已有停车车辆剩余
    + 本轮真正执行成功的点击车辆容量

    之后下一张稳定截图只要减去仍停在停车区的同色剩余总量，
    就能得到本轮该色真实吸收数量。这个结论与同色车辆分配顺序无关。
    """
    out = _parking_remaining_by_color(parked)

    for color, capacity in executed_actions:
        if color is None or capacity is None:
            continue
        color = int(color)
        capacity = int(capacity)
        if color <= 0 or capacity <= 0:
            continue
        out[color] = out.get(color, 0) + capacity

    return out


def _actual_consumed_from_capacity_delta(
    capacity_before_by_color: Dict[int, int],
    parked_after,
) -> Tuple[Dict[int, int], str]:
    """
    容量守恒：
        actual_consumed[color]
        = before_total_remaining[color]
        - after_stable_parked_remaining[color]

    如果 after > before，说明至少有颜色/OCR/动作确认发生了矛盾；
    这种数据不能驱动棋盘更新。
    """
    after = _parking_remaining_by_color(parked_after)
    colors = set(capacity_before_by_color) | set(after)

    consumed: Dict[int, int] = {}
    invalid_parts = []

    for color in sorted(colors):
        before_n = int(capacity_before_by_color.get(color, 0))
        after_n = int(after.get(color, 0))
        delta = before_n - after_n

        if delta < 0:
            invalid_parts.append(
                f"{ctag(color)}: before={before_n}, after={after_n}"
            )
            continue

        if delta > 0:
            consumed[color] = delta

    if invalid_parts:
        return {}, (
            "稳定后同色剩余容量大于本轮分流开始前总容量，"
            "无法建立容量守恒: "
            + "；".join(invalid_parts)
        )

    return consumed, ""


def _format_observed_board_observation(observation: ObservedBoard) -> str:
    health = observation.health
    lines = [
        "[OBSERVED_BOARD]",
        "  spatial authority: current stable frame first; history resolves current UNKNOWN only",
        (
            f"  cells: COLOR={observation.current_color_cells}, "
            f"EMPTY={observation.current_empty_cells}, "
            f"UNKNOWN={observation.current_unknown_cells}"
        ),
        (
            f"  history_resolved={observation.history_resolved_cells}, "
            f"temporal_empty={observation.temporal_resolved_empty_cells}, "
            f"visual_removed={observation.removed_cells}"
        ),
        f"  trusted={'yes' if health.trusted else 'no'}",
    ]
    if health.reasons:
        lines.append("  reasons: " + "；".join(health.reasons))
    if health.warnings:
        lines.append("  warnings: " + "；".join(health.warnings))
    if observation.capacity_expected_by_color:
        lines.append(
            "  capacity audit expected: "
            + ", ".join(
                f"{ctag(c)}={n}"
                for c, n in observation.capacity_expected_by_color.items()
            )
        )
    if observation.visual_removed_by_color:
        lines.append(
            "  visual removals: "
            + ", ".join(
                f"{ctag(c)}={n}"
                for c, n in observation.visual_removed_by_color.items()
            )
        )
    if health.transition_conflicts:
        lines.append(
            "  forbidden transitions: "
            + ", ".join(
                f"R{r + 1:02d}C{c + 1:02d}:"
                f"{ctag(old) if old > 0 else 'EMPTY'}->{ctag(cur)}"
                for r, c, old, cur in health.transition_conflicts
            )
        )
    return "\n".join(lines)


def _observed_board_allows_planning(
    observation: Optional[ObservedBoard],
    *,
    experimental_continue: bool = False,
) -> bool:
    """Default policy is strict; explicit experiment may plan one conservative step."""
    if observation is None:
        return False
    return bool(observation.health.trusted or experimental_continue)


def _plan_current_observation(
    observation: ObservedBoard,
    grid: np.ndarray,
    front,
    nxt,
    parked,
    slots: int,
    occupied_slots: int,
    *,
    experimental_continue: bool,
):
    """Small execution policy: trusted=normal plan; untrusted=none, or one-step experiment."""
    trusted = bool(observation.health.trusted)
    if not trusted and not experimental_continue:
        return [], None

    candidates = evaluate_candidates(
        grid,
        front,
        nxt,
        parked,
        slots,
        occupied_slots,
        # An explicit untrusted experiment is intentionally one-step only.
        include_queue_lookahead=trusted,
    )
    if not trusted:
        return candidates, None

    return candidates, choose_two_step_plan(
        grid,
        front,
        nxt,
        parked,
        slots,
        occupied_slots,
        candidates,
    )


def analyze_image(
    image_path: Path,
    state_path: Path,
    reset: bool,
    slots: int,
    flow_capacity_before_by_color: Optional[Dict[int, int]] = None,
    previous_prediction_upper: Optional[int] = None,
    previous_prediction_basis: str = "",
    experimental_continue: bool = False,
    commit_state: bool = True,
    prev_grid_image_rgb: Optional[np.ndarray] = None,
) -> AnalysisResult:
    """
    Analyze one stable screenshot.

    Issue 006 trust policy:
      - ObservationHealth describes this frame only;
      - trusted -> normal planner;
      - untrusted -> no planner/no click by default;
      - explicit experimental continuation -> single-step candidates only.
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)

    front_centers = detect_front_centers(image_bgr)
    front_numbers = read_front_numbers_at_centers(
        image_bgr,
        front_centers,
    )
    front_ocr_reads = len(front_centers)

    removed_since_last: Optional[int]
    capacity_validation_error = ""
    new_state = reset or not state_path.exists()

    if new_state:
        palette = learn_palette(image_rgb)
        palette, new_colors_added = extend_palette_from_front_numbers(
            image_rgb,
            palette,
            front_numbers,
            front_centers,
        )

        front, nxt = detect_front_and_next(
            image_rgb,
            image_bgr,
            palette,
            front_centers,
            front_numbers,
        )
        parked = detect_parked(
            image_rgb,
            image_bgr,
            palette,
            read_numbers=True,
        )

        board_observation = observe_board(image_rgb, palette)
        grid = board_observation.grid
        current_grid_rgb = board_observation.grid_rgb_snapshot
        turn = 0
        removed_since_last = None
        parking_empty_ref = parking_roi(image_bgr)

    else:
        trusted_state = load_state(state_path)
        palette = trusted_state.palette
        prev_grid = trusted_state.previous_trusted_grid
        turn = trusted_state.turn
        saved_size = trusted_state.screen_size
        parking_empty_ref = trusted_state.parking_empty_ref
        prev_grid_rgb = trusted_state.previous_trusted_grid_rgb
        if saved_size != (image_w, image_h):
            raise RuntimeError(
                f"截图尺寸从 {saved_size[0]}x{saved_size[1]} 变成 "
                f"{image_w}x{image_h}，请使用 --reset。"
            )

        palette, new_colors_added = extend_palette_from_front_numbers(
            image_rgb,
            palette,
            front_numbers,
            front_centers,
        )

        front, nxt = detect_front_and_next(
            image_rgb,
            image_bgr,
            palette,
            front_centers,
            front_numbers,
        )
        parked = detect_parked(
            image_rgb,
            image_bgr,
            palette,
            read_numbers=True,
        )

        expected_consumed: Dict[int, int] = {}
        has_capacity_context = flow_capacity_before_by_color is not None
        if has_capacity_context:
            expected_consumed, capacity_validation_error = (
                _actual_consumed_from_capacity_delta(
                    flow_capacity_before_by_color,
                    parked,
                )
            )

        board_observation = observe_board(
            image_rgb,
            palette,
            prev_grid,
            previous_trusted_frame_rgb=prev_grid_image_rgb,
            previous_trusted_grid_rgb=prev_grid_rgb,
            consumed_by_color=(
                expected_consumed
                if has_capacity_context and not capacity_validation_error
                else None
            ),
        )
        grid = board_observation.grid
        current_grid_rgb = board_observation.grid_rgb_snapshot
        removed_since_last = board_observation.removed_cells

    health = board_observation.health
    occupied_slots = len(parked)

    # Current-frame OCR / quantity / consistency checks now feed one health object.
    if occupied_slots > slots:
        health.add_reason(
            f"parking_count_exceeds_slots={occupied_slots}/{slots}"
        )

    incomplete_parked = [
        c for c in parked
        if c.color is None or c.remain is None
    ]
    if incomplete_parked:
        health.add_reason(
            "parking_ocr_incomplete="
            f"{len(incomplete_parked)}/{occupied_slots}"
        )

    if front and all(c.remain is None for c in front):
        health.add_reason(f"front_ocr_all_failed={len(front)}")

    if capacity_validation_error:
        health.add_reason(
            "capacity_conservation_invalid: " + capacity_validation_error
        )

    stable_conflicts = _stable_state_conflicts(grid, parked)
    for color, (supply, remains) in sorted(stable_conflicts.items()):
        health.add_reason(
            f"stable_parking_reachable_conflict {ctag(color)}:"
            f"reachable={supply},parked_remaining={list(remains)}"
        )

    prediction_mismatch = bool(
        previous_prediction_upper is not None
        and occupied_slots > int(previous_prediction_upper)
    )
    if prediction_mismatch:
        health.add_warning(
            "post_action_prediction_mismatch: "
            f"expected_parking<={int(previous_prediction_upper)},"
            f"actual={occupied_slots}; basis={previous_prediction_basis or 'unknown'}"
        )

    candidates, two_step_plan = _plan_current_observation(
        board_observation,
        grid,
        front,
        nxt,
        parked,
        slots,
        occupied_slots,
        experimental_continue=bool(experimental_continue),
    )

    report = format_report(
        grid,
        front,
        nxt,
        parked,
        candidates,
        removed_since_last,
        slots,
        len(palette),
        occupied_slots,
        new_colors_added,
    )
    report += "\n\n" + _format_observed_board_observation(board_observation)

    if capacity_validation_error:
        report += (
            "\n\n!!! CAPACITY_VALIDATION_UNTRUSTED / RETRY_OBSERVATION !!!\n"
            f"{capacity_validation_error}\n"
            "容量侧信道只参与数量校验；当前 observation 已标记不可信，"
            "默认不提交、不点击。"
        )

    palette_diag = _format_palette_diagnostics(palette)
    if palette_diag:
        report += (
            "\n\n颜色类别诊断（最近 palette 色对）:\n"
            + palette_diag
        )

    if not health.trusted:
        problem_colors = sorted(
            set(health.capacity_remaining_by_color)
            | set(health.capacity_excess_by_color)
        )
        report += (
            "\n\n!!! OBSERVATION_UNTRUSTED / RETRY_OBSERVATION !!!\n"
            "当前稳定帧 ObservationHealth 未通过 trust gate。默认严格模式"
            "不提交、不点击。"
            f"\n问题颜色: "
            f"{', '.join(ctag(c) for c in problem_colors) or '无特定颜色'}"
            f"\n原因: {'；'.join(health.reasons) or '未提供'}"
        )
        if experimental_continue:
            report += (
                "\n显式 experimental continuation：仅生成单步候选，"
                "不生成两步计划。"
            )

    if stable_conflicts:
        conflict_text = "；".join(
            (
                f"{ctag(color)}: reachable={supply}, "
                f"停车剩余={list(remains)}"
            )
            for color, (supply, remains)
            in sorted(stable_conflicts.items())
        )
        report += (
            "\n\n!!! OBSERVATION_VALIDATION_CONFLICT !!!\n"
            f"矛盾项: {conflict_text}\n"
            "稳定截图中停车车仍有同色 reachable，属于当前 observation "
            "一致性失败；默认 retry/no-click。"
        )

    if prediction_mismatch:
        report += (
            "\n\n!!! POST_ACTION_PREDICTION_MISMATCH / DIAGNOSTIC !!!\n"
            f"上一动作预测稳定停车占用 <= {previous_prediction_upper}，"
            f"当前稳定截图实际为 {occupied_slots}/{slots}。\n"
            f"上一动作依据: {previous_prediction_basis or '未记录'}\n"
            "该信息只作为当前帧 warning，不累计模式、不触发多轮恢复状态。"
        )

    if two_step_plan is not None:
        report += "\n\n" + format_two_step_plan(two_step_plan)
        best = two_step_plan.first
    else:
        best = best_valid_candidate(candidates)

    turn += 1
    state_saved = bool(commit_state and health.trusted)
    if state_saved:
        save_state(
            state_path,
            TrustedSessionState(
                palette=palette,
                previous_trusted_grid=grid,
                turn=turn,
                screen_size=(image_w, image_h),
                parking_empty_ref=parking_empty_ref,
                previous_trusted_grid_rgb=current_grid_rgb,
            ),
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
        front_ocr_reads=front_ocr_reads,
        two_step_plan=two_step_plan,
        observation_health=health,
        state_saved=state_saved,
        grid_rgb_snapshot=current_grid_rgb,
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

        promoted_ocr = read_number_detailed_at(
            frame,
            float(x),
            float(y),
            source="queue-promote",
        )
        last_num = promoted_ocr.value
        last_num_conf = promoted_ocr.confidence
        last_num_votes = promoted_ocr.agreeing_crops

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

def _front_candidate_signature_on_frame(
    frame_bgr: np.ndarray,
    result: AnalysisResult,
    candidate: Candidate,
) -> Tuple[Optional[int], Optional[int]]:
    """v5.12 click confirmation: read color/number at the old front position."""
    car = next(
        (c for c in result.front if c.column == candidate.column),
        None,
    )
    if car is None:
        return None, None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    color = car_color_at(rgb, float(car.x), float(car.y), result.palette)
    number, _conf, _votes = read_number_detailed_at(
        frame_bgr,
        float(car.x),
        float(car.y),
    )
    return color, number


def _has_distinguishable_known_next(
    result: AnalysisResult,
    candidate: Candidate,
) -> bool:
    nxt = next(
        (c for c in result.nxt if c.column == candidate.column),
        None,
    )
    if nxt is None:
        return False
    return (nxt.color, nxt.remain) != (candidate.color, candidate.capacity)


def _tap_proven_not_departed(
    frame_bgr: Optional[np.ndarray],
    result: AnalysisResult,
    candidate: Candidate,
) -> bool:
    """Return True only when a distinguishable next exists and old car remains."""
    if frame_bgr is None or not _has_distinguishable_known_next(result, candidate):
        return False
    color, number = _front_candidate_signature_on_frame(
        frame_bgr,
        result,
        candidate,
    )
    return color == candidate.color and number == candidate.capacity


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
    color_log_path = _resolve_color_log_path(args)
    number_log_path = _resolve_number_log_path(args)
    _append_session_marker(log_path, reset=args.reset)
    _append_session_marker(color_log_path, reset=args.reset)
    _append_session_marker(number_log_path, reset=args.reset)

    print("人工逐步调试模式。")
    print(f"决策日志: {log_path}")
    print(f"颜色表日志: {color_log_path}")
    print(f"数字表日志: {number_log_path}")

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
        _append_observation_table_logs(
            color_log_path,
            number_log_path,
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



def _commit_analysis_result_state(
    state_path: Path,
    result: AnalysisResult,
) -> None:
    """Commit the selected observation as historical context after the gate."""
    save_state(
        state_path,
        TrustedSessionState(
            palette=result.palette,
            previous_trusted_grid=result.grid,
            turn=result.turn,
            screen_size=(result.image_w, result.image_h),
            parking_empty_ref=result.parking_empty_ref,
            previous_trusted_grid_rgb=result.grid_rgb_snapshot,
        ),
    )
    result.state_saved = True


def _observation_quality(result: AnalysisResult) -> Tuple[int, int, int]:
    """Lower is better: trusted first, then fewer reasons, then fewer UNKNOWN cells."""
    health = getattr(result, "observation_health", None)
    if health is None:
        return (2, 10**9, 10**9)
    return (
        0 if bool(health.trusted) else 1,
        len(health.reasons),
        int(health.unknown_cells),
    )


def _observation_trust(
    result: AnalysisResult,
) -> Tuple[bool, Tuple[str, ...]]:
    """Read the one canonical current-observation trust signal."""
    health = getattr(result, "observation_health", None)
    if health is None:
        return False, ("observation_health_missing",)
    reasons = tuple(str(reason) for reason in (health.reasons or ()))
    if not bool(health.trusted) and not reasons:
        reasons = ("observation_untrusted_without_reason",)
    return bool(health.trusted), reasons


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
    pending_prediction: Optional[Tuple[int, str, str]] = None
    pending_flow_capacity_before_by_color: Optional[Dict[int, int]] = None
    committed_analysis_rgb: Optional[np.ndarray] = None
    log_path = _resolve_decision_log_path(args)
    color_log_path = _resolve_color_log_path(args)
    number_log_path = _resolve_number_log_path(args)
    _append_session_marker(log_path, reset=args.reset)
    _append_session_marker(color_log_path, reset=args.reset)
    _append_session_marker(number_log_path, reset=args.reset)
    experimental_continue = bool(
        getattr(args, "experimental_continue", False)
    )

    print("完全自动模式已启动。按 Ctrl+C 可随时停止。")
    print(f"决策日志: {log_path}")
    print(f"颜色表日志: {color_log_path}")
    print(f"数字表日志: {number_log_path}")

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
    if experimental_continue:
        print(
            "EXPERIMENTAL_CONTINUE 已启用：bounded retry 后仍不可信的 observation "
            "可显式提交，但只生成单步候选；不维护恢复模式历史。"
        )
    else:
        print(
            "严格观测门已启用：ObservationHealth.trusted=false 经 bounded retry 后 "
            "仍为 NO_CLICK_UNTRUSTED；不提交、不点击。"
        )

    while True:
        round_no += 1
        step_label = f"auto-{round_no}"
        print("\n" + "#" * 72)
        print(f"自动轮次 {round_no}")
        print("#" * 72)

        max_observation_attempts = max(
            1,
            int(getattr(args, "observation_retries", 3)),
        )
        retry_delay = max(
            0.05,
            float(getattr(args, "observation_retry_delay", 0.45)),
        )

        flow_context = pending_flow_capacity_before_by_color
        prediction_context = pending_prediction

        best_observation_result: Optional[AnalysisResult] = None
        best_observation_shot: Optional[Path] = None
        best_observation_bgr: Optional[np.ndarray] = None

        for observation_attempt in range(1, max_observation_attempts + 1):
            if observation_attempt == 1:
                shot = args.shots_dir / f"analysis_{shot_stamp()}.png"
            else:
                shot = (
                    args.shots_dir
                    / f"analysis_retry{observation_attempt}_{shot_stamp()}.png"
                )

            adb_screencap(shot, args.serial)
            print(
                f"分析截图: {shot}"
                + (
                    ""
                    if observation_attempt == 1
                    else f"（观测重试 {observation_attempt}/{max_observation_attempts}）"
                )
            )

            analysis_bgr = cv2.imread(str(shot), cv2.IMREAD_COLOR)
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage=(
                        f"自动轮次 {round_no} · 正在分析"
                        if observation_attempt == 1
                        else (
                            f"自动轮次 {round_no} · RETRY_OBSERVATION "
                            f"{observation_attempt}/{max_observation_attempts}"
                        )
                    ),
                    hint=(
                        "正在识别棋盘、队列与停车车辆。"
                        if observation_attempt == 1
                        else "上一张 ObservationHealth 不可信；重新截图确认，不点击。"
                    ),
                )

            result = analyze_image(
                shot,
                args.state,
                reset=(reset_current or not args.state.exists()),
                slots=args.slots,
                flow_capacity_before_by_color=flow_context,
                previous_prediction_upper=(
                    prediction_context[0]
                    if prediction_context is not None
                    else None
                ),
                previous_prediction_basis=(
                    prediction_context[2]
                    if prediction_context is not None
                    else ""
                ),
                experimental_continue=experimental_continue,
                # Retry attempts are read-only and share the same committed state.
                commit_state=False,
                prev_grid_image_rgb=committed_analysis_rgb,
            )

            if (
                best_observation_result is None
                or _observation_quality(result)
                < _observation_quality(best_observation_result)
            ):
                best_observation_result = result
                best_observation_shot = shot
                best_observation_bgr = analysis_bgr

            attempt_trusted, attempt_reasons = _observation_trust(result)
            if not attempt_trusted and observation_attempt < max_observation_attempts:
                health = result.observation_health
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=(
                        "RETRY_OBSERVATION "
                        f"attempt={observation_attempt}/{max_observation_attempts}; "
                        f"reasons={'; '.join(attempt_reasons)}; "
                        f"unknown={health.unknown_cells}; "
                        f"capacity_remaining={health.capacity_remaining_by_color}; "
                        f"capacity_excess={health.capacity_excess_by_color}"
                    ),
                )
                time.sleep(retry_delay)
                continue

            break

        # analyze_image either returns a structured observation or raises a fatal error.
        assert best_observation_result is not None
        assert best_observation_shot is not None
        result = best_observation_result
        shot = best_observation_shot
        analysis_bgr = best_observation_bgr

        observation_trusted, untrusted_reasons = _observation_trust(result)
        untrusted_reason_text = "; ".join(untrusted_reasons) or "none"

        if not observation_trusted and not experimental_continue:
            result.report += (
                "\n\n[NO_CLICK_UNTRUSTED]\n"
                f"reasons={untrusted_reason_text}\n"
                "严格模式拒绝本轮 observation：不提交 solver_state、不结束上一动作 "
                "checkpoint、不进入候选点击；继续重新观测。"
            )
            print(result.report)
            _append_decision_log(
                log_path,
                screenshot=shot,
                result=result,
                step_label=step_label,
            )
            _append_observation_table_logs(
                color_log_path,
                number_log_path,
                screenshot=shot,
                result=result,
                step_label=step_label,
            )
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "NO_CLICK_UNTRUSTED; "
                    f"reasons={untrusted_reason_text}; "
                    "committed-state-unchanged; pending-checkpoint-preserved"
                ),
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="NO_CLICK_UNTRUSTED",
                    hint=(
                        "重试后 observation 仍不可信；保持上一份 trusted state，"
                        "不点击，继续重新观测。"
                    ),
                )
            print(
                "NO_CLICK_UNTRUSTED：重试后 observation 仍不可信；"
                "未提交状态、未结束 checkpoint、未执行点击。"
            )
            time.sleep(retry_delay)
            continue

        if not observation_trusted:
            result.report += (
                "\n\n[EXPERIMENT_CONTINUE_UNTRUSTED]\n"
                f"reasons={untrusted_reason_text}\n"
                "--experimental-continue 已显式启用；提交当前保守 grid，"
                "且本轮只允许单步候选。"
            )

        # Trusted observation, or one explicit experimental opt-in, advances context.
        _commit_analysis_result_state(args.state, result)
        if analysis_bgr is not None:
            committed_analysis_rgb = cv2.cvtColor(
                analysis_bgr,
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)

        reset_current = False
        pending_flow_capacity_before_by_color = None
        pending_prediction = None

        print(result.report)
        print(
            f"OCR信息: 本轮第一排直接 OCR {result.front_ocr_reads} 列；"
            "每个稳定 observation 均重新读取，不跨车辆复用旧数字。"
        )

        _append_decision_log(
            log_path,
            screenshot=shot,
            result=result,
            step_label=step_label,
        )
        _append_observation_table_logs(
            color_log_path,
            number_log_path,
            screenshot=shot,
            result=result,
            step_label=step_label,
        )

        if not observation_trusted:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "EXPERIMENT_CONTINUE_UNTRUSTED; "
                    f"reasons={untrusted_reason_text}; single-step-only"
                ),
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="实验继续 · 不可信观测",
                    hint=(
                        "--experimental-continue 已显式启用；"
                        "当前保守状态已提交，本轮仅允许单步候选。"
                    ),
                )
            print(
                "EXPERIMENT_CONTINUE_UNTRUSTED：显式实验开关已启用；"
                "当前轮仅执行单步候选。"
            )

        if len(result.front) == 0 and len(result.nxt) == 0:
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="胜利确认（兜底）",
                    hint=(
                        "进入分析前排队区已经为空；"
                        "这通常表示上一轮点击已经点完最后一辆车。"
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
                    hint=(
                        "排队区连续两次无车辆；"
                        "按最后一辆排队车已点击规则判定本局结束。"
                        "停车区是否仍有车不参与胜利判定。"
                    ),
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

        if result.occupied_slots >= args.slots:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    f"HARD_STOP stable parking full "
                    f"{result.occupied_slots}/{args.slots}"
                ),
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="HARD_STOP",
                    hint=(
                        f"稳定停车已经 {result.occupied_slots}/{args.slots}；"
                        "按已确认规则本局已经失败，停止自动流程。"
                    ),
                )
            print(
                f"HARD_STOP：稳定停车已经 "
                f"{result.occupied_slots}/{args.slots}，本局失败。"
            )
            return 2

        if result.best is None:
            # v5.12: best_valid_candidate 同时承担稳定停车 hard veto。
            # 没有安全候选时必须保持不点击，而不是为了实验继续送车入场。
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution="NO_SAFE_CANDIDATE no-click; reobserve",
            )
            print(
                "当前没有可安全执行的候选；保持不点击并重新观测。"
                "若候选均 rejected，表示点击会导致稳定停车满位。"
            )
            time.sleep(
                max(
                    0.15,
                    float(getattr(args, "observation_retry_delay", 0.45)),
                )
            )
            continue

        if args.tap_delay > 0:
            time.sleep(args.tap_delay)

        plan = result.two_step_plan
        execution_parts = []
        executed_actions = []
        known_queue_count = len(result.front) + len(result.nxt)
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
            executed_actions.append(
                (int(plan.first.color), int(plan.first.capacity))
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
                    executed_actions.append(
                        (int(plan.second.color), int(plan.second.capacity))
                    )
                    print(
                        f"连续两步 2/2: 同列补位确认通过，第 {plan.second.column} 列 "
                        f"{ctag(plan.second.color)}×{plan.second.capacity}，"
                        f"坐标=({x2}, {y2})"
                    )
                else:
                    if _tap_proven_not_departed(
                        _confirm_frame,
                        result,
                        plan.first,
                    ):
                        # v5.12 click confirmation: ADB tap was sent but the old
                        # first-row car is still there. Do not invent capacity.
                        if executed_actions and executed_actions[-1] == (
                            int(plan.first.color),
                            int(plan.first.capacity),
                        ):
                            executed_actions.pop()
                        execution_parts.append(
                            "step1 CLICK_NOT_CONFIRMED: original car still front; "
                            "step2-next cancelled"
                        )
                        print(
                            "第一步点击未生效：原车仍在第一排；"
                            "不计入 executed_actions，不建立虚假容量。"
                        )
                    else:
                        execution_parts.append(
                            "step2-next cancelled: promote confirmation failed; "
                            "step1 departure not disproven"
                        )
                        print(
                            "连续两步第二步取消：补位未确认；"
                            "但当前画面不能证明第一步漏点。"
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
                executed_actions.append(
                    (int(plan.second.color), int(plan.second.capacity))
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
            # v5.12 click confirmation: when the same column has a
            # distinguishable known next car, briefly confirm that the original car
            # did not simply remain after a missed ADB tap.
            single_confirm_failed = False
            if _has_distinguishable_known_next(result, result.best):
                confirm_deadline = time.monotonic() + min(
                    1.2,
                    max(0.35, args.queue_promote_timeout),
                )
                last_confirm_frame = None
                while time.monotonic() < confirm_deadline:
                    last_confirm_frame = adb_capture_bgr(args.serial)
                    if not _tap_proven_not_departed(
                        last_confirm_frame,
                        result,
                        result.best,
                    ):
                        break
                    time.sleep(max(0.03, args.queue_promote_poll_interval))
                if _tap_proven_not_departed(
                    last_confirm_frame,
                    result,
                    result.best,
                ):
                    single_confirm_failed = True

            if single_confirm_failed:
                execution_parts.append(
                    "CLICK_NOT_CONFIRMED: original car still front"
                )
                print(
                    "ADB 点击已发送但原车仍在第一排；"
                    "不计入 executed_actions，重新观测。"
                )
            else:
                executed_actions.append(
                    (int(result.best.color), int(result.best.capacity))
                )
                print(
                    f"自动点击完成: 第一排第 {result.best.column} 列 "
                    f"{ctag(result.best.color)}×{result.best.capacity}，"
                    f"坐标=({x}, {y})"
                )

        if not executed_actions:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "; ".join(execution_parts)
                    + "; CLICK_NOT_CONFIRMED no executed action"
                ),
            )
            print(
                "本轮没有确认成功的点击；立即重新观测，"
                "不进入停车监控，也不建立容量 checkpoint。"
            )
            time.sleep(max(0.1, args.queue_promote_poll_interval))
            continue

        last_queue_car_clicked = (
            known_queue_count > 0
            and len(executed_actions) >= known_queue_count
        )
        if last_queue_car_clicked:
            execution_parts.append(
                "last-queue-clicked-known-before-monitor "
                f"known_queue={known_queue_count} "
                f"executed={len(executed_actions)}"
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

        # v5.11 primary completion path: if the pre-click analysis proved that
        # this click batch consumes every remaining queue entry, the confirmed
        # game rule says the level is complete after this batch's parking flow
        # stabilizes. Do not require post-click queue OCR.
        if last_queue_car_clicked:
            final_bgr = adb_capture_bgr(args.serial)
            final_path = (
                args.shots_dir
                / f"game_complete_last_click_{shot_stamp()}.png"
            )
            save_bgr(final_path, final_bgr)
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "game-complete last queued car clicked directly; "
                    f"known_queue={known_queue_count}; "
                    f"executed={len(executed_actions)}; "
                    f"final={final_path.name}; "
                    f"parking_before_click={result.occupied_slots}/{args.slots}"
                ),
            )
            display.show(
                final_bgr,
                stage="本局完成",
                hint=(
                    "本轮已经点击完分析时剩余的全部排队车辆，"
                    "且停车分流已稳定；按已确认规则本局结束。"
                    "停车位仍有车不影响胜利判定。"
                ),
            )
            print(
                "本轮已点击完最后一辆排队车，且停车分流已稳定；"
                "按已确认规则本局结束。"
                f"最终截图: {final_path}"
            )
            return 0

        # Recovery fallback only: if the pre-click model did not prove queue
        # exhaustion, keep the old two-frame empty visual check.
        queue_check_bgr = adb_capture_bgr(args.serial)
        queue_empty1, _queue_added1 = queue_empty_on_image(
            queue_check_bgr,
            result.palette,
        )

        if queue_empty1:
            if args.queue_empty_confirm_delay > 0:
                time.sleep(args.queue_empty_confirm_delay)

            queue_confirm_bgr = adb_capture_bgr(args.serial)
            queue_empty2, _queue_added2 = queue_empty_on_image(
                queue_confirm_bgr,
                result.palette,
            )
            queue_confirm_path = (
                args.shots_dir
                / f"queue_after_click_confirm_{shot_stamp()}.png"
            )
            save_bgr(queue_confirm_path, queue_confirm_bgr)

            if queue_empty2:
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=(
                        "game-complete last queued car clicked; "
                        f"confirm={queue_confirm_path.name}; "
                        f"parking_before_last_click={result.occupied_slots}/{args.slots}"
                    ),
                )
                display.show(
                    queue_confirm_bgr,
                    stage="本局完成",
                    hint=(
                        "上一轮点击后排队区连续两次为空："
                        "已点击完排队区最后一辆车。"
                        "停车位仍有车不影响胜利判定。"
                    ),
                )
                print(
                    "排队区在上一轮点击后连续两次为空；"
                    "确认已点击完最后一辆排队车，本局结束。"
                    "停车状态不参与胜利判定。"
                    f"确认截图: {queue_confirm_path}"
                )
                return 0

            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "queue-after-click-empty-cancelled; "
                    f"confirm={queue_confirm_path.name}; "
                    "second check found queued vehicles"
                ),
            )
        else:
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "queue-after-click-fallback first_empty=False; "
                    f"known_queue={known_queue_count}; "
                    f"executed={len(executed_actions)}"
                ),
            )

        pending_flow_capacity_before_by_color = (
            _capacity_before_flow_by_color(
                result.parked,
                executed_actions,
            )
        )
        causal_basis = ", ".join(
            f"{ctag(color)}={capacity}"
            for color, capacity in sorted(
                pending_flow_capacity_before_by_color.items()
            )
        ) or "无"
        _append_execution_update(
            log_path,
            screenshot=shot,
            step_label=step_label,
            execution=(
                "causal_capacity_checkpoint "
                f"before_total_remaining_by_color={causal_basis}"
            ),
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