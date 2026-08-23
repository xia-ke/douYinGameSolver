from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .board import ObservedBoard, ctag, reachable_summary
from .config import EMPTY, UNKNOWN
from .models import AnalysisResult, Candidate, Car, TwoStepPlan


# Presentation mirror of strategy's execution threshold.  This value is used
# only in human-readable plan text; strategy.py remains the rule authority.
_TWO_STEP_MIN_FREE_SLOTS = 3


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

def format_two_step_plan(plan: Optional[TwoStepPlan]) -> str:
    if plan is None:
        return ""

    if plan.second_source == "next":
        second_text = (
            f"随后等待同列第二排 {ctag(plan.second.color)}×{plan.second.capacity} "
            "顶到第一排，颜色+数字确认后立即点击"
        )
    else:
        second_text = (
            f"随后不等分流结束直接点击当前第一排第{plan.second.column}列 "
            f"{ctag(plan.second.color)}×{plan.second.capacity}"
        )

    exact_text = "确定性闭包" if plan.flow_exact else "保守闭包"
    return (
        f"连续两步计划: 当前空位 {plan.free_slots_before} >= "
        f"{_TWO_STEP_MIN_FREE_SLOTS}，"
        f"先点击第{plan.first.column}列 {ctag(plan.first.color)}×{plan.first.capacity}；"
        f"{second_text}。\n"
        f"联合动作预测: {exact_text}；稳定后停车占用上界 "
        f"{plan.final_occupied_upper}；保证完成 {plan.guaranteed_completions} 辆，"
        f"已有停车车至少完成 {plan.guaranteed_parked_completions} 辆；"
        f"确定性清除 {plan.cleared_cells} 个色块。\n"
        f"词典序 utility: {plan.utility_reason}\n"
        f"两步预测依据: {plan.reason}"
    )

def format_report(
    grid: np.ndarray,
    front: Sequence[Car],
    nxt: Sequence[Car],
    parked: Sequence[Car],
    candidates: Sequence[Candidate],
    removed_since_last: Optional[int],
    slots: int,
    palette_count: int,
    occupied_slots: int,
    new_colors_added: int,
) -> str:
    reachable, _neighbors = reachable_summary(grid)
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("程序分析结果（当前截图 + 持久化棋盘状态）")
    lines.append("=" * 72)

    known = int(np.sum(grid > 0))
    empty = int(np.sum(grid == EMPTY))
    unknown = int(np.sum(grid == UNKNOWN))
    lines.append(
        f"本关动态颜色类别: {palette_count} 种"
        + (f"（本帧新增 {new_colors_added} 种）" if new_colors_added else "")
    )
    lines.append(f"棋盘: 已知色块 {known} | 已空 {empty} | UI/未知 {unknown}")
    if removed_since_last is not None:
        lines.append(f"相对上一张完整分析截图新消失: {removed_since_last} 个小色块")

    if reachable:
        reach_text = ", ".join(
            f"{ctag(c)}={n}"
            for c, n in sorted(
                reachable.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        lines.append(f"当前从下方可连续触达: {reach_text}")
    else:
        lines.append("当前从下方没有识别到可触达的已知颜色区域")

    lines.append("")
    front_map = {c.column: c for c in front if c.column is not None}
    next_map = {c.column: c for c in nxt if c.column is not None}
    max_col = max(
        [0]
        + [int(c) for c in front_map.keys()]
        + [int(c) for c in next_map.keys()]
    )
    for col in range(1, max_col + 1):
        f = front_map.get(col)
        n = next_map.get(col)
        if f:
            if n:
                next_text = (
                    f"{ctag(n.color)} / 数字 "
                    f"{n.remain if n.remain is not None else '浅色OCR未确认'}"
                )
            else:
                next_text = "UNKNOWN"
            lines.append(
                f"第{col}列第一排: {ctag(f.color)} / 数字 "
                f"{f.remain if f.remain is not None else '识别失败'}"
                f"    第二排: {next_text}"
            )
        else:
            lines.append(f"第{col}列第一排: 未检测到可点击车辆")

    lines.append("")
    lines.append(f"停车位: 数字锚点检测 {occupied_slots} / {slots}")
    for i, car in enumerate(parked, 1):
        lines.append(
            f"  停车车{i}: {ctag(car.color)} / 剩余 "
            f"{car.remain if car.remain is not None else '识别失败'}"
        )

    lines.append("")
    lines.append(
        "策略排序: 先执行稳定停车 < slots 的硬安全过滤；安全候选再按 "
        "已有停车车释放 → 总保证完成 → 确定性清除 → 有用新暴露 → 队列推进 "
        "做词典序比较。nearest 预测只允许进入最后的小型 tie-break。"
    )
    lines.append("候选动作（安全候选已按 lexicographic utility 排序）:")
    for c in candidates:
        flags: List[str] = []
        if c.self_clear_guaranteed:
            flags.append("确定该点击车自己装满")
        if c.guaranteed_completions > 0:
            flags.append(f"闭包保证完成 {c.guaranteed_completions} 辆")
        if c.chain_parked_completions > 0:
            detail = ", ".join(
                f"{ctag(color)}×{count}"
                for color, count in c.chain_parked_completion_by_color.items()
            )
            flags.append(
                f"已有停车车至少完成 {c.chain_parked_completions} 辆"
                + (f"（{detail}）" if detail else "")
            )
        if c.deterministic_clear_reachable:
            flags.append("确定清空当前可达同色")
        if c.queue_progress >= 2:
            flags.append("同列第二排存在稳定安全的继续推进路径")
        if c.rejected:
            flags.append("硬安全拒绝（稳定后停车将满位）")
        if c.next_color is not None:
            if c.next_capacity is not None:
                flags.append(f"第二排 {ctag(c.next_color)}×{c.next_capacity}")
            else:
                flags.append(f"第二排 {ctag(c.next_color)}×?（浅色OCR未确认）")

        lines.append(
            f"  第{c.column}列 {ctag(c.color)}×{c.capacity}: "
            f"可触达={c.reachable}, utility={c.utility}, "
            f"稳定占用上界={c.flow_final_occupied_upper}/{slots}"
            + (" | " + "；".join(flags) if flags else "")
        )
        lines.append(f"      词典序依据: {c.utility_reason}")

        if c.unlocked_by_color:
            exposure_text = ", ".join(
                f"{ctag(k)}+{v}" for k, v in c.unlocked_by_color.items()
            )
            lines.append(f"      确定性新暴露: {exposure_text}")

        if not c.flow_exact:
            lines.append(
                "      闭包说明: 至少一种颜色容量不足以清空当前可达区；"
                "只记确定消费数量，不猜空间删除位置。"
            )
        if c.rejected:
            lines.append(f"      原因: {c.reject_reason}")

    valid = [c for c in candidates if not c.rejected]
    lines.append("")
    if not valid:
        lines.append("建议: 不点击。当前没有满足稳定状态硬安全约束的单步候选。")
    else:
        best = max(valid, key=lambda c: c.utility)
        lines.append(
            f"下一步建议: 点击【第一排第 {best.column} 列】 "
            f"{ctag(best.color)} / 数字 {best.capacity}"
        )
        lines.append(
            f"程序依据: {best.utility_reason}；"
            f"稳定后停车占用最坏上界 {best.flow_final_occupied_upper}/{slots}。"
        )
        if best.chain_parked_completions > 0:
            lines.append(
                f"停车释放依据: 按已确认的同色车‘剩余数字小者优先’规则，"
                f"保证已有停车车至少完成 {best.chain_parked_completions} 辆。"
            )
        if best.queue_progress >= 2 and best.next_capacity is not None:
            lines.append(
                f"队列依据: 同列第二排 {ctag(best.next_color)}×{best.next_capacity} "
                "存在稳定安全的联合推进路径；该信息只位于前四个确定性 utility "
                "项之后。"
            )

    return "\n".join(lines)
