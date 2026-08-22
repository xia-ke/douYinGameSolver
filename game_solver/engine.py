from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .adb import (
    adb_capture_bgr, adb_screencap, adb_tap, save_bgr, shot_stamp,
)
from .board import (
    CausalBoardUpdate, ctag, initial_grid, learn_palette,
    reachable_summary, sample_grid_rgb_snapshot, update_grid, update_grid_causal,
)
from .config import FRONT_Y_N, UNKNOWN
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
from .state import load_state_with_grid_rgb, save_state
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


# ---------------------------------------------------------------------------
# v5.6 运行时安全状态机
# ---------------------------------------------------------------------------
# 目标不是“发现一次异常就退出 Python”，而是：
#   NORMAL -> RETRY_OBSERVATION -> PAUSED_SYNC / DEGRADED -> PAUSED_SAFE -> HARD_STOP
#
# 常态下尽量自动运行；局部模型不可信时隔离对应颜色、关闭多步和停车连锁
# guarantee；只有持续且无法恢复的高风险状态才真正退出。
@dataclass
class _SafetyRuntime:
    mode: str = "NORMAL"
    untrusted_colors: Set[int] = field(default_factory=set)
    clean_streak: int = 0
    conservative_turns: int = 0
    board_incomplete_streak: int = 0
    guarantee_break_rounds: List[int] = field(default_factory=list)
    pause_reason: str = ""

    @property
    def force_single_step(self) -> bool:
        return self.mode != "NORMAL" or self.conservative_turns > 0

    @property
    def disable_parked_chain(self) -> bool:
        return self.mode != "NORMAL" or self.conservative_turns > 0

    def degrade(
        self,
        *,
        colors=(),
        reason: str,
        turns: int = 3,
    ) -> None:
        self.untrusted_colors.update(int(c) for c in colors if int(c) > 0)
        if self.mode != "PAUSED_SAFE":
            self.mode = "DEGRADED"
        self.conservative_turns = max(self.conservative_turns, int(turns))
        self.clean_streak = 0
        self.pause_reason = reason

    def pause(self, reason: str) -> None:
        self.mode = "PAUSED_SAFE"
        self.pause_reason = reason
        self.clean_streak = 0

    def pause_sync(self, reason: str) -> None:
        # 当前动作的棋盘转移尚未完整落盘。此状态严禁产生新点击，
        # 也不把 unresolved consumption 跨动作累计。
        self.mode = "PAUSED_SYNC"
        self.pause_reason = reason
        self.clean_streak = 0
        self.board_incomplete_streak += 1


    def note_model_conflict(self, colors) -> None:
        self.degrade(
            colors=colors,
            reason="稳定停车车与 reachable 冲突",
            turns=3,
        )

    def note_guarantee_broken(self, round_no: int) -> None:
        self.guarantee_break_rounds.append(int(round_no))
        self.guarantee_break_rounds = [
            r for r in self.guarantee_break_rounds
            if int(round_no) - r < 5
        ]
        if len(self.guarantee_break_rounds) >= 2:
            self.pause(
                "最近5轮内至少2次 guaranteed parking upper 被真实结果击穿"
            )
        else:
            self.degrade(
                reason="guaranteed parking upper 被真实结果击穿",
                turns=3,
            )

    def note_clean(self) -> None:
        self.board_incomplete_streak = 0
        self.clean_streak += 1
        if self.conservative_turns > 0:
            self.conservative_turns -= 1

        if self.mode == "PAUSED_SYNC":
            # 当前转移终于完整同步。先以单步/自完成门槛运行一小段，
            # 避免刚恢复时立刻重新使用激进多步。
            self.mode = "DEGRADED"
            self.conservative_turns = max(self.conservative_turns, 2)
            self.pause_reason = "棋盘转移已重新同步，暂以保守模式恢复"
            self.clean_streak = 0
            return

        if self.mode == "PAUSED_SAFE":
            # PAUSED_SAFE 不自动跳回 NORMAL。连续两张稳定、无冲突截图后，
            # 先恢复到 DEGRADED，再观察3轮。
            if self.clean_streak >= 2:
                self.mode = "DEGRADED"
                self.conservative_turns = max(self.conservative_turns, 3)
                self.pause_reason = "观测连续恢复，先以保守模式继续"
                self.clean_streak = 0
            return

        if (
            self.mode == "DEGRADED"
            and self.clean_streak >= 3
            and self.conservative_turns <= 0
        ):
            self.mode = "NORMAL"
            self.untrusted_colors.clear()
            self.pause_reason = ""
            self.clean_streak = 0

    def status_line(self) -> str:
        colors = ",".join(
            ctag(c) for c in sorted(self.untrusted_colors)
        ) or "无"
        return (
            f"安全模式={self.mode}; "
            f"不可信颜色={colors}; "
            f"保守剩余轮={self.conservative_turns}; "
            f"原因={self.pause_reason or '无'}"
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


def _format_causal_board_update(
    update: CausalBoardUpdate,
) -> str:
    if not update.expected_by_color and update.complete:
        return "因果棋盘更新: 本轮容量守恒确认没有色块被吸收，棋盘不应发生消失。"

    expected = ", ".join(
        f"{ctag(color)}={count}"
        for color, count in update.expected_by_color.items()
    ) or "无"
    confirmed = ", ".join(
        f"{ctag(color)}={count}"
        for color, count in update.confirmed_by_color.items()
    ) or "无"

    lines = [
        "因果棋盘更新:",
        f"  数学确定实际吸收: {expected}",
        f"  拓扑+截图已落实为空: {confirmed}",
        f"  本轮检查 reachable 同色连通块候选 {update.checked_cells} 个格子",
        (
            "  temporal snapshot: "
            + ("可用" if update.temporal_snapshot_available else "不可用（退回背景判定）")
            + f"，变化阈值={update.temporal_change_threshold:.1f}"
        ),
    ]

    if update.background_confirmed_by_color:
        bg = ", ".join(
            f"{ctag(color)}={count}"
            for color, count in update.background_confirmed_by_color.items()
        )
        lines.append(f"  灰背景直接确认: {bg}")

    if update.temporal_confirmed_by_color:
        temporal = ", ".join(
            f"{ctag(color)}={count}"
            for color, count in update.temporal_confirmed_by_color.items()
        )
        lines.append(f"  时间差分确认: {temporal}")

    lines.append(
        "  local patch previous frame: "
        + ("可用" if update.patch_previous_frame_available else "不可用")
        + f"，主体覆盖下降阈值={update.patch_coverage_drop_threshold:.2f}"
    )
    if update.patch_confirmed_by_color:
        patch = ", ".join(
            f"{ctag(color)}={count}"
            for color, count in update.patch_confirmed_by_color.items()
        )
        lines.append(f"  局部主体消失确认: {patch}")

    if update.strong_nonfrontier_confirmed_by_color:
        strong = ", ".join(
            f"{ctag(color)}={count}"
            for color, count
            in update.strong_nonfrontier_confirmed_by_color.items()
        )
        lines.append(f"  强视觉非frontier确认: {strong}")

    if update.ambiguous_changed_by_color:
        ambiguous = ", ".join(
            f"{ctag(color)}多{count}"
            for color, count in update.ambiguous_changed_by_color.items()
        )
        lines.append(
            "  额外变化候选（仅telemetry，容量数学已裁剪）: "
            + ambiguous
        )

    if update.remaining_by_color:
        missing = ", ".join(
            f"{ctag(color)}缺{count}"
            for color, count in update.remaining_by_color.items()
        )
        lines.append(f"  未落实预算: {missing}")

    if update.excess_by_color:
        excess = ", ".join(
            f"{ctag(color)}多{count}"
            for color, count in update.excess_by_color.items()
        )
        lines.append(f"  超预算空格: {excess}")

    if update.invalid_reason:
        lines.append(f"  更新异常: {update.invalid_reason}")

    return "\n".join(lines)


def analyze_image(
    image_path: Path,
    state_path: Path,
    reset: bool,
    slots: int,
    front_number_cache: Optional[Dict[int, FrontNumberCacheEntry]] = None,
    flow_capacity_before_by_color: Optional[Dict[int, int]] = None,
    strategy_untrusted_colors: Optional[Set[int]] = None,
    force_single_step: bool = False,
    disable_parked_chain: bool = False,
    previous_prediction_upper: Optional[int] = None,
    previous_prediction_basis: str = "",
    commit_state: bool = True,
    prev_grid_image_rgb: Optional[np.ndarray] = None,
) -> AnalysisResult:
    """
    分析一张已经稳定的游戏截图。

    v5.7 关键变化：
      - 因果棋盘异常只返回结构化状态，不在这里直接“杀进程”；
      - MODEL_INCONSISTENT 只隔离冲突颜色；
      - GUARANTEE_BROKEN 当前轮直接切换保守策略；
      - 由自动运行层负责重试、降级、PAUSED_SAFE 与最终 HARD_STOP。
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    image_h, image_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)
    current_grid_rgb = sample_grid_rgb_snapshot(image_rgb)

    front_centers = detect_front_centers(image_bgr)

    # 安全模式下暂不跨车辆复用数字缓存。
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
    causal_update: Optional[CausalBoardUpdate] = None
    causal_input_invalid = ""
    board_update_status = "ok"
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

        grid = initial_grid(image_rgb, palette)
        turn = 0
        removed_since_last = None
        parking_empty_ref = parking_roi(image_bgr)

    else:
        (
            palette,
            prev_grid,
            turn,
            saved_size,
            parking_empty_ref,
            prev_grid_rgb,
        ) = load_state_with_grid_rgb(state_path)
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

        # 车辆状态先于棋盘因果更新读取，因为“下一稳定停车剩余”是容量守恒右端。
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
        if flow_capacity_before_by_color is not None:
            expected_consumed, causal_input_invalid = (
                _actual_consumed_from_capacity_delta(
                    flow_capacity_before_by_color,
                    parked,
                )
            )

        # unresolved consumption 只属于“上一动作 -> 当前稳定截图”这一次转移。
        # 若无法完整定位，实验模式记录异常并提交保守 grid；未落实数量不跨动作 carry。
        has_causal_context = flow_capacity_before_by_color is not None

        if causal_input_invalid:
            # 容量守恒本身失败：不能拿任何数量修改棋盘。
            grid = prev_grid.copy()
            removed_since_last = 0
            board_update_status = "causal_invalid"
        elif has_causal_context:
            causal_update = update_grid_causal(
                prev_grid,
                image_rgb,
                palette,
                expected_consumed,
                prev_grid_rgb=prev_grid_rgb,
                prev_image_rgb=prev_grid_image_rgb,
            )
            grid = causal_update.grid
            removed_since_last = causal_update.removed
            if not causal_update.complete:
                board_update_status = "incomplete"
        else:
            # 单图/人工兼容路径。
            grid, removed_since_last = update_grid(
                prev_grid,
                image_rgb,
                palette,
            )

    occupied_slots = len(parked)

    if occupied_slots > slots:
        raise RuntimeError(
            f"停车数字锚点检测到 {occupied_slots} 辆，超过设定停车位 {slots}，"
            "状态已不可解释。"
        )

    incomplete_parked = [
        c for c in parked
        if c.color is None or c.remain is None
    ]
    if incomplete_parked:
        raise RuntimeError(
            "停车数字已检测到，但有停车车的颜色/剩余数字无法可靠识别。"
        )

    if front and all(c.remain is None for c in front):
        raise RuntimeError(
            "第一排车辆数字全部识别失败。"
        )

    stable_conflicts: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
    if board_update_status == "ok":
        stable_conflicts = _stable_state_conflicts(grid, parked)

    guarantee_broken = bool(
        previous_prediction_upper is not None
        and occupied_slots > int(previous_prediction_upper)
    )

    # ---------- v5.8 实验模式 ----------
    #
    # 所有异常都保留为 telemetry，不再从 strategy_grid 中删除颜色、
    # 不再把 Candidate.rejected 当作禁止点击条件。
    #
    # 原因：当前项目已经具备完整 decision_log，现阶段更需要持续收集
    # “模型预测 vs 真实下一帧”的偏差，而不是在轻微矛盾处停止运行。
    untrusted: Set[int] = {
        int(c)
        for c in (strategy_untrusted_colors or set())
        if int(c) > 0
    }
    untrusted.update(stable_conflicts)

    if disable_parked_chain or guarantee_broken:
        untrusted.update(
            int(car.color)
            for car in parked
            if car.color is not None and int(car.color) > 0
        )

    strategy_grid = grid.copy()

    # 异常状态下最多降为单步实验，避免一次叠加过多变量；
    # 但绝不阻止游戏继续点击。
    local_force_single = bool(
        force_single_step
        or guarantee_broken
        or stable_conflicts
        or board_update_status != "ok"
    )

    candidates = evaluate_candidates(
        strategy_grid,
        front,
        nxt,
        parked,
        slots,
        occupied_slots,
        include_queue_lookahead=not local_force_single,
    )

    if local_force_single:
        two_step_plan = None
    else:
        two_step_plan = choose_two_step_plan(
            strategy_grid,
            front,
            nxt,
            parked,
            slots,
            occupied_slots,
            candidates,
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

    if causal_update is not None:
        report += "\n\n" + _format_causal_board_update(causal_update)

    if causal_input_invalid:
        report += (
            "\n\n!!! CAUSAL_CAPACITY_INVALID / RETRY_OBSERVATION !!!\n"
            f"{causal_input_invalid}\n"
            "本次不修改持久化棋盘、不点击；自动运行层会重新截图尝试恢复，"
            "不会立即退出程序。"
        )

    palette_diag = _format_palette_diagnostics(palette)
    if palette_diag:
        report += (
            "\n\n颜色类别诊断（最近 palette 色对）:\n"
            + palette_diag
        )

    if causal_update is not None and not causal_update.complete:
        problem_colors = sorted(
            set(causal_update.remaining_by_color)
            | set(causal_update.excess_by_color)
        )
        report += (
            "\n\n!!! BOARD_UPDATE_INCOMPLETE / EXPERIMENT_WARNING !!!\n"
            "容量守恒预算尚未完整落实到具体格子。"
            "实验模式会提交当前已确认的保守棋盘并继续单步运行；"
            "未落实数量只写入日志，不跨动作 carry，也不阻止下一次点击。"
            f"\n问题颜色: "
            f"{', '.join(ctag(c) for c in problem_colors) or '未知'}"
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
            "\n\n!!! MODEL_INCONSISTENT / EXPERIMENT_WARNING !!!\n"
            f"矛盾项: {conflict_text}\n"
            "仅记录模型矛盾，不隔离颜色、不 veto 候选。"
            "当前轮最多降为单步，继续收集下一稳定截图验证模型。"
        )

    if guarantee_broken:
        report += (
            "\n\n!!! GUARANTEE_BROKEN / EXPERIMENT_WARNING !!!\n"
            f"上一轮保证稳定停车占用 <= {previous_prediction_upper}，"
            f"当前稳定截图实际为 {occupied_slots}/{slots}。\n"
            f"上一轮依据: {previous_prediction_basis or '未记录'}\n"
            "该结果只作为模型校准证据写入日志；当前轮继续运行，"
            "最多降为单步实验。"
        )

    if untrusted:
        report += (
            "\n\n模型不可信颜色（仅日志标记，不阻止点击）: "
            + ", ".join(ctag(c) for c in sorted(untrusted))
        )

    if two_step_plan is not None:
        report += "\n\n" + format_two_step_plan(two_step_plan)
        best = two_step_plan.first
    else:
        best = best_valid_candidate(candidates)

    # v5.9：
    # analyze_image 在观测重试期间必须是“只读 committed state”的纯分析。
    # 所有 retry 都从同一个 solver_state 重新计算，只有调用方最终选定一次
    # observation 后才 commit。单图模式仍使用默认 commit_state=True。
    turn += 1
    state_saved = bool(commit_state)
    if commit_state:
        save_state(
            state_path,
            palette,
            grid,
            turn,
            image_w,
            image_h,
            parking_empty_ref,
            grid_rgb_snapshot=current_grid_rgb,
        )

    remaining_by_color = (
        dict(causal_update.remaining_by_color)
        if causal_update is not None
        else {}
    )
    excess_by_color = (
        dict(causal_update.excess_by_color)
        if causal_update is not None
        else {}
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
        board_update_status=board_update_status,
        board_update_remaining_by_color=remaining_by_color,
        board_update_excess_by_color=excess_by_color,
        causal_input_invalid=causal_input_invalid,
        model_conflict_colors=sorted(stable_conflicts),
        strategy_untrusted_colors=sorted(untrusted),
        guarantee_broken=guarantee_broken,
        guarantee_expected_upper=previous_prediction_upper,
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



def _retriable_analysis_error(exc: RuntimeError) -> bool:
    text = str(exc)
    return any(
        token in text
        for token in (
            "停车数字已检测到",
            "第一排车辆数字全部识别失败",
            "无法可靠识别",
        )
    )



def _commit_analysis_result_state(
    state_path: Path,
    result: AnalysisResult,
) -> None:
    """
    v5.9 retry-safe commit：
    一轮多个 observation 全部只读同一个 committed solver_state；
    最终只把选中的一次结果写盘。
    """
    save_state(
        state_path,
        result.palette,
        result.grid,
        result.turn,
        result.image_w,
        result.image_h,
        result.parking_empty_ref,
        grid_rgb_snapshot=result.grid_rgb_snapshot,
    )
    result.state_saved = True


def _observation_quality(result: AnalysisResult) -> Tuple[int, int, int]:
    """
    越小越好。用于多次稳定观测中选择最完整的一次，而不是机械使用最后一次。

    0: 完整同步
    1: incomplete，按剩余预算总量排序
    2: causal_invalid
    """
    if result.board_update_status == "ok":
        return (0, 0, -int(result.grid.size))
    if result.board_update_status == "incomplete":
        remaining = sum(
            max(0, int(v))
            for v in result.board_update_remaining_by_color.values()
        )
        excess = sum(
            max(0, int(v))
            for v in result.board_update_excess_by_color.values()
        )
        return (1, remaining + excess, -int(np.count_nonzero(result.grid == 0)))
    return (2, 10**9, 0)

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
    pending_flow_capacity_before_by_color: Optional[Dict[int, int]] = None
    committed_analysis_rgb: Optional[np.ndarray] = None
    safety = _SafetyRuntime()
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
    print(
        "实验模式: observation retry 只读同一 committed state；"
        "最终只提交一次最完整观测，异常继续写日志而不形成策略 veto。"
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

        result: Optional[AnalysisResult] = None
        analysis_bgr: Optional[np.ndarray] = None
        shot: Optional[Path] = None
        last_retry_error = ""

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
                        else "上一张观测存在可恢复异常；重新截图确认，不执行点击。"
                    ),
                )

            try:
                result = analyze_image(
                    shot,
                    args.state,
                    reset=(reset_current or not args.state.exists()),
                    slots=args.slots,
                    front_number_cache=front_number_cache,
                    flow_capacity_before_by_color=flow_context,
                    strategy_untrusted_colors=safety.untrusted_colors,
                    force_single_step=safety.force_single_step,
                    disable_parked_chain=safety.disable_parked_chain,
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
                    # v5.9：retry 期间禁止写 solver_state。
                    # 所有 attempt 必须从同一个 committed state 重算。
                    commit_state=False,
                    prev_grid_image_rgb=committed_analysis_rgb,
                )
            except RuntimeError as exc:
                if not _retriable_analysis_error(exc):
                    raise

                last_retry_error = str(exc)
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=(
                        "RETRY_OBSERVATION "
                        f"attempt={observation_attempt}/{max_observation_attempts}; "
                        f"reason={last_retry_error}"
                    ),
                )
                if observation_attempt < max_observation_attempts:
                    time.sleep(retry_delay)
                    continue

                safety.pause(
                    "停车/第一排 OCR 连续多次无法形成可信稳定状态"
                )
                result = None
                break

            # 记录当前轮最完整的一次观测。即使后续 retry 更差，也不会覆盖它。
            if (
                best_observation_result is None
                or _observation_quality(result)
                < _observation_quality(best_observation_result)
            ):
                best_observation_result = result
                best_observation_shot = shot
                best_observation_bgr = analysis_bgr

            # 容量守恒无效或棋盘预算没有落实完整：先重新截图，不立即降级/退出。
            if (
                result.board_update_status in ("incomplete", "causal_invalid")
                and observation_attempt < max_observation_attempts
            ):
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=(
                        "RETRY_OBSERVATION "
                        f"attempt={observation_attempt}/{max_observation_attempts}; "
                        f"board_status={result.board_update_status}; "
                        f"remaining={result.board_update_remaining_by_color}; "
                        f"excess={result.board_update_excess_by_color}; "
                        f"causal_invalid={result.causal_input_invalid or 'none'}"
                    ),
                )
                time.sleep(retry_delay)
                continue

            break

        # 如果至少有一次成功分析，使用本轮质量最好的 observation。
        # 这同时防止“第一次 39/55，后续 retry 反而 0/55”之类的退化覆盖。
        if best_observation_result is not None:
            result = best_observation_result
            shot = best_observation_shot
            analysis_bgr = best_observation_bgr

        reset_current = False

        if result is None:
            print(
                "进入 PAUSED_SAFE：观测连续失败。程序保持运行，"
                "稍后重新截图尝试恢复，不执行点击。"
            )
            if shot is not None:
                _append_execution_update(
                    log_path,
                    screenshot=shot,
                    step_label=step_label,
                    execution=(
                        "PAUSED_SAFE observation failure; "
                        f"reason={last_retry_error or safety.pause_reason}"
                    ),
                )
            time.sleep(
                max(
                    0.2,
                    float(getattr(args, "safe_pause_retry_delay", 1.0)),
                )
            )
            continue

        assert shot is not None

        # v5.9.2 experiment mode: commit the best conservative
        # observation and continue; incomplete sync is telemetry.
        _commit_analysis_result_state(args.state, result)
        if analysis_bgr is not None:
            committed_analysis_rgb = cv2.cvtColor(
                analysis_bgr,
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)

        front_number_cache = result.front_number_cache

        # ----- v5.8 实验模式：异常只记录，不阻止游戏继续 -----
        sync_pending = result.board_update_status in (
            "incomplete",
            "causal_invalid",
        )

        if sync_pending:
            result.report += (
                "\n\n[EXPERIMENT_WARNING]\n"
                "本轮棋盘因果同步不完整，但实验模式不会暂停。"
                "已提交当前保守 grid；旧 checkpoint 在这里结束，"
                "未落实数量不跨动作 carry。"
            )

        # 无论本轮是否完整，都结束上一动作 checkpoint，
        # 避免历史 unresolved consumption 污染新动作。
        pending_flow_capacity_before_by_color = None
        pending_prediction = None

        # SafetyRuntime 现在只保留 telemetry/单步降级信息，不再拥有 veto 权。
        if result.guarantee_broken:
            safety.note_guarantee_broken(round_no)
        if result.model_conflict_colors:
            safety.note_model_conflict(result.model_conflict_colors)
        if not result.guarantee_broken and not result.model_conflict_colors:
            safety.note_clean()

        result.report += (
            "\n\n[EXPERIMENT_STATE]\n"
            + safety.status_line()
            + "\n上述状态仅影响诊断和必要时的单步/多步选择，不阻止点击。"
        )

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

        if result.board_update_status in ("incomplete", "causal_invalid"):
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution=(
                    "EXPERIMENT_WARNING board-sync-incomplete; "
                    "continue-with-conservative-grid; "
                    + safety.status_line()
                ),
            )
            if analysis_bgr is not None:
                display.show(
                    analysis_bgr,
                    stage="实验模式 · 棋盘同步警告",
                    hint=(
                        "上一动作棋盘变化未完全解释；当前保守状态已写日志，"
                        "继续执行评分最高的单步候选。"
                    ),
                )
            print(
                "EXPERIMENT_WARNING：棋盘同步未完全解释，"
                "但不中断运行；继续使用当前保守 grid 做下一步实验。"
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
            # 只有“当前第一排根本没有形成任何可评分候选”才会到这里，
            # 通常意味着 OCR/车辆识别本身没有足够信息，而不是策略 veto。
            _append_execution_update(
                log_path,
                screenshot=shot,
                step_label=step_label,
                execution="NO_CANDIDATE perception-only retry; no strategy veto",
            )
            print(
                "当前没有形成可评分候选（通常是车辆/OCR识别不足）；"
                "不退出程序，立即重新截图。"
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
            executed_actions.append(
                (int(result.best.color), int(result.best.capacity))
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

        # v5.10 confirmed rule: last queued car clicked == game complete.
        # Parking does not need to become empty.  Check after the click batch's
        # flow has stabilized and require two consecutive empty queue reads.
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