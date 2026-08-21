from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .board import ctag, reachable_components, reachable_summary
from .config import EMPTY, UNKNOWN
from .models import Car, Candidate, TwoStepPlan


_QUEUE_LOOKAHEAD_WEIGHT = 0.70
_QUEUE_LOOKAHEAD_BONUS_CAP = 120000.0
_TWO_STEP_MIN_GAIN = 4000.0


@dataclass
class _FlowSimulation:
    grid: np.ndarray
    consumed_by_color: Dict[int, int]
    exposed_by_color: Dict[int, int]
    cleared_by_color: Dict[int, int]
    guaranteed_completions: int
    guaranteed_completions_by_color: Dict[int, int]
    guaranteed_parked_completions: int
    guaranteed_parked_completions_by_color: Dict[int, int]
    final_occupied_upper: int
    stable_safe: bool
    exact_grid: bool
    rounds: int


def parked_remainders_by_color(parked: Sequence[Car]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for car in parked:
        if car.color is None:
            continue
        out[car.color].append(car.remain if car.remain is not None else 99)
    return out


def guaranteed_completions_from_supply(
    remainders: Sequence[int],
    supply: int,
) -> int:
    """按最坏同色分配计算至少能保证完成多少辆车。"""
    rems = [max(1, int(x)) for x in remainders]
    if not rems or supply <= 0:
        return 0

    total_capacity = sum(rems)
    guaranteed_moved = min(int(supply), total_capacity)
    no_completion_max = sum(max(0, remain - 1) for remain in rems)
    return max(0, guaranteed_moved - no_completion_max)


def _guaranteed_parked_completions_from_supply(
    parked_remainders: Sequence[int],
    new_remainders: Sequence[int],
    supply: int,
) -> int:
    """
    同色块可在“已有停车车 + 本次点击车”之间任意分配时，
    计算至少能保证完成多少辆已有停车车。

    为尽量不让停车车完成，先允许：
      1) 所有新点击车吸满；
      2) 每辆停车车只吸到 remain-1。
    超过这部分的每一个块都会强迫至少一辆停车车完成。
    """
    parked_rems = [max(1, int(x)) for x in parked_remainders]
    new_rems = [max(1, int(x)) for x in new_remainders]
    if not parked_rems or supply <= 0:
        return 0

    total_capacity = sum(parked_rems) + sum(new_rems)
    moved = min(int(supply), total_capacity)
    no_parked_completion_max = (
        sum(new_rems)
        + sum(max(0, remain - 1) for remain in parked_rems)
    )
    return max(0, moved - no_parked_completion_max)


def _clear_reachable_color(grid: np.ndarray, color: int) -> Tuple[np.ndarray, int]:
    comps, _ = reachable_components(grid)
    groups = comps.get(color, [])
    sim = grid.copy()
    removed = 0
    for group in groups:
        for r, c in group:
            if int(sim[r, c]) == color:
                sim[r, c] = EMPTY
                removed += 1
    return sim, removed


def simulate_clear_current_reachable_color(
    grid: np.ndarray,
    color: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    兼容旧调用：确定性移除当前全部可达该色并返回新暴露颜色。
    v5.3 的正式策略使用 simulate_flow_closure()。
    """
    before, _ = reachable_summary(grid)
    sim, _removed = _clear_reachable_color(grid, color)
    after, _ = reachable_summary(sim)

    unlocked: Dict[int, int] = {}
    for other in set(before) | set(after):
        if other == color:
            continue
        delta = int(after.get(other, 0)) - int(before.get(other, 0))
        if delta > 0:
            unlocked[other] = delta
    return sim, unlocked


def simulate_flow_closure(
    grid: np.ndarray,
    parked: Sequence[Car],
    action_cars: Sequence[Car],
    slots: int,
) -> _FlowSimulation:
    """
    对真实游戏规则做保守的“自动分流闭包”：

    - 停车车与本次点击车辆同时参与吸收；
    - 只要还有同色车辆有容量，可达同色块就持续被吃；
    - 同色分配顺序未知，因此车辆完成数按最坏分配保证；
    - 若同色总剩余容量 >= 当前全部可达块，则该颜色可确定全部清空，
      从棋盘删除并继续计算下一层暴露；
    - 若容量不足以清空，则能确定这些容量一定被吃满，但具体删除位置未知，
      因此不从棋盘删除这部分，后续连锁按保守下界继续；
    - 直到没有新的确定性清层，得到本轮稳定状态上界。

    final_occupied_upper 是“分流结束后最多还会留下多少辆车”。
    真实规则允许过程中短暂 6/6，但稳定后若仍为 6/6 则失败，
    因此 stable_safe 要求 final_occupied_upper < slots。
    """
    sim = grid.copy()

    parked_by_color = parked_remainders_by_color(parked)
    action_by_color: Dict[int, List[int]] = defaultdict(list)
    for car in action_cars:
        if car.color is None or car.remain is None:
            continue
        action_by_color[int(car.color)].append(max(1, int(car.remain)))

    all_colors = set(parked_by_color) | set(action_by_color)
    total_capacity: Dict[int, int] = {
        color: sum(parked_by_color.get(color, []))
        + sum(action_by_color.get(color, []))
        for color in all_colors
    }

    consumed: Dict[int, int] = defaultdict(int)
    exposed: Counter = Counter()
    cleared: Counter = Counter()
    uncertain_colors: set[int] = set()
    rounds = 0

    # 每轮先让“容量不足以清空当前可达区”的颜色确定性耗尽容量；
    # 再挑一个可以完全清空的颜色移除，重新计算 reachability。
    max_rounds = int(grid.size) + 32
    while rounds < max_rounds:
        rounds += 1
        reachable, _ = reachable_summary(sim)

        # 容量不足时：所有剩余容量一定被吃光，但删除位置未知。
        for color in sorted(all_colors):
            remaining_capacity = max(
                0, int(total_capacity.get(color, 0)) - int(consumed.get(color, 0))
            )
            r = int(reachable.get(color, 0))
            if remaining_capacity <= 0 or r <= 0:
                continue
            if remaining_capacity < r:
                consumed[color] += remaining_capacity
                uncertain_colors.add(color)

        # 找一个仍有容量且能把当前全部可达同色块吃完的颜色。
        reachable, _ = reachable_summary(sim)
        clear_color: Optional[int] = None
        for color in sorted(all_colors):
            remaining_capacity = max(
                0, int(total_capacity.get(color, 0)) - int(consumed.get(color, 0))
            )
            r = int(reachable.get(color, 0))
            if r > 0 and remaining_capacity >= r:
                clear_color = color
                break

        if clear_color is None:
            break

        before = dict(reachable)
        r = int(before.get(clear_color, 0))
        new_grid, removed = _clear_reachable_color(sim, clear_color)
        if removed <= 0:
            break

        sim = new_grid
        consumed[clear_color] += r
        cleared[clear_color] += removed

        after, _ = reachable_summary(sim)
        for other in set(before) | set(after):
            if other == clear_color:
                continue
            delta = int(after.get(other, 0)) - int(before.get(other, 0))
            if delta > 0:
                exposed[other] += delta

    guaranteed_total = 0
    guaranteed_by_color: Dict[int, int] = {}
    parked_total = 0
    parked_completed_by_color: Dict[int, int] = {}

    for color in sorted(all_colors):
        supply = int(consumed.get(color, 0))
        parked_rems = parked_by_color.get(color, [])
        new_rems = action_by_color.get(color, [])
        all_rems = list(parked_rems) + list(new_rems)

        completed = guaranteed_completions_from_supply(all_rems, supply)
        if completed > 0:
            guaranteed_by_color[color] = completed
            guaranteed_total += completed

        parked_completed = _guaranteed_parked_completions_from_supply(
            parked_rems,
            new_rems,
            supply,
        )
        if parked_completed > 0:
            parked_completed_by_color[color] = parked_completed
            parked_total += parked_completed

    total_cars_after_clicks = len(parked) + len(action_cars)
    final_occupied_upper = max(0, total_cars_after_clicks - guaranteed_total)
    stable_safe = final_occupied_upper < slots

    return _FlowSimulation(
        grid=sim,
        consumed_by_color=dict(sorted(consumed.items())),
        exposed_by_color=dict(sorted(exposed.items())),
        cleared_by_color=dict(sorted(cleared.items())),
        guaranteed_completions=guaranteed_total,
        guaranteed_completions_by_color=guaranteed_by_color,
        guaranteed_parked_completions=parked_total,
        guaranteed_parked_completions_by_color=parked_completed_by_color,
        final_occupied_upper=final_occupied_upper,
        stable_safe=stable_safe,
        exact_grid=(len(uncertain_colors) == 0),
        rounds=rounds,
    )


def _parking_release_weight(occupied_slots: int, slots: int) -> float:
    if slots <= 0:
        return 35000.0
    pressure = min(1.0, max(0.0, occupied_slots / slots))
    return 35000.0 + 50000.0 * pressure * pressure


def _score_flow(
    sim: _FlowSimulation,
    *,
    occupied_slots: int,
    slots: int,
    action_count: int,
    useful_colors: Sequence[int],
    next_color: Optional[int] = None,
) -> float:
    """
    只基于“联合动作完成后的稳定状态变化”评分。
    不再把 A 的奖励和 B 的奖励机械相加，从根源避免同一批新暴露块重复计分。
    """
    useful = set(useful_colors)
    cleared_cells = sum(sim.cleared_by_color.values())
    useful_exposed = sum(
        n for color, n in sim.exposed_by_color.items() if color in useful
    )
    next_exposed = (
        int(sim.exposed_by_color.get(next_color, 0))
        if next_color is not None
        else 0
    )

    score = 0.0

    # 队列推进本身有价值，但远低于确定完成/释放停车位。
    score += 3500.0 * action_count

    # 所有车辆的确定完成都会减少最终占位；已有停车车完成额外高权重。
    score += 22000.0 * sim.guaranteed_completions
    score += (
        _parking_release_weight(occupied_slots, slots)
        * sim.guaranteed_parked_completions
    )

    # 确定性清盘比单纯 reachable 更重要。
    score += min(cleared_cells, 160) * 650.0
    score += min(useful_exposed, 80) * 450.0
    score += min(next_exposed, 50) * 1100.0

    # 稳定后停车越拥堵，惩罚越强；5/6 时尤其不鼓励留下新车。
    score -= 550.0 * (sim.final_occupied_upper ** 2)
    if sim.final_occupied_upper == slots - 1:
        score -= 4000.0

    # 精确闭包可稍微优先；存在局部删除位置未知时仍然安全，但价值按保守下界。
    if sim.exact_grid:
        score += 1500.0
    else:
        score -= 1200.0

    return score


def _specific_action_car_completion_guaranteed(
    parked_same_color: Sequence[int],
    action_capacity: int,
    total_supply: int,
) -> bool:
    """
    分配顺序完全未知时，要保证“这辆新点击车自己”完成，
    最坏情况可以先把所有同色停车车喂满。
    因此只有总供给足以填满所有同色车辆时，才能保证指定新车完成。
    """
    total = sum(max(1, int(x)) for x in parked_same_color) + max(
        1, int(action_capacity)
    )
    return int(total_supply) >= total


def _build_candidate(
    grid: np.ndarray,
    car: Car,
    next_car: Optional[Car],
    parked: Sequence[Car],
    slots: int,
    occupied_slots: int,
    useful_colors: Sequence[int],
    neighbor_contacts: Dict[int, Counter],
) -> Candidate:
    assert car.color is not None
    assert car.remain is not None
    assert car.column is not None

    color = int(car.color)
    cap = int(car.remain)
    reachable, _ = reachable_summary(grid)
    r = int(reachable.get(color, 0))
    contacts = neighbor_contacts.get(color, Counter())

    sim = simulate_flow_closure(grid, parked, [car], slots)
    parked_same = parked_remainders_by_color(parked).get(color, [])

    total_supply_for_color = int(sim.consumed_by_color.get(color, 0))
    self_clear = _specific_action_car_completion_guaranteed(
        parked_same,
        cap,
        total_supply_for_color,
    )
    some_completion = sim.guaranteed_completions > 0

    deterministic_clear = (
        r > 0
        and int(sim.cleared_by_color.get(color, 0)) >= r
    )

    next_color = next_car.color if next_car is not None else None
    next_capacity = next_car.remain if next_car is not None else None
    next_new = (
        int(sim.exposed_by_color.get(next_color, 0))
        if next_color is not None
        else 0
    )
    useful_new = sum(
        n
        for c, n in sim.exposed_by_color.items()
        if c in set(useful_colors)
    )
    next_match_contacts = (
        int(contacts.get(next_color, 0))
        if next_color is not None
        else 0
    )

    score = _score_flow(
        sim,
        occupied_slots=occupied_slots,
        slots=slots,
        action_count=1,
        useful_colors=useful_colors,
        next_color=next_color,
    )

    # 对无法产生确定性清层的动作，保留少量即时填充启发，作为安全候选间的次级排序。
    fill_ratio = min(1.0, r / max(1, cap))
    score += 2200.0 * fill_ratio
    score += min(r, 100) * 6.0
    score += next_match_contacts * 80.0

    if r == 0:
        score -= 1800.0

    rejected = not sim.stable_safe
    reason = ""
    if rejected:
        reason = (
            f"按自动分流闭包的最坏分配，分流稳定后停车占用上界仍为 "
            f"{sim.final_occupied_upper}/{slots}；本游戏稳定后 6/6 会立即失败"
        )

    return Candidate(
        column=int(car.column),
        color=color,
        capacity=cap,
        reachable=r,
        self_clear_guaranteed=self_clear,
        some_completion_guaranteed=some_completion,
        guaranteed_completions=sim.guaranteed_completions,
        chain_parked_completions=sim.guaranteed_parked_completions,
        chain_parked_completion_by_color=dict(
            sim.guaranteed_parked_completions_by_color
        ),
        deterministic_clear_reachable=deterministic_clear,
        next_color_newly_reachable=next_new,
        useful_newly_reachable=useful_new,
        unlocked_by_color=dict(sim.exposed_by_color),
        rejected=rejected,
        reject_reason=reason,
        score=float(score),
        next_color=next_color,
        next_capacity=next_capacity,
        queue_unlock_bonus=0.0,
        next_vehicle_score=0.0,
        next_vehicle_chain_parked_completions=0,
        next_vehicle_exact=False,
        next_match_contacts=next_match_contacts,
        neighbor_contacts=dict(contacts),
        flow_cleared_cells=sum(sim.cleared_by_color.values()),
        flow_final_occupied_upper=sim.final_occupied_upper,
        flow_exact=sim.exact_grid,
        flow_rounds=sim.rounds,
    )


def evaluate_candidates(
    grid: np.ndarray,
    front: Sequence[Car],
    nxt: Sequence[Car],
    parked: Sequence[Car],
    slots: int,
    occupied_slots: int,
    *,
    include_queue_lookahead: bool = True,
    allow_deterministic_unlock: bool = True,
) -> List[Candidate]:
    # allow_deterministic_unlock 保留在接口中兼容旧调用；v5.3 的闭包自身决定
    # 哪些颜色能被数学证明清空，不再由外部强制禁用。
    del allow_deterministic_unlock

    _reachable, neighbor_contacts = reachable_summary(grid)
    next_by_col = {c.column: c for c in nxt if c.column is not None}
    front_by_col = {c.column: c for c in front if c.column is not None}

    active_front_colors = {c.color for c in front if c.color is not None}
    parked_colors = {c.color for c in parked if c.color is not None}
    useful_colors = {
        int(c) for c in (active_front_colors | parked_colors) if c is not None
    }

    candidates: List[Candidate] = []
    for car in front:
        if car.color is None or car.remain is None or car.column is None:
            continue
        candidate = _build_candidate(
            grid,
            car,
            next_by_col.get(car.column),
            parked,
            slots,
            occupied_slots,
            useful_colors,
            neighbor_contacts,
        )
        candidates.append(candidate)

    # ---------- 第二排价值传播 ----------
    # B 顶上是立即发生的，而 A/B 的吸收过程可以并发。
    # 因此直接模拟联合动作 [A, B]，不再先生成 A 的“串行 post-state”。
    if include_queue_lookahead:
        for candidate in candidates:
            if candidate.rejected:
                continue

            first_car = front_by_col.get(candidate.column)
            next_car = next_by_col.get(candidate.column)
            if (
                first_car is None
                or next_car is None
                or next_car.color is None
                or next_car.remain is None
            ):
                continue

            promoted = Car(
                source="front",
                column=next_car.column,
                color=next_car.color,
                remain=next_car.remain,
                x=next_car.x,
                y=first_car.y,
            )
            pair_sim = simulate_flow_closure(
                grid,
                parked,
                [first_car, promoted],
                slots,
            )
            if not pair_sim.stable_safe:
                continue

            pair_score = _score_flow(
                pair_sim,
                occupied_slots=occupied_slots,
                slots=slots,
                action_count=2,
                useful_colors=useful_colors,
            )

            candidate.next_vehicle_score = float(pair_score)
            candidate.next_vehicle_chain_parked_completions = (
                pair_sim.guaranteed_parked_completions
            )
            candidate.next_vehicle_exact = pair_sim.exact_grid

            # 只传播“联合动作比 A 单独更好”的增量，避免 A 的已有收益被重复计算。
            incremental = max(0.0, pair_score - candidate.score)
            bonus = incremental * _QUEUE_LOOKAHEAD_WEIGHT

            extra_parked_release = max(
                0,
                pair_sim.guaranteed_parked_completions
                - candidate.chain_parked_completions,
            )
            if extra_parked_release > 0:
                bonus = max(
                    bonus,
                    35000.0 * extra_parked_release
                    + 0.25 * incremental,
                )

            bonus = min(_QUEUE_LOOKAHEAD_BONUS_CAP, bonus)
            candidate.queue_unlock_bonus = float(bonus)
            candidate.score += bonus

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _candidate_for_promoted_car(
    grid: np.ndarray,
    first_car: Car,
    next_car: Car,
    parked: Sequence[Car],
    slots: int,
    occupied_slots: int,
    useful_colors: Sequence[int],
    neighbor_contacts: Dict[int, Counter],
) -> Candidate:
    promoted = Car(
        source="front",
        column=next_car.column,
        color=next_car.color,
        remain=next_car.remain,
        x=next_car.x,
        y=first_car.y,
    )
    return _build_candidate(
        grid,
        promoted,
        None,
        parked,
        slots,
        occupied_slots,
        useful_colors,
        neighbor_contacts,
    )


def choose_two_step_plan(
    grid: np.ndarray,
    front: Sequence[Car],
    nxt: Sequence[Car],
    parked: Sequence[Car],
    slots: int,
    occupied_slots: int,
    candidates: Sequence[Candidate],
) -> Optional[TwoStepPlan]:
    """
    v5.3 连续两步：

    - 游戏允许分流过程中短暂 6/6，因此最坏两辆新车都留下时，
      只要点击前至少有 2 个空位，就不会出现 >6 的瞬时占用。
    - 第一动作仍要求“单独执行也稳定安全”。这是执行层的兜底：
      同列第二排补位确认若失败，只执行第一步也不会输。
    - A+B 不是串行 score(A)+score(B)，而是作为并发联合动作一次性跑
      simulate_flow_closure()，按最终稳定状态评分。
    - 只有联合动作评分明确高于当前最佳单步，才批量执行。
    """
    free_slots = slots - occupied_slots
    if free_slots < 2:
        return None

    valid_first = [c for c in candidates if not c.rejected]
    if not valid_first:
        return None

    best_single_score = max(c.score for c in valid_first)
    front_by_col = {c.column: c for c in front if c.column is not None}
    next_by_col = {c.column: c for c in nxt if c.column is not None}

    _reachable, neighbor_contacts = reachable_summary(grid)
    active_front_colors = {c.color for c in front if c.color is not None}
    parked_colors = {c.color for c in parked if c.color is not None}
    useful_colors = {
        int(c) for c in (active_front_colors | parked_colors) if c is not None
    }

    best_plan: Optional[TwoStepPlan] = None

    def consider(
        first: Candidate,
        second: Candidate,
        second_source: str,
        action_cars: Sequence[Car],
        reason_prefix: str,
    ) -> None:
        nonlocal best_plan

        pair_sim = simulate_flow_closure(
            grid,
            parked,
            action_cars,
            slots,
        )
        if not pair_sim.stable_safe:
            return

        pair_score = _score_flow(
            pair_sim,
            occupied_slots=occupied_slots,
            slots=slots,
            action_count=2,
            useful_colors=useful_colors,
        )

        # 双步必须比当前最优单步有明确增益，否则宁可点击一次后重新观察。
        if pair_score < best_single_score + _TWO_STEP_MIN_GAIN:
            return

        reason = (
            f"{reason_prefix}；A+B 按并发联合动作执行自动分流闭包；"
            f"稳定后停车占用最坏上界 {pair_sim.final_occupied_upper}/{slots}；"
            f"保证完成 {pair_sim.guaranteed_completions} 辆，"
            f"其中已有停车车至少 {pair_sim.guaranteed_parked_completions} 辆；"
            f"确定性清除色块 {sum(pair_sim.cleared_by_color.values())} 个"
        )
        if not pair_sim.exact_grid:
            reason += "；存在容量不足颜色，其删除位置未知，后续开路按保守下界"

        plan = TwoStepPlan(
            first=first,
            second=second,
            second_source=second_source,
            score=float(pair_score),
            free_slots_before=free_slots,
            first_simulated_exactly=first.flow_exact,
            reason=reason,
            guaranteed_completions=pair_sim.guaranteed_completions,
            guaranteed_parked_completions=pair_sim.guaranteed_parked_completions,
            cleared_cells=sum(pair_sim.cleared_by_color.values()),
            final_occupied_upper=pair_sim.final_occupied_upper,
            flow_exact=pair_sim.exact_grid,
        )
        if best_plan is None or plan.score > best_plan.score:
            best_plan = plan

    for first in valid_first:
        first_car = front_by_col.get(first.column)
        if first_car is None:
            continue

        # 类型1：当前另一列第一排。两辆都已经确定存在，可直接作为联合动作。
        for second in valid_first:
            if second.column == first.column:
                continue
            second_car = front_by_col.get(second.column)
            if second_car is None:
                continue
            consider(
                first,
                second,
                "front",
                [first_car, second_car],
                "第二步为当前另一列第一排车辆",
            )

        # 类型2：同列第二排 B。B 会在 A 离开第一排后立即顶上，
        # 执行层仍用颜色+数字快速确认后才点击。
        next_car = next_by_col.get(first.column)
        if (
            next_car is None
            or next_car.color is None
            or next_car.remain is None
            or next_car.column is None
        ):
            continue

        promoted_candidate = _candidate_for_promoted_car(
            grid,
            first_car,
            next_car,
            parked,
            slots,
            occupied_slots,
            useful_colors,
            neighbor_contacts,
        )
        promoted_car = Car(
            source="front",
            column=next_car.column,
            color=next_car.color,
            remain=next_car.remain,
            x=next_car.x,
            y=first_car.y,
        )
        consider(
            first,
            promoted_candidate,
            "next",
            [first_car, promoted_car],
            "第一步离开队列后，同列第二排立即补位",
        )

    return best_plan


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
        f"连续两步计划: 当前空位 {plan.free_slots_before} >= 2，"
        f"先点击第{plan.first.column}列 {ctag(plan.first.color)}×{plan.first.capacity}；"
        f"{second_text}。\n"
        f"联合动作预测: {exact_text}；稳定后停车占用上界 "
        f"{plan.final_occupied_upper}；保证完成 {plan.guaranteed_completions} 辆，"
        f"已有停车车至少完成 {plan.guaranteed_parked_completions} 辆；"
        f"确定性清除 {plan.cleared_cells} 个色块；pair_score={plan.score:.1f}\n"
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
    lines.append("候选动作（按并发自动分流闭包评分）:")
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
            flags.append("同色总容量可确定清空当前可达该色")
        if c.queue_unlock_bonus > 0 and c.next_capacity is not None:
            flags.append(
                f"同列第二排联合前瞻 +{c.queue_unlock_bonus:.0f}"
                f"（{'确定' if c.next_vehicle_exact else '保守'}闭包）"
            )
        if c.rejected:
            flags.append("稳定状态硬禁止")
        if c.next_color is not None:
            if c.next_capacity is not None:
                flags.append(f"第二排 {ctag(c.next_color)}×{c.next_capacity}")
            else:
                flags.append(f"第二排 {ctag(c.next_color)}×?（浅色OCR未确认）")

        lines.append(
            f"  第{c.column}列 {ctag(c.color)}×{c.capacity}: "
            f"可触达={c.reachable}, score={c.score:.1f}, "
            f"闭包清除={c.flow_cleared_cells}, "
            f"稳定占用上界={c.flow_final_occupied_upper}/{slots}"
            + (" | " + "；".join(flags) if flags else "")
        )

        if c.unlocked_by_color:
            exposure_text = ", ".join(
                f"{ctag(k)}+{v}" for k, v in c.unlocked_by_color.items()
            )
            lines.append(f"      闭包过程中确定新暴露: {exposure_text}")

        if c.queue_unlock_bonus > 0:
            lines.append(
                f"      第二排联合动作评分 {c.next_vehicle_score:.1f}；"
                f"只把相对当前动作的增量折算 +{c.queue_unlock_bonus:.1f}，"
                "避免重复使用同一批新暴露色块。"
            )

        if not c.flow_exact:
            lines.append(
                "      闭包说明: 至少一种颜色容量不足以清空当前可达区，"
                "具体删除位置未知；后续开路只按保守下界计算。"
            )

        if c.rejected:
            lines.append(f"      原因: {c.reject_reason}")

    valid = [c for c in candidates if not c.rejected]
    lines.append("")
    if not valid:
        lines.append("建议: 不点击。当前没有满足稳定状态硬安全约束的单步候选。")
    else:
        best = max(valid, key=lambda c: c.score)
        lines.append(
            f"下一步建议: 点击【第一排第 {best.column} 列】 "
            f"{ctag(best.color)} / 数字 {best.capacity}"
        )
        lines.append(
            "程序依据: 以本次点击车与已有停车车作为并发活跃车辆，"
            "持续执行“可达同色自动吸收 → 可证明清层 → 新颜色暴露 → "
            "停车车继续自动吸收”的闭包直到稳定；"
            f"稳定后停车占用最坏上界 {best.flow_final_occupied_upper}/{slots}，"
            f"确定性清除 {best.flow_cleared_cells} 个色块。"
        )
        if best.chain_parked_completions > 0:
            lines.append(
                f"停车释放依据: 不假设同色分配顺序，按最坏分配仍保证已有停车车 "
                f"至少完成 {best.chain_parked_completions} 辆。"
            )
        if best.queue_unlock_bonus > 0 and best.next_capacity is not None:
            lines.append(
                f"队列依据: 点击后同列第二排 {ctag(best.next_color)}×"
                f"{best.next_capacity} 会立即补位；A+B 已按并发联合动作重新模拟，"
                f"只传播联合动作相对 A 的增量价值 +{best.queue_unlock_bonus:.0f}。"
            )

    return "\n".join(lines)


def best_valid_candidate(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    valid = [candidate for candidate in candidates if not candidate.rejected]
    if not valid:
        return None
    return max(valid, key=lambda c: c.score)