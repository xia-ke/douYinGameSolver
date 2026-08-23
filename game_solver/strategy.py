from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .board import ctag, reachable_components, reachable_summary
from .config import (
    EMPTY, UNKNOWN,
    X_EDGE0_REF, X_STEP_REF, Y_EDGE0_REF, Y_STEP_REF,
)
from .models import Car, Candidate, TwoStepPlan


_TWO_STEP_MIN_FREE_SLOTS = 3

# Nearest-cell routing is only a weak spatial lookahead.  Euclidean distance is
# a working geometry heuristic, not confirmed game truth.  If the closest two
# movable frontier cells are separated by less than roughly one third of a
# logical cell step, the prediction abstains instead of inventing a tie-break.
_NEAREST_MIN_DISTANCE_MARGIN_REF = 0.35 * min(X_STEP_REF, Y_STEP_REF)


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
    nearest_predicted_exposed_by_color: Dict[int, int] = field(default_factory=dict)
    nearest_predicted_removed_by_color: Dict[int, int] = field(default_factory=dict)


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
    """
    已确认游戏规则：同色停车车优先给“剩余数字更小”的车吸收。

    一辆车一旦先拿到色块，remain 会继续变小，因此它会持续保持优先级
    直到归零。于是总完成数不再需要按任意分配做最坏下界，而是把 remain
    从小到大排序，按供给依次填满即可。
    """
    left = max(0, int(supply))
    completed = 0
    for remain in sorted(max(1, int(x)) for x in remainders):
        if left < remain:
            break
        left -= remain
        completed += 1
    return completed


def _guaranteed_parked_completions_from_supply(
    parked_remainders: Sequence[int],
    new_remainders: Sequence[int],
    supply: int,
) -> int:
    """
    同色车统一按剩余数字从小到大吸收。

    本轮新点击车进入停车区后也参与该优先级。唯一仍未确认的是“相同
    remain 的平局顺序”，因此为了证明旧停车位一定释放，平局时故意把
    新车排在旧停车车之前，保留硬安全下界。
    """
    # kind: 0 = 本轮新车；1 = 已停车车。
    # sort 后相同 remain 的新车先走，是“旧停车位释放”的保守 tie-break。
    cars: List[Tuple[int, int]] = []
    cars.extend((max(1, int(x)), 1) for x in parked_remainders)
    cars.extend((max(1, int(x)), 0) for x in new_remainders)
    cars.sort(key=lambda item: (item[0], item[1]))

    left = max(0, int(supply))
    completed_parked = 0
    for remain, kind in cars:
        if left < remain:
            break
        left -= remain
        if kind == 1:
            completed_parked += 1
    return completed_parked


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


def _grid_cell_center_ref(r: int, c: int) -> Tuple[float, float]:
    """52x38 逻辑格在 940x2048 reference 坐标系中的中心。"""
    return (
        X_EDGE0_REF + (c + 0.5) * X_STEP_REF,
        Y_EDGE0_REF + (r + 0.5) * Y_STEP_REF,
    )


def _parked_states_after_known_consumption(
    cars: Sequence[Car],
    consumed: int,
) -> Optional[List[Tuple[Car, int]]]:
    """
    把已经由确定性清层消耗掉的供给落实到具体停车车 remain。

    若两辆物理位置不同的同色车 remain 完全相同，则平局规则仍未知；
    空间预测直接放弃，不自行用 x/slot 顺序猜赢家。
    """
    states: List[Tuple[Car, int]] = [
        (car, max(1, int(car.remain)))
        for car in cars
        if car.remain is not None
    ]
    if not states:
        return []

    initial_rems = [remain for _car, remain in states]
    if len(initial_rems) != len(set(initial_rems)):
        return None

    left = max(0, int(consumed))
    while left > 0 and states:
        states.sort(key=lambda item: item[1])
        car, remain = states[0]
        take = min(left, remain)
        left -= take
        remain -= take
        if remain <= 0:
            states.pop(0)
        else:
            states[0] = (car, remain)

    if left > 0:
        return []
    return states


def _confident_nearest_frontier_cell(
    cells: Sequence[Tuple[int, int]],
    car: Car,
) -> Optional[Tuple[int, int]]:
    """Return one clearly closest frontier cell, otherwise abstain.

    The distance model is deliberately not promoted to a game rule.  It is used
    only when the parked car position is known and the geometric winner is
    separated from the runner-up by a visible margin.
    """
    if not cells:
        return None
    if not np.isfinite(float(car.x)) or not np.isfinite(float(car.y)):
        return None

    ranked: List[Tuple[float, Tuple[int, int]]] = []
    for r, c in cells:
        x, y = _grid_cell_center_ref(r, c)
        distance = float(np.hypot(x - float(car.x), y - float(car.y)))
        ranked.append((distance, (r, c)))
    ranked.sort(key=lambda item: item[0])

    if len(ranked) >= 2:
        best_distance = ranked[0][0]
        second_distance = ranked[1][0]
        if second_distance - best_distance < _NEAREST_MIN_DISTANCE_MARGIN_REF:
            return None
    return ranked[0][1]


def _predict_nearest_partial_consumption(
    grid: np.ndarray,
    color: int,
    budget: int,
    parked_states: List[Tuple[Car, int]],
) -> Tuple[np.ndarray, int]:
    """Weak lookahead for partial consumption by already parked known-position cars.

    Same-remain assignment ties and spatially ambiguous nearest cells both stop
    prediction.  The returned grid is never used for stable_safe or completion
    proof; it feeds only the final bounded strategy tie-break.
    """
    sim = grid.copy()
    left = max(0, int(budget))
    removed = 0

    while left > 0 and parked_states:
        parked_states.sort(key=lambda item: item[1])
        if (
            len(parked_states) >= 2
            and parked_states[0][1] == parked_states[1][1]
        ):
            break

        car, remain = parked_states[0]
        comps, opened = reachable_components(sim)
        rows, cols = sim.shape
        cells: List[Tuple[int, int]] = []
        for group in comps.get(color, []):
            for r, c in group:
                touches_open = r == rows - 1
                if not touches_open:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols and opened[rr, cc]:
                            touches_open = True
                            break
                if touches_open:
                    cells.append((r, c))

        target = _confident_nearest_frontier_cell(cells, car)
        if target is None:
            break
        target_r, target_c = target
        if int(sim[target_r, target_c]) != color:
            break

        sim[target_r, target_c] = EMPTY
        removed += 1
        left -= 1
        remain -= 1
        if remain <= 0:
            parked_states.pop(0)
        else:
            parked_states[0] = (car, remain)

    return sim, removed


def _nearest_partial_exposure_prediction(
    base_grid: np.ndarray,
    parked: Sequence[Car],
    action_cars: Sequence[Car],
    consumed_by_color: Dict[int, int],
    cleared_by_color: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Confident nearest-cell lookahead for unresolved partial consumption only.

    Requirements are intentionally strict: the absorbing car must already be
    parked with a known finite position; a newly clicked same-color car disables
    the spatial guess because its eventual parking position is not modeled;
    equal remain assignment ties and close nearest-cell geometry abstain.
    """
    action_colors = {
        int(car.color)
        for car in action_cars
        if car.color is not None and car.remain is not None
    }

    parked_by_color: Dict[int, List[Car]] = defaultdict(list)
    for car in parked:
        if car.color is None or car.remain is None:
            continue
        if car.source != "parked":
            continue
        if not np.isfinite(float(car.x)) or not np.isfinite(float(car.y)):
            continue
        parked_by_color[int(car.color)].append(car)

    predicted_exposed: Dict[int, int] = {}
    predicted_removed: Dict[int, int] = {}
    before, _ = reachable_summary(base_grid)

    for color, cars in parked_by_color.items():
        if color in action_colors:
            continue

        consumed = int(consumed_by_color.get(color, 0))
        already_cleared = int(cleared_by_color.get(color, 0))
        partial_budget = max(0, consumed - already_cleared)
        if partial_budget <= 0:
            continue

        states = _parked_states_after_known_consumption(cars, already_cleared)
        if states is None or not states:
            continue

        predicted_grid, removed = _predict_nearest_partial_consumption(
            base_grid,
            color,
            partial_budget,
            states,
        )
        if removed <= 0:
            continue

        predicted_removed[color] = removed
        after, _ = reachable_summary(predicted_grid)
        for other in set(before) | set(after):
            if other == color:
                continue
            delta = int(after.get(other, 0)) - int(before.get(other, 0))
            if delta > 0:
                predicted_exposed[other] = max(
                    int(predicted_exposed.get(other, 0)),
                    delta,
                )

    return dict(sorted(predicted_exposed.items())), dict(sorted(predicted_removed.items()))


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
    - 同色停车车按“剩余数字较小优先”持续吸收直到归零；
    - 相同剩余数字的平局顺序仍按保守规则处理；
    - 对位置已知的旧停车车，容量不足时额外按“最近可达同色格”做空间前瞻；
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

        parked_completed = _guaranteed_parked_completions_from_supply(
            parked_rems,
            new_rems,
            supply,
        )

        # 停车位 smallest-remain-first 可能证明更多停车车必然完成；
        # 这些完成当然也属于所有 active 车辆的总完成数。
        completed = max(
            guaranteed_completions_from_supply(all_rems, supply),
            parked_completed,
        )
        if completed > 0:
            guaranteed_by_color[color] = completed
            guaranteed_total += completed

        if parked_completed > 0:
            parked_completed_by_color[color] = parked_completed
            parked_total += parked_completed

    total_cars_after_clicks = len(parked) + len(action_cars)
    final_occupied_upper = max(0, total_cars_after_clicks - guaranteed_total)
    stable_safe = final_occupied_upper < slots

    nearest_exposed, nearest_removed = _nearest_partial_exposure_prediction(
        sim,
        parked,
        action_cars,
        dict(consumed),
        dict(cleared),
    )

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
        nearest_predicted_exposed_by_color=nearest_exposed,
        nearest_predicted_removed_by_color=nearest_removed,
    )


def _useful_exposed_cells(
    exposed_by_color: Dict[int, int],
    useful_colors: Sequence[int],
) -> int:
    useful = {int(c) for c in useful_colors}
    return int(sum(int(n) for c, n in exposed_by_color.items() if int(c) in useful))


def _utility_key(
    sim: _FlowSimulation,
    *,
    useful_colors: Sequence[int],
    queue_progress: int,
    reachable_hint: int = 0,
) -> Tuple[int, ...]:
    """Rule-first lexicographic utility; no large additive score exists here."""
    deterministic_cleared = int(sum(sim.cleared_by_color.values()))
    useful_exposed = _useful_exposed_cells(sim.exposed_by_color, useful_colors)

    # Nearest prediction is deliberately confined to the final bounded tie-break.
    nearest_useful = _useful_exposed_cells(
        sim.nearest_predicted_exposed_by_color,
        useful_colors,
    )
    heuristic_tiebreak = (
        1 if sim.exact_grid else 0,
        min(9, max(0, int(nearest_useful))),
        min(9, max(0, int(reachable_hint))),
    )

    return (
        int(sim.guaranteed_parked_completions),
        int(sim.guaranteed_completions),
        deterministic_cleared,
        useful_exposed,
        max(0, int(queue_progress)),
        *heuristic_tiebreak,
    )


def _utility_reason(utility: Sequence[int]) -> str:
    values = tuple(int(v) for v in utility)
    padded = values + (0,) * max(0, 8 - len(values))
    return (
        f"parked_release={padded[0]}, completions={padded[1]}, "
        f"cleared={padded[2]}, useful_exposure={padded[3]}, "
        f"queue_progress={padded[4]}, "
        f"tie(exact={padded[5]}, nearest_useful={padded[6]}, reachable={padded[7]})"
    )


def _two_step_primary_improves(
    pair_utility: Sequence[int],
    best_single_utility: Sequence[int],
) -> bool:
    """Two-step must improve a rule term; tie-break-only improvement is insufficient."""
    return tuple(pair_utility[:5]) > tuple(best_single_utility[:5])


def _specific_action_car_completion_guaranteed(
    parked_same_color: Sequence[int],
    action_capacity: int,
    total_supply: int,
) -> bool:
    """
    按“剩余数字小者优先”判断这辆新点击车自己是否必然完成。

    remain 比它小的停车车必然先完成；remain 比它大的车不会抢在它前面。
    相同 remain 的平局仍未知，因此为了保证该新车完成，保守地把所有相同
    remain 的旧停车车也视为可能排在它前面。
    """
    cap = max(1, int(action_capacity))
    supply_needed = cap + sum(
        max(1, int(x))
        for x in parked_same_color
        if max(1, int(x)) <= cap
    )
    return int(total_supply) >= supply_needed


def _build_candidate(
    grid: np.ndarray,
    car: Car,
    next_car: Optional[Car],
    parked: Sequence[Car],
    slots: int,
    useful_colors: Sequence[int],
) -> Candidate:
    assert car.color is not None
    assert car.remain is not None
    assert car.column is not None

    color = int(car.color)
    cap = int(car.remain)
    reachable, _ = reachable_summary(grid)
    r = int(reachable.get(color, 0))

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
    useful_new = _useful_exposed_cells(sim.exposed_by_color, useful_colors)

    utility = _utility_key(
        sim,
        useful_colors=useful_colors,
        queue_progress=1,
        reachable_hint=r,
    )
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
        utility=utility,
        utility_reason=_utility_reason(utility),
        queue_progress=1,
        heuristic_tiebreak=tuple(utility[5:]),
        next_color=next_color,
        next_capacity=next_capacity,
        flow_cleared_cells=sum(sim.cleared_by_color.values()),
        flow_final_occupied_upper=sim.final_occupied_upper,
        flow_exact=sim.exact_grid,
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
) -> List[Candidate]:
    del occupied_slots  # stable feasibility is already encoded by flow closure.

    next_by_col = {c.column: c for c in nxt if c.column is not None}
    front_by_col = {c.column: c for c in front if c.column is not None}

    active_front_colors = {c.color for c in front if c.color is not None}
    parked_colors = {c.color for c in parked if c.color is not None}
    next_colors = {c.color for c in nxt if c.color is not None}
    useful_colors = {
        int(c)
        for c in (active_front_colors | parked_colors | next_colors)
        if c is not None
    }

    candidates: List[Candidate] = []
    for car in front:
        if car.color is None or car.remain is None or car.column is None:
            continue
        candidates.append(
            _build_candidate(
                grid,
                car,
                next_by_col.get(car.column),
                parked,
                slots,
                useful_colors,
            )
        )

    # Queue progress is the fifth rule term, after deterministic board utility.
    # It can prefer a lane whose known next row can also be executed safely, but
    # it cannot outrank parked release/completion/clear/useful exposure.
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
                or next_car.column is None
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
            pair_sim = simulate_flow_closure(grid, parked, [first_car, promoted], slots)
            if not pair_sim.stable_safe:
                continue

            candidate.queue_progress = 2
            candidate.utility = (
                candidate.chain_parked_completions,
                candidate.guaranteed_completions,
                candidate.flow_cleared_cells,
                candidate.useful_newly_reachable,
                candidate.queue_progress,
                *candidate.heuristic_tiebreak,
            )
            candidate.utility_reason = _utility_reason(candidate.utility)

    safe = [candidate for candidate in candidates if not candidate.rejected]
    unsafe = [candidate for candidate in candidates if candidate.rejected]
    safe.sort(key=lambda candidate: candidate.utility, reverse=True)
    unsafe.sort(key=lambda candidate: candidate.column)
    return safe + unsafe


def _candidate_for_promoted_car(
    grid: np.ndarray,
    first_car: Car,
    next_car: Car,
    parked: Sequence[Car],
    slots: int,
    useful_colors: Sequence[int],
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
        useful_colors,
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
    """Choose a stable-safe pair only when ordered rule utility clearly improves."""
    free_slots = slots - occupied_slots
    if free_slots < _TWO_STEP_MIN_FREE_SLOTS:
        return None

    valid_first = [candidate for candidate in candidates if not candidate.rejected]
    if not valid_first:
        return None

    best_single = max(valid_first, key=lambda candidate: candidate.utility)
    front_by_col = {c.column: c for c in front if c.column is not None}
    next_by_col = {c.column: c for c in nxt if c.column is not None}

    active_front_colors = {c.color for c in front if c.color is not None}
    parked_colors = {c.color for c in parked if c.color is not None}
    next_colors = {c.color for c in nxt if c.color is not None}
    useful_colors = {
        int(c)
        for c in (active_front_colors | parked_colors | next_colors)
        if c is not None
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

        # First action must already be a valid single fallback; pair must also be
        # stable-safe.  No score can relax either condition.
        if first.rejected:
            return
        pair_sim = simulate_flow_closure(grid, parked, action_cars, slots)
        if not pair_sim.stable_safe:
            return

        pair_utility = _utility_key(
            pair_sim,
            useful_colors=useful_colors,
            queue_progress=2,
        )
        if not _two_step_primary_improves(pair_utility, best_single.utility):
            return

        useful_exposed = _useful_exposed_cells(
            pair_sim.exposed_by_color,
            useful_colors,
        )
        reason = (
            f"{reason_prefix}；A+B 共用单步相同的 hard feasibility + lexicographic utility；"
            f"稳定后停车占用最坏上界 {pair_sim.final_occupied_upper}/{slots}；"
            f"utility={_utility_reason(pair_utility)}"
        )
        if not pair_sim.exact_grid:
            reason += "；容量不足颜色只记数量，不猜删除坐标"

        plan = TwoStepPlan(
            first=first,
            second=second,
            second_source=second_source,
            utility=pair_utility,
            utility_reason=_utility_reason(pair_utility),
            free_slots_before=free_slots,
            reason=reason,
            guaranteed_completions=pair_sim.guaranteed_completions,
            guaranteed_parked_completions=pair_sim.guaranteed_parked_completions,
            cleared_cells=sum(pair_sim.cleared_by_color.values()),
            useful_exposed_cells=useful_exposed,
            queue_progress=2,
            heuristic_tiebreak=tuple(pair_utility[5:]),
            final_occupied_upper=pair_sim.final_occupied_upper,
            flow_exact=pair_sim.exact_grid,
        )
        if best_plan is None or plan.utility > best_plan.utility:
            best_plan = plan

    for first in valid_first:
        first_car = front_by_col.get(first.column)
        if first_car is None:
            continue

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
            useful_colors,
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


def best_valid_candidate(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    """Return the lexicographically best stable-safe candidate, or None."""
    valid = [candidate for candidate in candidates if not candidate.rejected]
    if not valid:
        return None
    return max(valid, key=lambda candidate: candidate.utility)
