from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .board import ctag, reachable_components, reachable_summary
from .config import EMPTY, UNKNOWN
from .models import Car, Candidate, TwoStepPlan

def parked_remainders_by_color(parked: Sequence[Car]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for car in parked:
        if car.color is None:
            continue
        out[car.color].append(car.remain if car.remain is not None else 99)
    return out


def simulate_clear_current_reachable_color(
    grid: np.ndarray,
    color: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """确定性一层模拟：把“当前已经可达”的该颜色连通块全部移除，再看新暴露颜色。"""
    before, _ = reachable_summary(grid)
    comps, _ = reachable_components(grid)
    groups = comps.get(color, [])
    sim = grid.copy()
    for group in groups:
        for r, c in group:
            sim[r, c] = EMPTY
    after, _ = reachable_summary(sim)
    unlocked: Dict[int, int] = {}
    for other in set(before) | set(after):
        if other == color:
            continue
        delta = int(after.get(other, 0)) - int(before.get(other, 0))
        if delta > 0:
            unlocked[other] = delta
    return sim, unlocked


def evaluate_candidates(
    grid: np.ndarray,
    front: Sequence[Car],
    nxt: Sequence[Car],
    parked: Sequence[Car],
    slots: int,
    occupied_slots: int,
) -> List[Candidate]:
    reachable, neighbor_contacts = reachable_summary(grid)
    parked_by_color = parked_remainders_by_color(parked)
    next_by_col = {c.column: c for c in nxt if c.column is not None}

    candidates: List[Candidate] = []
    occupied = occupied_slots
    active_front_colors = {c.color for c in front if c.color is not None}
    parked_colors = {c.color for c in parked if c.color is not None}
    useful_colors = active_front_colors | parked_colors

    # 同一颜色可能同时出现在多列；确定性清层结果只需算一次。
    deterministic_unlock_cache: Dict[int, Dict[int, int]] = {}

    for car in front:
        if car.color is None or car.remain is None or car.column is None:
            continue

        color = car.color
        cap = car.remain
        r = int(reachable.get(color, 0))
        existing = list(parked_by_color.get(color, []))
        all_rems = existing + [cap]

        self_clear = (len(existing) == 0 and r >= cap)
        total_capacity = sum(all_rems)
        guaranteed_moved = min(r, total_capacity)
        no_completion_max = sum(max(0, x - 1) for x in all_rems)
        guaranteed_completions = max(0, guaranteed_moved - no_completion_max)
        some_completion = guaranteed_completions >= 1

        rejected = False
        reason = ""
        if occupied >= slots - 1 and not (self_clear or some_completion):
            rejected = True
            reason = (
                f"只剩最后一个停车位；当前只能确定 {r} 个 {ctag(color)} 可进入，"
                "不能保证至少一辆车消失"
            )

        contacts = neighbor_contacts.get(color, Counter())
        next_car = next_by_col.get(car.column)
        next_color = next_car.color if next_car else None
        next_match_contacts = int(contacts.get(next_color, 0)) if next_color else 0

        # 若没有同色停车车，且新车容量足以吃完“当前全部可达该色”，
        # 那么这一层移除是确定的，可以安全模拟它会新暴露什么。
        deterministic_clear = (r > 0 and len(existing) == 0 and cap >= r)
        unlocked: Dict[int, int] = {}
        if deterministic_clear:
            if color not in deterministic_unlock_cache:
                _sim, cached_unlocked = simulate_clear_current_reachable_color(grid, color)
                deterministic_unlock_cache[color] = dict(cached_unlocked)
            unlocked = deterministic_unlock_cache[color]

        next_new = int(unlocked.get(next_color, 0)) if next_color is not None else 0
        useful_new = sum(v for k, v in unlocked.items() if k in useful_colors)
        all_new = sum(unlocked.values())

        # ----- 启发式评分；硬安全规则仍与评分完全独立 -----
        score = 0.0
        if self_clear:
            score += 50000.0
        elif some_completion:
            score += 42000.0

        if deterministic_clear:
            score += 20000.0
            # “点这一列后顶上来的下一辆颜色”被确定性打开，奖励最高。
            score += min(next_new, 30) * 2500.0
            # 已停车/已在第一排的其他颜色被打开，也有明显价值。
            score += min(useful_new, 40) * 600.0
            score += min(all_new, 60) * 100.0

        fill_ratio = min(1.0, r / max(1, cap))
        score += 2500.0 * fill_ratio
        score += min(r, 100) * 6.0

        if self_clear:
            score += max(0, r - cap) * 15.0
            score += max(0, 60 - cap) * 2.0
        else:
            score -= 1000.0 + occupied * 450.0

        # 仍保留旧的邻层轻量信息，作为没有确定性模拟时的次级依据。
        score += next_match_contacts * 100.0
        if next_color is not None:
            score += min(40, int(reachable.get(next_color, 0))) * 35.0
        for other_color, contact_n in contacts.items():
            if other_color in useful_colors:
                score += min(contact_n, 30) * 15.0

        if existing and not some_completion:
            score -= 500.0
        if r == 0:
            score -= 1800.0
        if rejected:
            score = -1e12

        candidates.append(Candidate(
            column=car.column,
            color=color,
            capacity=cap,
            reachable=r,
            self_clear_guaranteed=self_clear,
            some_completion_guaranteed=some_completion,
            guaranteed_completions=guaranteed_completions,
            deterministic_clear_reachable=deterministic_clear,
            next_color_newly_reachable=next_new,
            useful_newly_reachable=useful_new,
            unlocked_by_color=dict(sorted(unlocked.items())),
            rejected=rejected,
            reject_reason=reason,
            score=score,
            next_color=next_color,
            next_match_contacts=next_match_contacts,
            neighbor_contacts=dict(contacts),
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


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
    停车空位 >= 3 时，允许同一分析帧连续点击两辆“当前第一排”的车。

    为避免依赖队列补位动画：
    - 两步必须来自不同列；
    - 第二步仍是当前截图里已经可点击的第一排车辆；
    - 不点击第一步之后才新顶上来的下一辆。

    预测规则：
    1. 如果第一步能确定清空当前可达该色，真正模拟棋盘后再评价第二步；
    2. 如果第一步能确定自己装满消失但清除位置不唯一，
       对不同颜色第二步使用当前棋盘作为保守下界；
    3. 如果两步同色且第一步只确定自己装满，
       只有“当前可达数 - 第一车容量 >= 第二车容量”时才允许，
       这样第二辆也仍能由当前已知可达色块保证装满；
    4. 第一车若可能停留，则把它保守加入 parked 再评价第二步。

    因为空位至少3个，哪怕两辆都完全不消失，连续点击后最多只占到5/6，
    不会触碰最后一个车位的硬风险。
    """
    free_slots = slots - occupied_slots
    if free_slots < 3:
        return None

    valid_first = [c for c in candidates if not c.rejected]
    if len(valid_first) < 2:
        return None

    front_by_col = {
        c.column: c for c in front
        if c.column is not None
    }
    original_by_col = {
        c.column: c for c in candidates
        if c.column is not None
    }

    best_plan: Optional[TwoStepPlan] = None

    for first in valid_first:
        first_car = front_by_col.get(first.column)
        if first_car is None:
            continue

        sim_grid = grid
        simulated_exactly = False

        if first.deterministic_clear_reachable:
            sim_grid, _unlocked = simulate_clear_current_reachable_color(
                grid, first.color
            )
            simulated_exactly = True

        parked_after = list(parked)
        occupied_after = occupied_slots

        if not first.self_clear_guaranteed:
            # 保守：不假定同色已有车真的消失，也不假定新车已经吃了多少。
            remain_after = first.capacity

            # deterministic_clear 只会在“此前没有同色停车车”时成立，
            # 因此此时可精确知道第一车吃完当前可达色后还剩多少。
            if first.deterministic_clear_reachable:
                remain_after = max(
                    1,
                    first.capacity - first.reachable,
                )

            parked_after.append(
                Car(
                    source="parked",
                    column=None,
                    color=first.color,
                    remain=remain_after,
                    x=first_car.x,
                    y=first_car.y,
                )
            )
            occupied_after += 1

        remaining_front = [
            car for car in front
            if car.column != first.column
        ]

        second_candidates = evaluate_candidates(
            sim_grid,
            remaining_front,
            nxt,
            parked_after,
            slots,
            occupied_after,
        )

        for second in second_candidates:
            if second.rejected:
                continue
            if second.column == first.column:
                continue

            # 连续第二步至少应该能接到当前已知色块；
            # 否则只是为了加速而白占车位，不符合保守策略。
            if second.reachable <= 0 and not second.some_completion_guaranteed:
                continue

            reason_parts: List[str] = []

            # 第一车和第二车同色时，如果第一步不是“整层确定清除”，
            # 原棋盘的 reachable 会包含已被第一车吃掉的部分。
            if (
                first.color == second.color
                and not first.deterministic_clear_reachable
            ):
                if not first.self_clear_guaranteed:
                    # 第一车可能停留，同色分流具体分配不可预测。
                    continue

                remaining_lower_bound = max(
                    0,
                    first.reachable - first.capacity,
                )

                # 只有当前已经确定剩下的同色块仍足够装满第二辆，
                # 才允许把两辆同色车连续点击。
                if remaining_lower_bound < second.capacity:
                    continue

                reason_parts.append(
                    f"同色保守下界仍有 {remaining_lower_bound} 个，"
                    f"足够第二辆 {second.capacity}"
                )

            original_second = original_by_col.get(second.column)
            improvement = 0.0
            if original_second is not None:
                improvement = second.score - original_second.score

            pair_score = first.score + 0.92 * second.score

            # 第一动作如果确定打开了第二动作所需颜色，
            # 让“先开路、再吃新层”的组合优先。
            if simulated_exactly and improvement > 0:
                pair_score += 0.35 * improvement
                reason_parts.append("第一步确定性开路后第二步评分提高")

            # 两辆都可能留下时虽然停车位仍安全，但略微降权，
            # 避免为了追求双击速度无谓堆车。
            if (
                not first.self_clear_guaranteed
                and not first.some_completion_guaranteed
            ):
                pair_score -= 700.0
            if (
                not second.self_clear_guaranteed
                and not second.some_completion_guaranteed
            ):
                pair_score -= 500.0

            if not reason_parts:
                if simulated_exactly:
                    reason_parts.append("按第一步确定性清层后的棋盘评价第二步")
                else:
                    reason_parts.append("第二步使用当前棋盘的保守可达信息")

            plan = TwoStepPlan(
                first=first,
                second=second,
                score=float(pair_score),
                free_slots_before=free_slots,
                first_simulated_exactly=simulated_exactly,
                reason="；".join(reason_parts),
            )

            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan

    return best_plan


def format_two_step_plan(plan: Optional[TwoStepPlan]) -> str:
    if plan is None:
        return ""

    return (
        f"连续两步计划: 当前空位 {plan.free_slots_before} >= 3，"
        f"优先点击第{plan.first.column}列 "
        f"{ctag(plan.first.color)}×{plan.first.capacity}，"
        f"随后不等分流结束直接点击第{plan.second.column}列 "
        f"{ctag(plan.second.color)}×{plan.second.capacity}。\\n"
        f"两步预测依据: {plan.reason}；pair_score={plan.score:.1f}"
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
    lines.append(f"本关动态颜色类别: {palette_count} 种" + (f"（本帧新增 {new_colors_added} 种）" if new_colors_added else ""))
    lines.append(f"棋盘: 已知色块 {known} | 已空 {empty} | UI/未知 {unknown}")
    if removed_since_last is not None:
        lines.append(f"相对上一张完整分析截图新消失: {removed_since_last} 个小色块")

    if reachable:
        reach_text = ", ".join(
            f"{ctag(c)}={n}" for c, n in sorted(reachable.items(), key=lambda kv: kv[1], reverse=True)
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
            lines.append(
                f"第{col}列第一排: {ctag(f.color)} / 数字 {f.remain if f.remain is not None else '识别失败'}"
                f"    下一辆: {ctag(n.color) if n else 'UNKNOWN'}"
            )
        else:
            lines.append(f"第{col}列第一排: 未检测到可点击车辆")

    lines.append("")
    lines.append(f"停车位: 数字锚点检测 {occupied_slots} / {slots}")
    for i, car in enumerate(parked, 1):
        lines.append(f"  停车车{i}: {ctag(car.color)} / 剩余 {car.remain if car.remain is not None else '识别失败'}")

    lines.append("")
    lines.append("候选动作:")
    for c in candidates:
        flags: List[str] = []
        if c.self_clear_guaranteed:
            flags.append("确定自己装满消失")
        elif c.some_completion_guaranteed:
            flags.append(f"确定至少消失 {c.guaranteed_completions} 辆同色车")
        if c.deterministic_clear_reachable:
            flags.append("确定清空当前可达该色")
        if c.next_color_newly_reachable > 0:
            flags.append(f"确定新开下一辆颜色 +{c.next_color_newly_reachable}")
        if c.rejected:
            flags.append("硬禁止")
        if c.next_color:
            flags.append(f"下一辆 {ctag(c.next_color)}")
        lines.append(
            f"  第{c.column}列 {ctag(c.color)}×{c.capacity}: "
            f"可触达={c.reachable}, score={c.score:.1f}"
            + (" | " + "；".join(flags) if flags else "")
        )
        if c.unlocked_by_color:
            unlock_text = ", ".join(f"{ctag(k)}+{v}" for k, v in c.unlocked_by_color.items())
            lines.append(f"      一层确定性模拟新暴露: {unlock_text}")
        if c.rejected:
            lines.append(f"      原因: {c.reject_reason}")

    valid = [c for c in candidates if not c.rejected]
    lines.append("")
    if not valid:
        lines.append("建议: 不点击。当前没有满足硬安全约束的候选动作。")
    else:
        best = valid[0]
        lines.append(
            f"下一步建议: 点击【第一排第 {best.column} 列】 "
            f"{ctag(best.color)} / 数字 {best.capacity}"
        )
        if best.next_color_newly_reachable > 0:
            lines.append(
                f"程序依据: 一层确定性模拟表明，该动作会把下一辆 {ctag(best.next_color)} "
                f"对应颜色新增暴露 {best.next_color_newly_reachable} 个。"
            )
        elif best.self_clear_guaranteed:
            lines.append(
                f"程序依据: 当前确定可触达 {best.reachable} 个同类色块 >= {best.capacity}，"
                "且停车区没有检测到同色分流车，因此可确定该车装满并消失。"
            )
        elif best.some_completion_guaranteed:
            lines.append(
                f"程序依据: 不预测同色分流位置，但按最坏分配仍可保证至少 {best.guaranteed_completions} 辆同色车消失。"
            )
        else:
            lines.append("程序依据: 该动作通过停车位硬安全检查，并在当前确定性解锁与启发式评分中最高。")
    return "\n".join(lines)


def best_valid_candidate(candidates: Sequence[Candidate]) -> Optional[Candidate]:
    for candidate in candidates:
        if not candidate.rejected:
            return candidate
    return None