from __future__ import annotations

from pathlib import Path

import numpy as np

from game_solver.config import EMPTY, GRID_COLS, GRID_ROWS, X_EDGE0_REF, X_STEP_REF, Y_EDGE0_REF, Y_STEP_REF
from game_solver.models import Candidate, Car
from game_solver import strategy


def _sim(*, parked: int, total: int, cleared: int, exposed: int, exact: bool = True):
    return strategy._FlowSimulation(
        grid=np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16),
        consumed_by_color={},
        exposed_by_color={2: exposed} if exposed else {},
        cleared_by_color={1: cleared} if cleared else {},
        guaranteed_completions=total,
        guaranteed_completions_by_color={},
        guaranteed_parked_completions=parked,
        guaranteed_parked_completions_by_color={},
        final_occupied_upper=0,
        stable_safe=True,
        exact_grid=exact,
        rounds=1,
    )


def _candidate(*, rejected: bool, utility: tuple[int, ...], column: int) -> Candidate:
    return Candidate(
        column=column,
        color=1,
        capacity=3,
        reachable=3,
        self_clear_guaranteed=False,
        some_completion_guaranteed=False,
        guaranteed_completions=0,
        chain_parked_completions=0,
        chain_parked_completion_by_color={},
        deterministic_clear_reachable=False,
        next_color_newly_reachable=0,
        useful_newly_reachable=0,
        unlocked_by_color={},
        rejected=rejected,
        reject_reason="unsafe" if rejected else "",
        utility=utility,
        utility_reason=str(utility),
        queue_progress=1,
        heuristic_tiebreak=tuple(utility[5:]),
        next_color=None,
        next_capacity=None,
        flow_cleared_cells=0,
        flow_final_occupied_upper=6 if rejected else 1,
        flow_exact=True,
    )


def test_existing_parked_release_is_first_lexicographic_term() -> None:
    release = strategy._utility_key(
        _sim(parked=1, total=1, cleared=0, exposed=0),
        useful_colors=[2],
        queue_progress=1,
    )
    huge_lower_level = strategy._utility_key(
        _sim(parked=0, total=9, cleared=999, exposed=999),
        useful_colors=[2],
        queue_progress=2,
    )
    assert release > huge_lower_level


def test_total_completion_precedes_clear_and_exposure() -> None:
    completion = strategy._utility_key(
        _sim(parked=0, total=2, cleared=0, exposed=0),
        useful_colors=[2],
        queue_progress=1,
    )
    lower = strategy._utility_key(
        _sim(parked=0, total=1, cleared=999, exposed=999),
        useful_colors=[2],
        queue_progress=2,
    )
    assert completion > lower


def test_rejected_candidate_cannot_win_even_with_huge_utility() -> None:
    unsafe = _candidate(rejected=True, utility=(99, 99, 99, 99, 99, 9, 9, 9), column=1)
    safe = _candidate(rejected=False, utility=(0, 0, 0, 0, 1, 0, 0, 0), column=2)
    assert strategy.best_valid_candidate([unsafe, safe]) is safe


def test_two_step_requires_primary_rule_improvement_not_tiebreak_only() -> None:
    single = (1, 2, 3, 4, 1, 0, 0, 0)
    tie_only = (1, 2, 3, 4, 1, 1, 9, 9)
    queue_progress = (1, 2, 3, 4, 2, 0, 0, 0)
    assert not strategy._two_step_primary_improves(tie_only, single)
    assert strategy._two_step_primary_improves(queue_progress, single)


def _cell_center(r: int, c: int) -> tuple[float, float]:
    return (
        X_EDGE0_REF + (c + 0.5) * X_STEP_REF,
        Y_EDGE0_REF + (r + 0.5) * Y_STEP_REF,
    )


def test_ambiguous_nearest_geometry_abstains() -> None:
    r = GRID_ROWS - 1
    left_c, right_c = 10, 12
    grid = np.full((GRID_ROWS, GRID_COLS), EMPTY, dtype=np.int16)
    grid[r, left_c] = 1
    grid[r, right_c] = 1
    lx, ly = _cell_center(r, left_c)
    rx, _ = _cell_center(r, right_c)
    car = Car("parked", None, 1, 3, (lx + rx) / 2.0, ly)

    predicted, removed = strategy._predict_nearest_partial_consumption(
        grid,
        1,
        1,
        [(car, 3)],
    )
    assert removed == 0
    np.testing.assert_array_equal(predicted, grid)


def test_clear_nearest_winner_is_lookahead_only_not_hard_grid() -> None:
    r = GRID_ROWS - 1
    near_c, far_c = 8, 18
    grid = np.full((GRID_ROWS, GRID_COLS), EMPTY, dtype=np.int16)
    grid[r, near_c] = 1
    grid[r, far_c] = 1
    x, y = _cell_center(r, near_c)
    parked = [Car("parked", None, 1, 1, x, y)]

    sim = strategy.simulate_flow_closure(grid, parked, [], slots=6)
    # Hard conservative grid is unchanged because capacity 1 cannot deterministically
    # choose which of the 2 reachable C01 cells disappears.
    assert int(np.count_nonzero(sim.grid == 1)) == 2
    assert sim.nearest_predicted_removed_by_color.get(1) == 1
    assert sim.stable_safe


def test_retired_additive_strategy_paths_are_absent() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "game_solver" / "strategy.py").read_text(encoding="utf-8")
    models = (root / "game_solver" / "models.py").read_text(encoding="utf-8")
    for token in (
        "def _score_flow(",
        "_QUEUE_LOOKAHEAD_WEIGHT",
        "_QUEUE_LOOKAHEAD_BONUS_CAP",
        "_TWO_STEP_MIN_GAIN",
        "_NEAREST_USEFUL_EXPOSE_WEIGHT",
        "_NEAREST_NEXT_EXPOSE_WEIGHT",
        "_NEAREST_SCORE_BONUS_CAP",
        "def simulate_clear_current_reachable_color(",
        "allow_deterministic_unlock",
    ):
        assert token not in text
    for token in (
        "score: float",
        "queue_unlock_bonus",
        "next_vehicle_score",
        "next_vehicle_chain_parked_completions",
        "next_vehicle_exact",
        "next_match_contacts",
        "neighbor_contacts",
    ):
        assert token not in models
