from __future__ import annotations

import numpy as np

import game_solver.strategy as strategy
from game_solver.models import Car


def _car(
    column: int,
    color: int | None,
    remain: int | None,
    *,
    source: str = "front",
) -> Car:
    return Car(
        source=source,
        column=column,
        color=color,
        remain=remain,
        x=float(column * 100),
        y=1800.0 if source == "front" else 1700.0,
    )


def _empty_grid() -> np.ndarray:
    return np.zeros((6, 6), dtype=np.int32)


def test_single_column_selects_unique_front_without_selection_work(monkeypatch):
    grid = _empty_grid()
    front = [_car(1, 1, 20)]
    nxt = [_car(1, 2, 18, source="next")]

    original_flow = strategy.simulate_flow_closure
    flow_calls = []

    def flow_spy(*args, **kwargs):
        flow_calls.append(kwargs.copy())
        return original_flow(*args, **kwargs)

    def selection_work_must_not_run(*args, **kwargs):
        raise AssertionError("single-column fast path must skip selection-only work")

    monkeypatch.setattr(strategy, "simulate_flow_closure", flow_spy)
    monkeypatch.setattr(strategy, "_utility_key", selection_work_must_not_run)
    monkeypatch.setattr(
        strategy,
        "_nearest_partial_exposure_prediction",
        selection_work_must_not_run,
    )

    candidates = strategy.evaluate_candidates(
        grid,
        front,
        nxt,
        parked=[],
        slots=6,
        occupied_slots=0,
        include_queue_lookahead=True,
        detected_queue_columns=1,
    )

    assert len(flow_calls) == 1
    assert flow_calls[0]["include_nearest_prediction"] is False

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.column == 1
    assert candidate.utility == ()
    assert "single-column fast path" in candidate.utility_reason
    assert candidate.queue_progress == 1
    assert candidate.rejected is False

    def flow_must_not_run(*args, **kwargs):
        raise AssertionError("two-step flow simulation must be skipped")

    monkeypatch.setattr(strategy, "simulate_flow_closure", flow_must_not_run)
    assert strategy.choose_two_step_plan(
        grid,
        front,
        nxt,
        parked=[],
        slots=6,
        occupied_slots=0,
        candidates=candidates,
    ) is None
    assert strategy.best_valid_candidate(candidates) is candidate


def test_single_column_hard_unsafe_is_still_rejected():
    grid = _empty_grid()
    front = [_car(1, 1, 20)]
    parked = [
        _car(i, 2 + i, 99, source="parked")
        for i in range(1, 6)
    ]

    candidates = strategy.evaluate_candidates(
        grid,
        front,
        nxt=[],
        parked=parked,
        slots=6,
        occupied_slots=5,
        include_queue_lookahead=True,
        detected_queue_columns=1,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.utility == ()
    assert candidate.flow_final_occupied_upper == 6
    assert candidate.rejected is True
    assert strategy.best_valid_candidate(candidates) is None
    assert strategy.choose_two_step_plan(
        grid,
        front,
        nxt=[],
        parked=parked,
        slots=6,
        occupied_slots=5,
        candidates=candidates,
    ) is None


def test_two_columns_keep_full_lexicographic_and_two_step_planning():
    grid = _empty_grid()
    front = [
        _car(1, 1, 20),
        _car(2, 2, 20),
    ]

    candidates = strategy.evaluate_candidates(
        grid,
        front,
        nxt=[],
        parked=[],
        slots=6,
        occupied_slots=0,
        include_queue_lookahead=True,
        detected_queue_columns=2,
    )

    assert len(candidates) == 2
    assert all(candidate.utility for candidate in candidates)
    assert all(
        "single-column fast path" not in candidate.utility_reason
        for candidate in candidates
    )

    plan = strategy.choose_two_step_plan(
        grid,
        front,
        nxt=[],
        parked=[],
        slots=6,
        occupied_slots=0,
        candidates=candidates,
    )
    assert plan is not None
    assert plan.queue_progress == 2
    assert strategy.best_valid_candidate(candidates) in candidates


def test_two_detected_columns_with_one_buildable_candidate_do_not_fast_path():
    grid = _empty_grid()
    front = [
        _car(1, 1, 20),
        _car(2, None, None),
    ]

    candidates = strategy.evaluate_candidates(
        grid,
        front,
        nxt=[],
        parked=[],
        slots=6,
        occupied_slots=0,
        include_queue_lookahead=True,
        detected_queue_columns=2,
    )

    assert len(candidates) == 1
    assert candidates[0].column == 1
    assert candidates[0].utility
    assert "single-column fast path" not in candidates[0].utility_reason


def test_untrusted_single_step_mode_does_not_use_trusted_fast_path():
    grid = _empty_grid()
    front = [_car(1, 1, 20)]

    candidates = strategy.evaluate_candidates(
        grid,
        front,
        nxt=[],
        parked=[],
        slots=6,
        occupied_slots=0,
        include_queue_lookahead=False,
        detected_queue_columns=1,
    )

    assert len(candidates) == 1
    assert candidates[0].utility
    assert "single-column fast path" not in candidates[0].utility_reason
