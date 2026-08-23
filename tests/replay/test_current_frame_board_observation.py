from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from game_solver import board, engine
from game_solver.config import EMPTY, GRID_COLS, GRID_ROWS, UNKNOWN
from game_solver.models import ObservationHealth


def _image() -> np.ndarray:
    return np.zeros((24, 24, 3), dtype=np.float32)


def _palette() -> np.ndarray:
    return np.asarray(
        [[220.0, 40.0, 40.0], [40.0, 220.0, 40.0]],
        dtype=np.float32,
    )


def _direct_grid(default: int = UNKNOWN) -> np.ndarray:
    return np.full((GRID_ROWS, GRID_COLS), default, dtype=np.int16)


def _patch_current_classifier(monkeypatch, direct: np.ndarray) -> None:
    monkeypatch.setattr(
        board,
        "_fallback_background_rgb",
        lambda _image: np.asarray([128.0, 128.0, 128.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        board,
        "_classify_current_frame_grid",
        lambda *_a, **_k: (
            direct.copy(),
            {
                (r, c): "test:current"
                for r in range(GRID_ROWS)
                for c in range(GRID_COLS)
            },
        ),
    )


def test_current_frame_is_not_previous_mother_state(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    previous = _direct_grid(1)
    _patch_current_classifier(monkeypatch, direct)

    # Explicitly keep every current UNKNOWN unresolved. If observe_board started
    # from previous.copy(), this assertion would fail with a board full of C01.
    monkeypatch.setattr(
        board,
        "_resolve_unknown_from_history",
        lambda grid, evidence, *_a, **_k: (grid.copy(), dict(evidence), 0, {}, {}),
    )

    observed = board.observe_board(_image(), _palette(), previous)
    assert np.all(observed.grid == UNKNOWN)


def test_previous_empty_only_resolves_current_unknown(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    previous = _direct_grid(UNKNOWN)
    previous[0, 0] = EMPTY
    _patch_current_classifier(monkeypatch, direct)

    observed = board.observe_board(_image(), _palette(), previous)

    assert observed.grid[0, 0] == EMPTY
    assert "history:empty-invariant" in observed.evidence_by_cell[(0, 0)]
    assert observed.grid[0, 1] == UNKNOWN


def test_color_to_different_color_is_visible_conflict(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    direct[0, 0] = 2
    previous = _direct_grid(UNKNOWN)
    previous[0, 0] = 1
    _patch_current_classifier(monkeypatch, direct)

    observed = board.observe_board(_image(), _palette(), previous)

    assert observed.grid[0, 0] == 2  # current contradiction remains visible
    assert not observed.health.trusted
    assert (0, 0, 1, 2) in observed.health.transition_conflicts
    assert any("forbidden_transition" in reason for reason in observed.health.reasons)


def test_previous_empty_invariant_overrides_direct_sprite_color(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    direct[0, 0] = 1
    previous = _direct_grid(UNKNOWN)
    previous[0, 0] = EMPTY
    _patch_current_classifier(monkeypatch, direct)

    observed = board.observe_board(_image(), _palette(), previous)

    assert observed.grid[0, 0] == EMPTY
    assert observed.health.trusted
    assert observed.health.transition_conflicts == []
    assert (
        "empty-invariant-overrides-direct-color"
        in observed.evidence_by_cell[(0, 0)]
    )
    assert any(
        "persistent_empty_direct_color_overrides=1" in warning
        for warning in observed.health.warnings
    )


def test_persistent_empty_bounce_does_not_create_capacity_excess(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    previous = _direct_grid(UNKNOWN)

    # Already-empty cell with a neighboring sprite painted into it.
    previous[0, 0] = EMPTY
    direct[0, 0] = 1

    # The one real C01 removal caused by this action.
    previous[0, 1] = 1
    direct[0, 1] = EMPTY

    _patch_current_classifier(monkeypatch, direct)

    observed = board.observe_board(
        _image(),
        _palette(),
        previous,
        consumed_by_color={1: 1},
    )

    assert observed.grid[0, 0] == EMPTY
    assert observed.grid[0, 1] == EMPTY
    assert observed.visual_removed_by_color == {1: 1}
    assert observed.health.capacity_remaining_by_color == {}
    assert observed.health.capacity_excess_by_color == {}
    assert observed.health.trusted


def test_capacity_shortfall_does_not_force_another_cell_empty(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    direct[0, 0] = EMPTY
    direct[0, 1] = 1
    previous = _direct_grid(UNKNOWN)
    previous[0, 0] = 1
    previous[0, 1] = 1
    _patch_current_classifier(monkeypatch, direct)

    observed = board.observe_board(
        _image(),
        _palette(),
        previous,
        consumed_by_color={1: 2},
    )

    assert observed.grid[0, 0] == EMPTY
    assert observed.grid[0, 1] == 1  # quantity budget cannot mutate this coordinate
    assert observed.health.capacity_remaining_by_color == {1: 1}
    assert not observed.health.trusted


def test_unknown_is_not_filled_to_satisfy_capacity(monkeypatch):
    direct = _direct_grid(UNKNOWN)
    direct[0, 0] = EMPTY
    previous = _direct_grid(UNKNOWN)
    previous[0, 0] = 1
    previous[0, 1] = 1
    _patch_current_classifier(monkeypatch, direct)

    # Keep [0,1] unresolved so the test isolates the capacity contract.
    monkeypatch.setattr(
        board,
        "_resolve_unknown_from_history",
        lambda grid, evidence, *_a, **_k: (grid.copy(), dict(evidence), 0, {}, {}),
    )

    observed = board.observe_board(
        _image(),
        _palette(),
        previous,
        consumed_by_color={1: 2},
    )

    assert observed.grid[0, 1] == UNKNOWN
    assert observed.health.capacity_remaining_by_color == {1: 1}


def test_bottom_only_reachability_is_unchanged():
    grid = np.asarray(
        [
            [2, 2, 2],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int16,
    )
    totals, _neighbors = board.reachable_summary(grid)
    assert totals == {1: 3}


def test_untrusted_observed_board_cannot_enter_planner():
    observation = SimpleNamespace(
        health=ObservationHealth(
            trusted=False,
            reasons=["forbidden_transition R01C01:C01->C02"],
        )
    )
    assert not engine._observed_board_allows_planning(observation)
    assert engine._observed_board_allows_planning(
        observation, experimental_continue=True
    )
