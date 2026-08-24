from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from game_solver import board
from game_solver.config import EMPTY, GRID_COLS, GRID_ROWS, UNKNOWN


_FIXTURES = Path(__file__).parent / "fixtures"

_UNKNOWN_COORDS_1BASED = [(1, 1), (1, 2), (1, 3), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 34), (1, 36), (1, 37), (1, 38), (2, 2), (2, 3), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23), (2, 28), (2, 29), (2, 30), (2, 32), (2, 33), (2, 34), (2, 35), (2, 36), (2, 37), (2, 38), (3, 1), (3, 2), (3, 16), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 23), (3, 28), (3, 29), (3, 30), (3, 31), (3, 32), (3, 33), (3, 34), (3, 35), (3, 36), (3, 37), (3, 38), (4, 1), (4, 28), (4, 29), (4, 30), (4, 32), (4, 33), (4, 34), (4, 35), (4, 36), (4, 37), (4, 38)]
_EXPECTED_HIDDEN_REMOVALS_1BASED = [(1, 19), (1, 21), (1, 22), (2, 16), (2, 19), (2, 21), (2, 28), (2, 29), (2, 30), (2, 32), (3, 17), (3, 18), (3, 19), (3, 20), (3, 21), (3, 22), (3, 28), (3, 29), (3, 30), (3, 31), (3, 32), (4, 28), (4, 29), (4, 30)]


def _load_rgb(name: str) -> np.ndarray:
    bgr = cv2.imread(str(_FIXTURES / name), cv2.IMREAD_COLOR)
    assert bgr is not None, name
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def test_level16_fixed_ui_exposes_24_anonymous_temporal_removals():
    before = _load_rgb("level16_ui_c02_before.png")
    after = _load_rgb("level16_ui_c02_after.png")

    previous = np.full((GRID_ROWS, GRID_COLS), EMPTY, dtype=np.int16)
    current = previous.copy()
    for r1, c1 in _UNKNOWN_COORDS_1BASED:
        previous[r1 - 1, c1 - 1] = UNKNOWN
        current[r1 - 1, c1 - 1] = UNKNOWN

    evidence = {
        (r, c): "test:unknown"
        for r in range(GRID_ROWS)
        for c in range(GRID_COLS)
    }
    resolved, _evidence, count, coords = (
        board._resolve_occluded_unknown_temporal_empty(
            current,
            evidence,
            previous,
            before,
            after,
        )
    )

    assert count == 24
    assert {(r + 1, c + 1) for r, c in coords} == set(
        _EXPECTED_HIDDEN_REMOVALS_1BASED
    )
    for r1, c1 in _EXPECTED_HIDDEN_REMOVALS_1BASED:
        assert int(resolved[r1 - 1, c1 - 1]) == EMPTY


def test_anonymous_ui_removals_can_explain_quantity_gap_without_choosing_colors():
    expected, remaining, excess, explained = (
        board._capacity_audit_with_occluded_unknown(
            {2: 33},
            {2: 9},
            24,
        )
    )
    assert expected == {2: 33}
    assert remaining == {}
    assert excess == {}
    assert explained == {2: 24}


def test_anonymous_ui_count_must_exactly_match_total_shortfall():
    _expected, remaining, excess, explained = (
        board._capacity_audit_with_occluded_unknown(
            {2: 33},
            {2: 9},
            23,
        )
    )
    assert remaining == {2: 24}
    assert excess == {}
    assert explained == {}
