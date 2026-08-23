from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from game_solver.config import GRID_COLS, GRID_ROWS, STATE_VERSION
from game_solver.state import TrustedSessionState, load_state, save_state


def _state() -> TrustedSessionState:
    return TrustedSessionState(
        palette=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        previous_trusted_grid=np.ones((GRID_ROWS, GRID_COLS), dtype=np.int16),
        turn=7,
        screen_size=(940, 2048),
        parking_empty_ref=np.zeros((8, 12, 3), dtype=np.uint8),
        previous_trusted_grid_rgb=np.zeros(
            (GRID_ROWS, GRID_COLS, 3), dtype=np.float32
        ),
    )


def test_canonical_trusted_session_state_round_trip(tmp_path: Path):
    path = tmp_path / "solver_state.npz"
    save_state(path, _state())
    loaded = load_state(path)

    assert STATE_VERSION == 4
    assert loaded.turn == 7
    assert loaded.screen_size == (940, 2048)
    np.testing.assert_array_equal(loaded.previous_trusted_grid, _state().previous_trusted_grid)
    np.testing.assert_allclose(loaded.palette, _state().palette)
    assert loaded.previous_trusted_grid_rgb is not None

    with np.load(path) as raw:
        assert "previous_trusted_grid" in raw.files
        assert "previous_trusted_grid_rgb" in raw.files
        assert "grid" not in raw.files
        assert "grid_rgb_snapshot" not in raw.files


def test_old_v3_state_requires_explicit_reset(tmp_path: Path):
    path = tmp_path / "old_state.npz"
    np.savez_compressed(
        path,
        version=np.asarray([3], dtype=np.int32),
        palette=np.zeros((1, 3), dtype=np.float32),
        grid=np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int16),
        turn=np.asarray([1], dtype=np.int32),
        screen_size=np.asarray([940, 2048], dtype=np.int32),
        parking_empty_ref=np.zeros((4, 4, 3), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="--reset"):
        load_state(path)


def test_production_legacy_board_authorities_are_deleted():
    root = Path(__file__).resolve().parents[2]
    board_text = (root / "game_solver" / "board.py").read_text(encoding="utf-8")
    engine_text = (root / "game_solver" / "engine.py").read_text(encoding="utf-8")
    state_text = (root / "game_solver" / "state.py").read_text(encoding="utf-8")

    for forbidden in (
        "class CausalBoardUpdate",
        "def update_grid_causal(",
        "def update_grid(",
        "def observed_board_to_causal_update(",
    ):
        assert forbidden not in board_text

    assert "observe_board(" in engine_text
    assert "load_state_with_grid_rgb" not in engine_text
    assert "CausalBoardUpdate" not in engine_text
    assert "load_state_with_grid_rgb" not in state_text
    assert "def _load_state_core(" not in state_text


def test_state_names_make_history_role_explicit():
    fields = set(TrustedSessionState.__dataclass_fields__)
    assert "previous_trusted_grid" in fields
    assert "previous_trusted_grid_rgb" in fields
    assert "grid" not in fields
    assert "grid_rgb_snapshot" not in fields
