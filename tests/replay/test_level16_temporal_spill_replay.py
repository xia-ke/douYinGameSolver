from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from game_solver import board
from game_solver.config import EMPTY


_FIXTURES = Path(__file__).parent / "fixtures"


def _load_rgb(name: str) -> np.ndarray:
    bgr = cv2.imread(str(_FIXTURES / name), cv2.IMREAD_COLOR)
    assert bgr is not None, name
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def test_level16_c01_64_removal_ignores_adjacent_sprite_spill():
    before = _load_rgb("level16_c01_64_before.png")
    after = _load_rgb("level16_c01_64_after.png")

    palette = board.learn_palette(before)
    previous = board.initial_grid(before, palette)
    previous_rgb = board.sample_grid_rgb_snapshot(before)

    # R48C28 is one of the C01 cells that really disappears in this transition.
    consumed_color = int(previous[47, 27])
    assert consumed_color > 0

    observed = board.observe_board(
        after,
        palette,
        previous,
        previous_trusted_frame_rgb=before,
        previous_trusted_grid_rgb=previous_rgb,
        consumed_by_color={consumed_color: 64},
    )

    # 44 cells are directly seen as background. The remaining 20 are visually
    # contaminated by neighboring sprites and must be recovered by independent
    # old-body-loss evidence, not by a capacity top-N guess.
    spill_cells = [
        (47, 27), (47, 28), (47, 29),
        (48, 8), (48, 12), (48, 13), (48, 25), (48, 26), (48, 30),
        (49, 7), (49, 9), (49, 10), (49, 11), (49, 14), (49, 24),
        (50, 15),
        (50, 6), (50, 31), (51, 5), (51, 32),
    ]
    for rc in spill_cells:
        assert int(observed.grid[rc]) == EMPTY, rc

    assert observed.visual_removed_by_color.get(consumed_color) == 64
    assert observed.direct_empty_by_color.get(consumed_color) == 44
    assert observed.temporal_empty_by_color.get(consumed_color) == 20
    assert observed.health.transition_conflicts == []
    assert observed.health.capacity_remaining_by_color == {}
    assert observed.health.capacity_excess_by_color == {}
    assert observed.health.trusted
