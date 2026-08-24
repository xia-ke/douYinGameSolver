from __future__ import annotations

from pathlib import Path

import cv2

from game_solver.vehicles import extract_parking_digit_groups


_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "level16_parking_16_after_c04x38.png"
)


def test_level16_parking_16_is_not_truncated_to_single_1():
    image_bgr = cv2.imread(str(_FIXTURE), cv2.IMREAD_COLOR)
    assert image_bgr is not None

    groups, _mask = extract_parking_digit_groups(
        image_bgr,
        full_ocr=True,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.value == 16
    assert tuple(component.digit for component in group.components) == (1, 6)
