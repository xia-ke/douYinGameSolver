"""Layer runners for active real replay fixtures.

Register a runner here when a pending case is promoted to active. Keeping runner
registration in one module makes active coverage explicit and prevents a manifest-only
status change from silently passing.
"""

# Intentionally empty until the first real visual fixture is promoted to active.
# Example:
#
# from .harness import register_runner
#
# @register_runner("parking_ocr")
# def run_parking_ocr(case, repo_root):
#     ...

# BEGIN REAL-FIXTURE RUNNERS
from pathlib import Path

import cv2

from game_solver.unlock import _looks_like_game_screen
from .harness import register_runner


def _load_bgr_artifact(case, repo_root: Path, key: str):
    relative = case.artifacts.get(key)
    if not relative:
        raise AssertionError(
            f"{case.case_id}: missing artifact path for {key}"
        )
    path = repo_root / relative
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError(
            f"{case.case_id}: cannot read image artifact: {relative}"
        )
    return image


@register_runner("unlock_game_screen")
def run_unlock_game_screen(case, repo_root: Path):
    image_bgr = _load_bgr_artifact(
        case,
        repo_root,
        "stable_screenshot",
    )
    return {
        "is_game_screen": bool(_looks_like_game_screen(image_bgr)),
    }
# END REAL-FIXTURE RUNNERS
