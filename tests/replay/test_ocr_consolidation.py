from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from game_solver import ocr
from game_solver.vehicles import ParkingDigitComponent
from tests.replay.manifest import cases_for


REPO_ROOT = Path(__file__).resolve().parents[2]


def _template_mask(digit: int) -> np.ndarray:
    packed = np.frombuffer(
        base64.b64decode(ocr._GAME_DIGIT_B64[digit]),
        dtype=np.uint8,
    )
    return np.unpackbits(packed)[: 32 * 48].reshape(48, 32)


def test_real_game_templates_are_the_canonical_classifier_source() -> None:
    for digit in range(10):
        result = ocr.recognize_digit(
            _template_mask(digit),
            source=f"template-contract-{digit}",
        )
        assert result.accepted
        assert result.digit == digit
        assert result.score >= 0.99
        assert result.margin > 0.0
        assert result.hole_count >= 0
        assert result.source == f"template-contract-{digit}"


def test_normal_and_preview_paths_share_one_digit_classifier() -> None:
    mask = _template_mask(6)
    normal = ocr.recognize_digit(mask, source="normal", profile="normal")
    preview = ocr.recognize_digit(mask, source="preview", profile="preview")

    assert normal.digit == preview.digit == 6
    assert normal.score == preview.score
    assert normal.margin == preview.margin
    assert normal.hole_count == preview.hole_count
    assert normal.accepted and preview.accepted


def test_template_contract_exposes_number_crop_vote_diagnostics(monkeypatch) -> None:
    masks = [_template_mask(2), _template_mask(6)]
    monkeypatch.setattr(
        ocr,
        "_extract_white_digit_masks",
        lambda _crop: masks,
    )
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    result = ocr.read_number_detailed_at(
        image,
        150.0,
        220.0,
        source="template-26-contract",
    )

    # This is a classifier/API contract using stored templates, not a claim
    # that the missing historical parked-26 screenshot has been replayed.
    assert result.value == 26
    assert result.candidate_value == 26
    assert result.agreeing_crops == 5
    assert result.vote_counts == ((26, 5),)
    assert len(result.crops) == 5
    for crop in result.crops:
        assert crop.value == 26
        assert len(crop.digits) == 2
        assert [item.digit for item in crop.digits] == [2, 6]
        assert all(item.margin > 0 for item in crop.digits)
        assert all(item.hole_count >= 0 for item in crop.digits)
        assert crop.source.startswith("template-26-contract:crop")



def test_parking_digit_components_keep_structural_diagnostics() -> None:
    fields = ParkingDigitComponent.__dataclass_fields__
    assert "score" in fields
    assert "margin" in fields
    assert "hole_count" in fields
    assert "ocr_source" in fields


def test_historical_parked_26_case_remains_pending_without_real_fixture() -> None:
    cases = {case.case_id: case for case in cases_for(subsystem="ocr")}
    case = cases["ocr-parking-26-not-20"]
    assert case.status == "pending_fixture"
    assert case.pending_reason


def test_duplicate_ocr_and_front_cache_architecture_is_removed() -> None:
    game_solver = REPO_ROOT / "game_solver"
    assert not (game_solver / "game_ocr.py").exists()

    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(game_solver.glob("*.py"))
    )
    for banned in (
        "install_game_digit_ocr",
        "_ORIGINAL_FAST",
        "_ORIGINAL_FULL",
        "_ORIGINAL_READ_NUMBER",
        "_PREVIEW_GAME_DIGIT_B64",
        "_preview_game_bank",
        "_recognize_preview_digit",
        "_DIGIT_SLOW_TEMPLATE_CACHE",
        "_digit_slow_templates",
        "_build_digit_templates",
        "FrontNumberCacheEntry",
        "front_number_cache",
        "_front_number_fingerprint",
        "read_front_numbers_cached",
    ):
        assert banned not in production


def test_raw_parking_monitor_remains_ocr_independent() -> None:
    text = (REPO_ROOT / "game_solver" / "monitor.py").read_text(encoding="utf-8")
    assert "from .ocr" not in text
    assert "import ocr" not in text
    assert "game_ocr" not in text
