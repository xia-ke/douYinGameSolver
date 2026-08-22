from __future__ import annotations

from .manifest import REPO_ROOT, load_manifest


EXPECTED_HISTORICAL_CASES = {
    "board-c01-75-removal",
    "ocr-parking-26-not-20",
    "board-c08-ghost-cells-empty",
    "unlock-three-column-game-screen",
    "strategy-nearest-cell-routing",
    "strategy-same-color-low-remain-priority",
}


def test_manifest_registers_all_audit_historical_cases() -> None:
    cases = load_manifest()
    assert {case.case_id for case in cases} == EXPECTED_HISTORICAL_CASES


def test_missing_historical_visual_evidence_is_explicitly_pending() -> None:
    cases = load_manifest()
    assert cases
    for case in cases:
        assert case.status == "pending_fixture"
        assert case.pending_reason
        assert case.expected


def test_pending_cases_never_claim_nonexistent_artifacts() -> None:
    for case in load_manifest():
        if not case.is_pending:
            continue
        for artifact_path in case.artifacts.values():
            if artifact_path is not None:
                assert (REPO_ROOT / artifact_path).is_file()
