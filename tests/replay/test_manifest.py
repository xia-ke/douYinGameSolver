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

PROMOTED_REAL_CASES = {
    "unlock-three-column-game-screen",
}


def test_manifest_registers_all_audit_historical_cases() -> None:
    cases = load_manifest()
    assert {case.case_id for case in cases} == EXPECTED_HISTORICAL_CASES


def test_only_real_recovered_historical_cases_are_active() -> None:
    cases = load_manifest()
    active = {case.case_id for case in cases if case.is_active}
    assert active == PROMOTED_REAL_CASES

    for case in cases:
        assert case.expected
        if case.case_id in PROMOTED_REAL_CASES:
            assert case.status == "active"
            assert not case.pending_reason
        else:
            assert case.status == "pending_fixture"
            assert case.pending_reason


def test_pending_cases_never_claim_nonexistent_artifacts() -> None:
    for case in load_manifest():
        if not case.is_pending:
            continue
        for artifact_path in case.artifacts.values():
            if artifact_path is not None:
                assert (REPO_ROOT / artifact_path).is_file()
