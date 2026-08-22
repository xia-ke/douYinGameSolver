from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

VALID_STATUSES = {"active", "pending_fixture"}
VALID_KINDS = {"frame", "transition", "rule"}
MANIFEST_PATH = Path(__file__).with_name("cases.json")
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    subsystem: str
    status: str
    kind: str
    runner: str | None
    artifacts: Mapping[str, str | None]
    inputs: Mapping[str, Any]
    expected: Mapping[str, Any]
    provenance: str
    notes: str
    pending_reason: str | None

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_pending(self) -> bool:
        return self.status == "pending_fixture"


class ReplayManifestError(ValueError):
    pass


def _require_mapping(value: Any, field: str, case_id: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ReplayManifestError(f"{case_id}: {field} must be an object")
    return value


def _validate_case(raw: Mapping[str, Any], seen_ids: set[str]) -> ReplayCase:
    required = {
        "id",
        "subsystem",
        "status",
        "kind",
        "artifacts",
        "inputs",
        "expected",
        "provenance",
        "notes",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ReplayManifestError(f"case missing required fields: {', '.join(missing)}")

    case_id = str(raw["id"]).strip()
    if not case_id:
        raise ReplayManifestError("case id must not be empty")
    if case_id in seen_ids:
        raise ReplayManifestError(f"duplicate case id: {case_id}")
    seen_ids.add(case_id)

    subsystem = str(raw["subsystem"]).strip()
    status = str(raw["status"]).strip()
    kind = str(raw["kind"]).strip()
    runner_raw = raw.get("runner")
    runner = str(runner_raw).strip() if runner_raw else None
    artifacts = _require_mapping(raw["artifacts"], "artifacts", case_id)
    inputs = _require_mapping(raw["inputs"], "inputs", case_id)
    expected = _require_mapping(raw["expected"], "expected", case_id)
    provenance = str(raw["provenance"]).strip()
    notes = str(raw["notes"]).strip()
    pending_raw = raw.get("pending_reason")
    pending_reason = str(pending_raw).strip() if pending_raw else None

    if not subsystem:
        raise ReplayManifestError(f"{case_id}: subsystem must not be empty")
    if status not in VALID_STATUSES:
        raise ReplayManifestError(
            f"{case_id}: status must be one of {sorted(VALID_STATUSES)}, got {status!r}"
        )
    if kind not in VALID_KINDS:
        raise ReplayManifestError(
            f"{case_id}: kind must be one of {sorted(VALID_KINDS)}, got {kind!r}"
        )
    if not expected:
        raise ReplayManifestError(f"{case_id}: expected must contain structured assertions")
    if not provenance:
        raise ReplayManifestError(f"{case_id}: provenance must not be empty")

    if status == "pending_fixture":
        if not pending_reason:
            raise ReplayManifestError(f"{case_id}: pending_fixture requires pending_reason")
    else:
        if kind in {"frame", "transition"}:
            artifact_paths = [value for value in artifacts.values() if value]
            if not artifact_paths:
                raise ReplayManifestError(
                    f"{case_id}: active visual/transition case requires real artifact paths"
                )
            for relative_path in artifact_paths:
                path = Path(str(relative_path))
                if path.is_absolute():
                    raise ReplayManifestError(
                        f"{case_id}: artifact paths must be repository-relative: {path}"
                    )
                if not (REPO_ROOT / path).is_file():
                    raise ReplayManifestError(
                        f"{case_id}: active artifact is missing: {relative_path}"
                    )
        if not runner:
            raise ReplayManifestError(
                f"{case_id}: active case requires a registered runner name"
            )

    return ReplayCase(
        case_id=case_id,
        subsystem=subsystem,
        status=status,
        kind=kind,
        runner=runner,
        artifacts=dict(artifacts),
        inputs=dict(inputs),
        expected=dict(expected),
        provenance=provenance,
        notes=notes,
        pending_reason=pending_reason,
    )


def load_manifest(path: Path = MANIFEST_PATH) -> list[ReplayCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReplayManifestError("manifest root must be an object")
    if raw.get("schema_version") != 1:
        raise ReplayManifestError("manifest schema_version must be 1")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list):
        raise ReplayManifestError("manifest cases must be a list")

    seen_ids: set[str] = set()
    cases: list[ReplayCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ReplayManifestError("each case must be an object")
        cases.append(_validate_case(raw_case, seen_ids))
    return cases


def cases_for(*, subsystem: str | None = None, status: str | None = None) -> list[ReplayCase]:
    cases = load_manifest()
    if subsystem is not None:
        cases = [case for case in cases if case.subsystem == subsystem]
    if status is not None:
        cases = [case for case in cases if case.status == status]
    return cases
