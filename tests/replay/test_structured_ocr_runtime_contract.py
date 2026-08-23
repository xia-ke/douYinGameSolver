from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from game_solver import engine
from game_solver.ocr import NumberOcrResult


ROOT = Path(__file__).resolve().parents[2]


def _ocr_result(value: int | None) -> NumberOcrResult:
    return NumberOcrResult(
        value=value,
        candidate_value=value,
        confidence=0.99,
        agreeing_crops=5,
        source="runtime-contract",
        crops=(),
        vote_counts=((value, 5),) if value is not None else (),
    )


def test_front_candidate_signature_consumes_structured_ocr_result(monkeypatch) -> None:
    monkeypatch.setattr(engine, "car_color_at", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(
        engine,
        "read_number_detailed_at",
        lambda *_args, **_kwargs: _ocr_result(26),
    )

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result = SimpleNamespace(
        front=[SimpleNamespace(column=2, x=60.0, y=90.0)],
        palette=np.zeros((1, 3), dtype=np.float32),
    )
    candidate = SimpleNamespace(column=2)

    assert engine._front_candidate_signature_on_frame(
        frame,
        result,
        candidate,
    ) == (7, 26)


def test_structured_number_ocr_is_never_tuple_unpacked_in_production() -> None:
    violations: list[str] = []

    for path in sorted((ROOT / "game_solver").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue

            value = getattr(node, "value", None)
            if not isinstance(value, ast.Call):
                continue

            func = value.func
            called_name = None
            if isinstance(func, ast.Name):
                called_name = func.id
            elif isinstance(func, ast.Attribute):
                called_name = func.attr

            if called_name != "read_number_detailed_at":
                continue

            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, (ast.Tuple, ast.List)) for target in targets):
                violations.append(f"{path.name}:{node.lineno}")

    assert not violations, (
        "read_number_detailed_at() returns NumberOcrResult; "
        "tuple-unpack call sites are invalid: " + ", ".join(violations)
    )


def test_final_production_has_no_version_patch_labels_or_fixed_front_x_constant() -> None:
    paths = [ROOT / "README.md"]
    paths.extend(sorted((ROOT / "game_solver").glob("*.py")))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert re.search(r"\bv5\.\d+\b", combined) is None
    assert "FRONT_X_N" not in combined
