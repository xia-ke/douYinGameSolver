from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from game_solver import engine
from game_solver.models import ObservationHealth


def _observation(trusted: bool):
    return SimpleNamespace(
        health=ObservationHealth(
            trusted=trusted,
            reasons=[] if trusted else ["test current-frame conflict"],
        )
    )


def test_observation_health_reason_immediately_revokes_trust() -> None:
    health = ObservationHealth(trusted=True)
    health.add_warning("diagnostic only")
    assert health.trusted
    assert health.warnings == ["diagnostic only"]

    health.add_reason("parking_ocr_incomplete=1/2")
    assert not health.trusted
    assert health.reasons == ["parking_ocr_incomplete=1/2"]


def test_default_untrusted_observation_does_not_invoke_planner(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("planner must not run for default untrusted observation")

    monkeypatch.setattr(engine, "evaluate_candidates", forbidden)
    candidates, plan = engine._plan_current_observation(
        _observation(False),
        np.zeros((2, 2), dtype=np.int16),
        [], [], [], 6, 0,
        experimental_continue=False,
    )
    assert candidates == []
    assert plan is None


def test_experimental_untrusted_observation_is_single_step_only(monkeypatch) -> None:
    calls = []
    candidate = object()

    def fake_evaluate(*_args, **kwargs):
        calls.append(kwargs)
        return [candidate]

    def forbidden_two_step(*_args, **_kwargs):
        raise AssertionError("untrusted experiment must not create a two-step plan")

    monkeypatch.setattr(engine, "evaluate_candidates", fake_evaluate)
    monkeypatch.setattr(engine, "choose_two_step_plan", forbidden_two_step)

    candidates, plan = engine._plan_current_observation(
        _observation(False),
        np.zeros((2, 2), dtype=np.int16),
        [], [], [], 6, 0,
        experimental_continue=True,
    )
    assert candidates == [candidate]
    assert plan is None
    assert calls == [{"include_queue_lookahead": False}]


def test_trusted_observation_uses_normal_planner(monkeypatch) -> None:
    candidate = object()
    plan = object()
    calls = []

    def fake_evaluate(*_args, **kwargs):
        calls.append(kwargs)
        return [candidate]

    monkeypatch.setattr(engine, "evaluate_candidates", fake_evaluate)
    monkeypatch.setattr(engine, "choose_two_step_plan", lambda *_a, **_k: plan)

    candidates, selected = engine._plan_current_observation(
        _observation(True),
        np.zeros((2, 2), dtype=np.int16),
        [], [], [], 6, 0,
        experimental_continue=False,
    )
    assert candidates == [candidate]
    assert selected is plan
    assert calls == [{"include_queue_lookahead": True}]


def test_safety_runtime_and_legacy_gate_plumbing_are_removed() -> None:
    root = Path(__file__).resolve().parents[2]
    engine_text = (root / "game_solver" / "engine.py").read_text(encoding="utf-8")
    models_text = (root / "game_solver" / "models.py").read_text(encoding="utf-8")
    cli_text = (root / "game_solver" / "cli.py").read_text(encoding="utf-8")

    for forbidden in (
        "class _SafetyRuntime",
        "PAUSED_SYNC",
        "PAUSED_SAFE",
        "conservative_turns",
        "clean_streak",
        "board_incomplete_streak",
        "guarantee_break_rounds",
        "strategy_untrusted_colors",
        "force_single_step",
        "disable_parked_chain",
        "safe_pause_retry_delay",
        "board_update_status",
        "causal_input_invalid",
        "model_conflict_colors",
    ):
        assert forbidden not in engine_text
        assert forbidden not in models_text
        assert forbidden not in cli_text

    fields = set(engine.AnalysisResult.__dataclass_fields__)
    assert "observation_health" in fields
    assert "board_update_status" not in fields
    assert "strategy_untrusted_colors" not in fields
    assert "model_conflict_colors" not in fields
