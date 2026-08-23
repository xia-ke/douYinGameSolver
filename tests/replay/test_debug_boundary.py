from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from game_solver import debug, engine, strategy
from game_solver.models import ObservationHealth


def _diagnostic_result():
    return SimpleNamespace(
        palette=np.asarray([[200.0, 100.0, 50.0]], dtype=np.float32),
        grid=np.zeros((2, 3), dtype=np.int16),
        turn=4,
        front=[],
        nxt=[],
        parked=[],
        occupied_slots=0,
        report="planner-report-contract",
        observation_health=ObservationHealth(trusted=True),
    )


def test_debug_module_owns_three_log_boundaries(tmp_path: Path) -> None:
    result = _diagnostic_result()
    screenshot = tmp_path / "stable.png"
    color_log = tmp_path / "color.txt"
    number_log = tmp_path / "number.txt"
    decision_log = tmp_path / "decision.txt"

    debug._append_color_observation_log(
        color_log, screenshot=screenshot, result=result, step_label="auto-4"
    )
    debug._append_number_observation_log(
        number_log, screenshot=screenshot, result=result, step_label="auto-4"
    )
    debug._append_decision_log(
        decision_log, screenshot=screenshot, result=result, step_label="auto-4"
    )

    color = color_log.read_text(encoding="utf-8")
    number = number_log.read_text(encoding="utf-8")
    decision = decision_log.read_text(encoding="utf-8")
    assert "[COLOR_SNAPSHOT]" in color
    assert "[BOARD_GRID]" in color
    assert "observation_trusted=yes" in color
    assert "[NUMBER_SNAPSHOT]" in number
    assert "[QUEUE_NUMBERS]" in number
    assert "observation_trusted=yes" in number
    assert "[DECISION]" in decision
    assert "planner-report-contract" in decision
    assert "step=auto-4" in color and "step=auto-4" in number and "step=auto-4" in decision


def test_engine_and_strategy_no_longer_define_presentation_formatters() -> None:
    root = Path(__file__).resolve().parents[2]
    engine_text = (root / "game_solver" / "engine.py").read_text(encoding="utf-8")
    strategy_text = (root / "game_solver" / "strategy.py").read_text(encoding="utf-8")
    debug_text = (root / "game_solver" / "debug.py").read_text(encoding="utf-8")

    for name in (
        "_append_color_observation_log",
        "_append_number_observation_log",
        "_append_decision_log",
        "_format_palette_diagnostics",
        "_format_observed_board_observation",
    ):
        assert f"def {name}(" not in engine_text
        assert f"def {name}(" in debug_text

    assert "def format_report(" not in strategy_text
    assert "def format_two_step_plan(" not in strategy_text
    assert "def format_report(" in debug_text
    assert "def format_two_step_plan(" in debug_text
    assert "from .debug import (" in engine_text


def test_auto_runner_exposes_explicit_orchestration_phases() -> None:
    root = Path(__file__).resolve().parents[2]
    engine_text = (root / "game_solver" / "engine.py").read_text(encoding="utf-8")
    for phase in (
        "PHASE 1 — stable capture",
        "PHASE 2 — perception + deterministic planning",
        "PHASE 3 — validation / strict trust gate",
        "PHASE 4 — trusted context commit",
        "PHASE 5 — execute the selected stable-safe action(s)",
        "PHASE 6 — wait for parking absorption/animation to become stable",
    ):
        assert phase in engine_text


def test_capture_helper_keeps_adb_and_display_optional_contract(tmp_path, monkeypatch) -> None:
    shot_calls = []
    display_calls = []
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    args = SimpleNamespace(shots_dir=tmp_path, serial="serial-1")
    display = SimpleNamespace(show=lambda *a, **k: display_calls.append((a, k)))

    monkeypatch.setattr(engine, "shot_stamp", lambda: "STAMP")
    monkeypatch.setattr(engine, "adb_screencap", lambda path, serial: shot_calls.append((path, serial)))
    monkeypatch.setattr(engine.cv2, "imread", lambda *_a, **_k: image)

    shot, captured = engine._capture_analysis_frame(
        args,
        display,
        round_no=2,
        observation_attempt=1,
        max_observation_attempts=3,
    )
    assert shot == tmp_path / "analysis_STAMP.png"
    assert shot_calls == [(shot, "serial-1")]
    assert captured is image
    assert len(display_calls) == 1


def test_strict_observation_gate_remains_engine_rule() -> None:
    result = SimpleNamespace(
        observation_health=ObservationHealth(
            trusted=False,
            reasons=["current-frame-test-conflict"],
        )
    )
    trusted, reasons = engine._observation_trust(result)
    assert not trusted
    assert reasons == ("current-frame-test-conflict",)


def test_strategy_keeps_rule_logic_not_debug_logging() -> None:
    root = Path(__file__).resolve().parents[2]
    strategy_text = (root / "game_solver" / "strategy.py").read_text(encoding="utf-8")
    assert "def simulate_flow_closure(" in strategy_text
    assert "def evaluate_candidates(" in strategy_text
    assert "def choose_two_step_plan(" in strategy_text
    assert "COLOR_SNAPSHOT" not in strategy_text
    assert "NUMBER_SNAPSHOT" not in strategy_text
    assert "DECISION]" not in strategy_text
