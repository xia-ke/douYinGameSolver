from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from game_solver import engine


class _FakeDisplay:
    def show(self, *_args, **_kwargs) -> None:
        return None


class _StopAfterGate(RuntimeError):
    pass


class _TapObserved(RuntimeError):
    pass


def _analysis(*, status: str, causal_invalid: str = "") -> SimpleNamespace:
    car = SimpleNamespace(column=1, color=1, remain=3, x=120.0, y=1800.0)
    candidate = SimpleNamespace(
        column=1,
        color=1,
        capacity=3,
        flow_final_occupied_upper=1,
    )
    return SimpleNamespace(
        board_update_status=status,
        causal_input_invalid=causal_invalid,
        board_update_remaining_by_color={1: 1} if status == "incomplete" else {},
        board_update_excess_by_color={},
        grid=np.ones((2, 2), dtype=np.int16),
        report="diagnostic observation",
        front_number_cache={1: object()},
        front_ocr_reads=1,
        guarantee_broken=False,
        model_conflict_colors=set(),
        front=[car],
        nxt=[],
        occupied_slots=0,
        best=candidate,
        two_step_plan=None,
        palette=np.asarray([[200.0, 100.0, 50.0]], dtype=np.float32),
    )


def _args(tmp_path: Path, *, experimental_continue: bool) -> SimpleNamespace:
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    state = tmp_path / "solver_state.npz"
    state.write_bytes(b"trusted-state")
    return SimpleNamespace(
        reset=False,
        serial=None,
        state=state,
        shots_dir=shots_dir,
        decision_log=None,
        color_log=None,
        number_log=None,
        slots=6,
        skip_sixth_slot_unlock=True,
        unlock_ad_wait=0.0,
        unlock_return_settle_delay=0.0,
        flow_start_delay=0.0,
        parking_check_interval=1.0,
        parking_idle_timeout=1.0,
        observation_retries=1,
        observation_retry_delay=0.01,
        safe_pause_retry_delay=0.01,
        experimental_continue=experimental_continue,
        no_auto_tap=False,
        tap_delay=0.0,
        double_step_gap=0.0,
        queue_promote_timeout=0.1,
        queue_promote_poll_interval=0.01,
        queue_empty_confirm_delay=0.0,
        analysis_settle_delay=0.0,
    )


def _patch_auto_dependencies(monkeypatch, result, execution_log, commits, taps):
    monkeypatch.setattr(engine, "_append_session_marker", lambda *_a, **_k: None)
    monkeypatch.setattr(engine, "_append_decision_log", lambda *_a, **_k: None)
    monkeypatch.setattr(
        engine, "_append_observation_table_logs", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        engine,
        "_append_execution_update",
        lambda *_a, **k: execution_log.append(k.get("execution", "")),
    )
    monkeypatch.setattr(engine, "adb_screencap", lambda *_a, **_k: None)
    monkeypatch.setattr(engine.cv2, "imread", lambda *_a, **_k: None)
    monkeypatch.setattr(engine, "analyze_image", lambda *_a, **_k: result)

    def fake_commit(path, committed_result):
        commits.append(committed_result)
        path.write_bytes(b"advanced-state")

    monkeypatch.setattr(engine, "_commit_analysis_result_state", fake_commit)

    def fake_tap(*_a, **_k):
        taps.append(True)
        raise _TapObserved("tap reached")

    monkeypatch.setattr(engine, "tap_candidate_from_result", fake_tap)
    monkeypatch.setattr(engine, "adb_tap", lambda *_a, **_k: taps.append(True))


@pytest.mark.parametrize(
    ("status", "causal_invalid"),
    [
        ("incomplete", ""),
        ("causal_invalid", "capacity conservation invalid"),
        ("ok", "capacity conservation invalid"),
    ],
)
def test_strict_default_rejects_untrusted_without_commit_or_tap(
    tmp_path, monkeypatch, status, causal_invalid
):
    result = _analysis(status=status, causal_invalid=causal_invalid)
    args = _args(tmp_path, experimental_continue=False)
    execution_log = []
    commits = []
    taps = []
    _patch_auto_dependencies(monkeypatch, result, execution_log, commits, taps)

    monkeypatch.setattr(
        engine.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(_StopAfterGate()),
    )

    with pytest.raises(_StopAfterGate):
        engine._run_auto_flow_mode_impl(args, _FakeDisplay())

    assert args.state.read_bytes() == b"trusted-state"
    assert commits == []
    assert taps == []
    assert any("NO_CLICK_UNTRUSTED" in item for item in execution_log)
    assert "[NO_CLICK_UNTRUSTED]" in result.report


def test_experimental_continue_is_explicit_opt_in(tmp_path, monkeypatch):
    result = _analysis(status="incomplete")
    args = _args(tmp_path, experimental_continue=True)
    execution_log = []
    commits = []
    taps = []
    _patch_auto_dependencies(monkeypatch, result, execution_log, commits, taps)
    monkeypatch.setattr(engine.time, "sleep", lambda _seconds: None)

    with pytest.raises(_TapObserved):
        engine._run_auto_flow_mode_impl(args, _FakeDisplay())

    assert commits == [result]
    assert args.state.read_bytes() == b"advanced-state"
    assert taps == [True]
    assert "[EXPERIMENT_CONTINUE_UNTRUSTED]" in result.report
    assert any("EXPERIMENT_CONTINUE_UNTRUSTED" in item for item in execution_log)
