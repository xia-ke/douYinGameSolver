from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_readme_describes_current_mission_first_system() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "ObservationHealth",
        "NO_CLICK_UNTRUSTED",
        "simulate_flow_closure()",
        "6/6",
        "bottom-only reachability",
        "pending_fixture",
        "game_solver/debug.py",
        "pytest tests\\replay\\",
    ):
        assert required in text
    assert "v5.1 dynamic queue" not in text


def test_cli_help_is_strict_by_default_and_has_no_dead_transition_option() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "game_solver_v5.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    help_text = proc.stdout
    assert "trusted-only" in help_text
    assert "--experimental-continue" in help_text
    assert "默认关闭" in help_text
    assert "--transition-timeout" not in help_text
    assert "实验优先运行" not in help_text


def test_retired_production_architecture_symbols_are_absent() -> None:
    forbidden = (
        "game" + "_ocr",
        "install_game_digit" + "_ocr",
        "_Safety" + "Runtime",
        "PAUSED_" + "SYNC",
        "PAUSED_" + "SAFE",
        "update_grid_" + "causal",
        "CausalBoard" + "Update",
        "load_state_with_" + "grid_rgb",
        "simulate_clear_current_" + "reachable_color",
        "allow_deterministic_" + "unlock",
        "FrontNumberCache" + "Entry",
        "front_number_" + "cache",
        "board_update_" + "status",
        "causal_input_" + "invalid",
        "strategy_untrusted_" + "colors",
        "force_single_" + "step",
        "disable_parked_" + "chain",
        "safe_pause_retry_" + "delay",
    )
    paths = [ROOT / "README.md", ROOT / "game_solver_v5.py"]
    paths.extend(sorted((ROOT / "game_solver").glob("*.py")))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for symbol in forbidden:
        assert symbol not in combined, symbol


def test_final_production_docs_do_not_present_patch_history_as_architecture() -> None:
    paths = [ROOT / "README.md"]
    paths.extend(sorted((ROOT / "game_solver").glob("*.py")))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "Issue 004 已切换" not in combined
    assert "Issue 006 trust policy" not in combined
    assert "__version__ = \"5.1-dynamic-queue\"" not in combined
