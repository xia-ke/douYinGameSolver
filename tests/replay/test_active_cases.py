from __future__ import annotations

from . import runners  # noqa: F401  # register active layer runners by import side effect
from .harness import run_active_case
from .manifest import cases_for


def test_all_active_replay_cases_execute_structured_assertions() -> None:
    """
    Pending cases are intentionally not executed as pixel regressions.

    Once a case is promoted to active, this test requires a registered layer runner;
    an active case with no executable runner fails instead of silently passing.
    """
    for case in cases_for(status="active"):
        run_active_case(case)
