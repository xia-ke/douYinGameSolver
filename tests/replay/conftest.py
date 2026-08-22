from __future__ import annotations

from .manifest import load_manifest


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # type: ignore[no-untyped-def]
    pending = [case for case in load_manifest() if case.is_pending]
    terminalreporter.section("replay registry")
    terminalreporter.write_line(f"pending_fixture: {len(pending)}")
    for case in pending:
        terminalreporter.write_line(
            f"  - {case.case_id} [{case.subsystem}]: {case.pending_reason}"
        )
