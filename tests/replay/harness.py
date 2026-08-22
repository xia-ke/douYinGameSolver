from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .manifest import REPO_ROOT, ReplayCase

ReplayRunner = Callable[[ReplayCase, Path], Mapping[str, Any]]
_RUNNERS: dict[str, ReplayRunner] = {}


def register_runner(name: str) -> Callable[[ReplayRunner], ReplayRunner]:
    def decorator(func: ReplayRunner) -> ReplayRunner:
        if name in _RUNNERS:
            raise RuntimeError(f"duplicate replay runner: {name}")
        _RUNNERS[name] = func
        return func

    return decorator


def assert_expected_subset(expected: Any, actual: Any, path: str = "expected") -> None:
    """Recursively require every expected value to exist and match in actual output."""
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            raise AssertionError(f"{path}: expected mapping, got {type(actual).__name__}")
        for key, expected_value in expected.items():
            if key not in actual:
                raise AssertionError(f"{path}.{key}: missing from replay result")
            assert_expected_subset(expected_value, actual[key], f"{path}.{key}")
        return

    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)):
            raise AssertionError(f"{path}: expected sequence, got {type(actual).__name__}")
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path}: expected sequence length {len(expected)}, got {len(actual)}"
            )
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            assert_expected_subset(expected_value, actual_value, f"{path}[{index}]")
        return

    if actual != expected:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")


def run_active_case(case: ReplayCase) -> None:
    if not case.is_active:
        raise AssertionError(f"{case.case_id}: only active cases may execute")
    if not case.runner:
        raise AssertionError(f"{case.case_id}: active case has no runner")
    runner = _RUNNERS.get(case.runner)
    if runner is None:
        raise AssertionError(
            f"{case.case_id}: active runner {case.runner!r} is not registered; "
            "promotion to active must include an executable layer runner"
        )
    actual = runner(case, REPO_ROOT)
    if not isinstance(actual, Mapping):
        raise AssertionError(f"{case.case_id}: runner must return structured mapping output")
    assert_expected_subset(case.expected, actual, path=case.case_id)
