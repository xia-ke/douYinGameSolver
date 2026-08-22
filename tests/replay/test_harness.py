from __future__ import annotations

import pytest

from .harness import assert_expected_subset


def test_structured_assertion_accepts_nested_expected_subset() -> None:
    expected = {"board": {"removed": 75}, "parking": {"remain": 5}}
    actual = {
        "board": {"removed": 75, "extra_debug": [1, 2, 3]},
        "parking": {"remain": 5},
        "diagnostics": {"trusted": True},
    }
    assert_expected_subset(expected, actual)


def test_structured_assertion_reports_the_failing_path() -> None:
    with pytest.raises(AssertionError, match=r"case\.board\.removed"):
        assert_expected_subset(
            {"board": {"removed": 75}},
            {"board": {"removed": 47}},
            path="case",
        )
