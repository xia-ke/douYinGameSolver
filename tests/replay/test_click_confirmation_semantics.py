from __future__ import annotations

from types import SimpleNamespace

from game_solver import engine


def _result(next_color, next_remain):
    return SimpleNamespace(
        nxt=[
            SimpleNamespace(
                column=1,
                color=next_color,
                remain=next_remain,
            )
        ]
    )


def _candidate():
    return SimpleNamespace(
        column=1,
        color=6,
        capacity=32,
    )


def test_same_color_unknown_next_number_is_not_distinguishable():
    # Real Level 16 failure: the clicked C06x32 departed, then another C06x32
    # promoted from a next row whose pale number had been UNKNOWN.
    assert not engine._has_distinguishable_known_next(
        _result(6, None),
        _candidate(),
    )


def test_known_color_difference_is_distinguishable_even_when_number_unknown():
    assert engine._has_distinguishable_known_next(
        _result(5, None),
        _candidate(),
    )


def test_known_number_difference_is_distinguishable_for_same_color():
    assert engine._has_distinguishable_known_next(
        _result(6, 39),
        _candidate(),
    )
