from __future__ import annotations

from game_solver.strategy import guaranteed_completions_from_supply


def test_smaller_remain_same_color_car_completes_first() -> None:
    # With supply=2, the confirmed low-remain rule completes the remain=2 car.
    # The old arbitrary-distribution lower bound returned zero for this case.
    assert guaranteed_completions_from_supply([5, 2], supply=2) == 1
    assert guaranteed_completions_from_supply([2, 5], supply=2) == 1


def test_low_remain_rule_consumes_supply_in_sorted_completion_order() -> None:
    assert guaranteed_completions_from_supply([5, 2], supply=6) == 1
    assert guaranteed_completions_from_supply([5, 2], supply=7) == 2
