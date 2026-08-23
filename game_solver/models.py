from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Car:
    source: str              # front / next / parked
    column: Optional[int]
    color: Optional[int]     # 1..N（N 动态）；None=识别失败
    remain: Optional[int]    # 车上剩余数字；next 可为 None
    x: float
    y: float
    scale_hint: float = 1.0


@dataclass
class Candidate:
    column: int
    color: int
    capacity: int
    reachable: int
    self_clear_guaranteed: bool
    some_completion_guaranteed: bool
    guaranteed_completions: int
    chain_parked_completions: int
    chain_parked_completion_by_color: Dict[int, int]
    deterministic_clear_reachable: bool
    next_color_newly_reachable: int
    useful_newly_reachable: int
    unlocked_by_color: Dict[int, int]
    rejected: bool
    reject_reason: str
    score: float
    next_color: Optional[int]
    next_capacity: Optional[int]
    queue_unlock_bonus: float
    next_vehicle_score: float
    next_vehicle_chain_parked_completions: int
    next_vehicle_exact: bool
    next_match_contacts: int
    neighbor_contacts: Dict[int, int]

    # v5.3：自动分流闭包的稳定状态摘要。
    flow_cleared_cells: int = 0
    flow_final_occupied_upper: int = 0
    flow_exact: bool = True
    flow_rounds: int = 0


@dataclass
class TwoStepPlan:
    first: Candidate
    second: Candidate
    second_source: str          # front / next
    score: float
    free_slots_before: int
    first_simulated_exactly: bool
    reason: str

    # v5.3：A+B 作为联合动作模拟后的稳定状态。
    guaranteed_completions: int = 0
    guaranteed_parked_completions: int = 0
    cleared_cells: int = 0
    final_occupied_upper: int = 0
    flow_exact: bool = True


@dataclass
class ObservationHealth:
    """Current stable-frame trust result. No multi-round recovery state lives here."""

    trusted: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unknown_cells: int = 0
    transition_conflicts: List[Tuple[int, int, int, int]] = field(default_factory=list)
    capacity_remaining_by_color: Dict[int, int] = field(default_factory=dict)
    capacity_excess_by_color: Dict[int, int] = field(default_factory=dict)

    def add_reason(self, reason: str) -> None:
        """Mark this observation untrusted for one explicit current-frame reason."""
        value = str(reason).strip()
        if not value:
            return
        if value not in self.reasons:
            self.reasons.append(value)
        self.trusted = False

    def add_warning(self, warning: str) -> None:
        """Attach diagnostic context without changing current-frame trust."""
        value = str(warning).strip()
        if value and value not in self.warnings:
            self.warnings.append(value)


@dataclass
class AnalysisResult:
    report: str
    palette: np.ndarray
    grid: np.ndarray
    turn: int
    best: Optional[Candidate]
    image_w: int
    image_h: int
    front: List[Car]
    nxt: List[Car]
    parked: List[Car]
    parking_empty_ref: np.ndarray
    occupied_slots: int
    new_colors_added: int
    front_ocr_reads: int
    two_step_plan: Optional[TwoStepPlan]

    # Issue 006: the sole execution trust signal for this current observation.
    observation_health: ObservationHealth

    state_saved: bool = True

    # Current stable screenshot's 52x38x3 cell RGB snapshot. Retry attempts are
    # read-only; only the selected committed observation becomes history.
    grid_rgb_snapshot: Optional[np.ndarray] = None
