from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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
    deterministic_clear_reachable: bool
    next_color_newly_reachable: int
    useful_newly_reachable: int
    unlocked_by_color: Dict[int, int]
    rejected: bool
    reject_reason: str
    score: float
    next_color: Optional[int]
    next_match_contacts: int
    neighbor_contacts: Dict[int, int]


@dataclass
class TwoStepPlan:
    first: Candidate
    second: Candidate
    score: float
    free_slots_before: int
    first_simulated_exactly: bool
    reason: str


@dataclass
class FrontNumberCacheEntry:
    value: Optional[int]
    fingerprint: np.ndarray


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
    front_number_cache: Dict[int, FrontNumberCacheEntry]
    front_ocr_reads: int
    two_step_plan: Optional[TwoStepPlan]