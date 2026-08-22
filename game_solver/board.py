from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import (
    REF_W, REF_H, GRID_ROWS, GRID_COLS,
    X_EDGE0_REF, X_STEP_REF, Y_EDGE0_REF, Y_STEP_REF,
    PALETTE_DBSCAN_EPS, PALETTE_DBSCAN_MIN_SAMPLES,
    PALETTE_MIN_CLUSTER_SIZE, KEEP_COLOR_DIST, INITIAL_COLOR_DIST,
    UNKNOWN, EMPTY,
)


# ------------------ 增量棋盘更新 ------------------
#
# 游戏中的棋盘色块具有很强的物理约束：
#   1) 色块不会移动；
#   2) 色块不会变色；
#   3) 一个已知色块只有在“从棋盘下方开放区可连通到它”时才可能消失；
#   4) UNKNOWN 不能直接推断成 EMPTY。
#
# 因此后续帧不再全棋盘重识别颜色，而只检查当前开放边界上的格子。
# 当某个边界格被截图明确确认成棋盘背景后，再把它置 EMPTY，
# 开放区随之扩大，继续检查新暴露的边界。
#
# 这同时解决：
#   - 微信/系统通知遮挡导致大量已知色块被误写 EMPTY；
#   - 单帧颜色、高光、动画污染导致永久状态损坏；
#   - 每轮重复扫描 52x38 全棋盘的无谓开销。
_INCREMENTAL_UNKNOWN_RECOVER_DIST = 30.0
_INCREMENTAL_EMPTY_BG_DIST = 34.0
_INCREMENTAL_EMPTY_PALETTE_MARGIN = 44.0
_INCREMENTAL_BG_SAMPLE_LIMIT = 256

# v5.9 temporal causal sync
#
# 容量守恒已经告诉我们“本轮某颜色实际消失多少格”，因此稳定截图中
# 不需要每个已吸收格都必须露出纯灰背景；只要：
#   - 该格属于上一 committed state 的当前 reachable 同色连通块；
#   - 与上一 committed 稳定截图相比发生了明显视觉变化；
# 就可以作为因果删除证据。
_CAUSAL_TEMPORAL_CHANGE_DIST = 18.0
_CAUSAL_TEMPORAL_STRONG_CHANGE_DIST = 32.0

# v5.9.2: local sprite-body coverage fallback.
# The exact 2x2 center may remain painted by a neighboring sprite even
# after this logical block is gone. Compare a small in-cell patch against
# the previous committed analysis frame.
_CAUSAL_PATCH_COLOR_DIST = 44.0
_CAUSAL_PATCH_COVERAGE_DROP = 0.18
_CAUSAL_PATCH_MIN_PREV_COVERAGE = 0.55
_CAUSAL_PATCH_HALF_X_FRAC = 0.38
_CAUSAL_PATCH_HALF_Y_FRAC = 0.38

# v5.10: strong non-frontier fallback for unresolved mathematical budget.
_CAUSAL_GLOBAL_STRONG_PREV_COVERAGE = 0.75
_CAUSAL_GLOBAL_STRONG_CURR_COVERAGE_MAX = 0.20
_CAUSAL_GLOBAL_STRONG_COVERAGE_DROP = 0.55
_CAUSAL_GLOBAL_STRONG_CENTER_CHANGE = 32.0

# v5.11: UI occlusion is a confidence penalty, not a permanent hard veto.
_UI_VISIBLE_CENTER_DIST = 30.0
_UI_VISIBLE_PATCH_COVERAGE = 0.75
_UI_UNKNOWN_PREV_COVERAGE = 0.60
_UI_UNKNOWN_COLOR_MARGIN = 0.18
_UI_UNKNOWN_CURR_COVERAGE_MAX = 0.20
_UI_UNKNOWN_COVERAGE_DROP = 0.50
_UI_UNKNOWN_CENTER_CHANGE = 32.0

# v5.14: perception-first board recognition.
#
# A logical cell is ~21x18 px at the reference resolution.  Center 2x2 RGB
# is too fragile when a disappearing block is covered by a car body / foot /
# shadow, and absolute distance-to-one-palette-color is ambiguous for close
# palette pairs.  The v5.14 classifier therefore:
#   - uses an inner cell patch;
#   - labels pixels by nearest palette color with a separation margin;
#   - compares exactly the pixels that belonged to the old color;
#   - measures loss / temporal change / transition-to-background per pixel.
_RECOG_PATCH_HALF_X_FRAC = 0.42
_RECOG_PATCH_HALF_Y_FRAC = 0.42
_RECOG_PIXEL_COLOR_MAX_DIST = 54.0
_RECOG_PIXEL_COLOR_HARD_DIST = 28.0
_RECOG_PIXEL_COLOR_MIN_MARGIN = 7.0
_RECOG_CELL_MIN_COLOR_PIXELS = 5
_RECOG_CELL_MIN_COLOR_COVERAGE = 0.10
_RECOG_CELL_MIN_WIN_MARGIN = 0.035

_RECOG_PIXEL_BG_DIST = 40.0
_RECOG_PIXEL_BG_PALETTE_MARGIN = 36.0

_RECOG_TEMPORAL_CHANGE_DIST = 22.0
_RECOG_TEMPORAL_STRONG_CHANGE_DIST = 30.0
_RECOG_TEMPORAL_STABLE_DIST = 12.0

_RECOG_DISAPPEAR_MIN_PREV_PIXELS = 5
_RECOG_DISAPPEAR_LOSS_RATIO = 0.55
_RECOG_DISAPPEAR_CHANGE_RATIO = 0.46
_RECOG_DISAPPEAR_STABLE_MAX = 0.38
_RECOG_DISAPPEAR_BG_RATIO = 0.10
_RECOG_DISAPPEAR_CURR_COVERAGE_MAX = 0.34

_RECOG_VERY_STRONG_LOSS_RATIO = 0.70
_RECOG_VERY_STRONG_CHANGE_RATIO = 0.62
_RECOG_VERY_STRONG_STABLE_MAX = 0.22
_RECOG_VERY_STRONG_CURR_COVERAGE_MAX = 0.24

# v5.16: visual truth is authoritative for spatial state.
#
# Background is a real visual class.  Do NOT reject a true gray background
# merely because it happens to be numerically close to one palette center.
# Compare which model wins: background vs nearest palette color.
_RECOG_BACKGROUND_MAX_DIST = 42.0
_RECOG_BACKGROUND_MIN_ADVANTAGE = 10.0

# A COLOR -> EMPTY commit must be supported by the current stable frame itself.
# Capacity conservation is an audit signal only; it never chooses which cells
# disappear and never truncates visual candidates to a top-N set.
_RECOG_PRESENT_MIN_OLD_COLOR_COVERAGE = 0.18
_RECOG_EMPTY_MIN_BG_COVERAGE = 0.60
_RECOG_EMPTY_MIN_BG_ZONES = 5
_RECOG_EMPTY_ZONE_COVERAGE = 0.55
_RECOG_EMPTY_MAX_OLD_COLOR_COVERAGE = 0.12

# Fixed UI areas are allowed to remain UNCERTAIN.  They require much stronger
# direct background evidence before a persistent color is cleared.
_RECOG_UI_EMPTY_MIN_BG_COVERAGE = 0.90
_RECOG_UI_EMPTY_MIN_BG_ZONES = 8
_RECOG_UI_EMPTY_MAX_OLD_COLOR_COVERAGE = 0.03

# 当已存在 EMPTY 很少时，使用棋盘下缘和停车区之间的灰色游戏背景估计背景色。
# 这些比例针对整个截图而不是棋盘网格；只作为首轮/极少 EMPTY 时的兜底。
_INCREMENTAL_BG_FALLBACK_X1_N = 0.22
_INCREMENTAL_BG_FALLBACK_X2_N = 0.78
_INCREMENTAL_BG_FALLBACK_Y1_N = 0.515
_INCREMENTAL_BG_FALLBACK_Y2_N = 0.545


@dataclass
class CausalBoardUpdate:
    """
    一轮真实分流结束后的因果棋盘更新结果。

    expected_by_color:
        根据“上一稳定状态的停车剩余 + 本轮实际点击车辆容量 - 下一稳定状态停车剩余”
        计算出的、本轮每种颜色确定实际吸收数量。

    confirmed_by_color:
        在拓扑允许的位置上，由当前稳定截图明确落实为 EMPTY 的数量。

    remaining_by_color:
        数学上应该已经消失、但视觉/拓扑尚未能落实的位置数量。
        非空时进入观测重试；持续无法落实时由运行时安全状态机降级处理。

    excess_by_color:
        当前截图在 reachable 同色连通块中显示为 EMPTY 的格数量超过实际吸收预算。
        这通常意味着旧 grid 颜色分类、OCR 容量或当前截图存在矛盾；该颜色应被隔离，不能猜位置。
    """
    grid: np.ndarray
    removed: int
    expected_by_color: Dict[int, int]
    confirmed_by_color: Dict[int, int]
    remaining_by_color: Dict[int, int]
    excess_by_color: Dict[int, int]
    invalid_reason: str
    checked_cells: int

    # v5.9 诊断：同一 committed stable frame -> current stable frame 的时间差分。
    temporal_confirmed_by_color: Dict[int, int]
    background_confirmed_by_color: Dict[int, int]
    ambiguous_changed_by_color: Dict[int, int]
    temporal_change_threshold: float
    temporal_snapshot_available: bool
    patch_confirmed_by_color: Dict[int, int]
    patch_coverage_drop_threshold: float
    patch_previous_frame_available: bool
    strong_nonfrontier_confirmed_by_color: Dict[int, int]
    ui_unknown_confirmed_by_color: Dict[int, int]

    @property
    def complete(self) -> bool:
        return (
            not self.invalid_reason
            and not self.remaining_by_color
            and not self.excess_by_color
        )


def ctag(color: Optional[int]) -> str:
    if color is None or color <= 0:
        return "UNKNOWN"
    return f"C{color:02d}"


def scaled_grid_geometry(w: int, h: int) -> Tuple[float, float, float, float]:
    sx = w / REF_W
    sy = h / REF_H
    return (
        X_EDGE0_REF * sx,
        X_STEP_REF * sx,
        Y_EDGE0_REF * sy,
        Y_STEP_REF * sy,
    )


def sample_center_2x2(arr: np.ndarray, x: float, y: float) -> np.ndarray:
    h, w = arr.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, min(w - 2, cx - 1))
    y0 = max(0, min(h - 2, cy - 1))
    patch = arr[y0:y0 + 2, x0:x0 + 2, :3]
    return patch.reshape(-1, 3).mean(axis=0)


def _grid_cell_center(
    r: int,
    c: int,
    w: int,
    h: int,
) -> Tuple[float, float]:
    x0, dx, y0, dy = scaled_grid_geometry(w, h)
    return (
        x0 + (c + 0.5) * dx,
        y0 + (r + 0.5) * dy,
    )


def _sample_grid_cell(
    image_rgb: np.ndarray,
    r: int,
    c: int,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x, y = _grid_cell_center(r, c, w, h)
    return sample_center_2x2(image_rgb, x, y).astype(np.float32)



def sample_grid_rgb_snapshot(image_rgb: np.ndarray) -> np.ndarray:
    """
    一次性采样 52x38 个逻辑格子的中心 RGB。

    该快照跟随“已提交稳定状态”持久化。下一次动作完成后用 current-prev
    时间差分判断哪些物理可达格确实发生了变化，从而避免把
    “必须露出纯灰背景”当成唯一 COLOR->EMPTY 证据。
    """
    snap = np.empty((GRID_ROWS, GRID_COLS, 3), dtype=np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            snap[r, c] = _sample_grid_cell(image_rgb, r, c)
    return snap



@dataclass(frozen=True)
class _CellVisualEvidence:
    """Pixel-level evidence that one old logical color cell disappeared."""

    score: float
    strong: bool
    very_strong: bool
    is_background: bool
    center_change: float
    prev_color_coverage: float
    curr_color_coverage: float
    loss_ratio: float
    change_ratio: float
    strong_change_ratio: float
    stable_ratio: float
    background_ratio: float
    zone_support: int


def _sample_grid_cell_inner_patch(
    image_rgb: np.ndarray,
    r: int,
    c: int,
) -> np.ndarray:
    """Return a robust inner patch without crossing into adjacent logical cells."""
    h, w = image_rgb.shape[:2]
    x, y = _grid_cell_center(r, c, w, h)
    _x0, dx, _y0, dy = scaled_grid_geometry(w, h)

    rx = max(2, int(round(dx * _RECOG_PATCH_HALF_X_FRAC)))
    ry = max(2, int(round(dy * _RECOG_PATCH_HALF_Y_FRAC)))
    cx = int(round(x))
    cy = int(round(y))

    x1 = max(0, cx - rx)
    x2 = min(w, cx + rx + 1)
    y1 = max(0, cy - ry)
    y2 = min(h, cy + ry + 1)
    return image_rgb[y1:y2, x1:x2, :3].astype(np.float32)


def _palette_pixel_labels(
    patch_rgb: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """
    Classify every pixel exclusively to at most one palette color.

    This is intentionally not ``distance(target) < threshold``.  For close
    colors (for example two palette centers only ~58 RGB units apart), a pixel
    may satisfy the absolute threshold for both colors.  Nearest-color +
    separation margin prevents the other color from being counted as evidence
    that the old color is still present.
    """
    if patch_rgb.size == 0:
        return np.empty(patch_rgb.shape[:2], dtype=np.int16)
    if len(palette) == 0:
        return np.full(patch_rgb.shape[:2], UNKNOWN, dtype=np.int16)

    flat = patch_rgb.reshape(-1, 3).astype(np.float32)
    pal = np.asarray(palette, dtype=np.float32)
    dist = np.linalg.norm(
        flat[:, None, :] - pal[None, :, :],
        axis=2,
    )

    order = np.argsort(dist, axis=1)
    best_idx = order[:, 0]
    best_dist = dist[np.arange(len(flat)), best_idx]
    if len(pal) >= 2:
        second_dist = dist[np.arange(len(flat)), order[:, 1]]
    else:
        second_dist = np.full(len(flat), np.inf, dtype=np.float32)

    hard_match = best_dist <= _RECOG_PIXEL_COLOR_HARD_DIST
    separated_match = (
        (best_dist <= _RECOG_PIXEL_COLOR_MAX_DIST)
        & ((second_dist - best_dist) >= _RECOG_PIXEL_COLOR_MIN_MARGIN)
    )
    accepted = hard_match | separated_match

    labels = np.full(len(flat), UNKNOWN, dtype=np.int16)
    labels[accepted] = best_idx[accepted].astype(np.int16) + 1
    return labels.reshape(patch_rgb.shape[:2])


def _classify_grid_cell_color_patch(
    image_rgb: np.ndarray,
    r: int,
    c: int,
    palette: np.ndarray,
) -> Tuple[Optional[int], float, float]:
    """
    Robust logical-cell color vote.

    Returns ``(color, winner_coverage, winner_minus_runner_up)``.
    Ambiguous/shadow/background pixels simply abstain instead of voting for the
    nearest color.
    """
    patch = _sample_grid_cell_inner_patch(image_rgb, r, c)
    labels = _palette_pixel_labels(patch, palette)
    if labels.size == 0 or len(palette) == 0:
        return None, 0.0, 0.0

    counts = np.asarray(
        [int(np.count_nonzero(labels == color)) for color in range(1, len(palette) + 1)],
        dtype=np.int32,
    )
    winner_idx = int(np.argmax(counts))
    winner_count = int(counts[winner_idx])
    if winner_count < _RECOG_CELL_MIN_COLOR_PIXELS:
        return None, 0.0, 0.0

    coverage = float(winner_count / labels.size)
    if len(counts) >= 2:
        runner_up = int(np.partition(counts, -2)[-2])
    else:
        runner_up = 0
    margin = float((winner_count - runner_up) / labels.size)

    if coverage < _RECOG_CELL_MIN_COLOR_COVERAGE:
        return None, coverage, margin
    if margin < _RECOG_CELL_MIN_WIN_MARGIN:
        return None, coverage, margin
    return winner_idx + 1, coverage, margin


def _pixel_background_mask(
    patch_rgb: np.ndarray,
    background_rgb: Optional[np.ndarray],
    palette: np.ndarray,
) -> np.ndarray:
    """
    Return pixels that are visually better explained by board background.

    v5.16 fixes an important false-negative:
    the real gray background can be only ~35 RGB units away from a palette
    center (for example C07 in level 15).  The old absolute rule
    ``nearest_palette >= 36`` therefore classified exact gray background as a
    palette color.

    The new rule is relative:
        - pixel must be close to the learned board background; and
        - background must beat the nearest palette center by a safe margin.
    """
    if patch_rgb.size == 0 or background_rgb is None:
        return np.zeros(patch_rgb.shape[:2], dtype=bool)

    patch = patch_rgb.astype(np.float32)
    bg_dist = np.linalg.norm(
        patch - background_rgb[None, None, :],
        axis=2,
    )
    mask = bg_dist <= _RECOG_BACKGROUND_MAX_DIST

    if len(palette) > 0:
        flat = patch.reshape(-1, 3)
        pal_dist = np.linalg.norm(
            flat[:, None, :] - np.asarray(palette, dtype=np.float32)[None, :, :],
            axis=2,
        )
        nearest_palette = pal_dist.min(axis=1).reshape(patch.shape[:2])
        mask &= (
            (nearest_palette - bg_dist)
            >= _RECOG_BACKGROUND_MIN_ADVANTAGE
        )
    return mask


def _cell_disappearance_evidence(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    current_grid_rgb: np.ndarray,
    previous_grid_rgb: Optional[np.ndarray],
    r: int,
    c: int,
    color: int,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
) -> _CellVisualEvidence:
    """
    Recognize disappearance by tracking old-color pixels, not patch averages.

    The decisive population is: pixels that were confidently this logical
    cell's color in the previous committed stable frame.  We ask what happened
    to those exact pixels in the current stable frame.  This remains useful
    when a sprite covers the cell, because "changed to another color/shadow"
    is valid disappearance evidence while an adjacent close palette color is
    no longer incorrectly counted as the old color.
    """
    prev_patch = _sample_grid_cell_inner_patch(previous_frame, r, c)
    curr_patch = _sample_grid_cell_inner_patch(current_frame, r, c)

    # Screenshots are expected to keep the same geometry.  Be defensive around
    # edge rounding so pixel-wise comparison never broadcasts accidentally.
    hh = min(prev_patch.shape[0], curr_patch.shape[0])
    ww = min(prev_patch.shape[1], curr_patch.shape[1])
    prev_patch = prev_patch[:hh, :ww]
    curr_patch = curr_patch[:hh, :ww]

    if hh == 0 or ww == 0 or not (0 < color <= len(palette)):
        return _CellVisualEvidence(
            0.0, False, False, False, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
        )

    prev_labels = _palette_pixel_labels(prev_patch, palette)
    curr_labels = _palette_pixel_labels(curr_patch, palette)
    prev_mask = prev_labels == int(color)

    # If shading made the exclusive vote sparse, recover only pixels for which
    # the requested old color is still the nearest palette center.  This does
    # not let another close color vote as the old color.
    if int(np.count_nonzero(prev_mask)) < _RECOG_DISAPPEAR_MIN_PREV_PIXELS:
        target = np.asarray(palette[color - 1], dtype=np.float32)
        flat = prev_patch.reshape(-1, 3)
        d_all = np.linalg.norm(
            flat[:, None, :] - np.asarray(palette, dtype=np.float32)[None, :, :],
            axis=2,
        )
        nearest = np.argmin(d_all, axis=1).reshape(prev_patch.shape[:2])
        target_dist = np.linalg.norm(
            prev_patch - target[None, None, :],
            axis=2,
        )
        prev_mask = (
            (nearest == color - 1)
            & (target_dist <= _CAUSAL_PATCH_COLOR_DIST)
        )

    prev_count = int(np.count_nonzero(prev_mask))
    total = int(prev_mask.size)
    prev_cov = float(prev_count / total) if total else 0.0
    curr_cov = float(np.mean(curr_labels == int(color))) if total else 0.0

    center_change = 0.0
    if previous_grid_rgb is not None:
        center_change = float(
            np.linalg.norm(current_grid_rgb[r, c] - previous_grid_rgb[r, c])
        )

    center_background = _looks_like_empty_background(
        current_grid_rgb[r, c],
        background_rgb,
        palette,
    )

    if prev_count < _RECOG_DISAPPEAR_MIN_PREV_PIXELS:
        return _CellVisualEvidence(
            float(3.0 if center_background else 0.0),
            bool(center_background),
            bool(center_background),
            bool(center_background),
            center_change,
            prev_cov,
            curr_cov,
            0.0, 0.0, 0.0, 0.0, 0.0, 0,
        )

    temporal = np.linalg.norm(curr_patch - prev_patch, axis=2)
    curr_same = curr_labels == int(color)
    lost = prev_mask & (~curr_same)
    changed = prev_mask & (temporal >= _RECOG_TEMPORAL_CHANGE_DIST)
    strong_changed = prev_mask & (temporal >= _RECOG_TEMPORAL_STRONG_CHANGE_DIST)
    stable = (
        prev_mask
        & curr_same
        & (temporal <= _RECOG_TEMPORAL_STABLE_DIST)
    )
    bg_mask = _pixel_background_mask(curr_patch, background_rgb, palette)
    to_background = prev_mask & bg_mask

    loss_ratio = float(np.count_nonzero(lost) / prev_count)
    change_ratio = float(np.count_nonzero(changed) / prev_count)
    strong_change_ratio = float(np.count_nonzero(strong_changed) / prev_count)
    stable_ratio = float(np.count_nonzero(stable) / prev_count)
    background_ratio = float(np.count_nonzero(to_background) / prev_count)

    # 3x3 local support prevents one tiny moving sprite fragment from deciding
    # the whole logical cell.  A real disappearance usually affects multiple
    # portions of the old block body.
    zone_support = 0
    for yi in range(3):
        y1 = (hh * yi) // 3
        y2 = (hh * (yi + 1)) // 3
        for xi in range(3):
            x1 = (ww * xi) // 3
            x2 = (ww * (xi + 1)) // 3
            zm = prev_mask[y1:y2, x1:x2]
            zn = int(np.count_nonzero(zm))
            if zn < 2:
                continue
            zl = float(np.count_nonzero(lost[y1:y2, x1:x2] & zm) / zn)
            zc = float(np.count_nonzero(changed[y1:y2, x1:x2] & zm) / zn)
            zb = float(np.count_nonzero(to_background[y1:y2, x1:x2] & zm) / zn)
            if (zl >= 0.50 and zc >= 0.40) or zb >= 0.25:
                zone_support += 1

    coverage_drop = max(0.0, prev_cov - curr_cov)
    score = (
        2.5 * loss_ratio
        + 2.1 * change_ratio
        + 1.2 * strong_change_ratio
        + 2.8 * background_ratio
        + 1.8 * coverage_drop
        + 0.22 * min(zone_support, 5)
        - 2.4 * stable_ratio
    )
    if center_background:
        score += 2.0

    strong = bool(
        center_background
        or (
            loss_ratio >= _RECOG_DISAPPEAR_LOSS_RATIO
            and change_ratio >= _RECOG_DISAPPEAR_CHANGE_RATIO
            and stable_ratio <= _RECOG_DISAPPEAR_STABLE_MAX
            and curr_cov <= _RECOG_DISAPPEAR_CURR_COVERAGE_MAX
            and (
                zone_support >= 2
                or background_ratio >= _RECOG_DISAPPEAR_BG_RATIO
                or strong_change_ratio >= 0.48
            )
        )
    )

    very_strong = bool(
        center_background
        or (
            loss_ratio >= _RECOG_VERY_STRONG_LOSS_RATIO
            and change_ratio >= _RECOG_VERY_STRONG_CHANGE_RATIO
            and stable_ratio <= _RECOG_VERY_STRONG_STABLE_MAX
            and curr_cov <= _RECOG_VERY_STRONG_CURR_COVERAGE_MAX
            and (
                zone_support >= 2
                or background_ratio >= 0.20
                or strong_change_ratio >= 0.68
            )
        )
    )

    return _CellVisualEvidence(
        float(score),
        strong,
        very_strong,
        bool(center_background),
        float(center_change),
        prev_cov,
        curr_cov,
        loss_ratio,
        change_ratio,
        strong_change_ratio,
        stable_ratio,
        background_ratio,
        int(zone_support),
    )


def _current_cell_visual_state(
    image_rgb: np.ndarray,
    r: int,
    c: int,
    old_color: int,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
    *,
    covered_by_fixed_ui: bool,
) -> Tuple[str, float, float, int]:
    """
    Classify one persistent COLOR cell from the current stable screenshot.

    Returns:
        ("PRESENT" | "EMPTY" | "UNCERTAIN",
         old_color_coverage,
         background_coverage,
         background_zone_support)

    Spatial state is decided only from visual evidence.  Capacity conservation
    is deliberately absent from this function.
    """
    if old_color <= 0 or old_color > len(palette):
        return "UNCERTAIN", 0.0, 0.0, 0

    patch = _sample_grid_cell_inner_patch(image_rgb, r, c)
    if patch.size == 0 or background_rgb is None:
        return "UNCERTAIN", 0.0, 0.0, 0

    labels = _palette_pixel_labels(patch, palette)
    background_mask = _pixel_background_mask(
        patch,
        background_rgb,
        palette,
    )

    # Background wins before palette voting.  This is critical for the level-15
    # gray background, which is numerically close to C07.
    labels = labels.copy()
    labels[background_mask] = UNKNOWN

    old_coverage = float(np.mean(labels == int(old_color)))
    background_coverage = float(np.mean(background_mask))

    hh, ww = background_mask.shape
    background_zones = 0
    for yi in range(3):
        y1 = (hh * yi) // 3
        y2 = (hh * (yi + 1)) // 3
        for xi in range(3):
            x1 = (ww * xi) // 3
            x2 = (ww * (xi + 1)) // 3
            zone = background_mask[y1:y2, x1:x2]
            if zone.size == 0:
                continue
            if float(np.mean(zone)) >= _RECOG_EMPTY_ZONE_COVERAGE:
                background_zones += 1

    if old_coverage >= _RECOG_PRESENT_MIN_OLD_COLOR_COVERAGE:
        return (
            "PRESENT",
            old_coverage,
            background_coverage,
            background_zones,
        )

    if covered_by_fixed_ui:
        center_rgb = _sample_grid_cell(image_rgb, r, c)
        center_is_background = _looks_like_empty_background(
            center_rgb,
            background_rgb,
            palette,
        )
        if (
            center_is_background
            and background_coverage >= _RECOG_UI_EMPTY_MIN_BG_COVERAGE
            and background_zones >= _RECOG_UI_EMPTY_MIN_BG_ZONES
            and old_coverage <= _RECOG_UI_EMPTY_MAX_OLD_COLOR_COVERAGE
        ):
            return (
                "EMPTY",
                old_coverage,
                background_coverage,
                background_zones,
            )
        return (
            "UNCERTAIN",
            old_coverage,
            background_coverage,
            background_zones,
        )

    if (
        background_coverage >= _RECOG_EMPTY_MIN_BG_COVERAGE
        and background_zones >= _RECOG_EMPTY_MIN_BG_ZONES
        and old_coverage <= _RECOG_EMPTY_MAX_OLD_COLOR_COVERAGE
    ):
        return (
            "EMPTY",
            old_coverage,
            background_coverage,
            background_zones,
        )

    return (
        "UNCERTAIN",
        old_coverage,
        background_coverage,
        background_zones,
    )


def _sample_grid_cell_color_coverage(
    image_rgb: np.ndarray,
    r: int,
    c: int,
    color: int,
    palette: np.ndarray,
) -> float:
    """
    Return exclusive pixel coverage for one palette color in a logical cell.

    v5.14: pixels are assigned to the nearest sufficiently separated palette
    color.  This fixes the old failure where a nearby palette color could also
    satisfy ``distance(target) < 44`` and be counted as survival of ``color``.
    """
    if color <= 0 or color > len(palette):
        return -1.0

    patch = _sample_grid_cell_inner_patch(image_rgb, r, c)
    if patch.size == 0:
        return -1.0
    labels = _palette_pixel_labels(patch, palette)
    return float(np.mean(labels == int(color)))

def ui_covered(x: float, y: float, w: int, h: int) -> bool:
    """顶部固定 UI 遮挡。坐标先映射回 940x2048 再判断。"""
    xr = x * REF_W / w
    yr = y * REF_H / h
    return yr < 165 and (
        xr < 125
        or 330 < xr < 560
        or 610 < xr < 790
        or xr > 820
    )


def grid_samples(image_rgb: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    完整采样只用于：
      - 新局建立 palette；
      - 新局 initial_grid。

    后续 update_grid() 已不再调用本函数扫描整张棋盘。
    """
    h, w = image_rgb.shape[:2]
    x0, dx, y0, dy = scaled_grid_geometry(w, h)
    samples: List[np.ndarray] = []
    positions: List[Tuple[int, int]] = []
    for r in range(GRID_ROWS):
        cy = y0 + (r + 0.5) * dy
        for c in range(GRID_COLS):
            cx = x0 + (c + 0.5) * dx
            samples.append(sample_center_2x2(image_rgb, cx, cy))
            positions.append((r, c))
    return np.asarray(samples, dtype=np.float32), positions


def learn_palette(image_rgb: np.ndarray) -> np.ndarray:
    """
    从棋盘自动学习本关实际存在的颜色类别。

    关键点：
    - 不预设任何固定颜色种数；DBSCAN 得到多少个稳定主色就保留多少个。
    - 顶部 UI 覆盖较多，因此只用第 5..51 行建立调色板。
    - 同色小块通常至少 4~5 个相连，因此允许较小但稳定的真实颜色簇存在。
    - 调色板只在 --reset / 新状态时建立；后续截图沿用同一 palette，类别 ID 不会重排。
    """
    samples, positions = grid_samples(image_rgb)

    train = np.asarray([
        rgb for rgb, (r, _c) in zip(samples, positions)
        if 5 <= r <= 51
    ], dtype=np.float32)

    if len(train) == 0:
        raise RuntimeError("棋盘采样为空，无法建立颜色类别。")

    labels = DBSCAN(
        eps=PALETTE_DBSCAN_EPS,
        min_samples=PALETTE_DBSCAN_MIN_SAMPLES,
    ).fit_predict(train)

    clusters: List[Tuple[int, np.ndarray, float]] = []
    for lab in sorted(set(labels) - {-1}):
        pts = train[labels == lab]
        if len(pts) < PALETTE_MIN_CLUSTER_SIZE:
            continue

        center = pts.mean(axis=0)
        # 中心 2x2 采样的真实车/色块主色应很集中。
        # 用到中心的 90 分位距离过滤异常杂色簇，而不是按“颜色总数”截断。
        radii = np.linalg.norm(pts - center, axis=1)
        compact90 = float(np.percentile(radii, 90)) if len(radii) else 999.0
        if compact90 <= PALETTE_DBSCAN_EPS * 1.25:
            clusters.append((len(pts), center.astype(np.float32), compact90))

    if not clusters:
        raise RuntimeError(
            "没有学习到稳定颜色类别。请确认截图版式正确，且棋盘区域可见。"
        )

    # 仅为了给 C01..Cn 一个可复现的顺序：先按簇大小降序，再按 RGB 排序。
    # 后续帧不重新学习，所以整局中的 ID 永久稳定。
    clusters.sort(
        key=lambda item: (
            -item[0],
            round(float(item[1][0]), 3),
            round(float(item[1][1]), 3),
            round(float(item[1][2]), 3),
        )
    )
    return np.asarray(
        [center for _n, center, _compact in clusters],
        dtype=np.float32,
    )


def initial_grid(image_rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Build the initial board with robust patch voting instead of center-2x2 only.

    The center sample is retained only as an extra guard inside fixed UI areas.
    Outside UI, ambiguous pixels abstain and the logical cell is decided by
    multiple exclusive palette-colored pixels across the cell body.
    """
    h, w = image_rgb.shape[:2]
    x0, dx, y0, dy = scaled_grid_geometry(w, h)

    grid = np.full((GRID_ROWS, GRID_COLS), UNKNOWN, dtype=np.int16)
    if len(palette) == 0:
        return grid

    for r in range(GRID_ROWS):
        cy = y0 + (r + 0.5) * dy
        for c in range(GRID_COLS):
            cx = x0 + (c + 0.5) * dx
            color, coverage, margin = _classify_grid_cell_color_patch(
                image_rgb,
                r,
                c,
                palette,
            )
            if color is None:
                continue

            if ui_covered(cx, cy, w, h):
                # UI remains a confidence penalty.  Require both a strong patch
                # vote and a center sample that agrees with the same color.
                center = _sample_grid_cell(image_rgb, r, c)
                d = np.linalg.norm(palette - center[None, :], axis=1)
                idx = int(d.argmin())
                if (
                    idx + 1 == color
                    and float(d[idx]) <= _UI_VISIBLE_CENTER_DIST
                    and coverage >= 0.50
                    and margin >= 0.20
                ):
                    grid[r, c] = int(color)
                continue

            grid[r, c] = int(color)

    return grid

def open_empty_mask(grid: np.ndarray) -> np.ndarray:
    """只从棋盘下方进入；上、左、右均视为墙。"""
    rows, cols = grid.shape
    opened = np.zeros_like(grid, dtype=bool)
    q: deque[Tuple[int, int]] = deque()

    # 虚拟的棋盘下方只与最后一行相邻。
    for c in range(cols):
        if int(grid[rows - 1, c]) == EMPTY:
            opened[rows - 1, c] = True
            q.append((rows - 1, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                if not opened[rr, cc] and int(grid[rr, cc]) == EMPTY:
                    opened[rr, cc] = True
                    q.append((rr, cc))
    return opened


def _frontier_cells(grid: np.ndarray) -> Set[Tuple[int, int]]:
    """
    返回当前唯一具有“物理上可能消失”资格的格子。

    资格来源只有两种：
      1) 最后一行，直接接触棋盘下方；
      2) 四邻域中至少一个格子属于从下方连通进来的 EMPTY 开放区。

    注意：
      - 这里不因为“同色连通块”而一次性把整个连通块放进候选。
      - 必须逐层剥离：边界格确认消失后，下一层格子才获得检查资格。
    """
    rows, cols = grid.shape
    opened = open_empty_mask(grid)
    frontier: Set[Tuple[int, int]] = set()

    # 底边直接接触棋盘外部。
    for c in range(cols):
        if int(grid[rows - 1, c]) != EMPTY:
            frontier.add((rows - 1, c))

    # 已开放 EMPTY 的四邻域。
    opened_positions = np.argwhere(opened)
    for r, c in opened_positions:
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if int(grid[rr, cc]) != EMPTY:
                frontier.add((rr, cc))

    return frontier


def _fallback_background_rgb(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    当 prev_grid 尚没有足够 EMPTY 时，从棋盘下缘与停车区之间的灰色游戏背景
    估计 EMPTY 背景色。

    只保留近灰色、中等亮度像素，再取中位数，避免偶发悬挂色块污染。
    """
    h, w = image_rgb.shape[:2]
    x1 = max(0, int(_INCREMENTAL_BG_FALLBACK_X1_N * w))
    x2 = min(w, int(_INCREMENTAL_BG_FALLBACK_X2_N * w))
    y1 = max(0, int(_INCREMENTAL_BG_FALLBACK_Y1_N * h))
    y2 = min(h, int(_INCREMENTAL_BG_FALLBACK_Y2_N * h))

    roi = image_rgb[y1:y2, x1:x2, :3].reshape(-1, 3)
    if len(roi) == 0:
        return None

    spread = roi.max(axis=1) - roi.min(axis=1)
    brightness = roi.mean(axis=1)
    good = roi[
        (spread <= 32.0)
        & (brightness >= 90.0)
        & (brightness <= 225.0)
    ]
    if len(good) < 20:
        return None

    return np.median(good, axis=0).astype(np.float32)


def _estimate_background_rgb(
    prev: np.ndarray,
    image_rgb: np.ndarray,
) -> Optional[np.ndarray]:
    """
    优先用“上一稳定状态中已经确定为空且从下方连通”的格子估计当前棋盘背景。

    即使当前截图顶部出现通知遮挡：
      - 遮挡通常只影响少量已空格；
      - 近灰色/亮度过滤 + 中位数会把异常像素剔除。
    """
    opened = open_empty_mask(prev)
    coords = np.argwhere(opened)

    if len(coords) > _INCREMENTAL_BG_SAMPLE_LIMIT:
        # 均匀抽样，避免只取左上角或某一局部。
        idx = np.linspace(
            0,
            len(coords) - 1,
            _INCREMENTAL_BG_SAMPLE_LIMIT,
            dtype=np.int32,
        )
        coords = coords[idx]

    samples: List[np.ndarray] = []
    for r, c in coords:
        rgb = _sample_grid_cell(image_rgb, int(r), int(c))
        spread = float(rgb.max() - rgb.min())
        brightness = float(rgb.mean())
        if spread <= 38.0 and 80.0 <= brightness <= 230.0:
            samples.append(rgb)

    if len(samples) >= 6:
        return np.median(
            np.asarray(samples, dtype=np.float32),
            axis=0,
        ).astype(np.float32)

    return _fallback_background_rgb(image_rgb)


def _looks_like_empty_background(
    rgb: np.ndarray,
    background_rgb: Optional[np.ndarray],
    palette: np.ndarray,
) -> bool:
    """
    Decide whether one RGB sample is genuinely board background.

    v5.16 uses background-vs-palette competition instead of an absolute
    distance from every palette center.  Exact gray background is allowed to
    be geometrically close to a real color as long as it is substantially
    closer to the learned background model.
    """
    if background_rgb is None:
        return False

    rgb = np.asarray(rgb, dtype=np.float32)
    bg_dist = float(np.linalg.norm(rgb - background_rgb))
    if bg_dist > _RECOG_BACKGROUND_MAX_DIST:
        return False

    if len(palette) > 0:
        d = np.linalg.norm(
            np.asarray(palette, dtype=np.float32) - rgb[None, :],
            axis=1,
        )
        nearest_palette = float(d.min())
        if (
            nearest_palette - bg_dist
            < _RECOG_BACKGROUND_MIN_ADVANTAGE
        ):
            return False

    return True


def _recover_unknown_color(
    rgb: np.ndarray,
    palette: np.ndarray,
) -> Optional[int]:
    """
    UNKNOWN 只能在成为当前开放 frontier 后尝试恢复成真实颜色。

    UNKNOWN 永远不会因为“看起来像背景”直接变 EMPTY，
    从而继续遵守“UNKNOWN 不能用于安全证明”的项目原则。
    """
    if len(palette) == 0:
        return None

    d = np.linalg.norm(palette - rgb[None, :], axis=1)
    idx = int(d.argmin())
    if float(d[idx]) < _INCREMENTAL_UNKNOWN_RECOVER_DIST:
        return 1 + idx
    return None


def update_grid_causal(
    prev: np.ndarray,
    image_rgb: np.ndarray,
    palette: np.ndarray,
    consumed_by_color: Dict[int, int],
    prev_grid_rgb: Optional[np.ndarray] = None,
    prev_image_rgb: Optional[np.ndarray] = None,
) -> CausalBoardUpdate:
    """
    v5.16 visual-truth board synchronization.

    The responsibility split is strict:

      visual recognition
          decides WHICH logical cells are EMPTY;

      capacity conservation
          only audits HOW MANY cells should have been consumed by color.

    In particular this function never:
      - takes the top-N visual candidates because capacity says N;
      - forces a remaining budget onto frontier cells;
      - keeps a clearly gray historical ghost merely because this turn has
        zero capacity change for that color.

    Every stable observation re-checks all persistent colored cells.  A cell is
    cleared only when the current stable frame independently classifies it as
    EMPTY.  PRESENT and UNCERTAIN both preserve the previous persistent state.
    """
    if prev.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(
            f"棋盘状态尺寸异常: {prev.shape}，期望 {(GRID_ROWS, GRID_COLS)}"
        )

    expected: Dict[int, int] = {
        int(color): int(count)
        for color, count in consumed_by_color.items()
        if int(color) > 0 and int(count) > 0
    }

    current_grid_rgb = sample_grid_rgb_snapshot(image_rgb)

    snapshot_ok = (
        prev_grid_rgb is not None
        and np.asarray(prev_grid_rgb).shape == (GRID_ROWS, GRID_COLS, 3)
    )
    previous_frame_ok = (
        prev_image_rgb is not None
        and np.asarray(prev_image_rgb).shape == image_rgb.shape
    )

    grid = prev.copy()
    background_rgb = _estimate_background_rgb(prev, image_rgb)

    confirmed: Dict[int, int] = defaultdict(int)
    background_confirmed: Dict[int, int] = defaultdict(int)
    reconciled_without_capacity: Dict[int, int] = defaultdict(int)
    checked_cells = 0

    invalid_reason = ""
    if background_rgb is None:
        invalid_reason = "无法可靠估计棋盘 EMPTY 背景色，视觉空间状态不提交"

    if not invalid_reason:
        h, w = image_rgb.shape[:2]

        # Global visual reconciliation is intentional.  A previously missed
        # ghost may no longer be on the current frontier and may have zero
        # current-turn capacity.  Clear it if the current stable image directly
        # proves that the logical cell is background.
        coords = np.argwhere(prev > 0)
        for rr, cc in coords:
            r, c = int(rr), int(cc)
            old_color = int(prev[r, c])
            checked_cells += 1

            cx, cy = _grid_cell_center(r, c, w, h)
            covered = ui_covered(cx, cy, w, h)

            state, _old_cov, _bg_cov, _bg_zones = (
                _current_cell_visual_state(
                    image_rgb,
                    r,
                    c,
                    old_color,
                    palette,
                    background_rgb,
                    covered_by_fixed_ui=covered,
                )
            )

            if state != "EMPTY":
                continue

            grid[r, c] = EMPTY
            confirmed[old_color] += 1
            background_confirmed[old_color] += 1

            if old_color not in expected:
                reconciled_without_capacity[old_color] += 1

        # UNKNOWN remains conservative in the causal path.  v5.16 does not
        # globally repaint UNKNOWN cells from a single frame, because the gray
        # background can be close to a palette center.  Only previously known
        # COLOR cells participate in visual truth reconciliation here.

    # Capacity is an audit only.  Never use these numbers to pick cell
    # positions or to undo visually confirmed EMPTY cells.
    remaining: Dict[int, int] = {}
    excess: Dict[int, int] = {}
    for color, expected_count in sorted(expected.items()):
        visual_count = int(confirmed.get(color, 0))
        if visual_count < expected_count:
            remaining[color] = expected_count - visual_count
        elif visual_count > expected_count:
            excess[color] = visual_count - expected_count

    return CausalBoardUpdate(
        grid=grid,
        removed=sum(confirmed.values()),
        expected_by_color=dict(sorted(expected.items())),
        confirmed_by_color=dict(sorted(confirmed.items())),
        remaining_by_color=dict(sorted(remaining.items())),
        excess_by_color=dict(sorted(excess.items())),
        invalid_reason=invalid_reason,
        checked_cells=checked_cells,
        temporal_confirmed_by_color={},
        background_confirmed_by_color=dict(
            sorted(background_confirmed.items())
        ),
        ambiguous_changed_by_color={},
        temporal_change_threshold=_RECOG_TEMPORAL_CHANGE_DIST,
        temporal_snapshot_available=bool(snapshot_ok),
        patch_confirmed_by_color={},
        patch_coverage_drop_threshold=_CAUSAL_PATCH_COVERAGE_DROP,
        patch_previous_frame_available=bool(previous_frame_ok),
        # Reuse this existing telemetry field to make historical/global
        # reconciliation visible without changing AnalysisResult serialization.
        strong_nonfrontier_confirmed_by_color=dict(
            sorted(reconciled_without_capacity.items())
        ),
        ui_unknown_confirmed_by_color={},
    )


def update_grid(
    prev: np.ndarray,
    image_rgb: np.ndarray,
    palette: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    拓扑约束的增量棋盘更新。

    与旧实现的关键区别：
    --------------------
    旧实现：
        每帧重新扫描 52x38 全棋盘；
        已知颜色只要当前“不像原颜色”就直接永久写 EMPTY。

    新实现：
        1) 从上一稳定状态开始；
        2) 只检查“从棋盘下方开放区当前可接触”的 frontier；
        3) frontier 中的已知色块只有“明确变成棋盘灰色背景”才写 EMPTY；
        4) 每确认一个 EMPTY，重新扩大开放区，再检查新 frontier；
        5) frontier 之外的已知格完全不采样、不更新；
        6) UNKNOWN 只能在 frontier 上恢复成真实颜色，永不直接清空。

    这样系统通知即使覆盖半张棋盘，也不会造成：
        COLOR -> EMPTY -> 永久损坏。

    返回值保持旧接口兼容：
        (new_grid, removed_count)
    """
    if prev.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(
            f"棋盘状态尺寸异常: {prev.shape}，期望 {(GRID_ROWS, GRID_COLS)}"
        )

    grid = prev.copy()
    h, w = image_rgb.shape[:2]
    background_rgb = _estimate_background_rgb(prev, image_rgb)

    removed = 0

    # 同一张稳定截图里，一个已确认“仍存在/无法确认”的格子没有必要反复采样。
    settled: Set[Tuple[int, int]] = set()

    # 每一次真正删除至少一个格子，开放区都会严格扩大；
    # 最大循环次数不会超过棋盘格总数。
    for _round in range(GRID_ROWS * GRID_COLS + 1):
        frontier = _frontier_cells(grid)
        pending = [
            pos for pos in sorted(frontier)
            if pos not in settled
        ]

        if not pending:
            break

        newly_removed = 0

        for r, c in pending:
            old = int(grid[r, c])

            # 固定顶部 UI 遮挡区域不做视觉承诺。
            cx, cy = _grid_cell_center(r, c, w, h)
            if ui_covered(cx, cy, w, h):
                settled.add((r, c))
                continue

            rgb = _sample_grid_cell(image_rgb, r, c)

            if old > 0:
                # 色块不会变色。当前仍像原颜色 -> 明确保留。
                if old - 1 < len(palette):
                    ref = palette[old - 1]
                    if float(np.linalg.norm(rgb - ref)) < KEEP_COLOR_DIST:
                        settled.add((r, c))
                        continue

                # 只有明确看到棋盘背景，才允许 COLOR -> EMPTY。
                if _looks_like_empty_background(rgb, background_rgb, palette):
                    grid[r, c] = EMPTY
                    removed += 1
                    newly_removed += 1
                    # 不加入 settled：它已经变 EMPTY，会自然退出 frontier。
                    continue

                # 既不像旧颜色，也不像棋盘背景：
                # 通知/动画/高光/其它 UI 污染。保持上一确定状态。
                settled.add((r, c))
                continue

            if old == UNKNOWN:
                recovered = _recover_unknown_color(rgb, palette)
                if recovered is not None:
                    grid[r, c] = recovered
                settled.add((r, c))
                continue

            # EMPTY 不应进入 frontier，保险处理。
            settled.add((r, c))

        if newly_removed == 0:
            break

    return grid, removed


def reachable_components(
    grid: np.ndarray,
) -> Tuple[Dict[int, List[List[Tuple[int, int]]]], np.ndarray]:
    """
    返回每种颜色当前从下方开放区可触达的“整块同色连通区域”。
    一旦区域有一个格子接触开放区，若同色车持续装载，该整个同色连通区域都可以逐层剥离。
    """
    rows, cols = grid.shape
    opened = open_empty_mask(grid)
    visited = np.zeros_like(grid, dtype=bool)
    result: Dict[int, List[List[Tuple[int, int]]]] = defaultdict(list)

    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if color <= 0 or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: List[Tuple[int, int]] = []
            touches_open = False

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                # 最后一行直接接触棋盘下方。
                if rr == rows - 1:
                    touches_open = True

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if opened[nr, nc]:
                            touches_open = True
                        if (
                            not visited[nr, nc]
                            and int(grid[nr, nc]) == color
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            if touches_open:
                result[color].append(cells)

    return result, opened


def reachable_summary(
    grid: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, Counter]]:
    """每色可触达总数 + 这些可触达区域背后的邻色接触数量。"""
    comps, _opened = reachable_components(grid)
    totals: Dict[int, int] = {}
    neighbors: Dict[int, Counter] = {}
    rows, cols = grid.shape

    for color, groups in comps.items():
        totals[color] = sum(len(g) for g in groups)
        cnt: Counter = Counter()
        for group in groups:
            for r, c in group:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        other = int(grid[rr, cc])
                        if other > 0 and other != color:
                            cnt[other] += 1
        neighbors[color] = cnt

    return totals, neighbors