from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import (
    REF_W, REF_H, GRID_ROWS, GRID_COLS,
    X_EDGE0_REF, X_STEP_REF, Y_EDGE0_REF, Y_STEP_REF,
    PALETTE_DBSCAN_EPS, PALETTE_DBSCAN_MIN_SAMPLES,
    PALETTE_MIN_CLUSTER_SIZE, KEEP_COLOR_DIST, INITIAL_COLOR_DIST,
    UNKNOWN, EMPTY,
)

from .models import ObservationHealth


# ---------------------------------------------------------------------------
# Current-frame observation primitives
# ---------------------------------------------------------------------------
# History is secondary evidence only. This distance is used when a previous
# trusted color patch was sparsely classified and the resolver needs a targeted
# old-color pixel mask; it never selects disappearing coordinates by capacity.
_HISTORY_COLOR_FALLBACK_DIST = 44.0

# Fixed top UI is a confidence penalty for current-frame color confirmation.
_UI_VISIBLE_CENTER_DIST = 30.0

# perception-first board recognition.
#
# A logical cell is ~21x18 px at the reference resolution.  Center 2x2 RGB
# is too fragile when a disappearing block is covered by a car body / foot /
# shadow, and absolute distance-to-one-palette-color is ambiguous for close
# palette pairs.  The current classifier therefore:
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

# visual truth is authoritative for spatial state.
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

# Fixed-UI cells may stay UNKNOWN even though the UI is translucent enough for
# a real underlying block removal to create a large temporal pixel change.
# These thresholds detect an anonymous UNKNOWN->EMPTY event from visual time
# difference only. Capacity never selects the coordinate or its old color.
_RECOG_OCCLUDED_CHANGE_PIXEL_DIST = 30.0
_RECOG_OCCLUDED_CHANGE_MIN_RATIO = 0.38
_RECOG_OCCLUDED_CHANGE_MIN_MEAN = 28.0

# 当已存在 EMPTY 很少时，使用棋盘下缘和停车区之间的灰色游戏背景估计背景色。
# 这些比例针对整个截图而不是棋盘网格；只作为首轮/极少 EMPTY 时的兜底。
_BACKGROUND_FALLBACK_X1_N = 0.22
_BACKGROUND_FALLBACK_X2_N = 0.78
_BACKGROUND_FALLBACK_Y1_N = 0.515
_BACKGROUND_FALLBACK_Y2_N = 0.545


@dataclass
class ObservedBoard:
    """
    One stable-frame spatial observation.

    ``grid`` is created from the current frame first. Historical state is only
    allowed to resolve cells that the current frame left UNKNOWN, or to validate
    transition invariants. Capacity data is quantity-only diagnostics.
    """

    grid: np.ndarray
    health: ObservationHealth
    evidence_by_cell: Dict[Tuple[int, int], str]
    background_rgb: Optional[np.ndarray]
    grid_rgb_snapshot: np.ndarray
    current_color_cells: int
    current_empty_cells: int
    current_unknown_cells: int
    history_resolved_cells: int
    temporal_resolved_empty_cells: int
    removed_cells: int
    visual_removed_by_color: Dict[int, int]
    direct_empty_by_color: Dict[int, int]
    temporal_empty_by_color: Dict[int, int]
    history_resolved_by_color: Dict[int, int]
    capacity_expected_by_color: Dict[int, int]
    previous_grid_rgb_available: bool
    previous_frame_available: bool

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

    The current background model avoids an important false-negative:
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
            & (target_dist <= _HISTORY_COLOR_FALLBACK_DIST)
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

    Pixels are assigned to the nearest sufficiently separated palette
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


def _fallback_background_rgb(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    当 prev_grid 尚没有足够 EMPTY 时，从棋盘下缘与停车区之间的灰色游戏背景
    估计 EMPTY 背景色。

    只保留近灰色、中等亮度像素，再取中位数，避免偶发悬挂色块污染。
    """
    h, w = image_rgb.shape[:2]
    x1 = max(0, int(_BACKGROUND_FALLBACK_X1_N * w))
    x2 = min(w, int(_BACKGROUND_FALLBACK_X2_N * w))
    y1 = max(0, int(_BACKGROUND_FALLBACK_Y1_N * h))
    y2 = min(h, int(_BACKGROUND_FALLBACK_Y2_N * h))

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


def _looks_like_empty_background(
    rgb: np.ndarray,
    background_rgb: Optional[np.ndarray],
    palette: np.ndarray,
) -> bool:
    """
    Decide whether one RGB sample is genuinely board background.

    Uses background-vs-palette competition instead of an absolute
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


# ---------------------------------------------------------------------------
# current-stable-frame board observation authority
# ---------------------------------------------------------------------------

def _background_zone_support(background_mask: np.ndarray) -> int:
    """Count 3x3 sub-zones that are predominantly current-frame background."""
    if background_mask.size == 0:
        return 0
    hh, ww = background_mask.shape
    zones = 0
    for yi in range(3):
        y1 = (hh * yi) // 3
        y2 = (hh * (yi + 1)) // 3
        for xi in range(3):
            x1 = (ww * xi) // 3
            x2 = (ww * (xi + 1)) // 3
            zone = background_mask[y1:y2, x1:x2]
            if zone.size and float(np.mean(zone)) >= _RECOG_EMPTY_ZONE_COVERAGE:
                zones += 1
    return zones


def _classify_current_frame_cell(
    image_rgb: np.ndarray,
    r: int,
    c: int,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
) -> Tuple[int, str]:
    """Classify one cell using the current stable image only."""
    patch = _sample_grid_cell_inner_patch(image_rgb, r, c)
    if patch.size == 0:
        return UNKNOWN, "current:unknown:empty-patch"

    labels = _palette_pixel_labels(patch, palette)
    background_mask = _pixel_background_mask(patch, background_rgb, palette)

    # A pixel explained better by board background must not also vote for a
    # close palette color (the level-15 gray/C07 failure mode).
    if labels.size:
        labels = labels.copy()
        labels[background_mask] = UNKNOWN

    bg_coverage = float(np.mean(background_mask)) if background_mask.size else 0.0
    bg_zones = _background_zone_support(background_mask)

    winner_color: Optional[int] = None
    winner_coverage = 0.0
    winner_margin = 0.0
    winner_count = 0
    if labels.size and len(palette) > 0:
        counts = np.asarray(
            [
                int(np.count_nonzero(labels == color))
                for color in range(1, len(palette) + 1)
            ],
            dtype=np.int32,
        )
        winner_idx = int(np.argmax(counts))
        winner_count = int(counts[winner_idx])
        winner_coverage = float(winner_count / labels.size)
        if len(counts) >= 2:
            runner_up = int(np.partition(counts, -2)[-2])
        else:
            runner_up = 0
        winner_margin = float((winner_count - runner_up) / labels.size)
        if (
            winner_count >= _RECOG_CELL_MIN_COLOR_PIXELS
            and winner_coverage >= _RECOG_CELL_MIN_COLOR_COVERAGE
            and winner_margin >= _RECOG_CELL_MIN_WIN_MARGIN
        ):
            winner_color = winner_idx + 1

    h, w = image_rgb.shape[:2]
    cx, cy = _grid_cell_center(r, c, w, h)
    covered = ui_covered(cx, cy, w, h)

    if covered:
        center_rgb = _sample_grid_cell(image_rgb, r, c)
        center_bg = _looks_like_empty_background(center_rgb, background_rgb, palette)
        empty_confident = bool(
            center_bg
            and bg_coverage >= _RECOG_UI_EMPTY_MIN_BG_COVERAGE
            and bg_zones >= _RECOG_UI_EMPTY_MIN_BG_ZONES
        )

        color_confident = False
        if winner_color is not None:
            d = np.linalg.norm(
                np.asarray(palette, dtype=np.float32) - center_rgb[None, :],
                axis=1,
            )
            idx = int(d.argmin())
            color_confident = bool(
                idx + 1 == winner_color
                and float(d[idx]) <= _UI_VISIBLE_CENTER_DIST
                and winner_coverage >= 0.50
                and winner_margin >= 0.20
            )
    else:
        empty_confident = bool(
            background_rgb is not None
            and bg_coverage >= _RECOG_EMPTY_MIN_BG_COVERAGE
            and bg_zones >= _RECOG_EMPTY_MIN_BG_ZONES
        )
        color_confident = winner_color is not None

    if empty_confident and color_confident:
        return (
            UNKNOWN,
            "current:unknown:color-background-conflict:"
            f"color={winner_color},color_cov={winner_coverage:.3f},"
            f"bg_cov={bg_coverage:.3f},bg_zones={bg_zones}",
        )
    if empty_confident:
        return EMPTY, f"current:empty:bg_cov={bg_coverage:.3f},bg_zones={bg_zones}"
    if color_confident and winner_color is not None:
        return (
            int(winner_color),
            f"current:{ctag(winner_color)}:coverage={winner_coverage:.3f},"
            f"margin={winner_margin:.3f}",
        )

    return (
        UNKNOWN,
        "current:unknown:"
        f"best_color={winner_color or 0},coverage={winner_coverage:.3f},"
        f"margin={winner_margin:.3f},bg_cov={bg_coverage:.3f},"
        f"bg_zones={bg_zones}",
    )


def _classify_current_frame_grid(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], str]]:
    """Fresh 52x38 COLOR/EMPTY/UNKNOWN classification; no historical input."""
    grid = np.full((GRID_ROWS, GRID_COLS), UNKNOWN, dtype=np.int16)
    evidence: Dict[Tuple[int, int], str] = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            value, why = _classify_current_frame_cell(
                image_rgb, r, c, palette, background_rgb
            )
            grid[r, c] = int(value)
            evidence[(r, c)] = why
    return grid, evidence


def _direct_transition_conflicts(
    previous_grid: np.ndarray,
    current_direct_grid: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """Return forbidden transitions strongly asserted by the current frame."""
    conflicts: List[Tuple[int, int, int, int]] = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            old = int(previous_grid[r, c])
            cur = int(current_direct_grid[r, c])
            if cur <= 0:
                continue
            if old == EMPTY:
                conflicts.append((r, c, old, cur))
            elif old > 0 and old != cur:
                conflicts.append((r, c, old, cur))
    return conflicts


def _temporal_old_color_disappeared(
    evidence: _CellVisualEvidence,
    *,
    covered_by_fixed_ui: bool,
) -> bool:
    strong = bool(
        evidence.loss_ratio >= _RECOG_DISAPPEAR_LOSS_RATIO
        and evidence.change_ratio >= _RECOG_DISAPPEAR_CHANGE_RATIO
        and evidence.stable_ratio <= _RECOG_DISAPPEAR_STABLE_MAX
        and evidence.curr_color_coverage <= _RECOG_DISAPPEAR_CURR_COVERAGE_MAX
        and (
            evidence.zone_support >= 2
            or evidence.background_ratio >= _RECOG_DISAPPEAR_BG_RATIO
            or evidence.strong_change_ratio >= 0.48
        )
    )
    very_strong = bool(
        evidence.loss_ratio >= _RECOG_VERY_STRONG_LOSS_RATIO
        and evidence.change_ratio >= _RECOG_VERY_STRONG_CHANGE_RATIO
        and evidence.stable_ratio <= _RECOG_VERY_STRONG_STABLE_MAX
        and evidence.curr_color_coverage <= _RECOG_VERY_STRONG_CURR_COVERAGE_MAX
        and (
            evidence.zone_support >= 2
            or evidence.background_ratio >= 0.20
            or evidence.strong_change_ratio >= 0.68
        )
    )
    if covered_by_fixed_ui:
        return bool(very_strong and evidence.background_ratio >= 0.20)
    return bool(strong or very_strong)


def _apply_persistent_empty_invariant(
    current_grid: np.ndarray,
    evidence_by_cell: Dict[Tuple[int, int], str],
    previous_grid: np.ndarray,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], str], int]:
    """
    Preserve a previously confirmed logical EMPTY against direct sprite spill.

    Once a block has been consumed, that logical board cell cannot contain a
    block again. A later stable screenshot may still paint pixels from a
    neighboring surviving block inside the empty cell (body/feet/shadow).
    Such a direct COLOR vote is rendering contamination, not a legal
    EMPTY -> COLOR transition.

    Capacity data is deliberately absent.
    """
    grid = current_grid.copy()
    evidence = dict(evidence_by_cell)

    overridden = np.argwhere(
        (previous_grid == EMPTY) & (current_grid > 0)
    )
    for rr, cc in overridden:
        r, c = int(rr), int(cc)
        direct = int(current_grid[r, c])
        grid[r, c] = EMPTY
        evidence[(r, c)] = evidence.get((r, c), "") + (
            "|history:empty-invariant-overrides-direct-color:"
            f"direct={ctag(direct)}"
        )

    return grid, evidence, int(len(overridden))


def _apply_temporal_disappearance_overrides(
    current_grid: np.ndarray,
    evidence_by_cell: Dict[Tuple[int, int], str],
    image_rgb: np.ndarray,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
    previous_grid: np.ndarray,
    current_grid_rgb: np.ndarray,
    previous_grid_rgb: Optional[np.ndarray],
    previous_frame_rgb: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[Tuple[int, int], str], Dict[int, int]]:
    """
    Correct direct COLOR false positives caused by neighboring block sprites.

    A removed logical block can leave an EMPTY cell whose pixels are partly
    covered by the body/feet of an adjacent surviving block. The fresh-frame
    classifier may then assert either the old color again or a different color.
    When the previous trusted frame independently proves that the old block body
    disappeared, that temporal visual evidence is stronger than the contaminated
    direct color vote and the logical cell is EMPTY.

    Capacity data is deliberately absent here: every coordinate must prove its
    own disappearance from visual/temporal evidence.
    """
    grid = current_grid.copy()
    evidence = dict(evidence_by_cell)
    temporal_empty_by_color: Dict[int, int] = defaultdict(int)

    if previous_frame_rgb is None:
        return grid, evidence, {}

    h, w = image_rgb.shape[:2]
    candidates = np.argwhere((previous_grid > 0) & (current_grid > 0))
    for rr, cc in candidates:
        r, c = int(rr), int(cc)
        old = int(previous_grid[r, c])
        direct = int(current_grid[r, c])
        if old <= 0 or old > len(palette):
            continue

        cx, cy = _grid_cell_center(r, c, w, h)
        covered = ui_covered(cx, cy, w, h)
        temporal = _cell_disappearance_evidence(
            previous_frame_rgb,
            image_rgb,
            current_grid_rgb,
            previous_grid_rgb,
            r,
            c,
            old,
            palette,
            background_rgb,
        )
        if not _temporal_old_color_disappeared(
            temporal,
            covered_by_fixed_ui=covered,
        ):
            continue

        grid[r, c] = EMPTY
        evidence[(r, c)] = evidence.get((r, c), "") + (
            "|current-temporal:empty:old-body-loss:"
            f"old={ctag(old)},direct={ctag(direct)},"
            f"loss={temporal.loss_ratio:.3f},"
            f"change={temporal.change_ratio:.3f},"
            f"stable={temporal.stable_ratio:.3f},"
            f"zones={temporal.zone_support}"
        )
        temporal_empty_by_color[old] += 1

    return (
        grid,
        evidence,
        dict(sorted(temporal_empty_by_color.items())),
    )


def _resolve_unknown_from_history(
    current_grid: np.ndarray,
    evidence_by_cell: Dict[Tuple[int, int], str],
    image_rgb: np.ndarray,
    palette: np.ndarray,
    background_rgb: Optional[np.ndarray],
    previous_grid: np.ndarray,
    current_grid_rgb: np.ndarray,
    previous_grid_rgb: Optional[np.ndarray],
    previous_frame_rgb: Optional[np.ndarray],
) -> Tuple[
    np.ndarray,
    Dict[Tuple[int, int], str],
    int,
    Dict[int, int],
    Dict[int, int],
]:
    """
    Resolve *only current UNKNOWN cells* using the previous trusted state.

    History is never copied wholesale. For a previous COLOR, current targeted
    visual evidence must still say PRESENT/EMPTY, or temporal old-color body
    loss must prove COLOR->EMPTY. Otherwise the cell stays UNKNOWN.
    """
    grid = current_grid.copy()
    evidence = dict(evidence_by_cell)
    history_resolved = 0
    history_by_color: Dict[int, int] = defaultdict(int)
    temporal_empty_by_color: Dict[int, int] = defaultdict(int)

    h, w = image_rgb.shape[:2]
    unknown_positions = np.argwhere(current_grid == UNKNOWN)
    for rr, cc in unknown_positions:
        r, c = int(rr), int(cc)
        old = int(previous_grid[r, c])

        if old == UNKNOWN:
            continue
        if old == EMPTY:
            # The trusted invariant EMPTY -> COLOR is impossible. The current
            # frame did not contradict it (it was UNKNOWN), so history may
            # resolve this uncertainty without inventing a new spatial change.
            grid[r, c] = EMPTY
            evidence[(r, c)] += "|history:empty-invariant"
            history_resolved += 1
            continue
        if old <= 0 or old > len(palette):
            continue

        cx, cy = _grid_cell_center(r, c, w, h)
        covered = ui_covered(cx, cy, w, h)
        state, old_cov, bg_cov, bg_zones = _current_cell_visual_state(
            image_rgb,
            r,
            c,
            old,
            palette,
            background_rgb,
            covered_by_fixed_ui=covered,
        )
        if state == "PRESENT":
            grid[r, c] = old
            evidence[(r, c)] += (
                f"|history-targeted:present:{ctag(old)}:old_cov={old_cov:.3f}"
            )
            history_resolved += 1
            history_by_color[old] += 1
            continue
        if state == "EMPTY":
            grid[r, c] = EMPTY
            evidence[(r, c)] += (
                f"|history-targeted:empty:bg_cov={bg_cov:.3f},bg_zones={bg_zones}"
            )
            history_resolved += 1
            history_by_color[old] += 1
            continue

        if previous_frame_rgb is None:
            continue

        temporal = _cell_disappearance_evidence(
            previous_frame_rgb,
            image_rgb,
            current_grid_rgb,
            previous_grid_rgb,
            r,
            c,
            old,
            palette,
            background_rgb,
        )
        if not _temporal_old_color_disappeared(
            temporal,
            covered_by_fixed_ui=covered,
        ):
            continue

        grid[r, c] = EMPTY
        evidence[(r, c)] += (
            "|history-temporal:empty:"
            f"loss={temporal.loss_ratio:.3f},change={temporal.change_ratio:.3f},"
            f"stable={temporal.stable_ratio:.3f},zones={temporal.zone_support}"
        )
        history_resolved += 1
        history_by_color[old] += 1
        temporal_empty_by_color[old] += 1

    return (
        grid,
        evidence,
        history_resolved,
        dict(sorted(history_by_color.items())),
        dict(sorted(temporal_empty_by_color.items())),
    )


def _resolve_occluded_unknown_temporal_empty(
    current_grid: np.ndarray,
    evidence_by_cell: Dict[Tuple[int, int], str],
    previous_grid: np.ndarray,
    previous_frame_rgb: Optional[np.ndarray],
    current_frame_rgb: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[Tuple[int, int], str],
    int,
    List[Tuple[int, int]],
]:
    """
    Resolve fixed-UI UNKNOWN cells that visually prove an underlying removal.

    The fixed top UI is stable and partially translucent. If a logical cell was
    UNKNOWN before and remains UNKNOWN now, but a large fraction of the same
    inner-cell pixels changed between two stable frames, the underlying block
    has disappeared even though its old palette color was never observable.

    This is deliberately color-anonymous and uses no capacity information.
    """
    grid = current_grid.copy()
    evidence = dict(evidence_by_cell)
    coords: List[Tuple[int, int]] = []

    if previous_frame_rgb is None:
        return grid, evidence, 0, coords

    h, w = current_frame_rgb.shape[:2]
    for rr, cc in np.argwhere(
        (previous_grid == UNKNOWN) & (current_grid == UNKNOWN)
    ):
        r, c = int(rr), int(cc)
        cx, cy = _grid_cell_center(r, c, w, h)
        if not ui_covered(cx, cy, w, h):
            continue

        before = _sample_grid_cell_inner_patch(previous_frame_rgb, r, c)
        after = _sample_grid_cell_inner_patch(current_frame_rgb, r, c)
        hh = min(before.shape[0], after.shape[0])
        ww = min(before.shape[1], after.shape[1])
        if hh <= 0 or ww <= 0:
            continue

        before = before[:hh, :ww]
        after = after[:hh, :ww]
        delta = np.linalg.norm(
            after.astype(np.float32) - before.astype(np.float32),
            axis=2,
        )
        changed_ratio = float(
            np.mean(delta >= _RECOG_OCCLUDED_CHANGE_PIXEL_DIST)
        )
        mean_change = float(np.mean(delta))

        if (
            changed_ratio < _RECOG_OCCLUDED_CHANGE_MIN_RATIO
            or mean_change < _RECOG_OCCLUDED_CHANGE_MIN_MEAN
        ):
            continue

        grid[r, c] = EMPTY
        evidence[(r, c)] = evidence.get((r, c), "") + (
            "|ui-occluded-temporal:empty:"
            f"change_ratio={changed_ratio:.3f},"
            f"mean_change={mean_change:.1f}"
        )
        coords.append((r, c))

    return grid, evidence, len(coords), coords


def _capacity_audit_with_occluded_unknown(
    consumed_by_color: Optional[Dict[int, int]],
    visual_removed_by_color: Dict[int, int],
    occluded_unknown_removed: int,
) -> Tuple[
    Dict[int, int],
    Dict[int, int],
    Dict[int, int],
    Dict[int, int],
]:
    """
    Quantity-only reconciliation for color-anonymous fixed-UI removals.

    Spatial coordinates were already selected by independent temporal visual
    evidence. Capacity is allowed only to verify that the anonymous removal
    count exactly closes the total per-color quantity shortfall. It does not
    assign an old color to any coordinate.
    """
    expected, remaining, excess = _capacity_audit_only(
        consumed_by_color,
        visual_removed_by_color,
    )
    explained: Dict[int, int] = {}

    missing_total = int(sum(remaining.values()))
    if (
        remaining
        and not excess
        and int(occluded_unknown_removed) > 0
        and missing_total == int(occluded_unknown_removed)
    ):
        explained = dict(remaining)
        remaining = {}

    return expected, remaining, excess, explained

def _visual_removals_by_color(
    previous_grid: Optional[np.ndarray],
    current_grid: np.ndarray,
) -> Dict[int, int]:
    if previous_grid is None:
        return {}
    removed: Dict[int, int] = defaultdict(int)
    for rr, cc in np.argwhere((previous_grid > 0) & (current_grid == EMPTY)):
        old = int(previous_grid[int(rr), int(cc)])
        removed[old] += 1
    return dict(sorted(removed.items()))


def _capacity_audit_only(
    consumed_by_color: Optional[Dict[int, int]],
    visual_removed_by_color: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Compare quantities only. This function has no grid argument to mutate."""
    if consumed_by_color is None:
        return {}, {}, {}

    expected = {
        int(color): int(count)
        for color, count in consumed_by_color.items()
        if int(color) > 0 and int(count) > 0
    }
    remaining: Dict[int, int] = {}
    excess: Dict[int, int] = {}
    for color in sorted(set(expected) | set(visual_removed_by_color)):
        expected_n = int(expected.get(color, 0))
        visual_n = int(visual_removed_by_color.get(color, 0))
        if visual_n < expected_n:
            remaining[color] = expected_n - visual_n
        elif visual_n > expected_n:
            excess[color] = visual_n - expected_n
    return dict(sorted(expected.items())), remaining, excess


def observe_board(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    previous_trusted_grid: Optional[np.ndarray] = None,
    *,
    previous_trusted_frame_rgb: Optional[np.ndarray] = None,
    previous_trusted_grid_rgb: Optional[np.ndarray] = None,
    consumed_by_color: Optional[Dict[int, int]] = None,
) -> ObservedBoard:
    """
    Build one fresh board from the current stable frame.

    Authority order:
      1. current-frame COLOR / EMPTY / UNKNOWN classification for all 1976 cells;
      2. previous trusted state only for cells that are still UNKNOWN and for
         forbidden-transition validation;
      3. capacity conservation only as a quantity audit after spatial output.

    In particular this function never starts from ``previous_trusted_grid.copy()``
    and never uses a quantity budget to select coordinates.
    """
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    palette = np.asarray(palette, dtype=np.float32)
    background_rgb = _fallback_background_rgb(image_rgb)
    current_grid_rgb = sample_grid_rgb_snapshot(image_rgb)

    reasons: List[str] = []
    warnings: List[str] = []

    previous_grid: Optional[np.ndarray] = None
    if previous_trusted_grid is not None:
        candidate = np.asarray(previous_trusted_grid)
        if candidate.shape != (GRID_ROWS, GRID_COLS):
            reasons.append(
                "previous_trusted_grid_shape="
                f"{candidate.shape}, expected={(GRID_ROWS, GRID_COLS)}"
            )
        else:
            previous_grid = candidate.astype(np.int16, copy=False)

    previous_grid_rgb: Optional[np.ndarray] = None
    previous_grid_rgb_available = False
    if previous_trusted_grid_rgb is not None:
        candidate_rgb = np.asarray(previous_trusted_grid_rgb)
        if candidate_rgb.shape == (GRID_ROWS, GRID_COLS, 3):
            previous_grid_rgb = candidate_rgb.astype(np.float32, copy=False)
            previous_grid_rgb_available = True
        else:
            warnings.append(
                "previous_trusted_grid_rgb_shape="
                f"{candidate_rgb.shape}, ignored"
            )

    previous_frame_rgb: Optional[np.ndarray] = None
    previous_frame_available = False
    if previous_trusted_frame_rgb is not None:
        candidate_frame = np.asarray(previous_trusted_frame_rgb)
        if candidate_frame.shape == image_rgb.shape:
            previous_frame_rgb = candidate_frame.astype(np.float32, copy=False)
            previous_frame_available = True
        else:
            warnings.append(
                "previous_trusted_frame_shape="
                f"{candidate_frame.shape}, current={image_rgb.shape}, ignored"
            )

    if background_rgb is None:
        reasons.append("current_frame_background_unavailable")

    # PRIMARY AUTHORITY: fresh current-frame classification, never prev.copy().
    raw_direct_grid, evidence = _classify_current_frame_grid(
        image_rgb,
        palette,
        background_rgb,
    )

    # A stable current frame can still contain pixels from an adjacent block
    # sprite inside a logically EMPTY cell. Before treating a direct COLOR
    # assertion as a forbidden transition, let independent old-body-loss
    # evidence invalidate that contaminated color vote. Capacity is not used.
    direct_grid = raw_direct_grid
    persistent_empty_overrides = 0
    direct_temporal_empty_by_color: Dict[int, int] = {}
    if previous_grid is not None:
        (
            direct_grid,
            evidence,
            persistent_empty_overrides,
        ) = _apply_persistent_empty_invariant(
            direct_grid,
            evidence,
            previous_grid,
        )
        (
            direct_grid,
            evidence,
            direct_temporal_empty_by_color,
        ) = _apply_temporal_disappearance_overrides(
            direct_grid,
            evidence,
            image_rgb,
            palette,
            background_rgb,
            previous_grid,
            current_grid_rgb,
            previous_grid_rgb,
            previous_frame_rgb,
        )

    if persistent_empty_overrides:
        warnings.append(
            "persistent_empty_direct_color_overrides="
            f"{persistent_empty_overrides}"
        )

    conflicts: List[Tuple[int, int, int, int]] = []
    if previous_grid is not None:
        conflicts = _direct_transition_conflicts(previous_grid, direct_grid)
        for r, c, old, cur in conflicts:
            reasons.append(
                f"forbidden_transition R{r + 1:02d}C{c + 1:02d}:"
                f"{ctag(old) if old > 0 else 'EMPTY'}->{ctag(cur)}"
            )

    final_grid = direct_grid
    history_resolved = 0
    history_by_color: Dict[int, int] = {}
    temporal_empty_by_color: Dict[int, int] = dict(
        direct_temporal_empty_by_color
    )
    if previous_grid is not None:
        (
            final_grid,
            evidence,
            history_resolved,
            history_by_color,
            resolved_temporal_empty_by_color,
        ) = _resolve_unknown_from_history(
            direct_grid,
            evidence,
            image_rgb,
            palette,
            background_rgb,
            previous_grid,
            current_grid_rgb,
            previous_grid_rgb,
            previous_frame_rgb,
        )
        for color, count in resolved_temporal_empty_by_color.items():
            temporal_empty_by_color[color] = (
                temporal_empty_by_color.get(color, 0) + int(count)
            )
        temporal_empty_by_color = dict(sorted(temporal_empty_by_color.items()))

    occluded_ui_temporal_empty = 0
    if previous_grid is not None:
        (
            final_grid,
            evidence,
            occluded_ui_temporal_empty,
            _occluded_ui_coords,
        ) = _resolve_occluded_unknown_temporal_empty(
            final_grid,
            evidence,
            previous_grid,
            previous_frame_rgb,
            image_rgb,
        )
        if occluded_ui_temporal_empty:
            warnings.append(
                "occluded_ui_temporal_empty="
                f"{occluded_ui_temporal_empty}"
            )

    visual_removed = _visual_removals_by_color(previous_grid, final_grid)

    direct_empty_by_color: Dict[int, int] = defaultdict(int)
    if previous_grid is not None:
        # Keep this diagnostic strictly "current-frame direct background".
        # Temporal spill corrections are reported separately.
        for rr, cc in np.argwhere(
            (previous_grid > 0) & (raw_direct_grid == EMPTY)
        ):
            old = int(previous_grid[int(rr), int(cc)])
            direct_empty_by_color[old] += 1

    (
        expected,
        remaining,
        excess,
        occluded_explained,
    ) = _capacity_audit_with_occluded_unknown(
        consumed_by_color,
        visual_removed,
        occluded_ui_temporal_empty,
    )
    if occluded_explained:
        warnings.append(
            "capacity_remaining_explained_by_occluded_ui="
            + ",".join(
                f"{ctag(c)}:{n}"
                for c, n in sorted(occluded_explained.items())
            )
        )
    if remaining:
        reasons.append(
            "capacity_remaining="
            + ",".join(f"{ctag(c)}:{n}" for c, n in sorted(remaining.items()))
        )
    if excess:
        reasons.append(
            "capacity_excess="
            + ",".join(f"{ctag(c)}:{n}" for c, n in sorted(excess.items()))
        )

    unknown_cells = int(np.count_nonzero(final_grid == UNKNOWN))
    if unknown_cells:
        warnings.append(f"unresolved_unknown_cells={unknown_cells}")

    health = ObservationHealth(
        trusted=not reasons,
        reasons=reasons,
        warnings=warnings,
        unknown_cells=unknown_cells,
        transition_conflicts=conflicts,
        capacity_remaining_by_color=remaining,
        capacity_excess_by_color=excess,
    )

    return ObservedBoard(
        grid=final_grid,
        health=health,
        evidence_by_cell=evidence,
        background_rgb=background_rgb,
        grid_rgb_snapshot=current_grid_rgb,
        current_color_cells=int(np.count_nonzero(final_grid > 0)),
        current_empty_cells=int(np.count_nonzero(final_grid == EMPTY)),
        current_unknown_cells=unknown_cells,
        history_resolved_cells=int(history_resolved),
        temporal_resolved_empty_cells=int(
            sum(temporal_empty_by_color.values())
            + occluded_ui_temporal_empty
        ),
        removed_cells=int(
            sum(visual_removed.values())
            + occluded_ui_temporal_empty
        ),
        visual_removed_by_color=visual_removed,
        direct_empty_by_color=dict(sorted(direct_empty_by_color.items())),
        temporal_empty_by_color=temporal_empty_by_color,
        history_resolved_by_color=history_by_color,
        capacity_expected_by_color=expected,
        previous_grid_rgb_available=previous_grid_rgb_available,
        previous_frame_available=previous_frame_available,
    )


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