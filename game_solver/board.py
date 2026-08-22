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


def _sample_grid_cell_color_coverage(
    image_rgb: np.ndarray,
    r: int,
    c: int,
    color: int,
    palette: np.ndarray,
) -> float:
    if color <= 0 or color > len(palette):
        return -1.0

    h, w = image_rgb.shape[:2]
    x, y = _grid_cell_center(r, c, w, h)
    _x0, dx, _y0, dy = scaled_grid_geometry(w, h)

    rx = max(2, int(round(dx * _CAUSAL_PATCH_HALF_X_FRAC)))
    ry = max(2, int(round(dy * _CAUSAL_PATCH_HALF_Y_FRAC)))
    cx = int(round(x))
    cy = int(round(y))

    x1 = max(0, cx - rx)
    x2 = min(w, cx + rx + 1)
    y1 = max(0, cy - ry)
    y2 = min(h, cy + ry + 1)
    patch = image_rgb[y1:y2, x1:x2, :3]
    if patch.size == 0:
        return -1.0

    target = palette[color - 1].astype(np.float32)
    dist = np.linalg.norm(
        patch.astype(np.float32) - target[None, None, :],
        axis=2,
    )
    return float(np.mean(dist < _CAUSAL_PATCH_COLOR_DIST))


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
    h, w = image_rgb.shape[:2]
    x0, dx, y0, dy = scaled_grid_geometry(w, h)
    samples, positions = grid_samples(image_rgb)

    grid = np.full((GRID_ROWS, GRID_COLS), UNKNOWN, dtype=np.int16)
    for rgb, (r, c) in zip(samples, positions):
        cx = x0 + (c + 0.5) * dx
        cy = y0 + (r + 0.5) * dy
        if ui_covered(cx, cy, w, h):
            continue

        if len(palette) == 0:
            continue
        d = np.linalg.norm(palette - rgb, axis=1)
        if float(d.min()) < INITIAL_COLOR_DIST:
            grid[r, c] = 1 + int(d.argmin())
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
    “不像旧颜色”绝不等于 EMPTY。

    必须同时满足：
      - 明确接近当前棋盘灰色背景；
      - 与所有真实 palette 主色保持足够距离。

    因此通知条、白色 UI、动画、其它彩色覆盖只会成为“无法确认”，不会写 EMPTY。
    """
    if background_rgb is None:
        return False

    if float(np.linalg.norm(rgb - background_rgb)) > _INCREMENTAL_EMPTY_BG_DIST:
        return False

    if len(palette) > 0:
        d = np.linalg.norm(palette - rgb[None, :], axis=1)
        if float(d.min()) < _INCREMENTAL_EMPTY_PALETTE_MARGIN:
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
    使用“容量守恒 + committed stable frame 时间差分”更新棋盘。

    v5.9 修正两个已由真实日志/截图证明的问题：

    1) 不再要求所有被吸收格都必须露出纯灰背景。
       底部 sprite 的脚/阴影/相邻块覆盖会让逻辑格已消失，但中心采样仍非灰色。
       若某格属于当前 reachable 同色 component，且相对上一 committed 稳定截图
       发生明显视觉变化，则它也是有效的 causal removal evidence。

    2) update_grid_causal 本身完全纯函数化：
       它只基于传入的 prev / prev_grid_rgb 生成 candidate grid，不写持久化状态。
       观测重试必须始终从同一个 committed state 重新计算。

    数量仍由 consumed_by_color 决定；视觉只负责在物理可达候选中定位具体位置。
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
    previous_rgb = (
        np.asarray(prev_grid_rgb, dtype=np.float32)
        if snapshot_ok
        else None
    )

    previous_frame_ok = (
        prev_image_rgb is not None
        and np.asarray(prev_image_rgb).shape == image_rgb.shape
    )
    previous_frame = (
        np.asarray(prev_image_rgb, dtype=np.float32)
        if previous_frame_ok
        else None
    )

    def _empty_result(
        *,
        invalid_reason: str = "",
        remaining_override: Optional[Dict[int, int]] = None,
    ) -> CausalBoardUpdate:
        return CausalBoardUpdate(
            grid=prev.copy(),
            removed=0,
            expected_by_color=dict(sorted(expected.items())),
            confirmed_by_color={},
            remaining_by_color=(
                dict(sorted(remaining_override.items()))
                if remaining_override is not None
                else {}
            ),
            excess_by_color={},
            invalid_reason=invalid_reason,
            checked_cells=0,
            temporal_confirmed_by_color={},
            background_confirmed_by_color={},
            ambiguous_changed_by_color={},
            temporal_change_threshold=_CAUSAL_TEMPORAL_CHANGE_DIST,
            temporal_snapshot_available=bool(snapshot_ok),
            patch_confirmed_by_color={},
            patch_coverage_drop_threshold=_CAUSAL_PATCH_COVERAGE_DROP,
            patch_previous_frame_available=bool(previous_frame_ok),
            strong_nonfrontier_confirmed_by_color={},
        )

    if not expected:
        return _empty_result()

    grid = prev.copy()
    background_rgb = _estimate_background_rgb(prev, image_rgb)

    # 没有 temporal snapshot 时仍可退回纯背景证据，兼容旧 solver_state；
    # 新局 --reset 后会自动持久化 snapshot。
    remaining = dict(expected)
    confirmed: Dict[int, int] = defaultdict(int)
    temporal_confirmed: Dict[int, int] = defaultdict(int)
    patch_confirmed: Dict[int, int] = defaultdict(int)
    strong_nonfrontier_confirmed: Dict[int, int] = defaultdict(int)
    background_confirmed: Dict[int, int] = defaultdict(int)
    ambiguous_changed: Dict[int, int] = defaultdict(int)
    excess: Dict[int, int] = defaultdict(int)
    checked_cells = 0

    # 一张稳定截图中，每个格子的视觉证据固定；同一轮不用重复采样。
    checked: Set[Tuple[int, int]] = set()

    max_rounds = GRID_ROWS * GRID_COLS + 1
    for _round in range(max_rounds):
        # v5.9.1 strict physical frontier:
        # temporal diff can localize a disappearance, but it must not punch an
        # arbitrary hole inside a reachable same-color component. Only cells
        # touching the currently open EMPTY region are physically eligible.
        # After confirmed removals, the next loop recomputes the frontier.
        frontier = _frontier_cells(grid)

        candidate_positions_by_color: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for r, c in sorted(frontier):
            if (r, c) in checked:
                continue
            color = int(grid[r, c])
            if color <= 0:
                continue
            if int(remaining.get(color, 0)) <= 0:
                continue
            candidate_positions_by_color[color].append((r, c))

        if not any(candidate_positions_by_color.values()):
            break

        newly_removed = 0

        for color, positions in sorted(candidate_positions_by_color.items()):
            budget = int(remaining.get(color, 0))
            if budget <= 0:
                continue

            # evidence tuple:
            # (rank, change_dist, is_background, (r,c))
            #
            # rank:
            #   3 = 明确灰背景
            #   2 = strong temporal change
            #   1 = normal temporal change
            evidences: List[Tuple[int, float, bool, Tuple[int, int]]] = []

            for r, c in sorted(set(positions)):
                checked.add((r, c))

                cx, cy = _grid_cell_center(
                    r,
                    c,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                )
                if ui_covered(
                    cx,
                    cy,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                ):
                    continue

                current_rgb = current_grid_rgb[r, c]
                checked_cells += 1

                is_background = _looks_like_empty_background(
                    current_rgb,
                    background_rgb,
                    palette,
                )

                change_dist = 0.0
                if previous_rgb is not None:
                    change_dist = float(
                        np.linalg.norm(current_rgb - previous_rgb[r, c])
                    )

                if is_background:
                    evidences.append((3, change_dist, True, (r, c)))
                    continue

                if previous_rgb is not None:
                    if change_dist >= _CAUSAL_TEMPORAL_STRONG_CHANGE_DIST:
                        evidences.append((2, change_dist, False, (r, c)))
                        continue
                    if change_dist >= _CAUSAL_TEMPORAL_CHANGE_DIST:
                        evidences.append((1, change_dist, False, (r, c)))
                        continue

                # v5.9.2 fallback: compare old-color body coverage in a
                # small patch against the previous committed analysis frame.
                if previous_frame is not None:
                    prev_cov = _sample_grid_cell_color_coverage(
                        previous_frame, r, c, color, palette
                    )
                    if prev_cov >= _CAUSAL_PATCH_MIN_PREV_COVERAGE:
                        curr_cov = _sample_grid_cell_color_coverage(
                            image_rgb, r, c, color, palette
                        )
                        coverage_drop = prev_cov - curr_cov
                        if coverage_drop >= _CAUSAL_PATCH_COVERAGE_DROP:
                            # rank 0: weaker than direct center temporal.
                            evidences.append((0, coverage_drop, False, (r, c)))
                            continue

                # No background/center-temporal/patch-coverage evidence.

            if not evidences:
                continue

            # 优先：背景 > 强时间变化 > 一般时间变化；同级按变化距离降序。
            evidences.sort(
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )

            if len(evidences) > budget:
                # Experiment mode: record ambiguity but do not stall.
                # Candidates are already restricted to the frontier.
                ambiguous_changed[color] += len(evidences) - budget

            selected = evidences[:budget]

            for rank, _change_dist, is_background, (r, c) in selected:
                if int(grid[r, c]) != color:
                    continue

                grid[r, c] = EMPTY
                confirmed[color] += 1
                remaining[color] -= 1
                newly_removed += 1

                if is_background:
                    background_confirmed[color] += 1
                elif rank >= 1:
                    temporal_confirmed[color] += 1
                else:
                    patch_confirmed[color] += 1

        if all(v <= 0 for v in remaining.values()):
            break

        # 已删除位置可能暴露新的异色/同色 component。
        if newly_removed == 0:
            break

    # v5.10 second-stage causal localization.
    # Strict frontier remains the primary rule. Only unresolved capacity may
    # use this global fallback, and only with near-certain direct visual loss.
    if previous_frame is not None and any(v > 0 for v in remaining.values()):
        for color in sorted(remaining):
            budget = int(remaining.get(color, 0))
            if budget <= 0:
                continue

            strong_candidates: List[
                Tuple[int, float, float, Tuple[int, int]]
            ] = []

            coords = np.argwhere(
                (prev == int(color))
                & (grid == int(color))
            )
            for rr, cc in coords:
                r, c = int(rr), int(cc)

                cx, cy = _grid_cell_center(
                    r,
                    c,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                )
                if ui_covered(
                    cx,
                    cy,
                    image_rgb.shape[1],
                    image_rgb.shape[0],
                ):
                    continue

                prev_cov = _sample_grid_cell_color_coverage(
                    previous_frame,
                    r,
                    c,
                    color,
                    palette,
                )
                if prev_cov < _CAUSAL_GLOBAL_STRONG_PREV_COVERAGE:
                    continue

                curr_cov = _sample_grid_cell_color_coverage(
                    image_rgb,
                    r,
                    c,
                    color,
                    palette,
                )
                coverage_drop = prev_cov - curr_cov
                if curr_cov > _CAUSAL_GLOBAL_STRONG_CURR_COVERAGE_MAX:
                    continue
                if coverage_drop < _CAUSAL_GLOBAL_STRONG_COVERAGE_DROP:
                    continue

                current_rgb = current_grid_rgb[r, c]
                change_dist = 0.0
                if previous_rgb is not None:
                    change_dist = float(
                        np.linalg.norm(current_rgb - previous_rgb[r, c])
                    )

                is_background = _looks_like_empty_background(
                    current_rgb,
                    background_rgb,
                    palette,
                )
                if (
                    not is_background
                    and change_dist < _CAUSAL_GLOBAL_STRONG_CENTER_CHANGE
                ):
                    continue

                strong_candidates.append(
                    (
                        1 if is_background else 0,
                        float(coverage_drop),
                        float(change_dist),
                        (r, c),
                    )
                )

            strong_candidates.sort(
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )

            if len(strong_candidates) > budget:
                ambiguous_changed[color] += (
                    len(strong_candidates) - budget
                )

            for _bg_rank, _drop, _change, (r, c) in strong_candidates[:budget]:
                if int(grid[r, c]) != int(color):
                    continue
                if int(remaining.get(color, 0)) <= 0:
                    break

                grid[r, c] = EMPTY
                confirmed[color] += 1
                remaining[color] -= 1
                strong_nonfrontier_confirmed[color] += 1

    remaining_nonzero = {
        color: count
        for color, count in sorted(remaining.items())
        if count > 0
    }

    invalid_reason = ""
    if not snapshot_ok and background_rgb is None:
        invalid_reason = (
            "既没有上一 committed stable RGB snapshot，"
            "也无法可靠估计棋盘 EMPTY 背景色"
        )

    return CausalBoardUpdate(
        grid=grid,
        removed=sum(confirmed.values()),
        expected_by_color=dict(sorted(expected.items())),
        confirmed_by_color=dict(sorted(confirmed.items())),
        remaining_by_color=remaining_nonzero,
        excess_by_color=dict(sorted(excess.items())),
        invalid_reason=invalid_reason,
        checked_cells=checked_cells,
        temporal_confirmed_by_color=dict(sorted(temporal_confirmed.items())),
        background_confirmed_by_color=dict(sorted(background_confirmed.items())),
        ambiguous_changed_by_color=dict(sorted(ambiguous_changed.items())),
        temporal_change_threshold=_CAUSAL_TEMPORAL_CHANGE_DIST,
        temporal_snapshot_available=bool(snapshot_ok),
        patch_confirmed_by_color=dict(sorted(patch_confirmed.items())),
        patch_coverage_drop_threshold=_CAUSAL_PATCH_COVERAGE_DROP,
        patch_previous_frame_available=bool(previous_frame_ok),
        strong_nonfrontier_confirmed_by_color=dict(
            sorted(strong_nonfrontier_confirmed.items())
        ),
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