from __future__ import annotations

from collections import Counter, defaultdict, deque
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

# 当已存在 EMPTY 很少时，使用棋盘下缘和停车区之间的灰色游戏背景估计背景色。
# 这些比例针对整个截图而不是棋盘网格；只作为首轮/极少 EMPTY 时的兜底。
_INCREMENTAL_BG_FALLBACK_X1_N = 0.22
_INCREMENTAL_BG_FALLBACK_X2_N = 0.78
_INCREMENTAL_BG_FALLBACK_Y1_N = 0.515
_INCREMENTAL_BG_FALLBACK_Y2_N = 0.545


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