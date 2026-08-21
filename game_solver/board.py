from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import (
    REF_W, REF_H, GRID_ROWS, GRID_COLS,
    X_EDGE0_REF, X_STEP_REF, Y_EDGE0_REF, Y_STEP_REF,
    PALETTE_DBSCAN_EPS, PALETTE_DBSCAN_MIN_SAMPLES,
    PALETTE_MIN_CLUSTER_SIZE, KEEP_COLOR_DIST, INITIAL_COLOR_DIST,
    UNKNOWN, EMPTY,
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
    return np.asarray([center for _n, center, _compact in clusters], dtype=np.float32)


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


def update_grid(prev: np.ndarray, image_rgb: np.ndarray, palette: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    时间序列更新：
    - EMPTY 永远是 EMPTY。
    - 上一帧是已知颜色：本帧仍匹配原颜色 -> 保留；否则 -> EMPTY。
      这是本程序处理挂穗/阴影污染的关键。
    - UNKNOWN 只在本帧非常明确匹配某个颜色时恢复。
    """
    h, w = image_rgb.shape[:2]
    x0, dx, y0, dy = scaled_grid_geometry(w, h)
    samples, positions = grid_samples(image_rgb)
    grid = prev.copy()
    removed = 0

    for rgb, (r, c) in zip(samples, positions):
        old = int(prev[r, c])
        cx = x0 + (c + 0.5) * dx
        cy = y0 + (r + 0.5) * dy

        if old == EMPTY:
            grid[r, c] = EMPTY
            continue

        if ui_covered(cx, cy, w, h):
            grid[r, c] = UNKNOWN
            continue

        if old > 0:
            ref = palette[old - 1]
            if float(np.linalg.norm(rgb - ref)) < KEEP_COLOR_DIST:
                grid[r, c] = old
            else:
                grid[r, c] = EMPTY
                removed += 1
        else:
            if len(palette) == 0:
                grid[r, c] = UNKNOWN
                continue
            d = np.linalg.norm(palette - rgb, axis=1)
            if float(d.min()) < 35.0:
                grid[r, c] = 1 + int(d.argmin())
            else:
                grid[r, c] = UNKNOWN

    return grid, removed


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


def reachable_components(grid: np.ndarray) -> Tuple[Dict[int, List[List[Tuple[int, int]]]], np.ndarray]:
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
                        if not visited[nr, nc] and int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            if touches_open:
                result[color].append(cells)

    return result, opened


def reachable_summary(grid: np.ndarray) -> Tuple[Dict[int, int], Dict[int, Counter]]:
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
