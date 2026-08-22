from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import STATE_VERSION, GRID_ROWS, GRID_COLS


def save_state(
    path: Path,
    palette: np.ndarray,
    grid: np.ndarray,
    turn: int,
    image_w: int,
    image_h: int,
    parking_empty_ref: np.ndarray,
    grid_rgb_snapshot: Optional[np.ndarray] = None,
) -> None:
    """
    保存已提交的稳定状态。

    grid_rgb_snapshot 是 52x38x3 的格子中心 RGB 快照，用于下一次真实动作后的
    temporal-diff 因果同步。它是可选字段，因此仍兼容旧调用方。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")

    payload = dict(
        version=np.asarray([STATE_VERSION], dtype=np.int32),
        palette=palette,
        grid=grid,
        turn=np.asarray([turn], dtype=np.int32),
        screen_size=np.asarray([image_w, image_h], dtype=np.int32),
        parking_empty_ref=parking_empty_ref.astype(np.uint8),
    )
    if grid_rgb_snapshot is not None:
        snap = np.asarray(grid_rgb_snapshot, dtype=np.float32)
        if snap.shape != (GRID_ROWS, GRID_COLS, 3):
            raise ValueError(
                f"grid_rgb_snapshot 尺寸异常: {snap.shape}，"
                f"期望 {(GRID_ROWS, GRID_COLS, 3)}"
            )
        payload["grid_rgb_snapshot"] = snap

    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _load_state_core(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    Tuple[int, int],
    np.ndarray,
    Optional[np.ndarray],
]:
    # Windows 下必须显式关闭 npz，否则后续原子替换状态文件时可能遇到文件占用。
    with np.load(path) as data:
        version = int(data["version"][0]) if "version" in data else 0
        if version != STATE_VERSION:
            raise RuntimeError(
                f"状态文件版本 {version} 与当前程序版本 {STATE_VERSION} 不兼容。"
                "请在新局使用 --reset 重建 solver_state.npz。"
            )

        palette = data["palette"].astype(np.float32)
        grid = data["grid"].astype(np.int16)
        turn = int(data["turn"][0]) if "turn" in data else 0
        size = tuple(map(int, data["screen_size"].tolist()))
        parking_empty_ref = data["parking_empty_ref"].astype(np.uint8)

        grid_rgb_snapshot: Optional[np.ndarray]
        if "grid_rgb_snapshot" in data:
            snap = data["grid_rgb_snapshot"].astype(np.float32)
            if snap.shape == (GRID_ROWS, GRID_COLS, 3):
                grid_rgb_snapshot = snap
            else:
                grid_rgb_snapshot = None
        else:
            grid_rgb_snapshot = None

    if grid.shape != (GRID_ROWS, GRID_COLS):
        raise RuntimeError("状态文件网格尺寸与当前程序不一致，请使用 --reset。")

    return (
        palette,
        grid,
        turn,
        (size[0], size[1]),
        parking_empty_ref,
        grid_rgb_snapshot,
    )


def load_state(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int], np.ndarray]:
    """
    原接口保持不变，避免影响其它模块。
    """
    palette, grid, turn, size, parking_ref, _snapshot = _load_state_core(path)
    return palette, grid, turn, size, parking_ref


def load_state_with_grid_rgb(
    path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    int,
    Tuple[int, int],
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    engine 的新接口：额外读取上一份已提交稳定截图的格子 RGB 快照。
    """
    return _load_state_core(path)