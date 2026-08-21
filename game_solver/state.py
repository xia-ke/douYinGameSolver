from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        version=np.asarray([STATE_VERSION], dtype=np.int32),
        palette=palette,
        grid=grid,
        turn=np.asarray([turn], dtype=np.int32),
        screen_size=np.asarray([image_w, image_h], dtype=np.int32),
        parking_empty_ref=parking_empty_ref.astype(np.uint8),
    )
    tmp.replace(path)


def load_state(path: Path) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int], np.ndarray]:
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
    if grid.shape != (GRID_ROWS, GRID_COLS):
        raise RuntimeError("状态文件网格尺寸与当前程序不一致，请使用 --reset。")
    return palette, grid, turn, (size[0], size[1]), parking_empty_ref
