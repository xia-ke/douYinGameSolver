from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import GRID_COLS, GRID_ROWS, STATE_VERSION


@dataclass(frozen=True)
class TrustedSessionState:
    """
    Previous trusted observation context for the next stable-frame analysis.

    ``previous_trusted_grid`` is historical context only. It is never the
    authoritative board for a newly captured frame; ``observe_board()`` must
    reconstruct current spatial state first and may consult this context only
    for current UNKNOWN resolution / invariant validation.
    """

    palette: np.ndarray
    previous_trusted_grid: np.ndarray
    turn: int
    screen_size: Tuple[int, int]
    parking_empty_ref: np.ndarray
    previous_trusted_grid_rgb: Optional[np.ndarray] = None


def _validated_state(state: TrustedSessionState) -> TrustedSessionState:
    palette = np.asarray(state.palette, dtype=np.float32)
    grid = np.asarray(state.previous_trusted_grid, dtype=np.int16)
    if grid.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(
            f"previous_trusted_grid 尺寸异常: {grid.shape}，"
            f"期望 {(GRID_ROWS, GRID_COLS)}"
        )

    size = tuple(map(int, state.screen_size))
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"screen_size 非法: {state.screen_size!r}")

    parking_ref = np.asarray(state.parking_empty_ref, dtype=np.uint8)

    snapshot: Optional[np.ndarray] = None
    if state.previous_trusted_grid_rgb is not None:
        snapshot = np.asarray(state.previous_trusted_grid_rgb, dtype=np.float32)
        if snapshot.shape != (GRID_ROWS, GRID_COLS, 3):
            raise ValueError(
                f"previous_trusted_grid_rgb 尺寸异常: {snapshot.shape}，"
                f"期望 {(GRID_ROWS, GRID_COLS, 3)}"
            )

    return TrustedSessionState(
        palette=palette,
        previous_trusted_grid=grid,
        turn=int(state.turn),
        screen_size=(size[0], size[1]),
        parking_empty_ref=parking_ref,
        previous_trusted_grid_rgb=snapshot,
    )


def save_state(path: Path, state: TrustedSessionState) -> None:
    """Atomically persist exactly one previous-trusted-context schema."""
    state = _validated_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")

    payload = dict(
        version=np.asarray([STATE_VERSION], dtype=np.int32),
        palette=state.palette,
        previous_trusted_grid=state.previous_trusted_grid,
        turn=np.asarray([state.turn], dtype=np.int32),
        screen_size=np.asarray(state.screen_size, dtype=np.int32),
        parking_empty_ref=state.parking_empty_ref,
    )
    if state.previous_trusted_grid_rgb is not None:
        payload["previous_trusted_grid_rgb"] = state.previous_trusted_grid_rgb

    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_state(path: Path) -> TrustedSessionState:
    """
    Load the canonical previous-trusted-context schema.

    State v3 and earlier are deliberately not migrated internally. Their field
    name ``grid`` encouraged treating persisted history as current authority;
    use ``--reset`` once after this cutover instead of retaining dual readers.
    """
    with np.load(path) as data:
        version = int(data["version"][0]) if "version" in data else 0
        if version != STATE_VERSION:
            raise RuntimeError(
                f"状态文件版本 {version} 与当前程序版本 {STATE_VERSION} 不兼容。"
                "Issue 004 已切换到 previous trusted context 状态格式；"
                "请在新局使用 --reset 重建 solver_state.npz。"
            )

        required = {
            "palette",
            "previous_trusted_grid",
            "turn",
            "screen_size",
            "parking_empty_ref",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(
                "状态文件缺少当前格式字段: " + ", ".join(missing)
                + "。请使用 --reset 重建 solver_state.npz。"
            )

        snapshot = None
        if "previous_trusted_grid_rgb" in data:
            snapshot = data["previous_trusted_grid_rgb"].astype(np.float32)

        state = TrustedSessionState(
            palette=data["palette"].astype(np.float32),
            previous_trusted_grid=data["previous_trusted_grid"].astype(np.int16),
            turn=int(data["turn"][0]),
            screen_size=tuple(map(int, data["screen_size"].tolist())),
            parking_empty_ref=data["parking_empty_ref"].astype(np.uint8),
            previous_trusted_grid_rgb=snapshot,
        )

    try:
        return _validated_state(state)
    except ValueError as exc:
        raise RuntimeError(
            f"状态文件内容与当前格式不一致: {exc}。请使用 --reset。"
        ) from exc
