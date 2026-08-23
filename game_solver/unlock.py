from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .adb import adb_capture_bgr, adb_tap, save_bgr, shot_stamp
from .config import (
    AD_CLOSE_X_N,
    AD_CLOSE_Y_N,
    SIXTH_SLOT_UNLOCK_X_N,
    SIXTH_SLOT_UNLOCK_Y_N,
    UNLOCK_AD_CLOSE_RETRY_COUNT,
    UNLOCK_AD_CLOSE_RETRY_INTERVAL,
    UNLOCK_AD_MIN_WATCH_SECONDS,
    UNLOCK_AD_OPEN_CHECK_DELAY,
    UNLOCK_AD_RETURN_DELAY,
    UNLOCK_GAME_FRONT_MIN_COLUMNS,
    UNLOCK_GAME_FRONT_MIN_SPAN_N,
    UNLOCK_GAME_THREE_COL_MIN_SPAN_N,
    UNLOCK_GAME_THREE_COL_MAX_GAP_RATIO,
    UNLOCK_BUTTON_SEARCH_X1_N,
    UNLOCK_BUTTON_SEARCH_X2_N,
    UNLOCK_BUTTON_SEARCH_Y1_N,
    UNLOCK_BUTTON_SEARCH_Y2_N,
    UNLOCK_BUTTON_DARK_GRAY_MAX,
    UNLOCK_BUTTON_MIN_W_REF,
    UNLOCK_BUTTON_MAX_W_REF,
    UNLOCK_BUTTON_MIN_H_REF,
    UNLOCK_BUTTON_MAX_H_REF,
    UNLOCK_BUTTON_MIN_AREA_REF,
    UNLOCK_BUTTON_MAX_AREA_REF,
    UNLOCK_BUTTON_MIN_ASPECT,
    UNLOCK_BUTTON_MAX_ASPECT,
)
from .vehicles import detect_front_centers

if TYPE_CHECKING:
    from .display import SolverDisplay


# 点击解锁后广告页面可能需要几秒才真正切换。
# 1.2s 只作为第一次检查时间，不再作为失败超时。
_AD_OPEN_TIMEOUT_SECONDS = 6.0
_AD_OPEN_POLL_INTERVAL_SECONDS = 0.5



def _find_unlock_button_center(
    image_bgr: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """
    动态寻找右侧“解锁”按钮的暗色圆角主体。

    这样不再依赖之前写错的固定坐标。
    同时限制高度 <= 185（940x2048基准），可排除第6车位里停着的车辆：
    实测黑色停车车主体高度约 203px，而“解锁”按钮暗色主体约 155px。
    """
    h, w = image_bgr.shape[:2]
    sx = w / 940.0
    sy = h / 2048.0
    area_scale = sx * sy

    x1 = max(0, int(round(UNLOCK_BUTTON_SEARCH_X1_N * w)))
    x2 = min(w, int(round(UNLOCK_BUTTON_SEARCH_X2_N * w)))
    y1 = max(0, int(round(UNLOCK_BUTTON_SEARCH_Y1_N * h)))
    y2 = min(h, int(round(UNLOCK_BUTTON_SEARCH_Y2_N * h)))

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = (gray < UNLOCK_BUTTON_DARK_GRAY_MAX).astype(np.uint8) * 255

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        aspect = bw / max(1.0, float(bh))

        if bw < UNLOCK_BUTTON_MIN_W_REF * sx:
            continue
        if bw > UNLOCK_BUTTON_MAX_W_REF * sx:
            continue
        if bh < UNLOCK_BUTTON_MIN_H_REF * sy:
            continue
        if bh > UNLOCK_BUTTON_MAX_H_REF * sy:
            continue
        if area < UNLOCK_BUTTON_MIN_AREA_REF * area_scale:
            continue
        if area > UNLOCK_BUTTON_MAX_AREA_REF * area_scale:
            continue
        if aspect < UNLOCK_BUTTON_MIN_ASPECT or aspect > UNLOCK_BUTTON_MAX_ASPECT:
            continue

        cx = x1 + x + bw / 2.0
        cy = y1 + y + bh / 2.0

        # 解锁按钮应位于停车区右侧。
        if cx < 0.70 * w:
            continue

        # 越接近预期尺寸/右侧位置分数越高。
        size_penalty = (
            abs(bw / max(1e-6, sx) - 115.0)
            + 0.5 * abs(bh / max(1e-6, sy) - 155.0)
        )
        score = cx / w * 100.0 - size_penalty
        candidates.append((score, int(round(cx)), int(round(cy))))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _score, cx, cy = candidates[0]
    return cx, cy


def _tap_normalized(
    image_bgr: np.ndarray,
    x_n: float,
    y_n: float,
    serial: Optional[str],
) -> Tuple[int, int]:
    h, w = image_bgr.shape[:2]
    x = int(round(x_n * w))
    y = int(round(y_n * h))
    adb_tap(x, y, serial)
    return x, y


def _looks_like_game_screen(image_bgr: np.ndarray) -> bool:
    """
    只用于广告解锁流程判断“是否已经回到新局游戏界面”。

    当前判定规则：
      - 4/5列关卡使用 >=35% 横向跨度要求；
      - 3列关卡允许较小的天然跨度（>=30%）；
      - 3列时额外要求相邻车辆中心间距近似等距，
        防止广告页三组不规则文字被误判为正常游戏队列。
    """
    try:
        centers = sorted(float(x) for x in detect_front_centers(image_bgr))
    except Exception:
        return False

    if len(centers) < UNLOCK_GAME_FRONT_MIN_COLUMNS:
        return False

    w = float(image_bgr.shape[1])
    if w <= 0:
        return False

    span_n = (centers[-1] - centers[0]) / w

    if len(centers) == 3:
        if span_n < UNLOCK_GAME_THREE_COL_MIN_SPAN_N:
            return False

        gaps = np.diff(np.asarray(centers, dtype=np.float32))
        if len(gaps) != 2 or float(gaps.min()) <= 0:
            return False

        gap_ratio = float(gaps.max() / gaps.min())
        return gap_ratio <= UNLOCK_GAME_THREE_COL_MAX_GAP_RATIO

    return span_n >= UNLOCK_GAME_FRONT_MIN_SPAN_N

def unlock_sixth_slot_at_game_start(
    *,
    serial: Optional[str],
    shots_dir: Path,
    min_watch_seconds: float = UNLOCK_AD_MIN_WATCH_SECONDS,
    display: Optional["SolverDisplay"] = None,
) -> bool:
    """
    每局开始时自动领取广告解锁的第6停车位。

    流程：
      1. 在新局界面点击右侧“解锁”；
      2. 确认广告已经切入；
      3. 从点击时刻累计至少 min_watch_seconds；
      4. 点击广告右上角 X；
      5. 确认第一排车辆重新出现后才允许求解器继续。

    返回：
      True  - 实际点击了解锁并完成广告返回；
      False - 启动时就没有检测到解锁按钮，视为第6位已经可用。

    注意：
      一旦点击前确实检测到了【解锁】按钮，后续绝不会因为
      “还能看到游戏界面”就跳过15秒广告等待。
    """
    before = adb_capture_bgr(serial)

    if not _looks_like_game_screen(before):
        fail = shots_dir / f"unlock_not_game_{shot_stamp()}.png"
        save_bgr(fail, before)
        raise RuntimeError(
            "准备检查第6停车位时没有检测到正常游戏第一排，"
            f"为避免误点其他界面已停止。截图: {fail}"
        )

    unlock_center = _find_unlock_button_center(before)
    if unlock_center is None:
        if display is not None:
            display.show(
                before,
                stage="启动检查",
                hint="未检测到【解锁】按钮；第6停车位视为已经可用。",
            )
        print("启动检查：当前画面没有检测到【解锁】按钮，视为第6停车位已经可用。")
        return False

    start_time = time.monotonic()
    ux, uy = unlock_center

    if display is not None:
        from .display import ClickMark
        display.show(
            before,
            stage="启动检查 · 解锁第6停车位",
            hint="检测到【解锁】按钮，准备点击并进入广告。",
            marks=(ClickMark(ux, uy, "UNLOCK"),),
        )

    adb_tap(ux, uy, serial)
    print(f"检测到【解锁】按钮并点击，坐标=({ux}, {uy})")

    # 广告切换可能需要数秒。
    # 1.2 秒只作为第一次检查，不再因为第一次截图仍看到【解锁】按钮就报错。
    if UNLOCK_AD_OPEN_CHECK_DELAY > 0:
        time.sleep(UNLOCK_AD_OPEN_CHECK_DELAY)

    open_deadline = start_time + _AD_OPEN_TIMEOUT_SECONDS
    after_open = adb_capture_bgr(serial)
    still_unlock = _find_unlock_button_center(after_open)

    while still_unlock is not None and time.monotonic() < open_deadline:
        elapsed_open = time.monotonic() - start_time
        if display is not None:
            from .display import ClickMark
            sx, sy = still_unlock
            display.show(
                after_open,
                stage="广告切换中",
                hint=(
                    f"已点击解锁 {elapsed_open:.1f}s；"
                    "等待【解锁】按钮消失，确认广告开始。"
                ),
                marks=(ClickMark(sx, sy, "WAIT"),),
            )
        remain_open = max(0.0, open_deadline - time.monotonic())
        print(
            f"\r广告正在切换，解锁按钮暂时仍可见："
            f"{elapsed_open:.1f}s / {_AD_OPEN_TIMEOUT_SECONDS:.1f}s "
            f"(最多再等 {remain_open:.1f}s)      ",
            end="",
            flush=True,
        )
        time.sleep(_AD_OPEN_POLL_INTERVAL_SECONDS)
        after_open = adb_capture_bgr(serial)
        still_unlock = _find_unlock_button_center(after_open)

    if still_unlock is not None:
        print()
        fail = shots_dir / f"unlock_click_not_effective_{shot_stamp()}.png"
        save_bgr(fail, after_open)
        raise RuntimeError(
            f"点击【解锁】后等待 {_AD_OPEN_TIMEOUT_SECONDS:.1f}s，"
            "按钮仍然存在，说明点击可能没有生效。"
            f"为避免误进入求解流程已停止。截图: {fail}"
        )

    print()
    print("检测到【解锁】按钮已经消失，确认广告流程开始。")

    if display is not None:
        display.show(
            after_open,
            stage="广告播放中",
            hint=f"广告已开始；至少观看 {min_watch_seconds:.1f}s 后自动关闭。",
        )

    elapsed = time.monotonic() - start_time
    remaining = max(0.0, min_watch_seconds - elapsed)
    if remaining > 0:
        print(
            f"解锁按钮已消失，按广告流程等待 {remaining:.1f}s "
            "后再尝试关闭..."
        )
        time.sleep(remaining)

    current = adb_capture_bgr(serial)
    h, w = current.shape[:2]
    cx_preview = int(round(AD_CLOSE_X_N * w))
    cy_preview = int(round(AD_CLOSE_Y_N * h))
    if display is not None:
        from .display import ClickMark
        display.show(
            current,
            stage="广告关闭",
            hint="广告时间已到，点击右上角关闭按钮。",
            marks=(ClickMark(cx_preview, cy_preview, "CLOSE"),),
        )

    cx, cy = _tap_normalized(
        current,
        AD_CLOSE_X_N,
        AD_CLOSE_Y_N,
        serial,
    )
    print(f"广告时间已到，点击右上角关闭，坐标=({cx}, {cy})")

    if UNLOCK_AD_RETURN_DELAY > 0:
        time.sleep(UNLOCK_AD_RETURN_DELAY)

    returned = adb_capture_bgr(serial)
    if _looks_like_game_screen(returned):
        if display is not None:
            display.show(
                returned,
                stage="解锁完成",
                hint="广告已关闭并返回游戏；准备开始自动分析。",
            )
        print("第6停车位广告解锁完成，已返回游戏。")
        return True

    # 广告的关闭按钮有时会晚一点真正生效。
    # 只有“仍未回到游戏界面”时才重试右上角关闭位置。
    for attempt in range(1, UNLOCK_AD_CLOSE_RETRY_COUNT + 1):
        if UNLOCK_AD_CLOSE_RETRY_INTERVAL > 0:
            time.sleep(UNLOCK_AD_CLOSE_RETRY_INTERVAL)

        current = adb_capture_bgr(serial)
        if _looks_like_game_screen(current):
            if display is not None:
                display.show(
                    current,
                    stage="解锁完成",
                    hint="广告已关闭并返回游戏；准备开始自动分析。",
                )
            print("第6停车位广告解锁完成，已返回游戏。")
            return True

        cx, cy = _tap_normalized(
            current,
            AD_CLOSE_X_N,
            AD_CLOSE_Y_N,
            serial,
        )
        print(
            f"广告关闭尚未返回游戏，重试 {attempt}/"
            f"{UNLOCK_AD_CLOSE_RETRY_COUNT}，坐标=({cx}, {cy})"
        )

        if UNLOCK_AD_RETURN_DELAY > 0:
            time.sleep(UNLOCK_AD_RETURN_DELAY)

        returned = adb_capture_bgr(serial)
        if _looks_like_game_screen(returned):
            if display is not None:
                display.show(
                    returned,
                    stage="解锁完成",
                    hint="广告已关闭并返回游戏；准备开始自动分析。",
                )
            print("第6停车位广告解锁完成，已返回游戏。")
            return True

    fail = shots_dir / f"unlock_ad_close_failed_{shot_stamp()}.png"
    save_bgr(fail, returned)
    raise RuntimeError(
        "广告播放后未能确认返回游戏界面，自动流程已停止，"
        f"避免在广告界面继续误点。最后截图: {fail}"
    )