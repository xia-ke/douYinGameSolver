from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .adb import adb_capture_bgr, save_bgr, shot_stamp
from .config import (
    REF_W, REF_H,
    RAW_MONITOR_X1_N, RAW_MONITOR_X2_N,
    RAW_MONITOR_Y1_N, RAW_MONITOR_Y2_N,
    RAW_MONITOR_SAT_MAX, RAW_MONITOR_VAL_MIN,
    RAW_MONITOR_COMPONENT_MIN_AREA, RAW_MONITOR_COMPONENT_MAX_AREA,
    RAW_MONITOR_COMPONENT_MIN_H, RAW_MONITOR_COMPONENT_MAX_H,
    RAW_MONITOR_COMPONENT_MIN_W, RAW_MONITOR_COMPONENT_MAX_W,
    RAW_MONITOR_MAX_ASPECT,
    RAW_MONITOR_MASK_CHANGE_RATIO,
    RAW_MONITOR_EMPTY_CONFIRM_EXTRA_FRAMES,
)


@dataclass
class RawDigitMonitorState:
    mask: np.ndarray
    glyph_count: int
    foreground_pixels: int


def _raw_digit_roi(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1 = max(0, int(RAW_MONITOR_X1_N * w))
    x2 = min(w, int(RAW_MONITOR_X2_N * w))
    y1 = max(0, int(RAW_MONITOR_Y1_N * h))
    y2 = min(h, int(RAW_MONITOR_Y2_N * h))
    return image_bgr[y1:y2, x1:x2]


def parking_digit_monitor_state(
    image_bgr: np.ndarray,
) -> Tuple[Optional[RawDigitMonitorState], str]:
    """
    真正的纯数字视觉监控。

    这里只做：
      1) 固定停车数字窄带；
      2) HSV 提取高亮低饱和的数字白色填充；
      3) 用最基础的宽/高/面积几何过滤掉噪声；
      4) 保存这些字符的二值 mask。

    明确不做：
      - 0~9 分类
      - 模板匹配
      - 快速 OCR
      - 车身颜色
      - 车辆连通域
      - “1辆/2辆”的数字配对判断

    因此 20、13、9、7 都只是“若干白色字符像素”。
    """
    roi = _raw_digit_roi(image_bgr)
    if roi.size == 0:
        return None, "停车数字监控 ROI 为空"

    full_h, full_w = image_bgr.shape[:2]
    sx = full_w / REF_W
    sy = full_h / REF_H
    scale_area = sx * sy

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    raw = (
        (hsv[:, :, 1] < RAW_MONITOR_SAT_MAX)
        & (hsv[:, :, 2] > RAW_MONITOR_VAL_MIN)
    ).astype(np.uint8) * 255
    raw = cv2.medianBlur(raw, 3)

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)
    clean = np.zeros_like(raw)
    glyph_count = 0

    for label_id in range(1, n):
        x, y, w, h, area = map(int, stats[label_id])

        if area < RAW_MONITOR_COMPONENT_MIN_AREA * scale_area:
            continue
        if area > RAW_MONITOR_COMPONENT_MAX_AREA * scale_area:
            continue
        if h < RAW_MONITOR_COMPONENT_MIN_H * sy:
            continue
        if h > RAW_MONITOR_COMPONENT_MAX_H * sy:
            continue
        if w < RAW_MONITOR_COMPONENT_MIN_W * sx:
            continue
        if w > RAW_MONITOR_COMPONENT_MAX_W * sx:
            continue
        if w / max(1.0, float(h)) > RAW_MONITOR_MAX_ASPECT:
            continue

        clean[labels == label_id] = 255
        glyph_count += 1

    return RawDigitMonitorState(
        mask=clean,
        glyph_count=glyph_count,
        foreground_pixels=int(np.count_nonzero(clean)),
    ), ""


def parking_digit_mask_changed(
    previous: RawDigitMonitorState,
    current: RawDigitMonitorState,
) -> Tuple[bool, float]:
    """
    只比较数字像素变化。

    glyph_count 只用于快速判断“字符数量变了”；它不代表车辆数量。
    具体像素比较允许约 1px 抖动。
    """
    if previous.mask.shape != current.mask.shape:
        return True, 1.0

    if previous.glyph_count != current.glyph_count:
        return True, 1.0

    a = (previous.mask > 0).astype(np.uint8)
    b = (current.mask > 0).astype(np.uint8)

    na = int(a.sum())
    nb = int(b.sum())

    if na == 0 and nb == 0:
        return False, 0.0
    if na == 0 or nb == 0:
        return True, 1.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    ad = cv2.dilate(a, kernel, iterations=1)
    bd = cv2.dilate(b, kernel, iterations=1)

    missing_a = int(np.logical_and(a > 0, bd == 0).sum())
    missing_b = int(np.logical_and(b > 0, ad == 0).sum())
    unmatched_ratio = (missing_a + missing_b) / max(1.0, float(na + nb))

    return unmatched_ratio >= RAW_MONITOR_MASK_CHANGE_RATIO, float(unmatched_ratio)


def format_digit_monitor_state(state: RawDigitMonitorState) -> str:
    if state.glyph_count == 0:
        return "未检测到停车数字字符"
    return (
        f"检测到 {state.glyph_count} 个停车数字字符"
        f"（前景像素={state.foreground_pixels}）"
    )


def confirm_parking_digit_area_empty(
    first_frame: np.ndarray,
    *,
    serial: Optional[str],
    confirm_delay: float,
) -> Tuple[bool, np.ndarray, str]:
    last = first_frame

    for i in range(RAW_MONITOR_EMPTY_CONFIRM_EXTRA_FRAMES):
        if confirm_delay > 0:
            time.sleep(confirm_delay)

        last = adb_capture_bgr(serial)
        state, reason = parking_digit_monitor_state(last)

        if state is None:
            return False, last, f"第{i+2}次空数字确认失败: {reason}"

        if state.glyph_count > 0:
            return (
                False,
                last,
                f"第{i+2}次确认检测到数字字符：{format_digit_monitor_state(state)}",
            )

    total = 1 + RAW_MONITOR_EMPTY_CONFIRM_EXTRA_FRAMES
    return True, last, f"连续 {total} 帧都没有停车数字字符"


def wait_for_parking_idle(
    *,
    shots_dir: Path,
    serial: Optional[str],
    start_delay: float,
    check_interval: float,
    idle_timeout: float,
    max_failures: int,
    empty_settle_delay: float,
) -> Path:
    """
    点击后只看停车数字像素有没有变化。

    - 数字字符出现/消失/移动/形状变化 -> 重置静止计时
    - 连续 idle_timeout 秒数字像素不变 -> 本轮结束
    - 连续多帧完全没有数字字符 -> 立即结束

    监控过程中从不运行 OCR。
    """
    if start_delay > 0:
        print(f"等待 {start_delay:.1f}s，让本轮装载/分流启动...")
        time.sleep(start_delay)

    failures = 0
    previous: Optional[RawDigitMonitorState] = None
    last_activity: Optional[float] = None

    while True:
        frame = adb_capture_bgr(serial)
        current, reason = parking_digit_monitor_state(frame)
        now = time.monotonic()

        if current is None:
            failures += 1
            last_activity = now
            print(
                f"\r停车数字像素监控失败 {failures}/{max_failures}，"
                f"静止计时已重置: {reason}          ",
                end="",
                flush=True,
            )
            if failures >= max_failures:
                print()
                fail = shots_dir / f"digit_monitor_failed_{shot_stamp()}.png"
                save_bgr(fail, frame)
                raise RuntimeError(
                    f"停车数字像素监控连续失败，自动流程已暂停。最后截图: {fail}"
                )
            time.sleep(check_interval)
            continue

        failures = 0

        if current.glyph_count == 0:
            ok, confirm_frame, why = confirm_parking_digit_area_empty(
                frame,
                serial=serial,
                confirm_delay=empty_settle_delay,
            )
            if ok:
                print()
                monitor_end = shots_dir / f"monitor_end_no_digits_{shot_stamp()}.png"
                save_bgr(monitor_end, confirm_frame)
                print(
                    f"停车数字区域确认为空：{why}。"
                    f"判定本轮分流结束。监控快照: {monitor_end}"
                )
                return monitor_end

            print(
                f"\n停车数字区域单帧为空，但未通过连续确认：{why}。继续监控。"
            )
            confirm_state, _ = parking_digit_monitor_state(confirm_frame)
            previous = confirm_state
            last_activity = time.monotonic()
            if check_interval > 0:
                time.sleep(check_interval)
            continue

        if previous is None:
            previous = current
            last_activity = now
            print()
            print(f"停车数字像素监控基准: {format_digit_monitor_state(current)}")
            if check_interval > 0:
                time.sleep(check_interval)
            continue

        changed, ratio = parking_digit_mask_changed(previous, current)

        if changed:
            print()
            print(
                "检测到停车数字像素变化: "
                f"{format_digit_monitor_state(previous)} -> "
                f"{format_digit_monitor_state(current)} "
                f"(变化比例={ratio:.4f})"
            )
            last_activity = now
        else:
            if last_activity is None:
                last_activity = now

            idle_for = now - last_activity
            remain = max(0.0, idle_timeout - idle_for)
            print(
                f"\r停车数字像素无变化 {idle_for:5.1f}s / {idle_timeout:.1f}s "
                f"(还需 {remain:4.1f}s) "
                f"{format_digit_monitor_state(current)} "
                f"change={ratio:.4f}",
                end="",
                flush=True,
            )

            if idle_for >= idle_timeout:
                print()
                monitor_end = shots_dir / f"monitor_end_digits_stable_{shot_stamp()}.png"
                save_bgr(monitor_end, frame)
                print(
                    f"连续 {idle_timeout:.1f}s 停车数字像素没有变化，"
                    f"判定本轮分流结束。监控快照: {monitor_end}"
                )
                return monitor_end

        previous = current

        if check_interval > 0:
            time.sleep(check_interval)
