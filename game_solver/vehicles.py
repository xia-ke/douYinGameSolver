from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import *
from .models import Car
from .ocr import (
    read_number_at, read_preview_number_at, recognize_digit,
)

@dataclass
class ParkingDigitComponent:
    label_id: int
    x: int
    y: int
    w: int
    h: int
    area: int
    digit: Optional[int]
    score: float
    margin: float
    hole_count: int = 0
    ocr_source: str = ""


@dataclass
class ParkingDigitGroup:
    components: Tuple[ParkingDigitComponent, ...]
    value: Optional[int]
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float


def nearest_palette(rgb: np.ndarray, palette: np.ndarray, threshold: float = CAR_COLOR_DIST) -> Optional[int]:
    if palette is None or len(palette) == 0:
        return None
    d = np.linalg.norm(palette - rgb, axis=1)
    if float(d.min()) > threshold:
        return None
    return 1 + int(d.argmin())


def car_color_at(image_rgb: np.ndarray, cx: float, cy: float, palette: np.ndarray) -> Optional[int]:
    """取车头上方一块纯色区域；示例里它与小色块主色几乎一致。"""
    h, w = image_rgb.shape[:2]
    sx = w / REF_W
    sy = h / REF_H
    x1 = max(0, int(cx - 30 * sx))
    x2 = min(w, int(cx + 30 * sx))
    y1 = max(0, int(cy - 85 * sy))
    y2 = min(h, int(cy - 60 * sy))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image_rgb[y1:y2, x1:x2, :3]
    rgb = np.median(patch.reshape(-1, 3), axis=0)
    return nearest_palette(rgb.astype(np.float32), palette)


def car_body_rgb_at(image_rgb: np.ndarray, cx: float, cy: float) -> Optional[np.ndarray]:
    """读取车辆上方相对纯净的车身颜色，用于动态追加新颜色。"""
    h, w = image_rgb.shape[:2]
    sx = w / REF_W
    sy = h / REF_H
    x1 = max(0, int(cx - 30 * sx))
    x2 = min(w, int(cx + 30 * sx))
    y1 = max(0, int(cy - 85 * sy))
    y2 = min(h, int(cy - 60 * sy))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image_rgb[y1:y2, x1:x2, :3].reshape(-1, 3)
    if len(patch) == 0:
        return None
    # 用中位数抗高光，并要求区域整体比较纯，避免把背景误追加成新颜色。
    med = np.median(patch, axis=0).astype(np.float32)
    mad = np.median(np.abs(patch - med[None, :]), axis=0)
    if float(np.max(mad)) > 28.0:
        return None
    return med



def detect_front_centers(image_bgr: np.ndarray) -> List[float]:
    """
    动态检测第一排可点击车辆的横坐标。

    关键点：不再假定固定 4 列。
    只利用第一排车顶白色数字的几何结构寻找每辆车中心；
    当前已在 4 列与 5 列真实截图上验证。

    返回按 x 从左到右排序的像素中心坐标。
    """
    h, w = image_bgr.shape[:2]
    x1 = max(0, int(FRONT_DIGIT_SCAN_X1_N * w))
    x2 = min(w, int(FRONT_DIGIT_SCAN_X2_N * w))
    y1 = max(0, int(FRONT_DIGIT_SCAN_Y1_N * h))
    y2 = min(h, int(FRONT_DIGIT_SCAN_Y2_N * h))

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    sx = w / REF_W
    sy = h / REF_H
    scale_area = sx * sy

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    raw = (
        (hsv[:, :, 1] < FRONT_DIGIT_SAT_MAX)
        & (hsv[:, :, 2] > FRONT_DIGIT_VAL_MIN)
    ).astype(np.uint8) * 255
    raw = cv2.medianBlur(raw, 3)

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)

    comps: List[Tuple[int, int, int, int, int, int]] = []
    for label_id in range(1, n):
        x, y, cw, ch, area = map(int, stats[label_id])

        if area < FRONT_DIGIT_COMPONENT_MIN_AREA * scale_area:
            continue
        if area > FRONT_DIGIT_COMPONENT_MAX_AREA * scale_area:
            continue
        if ch < FRONT_DIGIT_COMPONENT_MIN_H * sy:
            continue
        if ch > FRONT_DIGIT_COMPONENT_MAX_H * sy:
            continue
        if cw < FRONT_DIGIT_COMPONENT_MIN_W * sx:
            continue
        if cw > FRONT_DIGIT_COMPONENT_MAX_W * sx:
            continue
        if cw / max(1.0, float(ch)) > FRONT_DIGIT_MAX_ASPECT:
            continue

        comps.append((x + x1, y + y1, cw, ch, area, label_id))

    pair_candidates: List[Tuple[float, int, int]] = []

    for ai in range(len(comps)):
        for bi in range(ai + 1, len(comps)):
            a = comps[ai]
            b = comps[bi]
            left, right = (a, b) if a[0] <= b[0] else (b, a)

            hsim = min(left[3], right[3]) / max(1.0, float(max(left[3], right[3])))
            if hsim < FRONT_DIGIT_PAIR_MIN_HEIGHT_SIM:
                continue

            cy1 = left[1] + left[3] / 2.0
            cy2 = right[1] + right[3] / 2.0
            yoff = abs(cy1 - cy2)
            if yoff > FRONT_DIGIT_PAIR_MAX_Y_OFFSET_RATIO * max(left[3], right[3]) + 2.0 * sy:
                continue

            gap = right[0] - (left[0] + left[2])
            if gap < -2.0 * sx or gap > FRONT_DIGIT_PAIR_MAX_GAP * sx:
                continue

            span = right[0] + right[2] - left[0]
            if span > FRONT_DIGIT_PAIR_MAX_SPAN * sx:
                continue

            # 同一车的两位数字通常高度接近、水平对齐、间距很小。
            score = (
                5.0 * hsim
                - 0.10 * (yoff / max(1.0, sy))
                - 0.02 * abs(gap / max(1e-6, sx) - 7.0)
            )
            pair_candidates.append((score, ai, bi))

    pair_candidates.sort(reverse=True)
    used: set[int] = set()
    centers: List[float] = []

    for _score, ai, bi in pair_candidates:
        if ai in used or bi in used:
            continue
        a, b = comps[ai], comps[bi]
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        center = (left[0] + right[0] + right[2]) / 2.0
        centers.append(float(center))
        used.add(ai)
        used.add(bi)

    # 未配对组件可能就是个位数车辆。
    # 用快速数字模板确认“确实像1~9”，避免把车身高光当成一列。
    for idx, comp in enumerate(comps):
        if idx in used:
            continue

        ax, ay, cw, ch, _area, label_id = comp
        lx = ax - x1
        ly = ay - y1
        digit_mask = (
            labels[ly:ly + ch, lx:lx + cw] == label_id
        ).astype(np.uint8)

        digit_result = recognize_digit(
            digit_mask,
            source="front-column-singleton",
        )
        digit = digit_result.digit if digit_result.accepted else None
        if digit is None or digit == 0 or digit_result.score < 0.72:
            continue

        centers.append(float(ax + cw / 2.0))

    centers.sort()

    # 去掉极近重复中心；正常列间距远大于 60px。
    deduped: List[float] = []
    for cx in centers:
        if not deduped or abs(cx - deduped[-1]) > 45.0 * sx:
            deduped.append(cx)

    return deduped


def read_front_numbers_at_centers(
    image_bgr: np.ndarray,
    centers_x: Sequence[float],
) -> Dict[int, Optional[int]]:
    h, _w = image_bgr.shape[:2]
    cy = FRONT_Y_N * h
    out: Dict[int, Optional[int]] = {}
    for i, cx in enumerate(centers_x, 1):
        out[i] = read_number_at(image_bgr, float(cx), cy)
    return out


def read_preview_numbers_at_centers(
    image_bgr: np.ndarray,
    centers_x: Sequence[float],
) -> Dict[int, Optional[int]]:
    """读取第二排浅色车辆数字；低置信度返回 None。"""
    h, _w = image_bgr.shape[:2]
    cy = NEXT_Y_N * h
    out: Dict[int, Optional[int]] = {}
    for i, cx in enumerate(centers_x, 1):
        out[i] = read_preview_number_at(image_bgr, float(cx), cy)
    return out


def read_front_numbers(image_bgr: np.ndarray) -> Dict[int, Optional[int]]:
    """动态发现第一排列中心后读取数字。"""
    centers = detect_front_centers(image_bgr)
    return read_front_numbers_at_centers(image_bgr, centers)


def extend_palette_from_front_numbers(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    front_numbers: Dict[int, Optional[int]],
    centers_x: Sequence[float],
) -> Tuple[np.ndarray, int]:
    """
    运行中动态追加新颜色。
    列中心完全来自当前截图的动态检测结果。
    """
    h, _w = image_rgb.shape[:2]
    centers: List[np.ndarray] = [p.astype(np.float32) for p in palette]
    added = 0
    cy = FRONT_Y_N * h

    for i, cx in enumerate(centers_x, 1):
        if front_numbers.get(i) is None:
            continue

        rgb = car_body_rgb_at(image_rgb, float(cx), cy)
        if rgb is None:
            continue

        if centers:
            d = np.linalg.norm(
                np.asarray(centers, dtype=np.float32) - rgb[None, :],
                axis=1,
            )
            if float(d.min()) < NEW_COLOR_APPEND_DIST:
                continue

        centers.append(rgb)
        added += 1

    if not centers:
        return palette, 0

    return np.asarray(centers, dtype=np.float32), added


def detect_front_and_next(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    palette: np.ndarray,
    centers_x: Sequence[float],
    front_numbers: Optional[Dict[int, Optional[int]]] = None,
    *,
    read_next_numbers: bool = True,
) -> Tuple[List[Car], List[Car]]:
    """
    当前截图有多少个第一排数字组，就有多少个当前队列列。

    第一排：读取正常高对比度数字；
    第二排：读取浅色预览数字。预览 OCR 低置信度时 remain=None，
    策略仍可知道颜色，但不会把未知容量用于确定性二层安全证明。
    """
    h, _w = image_rgb.shape[:2]
    front: List[Car] = []
    nxt: List[Car] = []

    if front_numbers is None:
        front_numbers = read_front_numbers_at_centers(image_bgr, centers_x)

    next_numbers: Dict[int, Optional[int]] = {}
    if read_next_numbers:
        next_numbers = read_preview_numbers_at_centers(image_bgr, centers_x)

    cy = FRONT_Y_N * h
    ncy = NEXT_Y_N * h

    for i, cx in enumerate(centers_x, 1):
        cx = float(cx)
        color = car_color_at(image_rgb, cx, cy, palette)
        remain = front_numbers.get(i)

        if color is not None or remain is not None:
            front.append(Car("front", i, color, remain, cx, cy))

        ncolor = car_color_at(image_rgb, cx, ncy, palette)
        nremain = next_numbers.get(i) if read_next_numbers else None
        if ncolor is not None or nremain is not None:
            nxt.append(Car("next", i, ncolor, nremain, cx, ncy))

    return front, nxt


def _parking_digit_roi_bounds(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = image_bgr.shape[:2]
    x1 = max(0, int(PARK_DIGIT_MONITOR_X1_N * w))
    x2 = min(w, int(PARK_DIGIT_MONITOR_X2_N * w))
    y1 = max(0, int(PARK_DIGIT_MONITOR_Y1_N * h))
    y2 = min(h, int(PARK_DIGIT_MONITOR_Y2_N * h))
    return x1, y1, x2, y2


def _extract_parking_digit_components(
    image_bgr: np.ndarray,
    *,
    full_ocr: bool,
) -> Tuple[List[ParkingDigitComponent], np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    从停车数字区域提取单个数字候选。
    full_ocr=False 时只做快速模板形状过滤，用于高频监控。
    full_ocr=True 时模糊数字允许走慢速兜底，用于完整决策。
    """
    x1, y1, x2, y2 = _parking_digit_roi_bounds(image_bgr)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return [], np.zeros((0, 0), np.int32), np.zeros((0, 0), np.uint8), (x1, y1, x2, y2)

    H, W = roi.shape[:2]
    full_h, full_w = image_bgr.shape[:2]
    sx = full_w / REF_W
    sy = full_h / REF_H
    scale_area = sx * sy

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    raw = (
        (hsv[:, :, 1] < PARK_DIGIT_MONITOR_SAT_MAX)
        & (hsv[:, :, 2] > PARK_DIGIT_MONITOR_VAL_MIN)
    ).astype(np.uint8) * 255
    raw = cv2.medianBlur(raw, 3)

    n, labels, stats, _cents = cv2.connectedComponentsWithStats(raw, 8)
    comps: List[ParkingDigitComponent] = []

    for label_id in range(1, n):
        x, y, cw, ch, area = map(int, stats[label_id])

        if area < PARK_DIGIT_COMPONENT_MIN_AREA * scale_area:
            continue
        if area > PARK_DIGIT_COMPONENT_MAX_AREA * scale_area:
            continue
        if ch < PARK_DIGIT_COMPONENT_MIN_H * sy or ch > PARK_DIGIT_COMPONENT_MAX_H * sy:
            continue
        if cw < PARK_DIGIT_COMPONENT_MIN_W * sx or cw > PARK_DIGIT_COMPONENT_MAX_W * sx:
            continue
        if cw / max(1.0, float(ch)) > 1.20:
            continue

        mask = (labels[y:y + ch, x:x + cw] == label_id).astype(np.uint8)
        digit_result = recognize_digit(
            mask,
            source=("parking-full" if full_ocr else "parking-fast"),
        )
        digit = digit_result.digit if digit_result.accepted else None
        score = digit_result.score
        margin = digit_result.margin

        if digit is None or score < PARK_DIGIT_SHAPE_MIN_SCORE:
            continue

        comps.append(ParkingDigitComponent(
            label_id=label_id,
            x=x,
            y=y,
            w=cw,
            h=ch,
            area=area,
            digit=digit,
            score=float(score),
            margin=float(margin),
            hole_count=int(digit_result.hole_count),
            ocr_source=digit_result.source,
        ))

    return comps, labels, raw, (x1, y1, x2, y2)


def _group_parking_digit_components(
    comps: Sequence[ParkingDigitComponent],
    *,
    sx: float,
    sy: float,
) -> List[Tuple[ParkingDigitComponent, ...]]:
    """
    先把明显属于同一辆车的左右两位数字配对，再把剩余有效数字作为个位数车辆。
    两辆不同停车车的数字中心相距远大于同一辆车的十位/个位间距。
    """
    pair_candidates: List[Tuple[float, int, int]] = []

    for ai in range(len(comps)):
        for bi in range(ai + 1, len(comps)):
            a, b = comps[ai], comps[bi]
            left, right = (a, b) if a.x <= b.x else (b, a)

            hsim = min(left.h, right.h) / max(1.0, float(max(left.h, right.h)))
            if hsim < PARK_DIGIT_PAIR_MIN_HEIGHT_SIM:
                continue

            cy1 = left.y + left.h / 2.0
            cy2 = right.y + right.h / 2.0
            yoff = abs(cy1 - cy2)
            max_h = max(left.h, right.h)
            if yoff > PARK_DIGIT_PAIR_MAX_Y_OFFSET_RATIO * max_h + 2.0 * sy:
                continue

            gap = right.x - (left.x + left.w)
            if gap < -2.0 * sx or gap > PARK_DIGIT_PAIR_MAX_GAP * sx:
                continue

            span = right.x + right.w - left.x
            if span > PARK_DIGIT_PAIR_MAX_SPAN * sx:
                continue

            # 第一位不能是 0；游戏个位数也不会显示前导 0。
            if left.digit == 0:
                continue

            score = (
                5.0 * hsim
                - 0.10 * (yoff / max(1.0, sy))
                - 0.025 * abs(gap / max(1e-6, sx) - 7.0)
                + 0.6 * min(left.score, right.score)
            )
            pair_candidates.append((score, ai, bi))

    pair_candidates.sort(reverse=True)
    used: set[int] = set()
    groups: List[Tuple[ParkingDigitComponent, ...]] = []

    for _score, ai, bi in pair_candidates:
        if ai in used or bi in used:
            continue
        a, b = comps[ai], comps[bi]
        pair = (a, b) if a.x <= b.x else (b, a)
        groups.append(pair)
        used.add(ai)
        used.add(bi)

    # 未配对的高质量数字就是个位数车辆（1..9）。
    for i, comp in enumerate(comps):
        if i in used:
            continue
        if comp.digit is None or comp.digit == 0:
            continue
        # 单数字比两位数更容易受 UI 干扰，因此要求略高一点的形状分数。
        if comp.score < max(PARK_DIGIT_SHAPE_MIN_SCORE, 0.74):
            continue
        groups.append((comp,))

    groups.sort(key=lambda g: min(c.x for c in g))
    return groups


def extract_parking_digit_groups(
    image_bgr: np.ndarray,
    *,
    full_ocr: bool,
) -> Tuple[List[ParkingDigitGroup], np.ndarray]:
    comps, labels, _raw, bounds = _extract_parking_digit_components(
        image_bgr, full_ocr=full_ocr
    )
    x0, y0, _x2, _y2 = bounds
    h, w = image_bgr.shape[:2]
    sx = w / REF_W
    sy = h / REF_H

    raw_groups = _group_parking_digit_components(comps, sx=sx, sy=sy)
    clean = np.zeros(labels.shape, dtype=np.uint8)
    groups: List[ParkingDigitGroup] = []

    for g in raw_groups:
        ordered = tuple(sorted(g, key=lambda c: c.x))
        digits = [c.digit for c in ordered]

        value: Optional[int]
        if any(d is None for d in digits):
            value = None
        elif len(digits) == 1:
            value = int(digits[0])
        else:
            value = int(digits[0]) * 10 + int(digits[1])

        lx = min(c.x for c in ordered)
        ty = min(c.y for c in ordered)
        rx = max(c.x + c.w for c in ordered)
        by = max(c.y + c.h for c in ordered)

        for c in ordered:
            clean[labels == c.label_id] = 255

        groups.append(ParkingDigitGroup(
            components=ordered,
            value=value,
            x1=x0 + lx,
            y1=y0 + ty,
            x2=x0 + rx,
            y2=y0 + by,
            cx=x0 + (lx + rx) / 2.0,
            cy=y0 + (ty + by) / 2.0,
        ))

    return groups, clean


def _car_color_from_digit_group(
    image_rgb: np.ndarray,
    palette: np.ndarray,
    group: ParkingDigitGroup,
) -> Optional[int]:
    """
    数字就是停车车最稳定的锚点。
    在数字正上/下及左右近邻取多个小块，选择最接近 palette 的车身主色投票。
    """
    if palette is None or len(palette) == 0:
        return None

    h, w = image_rgb.shape[:2]
    sx = w / REF_W
    sy = h / REF_H
    gh = max(1.0, float(group.y2 - group.y1))
    gw = max(1.0, float(group.x2 - group.x1))
    half = max(4, int(round(6 * (sx + sy) / 2.0)))

    probes = [
        (group.cx, group.cy - 1.05 * gh),
        (group.cx, group.cy + 1.05 * gh),
        (group.cx - max(0.75 * gw, 28 * sx), group.cy + 0.15 * gh),
        (group.cx + max(0.75 * gw, 28 * sx), group.cy + 0.15 * gh),
    ]

    votes: List[Tuple[int, float]] = []
    for px, py in probes:
        ix, iy = int(round(px)), int(round(py))
        xa, xb = max(0, ix - half), min(w, ix + half + 1)
        ya, yb = max(0, iy - half), min(h, iy + half + 1)
        patch = image_rgb[ya:yb, xa:xb, :3]
        if patch.size == 0:
            continue
        med = np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)
        d = np.linalg.norm(palette - med[None, :], axis=1)
        idx = int(d.argmin())
        dist = float(d[idx])
        if dist <= CAR_COLOR_DIST:
            votes.append((idx + 1, dist))

    if not votes:
        return None

    counts = Counter(v for v, _d in votes)
    best_count = max(counts.values())
    tied = [c for c, n in counts.items() if n == best_count]
    if len(tied) == 1:
        return tied[0]

    # 平票时取平均距离更小的颜色。
    best_color = min(
        tied,
        key=lambda c: np.mean([d for cc, d in votes if cc == c]),
    )
    return int(best_color)


def detect_parked(
    image_rgb: np.ndarray,
    image_bgr: np.ndarray,
    palette: np.ndarray,
    *,
    read_numbers: bool = True,
) -> List[Car]:
    """
    v4：停车车完全由数字锚点检测，不再做车身颜色大连通域。
    支持个位数与两位数；稳定画面下每个数字组对应一辆停车车。
    """
    groups, _mask = extract_parking_digit_groups(
        image_bgr, full_ocr=bool(read_numbers)
    )

    cars: List[Car] = []
    for group in groups:
        remain = group.value if read_numbers else None
        if read_numbers and (remain is None or remain <= 0 or remain > 99):
            continue

        color = _car_color_from_digit_group(image_rgb, palette, group)
        digit_h = max(1.0, float(group.y2 - group.y1))
        scale_hint = max(
            0.55,
            min(1.35, digit_h / max(1.0, 35.0 * (image_bgr.shape[0] / REF_H))),
        )
        cars.append(Car(
            "parked",
            None,
            color,
            remain,
            group.cx,
            group.cy + 45.0 * (image_bgr.shape[0] / REF_H) * scale_hint,
            scale_hint,
        ))

    cars.sort(key=lambda c: c.x)
    return cars


def parking_roi(image_bgr: np.ndarray) -> np.ndarray:
    """
    仅用于兼容 v3 状态文件中的 parking_empty_ref 字段。
    v4 的停车占用/车辆识别已经完全改为数字锚点，不再使用空场差分。
    """
    h, w = image_bgr.shape[:2]
    x1, x2 = int(PARK_X1_N * w), int(PARK_X2_N * w)
    y1, y2 = int(PARK_Y1_N * h), int(PARK_Y2_N * h)
    return image_bgr[y1:y2, x1:x2].copy()