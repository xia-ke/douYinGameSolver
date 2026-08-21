from __future__ import annotations

import base64
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import REF_W, REF_H


# v5.4
# ----
# 正常第一排/停车区数字使用游戏自己的高对比度字体。
# 旧实现主要拿 OpenCV 内置字体去拟合，真实截图已经反复出现：
#   30 -> 39
#   20 -> 29
#   31 -> 37
#   11 -> 77
#   1  -> 7
#
# 以下 32x48 二值原型来自本游戏真实高对比度数字字形。
# 只保存归一化后的 0/1 字形，不保存截图。
_GAME_DIGIT_B64 = {
    0: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAH/wAAB/8AAD//wAB//+AA///wAf//+AH///wD///8A////AP+B/4P/AP+D/wD/g/8A/4P/AP+D/wB/g/8Af8P/AH/D/wB/w/8Af8P/AH/D/wB/w/8Af8P/AH/D/wB/w/8Af4P/AP+D/wD/g/8A/4P/gP+A/8H/gP///wD///8Af///AH///gA///wAH//4AAf/8AAB/4AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    1: "AAAAAAAAAAAAAH/AAAB/wAAf/+AD///gD///4A///+AP///gD///4A///+AP///gD///4A///+ADB//gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAH/4AAB/+AAAf/gAAD/wAAAAAAAAAAA",
    2: "AAAAAAAAAAAAAAAAAAAAAAA/8AAAP/AAA///gAf//8AH///gB///8Af///gH///4B+H/+AAAP/gAAD/4AAAf/AAAH/wAAB/4AAAf+AAAP/gAAD/4AAB/8AAA//AAAf/gAAH/4AAD/8AAB//AAB//gAA//gAAf/wAAf/wAAP/4AAH/8AAD//AAA//wAA////4P////D////w////8P////D////w////8D///+AAAAAAAAAAAAAAAAAAAAAAAAAAA",
    3: "AAAAAAAAAAAAAAAAA//4AAP/+AAH//4AD///wA///+AP///wD///8A////AH///wAAD/+AAAf/gAAD/4AAA/+AAAP/gAAD/wAAD/8AAD//AA///gAf//wAH//4AB///AAf//wAH//+AA///wAAf/+AAAf/gAAD/4AAAf/AAAH/wAAB/8AAA//AAAf/wOA//4P///+D////g////4P///8D///+A////AP//+AA//4AAAAAAAAAAAAAAAAAAAAAAA",
    4: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/gAAA/4AAAf/AAAH/4AAD/+AAB//gAAf/4AAP/+AAH//gAB//4AA//+AAf//gAH//4AD/P+AB/j/gAfw/4AP8P+AH+D/gB/A/4A/wP+A/8H/wP///+D////w////8P////D////w////8P////A////gAAH/wAAA/4AAAP+AAAD/gAAA/4AAAH8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    5: "AAAAAAAAAAAD///gA///4Af///AH///wB///8A////AP///wD///8A////AP///gD/4AAA/+AAAP+AAAD/gAAA/4AAA//gAAP//+AD///gA////AP///4D////A////4P///+D////gP///4AAB//AAAP/wAAD/8AAA//AAAP/wAAD/8AAB//AAA//wAAP/8D///+D////g////wP///8D////A////gP///AA///AAH/4AAAAAAAAAAAAAAAAAA",
    6: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/AAAD/4AAH/+AAD//gAB//4AA//+AAf//gAP//4AH//8AB//wAA//AAAP/gAAD//4AB///4Af///AH///4B////Af///4P///+D////w////8P/wf/D/4D/w/+A/8P/gP/D/4D/wf+B/8H/w//B////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    7: "AAAAAAAAAAAAAAAAD///+A////g////8P////D////w////8P////D////w////8D////AAAP/wAAA/8AAAP/AAAP/wAAD/4AAB/+AAAf/gAAP/wAAD/8AAA//AAAf/gAAH/4AAB/8AAB//AAAf/wAAH/8AAD/8AAA//AAAf/gAAH/4AAB/+AAA//AAAP/wAAD/4AAB/+AAAf/gAAH/4AAH/8AAB//AAA//AAAH/gAAAAAAAAAAAAAAAAAAAAAAA",
    8: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAP/wAAD/8AAD//4AD///AB///4Af///AP///wD///8A/4D/gP8Af4D/AH+A/wB/gP8AfwD/AH8A/4D/AH/D/wB///4AP//8AB//+AAf//wAP//+AH///wD/4/8A/4D/g/8Af4P+AH/D/gB/w/4Af8P+AH/D/wB/w/+A/8P///+A////gP///wB///8AP//+AA//+AAD/8AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    9: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAH/wAAB/8AAB//4AB///AA///4Af///AH///wD///+A/8D/gP+Af8P/AH/D/wA/w/4AP8P+AD/D/gA/w/8AP8P/AH/A/4D/wP///8D////Af///wD///8Af///AD///wAD8/4AAAH+AAAB/gAAA/4AAAf+AAA//AAf//wAP//4AD//8AA//+AAP//AAD//AAA//AAAH4AAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
}

_GAME_BANK_CACHE: Optional[np.ndarray] = None
_GAME_HOLES_CACHE: Optional[np.ndarray] = None

_ORIGINAL_FAST = None
_ORIGINAL_FULL = None
_ORIGINAL_READ_NUMBER = None


def _normalize_digit_mask(
    mask: np.ndarray,
    out_w: int = 32,
    out_h: int = 48,
    pad: int = 2,
) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min(
        (out_w - 2 * pad) / max(1, m.shape[1]),
        (out_h - 2 * pad) / max(1, m.shape[0]),
    )
    nw = max(1, int(round(m.shape[1] * scale)))
    nh = max(1, int(round(m.shape[0] * scale)))
    resized = cv2.resize(
        m,
        (nw, nh),
        interpolation=cv2.INTER_NEAREST,
    )

    out = np.zeros((out_h, out_w), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _hole_count(mask: np.ndarray) -> int:
    m = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(
        m,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return 0

    holes = 0
    for i, contour in enumerate(contours):
        if int(hierarchy[0][i][3]) == -1:
            continue
        if cv2.contourArea(contour) >= 4:
            holes += 1
    return holes


def _game_bank() -> Tuple[np.ndarray, np.ndarray]:
    global _GAME_BANK_CACHE, _GAME_HOLES_CACHE
    if _GAME_BANK_CACHE is not None and _GAME_HOLES_CACHE is not None:
        return _GAME_BANK_CACHE, _GAME_HOLES_CACHE

    rows: List[np.ndarray] = []
    holes: List[int] = []
    for digit in range(10):
        packed = np.frombuffer(
            base64.b64decode(_GAME_DIGIT_B64[digit]),
            dtype=np.uint8,
        )
        bits = np.unpackbits(packed)[: 32 * 48].reshape(48, 32)
        rows.append(bits.astype(np.float32).reshape(-1))
        holes.append(_hole_count(bits))

    _GAME_BANK_CACHE = np.stack(rows, axis=0).astype(np.float32)
    _GAME_HOLES_CACHE = np.asarray(holes, dtype=np.int16)
    return _GAME_BANK_CACHE, _GAME_HOLES_CACHE


def _recognize_game_digit_raw(
    mask: np.ndarray,
) -> Tuple[Optional[int], float, float]:
    normalized = _normalize_digit_mask(mask)
    if int(np.count_nonzero(normalized)) == 0:
        return None, 0.0, 0.0

    v0 = normalized.astype(np.float32)
    bank, template_holes = _game_bank()
    areas = bank.sum(axis=1)
    input_holes = _hole_count(normalized)

    best = np.full(10, -1e9, dtype=np.float32)

    # 游戏字体原型与真实截图同源，只允许 ±1px 位置误差。
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(v0)

            ys0 = max(0, -dy)
            ys1 = min(v0.shape[0], v0.shape[0] - dy)
            xs0 = max(0, -dx)
            xs1 = min(v0.shape[1], v0.shape[1] - dx)

            if ys1 <= ys0 or xs1 <= xs0:
                continue

            yd0 = max(0, dy)
            xd0 = max(0, dx)
            shifted[
                yd0:yd0 + (ys1 - ys0),
                xd0:xd0 + (xs1 - xs0),
            ] = v0[ys0:ys1, xs0:xs1]

            v = shifted.reshape(-1)
            inter = bank @ v
            dice = 2.0 * inter / np.maximum(
                1.0,
                areas + float(v.sum()),
            )
            score = dice - 0.12 * np.abs(
                template_holes.astype(np.float32)
                - float(input_holes)
            )
            best = np.maximum(best, score)

    ranked = np.argsort(best)[::-1]
    digit = int(ranked[0])
    score = float(best[digit])
    second = float(best[int(ranked[1])])
    margin = score - second
    return digit, score, margin


def _game_digit_is_confident(score: float, margin: float) -> bool:
    # 真实回归样本中正常数字基本在 0.90+；
    # 5/6/9 等相近形状允许较小 margin，但要求更高绝对分。
    return (
        (score >= 0.84 and margin >= 0.055)
        or (score >= 0.92 and margin >= 0.030)
    )


def recognize_digit_fast_only(
    mask: np.ndarray,
) -> Tuple[Optional[int], float, float]:
    digit, score, margin = _recognize_game_digit_raw(mask)
    if (
        digit is not None
        and _game_digit_is_confident(score, margin)
    ):
        return digit, score, margin

    if _ORIGINAL_FAST is not None:
        return _ORIGINAL_FAST(mask)
    return None, score, margin


def recognize_digit(
    mask: np.ndarray,
) -> Tuple[Optional[int], float, float]:
    digit, score, margin = _recognize_game_digit_raw(mask)
    if (
        digit is not None
        and _game_digit_is_confident(score, margin)
    ):
        return digit, score, margin

    if _ORIGINAL_FULL is not None:
        return _ORIGINAL_FULL(mask)
    return None, score, margin


def _extract_white_digit_masks(
    crop_bgr: np.ndarray,
) -> List[np.ndarray]:
    """
    与旧 OCR 保持同样的数字组件提取几何，只替换“字形分类器”。

    正常高对比度车顶数字的白色填充是最稳定的观察量。
    """
    if crop_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    raw = ((sat < 100) & (val > 155)).astype(np.uint8) * 255
    raw = cv2.medianBlur(raw, 3)

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        raw,
        8,
    )
    H, W = raw.shape
    comps = []

    for label_id in range(1, n):
        x, y, cw, ch, area = map(int, stats[label_id])

        if area < max(45, int(H * W * 0.006)):
            continue
        if ch < int(H * 0.28) or ch > int(H * 0.85):
            continue
        if cw < 4 or cw > int(W * 0.48):
            continue
        if y + ch < int(H * 0.45):
            continue

        comps.append((x, y, cw, ch, area, label_id))

    if not comps:
        return []

    if len(comps) > 2:
        best_pair = None
        best_pair_score = -1e9

        for ai in range(len(comps)):
            for bi in range(ai + 1, len(comps)):
                a = comps[ai]
                b = comps[bi]
                if a[0] > b[0]:
                    a, b = b, a

                height_sim = (
                    1.0
                    - abs(a[3] - b[3])
                    / max(1.0, float(max(a[3], b[3])))
                )
                y_sim = (
                    1.0
                    - abs(a[1] - b[1])
                    / max(1.0, float(H))
                )
                gap = b[0] - (a[0] + a[2])
                center_x = (
                    (a[0] + a[2] / 2.0)
                    + (b[0] + b[2] / 2.0)
                ) / 2.0
                center_penalty = (
                    abs(center_x - W / 2.0)
                    / max(1.0, W / 2.0)
                )
                gap_penalty = (
                    max(0.0, gap - W * 0.18)
                    / max(1.0, W)
                )

                pair_score = (
                    3.0 * height_sim
                    + 1.5 * y_sim
                    - 2.0 * center_penalty
                    - gap_penalty
                )
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (a, b)

        if best_pair is not None and best_pair_score >= 2.2:
            comps = list(best_pair)
        else:
            comps = [
                max(
                    comps,
                    key=lambda c: (
                        -abs(
                            (c[0] + c[2] / 2.0)
                            - W / 2.0
                        ),
                        c[4],
                    ),
                )
            ]

    comps.sort(key=lambda item: item[0])
    comps = comps[:2]

    masks: List[np.ndarray] = []
    for x, y, cw, ch, _area, label_id in comps:
        masks.append(
            (
                labels[y:y + ch, x:x + cw] == label_id
            ).astype(np.uint8)
        )
    return masks


def _read_game_number_from_crop(
    crop_bgr: np.ndarray,
) -> Tuple[Optional[int], float]:
    masks = _extract_white_digit_masks(crop_bgr)
    if len(masks) not in (1, 2):
        return None, 0.0

    digits: List[int] = []
    confidence_parts: List[float] = []

    for mask in masks:
        digit, score, margin = _recognize_game_digit_raw(mask)
        if (
            digit is None
            or not _game_digit_is_confident(score, margin)
        ):
            return None, 0.0

        digits.append(int(digit))
        confidence_parts.append(
            float(score + min(0.12, max(0.0, margin)))
        )

    if digits[0] == 0:
        return None, 0.0

    if len(digits) == 1:
        value = digits[0]
    else:
        value = digits[0] * 10 + digits[1]

    return int(value), float(min(confidence_parts))


def read_number_detailed_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
) -> Tuple[Optional[int], float, int]:
    """
    返回 (value, confidence, agreeing_crops)。

    第二步补位确认会用 agreeing_crops 区分：
      - 真正稳定的数字识别；
      - 单裁剪偶发识别。
    """
    h, w = image_bgr.shape[:2]
    sx = (w / REF_W) * scale_hint
    sy = (h / REF_H) * scale_hint

    votes: List[Tuple[int, float]] = []

    for top, bottom, half in (
        (85, 15, 45),
        (80, 20, 45),
        (75, 10, 42),
        (90, 18, 48),
        (82, 12, 48),
    ):
        x1 = max(0, int(cx - half * sx))
        x2 = min(w, int(cx + half * sx))
        y1 = max(0, int(cy - top * sy))
        y2 = min(h, int(cy - bottom * sy))

        value, confidence = _read_game_number_from_crop(
            image_bgr[y1:y2, x1:x2]
        )
        if value is not None and 1 <= value <= 99:
            votes.append((int(value), float(confidence)))

    if votes:
        grouped: Dict[int, List[float]] = defaultdict(list)
        for value, confidence in votes:
            grouped[value].append(confidence)

        ranked = sorted(
            grouped.items(),
            key=lambda kv: (
                len(kv[1]),
                sum(kv[1]) / len(kv[1]),
            ),
            reverse=True,
        )
        value, confs = ranked[0]
        avg_conf = float(sum(confs) / len(confs))

        if len(confs) >= 2:
            return int(value), avg_conf, len(confs)

        if len(confs) == 1 and confs[0] >= 0.92:
            return int(value), avg_conf, 1

    # 对非常规版式保留旧 OCR 兜底；正常游戏帧优先真实字形。
    if _ORIGINAL_READ_NUMBER is not None:
        fallback = _ORIGINAL_READ_NUMBER(
            image_bgr,
            cx,
            cy,
            scale_hint,
        )
        if fallback is not None:
            return int(fallback), 0.50, 0

    return None, 0.0, 0


def read_number_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
) -> Optional[int]:
    value, _confidence, _agreeing = read_number_detailed_at(
        image_bgr,
        cx,
        cy,
        scale_hint,
    )
    return value


def install_game_digit_ocr(
    ocr_module,
    vehicles_module=None,
) -> None:
    """
    在不破坏第二排 preview OCR 的前提下，把正常高对比度数字识别切换为游戏字形。

    这样无需复制整个 ocr.py / vehicles.py：
      - ocr 模块中的公开 read_number_at 被替换；
      - vehicles 已经 import 过旧函数时，也显式替换其模块全局引用；
      - 第二排 read_preview_number_at 保持原实现。
    """
    global _ORIGINAL_FAST, _ORIGINAL_FULL, _ORIGINAL_READ_NUMBER

    if _ORIGINAL_FAST is None:
        _ORIGINAL_FAST = getattr(
            ocr_module,
            "_recognize_digit_fast_only",
            None,
        )
    if _ORIGINAL_FULL is None:
        _ORIGINAL_FULL = getattr(
            ocr_module,
            "_recognize_digit",
            None,
        )
    if _ORIGINAL_READ_NUMBER is None:
        _ORIGINAL_READ_NUMBER = getattr(
            ocr_module,
            "read_number_at",
            None,
        )

    ocr_module._recognize_digit_fast_only = recognize_digit_fast_only
    ocr_module._recognize_digit = recognize_digit
    ocr_module.read_number_at = read_number_at

    if vehicles_module is not None:
        vehicles_module._recognize_digit_fast_only = recognize_digit_fast_only
        vehicles_module._recognize_digit = recognize_digit
        vehicles_module.read_number_at = read_number_at
