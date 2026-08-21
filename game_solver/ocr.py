from __future__ import annotations

from collections import defaultdict
import base64
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    REF_W, REF_H,
    FAST_OCR_MIN_SCORE, FAST_OCR_MIN_MARGIN,
    FAST_OCR_STRONG_SCORE, FAST_OCR_STRONG_MARGIN,
)

_DIGIT_TEMPLATE_CACHE: Optional[Dict[int, List[Tuple[np.ndarray, int]]]] = None
_DIGIT_SLOW_TEMPLATE_CACHE: Optional[Dict[int, List[Tuple[np.ndarray, int]]]] = None
_DIGIT_FAST_BANK_CACHE: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

def _normalize_digit_mask(mask: np.ndarray, out_w: int = 32, out_h: int = 48, pad: int = 2) -> np.ndarray:
    """把单个数字二值图缩放到统一画布，保持宽高比并居中。"""
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
    resized = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)

    out = np.zeros((out_h, out_w), dtype=np.uint8)
    x0 = (out_w - nw) // 2
    y0 = (out_h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out


def _hole_count(mask: np.ndarray) -> int:
    """统计数字内部孔洞数：0/6/9≈1，8≈2，4在当前字体通常≈1。"""
    m = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0

    holes = 0
    for i, contour in enumerate(contours):
        parent = int(hierarchy[0][i][3])
        if parent != -1 and cv2.contourArea(contour) >= 4:
            holes += 1
    return holes


def _shifted_dice(a: np.ndarray, b: np.ndarray, max_shift: int = 2) -> float:
    """允许少量平移的 Dice 相似度，降低截图缩放/采样带来的偏移。"""
    aa = (a > 0)
    best = 0.0
    h, w = a.shape

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(
                b.astype(np.uint8),
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            denom = int(aa.sum()) + int(moved.sum())
            if denom == 0:
                continue
            score = 2.0 * float(np.logical_and(aa, moved).sum()) / denom
            if score > best:
                best = score
    return best


def _build_digit_templates(
    fonts: Sequence[int],
    scales: Sequence[float],
    thicknesses: Sequence[int],
) -> Dict[int, List[Tuple[np.ndarray, int]]]:
    templates: Dict[int, List[Tuple[np.ndarray, int]]] = defaultdict(list)

    for digit in range(10):
        for font in fonts:
            for scale in scales:
                for thickness in thicknesses:
                    canvas = np.zeros((90, 70), dtype=np.uint8)
                    (tw, th), _baseline = cv2.getTextSize(
                        str(digit), font, scale, thickness
                    )
                    org = ((70 - tw) // 2, (90 + th) // 2)
                    cv2.putText(
                        canvas,
                        str(digit),
                        org,
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    mask = _normalize_digit_mask(canvas > 100)
                    templates[digit].append((mask, _hole_count(mask)))

    return dict(templates)


def _digit_templates() -> Dict[int, List[Tuple[np.ndarray, int]]]:
    """
    日常快速模板库：120 个模板。
    已用现有真实 13/25/34/35/36 样本回归，均能在无平移路径正确识别。
    """
    global _DIGIT_TEMPLATE_CACHE
    if _DIGIT_TEMPLATE_CACHE is None:
        _DIGIT_TEMPLATE_CACHE = _build_digit_templates(
            fonts=(
                cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_DUPLEX,
            ),
            scales=(1.0, 1.3, 1.6),
            thicknesses=(3, 4),
        )
    return _DIGIT_TEMPLATE_CACHE


def _digit_slow_templates() -> Dict[int, List[Tuple[np.ndarray, int]]]:
    """
    只有快速路径置信度不足时才建立完整模板库。
    正常游戏帧不会支付这部分初始化成本。
    """
    global _DIGIT_SLOW_TEMPLATE_CACHE
    if _DIGIT_SLOW_TEMPLATE_CACHE is None:
        _DIGIT_SLOW_TEMPLATE_CACHE = _build_digit_templates(
            fonts=(
                cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_DUPLEX,
                cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX,
            ),
            scales=(0.90, 1.10, 1.30, 1.50, 1.70),
            thicknesses=(2, 3, 4, 5),
        )
    return _DIGIT_SLOW_TEMPLATE_CACHE


def _digit_fast_bank() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    把所有 OpenCV 字体模板堆成一个矩阵，一次矩阵乘法完成所有无平移 Dice 比较。
    返回:
      bank:  [N, 1536] float32 0/1
      labels:[N] digit
      holes: [N]
      areas: [N]
    """
    global _DIGIT_FAST_BANK_CACHE
    if _DIGIT_FAST_BANK_CACHE is not None:
        return _DIGIT_FAST_BANK_CACHE

    rows: List[np.ndarray] = []
    labels: List[int] = []
    holes: List[int] = []

    for digit, variants in _digit_templates().items():
        for tmpl, tmpl_holes in variants:
            rows.append((tmpl > 0).astype(np.float32).reshape(-1))
            labels.append(int(digit))
            holes.append(int(tmpl_holes))

    bank = np.stack(rows, axis=0).astype(np.float32)
    label_arr = np.asarray(labels, dtype=np.int16)
    hole_arr = np.asarray(holes, dtype=np.int16)
    areas = bank.sum(axis=1).astype(np.float32)
    _DIGIT_FAST_BANK_CACHE = (bank, label_arr, hole_arr, areas)
    return _DIGIT_FAST_BANK_CACHE


def _recognize_digit_fast_only(mask: np.ndarray) -> Tuple[Optional[int], float, float]:
    """不做任何平移搜索的快速数字匹配。"""
    normalized = _normalize_digit_mask(mask)
    v = (normalized > 0).astype(np.float32).reshape(-1)
    if float(v.sum()) <= 0:
        return None, 0.0, 0.0

    bank, labels, holes, areas = _digit_fast_bank()
    inter = bank @ v
    dice = 2.0 * inter / np.maximum(1.0, areas + float(v.sum()))
    input_holes = _hole_count(normalized)
    scores_all = dice - 0.16 * np.abs(holes.astype(np.float32) - float(input_holes))

    best_by_digit = np.full(10, -1e9, dtype=np.float32)
    for digit in range(10):
        sel = scores_all[labels == digit]
        if len(sel):
            best_by_digit[digit] = float(sel.max())

    ranked = np.argsort(best_by_digit)[::-1]
    best_digit = int(ranked[0])
    best_score = float(best_by_digit[best_digit])
    second_score = float(best_by_digit[int(ranked[1])])
    margin = best_score - second_score

    if best_score < 0.52:
        return None, best_score, margin
    return best_digit, best_score, margin


def _recognize_digit_slow(mask: np.ndarray) -> Tuple[Optional[int], float, float]:
    """旧版 ±2px 平移穷举，仅作为低置信度兜底。"""
    normalized = _normalize_digit_mask(mask)
    input_holes = _hole_count(normalized)
    templates = _digit_slow_templates()

    scores: Dict[int, float] = {}
    for digit, variants in templates.items():
        best = -1e9
        for tmpl, tmpl_holes in variants:
            score = _shifted_dice(normalized, tmpl) - 0.16 * abs(input_holes - tmpl_holes)
            if score > best:
                best = score
        scores[digit] = best

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_digit, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score

    if best_score < 0.52:
        return None, float(best_score), float(margin)
    return int(best_digit), float(best_score), float(margin)


def _recognize_digit(mask: np.ndarray) -> Tuple[Optional[int], float, float]:
    """
    快速优先：
    - 标准化后直接与全部模板做无平移向量化匹配；
    - 高置信度立即返回；
    - 只有边界/模糊情况才调用旧的慢速平移搜索。
    """
    digit, score, margin = _recognize_digit_fast_only(mask)
    if digit is not None and (
        (score >= FAST_OCR_MIN_SCORE and margin >= FAST_OCR_MIN_MARGIN)
        or (score >= FAST_OCR_STRONG_SCORE and margin >= FAST_OCR_STRONG_MARGIN)
    ):
        return digit, score, margin

    return _recognize_digit_slow(mask)


def _extract_white_digit_masks(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """
    提取车顶数字的白色填充。

    现在允许返回 1 个或 2 个数字组件：
    - 两位数：35、27、13...
    - 个位数：9、7、3...

    游戏个位数不会显示前导 0。
    """
    if crop_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = ((sat < 100) & (val > 155)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    H, W = mask.shape
    comps: List[Tuple[int, int, int, int, int, int]] = []

    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])

        if area < max(45, int(H * W * 0.006)):
            continue
        if h < int(H * 0.28) or h > int(H * 0.85):
            continue
        if w < 4 or w > int(W * 0.48):
            continue
        if y + h < int(H * 0.45):
            continue

        comps.append((x, y, w, h, area, i))

    if not comps:
        return []

    # 多于2个候选时，优先找最像真正两位数字的一对。
    if len(comps) > 2:
        best_pair = None
        best_pair_score = -1e9

        for a_idx in range(len(comps)):
            for b_idx in range(a_idx + 1, len(comps)):
                a = comps[a_idx]
                b = comps[b_idx]
                if a[0] > b[0]:
                    a, b = b, a

                height_sim = 1.0 - abs(a[3] - b[3]) / max(1.0, max(a[3], b[3]))
                y_sim = 1.0 - abs(a[1] - b[1]) / max(1.0, H)
                gap = b[0] - (a[0] + a[2])
                center_x = ((a[0] + a[2] / 2) + (b[0] + b[2] / 2)) / 2
                center_penalty = abs(center_x - W / 2) / max(1.0, W / 2)
                gap_penalty = max(0.0, gap - W * 0.18) / max(1.0, W)

                score = (
                    3.0 * height_sim
                    + 1.5 * y_sim
                    - 2.0 * center_penalty
                    - gap_penalty
                )
                if score > best_pair_score:
                    best_pair_score = score
                    best_pair = (a, b)

        # 如果能找到合理的双数字对就保留双数字；
        # 否则退化为最靠近裁剪中心、面积较大的单数字。
        if best_pair is not None and best_pair_score >= 2.2:
            comps = list(best_pair)
        else:
            comps = [
                max(
                    comps,
                    key=lambda c: (
                        -abs((c[0] + c[2] / 2) - W / 2),
                        c[4],
                    ),
                )
            ]

    comps.sort(key=lambda item: item[0])

    # 只允许 1 或 2 位。
    if len(comps) > 2:
        comps = comps[:2]

    result: List[np.ndarray] = []
    for x, y, w, h, _area, label_id in comps:
        digit = (labels[y:y + h, x:x + w] == label_id).astype(np.uint8)
        result.append(digit)

    return result


def _read_one_or_two_digit_number(crop_bgr: np.ndarray) -> Tuple[Optional[int], float]:
    masks = _extract_white_digit_masks(crop_bgr)
    if len(masks) not in (1, 2):
        return None, 0.0

    digits: List[int] = []
    confidences: List[float] = []

    for mask in masks:
        digit, score, margin = _recognize_digit(mask)
        if digit is None:
            return None, 0.0

        digits.append(digit)
        confidences.append(score + min(0.15, max(0.0, margin)))

    if len(digits) == 1:
        # 0 不会作为可见剩余数存在；到0车辆会离场。
        if digits[0] == 0:
            return None, 0.0
        value = digits[0]
    else:
        # 个位数不带前导0，因此两位数首位不能是0。
        if digits[0] == 0:
            return None, 0.0
        value = digits[0] * 10 + digits[1]

    return int(value), float(min(confidences))


# 兼容旧内部名字。
def _read_two_digit_number(crop_bgr: np.ndarray) -> Tuple[Optional[int], float]:
    return _read_one_or_two_digit_number(crop_bgr)


# 游戏第二排车辆的数字使用低对比度/半透明样式。
# 下面的 32x48 二值原型来自同一游戏字体的真实数字形状；只保存字形，不保存截图。
_PREVIEW_GAME_DIGIT_B64 = {
    0: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/4AAAP+AAAf/4AAf//gAP//8AD///gB///4Af///AP///wD///+A////g////4P///+D////g////4P////D////w////8P////D////w////8P////D////w////4P///+D////g////4D///+A////gP///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    1: "AAAAAAAAAAAAB//gAAf/4AP///gD///4D///+D////w////8P////D////w////8P////D////w////8P////D////wMB//8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAB//wAAf/8AAH//AAAAAAAAAAA",
    2: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/AAAD/wAAP//wAH///AB///4Af///AH///4B///+Af///gH///8AwD//AAAP/wAAB/8AAAf/AAAP/gAAD/4AAA/+AAA//gAAf/wAAP/8AAH/+AAD//AAB//gAA//wAAf/4AAP/8AAP/+AAH//AAD//gAD//4AA////4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    3: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH/4AAB/+AAB//8AA///4AP///AD///4A////AP///wD///8A////gGA//4AAH/+AAA//gAAP/4AAD/+AAB//AAB//wAf//4AP//8AD///AA///gAP//4AD///AA///4AH///AAH//4AAD/+AAAP/wAAD/8AAA//AAAP/wAAP/8CAP//D////gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    4: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/4AAA//AAAP/wAAH/8AAD//AAA//wAAf/8AAH//AAD//wAB//8AA///AAP//wAH//8AD///AA///wAf//8AP///AD///wB///8Af///gH///+B////wf///8P////D////w////8H////B////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    5: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//+AD///gB///8Af///AH///wD///8A////AP///wD///8A///+AP/gAAD/gAAA/4AAAP+AAAD/4AAD///AA///8AP///4D///+A////wP///+D////g////8P////A/n//wAAH/8AAA//AAAP/wAAD/8AAA//AAAf/wAAP/8DAf//D////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    6: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8AAAA/AAAD/4AAD/+AAD//gAB//4AA//+AAf//gAP//wAD/8AAB/4AAAf8AAAP+AAAD/gAAA/4AAAP//8AD///wD///+A////wP///+D////g////4P////D////w////8P////D////w////8P////D////wP///4D///+Af///gH///wA///4AH//8AAf/+AAB/8AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    7: "AAAAAAAAAAAAAAAAD///+A////g////8P////D////w////8P////D////w////8D////AAAP/wAAA/8AAAP/AAAP/wAAD/4AAB/+AAAf/AAAP/wAAD/8AAA/+AAAf/gAAH/wAAB/8AAB//AAAf/wAAH/wAAD/8AAA/+AAAf/gAAH/wAAB/8AAA//AAAP/gAAD/4AAB/+AAAf/AAAH/wAAH/8AAB/8AAA//AAAH/gAAAAAAAAAAAAAAAAAAAAAAA",
    8: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB/8AAAf/AAA//8AAf//wAP//+AH///wD///8A////gP///4P///+D////g////4D///+A////gP///wD///8Af//+AD///AA///wAP//8AH///gD///8D////g////4P////D////w////8P////D////w////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    9: "AAAAAAAAAAAAAAAAAAAAAAABwAAAAcAAAH//AAD//4AD///AB///4Af///AP///4D///+A////g////8P////D////w////8P////D////w////8P////A////wP///8B////Af///wD///8Af///AB///gAAA/4AAAP+AAAD/gAAB/4AAA/8AAP//AAf//gAP//4AD//8AA//+AAP/+AAD/+AAA/+AAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
}
_PREVIEW_GAME_BANK_CACHE: Optional[np.ndarray] = None


def _preview_game_bank() -> np.ndarray:
    global _PREVIEW_GAME_BANK_CACHE
    if _PREVIEW_GAME_BANK_CACHE is not None:
        return _PREVIEW_GAME_BANK_CACHE

    rows: List[np.ndarray] = []
    for digit in range(10):
        packed = np.frombuffer(
            base64.b64decode(_PREVIEW_GAME_DIGIT_B64[digit]),
            dtype=np.uint8,
        )
        bits = np.unpackbits(packed)[: 32 * 48].reshape(48, 32)
        rows.append(bits.astype(np.float32).reshape(-1))
    _PREVIEW_GAME_BANK_CACHE = np.stack(rows, axis=0)
    return _PREVIEW_GAME_BANK_CACHE


def _fill_external_contours(mask: np.ndarray) -> np.ndarray:
    """把低对比度数字的轮廓/半填充统一成实心字形。"""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if contours:
        cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return (out > 0).astype(np.uint8)


def _extract_preview_digit_masks(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """
    提取第二排浅色数字。

    彩色车：数字半透明后饱和度明显低于车身；
    白色车：饱和度本身很低，改用亮度下降提取灰色数字轮廓。
    识别不够可靠时宁可返回空，不参与二层安全预测。
    """
    if crop_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    med_sat = float(np.median(sat))
    med_val = float(np.median(val))

    if med_sat < 45.0:
        raw = (val < med_val - 35.0).astype(np.uint8) * 255
    else:
        raw = ((sat < med_sat - 18.0) & (val > 60)).astype(np.uint8) * 255

    raw = cv2.medianBlur(raw, 3)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)
    H, W = raw.shape

    comps: List[Tuple[int, int, int, int, int, int]] = []
    for label_id in range(1, n):
        x, y, w, h, area = map(int, stats[label_id])
        if x <= 2 or x + w >= W - 2:
            continue
        if y < int(H * 0.42):
            continue
        if not (8 <= w <= 38 and 18 <= h <= 40 and area >= 50):
            continue
        if w / max(1.0, float(h)) > 1.45:
            continue
        comps.append((x, y, w, h, area, label_id))

    if not comps:
        return []

    comps.sort(key=lambda item: item[0])

    if len(comps) > 2:
        pairs: List[Tuple[float, Tuple[int, ...], Tuple[int, ...]]] = []
        for ai in range(len(comps)):
            for bi in range(ai + 1, len(comps)):
                a, b = comps[ai], comps[bi]
                gap = b[0] - (a[0] + a[2])
                if gap < -3 or gap > 25:
                    continue
                hsim = min(a[3], b[3]) / max(1.0, float(max(a[3], b[3])))
                yoff = abs((a[1] + a[3] / 2.0) - (b[1] + b[3] / 2.0))
                span = b[0] + b[2] - a[0]
                center = (a[0] + b[0] + b[2]) / 2.0
                score = (
                    5.0 * hsim
                    - 0.15 * yoff
                    - 0.04 * abs(span - 58.0)
                    - 0.06 * abs(center - W / 2.0)
                )
                pairs.append((score, a, b))
        if pairs:
            _score, a, b = max(pairs, key=lambda item: item[0])
            comps = [a, b]

    comps = comps[:2]
    masks: List[np.ndarray] = []
    for x, y, w, h, _area, label_id in comps:
        digit = (labels[y:y + h, x:x + w] == label_id).astype(np.uint8)
        masks.append(_fill_external_contours(digit))
    return masks


def _recognize_preview_digit(mask: np.ndarray) -> Tuple[Optional[int], float, float]:
    normalized = _normalize_digit_mask(mask)
    v0 = (normalized > 0).astype(np.float32)
    if float(v0.sum()) <= 0:
        return None, 0.0, 0.0

    bank = _preview_game_bank()
    areas = bank.sum(axis=1)
    best = np.full(10, -1e9, dtype=np.float32)

    # 只允许 ±1px；实际游戏字体模板已经同源，不需要慢速 ±2 穷举。
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.zeros_like(v0)
            ys0 = max(0, -dy)
            ys1 = min(v0.shape[0], v0.shape[0] - dy)
            xs0 = max(0, -dx)
            xs1 = min(v0.shape[1], v0.shape[1] - dx)
            yd0 = max(0, dy)
            yd1 = yd0 + (ys1 - ys0)
            xd0 = max(0, dx)
            xd1 = xd0 + (xs1 - xs0)
            if ys1 <= ys0 or xs1 <= xs0:
                continue
            shifted[yd0:yd1, xd0:xd1] = v0[ys0:ys1, xs0:xs1]
            v = shifted.reshape(-1)
            inter = bank @ v
            dice = 2.0 * inter / np.maximum(1.0, areas + float(v.sum()))
            best = np.maximum(best, dice)

    ranked = np.argsort(best)[::-1]
    digit = int(ranked[0])
    score = float(best[digit])
    second = float(best[int(ranked[1])])
    margin = score - second

    # 低对比度 OCR 只用于策略前瞻，阈值宁可保守。
    if not (
        (score >= 0.80 and margin >= 0.025)
        or (score >= 0.72 and margin >= 0.060)
    ):
        return None, score, margin
    return digit, score, margin


def _read_preview_number_from_crop(crop_bgr: np.ndarray) -> Tuple[Optional[int], float]:
    masks = _extract_preview_digit_masks(crop_bgr)
    if len(masks) not in (1, 2):
        return None, 0.0

    digits: List[int] = []
    confidences: List[float] = []
    for mask in masks:
        digit, score, margin = _recognize_preview_digit(mask)
        if digit is None:
            return None, 0.0
        digits.append(digit)
        confidences.append(score + min(0.12, max(0.0, margin)))

    if len(digits) == 1:
        if digits[0] == 0:
            return None, 0.0
        return digits[0], float(min(confidences))

    if digits[0] == 0:
        return None, 0.0
    return digits[0] * 10 + digits[1], float(min(confidences))


def read_preview_number_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
) -> Optional[int]:
    """读取第二排浅色的 1~2 位车辆数字。"""
    h, w = image_bgr.shape[:2]
    sx = (w / REF_W) * scale_hint
    sy = (h / REF_H) * scale_hint
    votes: List[Tuple[int, float]] = []

    for top, bottom, half in (
        (90, 8, 50),
        (86, 10, 48),
        (94, 6, 52),
    ):
        x1 = max(0, int(cx - half * sx))
        x2 = min(w, int(cx + half * sx))
        y1 = max(0, int(cy - top * sy))
        y2 = min(h, int(cy - bottom * sy))
        value, confidence = _read_preview_number_from_crop(image_bgr[y1:y2, x1:x2])
        if value is not None and 1 <= value <= 99:
            votes.append((int(value), float(confidence)))

    if not votes:
        return None

    grouped: Dict[int, List[float]] = defaultdict(list)
    for value, confidence in votes:
        grouped[value].append(confidence)

    ranked = sorted(
        grouped.items(),
        key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])),
        reverse=True,
    )
    value, confs = ranked[0]
    if len(confs) >= 2:
        return int(value)
    if confs and confs[0] >= 0.88:
        return int(value)
    return None


def read_number_at(image_bgr: np.ndarray, cx: float, cy: float, scale_hint: float = 1.0) -> Optional[int]:
    """
    纯 OpenCV 的 1~2 位数字识别。

    支持个位数（例如 9、7）和两位数（例如 35、13）。
    会尝试几组略有不同的裁剪框并多数投票。
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
        value, confidence = _read_one_or_two_digit_number(image_bgr[y1:y2, x1:x2])
        if value is not None and 1 <= value <= 99:
            votes.append((value, confidence))

    if not votes:
        return None

    grouped: Dict[int, List[float]] = defaultdict(list)
    for value, confidence in votes:
        grouped[value].append(confidence)

    ranked = sorted(
        grouped.items(),
        key=lambda kv: (len(kv[1]), sum(kv[1]) / len(kv[1])),
        reverse=True,
    )
    best_value, best_conf = ranked[0]

    # 多个裁剪至少两票一致最稳；若只有一个成功结果，则要求形状分数更高。
    if len(best_conf) >= 2:
        return int(best_value)
    if len(best_conf) == 1 and best_conf[0] >= 0.72:
        return int(best_value)
    return None