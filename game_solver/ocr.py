from __future__ import annotations

import base64
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import REF_H, REF_W


# Canonical game digit templates captured from the game's real high-contrast
# glyph shapes. Only normalized 32x48 binary glyphs are stored here.
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
# Additional real-game glyph variants. Keep the original canonical template;
# variants are max-pooled per digit so an added sample cannot remove support
# for an older rendering of the same glyph.
_GAME_DIGIT_VARIANTS_B64 = {
    6: (
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8AAAAfAAAB/4AAD/+AAD//gAB//4AA//+AAf//gAP//wAD/+AAB/4AAAf8AAAP+AAAD/gAAA/4AAAP//8AP///wD///+A////wP///+D////g////4P/gP/D/wB/w/4Af8P+AH/D/gB/w/4Af8P/AH/D/wB/wP+A/4D///+A////gH///wA///4AH//8AA//+AAB/+AAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ),
}

_GAME_BANK_CACHE: Optional[np.ndarray] = None
_GAME_TEMPLATE_DIGITS_CACHE: Optional[np.ndarray] = None
_GAME_HOLES_CACHE: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DigitOcrDiagnostic:
    digit: Optional[int]
    score: float
    margin: float
    hole_count: int
    source: str
    accepted: bool


@dataclass(frozen=True)
class CropOcrDiagnostic:
    source: str
    value: Optional[int]
    confidence: float
    digits: Tuple[DigitOcrDiagnostic, ...]


@dataclass(frozen=True)
class NumberOcrResult:
    """Structured OCR result suitable for logs and replay diagnostics."""

    value: Optional[int]
    candidate_value: Optional[int]
    confidence: float
    agreeing_crops: int
    source: str
    crops: Tuple[CropOcrDiagnostic, ...]
    vote_counts: Tuple[Tuple[int, int], ...]


def _normalize_digit_mask(
    mask: np.ndarray,
    out_w: int = 32,
    out_h: int = 48,
    pad: int = 2,
) -> np.ndarray:
    """Normalize one binary digit to the canonical 32x48 game-glyph canvas."""
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
    """Count structural holes used to separate near glyphs such as 0/6/8/9."""
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


def _game_bank() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _GAME_BANK_CACHE, _GAME_TEMPLATE_DIGITS_CACHE, _GAME_HOLES_CACHE
    if (
        _GAME_BANK_CACHE is not None
        and _GAME_TEMPLATE_DIGITS_CACHE is not None
        and _GAME_HOLES_CACHE is not None
    ):
        return (
            _GAME_BANK_CACHE,
            _GAME_TEMPLATE_DIGITS_CACHE,
            _GAME_HOLES_CACHE,
        )

    rows: List[np.ndarray] = []
    template_digits: List[int] = []
    holes: List[int] = []

    def append_template(digit: int, encoded: str) -> None:
        packed = np.frombuffer(
            base64.b64decode(encoded),
            dtype=np.uint8,
        )
        bits = np.unpackbits(packed)[: 32 * 48].reshape(48, 32)
        rows.append(bits.astype(np.float32).reshape(-1))
        template_digits.append(int(digit))
        holes.append(_hole_count(bits))

    for digit in range(10):
        append_template(digit, _GAME_DIGIT_B64[digit])

    for digit, variants in sorted(_GAME_DIGIT_VARIANTS_B64.items()):
        for encoded in variants:
            append_template(int(digit), encoded)

    _GAME_BANK_CACHE = np.stack(rows, axis=0).astype(np.float32)
    _GAME_TEMPLATE_DIGITS_CACHE = np.asarray(
        template_digits,
        dtype=np.int16,
    )
    _GAME_HOLES_CACHE = np.asarray(holes, dtype=np.int16)
    return (
        _GAME_BANK_CACHE,
        _GAME_TEMPLATE_DIGITS_CACHE,
        _GAME_HOLES_CACHE,
    )


def _accept_digit(score: float, margin: float, profile: str) -> bool:
    if profile == "preview":
        # Preview extraction is lower contrast, but classification is identical.
        return (
            (score >= 0.80 and margin >= 0.025)
            or (score >= 0.72 and margin >= 0.060)
        )
    if profile != "normal":
        raise ValueError(f"unknown OCR profile: {profile}")
    return (
        (score >= 0.84 and margin >= 0.055)
        or (score >= 0.92 and margin >= 0.030)
    )


class GameDigitRecognizer:
    """The repository's single digit classifier, backed by real-game templates."""

    def recognize(
        self,
        mask: np.ndarray,
        *,
        source: str = "digit",
        profile: str = "normal",
    ) -> DigitOcrDiagnostic:
        normalized = _normalize_digit_mask(mask)
        hole_count = _hole_count(normalized)
        if int(np.count_nonzero(normalized)) == 0:
            return DigitOcrDiagnostic(
                digit=None,
                score=0.0,
                margin=0.0,
                hole_count=hole_count,
                source=source,
                accepted=False,
            )

        v0 = normalized.astype(np.float32)
        bank, template_digits, template_holes = _game_bank()
        areas = bank.sum(axis=1)
        template_best = np.full(len(bank), -1e9, dtype=np.float32)

        # Real-game templates are already normalized from the same font. Only a
        # ±1 px alignment tolerance is needed. Multiple real variants for one
        # digit are max-pooled after alignment instead of replacing the older
        # canonical rendering.
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
                scores = dice - 0.12 * np.abs(
                    template_holes.astype(np.float32) - float(hole_count)
                )
                template_best = np.maximum(template_best, scores)

        best = np.full(10, -1e9, dtype=np.float32)
        for digit in range(10):
            matches = template_best[template_digits == digit]
            if len(matches):
                best[digit] = float(np.max(matches))

        ranked = np.argsort(best)[::-1]
        digit = int(ranked[0])
        score = float(best[digit])
        second = float(best[int(ranked[1])])
        margin = score - second
        accepted = _accept_digit(score, margin, profile)
        return DigitOcrDiagnostic(
            digit=digit,
            score=score,
            margin=margin,
            hole_count=hole_count,
            source=source,
            accepted=accepted,
        )


_GAME_DIGIT_RECOGNIZER = GameDigitRecognizer()


def recognize_digit(
    mask: np.ndarray,
    *,
    source: str = "digit",
    profile: str = "normal",
) -> DigitOcrDiagnostic:
    """Canonical public digit-classification interface."""
    return _GAME_DIGIT_RECOGNIZER.recognize(
        mask,
        source=source,
        profile=profile,
    )


def _extract_white_digit_masks(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """Extract normal high-contrast white game digits from a vehicle crop."""
    if crop_bgr.size == 0:
        return []

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    raw = ((sat < 100) & (val > 155)).astype(np.uint8) * 255
    raw = cv2.medianBlur(raw, 3)

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw, 8)
    h, w = raw.shape
    comps: List[Tuple[int, int, int, int, int, int]] = []

    for label_id in range(1, n):
        x, y, cw, ch, area = map(int, stats[label_id])
        if area < max(45, int(h * w * 0.006)):
            continue
        if ch < int(h * 0.28) or ch > int(h * 0.85):
            continue
        if cw < 4 or cw > int(w * 0.48):
            continue
        if y + ch < int(h * 0.45):
            continue
        comps.append((x, y, cw, ch, area, label_id))

    if not comps:
        return []

    if len(comps) > 2:
        best_pair = None
        best_pair_score = -1e9
        for ai in range(len(comps)):
            for bi in range(ai + 1, len(comps)):
                a, b = comps[ai], comps[bi]
                if a[0] > b[0]:
                    a, b = b, a
                height_sim = 1.0 - abs(a[3] - b[3]) / max(
                    1.0, float(max(a[3], b[3]))
                )
                y_sim = 1.0 - abs(a[1] - b[1]) / max(1.0, float(h))
                gap = b[0] - (a[0] + a[2])
                center_x = ((a[0] + a[2] / 2.0) + (b[0] + b[2] / 2.0)) / 2.0
                center_penalty = abs(center_x - w / 2.0) / max(1.0, w / 2.0)
                gap_penalty = max(0.0, gap - w * 0.18) / max(1.0, w)
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
                    key=lambda comp: (
                        -abs((comp[0] + comp[2] / 2.0) - w / 2.0),
                        comp[4],
                    ),
                )
            ]

    comps.sort(key=lambda item: item[0])
    masks: List[np.ndarray] = []
    for x, y, cw, ch, _area, label_id in comps[:2]:
        masks.append(
            (labels[y:y + ch, x:x + cw] == label_id).astype(np.uint8)
        )
    return masks


def _fill_external_contours(mask: np.ndarray) -> np.ndarray:
    """Normalize preview outline/semi-filled extraction before classification."""
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if contours:
        cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return (out > 0).astype(np.uint8)


def _extract_preview_digit_masks(crop_bgr: np.ndarray) -> List[np.ndarray]:
    """Extract low-contrast second-row digits; classification remains shared."""
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
    h, w = raw.shape
    comps: List[Tuple[int, int, int, int, int, int]] = []
    for label_id in range(1, n):
        x, y, cw, ch, area = map(int, stats[label_id])
        if x <= 2 or x + cw >= w - 2:
            continue
        if y < int(h * 0.42):
            continue
        if not (8 <= cw <= 38 and 18 <= ch <= 40 and area >= 50):
            continue
        if cw / max(1.0, float(ch)) > 1.45:
            continue
        comps.append((x, y, cw, ch, area, label_id))

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
                    - 0.06 * abs(center - w / 2.0)
                )
                pairs.append((score, a, b))
        if pairs:
            _score, a, b = max(pairs, key=lambda item: item[0])
            comps = [a, b]

    masks: List[np.ndarray] = []
    for x, y, cw, ch, _area, label_id in comps[:2]:
        digit = (labels[y:y + ch, x:x + cw] == label_id).astype(np.uint8)
        masks.append(_fill_external_contours(digit))
    return masks


def _read_crop(
    crop_bgr: np.ndarray,
    *,
    source: str,
    extractor: Callable[[np.ndarray], List[np.ndarray]],
    profile: str,
) -> CropOcrDiagnostic:
    masks = extractor(crop_bgr)
    if len(masks) not in (1, 2):
        return CropOcrDiagnostic(source, None, 0.0, ())

    diagnostics = tuple(
        recognize_digit(
            mask,
            source=f"{source}:digit{index}",
            profile=profile,
        )
        for index, mask in enumerate(masks, 1)
    )
    if any(not item.accepted or item.digit is None for item in diagnostics):
        return CropOcrDiagnostic(source, None, 0.0, diagnostics)

    digits = [int(item.digit) for item in diagnostics if item.digit is not None]
    if not digits or digits[0] == 0:
        return CropOcrDiagnostic(source, None, 0.0, diagnostics)

    value = digits[0] if len(digits) == 1 else digits[0] * 10 + digits[1]
    confidence = min(
        item.score + min(0.12, max(0.0, item.margin))
        for item in diagnostics
    )
    return CropOcrDiagnostic(source, int(value), float(confidence), diagnostics)


def _multi_crop_number(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float,
    *,
    source: str,
    crop_specs: Sequence[Tuple[int, int, int]],
    extractor: Callable[[np.ndarray], List[np.ndarray]],
    profile: str,
    single_crop_min_confidence: float,
) -> NumberOcrResult:
    h, w = image_bgr.shape[:2]
    sx = (w / REF_W) * scale_hint
    sy = (h / REF_H) * scale_hint
    crop_results: List[CropOcrDiagnostic] = []

    for index, (top, bottom, half) in enumerate(crop_specs, 1):
        x1 = max(0, int(cx - half * sx))
        x2 = min(w, int(cx + half * sx))
        y1 = max(0, int(cy - top * sy))
        y2 = min(h, int(cy - bottom * sy))
        crop_results.append(
            _read_crop(
                image_bgr[y1:y2, x1:x2],
                source=f"{source}:crop{index}[{top},{bottom},{half}]",
                extractor=extractor,
                profile=profile,
            )
        )

    grouped: Dict[int, List[float]] = defaultdict(list)
    for result in crop_results:
        if result.value is not None and 1 <= result.value <= 99:
            grouped[int(result.value)].append(float(result.confidence))

    if not grouped:
        return NumberOcrResult(
            value=None,
            candidate_value=None,
            confidence=0.0,
            agreeing_crops=0,
            source=source,
            crops=tuple(crop_results),
            vote_counts=(),
        )

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len(item[1]),
            sum(item[1]) / len(item[1]),
        ),
        reverse=True,
    )
    candidate, confidences = ranked[0]
    avg_confidence = float(sum(confidences) / len(confidences))
    accepted = (
        len(confidences) >= 2
        or (
            len(confidences) == 1
            and confidences[0] >= single_crop_min_confidence
        )
    )
    vote_counts = tuple(
        sorted((int(value), len(confs)) for value, confs in grouped.items())
    )
    return NumberOcrResult(
        value=int(candidate) if accepted else None,
        candidate_value=int(candidate),
        confidence=avg_confidence,
        agreeing_crops=len(confidences),
        source=source,
        crops=tuple(crop_results),
        vote_counts=vote_counts,
    )


def read_number_detailed_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
    *,
    source: str = "normal",
) -> NumberOcrResult:
    """Read normal first-row/parked digits with structured crop diagnostics."""
    return _multi_crop_number(
        image_bgr,
        cx,
        cy,
        scale_hint,
        source=source,
        crop_specs=(
            (85, 15, 45),
            (80, 20, 45),
            (75, 10, 42),
            (90, 18, 48),
            (82, 12, 48),
        ),
        extractor=_extract_white_digit_masks,
        profile="normal",
        single_crop_min_confidence=0.92,
    )


def read_number_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
) -> Optional[int]:
    return read_number_detailed_at(
        image_bgr,
        cx,
        cy,
        scale_hint,
    ).value


def read_preview_number_detailed_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
    *,
    source: str = "preview",
) -> NumberOcrResult:
    """Read second-row digits using preview extraction + the same classifier."""
    return _multi_crop_number(
        image_bgr,
        cx,
        cy,
        scale_hint,
        source=source,
        crop_specs=(
            (90, 8, 50),
            (86, 10, 48),
            (94, 6, 52),
        ),
        extractor=_extract_preview_digit_masks,
        profile="preview",
        single_crop_min_confidence=0.88,
    )


def read_preview_number_at(
    image_bgr: np.ndarray,
    cx: float,
    cy: float,
    scale_hint: float = 1.0,
) -> Optional[int]:
    return read_preview_number_detailed_at(
        image_bgr,
        cx,
        cy,
        scale_hint,
    ).value
