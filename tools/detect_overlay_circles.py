"""Per-band scroll-overlay circle detection and persistent-circle discovery.

This module finds opaque circular UI overlays that occupy a fixed SCREEN
position across many bands (e.g. iOS Messenger scroll-to-latest button).
Such overlays are conditionally present (only shown when the user has
scrolled away from the bottom), so they appear in some fraction of bands
but at the same (cx, cy, r). The per-pixel agreement-fraction detector in
``detect_overlays.py`` misses them because the underlying content varies
beneath them; this module finds them via shape (HoughCircles) instead.

Two entry points:

1. :func:`discover_persistent_circles` -- run once across all placed bands
   to find tracked overlay positions (cx, cy, r) with prevalence stats.
2. :func:`detect_circle_in_band` -- per-band targeted detection inside a
   tight ROI around an expected (cx, cy, r). Returns a boolean band-shaped
   mask of detected button pixels (with AA padding) plus detection info.

The stitcher uses (1) to find the canonical position once, then (2) to
build a per-band augmentation mask layered on top of the shared
``clean_mask``. Bands where the button is absent receive an empty mask
and continue to contribute their clean content to the canvas, so pass-1
mean blending recovers the underlying conversation content cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


@dataclass
class CircleSpec:
    """A persistent screen-fixed circular overlay across many bands."""
    cx: int
    cy: int
    r: int
    prevalence: float           # fraction of bands where this was detected
    r_min: int = 0
    r_max: int = 0
    n_detected: int = 0
    n_total: int = 0

    def to_dict(self) -> dict:
        return {
            "cx": int(self.cx), "cy": int(self.cy), "r": int(self.r),
            "prevalence": round(float(self.prevalence), 4),
            "r_min": int(self.r_min), "r_max": int(self.r_max),
            "n_detected": int(self.n_detected),
            "n_total": int(self.n_total),
        }


@dataclass
class BandDetection:
    """Result of a per-band targeted detection."""
    detected: bool
    cx: int = -1
    cy: int = -1
    r: int = -1
    n_circles_in_roi: int = 0
    score_distance: float = -1.0  # px distance from expected center; -1 if not detected
    pixels_masked: int = 0


def _to_gray(band: np.ndarray) -> np.ndarray:
    if band.ndim == 2:
        return band
    if band.shape[-1] == 4:
        band = band[..., :3]
    # bands from ffmpeg pipes are RGB; weights don't matter much for Hough
    return cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)


def _filled_disk_mask(
    shape: tuple[int, int], cx: int, cy: int, r: int,
) -> np.ndarray:
    """Return a (H, W) bool mask with True inside the disk of radius r."""
    H, W = shape
    if r <= 0 or cx < 0 or cy < 0 or cx >= W or cy >= H:
        return np.zeros((H, W), dtype=bool)
    y0 = max(0, cy - r)
    y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r)
    x1 = min(W, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = inside
    return mask


def discover_persistent_circles(
    bands: Iterable[np.ndarray],
    *,
    min_prevalence: float = 0.4,
    r_min: int = 20,
    r_max: int = 100,
    bin_px: int = 8,
    param1: int = 100,
    param2: int = 30,
    min_dist: int = 60,
    dp: float = 1.2,
    median_blur_ksize: int = 5,
) -> tuple[list[CircleSpec], dict]:
    """Find screen-fixed circular overlays present in ≥min_prevalence of bands.

    Runs an unconstrained ``cv2.HoughCircles`` per band, bins all hits in
    (cx, cy) space at ``bin_px`` resolution, and promotes any bin with
    prevalence ≥ ``min_prevalence`` to a tracked overlay.

    Parameters
    ----------
    bands : iterable of (H, W) gray or (H, W, 3) RGB uint8 bands.
    min_prevalence : float in (0, 1]. Bin promoted to overlay when its hit
        count divided by the number of bands processed meets this.
    r_min, r_max : Hough radius sweep bounds (px).
    bin_px : spatial bin size for the (cx, cy) histogram. Default 8 px
        gives ±4 px tolerance per bin.
    param1, param2, min_dist, dp : forwarded to ``cv2.HoughCircles``.
    median_blur_ksize : pre-Hough ``cv2.medianBlur`` kernel size; 0 to skip.

    Returns
    -------
    (specs, report)
        specs : list of CircleSpec sorted by prevalence desc.
        report : dict with n_bands, n_unique_bins, top_bins, params.
    """
    bands_list = list(bands)
    n_bands = len(bands_list)
    if n_bands == 0:
        return [], {"n_bands": 0, "skipped": True}

    bin_counts: dict[tuple[int, int], list[int]] = {}
    for band in bands_list:
        gray = _to_gray(band)
        if median_blur_ksize and median_blur_ksize >= 3:
            gray = cv2.medianBlur(gray, int(median_blur_ksize))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2,
            minRadius=r_min, maxRadius=r_max,
        )
        if circles is None:
            continue
        # circles[0] shape (N, 3) = (cx, cy, r)
        # Keep only the closest hit per (cx_bin, cy_bin) within this band so
        # one band can never contribute more than one vote to a bin.
        seen_this_band: set[tuple[int, int]] = set()
        for cx, cy, cr in circles[0]:
            cxi, cyi, cri = int(round(cx)), int(round(cy)), int(round(cr))
            key = (cxi // bin_px, cyi // bin_px)
            if key in seen_this_band:
                continue
            seen_this_band.add(key)
            bin_counts.setdefault(key, []).append(cri)

    # Promote bins that meet the prevalence threshold.
    min_count = max(1, int(np.ceil(min_prevalence * n_bands)))
    specs: list[CircleSpec] = []
    for key, rs in sorted(bin_counts.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < min_count:
            continue
        cx_px = key[0] * bin_px + bin_px // 2
        cy_px = key[1] * bin_px + bin_px // 2
        r_med = int(np.median(rs))
        specs.append(CircleSpec(
            cx=cx_px, cy=cy_px, r=r_med,
            prevalence=len(rs) / float(n_bands),
            r_min=int(np.min(rs)), r_max=int(np.max(rs)),
            n_detected=len(rs), n_total=n_bands,
        ))

    top_bins = sorted(
        ((k, len(v)) for k, v in bin_counts.items()),
        key=lambda kv: -kv[1],
    )[:10]
    report = {
        "n_bands": n_bands,
        "n_unique_bins": len(bin_counts),
        "min_prevalence": float(min_prevalence),
        "min_count": min_count,
        "bin_px": int(bin_px),
        "top_bins": [
            {"cx_px": k[0] * bin_px + bin_px // 2,
             "cy_px": k[1] * bin_px + bin_px // 2,
             "count": c}
            for k, c in top_bins
        ],
        "params": {
            "r_min": int(r_min), "r_max": int(r_max),
            "param1": int(param1), "param2": int(param2),
            "min_dist": int(min_dist), "dp": float(dp),
            "median_blur_ksize": int(median_blur_ksize),
        },
        "specs": [s.to_dict() for s in specs],
    }
    return specs, report


def detect_circle_in_band(
    band: np.ndarray,
    *,
    expected_cx: int,
    expected_cy: int,
    expected_r: int,
    slack_xy: int = 10,
    slack_r: int = 4,
    pad: int = 4,
    param1: int = 100,
    param2: int = 30,
    dp: float = 1.2,
    median_blur_ksize: int = 5,
    min_dist: int = 60,
) -> tuple[np.ndarray, BandDetection]:
    """Detect a single circular overlay near an expected position.

    Runs ``cv2.HoughCircles`` inside a tight ROI centered on
    ``(expected_cx, expected_cy)`` with size ``(expected_r + slack_xy) * 2``.
    Returns the best detection (closest center to expected) within the ROI
    and a band-shaped boolean mask of the detected disk plus ``pad`` AA
    pixels. If no detection is found, returns an empty mask and
    ``BandDetection(detected=False)``.

    Parameters
    ----------
    band : (H, W) gray or (H, W, 3) RGB uint8.
    expected_cx, expected_cy, expected_r : center / radius to search near.
    slack_xy : allowed center offset from expected (px).
    slack_r : allowed radius deviation from expected (px). Hough radius
        sweep becomes [expected_r - slack_r, expected_r + slack_r].
    pad : additional pixels added to the detected radius when drawing the
        boolean mask (catches AA outline). Mask radius = detected_r + pad.
    param1, param2, dp, median_blur_ksize, min_dist : Hough params.

    Returns
    -------
    (mask, detection)
        mask : (H, W) bool, True at the disk to suppress.
        detection : BandDetection with hit details (or detected=False).
    """
    gray_full = _to_gray(band)
    H, W = gray_full.shape

    x0 = max(0, expected_cx - expected_r - slack_xy)
    x1 = min(W, expected_cx + expected_r + slack_xy + 1)
    y0 = max(0, expected_cy - expected_r - slack_xy)
    y1 = min(H, expected_cy + expected_r + slack_xy + 1)
    roi = gray_full[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros((H, W), dtype=bool), BandDetection(detected=False)

    if median_blur_ksize and median_blur_ksize >= 3:
        ksize = int(median_blur_ksize)
        if ksize % 2 == 0:
            ksize += 1
        roi_blur = cv2.medianBlur(roi, ksize)
    else:
        roi_blur = roi

    r_lo = max(1, expected_r - slack_r)
    r_hi = max(r_lo + 1, expected_r + slack_r)
    circles = cv2.HoughCircles(
        roi_blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=r_lo, maxRadius=r_hi,
    )
    if circles is None:
        return np.zeros((H, W), dtype=bool), BandDetection(detected=False)

    # Best hit = closest center to expected position. ROI is in band coords
    # offset by (y0, x0); convert back.
    best_dist = float("inf")
    best = None
    n_in_roi = 0
    for cx_roi, cy_roi, cr in circles[0]:
        cx_band = int(round(cx_roi)) + x0
        cy_band = int(round(cy_roi)) + y0
        cr_band = int(round(cr))
        dx = cx_band - expected_cx
        dy = cy_band - expected_cy
        d = float(np.hypot(dx, dy))
        if d > slack_xy:
            continue
        n_in_roi += 1
        if d < best_dist:
            best_dist = d
            best = (cx_band, cy_band, cr_band)

    if best is None:
        return np.zeros((H, W), dtype=bool), BandDetection(
            detected=False, n_circles_in_roi=0,
        )

    cx_b, cy_b, cr_b = best
    mask = _filled_disk_mask((H, W), cx_b, cy_b, cr_b + max(0, pad))
    return mask, BandDetection(
        detected=True, cx=cx_b, cy=cy_b, r=cr_b,
        n_circles_in_roi=n_in_roi,
        score_distance=round(best_dist, 2),
        pixels_masked=int(mask.sum()),
    )
