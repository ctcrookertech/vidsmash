"""Dynamic overlay detection: identify screen-fixed UI overlays inside the
dynamic chat band by exploiting their pixel-wise frame-invariance.

Principle
---------
Across many sampled bands taken from different scroll positions, the
underlying chat CONTENT differs per pixel while screen-fixed overlays
(scroll-to-latest button, scroll indicator, timestamp pill, etc.) do not.

A naive "max - min" range test fails for overlays that are CONDITIONALLY
visible (e.g. iOS scroll-to-latest button: hidden when the user is at the
bottom, shown otherwise). Even one absent frame blows the range up.

Approach (agreement-fraction)
-----------------------------
1. Stack N sample bands (RGB, all shape (dyn_h, W, 3)). Reduce to
   grayscale-ish int16 luma.
2. Per-pixel median across N bands → reference value `med`.
3. Per-pixel count of bands whose value lies within ``agreement_tol`` of
   `med` → `n_agree`. Fraction = n_agree / N.
4. Mask = (fraction >= agreement_frac). True at pixels where the dominant
   value commands a large share of bands — i.e. screen-fixed pixels and
   true-background gutter pixels.
5. Dilate by ``dilate`` pixels (square structuring element) to catch
   anti-aliased edges.
6. Drop connected components smaller than ``min_area``.

Returns
-------
mask : (dyn_h, W) bool — True = pixel is an overlay (should be masked when
                         stitching, fall back to non-overlay band pixels).
report : dict with stats (n_components, areas, bboxes, params used).

Notes
-----
- No color or shape priors. Works across light/dark mode, different apps,
  different phone layouts.
- A pixel that is consistently the same colour across the sample set (e.g.
  empty gutter background) will also be masked. This is harmless because
  the 2-pass stitcher fills any unfilled pixel from any covering band, and
  by construction those bands all show the same colour at that pixel.
- Agreement-fraction (e.g. 0.6) tolerates ~40% of bands disagreeing — so
  conditionally-visible overlays still get masked as long as they are
  present in most sampled scroll positions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from scipy.ndimage import binary_dilation, label, find_objects
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def detect_overlay_mask_from_bands(
    bands: Iterable[np.ndarray],
    *,
    agreement_frac: float = 0.6,
    agreement_tol: int = 12,
    dilate: int = 2,
    min_area: int = 30,
    # back-compat: ignored (was used by the older range-based detector).
    range_threshold: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Detect screen-fixed overlay pixels from a set of sampled bands.

    Parameters
    ----------
    bands : iterable of (dyn_h, W, 3) uint8 arrays. All must share shape.
    agreement_frac : float in (0, 1]. Fraction of bands that must agree on
        a dominant value for a pixel to be flagged as overlay. Higher =
        stricter (fewer false positives, may miss flicker overlays).
    agreement_tol : int (0..255). Pixels within this many luma units of the
        per-pixel median count as "agreeing".
    dilate : int. Square structuring element radius in pixels for
        post-detection dilation (catches anti-aliased edges).
    min_area : int. Discard connected components smaller than this.
    range_threshold : DEPRECATED. Kept for back-compat with older callers;
        ignored by the agreement-fraction detector.

    Returns
    -------
    (mask, report)
        mask : (dyn_h, W) bool — True at overlay pixels.
        report : dict with detection stats.
    """
    del range_threshold  # accepted but unused

    arr = np.stack(list(bands), axis=0)  # (N, dyn_h, W, 3) uint8
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"bands must be (dyn_h, W, 3) RGB arrays; got {arr.shape}")
    N, dyn_h, W, _ = arr.shape
    if N < 4:
        return (np.zeros((dyn_h, W), dtype=bool),
                {"n_bands": N, "skipped": True, "reason": "fewer than 4 sample bands"})

    # luma proxy (cheap; channels weighted equally is fine for our purposes)
    luma = arr.astype(np.int16).mean(axis=-1)  # (N, dyn_h, W)
    med = np.median(luma, axis=0)              # (dyn_h, W)
    diff = np.abs(luma - med[None, :, :])      # (N, dyn_h, W)
    n_agree = (diff <= agreement_tol).sum(axis=0).astype(np.int32)  # (dyn_h, W)
    frac = n_agree / float(N)

    raw = frac >= agreement_frac

    if dilate > 0:
        if _HAS_SCIPY:
            struct = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=bool)
            mask = binary_dilation(raw, structure=struct)
        else:
            mask = raw.copy()
            for dy in range(-dilate, dilate + 1):
                for dx in range(-dilate, dilate + 1):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = np.roll(raw, (dy, dx), axis=(0, 1))
                    if dy > 0:
                        shifted[:dy, :] = False
                    elif dy < 0:
                        shifted[dy:, :] = False
                    if dx > 0:
                        shifted[:, :dx] = False
                    elif dx < 0:
                        shifted[:, dx:] = False
                    mask |= shifted
    else:
        mask = raw

    components: list[dict] = []
    if _HAS_SCIPY:
        lab, n = label(mask)
        if n > 0:
            objs = find_objects(lab)
            for k, slc in enumerate(objs, start=1):
                if slc is None:
                    continue
                area = int((lab[slc] == k).sum())
                if area < min_area:
                    mask[slc][lab[slc] == k] = False
                    continue
                y0, y1 = slc[0].start, slc[0].stop
                x0, x1 = slc[1].start, slc[1].stop
                components.append({
                    "label": k,
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "area": area,
                    "aspect": round((x1 - x0) / max(1, y1 - y0), 3),
                })
    n_masked = int(mask.sum())
    report = {
        "n_bands": int(N),
        "dyn_h": int(dyn_h),
        "W": int(W),
        "agreement_frac": float(agreement_frac),
        "agreement_tol": int(agreement_tol),
        "dilate": int(dilate),
        "min_area": int(min_area),
        "pixels_masked": n_masked,
        "pct_masked": round(100.0 * n_masked / (dyn_h * W), 2),
        "n_components": len(components),
        "components": components,
        "scipy_available": _HAS_SCIPY,
    }
    return mask, report
