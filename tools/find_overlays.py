"""Locate the iOS Messenger overlay bboxes (scroll-to-latest circle + scrollbar)
in a single sample frame.

Heuristics:
  - Scroll-to-latest circle: a dark-gray (R≈G≈B, mid luma) filled disk inside
    the dynamic chat band. We find the largest connected component of
    "dark gray" pixels (low color saturation, mid-low luma) inside the band.
  - Scrollbar: a thin light-gray vertical strip near the right edge inside
    the dynamic band. We find the rightmost continuous column of high-luma
    pixels in the right margin.

Prints the bboxes in two forms:
  - absolute frame coordinates (y is from top of frame)
  - in-band coordinates (y is from top of the dynamic band, i.e. y - dyn_top)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def find_circle(rgb: np.ndarray, dyn_top: int, dyn_bot: int) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) of the largest dark-gray disk inside the dyn band, or None."""
    band = rgb[dyn_top:dyn_bot]
    r = band[..., 0].astype(np.int16)
    g = band[..., 1].astype(np.int16)
    b = band[..., 2].astype(np.int16)
    luma = (0.299 * r + 0.587 * g + 0.114 * b)
    max_chan = np.maximum(np.maximum(r, g), b)
    min_chan = np.minimum(np.minimum(r, g), b)
    sat = max_chan - min_chan
    mask = (luma > 30) & (luma < 120) & (sat < 25)  # dark gray, low sat
    # connected components via scipy if available, else fallback to bbox of all-mask
    try:
        from scipy.ndimage import label, find_objects
        lab, n = label(mask)
        if n == 0:
            return None
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0  # ignore background
        best = int(np.argmax(sizes))
        if sizes[best] < 1000:  # minimum disk area
            return None
        slc = find_objects(lab)[best - 1]
        if slc is None:
            return None
        y0_b, y1_b = slc[0].start, slc[0].stop
        x0, x1 = slc[1].start, slc[1].stop
        return (x0, y0_b + dyn_top, x1, y1_b + dyn_top)
    except ImportError:
        # naive fallback: bbox of entire mask
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return (int(xs.min()), int(ys.min()) + dyn_top, int(xs.max()) + 1, int(ys.max()) + dyn_top + 1)


def find_scrollbar(rgb: np.ndarray, dyn_top: int, dyn_bot: int,
                   right_margin: int = 30) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) of the right-edge scrollbar, or None."""
    band = rgb[dyn_top:dyn_bot]
    W = band.shape[1]
    right = band[:, W - right_margin:]
    luma = (0.299 * right[..., 0].astype(np.int16)
            + 0.587 * right[..., 1].astype(np.int16)
            + 0.114 * right[..., 2].astype(np.int16))
    # scrollbar is medium-bright pixels (gray on a dark background)
    bright = luma > 90
    # for each column, count bright pixels with long contiguous runs
    col_counts = bright.sum(axis=0)
    if col_counts.max() < 50:
        return None
    # find columns with at least half the band height as bright
    cand_cols = np.where(col_counts > (dyn_bot - dyn_top) * 0.25)[0]
    if cand_cols.size == 0:
        return None
    x_local_start = int(cand_cols.min())
    x_local_end = int(cand_cols.max()) + 1
    # vertical extent: rows where any candidate column is bright
    sub = bright[:, x_local_start:x_local_end]
    rows_any = sub.any(axis=1)
    ys = np.where(rows_any)[0]
    if ys.size == 0:
        return None
    return (W - right_margin + x_local_start, int(ys.min()) + dyn_top,
            W - right_margin + x_local_end, int(ys.max()) + dyn_top + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--dyn-top", type=int, default=291)
    ap.add_argument("--dyn-bot", type=int, default=1260)
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    print(f"[load] {args.input} {W}x{H}  dyn=[{args.dyn_top},{args.dyn_bot})")

    c = find_circle(rgb, args.dyn_top, args.dyn_bot)
    if c is None:
        print("[circle] not found")
    else:
        x0, y0, x1, y1 = c
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        r = max(x1 - x0, y1 - y0) // 2
        print(f"[circle] abs bbox=({x0},{y0},{x1},{y1})  cx={cx}  cy={cy}  r~{r}")
        print(f"[circle] in-band cy={cy - args.dyn_top}  bbox_y=({y0 - args.dyn_top},{y1 - args.dyn_top})")

    s = find_scrollbar(rgb, args.dyn_top, args.dyn_bot)
    if s is None:
        print("[scrollbar] not found")
    else:
        x0, y0, x1, y1 = s
        print(f"[scrollbar] abs bbox=({x0},{y0},{x1},{y1})  width={x1-x0}px  height={y1-y0}px")
        print(f"[scrollbar] in-band bbox_y=({y0 - args.dyn_top},{y1 - args.dyn_top})  right_strip_from_x={x0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
