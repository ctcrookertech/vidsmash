"""Per-row bubble extent detector using OpenCV Canny + morphology.

Usage:
    python tools/detect_bubble_extents.py --input <png> --out <annotated.png>
        [--y0 291 --y1 1260]
        [--canny-lo 40 --canny-hi 120]
        [--close-w 15 --close-h 3]
        [--min-strong-edge 32]
        [--scrollbar-pair-dx 6]
        [--out-json <extents.json>]

What it does (single frame):
  1. Crop dynamic band [y0..y1] (defaults match lexiconv.mp4 invariants).
  2. Convert to gray, run cv2.Canny(lo, hi).
  3. Morphological close horizontally to bridge text-stroke gaps inside bubbles
     (so the bubble outline is preferred over interior text edges).
  4. For each row, scan right-to-left, find the rightmost x with a strong edge.
  5. Filter scrollbar: if the rightmost edge has a matching opposite-direction
     edge within --scrollbar-pair-dx pixels to its left AND there is a stronger
     edge further left, treat the rightmost as scrollbar and use the inner one.
  6. Emit annotated PNG: red dot at rightmost-edge x per row.
  7. Optionally emit JSON with per-row extent.

Tool deps: opencv-python(-headless), numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def detect_band_extents(
    band_rgb_or_bgr: np.ndarray,
    canny_lo: int = 40,
    canny_hi: int = 120,
    close_w: int = 15,
    close_h: int = 3,
    scrollbar_pair_dx: int = 6,
    is_bgr: bool = False,
    r_exclude_from: int = -1,
    l_exclude_to: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (edges_closed, rightmost_x_per_row, leftmost_x_per_row).

    rightmost/leftmost are int arrays of length H with -1 where no edge found.

    `is_bgr` controls colour conversion (cv2's imread is BGR; ffmpeg pipes
    yield RGB). Grayscale conversion uses BT.601 weights either way; the
    weight difference between BGR2GRAY and RGB2GRAY is irrelevant for Canny.

    `r_exclude_from` (default -1 = auto = W-16): zero edges_closed at
    x >= r_exclude_from before computing rightmost. This kills the iOS
    right-edge scrollbar (and its morphological close spread), preventing
    it from being detected as the bubble's R. Use 0 or W to disable.

    `l_exclude_to` (default -1 = auto = 0): zero edges_closed at
    x < l_exclude_to before computing leftmost. iOS has no left scrollbar
    so auto=0 = no-op. Provided for symmetry.
    """
    if band_rgb_or_bgr.ndim == 3:
        code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(band_rgb_or_bgr, code)
    else:
        gray = band_rgb_or_bgr
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    if close_w > 1 or close_h > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    else:
        edges_closed = edges

    H, W = edges_closed.shape
    r_excl = (W - 16) if r_exclude_from < 0 else min(W, max(0, r_exclude_from))
    l_excl = 0 if l_exclude_to < 0 else min(W, max(0, l_exclude_to))
    if r_excl < W or l_excl > 0:
        edges_for_scan = edges_closed.copy()
        if r_excl < W:
            edges_for_scan[:, r_excl:] = 0
        if l_excl > 0:
            edges_for_scan[:, :l_excl] = 0
    else:
        edges_for_scan = edges_closed

    right = np.full(H, -1, dtype=np.int32)
    left = np.full(H, -1, dtype=np.int32)
    for y in range(H):
        xs = np.where(edges_for_scan[y] > 0)[0]
        if xs.size == 0:
            continue
        rx = int(xs[-1])
        lx = int(xs[0])
        if scrollbar_pair_dx > 0 and xs.size >= 3:
            inner = xs[xs < rx - scrollbar_pair_dx]
            if inner.size > 0 and (rx - int(inner[-1])) <= 40:
                rx = int(inner[-1])
        right[y] = rx
        left[y] = lx
    return edges_closed, right, left


def detect_extents(
    band_bgr: np.ndarray,
    canny_lo: int,
    canny_hi: int,
    close_w: int,
    close_h: int,
    scrollbar_pair_dx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CLI-tool entrypoint: BGR input (from cv2.imread)."""
    return detect_band_extents(
        band_bgr, canny_lo, canny_hi, close_w, close_h, scrollbar_pair_dx,
        is_bgr=True,
    )


def annotate(
    band_bgr: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
) -> np.ndarray:
    out = band_bgr.copy()
    H, W = out.shape[:2]
    for y in range(H):
        rx = int(right[y])
        lx = int(left[y])
        if 0 <= rx < W:
            cv2.circle(out, (rx, y), 2, (0, 0, 255), -1)
        if 0 <= lx < W:
            cv2.circle(out, (lx, y), 2, (0, 255, 0), -1)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--y0", type=int, default=291)
    p.add_argument("--y1", type=int, default=1260)
    p.add_argument("--canny-lo", type=int, default=40)
    p.add_argument("--canny-hi", type=int, default=120)
    p.add_argument("--close-w", type=int, default=15)
    p.add_argument("--close-h", type=int, default=3)
    p.add_argument("--scrollbar-pair-dx", type=int, default=6)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-edges", type=Path, default=None,
                   help="Optional: save the (closed) Canny edges as PNG")
    args = p.parse_args()

    if not args.input.exists():
        print(f"[err] missing input {args.input}", file=sys.stderr)
        return 2

    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[err] could not decode {args.input}", file=sys.stderr)
        return 2
    H, W = img.shape[:2]
    y0 = max(0, args.y0)
    y1 = min(H, args.y1)
    band = img[y0:y1, :, :]
    print(f"[input] {args.input} {W}x{H}  band=[y{y0}..y{y1}] {band.shape[1]}x{band.shape[0]}")

    t0 = time.perf_counter()
    edges, right, left = detect_extents(
        band,
        args.canny_lo,
        args.canny_hi,
        args.close_w,
        args.close_h,
        args.scrollbar_pair_dx,
    )
    dt = time.perf_counter() - t0
    valid_r = (right >= 0).sum()
    valid_l = (left >= 0).sum()
    r_med = int(np.median(right[right >= 0])) if valid_r else -1
    r_max = int(right.max()) if valid_r else -1
    l_med = int(np.median(left[left >= 0])) if valid_l else -1
    l_min = int(left[left >= 0].min()) if valid_l else -1
    print(f"[canny] lo={args.canny_lo} hi={args.canny_hi}  close={args.close_w}x{args.close_h}")
    print(f"[stats] {dt*1000:.1f} ms  rows={band.shape[0]}  "
          f"R: valid={valid_r} median={r_med} max={r_max}  "
          f"L: valid={valid_l} median={l_med} min={l_min}")

    annotated = annotate(band, right, left)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), annotated)
    print(f"[out] annotated -> {args.out}")

    if args.out_edges is not None:
        cv2.imwrite(str(args.out_edges), edges)
        print(f"[out] edges -> {args.out_edges}")

    if args.out_json is not None:
        data = {
            "input": str(args.input),
            "band_y0": y0,
            "band_y1": y1,
            "width": int(band.shape[1]),
            "height": int(band.shape[0]),
            "canny_lo": args.canny_lo,
            "canny_hi": args.canny_hi,
            "close": [args.close_w, args.close_h],
            "scrollbar_pair_dx": args.scrollbar_pair_dx,
            "rightmost_per_row": right.tolist(),
            "leftmost_per_row": left.tolist(),
        }
        args.out_json.write_text(json.dumps(data))
        print(f"[out] json -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
