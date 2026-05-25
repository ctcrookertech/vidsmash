"""Inspect brightness/colour distribution in a canvas region.

Usage:
    python tools/inspect_brightness.py --input out/stitch/keyframe_chunk_004.png \\
        --x0 850 --x1 1126 --y0 0 --y1 4096 --bg 0,0,0

Prints:
- Modal pixel and frequency
- Histogram of max(|RGB - bg|) ("brightness above background")
- Top-N most frequent distinct colours
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, required=True)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, required=True)
    ap.add_argument("--bg", type=str, default="0,0,0",
                    help="Background RGB as 'R,G,B'. Brightness = max(|RGB - bg|)")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Show top N most frequent colours")
    args = ap.parse_args()

    img = np.array(Image.open(args.input).convert("RGB"))
    H, W, _ = img.shape
    x0 = max(0, args.x0); x1 = min(W, args.x1)
    y0 = max(0, args.y0); y1 = min(H, args.y1)
    crop = img[y0:y1, x0:x1, :]
    npx = crop.shape[0] * crop.shape[1]
    print(f"[input] {args.input} {W}x{H}  crop=[{x0}..{x1}, {y0}..{y1}]  pixels={npx}")

    bg = np.array([int(v) for v in args.bg.split(",")], dtype=np.int16)
    diff = np.abs(crop.astype(np.int16) - bg).max(axis=2)  # (h, w)

    # Brightness histogram (buckets of 8)
    hist_bins = list(range(0, 256, 8))
    counts, edges = np.histogram(diff.flatten(), bins=hist_bins + [256])
    print(f"\n[brightness histogram]  bucket = max(|RGB - bg{tuple(bg.tolist())}|)")
    cum = 0
    for i, c in enumerate(counts):
        if c == 0:
            continue
        cum += c
        pct = 100.0 * c / npx
        cum_pct = 100.0 * cum / npx
        bar = "#" * min(60, int(pct))
        print(f"  [{edges[i]:3d}..{edges[i+1]:3d}) {c:>9d}  {pct:6.2f}%  cum={cum_pct:6.2f}%  {bar}")

    # Top-N distinct colours (packed uint32)
    packed = (crop[..., 0].astype(np.uint32) << 16
              | crop[..., 1].astype(np.uint32) << 8
              | crop[..., 2].astype(np.uint32))
    flat = packed.flatten()
    unique, freq = np.unique(flat, return_counts=True)
    order = np.argsort(freq)[::-1][: args.top_n]
    print(f"\n[top {args.top_n} colours]")
    for i, idx in enumerate(order):
        packed_v = int(unique[idx])
        r, g, b = (packed_v >> 16) & 0xFF, (packed_v >> 8) & 0xFF, packed_v & 0xFF
        c = int(freq[idx])
        pct = 100.0 * c / npx
        d = max(abs(r - int(bg[0])), abs(g - int(bg[1])), abs(b - int(bg[2])))
        print(f"  #{i+1:02d}  RGB({r:3d},{g:3d},{b:3d})  diff={d:3d}  count={c:>9d}  {pct:6.3f}%")


if __name__ == "__main__":
    main()
