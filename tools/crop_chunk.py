"""Crop a debug strip from a stitched chunk so overlay artifacts can be inspected.

Reads a single chunk PNG and a screen-space x range (e.g., the right 60 px),
and saves a tall narrow crop. Used to spot iOS scrollbar / scroll-to-latest
button overlays in stitch output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--x0", type=int, default=0)
    ap.add_argument("--x1", type=int, default=-1, help="-1 = image width")
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--y1", type=int, default=-1, help="-1 = image height")
    args = ap.parse_args()

    with Image.open(args.input) as im:
        W, H = im.size
        x0 = args.x0
        x1 = W if args.x1 < 0 else args.x1
        y0 = args.y0
        y1 = H if args.y1 < 0 else args.y1
        crop = im.crop((x0, y0, x1, y1))
        crop.save(args.out)
        print(f"[crop] {args.input} ({W}x{H}) -> {args.out} ({crop.size[0]}x{crop.size[1]}) bbox=({x0},{y0},{x1},{y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
