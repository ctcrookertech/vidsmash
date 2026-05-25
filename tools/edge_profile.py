"""Dump a per-row RGB profile across a horizontal slice of a PNG.

Used to diagnose whether the right-edge bubble blunt is iOS-native AA or
introduced by stitching. Prints one row per Y, columns X0..X1.

Usage:
    python tools/edge_profile.py --input path.png --y 650 --x0 1095 --x1 1115
    python tools/edge_profile.py --input path.png --y0 600 --y1 800 --y-step 20 --x0 1095 --x1 1115
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--y", type=int, default=-1, help="Single row (overrides y0/y1)")
    ap.add_argument("--y0", type=int, default=-1)
    ap.add_argument("--y1", type=int, default=-1)
    ap.add_argument("--y-step", type=int, default=1)
    ap.add_argument("--x0", type=int, required=True)
    ap.add_argument("--x1", type=int, required=True)
    ap.add_argument("--channel", choices=("R", "G", "B", "luma", "all"), default="luma",
                    help="Which channel to print (luma = 0.299R+0.587G+0.114B)")
    args = ap.parse_args()

    img = np.asarray(Image.open(args.input).convert("RGB"))
    H, W, _ = img.shape

    if args.y >= 0:
        rows = [args.y]
    else:
        y0 = args.y0 if args.y0 >= 0 else 0
        y1 = args.y1 if args.y1 >= 0 else H
        rows = list(range(y0, y1, args.y_step))

    x0, x1 = args.x0, args.x1
    print(f"# {args.input.name}  shape=({H},{W})  x={x0}..{x1-1}  channel={args.channel}")
    header = "  y  | " + " ".join(f"{x:4d}" for x in range(x0, x1))
    print(header)
    print("-" * len(header))

    for y in rows:
        if y < 0 or y >= H:
            continue
        slc = img[y, x0:x1, :].astype(np.float32)
        if args.channel == "luma":
            vals = 0.299 * slc[:, 0] + 0.587 * slc[:, 1] + 0.114 * slc[:, 2]
            row = " ".join(f"{int(round(v)):4d}" for v in vals)
        elif args.channel == "all":
            row = " ".join(f"({int(s[0]):3d},{int(s[1]):3d},{int(s[2]):3d})" for s in slc)
        else:
            ch = {"R": 0, "G": 1, "B": 2}[args.channel]
            row = " ".join(f"{int(s[ch]):4d}" for s in slc)
        print(f"{y:5d} | {row}")


if __name__ == "__main__":
    main()
