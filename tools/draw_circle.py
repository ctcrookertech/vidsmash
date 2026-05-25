"""Annotate a single frame with a debug circle at known (cx, cy, r) coordinates.

Used to inspect what HoughCircles is actually detecting (vs what we see).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cx", type=int, required=True)
    ap.add_argument("--cy", type=int, required=True,
                    help="Y in input image coordinates (NOT band coordinates)")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    img = np.asarray(Image.open(args.input).convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, (args.cx, args.cy), args.r, (0, 255, 0), 2)
    cv2.circle(bgr, (args.cx, args.cy), 2, (0, 0, 255), 3)
    label = args.label or f"({args.cx},{args.cy}) r={args.r}"
    cv2.putText(bgr, label,
                (max(0, args.cx - 80), max(15, args.cy - args.r - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(args.out)
    print(f"[draw] {args.input.name} -> {args.out.name}  circle @ ({args.cx},{args.cy}) r={args.r}")


if __name__ == "__main__":
    main()
