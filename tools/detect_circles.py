"""Discover circular UI overlays (scroll-to-bottom button, avatars, etc.) per frame.

Streams every frame's dynamic band as gray via ffmpeg, runs cv2.HoughCircles
with a wide radius sweep, and writes a JSON manifest of all detections plus a
binned position-histogram so we can identify stable circular UI elements.

Usage:
    python tools/detect_circles.py --input lexi_iphone_messenger_all.mp4 \\
        --keyframes out/lexi_iphone_messenger_all/keyframes.json \\
        --out out/lexi_iphone_messenger_all/circles \\
        --frame-step 5 --min-r 20 --max-r 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from video_io import (  # noqa: E402
    _resolve_ffmpeg, open_rgb_pipe, read_frame, close_proc,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--keyframes", required=True, type=Path,
                    help="Reads dyn_top / dyn_bot / w / h from this. "
                         "Per-video convention: out/<video_basename>/keyframes.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frame-step", type=int, default=1,
                    help="Process every Nth frame (default 1 = every frame)")
    ap.add_argument("--min-r", type=int, default=20,
                    help="HoughCircles minRadius (px). Default 20 covers small avatars.")
    ap.add_argument("--max-r", type=int, default=100,
                    help="HoughCircles maxRadius (px). Default 100 covers ~200px-diameter UI.")
    ap.add_argument("--param1", type=int, default=100,
                    help="HoughCircles Canny upper threshold (param1). Default 100.")
    ap.add_argument("--param2", type=int, default=30,
                    help="HoughCircles accumulator threshold (param2). Lower = more "
                         "candidates / more false positives. Default 30.")
    ap.add_argument("--min-dist", type=int, default=60,
                    help="HoughCircles minDist between detected centers (px). Default 60.")
    ap.add_argument("--dp", type=float, default=1.2,
                    help="HoughCircles inverse accumulator resolution. Default 1.2.")
    ap.add_argument("--bin-px", type=int, default=8,
                    help="Spatial bin size for position histogram (px). Default 8.")
    ap.add_argument("--save-debug", action="store_true",
                    help="Save annotated debug PNGs for first-seen frames in top bins.")
    ap.add_argument("--max-debug", type=int, default=10,
                    help="Max number of debug images to save.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ffmpeg, _ = _resolve_ffmpeg()

    # Read video meta from keyframes.json
    kf = json.loads(args.keyframes.read_text())
    vmeta = kf["video"]
    W = int(vmeta["w"])
    dyn_top = int(vmeta["dyn_top"])
    dyn_bot = int(vmeta["dyn_bot"])
    dyn_h = dyn_bot - dyn_top
    n_frames_meta = int(vmeta.get("n_frames", len(kf["dy_series"]) + 1))

    print(f"[circles] input={args.input.name}  W={W}  dyn=[{dyn_top}..{dyn_bot})  "
          f"dyn_h={dyn_h}  expected_frames={n_frames_meta}  step={args.frame_step}")
    print(f"[circles] HoughCircles  r=[{args.min_r}..{args.max_r}]  dp={args.dp}  "
          f"minDist={args.min_dist}  p1={args.param1}  p2={args.param2}")

    t0 = time.perf_counter()
    proc = open_rgb_pipe(ffmpeg, args.input, pix_fmt="gray",
                         crop=(W, dyn_h, 0, dyn_top))
    frame_bytes = W * dyn_h

    detections: list[dict] = []
    debug_saved = 0
    seen_bins: set[tuple[int, int, int]] = set()
    bin_px = max(1, args.bin_px)

    idx = 0
    while True:
        buf = read_frame(proc, frame_bytes)
        if buf is None:
            break
        if (idx % args.frame_step) != 0:
            idx += 1
            continue
        band = np.frombuffer(buf, dtype=np.uint8).reshape(dyn_h, W)
        # HoughCircles wants slight blur for stability
        blurred = cv2.medianBlur(band, 5)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=args.dp,
            minDist=args.min_dist,
            param1=args.param1, param2=args.param2,
            minRadius=args.min_r, maxRadius=args.max_r,
        )
        if circles is not None:
            for cx, cy, cr in circles[0]:
                cxi, cyi, cri = int(round(cx)), int(round(cy)), int(round(cr))
                bin_key = (cxi // bin_px, cyi // bin_px, cri // bin_px)
                detections.append({
                    "frame": idx, "cx": cxi, "cy": cyi, "r": cri,
                    "bin": bin_key,
                })
                if args.save_debug and bin_key not in seen_bins and debug_saved < args.max_debug:
                    seen_bins.add(bin_key)
                    dbg = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
                    cv2.circle(dbg, (cxi, cyi), cri, (0, 255, 0), 2)
                    cv2.circle(dbg, (cxi, cyi), 2, (0, 0, 255), 3)
                    cv2.putText(dbg, f"f{idx} ({cxi},{cyi}) r={cri}",
                                (max(0, cxi - 80), max(15, cyi - cri - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    out_dbg = args.out / f"debug_f{idx:04d}_cx{cxi}_cy{cyi}_r{cri}.png"
                    cv2.imwrite(str(out_dbg), dbg)
                    debug_saved += 1
        idx += 1

    close_proc(proc)
    dt = time.perf_counter() - t0
    n_processed = (idx + args.frame_step - 1) // args.frame_step

    # Position histogram: count detections per (cx_bin, cy_bin, r_bin) and per (cx_bin, cy_bin)
    bin_counts: dict[tuple[int, int, int], int] = {}
    pos_counts: dict[tuple[int, int], list[int]] = {}  # (cx_bin, cy_bin) -> [r values]
    for d in detections:
        bk = tuple(d["bin"])
        bin_counts[bk] = bin_counts.get(bk, 0) + 1
        pk = (bk[0], bk[1])
        pos_counts.setdefault(pk, []).append(d["r"])

    # Sort bins by frequency desc
    top_bins = sorted(bin_counts.items(), key=lambda kv: kv[1], reverse=True)[:30]
    top_positions = sorted(
        ((pk, rs) for pk, rs in pos_counts.items()),
        key=lambda kv: len(kv[1]), reverse=True,
    )[:20]

    report = {
        "params": {
            "frame_step": args.frame_step, "min_r": args.min_r, "max_r": args.max_r,
            "param1": args.param1, "param2": args.param2, "min_dist": args.min_dist,
            "dp": args.dp, "bin_px": bin_px,
        },
        "frames_processed": n_processed,
        "frames_total": idx,
        "detections": len(detections),
        "wall_s": round(dt, 3),
        "top_bins": [
            {"cx_bin": b[0], "cy_bin": b[1], "r_bin": b[2],
             "cx_px": b[0] * bin_px + bin_px // 2,
             "cy_px": b[1] * bin_px + bin_px // 2,
             "r_px": b[2] * bin_px + bin_px // 2,
             "count": c, "pct_of_frames": round(100.0 * c / n_processed, 2)}
            for b, c in top_bins
        ],
        "top_positions": [
            {"cx_bin": p[0], "cy_bin": p[1],
             "cx_px": p[0] * bin_px + bin_px // 2,
             "cy_px": p[1] * bin_px + bin_px // 2,
             "count": len(rs), "pct_of_frames": round(100.0 * len(rs) / n_processed, 2),
             "r_min": int(min(rs)), "r_max": int(max(rs)),
             "r_mean": round(float(np.mean(rs)), 1),
             "r_median": int(np.median(rs))}
            for p, rs in top_positions
        ],
    }
    (args.out / "circles.json").write_text(json.dumps(report, indent=2))
    print(f"[circles] processed {n_processed} frames in {dt:.2f}s, "
          f"{len(detections)} detections, {len(bin_counts)} unique bins, "
          f"{len(pos_counts)} unique positions, {debug_saved} debug PNGs saved")
    print(f"[circles] report -> {args.out / 'circles.json'}")
    print(f"[circles] top 5 positions (cx,cy,count,r_med):")
    for p in report["top_positions"][:5]:
        print(f"  ({p['cx_px']:4d},{p['cy_px']:4d}) count={p['count']:4d} "
              f"({p['pct_of_frames']:5.1f}% of frames)  r_med={p['r_median']:2d}  "
              f"r_range=[{p['r_min']}..{p['r_max']}]")


if __name__ == "__main__":
    main()
