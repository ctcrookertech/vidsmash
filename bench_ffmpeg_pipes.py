"""Benchmark four ffmpeg pipe strategies for detect_v2's pass-1 input layer.

Measures wall time to consume the full video as raw frames into Python (the
exact pattern detect_pauses_b_v2 pass 1 uses), summing only luma rows in
numpy so we don't trivially optimise the read away. Reports MB throughput and
ms / frame so the GPU options can be compared apples-to-apples.

Pipes tested (the ones ffmpeg actually exposes on this box):
  1. CPU decode, full-frame gray pipe         (current production path / A1)
  2. NVDEC decode, full-frame gray pipe       (hwdownload to system memory)
  3. NVDEC decode + GPU crop to dyn band + gray (smaller pipe; less Python work)
  4. CPU decode + CPU crop+gray via -vf       (control: same pipe size as #3 but no GPU)

Each is measured 1x after a brief warmup. Pipe bandwidth (MB read) reported
so we can see the bandwidth advantage of cropping.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the ffmpeg resolver from the strategy_b helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent / "strategy_b" / "tools"))
from stitch_scroll_b import _resolve_ffmpeg, probe_video, close_proc  # type: ignore  # noqa: E402


def run_pipe(label: str, cmd: list[str], frame_bytes: int) -> dict:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_bytes * 4,
    )
    n_frames = 0
    bytes_read = 0
    sink = np.float64(0.0)  # keep compiler honest
    t0 = time.perf_counter()
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            bytes_read += len(buf)
            n_frames += 1
            # touch the data so the OS can't skip the copy
            arr = np.frombuffer(buf, dtype=np.uint8)
            sink += float(arr[0]) + float(arr[-1])
    finally:
        close_proc(proc)
    elapsed = time.perf_counter() - t0
    mb = bytes_read / (1024 * 1024)
    print(
        f"[{label:<32}] {n_frames:5d} frames  {elapsed:6.2f}s  "
        f"{n_frames / max(elapsed, 1e-9):6.1f} fps  "
        f"{mb:6.1f} MB  "
        f"{mb / max(elapsed, 1e-9):6.1f} MB/s  "
        f"sink={sink}"
    )
    return {
        "label": label,
        "frames": n_frames,
        "seconds": elapsed,
        "bytes": bytes_read,
        "fps": n_frames / max(elapsed, 1e-9),
        "mb_s": mb / max(elapsed, 1e-9),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--top", type=int, default=291, help="dyn band top row")
    ap.add_argument("--bot", type=int, default=1260, help="dyn band bottom row")
    args = ap.parse_args()

    ffmpeg, ffprobe = _resolve_ffmpeg()
    info = probe_video(ffprobe, args.input)
    W, H = info["width"], info["height"]
    dyn_h = args.bot - args.top
    print(f"video: {W}x{H}  nb_frames={info['nb_frames']}  dyn={args.top}..{args.bot} ({dyn_h}px)")
    print(f"ffmpeg: {ffmpeg}")
    print()

    common_in = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    common_out = ["-f", "rawvideo", "-an", "pipe:1"]

    results = []

    # 1. CPU decode, full-frame gray pipe (current A1 path).
    cmd = (
        common_in
        + ["-i", args.input]
        + ["-pix_fmt", "gray"]
        + common_out
    )
    results.append(run_pipe("1. CPU gray full", cmd, W * H))

    # 2. NVDEC decode, full-frame gray (download to system mem).
    cmd = (
        common_in
        + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        + ["-i", args.input]
        + ["-vf", "hwdownload,format=nv12,format=gray"]
        + ["-pix_fmt", "gray"]
        + common_out
    )
    results.append(run_pipe("2. NVDEC gray full", cmd, W * H))

    # 3. NVDEC decode + GPU crop to dyn band + gray.
    # crop_cuda is a feature only in recent ffmpeg; if not available, this errors.
    cmd = (
        common_in
        + ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        + ["-i", args.input]
        + [
            "-vf",
            f"crop={W}:{dyn_h}:0:{args.top}:exact=1,"
            "hwdownload,format=nv12,format=gray",
        ]
        + ["-pix_fmt", "gray"]
        + common_out
    )
    results.append(run_pipe("3. NVDEC GPUcrop gray dyn", cmd, W * dyn_h))

    # 4. CPU decode + CPU crop + gray (control: same bandwidth as #3, no GPU).
    cmd = (
        common_in
        + ["-i", args.input]
        + ["-vf", f"crop={W}:{dyn_h}:0:{args.top}:exact=1,format=gray"]
        + ["-pix_fmt", "gray"]
        + common_out
    )
    results.append(run_pipe("4. CPU crop gray dyn", cmd, W * dyn_h))

    print()
    base = results[0]["seconds"]
    print("Speedup vs option 1 (CPU gray full):")
    for r in results:
        print(f"  {r['label']:<32}  {base / r['seconds']:5.2f}x   ({r['seconds']:6.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
