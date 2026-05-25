"""Characterize per-row variation across uniformly-sampled frames.

Prints, for each row index, the range/std of the row's mean luma across N
samples. Use the output to choose a static-band detection threshold.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


FF = (
    r"C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
)
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")


def probe(path: Path) -> dict:
    import json
    out = subprocess.check_output(
        [FP, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", str(path)]
    )
    s = json.loads(out)["streams"][0]
    return {"W": int(s["width"]), "H": int(s["height"]), "N": int(s["nb_frames"])}


def grab_profiles(path: Path, W: int, H: int, N: int, n_samples: int) -> np.ndarray:
    step = max(1, N // n_samples)
    indices = list(range(0, N, step))[:n_samples]
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    cmd = [FF, "-v", "error", "-i", str(path), "-vf", f"select='{expr}'",
           "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    fb = W * H * 3
    profiles = []
    while True:
        remaining = fb
        chunks = []
        while remaining > 0:
            b = p.stdout.read(remaining)
            if not b:
                break
            chunks.append(b)
            remaining -= len(b)
        if remaining > 0:
            break
        buf = b"".join(chunks)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        luma = (0.299 * arr[..., 0].astype(np.float32)
                + 0.587 * arr[..., 1].astype(np.float32)
                + 0.114 * arr[..., 2].astype(np.float32))
        profiles.append(luma.mean(axis=1))
    p.stdout.close()
    p.wait(timeout=10)
    return np.stack(profiles, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--rows-step", type=int, default=20, help="Print every Nth row.")
    args = ap.parse_args()

    info = probe(args.input)
    print(f"video: {info['W']}x{info['H']}  N={info['N']}")
    print(f"sampling {args.samples} frames...")
    mat = grab_profiles(args.input, info["W"], info["H"], info["N"], args.samples)
    print(f"got {mat.shape[0]} profiles")
    rng = mat.max(0) - mat.min(0)
    std = mat.std(0)
    # Sorted (largest first) std per row index? No, print per-row.
    print(f"\nGlobal stats: rng [min={rng.min():.2f} max={rng.max():.2f} mean={rng.mean():.2f}]  std [min={std.min():.2f} max={std.max():.2f} mean={std.mean():.2f}]")
    print()
    print(f"{'row':>5}  {'rng':>7}  {'std':>7}  rng-bar")
    for r in range(0, info["H"], args.rows_step):
        bar = "#" * min(60, int(rng[r] / 2))
        print(f"{r:>5}  {rng[r]:>7.2f}  {std[r]:>7.2f}  {bar}")

    # Recommend thresholds: try several and report what dynamic-band edges emerge.
    print("\nDetection sweeps (longest contiguous run with min_len=16):")
    for metric_name, metric in (("range", rng), ("std", std)):
        for thr in (4, 8, 12, 18, 25, 40):
            dyn = metric > thr
            best_s, best_l = -1, 0
            i = 0
            H = dyn.shape[0]
            while i < H:
                if dyn[i]:
                    j = i
                    while j < H and dyn[j]:
                        j += 1
                    L = j - i
                    if L >= 16 and L > best_l:
                        best_s, best_l = i, L
                    i = j
                else:
                    i += 1
            if best_l > 0:
                print(f"  {metric_name:>5} > {thr:>2}: dyn = {best_s}..{best_s+best_l} (h={best_l})")
            else:
                print(f"  {metric_name:>5} > {thr:>2}: no run")


if __name__ == "__main__":
    main()
