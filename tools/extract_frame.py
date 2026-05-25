"""Extract a single frame from a video as PNG via ffmpeg.

Usage:
  python tools/extract_frame.py --input lexi_iphone_messenger_all.mp4 --frame 1500 --out debug_overlays/frame_1500.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def resolve_ffmpeg() -> str:
    candidates = [
        r"C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
        "ffmpeg",
    ]
    for c in candidates:
        if Path(c).exists() or c == "ffmpeg":
            return c
    raise FileNotFoundError("ffmpeg not found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-i", str(args.input),
        "-vf", f"select=eq(n\\,{args.frame})",
        "-vframes", "1",
        "-y", str(args.out),
    ]
    print("[run]", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        return r.returncode
    print(f"[ok] frame {args.frame} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
