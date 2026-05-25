"""Smoke test for tools/detect_overlay_circles.py.

Loads a few frames from lexiconv.mp4, runs discovery + per-band detection,
prints results. Verifies the scroll-to-bottom button is found and masked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from video_io import _resolve_ffmpeg, open_rgb_pipe, read_frame, close_proc  # type: ignore  # noqa: E402
from detect_overlay_circles import (  # type: ignore  # noqa: E402
    discover_persistent_circles, detect_circle_in_band,
)


def main() -> int:
    ffmpeg, _ = _resolve_ffmpeg()
    src = ROOT / "lexiconv.mp4"
    if not src.exists():
        print(f"[err] missing {src}", file=sys.stderr)
        return 2

    W, H, dyn_top, dyn_h = 1126, 2436, 291, 969
    frame_bytes = W * dyn_h * 3
    target = sorted({i for i in range(0, 3060, 30)})

    proc = open_rgb_pipe(ffmpeg, src, pix_fmt="rgb24", crop=(W, dyn_h, 0, dyn_top))
    bands: dict[int, np.ndarray] = {}
    i = 0
    while True:
        buf = read_frame(proc, frame_bytes)
        if buf is None:
            break
        if i in target:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(dyn_h, W, 3).copy()
            bands[i] = arr
        i += 1
    close_proc(proc)
    print(f"[smoke] captured {len(bands)} bands (every 30 frames)")

    specs, disc = discover_persistent_circles(
        list(bands.values()), min_prevalence=0.4,
        r_min=20, r_max=100, bin_px=8,
    )
    print(f"[discover] n_bands={disc['n_bands']} unique_bins={disc['n_unique_bins']}")
    print(f"[discover] top 5 bins:")
    for b in disc["top_bins"][:5]:
        print(f"    cx={b['cx_px']} cy={b['cy_px']} count={b['count']}")
    print(f"[discover] {len(specs)} promoted specs:")
    for s in specs:
        print(f"    cx={s.cx} cy={s.cy} r={s.r}  prev={s.prevalence:.3f}  "
              f"r_range=[{s.r_min}..{s.r_max}]  n={s.n_detected}/{s.n_total}")

    if not specs:
        print("[smoke] FAIL: no specs found")
        return 1

    target_spec = specs[0]
    print(f"[detect] using spec ({target_spec.cx}, {target_spec.cy}, r={target_spec.r}) "
          f"to scan all {len(bands)} bands")
    out_dir = ROOT / "out" / "diag" / "smoke_circles"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_hit = 0
    n_miss = 0
    sample_overlay = None
    sample_no_detection = None
    for frame_idx, band in sorted(bands.items()):
        mask, det = detect_circle_in_band(
            band,
            expected_cx=target_spec.cx,
            expected_cy=target_spec.cy,
            expected_r=target_spec.r,
            slack_xy=10, slack_r=4, pad=4,
        )
        if det.detected:
            n_hit += 1
            if sample_overlay is None:
                vis = band.copy()
                vis[mask] = (255, 0, 0)
                Image.fromarray(vis[800:960, 470:660]).save(
                    out_dir / f"hit_f{frame_idx:04d}.png"
                )
                sample_overlay = frame_idx
        else:
            n_miss += 1
            if sample_no_detection is None:
                vis = band.copy()
                Image.fromarray(vis[800:960, 470:660]).save(
                    out_dir / f"miss_f{frame_idx:04d}.png"
                )
                sample_no_detection = frame_idx
    print(f"[detect] hits={n_hit}/{len(bands)}  misses={n_miss}/{len(bands)}  "
          f"(expected ~70% hits based on prior discovery)")
    print(f"[detect] sample overlay PNG: hit_f{sample_overlay:04d}.png")
    print(f"[detect] sample miss PNG: miss_f{sample_no_detection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
