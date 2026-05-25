"""Velocity-based pause detection (Strategy B, v2).

WHY v2: v1 used pixel-hash equality on the dynamic band to find pauses. That
fails because (a) the iOS top-status-bar clock ticks (if included), and more
fundamentally (b) frames during a slow scroll differ ONLY by a few rows of
vertical translation -- their absolute pixel bytes diverge even when nothing
"changes" in the conversation. A typical scrolling pair has MAD ~13 unshifted
but <1 at the correct vertical offset.

The right signal is SCROLL VELOCITY dy(t): the vertical shift that best aligns
frame t+1 with frame t. A pause = dy ~ 0 for a sustained run of frames.

Approach
--------
PASS 1 (cheap, one ffmpeg decode):
  For each frame:
    - Extract dyn band (rows top..bot, all cols).
    - Compute K-segmented per-row luma profile (H_dyn x K). Same signature
      Strategy A uses (luma_row_profile).
    - Store profile + a small downsampled "stripe" for the timeline viz.
  Memory: 3060 frames * 969 rows * 16 segments * 4 bytes = ~190 MB. Fine.

PASS 2 (no decode, just numpy):
  For each consecutive (i, i+1), call match_1d_offset(prev_profile,
  cur_profile, predicted_p=0, search_radius=R) and record:
    dy[i+1]   = best p
    mad[i+1]  = MAD at best p
    conf[i+1] = second-best/best ratio

PAUSE DETECTION:
  A frame i is "stationary" iff |dy[i]| <= dy_threshold AND mad[i] <=
  mad_threshold. A pause group = run of consecutive stationary frames of
  length >= min_pause_len.

KEYFRAME: midpoint of each pause group. Choosing a frame from inside a true
stationary run guarantees zero motion-blur and lets pairwise stitching be
done with exact-pixel alignment.

OUTPUT (--out controls the directory; per-video convention is out/<video_basename>/):
  <out>/keyframes.json    (same schema as v1 + dy_series, mad_series)
  <out>/timeline.png      (same layout; bottom heatmap is now |dy| instead of
                           MAD, since |dy| is what actually distinguishes
                           pauses from motion)

Limitations / future work:
  - Drag events (horizontal motion) also produce |dy| ~= 0 in this signal.
    For now they appear as "pauses" too. The stitcher will need to detect
    drag separately (via column-profile horizontal slide-search) and either
    skip them or process them as metadata. A stub flag `drag_suspect` is
    included per pause group when its first frame's column profile differs
    significantly from its neighbors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_io import (  # type: ignore  # noqa: E402
    _resolve_ffmpeg,
    close_proc,
    detect_static_bands,
    gray_col_profile,
    gray_row_profile,
    match_1d_offset,
    open_rgb_pipe,
    probe_video,
    read_frame,
)


# ---------------------------------------------------------------------------
# Pause-run detection
# ---------------------------------------------------------------------------

def find_pause_runs(
    dy: np.ndarray,
    mad: np.ndarray,
    dy_threshold: int,
    mad_threshold: float,
    min_len: int,
) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) runs of consecutive stationary frames.

    Frame i is stationary iff |dy[i]| <= dy_threshold AND mad[i] <=
    mad_threshold. dy[0] is undefined (no previous frame); we treat frame 0
    as stationary by convention.
    """
    n = dy.shape[0]
    stat = np.zeros(n, dtype=bool)
    stat[0] = True
    stat[1:] = (np.abs(dy[1:]) <= dy_threshold) & (mad[1:] <= mad_threshold)

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not stat[i]:
            i += 1
            continue
        j = i + 1
        while j < n and stat[j]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def coalesce_runs(
    runs: list[tuple[int, int]],
    dy: np.ndarray,
    mad: np.ndarray,
    inter_dy_threshold: int,
    inter_mad_threshold: float,
) -> list[tuple[int, int]]:
    """Greedily absorb single-frame motion blips between adjacent pauses.

    For each gap of length <= 2 frames between consecutive pause runs, if all
    frames in the gap have |dy| <= inter_dy_threshold and mad <=
    inter_mad_threshold, merge the two runs together.
    """
    if not runs:
        return runs
    merged: list[list[int]] = [list(runs[0])]
    for s, e in runs[1:]:
        prev_e = merged[-1][1]
        gap = s - prev_e
        if 0 < gap <= 2:
            gap_dy = np.abs(dy[prev_e:s])
            gap_mad = mad[prev_e:s]
            if (gap_dy <= inter_dy_threshold).all() and (gap_mad <= inter_mad_threshold).all():
                merged[-1][1] = e
                continue
        merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------------------

def render_timeline(
    out_path: Path,
    n_frames: int,
    dy: np.ndarray,
    mad: np.ndarray,
    pauses: list[tuple[int, int]],
    keyframes: list[int],
    drag_suspects: set[int],
    fps: float,
) -> None:
    """3-stripe timeline: classification | |dy| heatmap | tick row.

    Bright-green columns mark pauses; cyan = keyframe; orange = drag-suspect
    pause; dark gray = transition. |dy| heatmap: 0 = green, growing |dy| =
    yellow then red, log-scaled.
    """
    W = int(n_frames)
    H_class = 40
    H_dy = 60
    H_tick = 24
    H = H_class + H_dy + H_tick

    img = np.full((H, W, 3), 32, dtype=np.uint8)

    # --- classification stripe ---
    img[0:H_class, :, :] = (60, 60, 70)
    for k, (s, e) in enumerate(pauses):
        color = (220, 140, 0) if k in drag_suspects else (40, 170, 40)
        img[0:H_class, s:e, :] = color
        for col in (s, max(s, e - 1)):
            edge = (255, 200, 0) if k in drag_suspects else (60, 220, 60)
            img[0:H_class, col:col + 1, :] = edge
    for kf in keyframes:
        if 0 <= kf < W:
            img[0:H_class, kf:kf + 1, :] = (0, 220, 255)

    # --- |dy| heatmap (log scale, capped at p99) ---
    abs_dy = np.abs(dy).astype(np.float32)
    log_d = np.log1p(abs_dy)
    cap = float(np.percentile(log_d, 99)) if log_d.size else 0.0
    norm = np.clip(log_d / cap, 0.0, 1.0) if cap > 0 else np.zeros_like(log_d)
    r = np.where(norm < 0.5, 40 + (220 - 40) * (norm / 0.5), 220)
    g = np.where(norm < 0.5, 170 + (200 - 170) * (norm / 0.5), 200 - (200 - 40) * ((norm - 0.5) / 0.5))
    b = np.full_like(r, 40)
    dy_band = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img[H_class:H_class + H_dy, :W, :] = dy_band[None, :, :]

    # --- tick row ---
    tick_y0 = H_class + H_dy
    img[tick_y0:H, :, :] = (20, 20, 24)
    for f in range(0, W, 100):
        thick = (f % 500 == 0)
        col = (220, 220, 220) if thick else (140, 140, 150)
        w = 2 if thick else 1
        img[tick_y0:tick_y0 + (16 if thick else 10), f:f + w, :] = col

    base = Image.fromarray(img)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for f in range(0, W, 500):
        sec = f / fps if fps > 0 else 0
        draw.text((f + 3, tick_y0 + 6), f"{f}  ({sec:.1f}s)", fill=(230, 230, 230), font=font)
    legend = (
        "GREEN=pause  ORANGE=drag-suspect  CYAN=keyframe  GRAY=transition  |  "
        "|dy| heatmap (green=0 -> red=max, log)"
    )
    draw.rectangle([(2, 2), (4 + 7 * len(legend), 16)], fill=(0, 0, 0))
    draw.text((4, 2), legend, fill=(230, 230, 230), font=font)
    base.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-pause-len", type=int, default=3,
        help="Minimum consecutive stationary frames to register as a pause "
             "(default 3; user paused ~50ms+ at each unique state).",
    )
    ap.add_argument(
        "--dy-threshold", type=int, default=1,
        help="|dy| must be <= this to count as stationary (default 1 row).",
    )
    ap.add_argument(
        "--mad-threshold", type=float, default=2.0,
        help="MAD at best dy must be <= this to count as stationary "
             "(default 2.0 luma units).",
    )
    ap.add_argument(
        "--search-radius", type=int, default=400,
        help="Slide-search radius (pixels) for inter-frame dy. Default 400 "
             "covers ~400 px/frame scroll velocity; bump if scrolls are faster.",
    )
    ap.add_argument(
        "--n-segments", type=int, default=16,
        help="Columns per row signature (matches Strategy A default).",
    )
    ap.add_argument(
        "--hpad", type=int, default=40,
        help="Inner horizontal pad for row signatures (avoids scroll-bar / "
             "edge avatars dominating the signal).",
    )
    ap.add_argument(
        "--drag-col-threshold", type=float, default=4.0,
        help="Per-pause drag-suspect flag: if mean abs-diff of column "
             "profile vs prior pause's column profile exceeds this, mark "
             "this pause as drag-suspect (stub for later drag handling).",
    )
    ap.add_argument(
        "--coalesce", action="store_true", default=True,
        help="Coalesce small motion blips (<=2 frames) between adjacent pauses.",
    )
    ap.add_argument("--no-coalesce", action="store_false", dest="coalesce")
    ap.add_argument("--save-frames", action="store_true",
                    help="Write each keyframe as PNG under <out_dir>/keyframes/")
    ap.add_argument("--std-threshold", type=float, default=12.0)
    ap.add_argument("--min-static-run", type=int, default=16)
    ap.add_argument("--dynamic-top", type=int, default=-1)
    ap.add_argument("--dynamic-bottom", type=int, default=-1)
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg, _ = _resolve_ffmpeg()
    _, ffprobe = _resolve_ffmpeg()
    vinfo = probe_video(ffprobe, args.input)
    W, H = vinfo["width"], vinfo["height"]
    fps = vinfo["fps"]
    print(f"[probe] {W}x{H}, {vinfo['nb_frames']} frames, {fps:.3f} fps")

    if args.dynamic_top >= 0 and args.dynamic_bottom >= 0:
        top, bot = args.dynamic_top, args.dynamic_bottom
        print(f"[ui] dynamic band overridden: {top}..{bot}")
    else:
        print("[ui] detecting static bands...")
        top, bot = detect_static_bands(
            ffmpeg=ffmpeg, path=args.input, width=W, height=H,
            nb_frames=vinfo["nb_frames"], n_samples=32,
            std_threshold=args.std_threshold, min_run=args.min_static_run,
        )
        print(f"[ui] static_top=0..{top}  dynamic={top}..{bot}  static_bot={bot}..{H}")
    dyn_h = bot - top
    fbytes = W * dyn_h  # gray pipe + ffmpeg-side crop: 1 byte per pixel

    # ----- Pass 1: decode (gray + crop) + profiles -----
    profiles: list[np.ndarray] = []     # row signature per frame: (dyn_h, K)
    col_profiles: list[np.ndarray] = []  # column signature per frame: 1D
    t_pass1_start = time.perf_counter()
    t_decode = 0.0
    t_profile = 0.0
    # ffmpeg-side crop drops pipe bandwidth ~2.5x for the dyn band of
    # lexi_iphone_messenger_all.mp4 (1126x969 of 1126x2436). Measured 3.11x
    # decode speedup vs full-frame gray pipe; see bench_ffmpeg_pipes.py
    # option 4 + AGENTS.md "Performance -> A2".
    proc = open_rgb_pipe(
        ffmpeg, args.input, pix_fmt="gray",
        crop=(W, dyn_h, 0, top),
    )
    n_frames = 0
    try:
        while True:
            _t = time.perf_counter()
            buf = read_frame(proc, fbytes)
            t_decode += time.perf_counter() - _t
            if buf is None:
                break
            _t = time.perf_counter()
            dyn = np.frombuffer(buf, dtype=np.uint8).reshape(dyn_h, W)
            profiles.append(gray_row_profile(dyn, hpad=args.hpad, n_segments=args.n_segments))
            col_profiles.append(gray_col_profile(dyn, vpad=10))
            t_profile += time.perf_counter() - _t
            n_frames += 1
            if n_frames % 200 == 0:
                print(f"[pass1] decoded {n_frames} frames")
    finally:
        close_proc(proc)
    t_pass1 = time.perf_counter() - t_pass1_start
    print(f"[pass1] decoded {n_frames} frames total")
    print(
        f"[timing] pass1 total={t_pass1:.2f}s  decode={t_decode:.2f}s  "
        f"profile={t_profile:.2f}s  other={(t_pass1 - t_decode - t_profile):.2f}s"
    )

    # ----- Pass 2: per-pair slide search -----
    t_pass2_start = time.perf_counter()
    dy = np.zeros(n_frames, dtype=np.int32)
    mad = np.zeros(n_frames, dtype=np.float32)
    conf = np.ones(n_frames, dtype=np.float32)
    for i in range(1, n_frames):
        res = match_1d_offset(
            ref=profiles[i - 1], cur=profiles[i],
            predicted_p=0, search_radius=args.search_radius,
            min_overlap=max(50, dyn_h // 8),
            prior_alpha=0.0,
        )
        dy[i] = res.p
        mad[i] = res.sad
        conf[i] = res.confidence
        if i % 200 == 0:
            print(f"[pass2] frame {i}  dy={res.p:+5d}  mad={res.sad:.3f}  conf={res.confidence:.2f}")
    t_pass2 = time.perf_counter() - t_pass2_start
    print(f"[timing] pass2 total={t_pass2:.2f}s  ({t_pass2 / max(1, n_frames - 1) * 1000:.2f} ms/pair)")
    print(
        f"[pass2] dy stats: |dy| mean={float(np.abs(dy).mean()):.2f}  "
        f"median={float(np.median(np.abs(dy))):.1f}  max={int(np.abs(dy).max())}"
    )
    print(
        f"[pass2] mad stats: mean={float(mad[1:].mean()):.3f}  "
        f"median={float(np.median(mad[1:])):.3f}  max={float(mad[1:].max()):.3f}"
    )
    stationary_count = int(((np.abs(dy[1:]) <= args.dy_threshold) & (mad[1:] <= args.mad_threshold)).sum())
    print(f"[pass2] stationary frame count: {stationary_count}/{n_frames - 1}")

    # ----- Detect + coalesce pauses -----
    runs = find_pause_runs(dy, mad, args.dy_threshold, args.mad_threshold, args.min_pause_len)
    print(f"[pause] raw pause runs: {len(runs)} covering {sum(e - s for s, e in runs)} frames")
    if args.coalesce:
        runs = coalesce_runs(
            runs, dy, mad,
            inter_dy_threshold=args.dy_threshold + 1,
            inter_mad_threshold=args.mad_threshold * 2,
        )
        print(f"[coalesce] after coalesce: {len(runs)} runs covering {sum(e - s for s, e in runs)} frames")

    # ----- Drag-suspect flag per pause (stub) -----
    # Compare each pause's first-frame column profile to the prior pause's
    # last-frame column profile; large diff = horizontal drag occurred.
    drag_suspects: set[int] = set()
    for k in range(1, len(runs)):
        prev_last = col_profiles[runs[k - 1][1] - 1]
        cur_first = col_profiles[runs[k][0]]
        col_mad = float(np.abs(prev_last - cur_first).mean())
        if col_mad > args.drag_col_threshold:
            drag_suspects.add(k)
    if drag_suspects:
        print(f"[drag] {len(drag_suspects)} pause(s) flagged drag-suspect: {sorted(drag_suspects)}")

    # ----- Build keyframes -----
    pauses_out = []
    keyframes_out = []
    for k, (s, e) in enumerate(runs):
        length = e - s
        mid = s + length // 2
        pauses_out.append({
            "start": int(s), "end": int(e), "length": int(length),
            "mid": int(mid),
            "drag_suspect": (k in drag_suspects),
        })
        keyframes_out.append({
            "i": int(mid),
            "pause_index": k,
            "pause_length": int(length),
            "drag_suspect": (k in drag_suspects),
        })

    between = []
    for k in range(len(runs) - 1):
        a_end = runs[k][1]
        b_start = runs[k + 1][0]
        if b_start > a_end:
            between.append({
                "from_pause": k, "to_pause": k + 1,
                "gap_frames": int(b_start - a_end),
            })

    result = {
        "video": {
            "w": W, "h": H,
            "dyn_top": int(top), "dyn_bot": int(bot), "dyn_h": int(dyn_h),
            "frames_decoded": int(n_frames),
            "fps": fps,
        },
        "params": {
            "min_pause_len": args.min_pause_len,
            "dy_threshold": args.dy_threshold,
            "mad_threshold": args.mad_threshold,
            "search_radius": args.search_radius,
            "n_segments": args.n_segments,
            "hpad": args.hpad,
            "drag_col_threshold": args.drag_col_threshold,
            "coalesce": args.coalesce,
        },
        "pauses": pauses_out,
        "keyframes": keyframes_out,
        "between_runs": between,
        "dy_series": dy.tolist(),
        "mad_series": [float(x) for x in mad],
        "summary": {
            "n_frames": n_frames,
            "stationary_frames": stationary_count,
            "pause_groups": len(runs),
            "frames_in_pauses": int(sum(e - s for s, e in runs)),
            "drag_suspect_pauses": len(drag_suspects),
            "transitions": len(between),
        },
    }
    (out_dir / "keyframes.json").write_text(json.dumps(result, indent=2))
    print(f"[write] {out_dir / 'keyframes.json'}")

    if args.save_frames and keyframes_out:
        kf_dir = out_dir / "keyframes"
        kf_dir.mkdir(exist_ok=True)
        wanted = {kf["i"]: kf["pause_index"] for kf in keyframes_out}
        fbytes_rgb = W * H * 3
        proc = open_rgb_pipe(ffmpeg, args.input)  # default rgb24
        i = 0
        saved = 0
        try:
            while True:
                buf = read_frame(proc, fbytes_rgb)
                if buf is None:
                    break
                if i in wanted:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
                    Image.fromarray(frame).save(
                        kf_dir / f"kf_{wanted[i]:04d}_f{i:05d}.png"
                    )
                    saved += 1
                i += 1
        finally:
            close_proc(proc)
        print(f"[write] {saved} keyframes under {kf_dir}")

    render_timeline(
        out_dir / "timeline.png",
        n_frames=n_frames, dy=dy, mad=mad,
        pauses=runs,
        keyframes=[kf["i"] for kf in keyframes_out],
        drag_suspects=drag_suspects,
        fps=fps,
    )
    print(f"[write] {out_dir / 'timeline.png'}")

    print()
    print("=== Pause / keyframe summary (velocity-based) ===")
    print(f"frames_decoded         : {n_frames}")
    print(f"stationary frames      : {stationary_count}")
    print(f"pause groups           : {len(runs)}  ({sum(e - s for s, e in runs)} frames)")
    if runs:
        ll = [e - s for s, e in runs]
        print(f"  pause_lengths: min={min(ll)} median={int(np.median(ll))} mean={np.mean(ll):.1f} max={max(ll)}")
    print(f"drag-suspect pauses    : {len(drag_suspects)}")
    print(f"transitions (gaps)     : {len(between)}")
    if between:
        gl = [b['gap_frames'] for b in between]
        print(f"  gap_frames: min={min(gl)} median={int(np.median(gl))} max={max(gl)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
