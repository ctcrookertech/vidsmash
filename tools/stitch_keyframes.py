"""Stitch keyframes from detect_pauses into ordered vertical chunks.

Algorithm
---------
1. Load keyframes.json (produced by detect_pauses.py). It carries:
     - dy_series: per-frame velocity (cur vs prev)
     - keyframes: representative frames for each pause group
     - pauses, between_runs, drag_suspect flags
     - video meta (dyn_top, dyn_bot, dyn_h, w)
2. Compute cum_y[k] for every keyframe by summing dy_series across the
   inter-keyframe gap. Where chain_dy between consecutive keyframes
   exceeds (dyn_h - overlap_margin), the keyframes alone do not cover all
   content. Walk the transition frames between them and add the minimum
   set of "bridge" frames such that consecutive placed frames overlap by
   at least overlap_margin rows.
3. Decode the source video ONCE (RGB pipe, ffmpeg-side crop to dyn band).
   For each target frame index in (keyframes ∪ bridges) keep its RGB band
   in memory. From frame 0, also keep one full-frame RGB to extract the
   static_top + static_bot UI strips.
4. Optionally re-match each consecutive pair with match_1d_offset; if
   chain_dy and direct_dy disagree by > max_corr_disagreement, log a
   warning. Final cum_y always uses the chain (direct match is only a
   sanity tap for now).
5. Sort placed frames by cum_y, walk in order, append non-overlapping rows
   to a ChunkWriter. Prepend static_top and append static_bot in the
   "keep-once" UI mode (only first/last chunk show chrome).
6. Emit report.json with placements, gaps if any, and drag-suspect flags.

Outputs
-------
out/keyframe_chunk_NNN.png : ordered vertical chunks of the stitched canvas.
out/report.json            : per-frame placement, chain_dy vs direct_dy,
                              chunk metadata, drag-suspect annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_io import (  # type: ignore  # noqa: E402
    ChunkWriter,
    _resolve_ffmpeg,
    close_proc,
    gray_row_profile,
    match_1d_offset,
    open_rgb_pipe,
    probe_video,
    read_frame,
)
from detect_overlays import detect_overlay_mask_from_bands  # type: ignore  # noqa: E402
from detect_bubble_extents import detect_band_extents  # type: ignore  # noqa: E402
from detect_overlay_circles import (  # type: ignore  # noqa: E402
    discover_persistent_circles, detect_circle_in_band, _filled_disk_mask,
)


def select_bridge_frames(
    dys: list[int], k_prev_i: int, k_next_i: int,
    cum_y_prev: int, dyn_h: int, overlap_margin: int,
) -> list[tuple[int, int]]:
    """Return [(frame_index, cum_y_at_that_frame), ...] of bridge frames
    that need to be placed between two keyframes to avoid an uncovered gap.

    A bridge is appended whenever the cumulative position from the last
    placed frame would advance by more than (dyn_h - overlap_margin).
    """
    bridges: list[tuple[int, int]] = []
    last_placed_y = cum_y_prev
    cum_y = cum_y_prev
    for t in range(k_prev_i + 1, k_next_i + 1):
        cum_y += dys[t]
        if t == k_next_i:
            break
        if abs(cum_y - last_placed_y) >= (dyn_h - overlap_margin):
            bridges.append((t, cum_y))
            last_placed_y = cum_y
    return bridges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="Source video (same one passed to detect_pauses).")
    ap.add_argument("--keyframes", required=True, type=Path,
                    help="Path to keyframes.json from detect_pauses.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for chunk PNGs and report.json.")
    ap.add_argument("--chunk-height", type=int, default=4096,
                    help="Rows per output PNG chunk.")
    ap.add_argument("--overlap-margin", type=int, default=120,
                    help="Minimum overlap (px) between adjacent placed frames. "
                         "Drives how aggressively bridge frames are inserted.")
    ap.add_argument("--ui", choices=["keep-once", "strip"], default="keep-once",
                    help="keep-once: prepend static_top to first chunk, "
                         "append static_bot to last chunk. strip: omit UI bands.")
    ap.add_argument("--validate-match", action="store_true",
                    help="Re-match each consecutive pair with match_1d_offset "
                         "and log disagreement vs chain_dy (sanity check).")
    ap.add_argument("--max-disagreement", type=int, default=20,
                    help="Pixel threshold for logging chain-vs-direct disagreement.")
    ap.add_argument("--detect-overlays", choices=["auto", "off"], default="auto",
                    help="auto: derive overlay mask from per-pixel agreement across all "
                         "captured bands (color/layout/app-agnostic). off: no mask.")
    ap.add_argument("--overlay-agreement-frac", type=float, default=0.6,
                    help="Fraction of sample bands that must agree on the dominant "
                         "value for a pixel to be flagged overlay. Higher = stricter "
                         "(fewer false positives, may miss flicker overlays).")
    ap.add_argument("--overlay-agreement-tol", type=int, default=12,
                    help="Luma units within which bands count as 'agreeing' with the "
                         "per-pixel median.")
    ap.add_argument("--overlay-dilate", type=int, default=2,
                    help="Pixels of mask dilation post-detection (catches anti-aliased "
                         "overlay edges).")
    ap.add_argument("--overlay-min-area", type=int, default=30,
                    help="Drop overlay components smaller than this many pixels.")
    ap.add_argument("--overlay-max-strip-width", type=int, default=50,
                    help="Pass-3 horizontally inpaints contiguous runs of "
                         "densely-masked columns up to this width (covers thin "
                         "scrollbars). Wider runs are assumed to be gutters and "
                         "left to median fill.")
    ap.add_argument("--mask-circle", default="",
                    help="MANUAL OVERRIDE (added on top of auto). Comma-separated "
                         "cx,cy_in_band,r. Empty = none.")
    ap.add_argument("--mask-right-strip-from", type=int, default=-1,
                    help="MANUAL OVERRIDE (added on top of auto). x column (inclusive) "
                         "from which to mask to the right edge. -1 = none.")
    ap.add_argument("--no-clean-policy",
                    choices=["median", "variance-gate", "hard-skip", "variance-gate-dilated"],
                    default="median",
                    help="How to fill canvas pixels with clean_count==0 (every covering "
                         "band has overlay at this in-band position). "
                         "median (default): median of all covering bands, RGB output. "
                         "variance-gate: pass 2 fills only where per-pixel sample range "
                         "<= --pass2-variance-tol; else alpha=0 (RGBA output). k=1 always "
                         "transparent. "
                         "hard-skip: never fill via pass 2; every clean_count==0 pixel "
                         "becomes alpha=0 (RGBA output). "
                         "variance-gate-dilated: variance-gate then morphologically "
                         "dilate transparent regions (RGBA output). "
                         "Non-median policies were tested and produced worse visual "
                         "results than median; keep median unless you have a specific "
                         "reason.")
    ap.add_argument("--pass2-variance-tol", type=int, default=16,
                    help="Max per-channel range across covering bands below which a "
                         "pass-2 median is trusted (luma units). Used by variance-gate "
                         "policies. Higher = more permissive fill, more ghost risk.")
    ap.add_argument("--alpha-dilate-iters", type=int, default=2,
                    help="Iterations of binary dilation applied to α=0 regions when "
                         "--no-clean-policy=variance-gate-dilated. Each iteration grows "
                         "the hole by ~1px in each direction.")
    ap.add_argument("--scrollbar-rim-px", type=int, default=0,
                    help="Width (in px) of the rightmost rim to overwrite with the "
                         "background colour after all fill passes. Cleans up the thin "
                         "iOS scrollbar ghost that pass 2 fabricates in the gutter. "
                         "0 disables. Recommended 12 for iOS @3x (scrollbar is ~3pt=9px, "
                         "bubbles end ~48px from edge so 12 never touches bubble content). "
                         "Does NOT address scroll-to-latest button ghosts inside bubbles.")
    ap.add_argument("--scrollbar-rim-bg", type=str, default="auto",
                    help="Background colour for the scrollbar rim fill. 'auto' (default) "
                         "uses the modal pixel of the rightmost 1-px column. Or pass "
                         "'R,G,B' (e.g. '0,0,0') to force a specific colour.")
    ap.add_argument("--clear-beyond-bubble-extent", action="store_true",
                    help="After all fill passes, use OpenCV Canny edge detection to find "
                         "the leftmost/rightmost bubble edge per row in each placed band, "
                         "aggregate across covering bands, and force-fill canvas pixels "
                         "beyond those extents with the gutter background colour. This "
                         "removes wrong-colour gutter pixels that pass 2 can fabricate at "
                         "the canvas left/right edges. Requires opencv-python.")
    ap.add_argument("--bubble-extent-pad", type=int, default=0,
                    help="Number of pixels to keep beyond the detected bubble edge before "
                         "starting the gutter fill. Default 0 (pixel-perfect right at the "
                         "Canny outline). Increase to 1-3 if anti-aliased outline halos "
                         "are visibly clipped on a new input.")
    ap.add_argument("--bubble-extent-bg", type=str, default="auto",
                    help="Gutter colour for --clear-beyond-bubble-extent. 'auto' (default) "
                         "samples the modal pixel of x=0 across all placed bands. Or pass "
                         "'R,G,B' (e.g. '0,0,0') to force a specific colour.")
    ap.add_argument("--canny-lo", type=int, default=40,
                    help="Lower Canny hysteresis threshold for bubble-extent detection.")
    ap.add_argument("--canny-hi", type=int, default=120,
                    help="Upper Canny hysteresis threshold for bubble-extent detection.")
    ap.add_argument("--canny-close-w", type=int, default=15,
                    help="Horizontal morphological-close width (px) after Canny.")
    ap.add_argument("--canny-close-h", type=int, default=3,
                    help="Vertical morphological-close height (px) after Canny.")
    ap.add_argument("--bubble-scrollbar-pair-dx", type=int, default=6,
                    help="Min horizontal distance between the rightmost edge and the next "
                         "inner edge to treat the rightmost as the scrollbar (and prefer the "
                         "inner one). 0 disables this filter.")
    ap.add_argument("--bubble-extent-smooth-radius", type=int, default=30,
                    help="Vertical max-pool radius (px) applied to canvas_R before pass 7 "
                         "clearing. The bubble has a straight vertical right edge spanning "
                         "many rows, so a few text-edge under-detections (where white text "
                         "near the right edge fools Canny) are repaired by propagating the "
                         "true bubble R from neighbouring rows. Window size = 2*radius+1. "
                         "Set 0 to disable. Only applied to R (right side); L is not "
                         "smoothed to avoid weakening left-gutter cleanup at "
                         "incoming/outgoing bubble transitions.")
    ap.add_argument("--bubble-extent-r-max", type=int, default=-1,
                    help="Upper clamp on canvas_R before vertical smoothing, to prevent a "
                         "stray scrollbar-edge detection from inflating R across many rows "
                         "via the max-pool. -1 (default) means W-16 (iOS scrollbar lives "
                         "at the rightmost ~9 px; 16 gives margin). Set to W (e.g. 1126) "
                         "to disable clamping.")
    ap.add_argument("--bubble-detector-r-exclude-from", type=int, default=-1,
                    help="Zero out Canny edges at x >= this value INSIDE the per-band "
                         "detector, before per-row rightmost is computed. This kills the "
                         "iOS scrollbar (and its morphological-close spread) so it cannot "
                         "be reported as a bubble R. -1 (default) means W-16. Set to W "
                         "(e.g. 1126) to disable. The post-detection clamp "
                         "(--bubble-extent-r-max) remains active as defense-in-depth.")
    ap.add_argument("--bubble-extent-synthetic-aa", action="store_true",
                    help="After pass-7 clearing, repaint the rightmost 3 columns of every "
                         "bubble with iOS-native AA gradient (90 -> 63 -> 19 -> 1 -> 0 in "
                         "luma-90-bubble terms; bubble*ratio + bg*(1-ratio) per pixel using "
                         "ratios 0.70/0.20/0.01 at x=R-2/R-1/R). Per-row bubble colour is "
                         "sampled at x=R-3 so blends work for any bubble colour. Rows "
                         "where R-3 is bg-coloured (rounded corners, max-pool-raised rows "
                         "without a real bubble at that R) are skipped via the AA threshold. "
                         "Fixes the 'hard mechanical edge' look from pass-1 sub-pixel-mean "
                         "smearing.")
    ap.add_argument("--bubble-extent-aa-threshold", type=int, default=30,
                    help="Per-row luma distance from bg required at x=R-3 for synthetic AA "
                         "to apply on that row. Below this, the row has no bubble content at "
                         "the detected extent (corner / spurious R) and AA is skipped. "
                         "Default 30 is comfortably below all iOS bubble colours vs black bg.")
    ap.add_argument("--mask-detected-circles", action="store_true",
                    help="Auto-discover screen-fixed circular overlays (e.g. iOS "
                         "scroll-to-latest button) via HoughCircles across all placed "
                         "bands, then per-band redetect inside a tight ROI and add the "
                         "matched disk to a PER-BAND overlay mask. Pass-1 mean and pass-2 "
                         "median both ignore that band's pixels at those positions, so "
                         "bands without the button at the canonical position still "
                         "contribute clean conversation content. Eliminates the dark "
                         "ghost left by partial averaging of the button.")
    ap.add_argument("--circle-min-prevalence", type=float, default=0.4,
                    help="Discovery threshold: a (cx,cy) bin must appear in this fraction "
                         "of placed bands to be promoted to a tracked overlay. Lower = "
                         "more aggressive, may catch transient circles. Default 0.4.")
    ap.add_argument("--circle-r-min", type=int, default=20,
                    help="HoughCircles minRadius for discovery + per-band detection (px). "
                         "Default 20.")
    ap.add_argument("--circle-r-max", type=int, default=100,
                    help="HoughCircles maxRadius for discovery + per-band detection (px). "
                         "Default 100 covers up to ~200 px diameter UI.")
    ap.add_argument("--circle-pad", type=int, default=4,
                    help="Pixels added to detected radius when drawing the per-band mask. "
                         "Catches anti-aliased outline + minor jitter between bands. "
                         "Default 4.")
    ap.add_argument("--circle-param1", type=int, default=100,
                    help="HoughCircles Canny upper threshold. Default 100.")
    ap.add_argument("--circle-param2", type=int, default=30,
                    help="HoughCircles accumulator threshold. Lower = more permissive "
                         "per-band detection (catches faint button overlays on dark "
                         "backgrounds at the cost of more false positives in noisy bands). "
                         "Default 30.")
    ap.add_argument("--circle-slack-xy", type=int, default=10,
                    help="Per-band detection allowed center offset from the discovered "
                         "(cx,cy) (px). Default 10.")
    ap.add_argument("--circle-slack-r", type=int, default=4,
                    help="Per-band detection allowed radius deviation from the discovered "
                         "r (px). Default 4.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = _resolve_ffmpeg()

    # ----- Load keyframes.json -----
    kf_data = json.loads(args.keyframes.read_text())
    dys: list[int] = list(kf_data["dy_series"])
    kfs = kf_data["keyframes"]
    vmeta = kf_data["video"]
    W = int(vmeta["w"])
    H = int(vmeta["h"])
    dyn_top = int(vmeta["dyn_top"])
    dyn_bot = int(vmeta["dyn_bot"])
    dyn_h = int(vmeta["dyn_h"])
    print(f"[load] {len(kfs)} keyframes  dyn={dyn_top}..{dyn_bot} ({dyn_h}px)  W={W}")

    # ----- Compute cum_y for every keyframe + determine bridge frames -----
    placements: list[dict] = []  # {i, cum_y, source: "keyframe"|"bridge", ...}
    cum_y = 0
    prev_i = 0
    placements.append({
        "i": kfs[0]["i"],
        "cum_y": sum(dys[1 : kfs[0]["i"] + 1]),
        "source": "keyframe",
        "pause_index": kfs[0]["pause_index"],
        "drag_suspect": kfs[0]["drag_suspect"],
    })
    cum_y = placements[0]["cum_y"]
    prev_i = kfs[0]["i"]

    bridge_count = 0
    for k_next in kfs[1:]:
        next_i = k_next["i"]
        chain_dy = sum(dys[prev_i + 1 : next_i + 1])
        target_cum_y = cum_y + chain_dy
        # Insert bridges between prev_i and next_i if the leap would exceed
        # dyn_h - overlap_margin.
        if abs(chain_dy) >= (dyn_h - args.overlap_margin):
            bridges = select_bridge_frames(
                dys, prev_i, next_i, cum_y, dyn_h, args.overlap_margin,
            )
            for bi, by in bridges:
                placements.append({
                    "i": bi,
                    "cum_y": by,
                    "source": "bridge",
                    "pause_index": -1,
                    "drag_suspect": False,
                })
                bridge_count += 1
        placements.append({
            "i": next_i,
            "cum_y": target_cum_y,
            "source": "keyframe",
            "pause_index": k_next["pause_index"],
            "drag_suspect": k_next["drag_suspect"],
        })
        cum_y = target_cum_y
        prev_i = next_i
    print(f"[plan] {len(placements)} placements "
          f"({len(kfs)} keyframes + {bridge_count} bridges)")
    print(f"[plan] cum_y range: {min(p['cum_y'] for p in placements)} .. "
          f"{max(p['cum_y'] for p in placements)} "
          f"(span={max(p['cum_y'] for p in placements) - min(p['cum_y'] for p in placements) + dyn_h}px)")

    # ----- Decode source video and capture target RGB bands -----
    wanted = {p["i"] for p in placements}
    print(f"[decode] need RGB for {len(wanted)} frames")
    fbytes_band = W * dyn_h * 3
    rgb_bands: dict[int, np.ndarray] = {}
    fbytes_full = W * H * 3
    static_top_rgb: np.ndarray | None = None
    static_bot_rgb: np.ndarray | None = None

    # Pass 1: extract static UI strips from frame 0 with a full-frame pipe.
    t0 = time.perf_counter()
    proc = open_rgb_pipe(ffmpeg, args.input, pix_fmt="rgb24")
    buf = read_frame(proc, fbytes_full)
    if buf is None:
        raise RuntimeError("could not decode frame 0")
    f0 = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
    close_proc(proc)
    static_top_rgb = f0[:dyn_top].copy()
    static_bot_rgb = f0[dyn_bot:].copy()
    print(f"[decode] static_top={static_top_rgb.shape}  static_bot={static_bot_rgb.shape}  "
          f"({time.perf_counter() - t0:.2f}s)")

    # Pass 2: full video, crop to dyn band, keep only target frames in memory.
    t0 = time.perf_counter()
    proc = open_rgb_pipe(
        ffmpeg, args.input, pix_fmt="rgb24",
        crop=(W, dyn_h, 0, dyn_top),
    )
    i = 0
    try:
        while True:
            buf = read_frame(proc, fbytes_band)
            if buf is None:
                break
            if i in wanted:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(dyn_h, W, 3).copy()
                rgb_bands[i] = arr
            i += 1
            if i % 500 == 0:
                print(f"[decode] scanned {i} frames, captured {len(rgb_bands)}/{len(wanted)}")
    finally:
        close_proc(proc)
    print(f"[decode] scanned {i} frames, captured {len(rgb_bands)}/{len(wanted)} "
          f"({time.perf_counter() - t0:.2f}s)")
    if len(rgb_bands) != len(wanted):
        missing = sorted(wanted - set(rgb_bands.keys()))
        raise RuntimeError(f"missing RGB for {len(missing)} target frames: {missing[:10]}...")

    # ----- Optional: validate cum_y via direct match between consecutive placements -----
    if args.validate_match:
        t0 = time.perf_counter()
        n_check = 0
        n_disagree = 0
        for a, b in zip(placements[:-1], placements[1:]):
            chain_dy = b["cum_y"] - a["cum_y"]
            if abs(chain_dy) >= dyn_h:
                continue  # no overlap, can't direct-match
            band_a = rgb_bands[a["i"]]
            band_b = rgb_bands[b["i"]]
            gray_a = (
                0.299 * band_a[..., 0].astype(np.float32)
                + 0.587 * band_a[..., 1].astype(np.float32)
                + 0.114 * band_a[..., 2].astype(np.float32)
            ).astype(np.uint8)
            gray_b = (
                0.299 * band_b[..., 0].astype(np.float32)
                + 0.587 * band_b[..., 1].astype(np.float32)
                + 0.114 * band_b[..., 2].astype(np.float32)
            ).astype(np.uint8)
            prof_a = gray_row_profile(gray_a, hpad=40, n_segments=16)
            prof_b = gray_row_profile(gray_b, hpad=40, n_segments=16)
            res = match_1d_offset(
                ref=prof_a, cur=prof_b,
                predicted_p=chain_dy, search_radius=50,
                min_overlap=max(50, dyn_h // 8),
                prior_alpha=0.0,
            )
            diff = res.p - chain_dy
            if abs(diff) > args.max_disagreement:
                print(f"[validate] kf i={b['i']} chain_dy={chain_dy:+5d} "
                      f"direct={res.p:+5d} diff={diff:+5d} mad={res.sad:.2f} conf={res.confidence:.2f}")
                n_disagree += 1
            n_check += 1
        print(f"[validate] checked {n_check} overlapping pairs, "
              f"{n_disagree} disagreed by > {args.max_disagreement}px "
              f"({time.perf_counter() - t0:.2f}s)")

    # ----- Build canvas: sort by cum_y; per-pixel 2-pass fill (with overlay mask) -----
    # Normalize so min cum_y maps to row 0.
    min_y = min(p["cum_y"] for p in placements)
    for p in placements:
        p["abs_y"] = p["cum_y"] - min_y
    placements_sorted = sorted(placements, key=lambda p: p["abs_y"])
    canvas_h = max(p["abs_y"] for p in placements_sorted) + dyn_h
    print(f"[stitch] canvas_h={canvas_h} (W={W})  placements sorted by abs_y")

    # Build a shared per-band "clean" mask (True = pixel is overlay-free).
    clean_mask = np.ones((dyn_h, W), dtype=bool)
    overlay_report: dict | None = None
    if args.detect_overlays == "auto":
        t0 = time.perf_counter()
        bands_list = [rgb_bands[p["i"]] for p in placements]
        overlay_mask, overlay_report = detect_overlay_mask_from_bands(
            bands_list,
            agreement_frac=args.overlay_agreement_frac,
            agreement_tol=args.overlay_agreement_tol,
            dilate=args.overlay_dilate,
            min_area=args.overlay_min_area,
        )
        clean_mask &= ~overlay_mask
        print(f"[mask] auto-detected {overlay_report['pixels_masked']} overlay pixels "
              f"({overlay_report['pct_masked']}% of band) across "
              f"{overlay_report['n_components']} components "
              f"({time.perf_counter() - t0:.2f}s)")
        for c in overlay_report["components"][:8]:
            print(f"       component bbox={c['bbox']} area={c['area']} aspect={c['aspect']}")

    if args.mask_circle.strip():
        try:
            cx, cy, r = [int(x) for x in args.mask_circle.split(",")]
        except ValueError:
            raise SystemExit(f"--mask-circle expected 'cx,cy,r' got {args.mask_circle!r}")
        yy, xx = np.ogrid[:dyn_h, :W]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r + 2) ** 2
        clean_mask &= ~inside
        print(f"[mask] manual circle cx={cx} cy={cy} r={r} -> {int(inside.sum())} pixels masked")
    if args.mask_right_strip_from >= 0:
        x0 = max(0, args.mask_right_strip_from)
        clean_mask[:, x0:] = False
        print(f"[mask] manual right strip x>={x0} -> {(W - x0) * dyn_h} pixels masked")

    n_clean = int(clean_mask.sum())
    print(f"[mask] final clean pixels per band: {n_clean}/{dyn_h*W} ({100.0*n_clean/(dyn_h*W):.1f}%)")

    # ----- Optional: per-band detected-circle masks (scroll-to-latest button etc.) -----
    # Discovery pass over all placed bands → list of stable circular overlays.
    # Per-band targeted re-detection inside a tight ROI builds per-band augment
    # masks. These layer onto the shared clean_mask via per-band lookups in
    # passes 1 and 2 so that bands WITHOUT the button at the canonical position
    # still contribute their clean conversation content (essential because the
    # button is conditionally visible).
    per_band_extra_mask: dict[int, np.ndarray] = {}
    circle_specs: list = []
    circle_discovery_report: dict | None = None
    circle_per_band_report: dict | None = None
    if args.mask_detected_circles:
        t0 = time.perf_counter()
        unique_bands = list({p["i"] for p in placements})
        bands_for_discovery = [rgb_bands[i] for i in unique_bands]
        circle_specs, circle_discovery_report = discover_persistent_circles(
            bands_for_discovery,
            min_prevalence=args.circle_min_prevalence,
            r_min=args.circle_r_min, r_max=args.circle_r_max,
            param1=args.circle_param1, param2=args.circle_param2,
        )
        n_specs = len(circle_specs)
        print(f"[circles] discovery: {n_specs} persistent circle(s) across "
              f"{len(unique_bands)} bands "
              f"(min_prevalence={args.circle_min_prevalence}) "
              f"in {time.perf_counter() - t0:.2f}s")
        for s in circle_specs:
            print(f"[circles]   cx={s.cx} cy={s.cy} r={s.r}  "
                  f"prevalence={s.prevalence:.3f}  r_range=[{s.r_min}..{s.r_max}]")
        if circle_specs:
            t1 = time.perf_counter()
            # Strategy: for every placed band, OR together
            #   (a) the spec disc (always applied: cx, cy, r+pad)
            #   (b) the per-band detected disc if Hough re-detects in the band
            #       at slightly different (cx, cy, r) — typically <= 2 px off
            # Always-apply (a) is the robustness mechanism — the per-band
            # detector misses ~30% of the bands that visually contain the
            # button (Hough is finicky on the conversation background), and
            # the missed bands would otherwise leak button pixels into
            # pass-1 mean and pass-2 median. (b) is additive: covers minor
            # spec drift so bands with a slightly offset button are masked
            # at the actual position too.
            #
            # Cost of always-applying (a) to bands without the button:
            # ~(r+pad)^2 * pi ≈ 13k px of clean conversation content lost
            # per band per spec. Other bands covering the same canvas pixel
            # contribute clean content via pass-1, so the canvas is fine.
            per_band_hits: dict[int, int] = {}
            per_band_detections: list[dict] = []
            for i in unique_bands:
                acc_mask = np.zeros((dyn_h, W), dtype=bool)
                hits_this_band = 0
                for spec in circle_specs:
                    spec_mask = _filled_disk_mask(
                        (dyn_h, W), spec.cx, spec.cy, spec.r + args.circle_pad
                    )
                    acc_mask |= spec_mask
                    det_mask, det = detect_circle_in_band(
                        rgb_bands[i],
                        expected_cx=spec.cx, expected_cy=spec.cy,
                        expected_r=spec.r,
                        slack_xy=args.circle_slack_xy,
                        slack_r=args.circle_slack_r,
                        pad=args.circle_pad,
                        param1=args.circle_param1,
                        param2=args.circle_param2,
                    )
                    if det.detected:
                        acc_mask |= det_mask
                        hits_this_band += 1
                        per_band_detections.append({
                            "band_i": int(i),
                            "spec_cx": spec.cx, "spec_cy": spec.cy,
                            "det_cx": det.cx, "det_cy": det.cy, "det_r": det.r,
                            "score_distance": det.score_distance,
                            "pixels_masked": det.pixels_masked,
                        })
                per_band_extra_mask[i] = acc_mask
                per_band_hits[i] = hits_this_band
            n_bands_with_hit = sum(1 for v in per_band_hits.values() if v > 0)
            tot_pixels_masked = int(sum(m.sum() for m in per_band_extra_mask.values()))
            spec_disc_only_px = sum(
                _filled_disk_mask((dyn_h, W), s.cx, s.cy, s.r + args.circle_pad).sum()
                for s in circle_specs
            )
            circle_per_band_report = {
                "wall_s": round(time.perf_counter() - t1, 3),
                "strategy": "always-apply-spec + per-band-detection-union",
                "n_specs": int(n_specs),
                "n_bands": len(unique_bands),
                "n_bands_with_perband_detection": int(n_bands_with_hit),
                "spec_disc_pixels_per_band": int(spec_disc_only_px),
                "total_per_band_mask_pixels": tot_pixels_masked,
                "per_spec_prevalence": [s.prevalence for s in circle_specs],
                "detections": per_band_detections,
            }
            print(f"[circles] always-apply spec disc to all {len(unique_bands)} bands  "
                  f"+ per-band Hough refinement hit {n_bands_with_hit}/{len(unique_bands)}  "
                  f"total masked px (summed across bands)={tot_pixels_masked}  "
                  f"in {time.perf_counter() - t1:.2f}s")
        elif circle_discovery_report:
            top_count = circle_discovery_report["top_bins"][0]["count"] if circle_discovery_report["top_bins"] else 0
            print(f"[circles] no specs promoted (top bin had {top_count}/"
                  f"{len(unique_bands)} bands, below threshold)")

    # Helper: return effective clean mask (and its int16 version) for band i,
    # combining shared clean_mask with any per-band augmentation.
    _empty_extra = np.zeros((dyn_h, W), dtype=bool)
    clean_i16_shared = clean_mask.astype(np.int16)

    def _effective_clean_for_band(i: int) -> tuple[np.ndarray, np.ndarray]:
        extra = per_band_extra_mask.get(i, _empty_extra)
        if not extra.any():
            return clean_mask, clean_i16_shared
        eff = clean_mask & ~extra
        return eff, eff.astype(np.int16)

    # ----- Optional: per-band bubble extents via cv2.Canny (for pass 7) -----
    # iOS Messenger bubbles end with a high-contrast rounded edge against the
    # gutter. Canny finds those edges robustly across colours (blue, grey,
    # dark/light mode). We compute leftmost/rightmost edge per band row, then
    # aggregate (max-R, min-L) across all bands covering each canvas row.
    # Pass 7 uses these to overwrite gutter pixels that pass 2 may fabricate
    # with the wrong colour.
    band_R: dict[int, np.ndarray] = {}
    band_L: dict[int, np.ndarray] = {}
    extents_report: dict | None = None
    if args.clear_beyond_bubble_extent:
        t0 = time.perf_counter()
        for p in placements:
            i = p["i"]
            if i in band_R:
                continue
            _, R_arr, L_arr = detect_band_extents(
                rgb_bands[i],
                canny_lo=args.canny_lo,
                canny_hi=args.canny_hi,
                close_w=args.canny_close_w,
                close_h=args.canny_close_h,
                scrollbar_pair_dx=args.bubble_scrollbar_pair_dx,
                is_bgr=False,
                r_exclude_from=args.bubble_detector_r_exclude_from,
            )
            band_R[i] = R_arr
            band_L[i] = L_arr
        valid_R_total = sum(int((R >= 0).sum()) for R in band_R.values())
        valid_L_total = sum(int((L >= 0).sum()) for L in band_L.values())
        rows_total = sum(R.size for R in band_R.values())
        med_R = int(np.median(np.concatenate([R[R >= 0] for R in band_R.values()]))) if valid_R_total else -1
        med_L = int(np.median(np.concatenate([L[L >= 0] for L in band_L.values()]))) if valid_L_total else -1
        extents_report = {
            "wall_s": round(time.perf_counter() - t0, 3),
            "bands_processed": len(band_R),
            "rows_total": int(rows_total),
            "rows_with_R": int(valid_R_total),
            "rows_with_L": int(valid_L_total),
            "R_median_in_band": med_R,
            "L_median_in_band": med_L,
            "canny_lo": args.canny_lo,
            "canny_hi": args.canny_hi,
            "close_wxh": [args.canny_close_w, args.canny_close_h],
            "scrollbar_pair_dx": args.bubble_scrollbar_pair_dx,
            "r_exclude_from": args.bubble_detector_r_exclude_from,
        }
        print(f"[extents] cv2.Canny per band: {len(band_R)} bands in "
              f"{extents_report['wall_s']:.2f}s  "
              f"R_med={med_R} L_med={med_L}  "
              f"valid R={valid_R_total}/{rows_total} L={valid_L_total}/{rows_total}")

    # Multi-pass fill strategy:
    #   Pass 1: For each canvas pixel, accumulate values from covering bands
    #           ONLY where the in-band position is CLEAN. Result: average of
    #           clean candidates. Best signal where any band has clean data.
    #   Pass 2: For canvas pixels where no band had a clean value (every
    #           covering band has overlay at this in-band position), take the
    #           MEDIAN of ALL covering bands' values. This correctly recovers
    #           uniform background "gutter" regions (where every band agreed
    #           because the colour is constant). For thin strip overlays
    #           (scrollbar) where every band's value is overlay-colour, the
    #           median is still overlay-colour and pass 3 cleans it.
    #   Pass 3: Narrow contiguous runs of densely-masked columns (e.g. iOS
    #           scrollbar) are replaced via horizontal inpainting from the
    #           nearest non-strip column.
    #   Pass 4: Any still-uncovered (total coverage 0) pixels — typically
    #           none with proper bridge insertion — are spatially inpainted.
    t0 = time.perf_counter()
    canvas_acc = np.zeros((canvas_h, W, 3), dtype=np.float32)
    clean_count = np.zeros((canvas_h, W), dtype=np.int16)
    total_count = np.zeros((canvas_h, W), dtype=np.int16)
    canvas_R = np.full(canvas_h, -1, dtype=np.int32) if args.clear_beyond_bubble_extent else None
    canvas_L = np.full(canvas_h, W, dtype=np.int32) if args.clear_beyond_bubble_extent else None
    for p in placements_sorted:
        band = rgb_bands[p["i"]]
        eff_clean, eff_clean_i16 = _effective_clean_for_band(p["i"])
        eff_clean_3 = eff_clean[..., None]
        y0 = p["abs_y"]
        y1 = y0 + dyn_h
        np.add(canvas_acc[y0:y1], band, where=eff_clean_3, out=canvas_acc[y0:y1])
        clean_count[y0:y1] += eff_clean_i16
        total_count[y0:y1] += 1
        if args.clear_beyond_bubble_extent:
            R_b = band_R[p["i"]]
            L_b = band_L[p["i"]]
            valid_R = R_b >= 0
            valid_L = L_b >= 0
            existing_R = canvas_R[y0:y1]
            np.maximum(existing_R, R_b, out=existing_R, where=valid_R)
            existing_L = canvas_L[y0:y1]
            np.minimum(existing_L, L_b, out=existing_L, where=valid_L)
        p["status"] = "placed"
    have_clean = clean_count > 0
    canvas = np.zeros((canvas_h, W, 4), dtype=np.uint8)
    cnt3 = clean_count[..., None].astype(np.float32)
    np.divide(canvas_acc, cnt3, where=have_clean[..., None], out=canvas_acc)
    canvas[have_clean, :3] = canvas_acc[have_clean].astype(np.uint8)
    canvas[have_clean, 3] = 255
    n_pass1 = int(have_clean.sum())
    print(f"[stitch] pass1 (clean-mean) filled {n_pass1}/{canvas_h*W} pixels "
          f"({100.0*n_pass1/(canvas_h*W):.2f}%) in {time.perf_counter() - t0:.2f}s")
    print(f"[stitch] clean coverage: min={int(clean_count.min())} "
          f"max={int(clean_count.max())} mean={float(clean_count.mean()):.2f}  "
          f"total coverage: min={int(total_count.min())} "
          f"max={int(total_count.max())} mean={float(total_count.mean()):.2f}")

    # Pass 2: policy-driven fill for clean_count==0 pixels.
    #   median:        original behaviour (median of all covering bands).
    #   variance-gate: fill only where sample range <= variance_tol; else α=0.
    #                  k=1 samples always α=0 (cannot distinguish gutter from overlay).
    #   hard-skip:     never fill via pass 2; α=0 for all.
    #   variance-gate-dilated: same as variance-gate; dilation applied after pass 4.
    need_median = (~have_clean) & (total_count > 0)
    n_need_median = int(need_median.sum())
    pass2_count = 0
    pass2_skipped_variance = 0
    pass2_skipped_k1 = 0
    if n_need_median > 0 and args.no_clean_policy != "hard-skip":
        t0 = time.perf_counter()
        max_cov = int(total_count.max())
        CHUNK = 2048
        variance_tol = int(args.pass2_variance_tol)
        policy_variance = args.no_clean_policy in ("variance-gate", "variance-gate-dilated")
        for y_start in range(0, canvas_h, CHUNK):
            y_end = min(canvas_h, y_start + CHUNK)
            local_need = need_median[y_start:y_end]
            if not local_need.any():
                continue
            local_max_cov = int(total_count[y_start:y_end].max())
            if local_max_cov == 0:
                continue
            cand = np.zeros((local_max_cov, y_end - y_start, W, 3), dtype=np.uint8)
            cand_count = np.zeros((y_end - y_start, W), dtype=np.int16)
            for p in placements_sorted:
                y0 = p["abs_y"]
                y1 = y0 + dyn_h
                cy0 = max(y0, y_start)
                cy1 = min(y1, y_end)
                if cy0 >= cy1:
                    continue
                band = rgb_bands[p["i"]]
                band_y0 = cy0 - y0
                band_y1 = cy1 - y0
                local_y0 = cy0 - y_start
                local_y1 = cy1 - y_start
                sub_band = band[band_y0:band_y1]
                sub_need = local_need[local_y0:local_y1]
                extra = per_band_extra_mask.get(p["i"])
                if extra is not None:
                    sub_extra = extra[band_y0:band_y1]
                    sub_need = sub_need & ~sub_extra
                if not sub_need.any():
                    continue
                idx_y, idx_x = np.where(sub_need)
                global_idx_y = idx_y + local_y0
                slots = cand_count[global_idx_y, idx_x]
                cand[slots, global_idx_y, idx_x] = sub_band[idx_y, idx_x]
                cand_count[global_idx_y, idx_x] += 1
            uy, ux = np.where(local_need)
            ks = cand_count[uy, ux]
            for k in np.unique(ks):
                if k == 0:
                    continue
                sel = np.where(ks == k)[0]
                samples = cand[:k, uy[sel], ux[sel], :]  # (k, n_sel, 3)
                if policy_variance:
                    if k == 1:
                        pass2_skipped_k1 += len(sel)
                        continue
                    sample_range = np.ptp(samples.astype(np.int16), axis=0).max(axis=1)
                    accept = sample_range <= variance_tol
                    n_accept = int(accept.sum())
                    pass2_skipped_variance += len(sel) - n_accept
                    if n_accept == 0:
                        continue
                    med = np.median(samples[:, accept, :], axis=0).astype(np.uint8)
                    acc_uy = y_start + uy[sel[accept]]
                    acc_ux = ux[sel[accept]]
                    canvas[acc_uy, acc_ux, :3] = med
                    canvas[acc_uy, acc_ux, 3] = 255
                    pass2_count += n_accept
                else:
                    med = np.median(samples, axis=0).astype(np.uint8)
                    canvas[y_start + uy[sel], ux[sel], :3] = med
                    canvas[y_start + uy[sel], ux[sel], 3] = 255
                    pass2_count += len(sel)
        print(f"[stitch] pass2 ({args.no_clean_policy}) filled +{pass2_count} pixels "
              f"({100.0*pass2_count/(canvas_h*W):.2f}%) "
              f"[skipped: k=1 {pass2_skipped_k1}, variance {pass2_skipped_variance}] in "
              f"{time.perf_counter() - t0:.2f}s")
    elif args.no_clean_policy == "hard-skip" and n_need_median > 0:
        print(f"[stitch] pass2 (hard-skip) left {n_need_median} clean_count==0 pixels "
              f"transparent ({100.0*n_need_median/(canvas_h*W):.2f}%)")

    # Pass 3: narrow vertical strip overlays (scrollbar) → horizontal inpaint
    n_strip = 0
    runs: list[tuple[int, int]] = []
    if overlay_report is not None and not overlay_report.get("skipped"):
        t0 = time.perf_counter()
        col_masked_frac = (~clean_mask).sum(axis=0) / dyn_h
        dense_cols = col_masked_frac > 0.90
        strip_cols = np.zeros(W, dtype=bool)
        i = 0
        while i < W:
            if dense_cols[i]:
                j = i
                while j < W and dense_cols[j]:
                    j += 1
                if (j - i) <= args.overlay_max_strip_width:
                    strip_cols[i:j] = True
                    runs.append((i, j))
                i = j
            else:
                i += 1
        n_strip = int(strip_cols.sum())
        if 0 < n_strip < W:
            xs = np.arange(W)
            good_cols = xs[~strip_cols]
            for x in xs[strip_cols]:
                idx = np.searchsorted(good_cols, x)
                left = good_cols[idx - 1] if idx > 0 else good_cols[idx]
                right = good_cols[idx] if idx < len(good_cols) else good_cols[idx - 1]
                nearest = left if abs(x - left) <= abs(x - right) else right
                canvas[:, x] = canvas[:, nearest]
            print(f"[stitch] pass3 (horizontal inpaint) replaced {n_strip} "
                  f"strip cols in {len(runs)} runs "
                  f"(widths: {[r[1]-r[0] for r in runs]}) in "
                  f"{time.perf_counter() - t0:.2f}s")

    # Pass 4: spatial inpaint for (a) total_count == 0 pixels (truly uncovered)
    # AND (b) pixels that remained α=0 after passes 1-3 despite having coverage
    # — i.e. starved by per-band overlay masks (every band covering them had
    # the detected-circle mask hit at that position, so neither clean-mean nor
    # gated-median had a candidate). Without this, those starved pixels would
    # stay (0,0,0,0) → render as BLACK in the RGB output (the "black half-disc"
    # artifact when circle masking is on).
    #
    # Two algorithms:
    #   - distance_transform_edt: copies the nearest filled pixel into each
    #     hole pixel. Cheap; appropriate for thin holes (≤ ~4 px). For wide
    #     holes (a 120 px scroll-button disc) it produces visible wedge/streak
    #     artifacts because the nearest-pixel for a deep interior point is on
    #     the rim and the same rim pixel is reused for an entire sector.
    #   - cv2.inpaint (Telea): smooth boundary interpolation. Slightly more
    #     expensive but produces clean fills for disc-shaped holes. Used for
    #     the "starved" set (per-band-mask-induced holes).
    no_coverage = total_count == 0
    alpha_now = canvas[..., 3]
    starved = (alpha_now == 0) & ~no_coverage
    n_nocov = int(no_coverage.sum())
    n_starved = int(starved.sum())
    n_need_inpaint = n_nocov + n_starved
    if n_need_inpaint > 0:
        t0 = time.perf_counter()
        if n_starved > 0:
            rgb = canvas[..., :3].copy()
            inpaint_mask = starved.astype(np.uint8) * 255
            filled = cv2.inpaint(rgb, inpaint_mask, inpaintRadius=3,
                                 flags=cv2.INPAINT_TELEA)
            canvas[starved, :3] = filled[starved]
            canvas[starved, 3] = 255
        if n_nocov > 0:
            try:
                from scipy.ndimage import distance_transform_edt
            except ImportError:
                distance_transform_edt = None
            if distance_transform_edt is not None:
                inds = distance_transform_edt(no_coverage, return_distances=False,
                                              return_indices=True)
                ny, nx = inds[0], inds[1]
                canvas[no_coverage] = canvas[ny[no_coverage], nx[no_coverage]]
        print(f"[stitch] pass4 (spatial inpaint) filled {n_need_inpaint} "
              f"pixels (no_coverage={n_nocov} via EDT, "
              f"starved-by-mask={n_starved} via cv2.inpaint Telea) "
              f"in {time.perf_counter() - t0:.2f}s")

    # Pass 5 (policy=variance-gate-dilated only): morphologically dilate α=0
    # regions so isolated rejected pixels merge into solid blobs (tidier holes
    # for downstream review). Dilation grows transparent regions; each iter
    # expands by ~1px in each direction.
    n_alpha_zero_pre = int((canvas[..., 3] == 0).sum())
    n_dilated = 0
    if args.no_clean_policy == "variance-gate-dilated" and args.alpha_dilate_iters > 0:
        try:
            from scipy.ndimage import binary_dilation
        except ImportError:
            binary_dilation = None
        if binary_dilation is not None and n_alpha_zero_pre > 0:
            t0 = time.perf_counter()
            alpha0 = canvas[..., 3] == 0
            grown = binary_dilation(alpha0, iterations=int(args.alpha_dilate_iters))
            newly = grown & ~alpha0
            canvas[..., 3][newly] = 0
            canvas[..., :3][newly] = 0
            n_dilated = int(newly.sum())
            print(f"[stitch] pass5 (alpha dilation x{args.alpha_dilate_iters}) "
                  f"converted +{n_dilated} pixels to transparent in "
                  f"{time.perf_counter() - t0:.2f}s")
    n_alpha_zero_post = int((canvas[..., 3] == 0).sum())

    # ----- Pass 6: scrollbar rim trim (post-process, scope-limited) -----
    # iOS scrollbar is a thin (~3pt = 9px @3x) translucent vertical bar at the
    # extreme right edge. Pass 2 fabricates a wrong-coloured ghost there because
    # every covering band has the overlay at the same in-band x position. The
    # bubbles end ~48px from the edge, so overwriting a narrow rim (default 12px
    # if --scrollbar-rim-px > 0) cleans the ghost without touching any bubble
    # pixel. This pass does NOT address the scroll-to-latest button ghost, which
    # lives inside bubble interiors and needs a different (content-aware) fix.
    scrollbar_report: dict | None = None
    if args.scrollbar_rim_px > 0:
        t0 = time.perf_counter()
        rim = int(args.scrollbar_rim_px)
        if args.scrollbar_rim_bg == "auto":
            edge_col_rgb = canvas[:, W - 1, :3]
            packed = (
                edge_col_rgb[:, 0].astype(np.uint32) << 16
                | edge_col_rgb[:, 1].astype(np.uint32) << 8
                | edge_col_rgb[:, 2].astype(np.uint32)
            )
            unique, counts = np.unique(packed, return_counts=True)
            mode_packed = int(unique[int(counts.argmax())])
            mode_frac = float(counts.max()) / float(canvas_h)
            bg_rgb = np.array([
                (mode_packed >> 16) & 0xFF,
                (mode_packed >> 8) & 0xFF,
                mode_packed & 0xFF,
            ], dtype=np.uint8)
            bg_source = f"auto (modal of x={W-1}, frac={mode_frac:.3f})"
        else:
            try:
                parts = [int(v) for v in args.scrollbar_rim_bg.split(",")]
                assert len(parts) == 3
                bg_rgb = np.array(parts, dtype=np.uint8)
                bg_source = f"explicit ({args.scrollbar_rim_bg})"
            except (ValueError, AssertionError):
                raise SystemExit(f"--scrollbar-rim-bg must be 'auto' or 'R,G,B', got {args.scrollbar_rim_bg!r}")
        x0_rim = W - rim
        rim_view = canvas[:, x0_rim:W, :3]
        before = rim_view.copy()
        rim_view[...] = bg_rgb
        if args.no_clean_policy != "median":
            canvas[:, x0_rim:W, 3] = 255
        diff_mask = np.abs(before.astype(np.int16) - bg_rgb.astype(np.int16)).max(axis=2) > 0
        n_changed = int(diff_mask.sum())
        scrollbar_report = {
            "rim_px": rim, "x_range": [int(x0_rim), int(W)],
            "bg_rgb": bg_rgb.tolist(), "bg_source": bg_source,
            "pixels_changed": n_changed,
            "pct_changed_in_rim": round(100.0 * n_changed / max(1, rim * canvas_h), 3),
            "wall_s": round(time.perf_counter() - t0, 3),
        }
        print(f"[stitch] pass6 (scrollbar-rim) rim={rim}px x={x0_rim}..{W} "
              f"bg=RGB{tuple(int(v) for v in bg_rgb)} ({bg_source}) "
              f"changed {n_changed} pixels ({100.0 * n_changed / max(1, rim * canvas_h):.2f}% of rim) "
              f"in {scrollbar_report['wall_s']:.2f}s")

    # ----- Pass 7: clear beyond bubble extent (cv2.Canny-driven gutter fill) -----
    # For every canvas row, we know the max-rightmost / min-leftmost bubble
    # edge across all bands that contributed to it. Any pixel beyond those
    # extents (plus a small pad for anti-aliased outlines) is gutter; force-
    # fill with the gutter background colour. This removes wrong-colour pass-2
    # fabrications at the canvas left/right edges (the "stretched bubble"
    # artifact). bg is auto-detected from x=0 of placed source bands (always
    # gutter, never has scrollbar).
    extent_clear_report: dict | None = None
    if args.clear_beyond_bubble_extent:
        t0 = time.perf_counter()
        if args.bubble_extent_bg == "auto":
            edge_x0_col = np.concatenate([rgb_bands[p["i"]][:, 0, :] for p in placements])
            packed = (
                edge_x0_col[:, 0].astype(np.uint32) << 16
                | edge_x0_col[:, 1].astype(np.uint32) << 8
                | edge_x0_col[:, 2].astype(np.uint32)
            )
            unique, counts = np.unique(packed, return_counts=True)
            mode_packed = int(unique[int(counts.argmax())])
            mode_frac = float(counts.max()) / float(edge_x0_col.shape[0])
            ext_bg = np.array([
                (mode_packed >> 16) & 0xFF,
                (mode_packed >> 8) & 0xFF,
                mode_packed & 0xFF,
            ], dtype=np.uint8)
            ext_bg_src = f"auto (modal of x=0 across {len(placements)} bands, frac={mode_frac:.3f})"
        else:
            try:
                parts = [int(v) for v in args.bubble_extent_bg.split(",")]
                assert len(parts) == 3
                ext_bg = np.array(parts, dtype=np.uint8)
                ext_bg_src = f"explicit ({args.bubble_extent_bg})"
            except (ValueError, AssertionError):
                raise SystemExit(f"--bubble-extent-bg must be 'auto' or 'R,G,B', got {args.bubble_extent_bg!r}")

        pad = int(args.bubble_extent_pad)
        x_idx = np.arange(W, dtype=np.int32)[None, :]

        # ----- Robustify canvas_R: clamp + vertical max-pool -----
        # The bubble's right edge is straight and vertical across many rows.
        # Per-row Canny detection can fail on text-heavy rows (white text
        # near the bubble edge dominates the per-row rightmost edge), leaving
        # R too low and pass-7 carving notches out of the bubble. A vertical
        # max-pool propagates correct R from neighbouring good rows. We clamp
        # R upper bound first so a stray scrollbar-edge detection cannot
        # inflate R across the whole pool window.
        canvas_R_raw = canvas_R.copy()
        r_max_arg = int(args.bubble_extent_r_max)
        r_max = (W - 16) if r_max_arg < 0 else min(W, max(0, r_max_arg))
        rows_clamped_R = int(np.sum(canvas_R > r_max))
        canvas_R = np.where(canvas_R > r_max, r_max, canvas_R).astype(np.int32)

        smooth_radius = max(0, int(args.bubble_extent_smooth_radius))
        rows_pool_raised_R = 0
        max_pool_delta_R = 0
        if smooth_radius > 0:
            from scipy.ndimage import maximum_filter1d
            # -1 sentinel for invalid rows is the natural min element, so
            # max-pool propagates valid R into nearby invalid rows up to
            # `smooth_radius` away and never replaces a higher valid R with
            # an invalid -1. Rows with no valid neighbour stay at -1.
            canvas_R_smoothed = maximum_filter1d(
                canvas_R, size=2 * smooth_radius + 1, mode="nearest"
            ).astype(np.int32)
            pool_delta = canvas_R_smoothed - canvas_R
            rows_pool_raised_R = int(np.sum(pool_delta > 0))
            max_pool_delta_R = int(pool_delta.max()) if pool_delta.size else 0
            canvas_R = canvas_R_smoothed

        # Right side: clear where x > canvas_R + pad (only where canvas_R is valid)
        valid_R_rows = canvas_R >= 0
        R_thresh = np.where(valid_R_rows, canvas_R + pad, W).astype(np.int32)
        clear_R = (x_idx > R_thresh[:, None]) & valid_R_rows[:, None]

        # Left side: clear where x < canvas_L - pad (only where canvas_L was set < W).
        # NOTE: We deliberately do NOT vertical-min-pool canvas_L. Min-pooling
        # would pull L from a right-aligned outgoing-bubble row (L~313) down
        # to a nearby left-aligned incoming-bubble row's L (~24), shrinking
        # the left-clear region from "x<310" to "x<21" on outgoing rows and
        # leaving fabricated wrong-colour gutter at x=24..310 uncleaned.
        valid_L_rows = canvas_L < W
        L_thresh = np.where(valid_L_rows, np.maximum(canvas_L - pad, 0), 0).astype(np.int32)
        clear_L = (x_idx < L_thresh[:, None]) & valid_L_rows[:, None]

        clear_total = clear_R | clear_L
        canvas[..., :3][clear_total] = ext_bg
        if args.no_clean_policy != "median":
            canvas[..., 3][clear_total] = 255

        n_cleared_R = int(clear_R.sum())
        n_cleared_L = int(clear_L.sum())
        n_cleared = int(clear_total.sum())
        # Sanity check: nothing inside [L+pad..R-pad] should be cleared.
        # (Both clear_R and clear_L predicates ensure x is outside [L,R] band.)
        rows_with_R = int(valid_R_rows.sum())
        rows_with_L = int(valid_L_rows.sum())
        extent_clear_report = {
            "wall_s": round(time.perf_counter() - t0, 3),
            "bg_rgb": ext_bg.tolist(),
            "bg_source": ext_bg_src,
            "pad_px": pad,
            "smooth_radius_R": smooth_radius,
            "r_max_clamp": int(r_max),
            "rows_clamped_R": rows_clamped_R,
            "rows_pool_raised_R": rows_pool_raised_R,
            "max_pool_delta_R": max_pool_delta_R,
            "rows_with_R_extent": rows_with_R,
            "rows_with_L_extent": rows_with_L,
            "rows_total": int(canvas_h),
            "pixels_cleared_right": n_cleared_R,
            "pixels_cleared_left": n_cleared_L,
            "pixels_cleared_total": n_cleared,
            "pct_canvas_cleared": round(100.0 * n_cleared / (canvas_h * W), 3),
        }
        print(f"[stitch] pass7 (extent-clear) bg=RGB{tuple(int(v) for v in ext_bg)} "
              f"({ext_bg_src}) pad={pad} smooth_r={smooth_radius} r_max={r_max}  "
              f"R: clamped={rows_clamped_R} pool_raised={rows_pool_raised_R} "
              f"(max_delta={max_pool_delta_R})  "
              f"cleared R={n_cleared_R} L={n_cleared_L} (total={n_cleared}, "
              f"{extent_clear_report['pct_canvas_cleared']}% of canvas)  "
              f"rows_with_R={rows_with_R}/{canvas_h} L={rows_with_L}/{canvas_h}  "
              f"in {extent_clear_report['wall_s']:.2f}s")

        # ----- Pass 7b: synthetic AA on the right edge -----
        # Pass-1 mean blending across vertically-sub-pixel-misaligned bands
        # smears the bubble fill outward by 2-3 px AND erases the iOS-native
        # AA falloff (90 -> 63 -> 19 -> 1 -> 0). Pass-7 clearing turns the
        # smeared bubble into a hard 91->0 step, which reads as "blunted" /
        # mechanical compared to the source's natural taper.
        #
        # Reconstruction recipe (measured from source frame 472, y=295 vs
        # luma-90 blue bubble): the rightmost 3 columns of every bubble are
        # AA pixels with bubble-color intensity ratios (0.70, 0.20, 0.01).
        # We sample bubble RGB at x = R-3 (a column that is solid bubble in
        # both source and our smeared output) and repaint x = R-2, R-1, R as
        # bubble*ratio + bg*(1-ratio). Per-row sampling handles arbitrary
        # bubble colors automatically.
        #
        # Threshold-skip rows where x=R-3 is bg-coloured (rounded-corner
        # rows, or max-pool-raised rows where the row has no bubble at this
        # R extent) so we never paint AA into empty gutter.
        if args.bubble_extent_synthetic_aa:
            t0_aa = time.perf_counter()
            aa_recipe = [(2, 0.70), (1, 0.20), (0, 0.01)]
            bg_rgb_f = ext_bg.astype(np.float32)
            bg_luma = 0.299 * bg_rgb_f[0] + 0.587 * bg_rgb_f[1] + 0.114 * bg_rgb_f[2]
            valid_aa = valid_R_rows & (canvas_R >= 3) & (canvas_R < W)
            R_safe = np.where(valid_aa, canvas_R, 3).astype(np.int32)
            sample_x = np.clip(R_safe - 3, 0, W - 1)
            y_indices = np.arange(canvas_h, dtype=np.int32)
            bubble_rgb_per_row = canvas[..., :3][y_indices, sample_x].astype(np.float32)
            bubble_luma_per_row = (
                0.299 * bubble_rgb_per_row[:, 0]
                + 0.587 * bubble_rgb_per_row[:, 1]
                + 0.114 * bubble_rgb_per_row[:, 2]
            )
            aa_rows_mask = valid_aa & (
                np.abs(bubble_luma_per_row - bg_luma) >= args.bubble_extent_aa_threshold
            )
            aa_pixels_painted = 0
            aa_rows_painted = int(aa_rows_mask.sum())
            for dx, ratio in aa_recipe:
                x_arr = R_safe - dx
                col_valid = (x_arr >= 0) & (x_arr < W) & aa_rows_mask
                if not col_valid.any():
                    continue
                ys = np.where(col_valid)[0]
                xs = x_arr[col_valid]
                blended = (
                    bubble_rgb_per_row[col_valid] * ratio
                    + bg_rgb_f * (1.0 - ratio)
                ).astype(np.uint8)
                canvas[..., :3][ys, xs] = blended
                aa_pixels_painted += int(col_valid.sum())
            extent_clear_report["synthetic_aa_rows"] = aa_rows_painted
            extent_clear_report["synthetic_aa_pixels"] = aa_pixels_painted
            extent_clear_report["synthetic_aa_wall_s"] = round(
                time.perf_counter() - t0_aa, 3
            )
            print(f"[stitch] pass7b (synthetic-AA) rows={aa_rows_painted}/{rows_with_R} "
                  f"pixels={aa_pixels_painted} "
                  f"(recipe ratios 0.70/0.20/0.01 at R-2/R-1/R, "
                  f"threshold={args.bubble_extent_aa_threshold}) "
                  f"in {extent_clear_report['synthetic_aa_wall_s']:.2f}s")


    if overlay_report is not None:
        overlay_report["canvas_pixels_total"] = int(canvas_h * W)
        overlay_report["canvas_pass1_clean_mean"] = n_pass1
        overlay_report["canvas_pass2_filled"] = int(pass2_count)
        overlay_report["canvas_pass2_need_pixels"] = int(n_need_median)
        overlay_report["canvas_pass2_skipped_k1"] = int(pass2_skipped_k1)
        overlay_report["canvas_pass2_skipped_variance"] = int(pass2_skipped_variance)
        overlay_report["canvas_pass3_strip_inpainted"] = n_strip
        overlay_report["canvas_pass4_spatial_inpainted_no_coverage"] = n_nocov
        overlay_report["canvas_pass4_starved_by_per_band_mask"] = n_starved
        overlay_report["canvas_pass4_total_inpainted"] = n_need_inpaint
        overlay_report["canvas_pass5_alpha_dilated"] = n_dilated
        overlay_report["canvas_alpha_zero_final"] = n_alpha_zero_post
        overlay_report["canvas_alpha_zero_pct"] = round(100.0 * n_alpha_zero_post / (canvas_h * W), 3)
        overlay_report["clean_coverage_mean"] = round(float(clean_count.mean()), 3)
        overlay_report["total_coverage_mean"] = round(float(total_count.mean()), 3)
        overlay_report["strip_runs"] = [{"x0": r[0], "x1": r[1]} for r in runs]
    unfilled = n_nocov  # remains > 0 only if pass4 fell back and failed

    # ----- Write chunks -----
    writer = ChunkWriter(out_dir=args.out, width=W, chunk_height=args.chunk_height,
                         prefix="keyframe_chunk")
    # Canvas is RGBA internally. With the `median` policy every pixel ends up
    # alpha=255, so strip the alpha channel for clean RGB output (smaller
    # files; matches historical baseline). Non-median policies emit RGBA so
    # transparent regions survive into the PNGs.
    use_alpha_out = args.no_clean_policy != "median"
    if use_alpha_out:
        alpha_pad_top = np.full((static_top_rgb.shape[0], W, 1), 255, dtype=np.uint8)
        alpha_pad_bot = np.full((static_bot_rgb.shape[0], W, 1), 255, dtype=np.uint8)
        static_top_out = np.concatenate([static_top_rgb, alpha_pad_top], axis=2)
        static_bot_out = np.concatenate([static_bot_rgb, alpha_pad_bot], axis=2)
        canvas_out = canvas
    else:
        static_top_out = static_top_rgb
        static_bot_out = static_bot_rgb
        canvas_out = canvas[..., :3]
    if args.ui == "keep-once":
        writer.append(static_top_out)
    writer.append(canvas_out)
    if args.ui == "keep-once":
        writer.append(static_bot_out)
    chunks_log = writer.finalize()
    print(f"[write] {len(chunks_log)} chunks, {writer.total_rows} total rows, "
          f"alpha=0: {n_alpha_zero_post} ({100.0*n_alpha_zero_post/(canvas_h*W):.2f}% of canvas)"
          f"{' [stripped: RGB output]' if not use_alpha_out else ''}")

    # ----- Report -----
    report = {
        "input": str(args.input),
        "keyframes_json": str(args.keyframes),
        "video": vmeta,
        "params": {
            "chunk_height": args.chunk_height,
            "overlap_margin": args.overlap_margin,
            "ui": args.ui,
            "validate_match": args.validate_match,
            "detect_overlays": args.detect_overlays,
            "overlay_agreement_frac": args.overlay_agreement_frac,
            "overlay_agreement_tol": args.overlay_agreement_tol,
            "overlay_dilate": args.overlay_dilate,
            "overlay_min_area": args.overlay_min_area,
            "overlay_max_strip_width": args.overlay_max_strip_width,
            "mask_circle": args.mask_circle,
            "mask_right_strip_from": args.mask_right_strip_from,
            "no_clean_policy": args.no_clean_policy,
            "pass2_variance_tol": args.pass2_variance_tol,
            "alpha_dilate_iters": args.alpha_dilate_iters,
            "scrollbar_rim_px": args.scrollbar_rim_px,
            "scrollbar_rim_bg": args.scrollbar_rim_bg,
            "clear_beyond_bubble_extent": args.clear_beyond_bubble_extent,
            "bubble_extent_pad": args.bubble_extent_pad,
            "bubble_extent_bg": args.bubble_extent_bg,
            "canny_lo": args.canny_lo,
            "canny_hi": args.canny_hi,
            "canny_close_w": args.canny_close_w,
            "canny_close_h": args.canny_close_h,
            "bubble_scrollbar_pair_dx": args.bubble_scrollbar_pair_dx,
            "bubble_extent_smooth_radius": args.bubble_extent_smooth_radius,
            "bubble_extent_r_max": args.bubble_extent_r_max,
            "bubble_detector_r_exclude_from": args.bubble_detector_r_exclude_from,
        },
        "n_keyframes": len(kfs),
        "n_bridges": bridge_count,
        "n_placements": len(placements),
        "canvas_height": canvas_h,
        "total_rows_written": writer.total_rows,
        "mask_clean_pixels_per_band": int(clean_mask.sum()),
        "canvas_unfilled_pixels": int(unfilled),
        "overlay_detection": overlay_report,
        "circle_discovery": circle_discovery_report,
        "circle_per_band": circle_per_band_report,
        "circle_specs": [s.to_dict() for s in circle_specs] if circle_specs else [],
        "scrollbar_rim_report": scrollbar_report,
        "bubble_extents_report": extents_report,
        "extent_clear_report": extent_clear_report,
        "placements": placements_sorted,
        "chunks": chunks_log,
    }
    rpath = args.out / "report.json"
    rpath.write_text(json.dumps(report, indent=2))
    print(f"[write] {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
