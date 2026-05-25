"""Stitch keyframes from detect_pauses_b_v2 into ordered vertical chunks.

Algorithm
---------
1. Load keyframes.json (produced by detect_pauses_b_v2.py). It carries:
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_scroll_b import (  # type: ignore  # noqa: E402
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
                    help="Source video (same one passed to detect_pauses_b_v2).")
    ap.add_argument("--keyframes", required=True, type=Path,
                    help="Path to keyframes.json from detect_pauses_b_v2.")
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
    clean_i16 = clean_mask.astype(np.int16)
    clean_3 = clean_mask[..., None]
    for p in placements_sorted:
        band = rgb_bands[p["i"]]
        y0 = p["abs_y"]
        y1 = y0 + dyn_h
        np.add(canvas_acc[y0:y1], band, where=clean_3, out=canvas_acc[y0:y1])
        clean_count[y0:y1] += clean_i16
        total_count[y0:y1] += 1
        p["status"] = "placed"
    have_clean = clean_count > 0
    canvas = np.zeros((canvas_h, W, 3), dtype=np.uint8)
    cnt3 = clean_count[..., None].astype(np.float32)
    np.divide(canvas_acc, cnt3, where=have_clean[..., None], out=canvas_acc)
    canvas[have_clean] = canvas_acc[have_clean].astype(np.uint8)
    n_pass1 = int(have_clean.sum())
    print(f"[stitch] pass1 (clean-mean) filled {n_pass1}/{canvas_h*W} pixels "
          f"({100.0*n_pass1/(canvas_h*W):.2f}%) in {time.perf_counter() - t0:.2f}s")
    print(f"[stitch] clean coverage: min={int(clean_count.min())} "
          f"max={int(clean_count.max())} mean={float(clean_count.mean()):.2f}  "
          f"total coverage: min={int(total_count.min())} "
          f"max={int(total_count.max())} mean={float(total_count.mean()):.2f}")

    # Pass 2: median of ALL covering bands for clean-coverage-zero pixels
    # with total coverage >= 1. Recovers gutter backgrounds correctly.
    need_median = (~have_clean) & (total_count > 0)
    n_need_median = int(need_median.sum())
    if n_need_median > 0:
        t0 = time.perf_counter()
        max_cov = int(total_count.max())
        CHUNK = 2048
        pass2_count = 0
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
                samples = cand[:k, uy[sel], ux[sel], :]
                med = np.median(samples, axis=0).astype(np.uint8)
                canvas[y_start + uy[sel], ux[sel]] = med
                pass2_count += len(sel)
        print(f"[stitch] pass2 (all-band median) filled +{pass2_count} pixels "
              f"({100.0*pass2_count/(canvas_h*W):.2f}%) in "
              f"{time.perf_counter() - t0:.2f}s")

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

    # Pass 4: spatial inpaint for total_count == 0 pixels (truly uncovered).
    no_coverage = total_count == 0
    n_nocov = int(no_coverage.sum())
    if n_nocov > 0:
        t0 = time.perf_counter()
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            distance_transform_edt = None
        if distance_transform_edt is not None:
            inds = distance_transform_edt(no_coverage, return_distances=False,
                                          return_indices=True)
            ny, nx = inds[0], inds[1]
            canvas[no_coverage] = canvas[ny[no_coverage], nx[no_coverage]]
            print(f"[stitch] pass4 (spatial inpaint of uncovered) filled {n_nocov} "
                  f"pixels in {time.perf_counter() - t0:.2f}s")

    if overlay_report is not None:
        overlay_report["canvas_pixels_total"] = int(canvas_h * W)
        overlay_report["canvas_pass1_clean_mean"] = n_pass1
        overlay_report["canvas_pass2_all_band_median"] = int(n_need_median)
        overlay_report["canvas_pass3_strip_inpainted"] = n_strip
        overlay_report["canvas_pass4_spatial_inpainted_no_coverage"] = n_nocov
        overlay_report["clean_coverage_mean"] = round(float(clean_count.mean()), 3)
        overlay_report["total_coverage_mean"] = round(float(total_count.mean()), 3)
        overlay_report["strip_runs"] = [{"x0": r[0], "x1": r[1]} for r in runs]
    unfilled = n_nocov  # remains > 0 only if pass4 fell back and failed

    # ----- Write chunks -----
    writer = ChunkWriter(out_dir=args.out, width=W, chunk_height=args.chunk_height,
                         prefix="keyframe_chunk")
    if args.ui == "keep-once":
        writer.append(static_top_rgb)
    writer.append(canvas)
    if args.ui == "keep-once":
        writer.append(static_bot_rgb)
    chunks_log = writer.finalize()
    print(f"[write] {len(chunks_log)} chunks, {writer.total_rows} total rows")

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
        },
        "n_keyframes": len(kfs),
        "n_bridges": bridge_count,
        "n_placements": len(placements),
        "canvas_height": canvas_h,
        "total_rows_written": writer.total_rows,
        "mask_clean_pixels_per_band": int(clean_mask.sum()),
        "canvas_unfilled_pixels": int(unfilled),
        "overlay_detection": overlay_report,
        "placements": placements_sorted,
        "chunks": chunks_log,
    }
    rpath = args.out / "report.json"
    rpath.write_text(json.dumps(report, indent=2))
    print(f"[write] {rpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
