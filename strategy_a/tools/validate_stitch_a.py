"""Validate a stitched output against the source video.

Premise (user-supplied invariant): every line of conversation content is
present somewhere in the source video with perfect pixel alignment for at
least some subset of consecutive frames. Therefore:

  * For every consecutive frame pair (i, i+1), the recorded relative
    displacement dy = y_top[i+1] - y_top[i] implies an overlap window in
    the dynamic band. Within that window, frame i and frame i+1 MUST be
    pixel-identical (modulo trivial compression noise). Any disagreement
    is a matcher error (wrong dy → shear).
  * For every frame, the recorded canvas position y_top must place the
    frame's dynamic band on a region of the stitched canvas that is
    pixel-identical to the frame. Otherwise the frame was misplaced.
  * Every canvas row should have at least one supporting frame placement.
    A row with zero support is a true gap.

This validator re-decodes the source video, loads the stitched canvas from
chunk PNGs + report.json metadata, and runs the three checks above.

Outputs (all written under <out>/validation/):
  - per_frame.json    : i, y_top, dy, pair_mad, canvas_mad, best_dy_offset,
                        best_mad
  - bad_pairs.json    : list of frame pairs with pair_mad > pair_threshold
  - bad_placements.json : list of frames with canvas_mad > canvas_threshold
                          OR best_dy_offset != 0 with significantly better
                          MAD (the recorded placement disagrees with the
                          true best alignment).
  - coverage.png      : visualization of per-row support count over canvas
  - summary.txt       : human-readable summary

Run:
  python tools/validate_stitch.py --input lexiconv.mp4 --out out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse ffmpeg helpers from the stitcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_scroll_a import (  # type: ignore  # noqa: E402
    _resolve_ffmpeg,
    close_proc,
    open_rgb_pipe,
    probe_video,
    read_frame,
)


def to_luma(rgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 RGB to HxW float32 luma (Rec. 601 weights)."""
    return (
        0.299 * rgb[..., 0].astype(np.float32)
        + 0.587 * rgb[..., 1].astype(np.float32)
        + 0.114 * rgb[..., 2].astype(np.float32)
    )


def load_stitched_dynamic_canvas(
    report: dict, out_dir: Path
) -> tuple[np.ndarray, int]:
    """Load the dynamic portion of the stitched canvas as one big float32 luma.

    Returns
    -------
    canvas_luma : (canvas_rows, W) float32 luma
    canvas_top_y : int (== report['result']['canvas_extent']['min_top'])
    """
    chunks = report["result"]["chunks"]
    static_top_h = report["static"]["top_end"]  # rows of static-top UI
    extent = report["result"]["canvas_extent"]
    canvas_top_y = extent["min_top"]
    canvas_rows = extent["max_bot"] - extent["min_top"]

    parts = []
    total = 0
    for c in chunks:
        im = np.asarray(Image.open(out_dir.parent / Path(c["path"])))
        parts.append(im)
        total += im.shape[0]
    full = np.concatenate(parts, axis=0)  # (total_rows, W, 3)
    # Slice out only the dynamic portion (drop static_top header and
    # static_bot footer). The dynamic portion is contiguous and exactly
    # canvas_rows tall.
    static_bot_start_in_stitched = static_top_h + canvas_rows
    dyn_rgb = full[static_top_h:static_bot_start_in_stitched]
    if dyn_rgb.shape[0] != canvas_rows:
        raise RuntimeError(
            f"stitched canvas height mismatch: expected {canvas_rows} "
            f"rows, got {dyn_rgb.shape[0]}"
        )
    return to_luma(dyn_rgb), canvas_top_y


def overlap_mad(
    a_luma: np.ndarray, b_luma: np.ndarray, dy: int
) -> tuple[float, int]:
    """Mean absolute diff over the overlap of two same-shape dyn-band lumas.

    Both arrays are (dyn_h, W). dy is the canvas-y offset of b relative to a
    (i.e. b's content sits dy rows below a's). Positive dy: a's bottom rows
    overlap with b's top rows.

    Returns
    -------
    mad : mean abs diff over the overlap
    overlap_rows : number of rows in the overlap (0 if no overlap)
    """
    dyn_h = a_luma.shape[0]
    if abs(dy) >= dyn_h:
        return float("inf"), 0
    if dy >= 0:
        a_view = a_luma[dy:dyn_h]
        b_view = b_luma[0 : dyn_h - dy]
    else:
        a_view = a_luma[0 : dyn_h + dy]
        b_view = b_luma[-dy:dyn_h]
    overlap = a_view.shape[0]
    if overlap == 0:
        return float("inf"), 0
    diff = np.abs(a_view - b_view)
    return float(diff.mean()), int(overlap)


def best_offset_mad(
    frame_luma: np.ndarray, canvas_luma: np.ndarray, center_row: int, radius: int
) -> tuple[int, float]:
    """Slide frame's dyn band over a window of the canvas, return best (offset, mad).

    The search probes rows [center_row + d : center_row + d + dyn_h] for
    d in [-radius, +radius]. Returns the d that minimizes MAD and the MAD
    value at that d. Skips any position that falls outside the canvas.
    """
    dyn_h, W = frame_luma.shape
    cn = canvas_luma.shape[0]
    best_d = 0
    best_m = float("inf")
    for d in range(-radius, radius + 1):
        start = center_row + d
        end = start + dyn_h
        if start < 0 or end > cn:
            continue
        window = canvas_luma[start:end]
        m = float(np.abs(window - frame_luma).mean())
        if m < best_m:
            best_m = m
            best_d = d
    return best_d, best_m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Source video the stitch was made from")
    ap.add_argument("--out", required=True, type=Path,
                    help="Stitch output directory (containing report.json + chunk_*.png)")
    ap.add_argument("--pair-mad-warn", type=float, default=1.0,
                    help="Pair-overlap MAD warning threshold (luma 0..255)")
    ap.add_argument("--canvas-mad-warn", type=float, default=2.0,
                    help="Canvas-placement MAD warning threshold (luma 0..255)")
    ap.add_argument("--search-radius", type=int, default=200,
                    help="Radius (rows) for best-offset slide search around recorded y")
    ap.add_argument("--every", type=int, default=1,
                    help="Validate every Nth frame for canvas check (pair check always runs)")
    args = ap.parse_args()

    out_dir: Path = args.out
    report_path = out_dir / "report.json"
    report = json.loads(report_path.read_text())

    ffmpeg, ffprobe = _resolve_ffmpeg()
    vinfo = probe_video(ffprobe, args.input)
    W, H = vinfo["width"], vinfo["height"]
    top = report["static"]["top_end"]
    bot = report["static"]["bottom_start"]
    dyn_h = bot - top
    fbytes = W * H * 3

    canvas_luma, canvas_top_y = load_stitched_dynamic_canvas(report, out_dir)
    print(f"[load] canvas: {canvas_luma.shape}  top_y={canvas_top_y}")

    frames_meta = {f["i"]: f for f in report["frames"]}

    val_dir = out_dir / "validation"
    val_dir.mkdir(exist_ok=True)

    per_frame_records = []
    bad_pairs = []
    bad_placements = []

    # Coverage map: per canvas row, how many frames cover it?
    coverage = np.zeros(canvas_luma.shape[0], dtype=np.int32)

    proc = open_rgb_pipe(ffmpeg, args.input)
    prev_luma: np.ndarray | None = None
    prev_y: int | None = None
    prev_i: int | None = None

    # Track frames flagged for slide-search refinement (phase 2)
    flagged_for_search: list[int] = []
    # Cache: i -> (y_top, luma copy) for flagged frames we'll re-examine
    # (cheaper to re-decode in phase 2 than hold all frames in RAM)

    i = 0
    try:
        while True:
            buf = read_frame(proc, fbytes)
            if buf is None:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            dyn = frame[top:bot]
            luma = to_luma(dyn)

            meta = frames_meta.get(i)
            y_top = meta.get("y_top") if meta else None
            is_drag = bool(meta.get("is_drag")) if meta else False

            # Coverage (only count frames that actually placed; skip drag-frozen)
            if y_top is not None and not is_drag:
                rel = y_top - canvas_top_y
                if 0 <= rel and rel + dyn_h <= coverage.shape[0]:
                    coverage[rel:rel + dyn_h] += 1

            # ----- pair-overlap check (fast, every frame) -----
            pair_mad = None
            pair_overlap = None
            pair_dy = None
            if prev_luma is not None and y_top is not None and prev_y is not None:
                dy = y_top - prev_y
                pair_dy = int(dy)
                if not is_drag and not (meta and meta.get("stationary")):
                    mad, ov = overlap_mad(prev_luma, luma, dy)
                    pair_mad = mad
                    pair_overlap = ov
                    if mad != float("inf") and mad > args.pair_mad_warn and ov >= 50:
                        bad_pairs.append({
                            "from": prev_i, "to": i,
                            "dy": int(dy),
                            "overlap": int(ov),
                            "mad": float(mad),
                            "from_y_top": int(prev_y),
                            "to_y_top": int(y_top),
                        })

            # ----- canvas-placement check at RECORDED position (fast, every Nth) -----
            canvas_mad = None
            if (
                i % args.every == 0
                and y_top is not None
                and not is_drag
            ):
                rel = y_top - canvas_top_y
                if 0 <= rel and rel + dyn_h <= canvas_luma.shape[0]:
                    canvas_mad = float(
                        np.abs(canvas_luma[rel:rel + dyn_h] - luma).mean()
                    )
                    if canvas_mad > args.canvas_mad_warn:
                        flagged_for_search.append(i)

            # Also flag frames whose pair-MAD is bad — they likely need
            # a slide-search to find the true position.
            if pair_mad is not None and pair_mad > args.pair_mad_warn:
                if i not in flagged_for_search:
                    flagged_for_search.append(i)

            per_frame_records.append({
                "i": i,
                "y_top": int(y_top) if y_top is not None else None,
                "is_drag": is_drag,
                "stationary": bool(meta.get("stationary")) if meta else False,
                "pair_dy": pair_dy,
                "pair_overlap": pair_overlap,
                "pair_mad": pair_mad,
                "canvas_mad": canvas_mad,
                "best_dy_offset": None,
                "best_mad": None,
            })

            if i % 200 == 0:
                pm = "-" if pair_mad is None else f"{pair_mad:.3f}"
                cm = "-" if canvas_mad is None else f"{canvas_mad:.3f}"
                print(f"[val] phase1 i={i:4d}  y_top={y_top}  pair_mad={pm}  canvas_mad={cm}")

            prev_luma = luma
            prev_y = y_top if (y_top is not None and not is_drag) else prev_y
            prev_i = i if (y_top is not None and not is_drag) else prev_i
            i += 1
    finally:
        close_proc(proc)

    print(f"[phase1] complete. {len(flagged_for_search)} frames flagged for slide-search.")

    # ----- phase 2: slide-search only flagged frames -----
    if flagged_for_search:
        flagged_set = set(flagged_for_search)
        proc = open_rgb_pipe(ffmpeg, args.input)
        idx_to_record = {r["i"]: r for r in per_frame_records}
        i = 0
        try:
            while True:
                buf = read_frame(proc, fbytes)
                if buf is None:
                    break
                if i in flagged_set:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
                    luma = to_luma(frame[top:bot])
                    meta = frames_meta.get(i)
                    y_top = meta.get("y_top") if meta else None
                    if y_top is not None:
                        rel = y_top - canvas_top_y
                        if 0 <= rel and rel + dyn_h <= canvas_luma.shape[0]:
                            bd, bm = best_offset_mad(
                                luma, canvas_luma, rel, args.search_radius
                            )
                            rec = idx_to_record[i]
                            rec["best_dy_offset"] = int(bd)
                            rec["best_mad"] = float(bm)
                            cm = rec["canvas_mad"]
                            if cm is None or bm + 0.5 < cm or cm > args.canvas_mad_warn:
                                bad_placements.append({
                                    "i": i,
                                    "y_top": int(y_top),
                                    "canvas_mad": cm,
                                    "best_dy_offset": int(bd),
                                    "best_mad": float(bm),
                                    "is_drag": bool(meta.get("is_drag")) if meta else False,
                                    "stationary": bool(meta.get("stationary")) if meta else False,
                                })
                    if len(bad_placements) % 25 == 0 and bad_placements:
                        print(f"[phase2] i={i:4d}  bad_placements so far: {len(bad_placements)}")
                i += 1
        finally:
            close_proc(proc)

    # ----- coverage analysis -----
    gap_rows = int(np.sum(coverage == 0))
    weak_rows = int(np.sum((coverage > 0) & (coverage < 2)))
    cov_min = int(coverage.min())
    cov_max = int(coverage.max())
    cov_mean = float(coverage.mean())

    # ----- save artifacts -----
    (val_dir / "per_frame.json").write_text(json.dumps(per_frame_records))
    (val_dir / "bad_pairs.json").write_text(json.dumps(bad_pairs, indent=2))
    (val_dir / "bad_placements.json").write_text(json.dumps(bad_placements, indent=2))

    # Coverage visualization: 1-pixel-per-row bar, color-coded.
    cov_img = np.zeros((coverage.shape[0], 200, 3), dtype=np.uint8)
    for r, c in enumerate(coverage):
        if c == 0:
            cov_img[r] = (255, 0, 0)  # gap = red
        elif c == 1:
            cov_img[r] = (255, 180, 0)  # weak = orange
        else:
            # green scaled by support, capped
            g = min(80 + c * 4, 255)
            cov_img[r] = (0, g, 0)
    Image.fromarray(cov_img).save(val_dir / "coverage.png")

    # ----- summary -----
    lines = []
    lines.append(f"Validator summary")
    lines.append(f"=================")
    lines.append(f"Source video    : {args.input}")
    lines.append(f"Stitch output   : {out_dir}")
    lines.append(f"Frames validated: {len(per_frame_records)}")
    lines.append(f"Canvas size     : {canvas_luma.shape[0]} rows x {canvas_luma.shape[1]} cols")
    lines.append(f"Canvas top_y    : {canvas_top_y}")
    lines.append("")
    lines.append(f"Pair-overlap check (threshold MAD > {args.pair_mad_warn}):")
    lines.append(f"  bad pairs: {len(bad_pairs)}")
    if bad_pairs:
        worst = sorted(bad_pairs, key=lambda x: -x["mad"])[:10]
        for b in worst:
            lines.append(f"    pair {b['from']}->{b['to']}  dy={b['dy']:+d}  overlap={b['overlap']}  mad={b['mad']:.2f}")
    lines.append("")
    lines.append(f"Canvas-placement check (threshold MAD > {args.canvas_mad_warn}, search_radius={args.search_radius}):")
    lines.append(f"  bad placements: {len(bad_placements)}")
    if bad_placements:
        def _cm(x: dict) -> float:
            v = x.get("canvas_mad")
            return float(v) if v is not None else float("-inf")
        worst = sorted(bad_placements, key=lambda x: -_cm(x))[:10]
        for b in worst:
            cm_s = f"{b['canvas_mad']:.2f}" if b.get("canvas_mad") is not None else "n/a"
            bm_s = f"{b['best_mad']:.2f}" if b.get("best_mad") is not None else "n/a"
            lines.append(
                f"    frame {b['i']}  y_top={b['y_top']}  recorded_mad={cm_s}  "
                f"best_d={b['best_dy_offset']:+d} best_mad={bm_s}"
            )
    lines.append("")
    lines.append(f"Coverage:")
    lines.append(f"  rows with support=0 (true gap): {gap_rows}")
    lines.append(f"  rows with support=1 (weak)    : {weak_rows}")
    lines.append(f"  support min/mean/max          : {cov_min} / {cov_mean:.1f} / {cov_max}")
    summary = "\n".join(lines)
    (val_dir / "summary.txt").write_text(summary)
    print()
    print(summary)
    return 0


# ---------------------------------------------------------------------------
# STUB: drag-reveals-data detection (future work)
# ---------------------------------------------------------------------------
# Some videos (e.g. iOS SMS) reveal additional metadata (timestamps) only
# while the user drags the whole conversation horizontally. The dragged
# columns contain real data that must be preserved alongside the main
# stitch. The plan is:
#   1. During drag events (already detected), capture the deepest-drag
#      frame (already done as drag_NNN.png sidecars).
#   2. Extract the newly-revealed columns by diffing pre-drag and
#      deepest-drag frames over a band around the drag axis.
#   3. Associate those columns as metadata against the canvas rows they
#      were taken from (drag y-position at the time).
#   4. Emit a metadata sidecar JSON mapping canvas-row-range -> revealed
#      columns image path.
# For now: drag events are captured as full-frame sidecars only. No
# revealed-column extraction is performed.
def extract_drag_revealed_columns_stub(*_args, **_kwargs):  # pragma: no cover
    """Placeholder for future drag-reveals-data extraction. See note above."""
    raise NotImplementedError(
        "drag-reveals-data extraction not yet implemented; "
        "drag events are currently captured only as deepest-drag sidecars."
    )


if __name__ == "__main__":
    sys.exit(main())
