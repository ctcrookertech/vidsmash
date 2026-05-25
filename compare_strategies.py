"""Compare Strategy A vs Strategy B stitched outputs.

Reads chunk_*.png from each strategy's out/ directory, optionally trims
their static UI bands (read from each strategy's report.json), reassembles
each into a single tall RGB image in memory, and computes:

  - Total height, chunk count, file sizes for each strategy
  - Width sanity check (must match)
  - Vertical alignment offset between A and B using a row-profile match
    (gray row-mean over the content portion)
  - Per-row mean absolute difference along the overlap region after the
    best alignment is applied (low MAD = strategies agree)

Also emits a side-by-side preview PNG: A's content on the left, B's
content on the right, both downscaled to a target height for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str((Path(__file__).resolve().parent / "strategy_b" / "tools")))
from stitch_scroll_b import gray_row_profile, match_1d_offset  # type: ignore  # noqa: E402


def concat_chunks(chunk_paths: list[Path]) -> np.ndarray:
    """Load and vertically concatenate ordered chunk_*.png files into one RGB array."""
    bands: list[np.ndarray] = []
    for p in chunk_paths:
        with Image.open(p) as im:
            bands.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    if not bands:
        raise RuntimeError("no chunks to concat")
    W = bands[0].shape[1]
    for b in bands:
        if b.shape[1] != W:
            raise RuntimeError(f"chunk width mismatch: {b.shape[1]} vs {W}")
    return np.concatenate(bands, axis=0)


def load_strategy(dir_a_or_b: Path, prefix: str) -> tuple[np.ndarray, dict]:
    """Load all chunk_*.png from a strategy directory along with its report.json."""
    report_path = dir_a_or_b / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    chunks = sorted(dir_a_or_b.glob(f"{prefix}*.png"))
    if not chunks:
        raise FileNotFoundError(f"no {prefix}*.png in {dir_a_or_b}")
    print(f"[load] {dir_a_or_b}: {len(chunks)} chunks, prefix={prefix!r}")
    img = concat_chunks(chunks)
    return img, report


def trim_static(img: np.ndarray, top: int, bot_keep_from: int) -> np.ndarray:
    """Drop the top `top` rows and the bottom rows starting from height - bot_keep_from."""
    h = img.shape[0]
    if top + bot_keep_from >= h:
        raise RuntimeError(f"trim would empty image: h={h} top={top} bot_keep_from={bot_keep_from}")
    return img[top : h - bot_keep_from]


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return (
        0.299 * img[..., 0].astype(np.float32)
        + 0.587 * img[..., 1].astype(np.float32)
        + 0.114 * img[..., 2].astype(np.float32)
    ).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-dir", type=Path, default=Path("strategy_a/out"),
                    help="Directory containing Strategy A's chunk_*.png + report.json.")
    ap.add_argument("--a-prefix", default="chunk_",
                    help="Filename prefix for A's chunks.")
    ap.add_argument("--b-dir", type=Path, default=Path("strategy_b/out/stitch"),
                    help="Directory containing Strategy B's keyframe_chunk_*.png + report.json.")
    ap.add_argument("--b-prefix", default="keyframe_chunk_",
                    help="Filename prefix for B's chunks.")
    ap.add_argument("--out", type=Path, default=Path("compare_out"),
                    help="Output directory for compare_preview.png and compare_report.json.")
    ap.add_argument("--preview-height", type=int, default=4000,
                    help="Downscale each strategy's content image to this height in the preview.")
    ap.add_argument("--search-radius", type=int, default=500,
                    help="Vertical search radius (rows) when aligning A vs B content.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    img_a, rep_a = load_strategy(args.a_dir, args.a_prefix)
    img_b, rep_b = load_strategy(args.b_dir, args.b_prefix)
    if img_a.shape[1] != img_b.shape[1]:
        raise RuntimeError(f"width mismatch: A={img_a.shape[1]} B={img_b.shape[1]}")
    W = img_a.shape[1]

    # --- Determine static-band trim for each strategy from its report ---
    a_top = int(rep_a.get("static", {}).get("top_end", 0))
    a_h = int(rep_a.get("video", {}).get("height", img_a.shape[0]))
    a_bot_start = int(rep_a.get("static", {}).get("bottom_start", a_h))
    a_bot_keep = a_h - a_bot_start

    b_v = rep_b.get("video", {})
    b_top = int(b_v.get("dyn_top", 0))
    b_h = int(b_v.get("h", img_b.shape[0]))
    b_bot_start = int(b_v.get("dyn_bot", b_h))
    b_bot_keep = b_h - b_bot_start

    content_a = trim_static(img_a, a_top, a_bot_keep)
    content_b = trim_static(img_b, b_top, b_bot_keep)
    print(f"[trim] A: full={img_a.shape[0]} -> content={content_a.shape[0]} (cut top={a_top}, bot={a_bot_keep})")
    print(f"[trim] B: full={img_b.shape[0]} -> content={content_b.shape[0]} (cut top={b_top}, bot={b_bot_keep})")

    # --- Align content_a vs content_b along the vertical axis ---
    gray_a = rgb_to_gray(content_a)
    gray_b = rgb_to_gray(content_b)
    prof_a = gray_row_profile(gray_a, hpad=40, n_segments=16)
    prof_b = gray_row_profile(gray_b, hpad=40, n_segments=16)
    print(f"[align] prof_a len={len(prof_a)}  prof_b len={len(prof_b)}")
    res = match_1d_offset(
        ref=prof_a, cur=prof_b,
        predicted_p=0, search_radius=args.search_radius,
        min_overlap=max(500, min(len(prof_a), len(prof_b)) // 4),
        prior_alpha=0.0,
    )
    align_p = int(res.p)
    print(f"[align] best offset (B vs A) = {align_p:+d} rows  mad={res.sad:.2f}  conf={res.confidence:.2f}")

    # --- Per-row MAD over the overlap region ---
    if align_p >= 0:
        a_start = align_p
        b_start = 0
    else:
        a_start = 0
        b_start = -align_p
    overlap = min(content_a.shape[0] - a_start, content_b.shape[0] - b_start)
    if overlap <= 0:
        print(f"[mad] no overlap (align_p={align_p}, a={content_a.shape[0]}, b={content_b.shape[0]})")
        row_mad = -1.0
    else:
        ga = gray_a[a_start : a_start + overlap].astype(np.int16)
        gb = gray_b[b_start : b_start + overlap].astype(np.int16)
        per_row_mad = np.mean(np.abs(ga - gb), axis=1)
        row_mad = float(per_row_mad.mean())
        worst_idx = int(np.argmax(per_row_mad))
        print(f"[mad] overlap={overlap} rows  mean_row_mad={row_mad:.2f}  worst_row_mad={per_row_mad[worst_idx]:.2f} at row={worst_idx}")

    # --- Side-by-side preview ---
    def scale_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
        h, w = arr.shape[:2]
        if h <= target_h:
            return arr
        new_w = max(1, int(round(w * target_h / h)))
        im = Image.fromarray(arr, mode="RGB").resize((new_w, target_h), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)

    prev_a = scale_to_height(content_a, args.preview_height)
    prev_b = scale_to_height(content_b, args.preview_height)
    # pad shorter to match heights
    h_max = max(prev_a.shape[0], prev_b.shape[0])
    def pad_h(arr: np.ndarray, h: int) -> np.ndarray:
        if arr.shape[0] == h:
            return arr
        pad = np.zeros((h - arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        return np.concatenate([arr, pad], axis=0)
    prev_a = pad_h(prev_a, h_max)
    prev_b = pad_h(prev_b, h_max)
    gap = np.full((h_max, 8, 3), 64, dtype=np.uint8)
    side = np.concatenate([prev_a, gap, prev_b], axis=1)
    preview_path = args.out / "compare_preview.png"
    Image.fromarray(side, mode="RGB").save(preview_path)
    print(f"[preview] saved {preview_path}  shape={side.shape}")

    # --- Summary ---
    a_chunk_count = len(sorted(args.a_dir.glob(f"{args.a_prefix}*.png")))
    b_chunk_count = len(sorted(args.b_dir.glob(f"{args.b_prefix}*.png")))
    summary = {
        "width": W,
        "a": {
            "dir": str(args.a_dir),
            "chunks": a_chunk_count,
            "full_height": int(img_a.shape[0]),
            "content_height": int(content_a.shape[0]),
            "report_keys": sorted(rep_a.keys()),
        },
        "b": {
            "dir": str(args.b_dir),
            "chunks": b_chunk_count,
            "full_height": int(img_b.shape[0]),
            "content_height": int(content_b.shape[0]),
            "report_keys": sorted(rep_b.keys()),
        },
        "height_diff_rows": int(content_b.shape[0] - content_a.shape[0]),
        "alignment_offset_rows_b_vs_a": align_p,
        "alignment_mad": float(res.sad),
        "alignment_confidence": float(res.confidence),
        "overlap_rows_after_align": int(overlap),
        "mean_row_mad_over_overlap": float(row_mad),
        "preview_png": str(preview_path),
        "elapsed_s": round(time.perf_counter() - t_total, 2),
    }
    rpath = args.out / "compare_report.json"
    rpath.write_text(json.dumps(summary, indent=2))
    print(f"[write] {rpath}")
    print("--- summary ---")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
