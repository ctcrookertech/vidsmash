"""Side-by-side diff of button-region patches between two stitch outputs.

For each band where the circle detector fired, computes the canvas position of
the masked disk and crops a fixed-size patch from BOTH the baseline output and
the circle-masked output. Stitches the patches into a vertical strip:
  [baseline patch | circle-masked patch | abs-diff x4]

Use to visually verify the scroll-to-latest button ghost has been suppressed.

Usage:
    python tools\\diff_circle_regions.py \
        --baseline out\\stitch_extents \
        --circles  out\\stitch_circles \
        --out      out\\diag\\circle_diff.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_chunks(stitch_dir: Path) -> tuple[list[Image.Image], list[int]]:
    """Return (chunks, cumulative_top_y) where chunks[i] is the PNG and
    cumulative_top_y[i] is the canvas-y of its top row (after static_top is
    stripped, so chunk_000's first row is canvas-y 0)."""
    paths = sorted(stitch_dir.glob("keyframe_chunk_*.png"))
    chunks = [Image.open(p).convert("RGB") for p in paths]
    cum_y = []
    y = 0
    for c in chunks:
        cum_y.append(y)
        y += c.size[1]
    return chunks, cum_y


def _crop_at(chunks: list[Image.Image], cum_y: list[int], total_h: int,
             canvas_y: int, canvas_x: int, patch_h: int, patch_w: int) -> Image.Image:
    """Crop a (patch_h, patch_w) RGB patch centered at (canvas_y, canvas_x),
    walking the chunks. Falls back to black where out of range."""
    y0 = canvas_y - patch_h // 2
    y1 = y0 + patch_h
    x0 = max(0, canvas_x - patch_w // 2)
    x1 = x0 + patch_w
    out = Image.new("RGB", (patch_w, patch_h), (0, 0, 0))
    for ci, c in enumerate(chunks):
        ctop = cum_y[ci]
        cbot = ctop + c.size[1]
        if cbot <= y0 or ctop >= y1:
            continue
        src_y0 = max(0, y0 - ctop)
        src_y1 = min(c.size[1], y1 - ctop)
        dst_y0 = max(0, ctop - y0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        crop = c.crop((x0, src_y0, x1, src_y1))
        out.paste(crop, (0, dst_y0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path,
                    help="Stitch output dir WITHOUT circle masking (the artifact-prone one).")
    ap.add_argument("--circles", required=True, type=Path,
                    help="Stitch output dir WITH --mask-detected-circles (the fix).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--patch", type=int, default=180,
                    help="Patch height/width in px. Default 180 (covers r=60 + margin).")
    ap.add_argument("--max-samples", type=int, default=24,
                    help="Max number of detection sites to render. Default 24.")
    args = ap.parse_args()

    report_path = args.circles / "report.json"
    report = json.loads(report_path.read_text())
    circ = report.get("circle_per_band")
    if not circ:
        print(f"[diff] no circle_per_band in {report_path} - was --mask-detected-circles used?")
        return 1
    dets = circ["detections"]
    if not dets:
        print(f"[diff] zero detections in report")
        return 1
    placements = report["placements"]
    # placements has per-entry "i" (band frame idx) and "abs_y" (canvas top).
    # static_top is prepended to chunk_000, so chunk-cumulative-y == canvas_y + static_top_h.
    # We crop directly in chunk-cumulative-y space, so we need that offset.
    static_top_h = int(report["video"]["dyn_top"])  # static top = rows 0..dyn_top
    by_band: dict[int, int] = {}
    for p in placements:
        by_band[int(p["i"])] = int(p["abs_y"])

    min_top = min(by_band.values())  # canvas-y is anchored at frame-0 band; offset to 0..canvas_h
    # canvas first row = abs_y == min_top. In chunk-coords, that's at static_top_h.

    base_chunks, base_cum = _load_chunks(args.baseline)
    circ_chunks, circ_cum = _load_chunks(args.circles)
    base_h = sum(c.size[1] for c in base_chunks)
    circ_h = sum(c.size[1] for c in circ_chunks)
    print(f"[diff] baseline {len(base_chunks)} chunks total {base_h}px  "
          f"circles {len(circ_chunks)} chunks total {circ_h}px")

    # Pick samples: spread across detection list, capped at max_samples
    n = min(args.max_samples, len(dets))
    stride = max(1, len(dets) // n)
    selected = dets[::stride][:n]
    print(f"[diff] {len(dets)} detections, sampling {len(selected)}")

    patch_h = args.patch
    patch_w = args.patch
    rows = []
    for d in selected:
        band_i = int(d["band_i"])
        det_cx = int(d["det_cx"])
        det_cy = int(d["det_cy"])  # in band coords (0..dyn_h)
        if band_i not in by_band:
            continue
        abs_y = by_band[band_i]  # canvas-y of band top
        canvas_y = abs_y - min_top + det_cy  # canvas-y of button center
        chunk_y = canvas_y + static_top_h     # chunk-cumulative-y
        base_patch = _crop_at(base_chunks, base_cum, base_h, chunk_y, det_cx, patch_h, patch_w)
        circ_patch = _crop_at(circ_chunks, circ_cum, circ_h, chunk_y, det_cx, patch_h, patch_w)
        base_arr = np.asarray(base_patch).astype(np.int16)
        circ_arr = np.asarray(circ_patch).astype(np.int16)
        diff = np.clip(np.abs(base_arr - circ_arr).astype(np.int16) * 4, 0, 255).astype(np.uint8)
        diff_img = Image.fromarray(diff, mode="RGB")
        # Side-by-side with 2px black separator
        row = Image.new("RGB", (patch_w * 3 + 4, patch_h), (32, 32, 32))
        row.paste(base_patch, (0, 0))
        row.paste(circ_patch, (patch_w + 2, 0))
        row.paste(diff_img, (patch_w * 2 + 4, 0))
        rows.append(row)
        print(f"[diff] band={band_i} canvas_y={canvas_y} chunk_y={chunk_y} det=({det_cx},{det_cy}) r={d['det_r']}")

    if not rows:
        print(f"[diff] no rows produced")
        return 1
    sep_h = 4
    total_h = sum(r.size[1] for r in rows) + sep_h * (len(rows) - 1)
    out = Image.new("RGB", (rows[0].size[0], total_h), (16, 16, 16))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1] + sep_h
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"[diff] wrote {args.out} ({out.size[0]}x{out.size[1]})  "
          f"layout: [baseline | circles | abs-diff x4]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
