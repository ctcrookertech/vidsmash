"""Emit a minimal view.html that vertically renders ordered chunk_*.png files.

Generic helper used by output directories. Picks files by glob, sorts them
by filename, writes a 1126-pixel-wide stack to view.html in the same dir.

Usage:
  python tools/make_view_html.py --dir out/stitch --glob "keyframe_chunk_*.png" --title "vidsmash"
"""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width={width}, initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: {bg}; color: #ddd; font-family: -apple-system, Segoe UI, sans-serif; }}
  h1 {{ font-size: 14px; font-weight: 400; margin: 8px 12px; opacity: 0.6; }}
  .stream {{ display: block; width: {width}px; max-width: 100%; margin: 0 auto; background: {chunk_bg}; }}
  .stream img {{ display: block; width: 100%; height: auto; margin: 0; padding: 0; border: 0; }}
</style>
</head>
<body>
<h1>{title} — {n} image(s)</h1>
<div class="stream">
{imgs}
</div>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path,
                    help="Directory containing the images and where view.html will be written.")
    ap.add_argument("--glob", required=True,
                    help="Glob pattern (relative to --dir) selecting the images to include, in name-sorted order.")
    ap.add_argument("--title", default="stitched",
                    help="Page title and h1 caption.")
    ap.add_argument("--width", type=int, default=1126,
                    help="CSS render width for the image column.")
    ap.add_argument("--out-name", default="view.html",
                    help="Output filename written into --dir.")
    ap.add_argument("--bg", default="#000",
                    help="CSS background for the body (outside the chunk column).")
    ap.add_argument("--chunk-bg", default="#1f1f1f",
                    help="CSS background BEHIND the chunk images. Shows through any transparent (RGBA alpha=0) pixels — pick something that contrasts with both message bubbles and the iOS dark gutter so holes are obvious.")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"--dir is not a directory: {args.dir}")
    files = sorted(args.dir.glob(args.glob))
    if not files:
        raise SystemExit(f"no files match {args.glob} in {args.dir}")
    imgs = "\n".join(
        f'  <img src="{f.name}" alt="{f.stem}">' for f in files
    )
    html = TEMPLATE.format(
        title=args.title, width=args.width, n=len(files), imgs=imgs,
        bg=args.bg, chunk_bg=args.chunk_bg,
    )
    out = args.dir / args.out_name
    out.write_text(html, encoding="utf-8")
    print(f"[write] {out} ({len(files)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
