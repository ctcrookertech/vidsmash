"""OCR validation: compare source-video OCR against stitched-chunk OCR.

Uses Windows native OCR (winsdk.windows.media.ocr) since the input is
crisp screen-rendered iOS text, which Windows OCR handles well.

Process
-------
1. Decode the source video once and capture the RGB frame at every
   keyframe index from out/keyframes.json. Pause-midpoint frames are the
   most stable (no motion blur).
2. OCR each captured keyframe frame -> list of normalized lines.
3. OCR each keyframe_chunk_*.png in the stitch directory -> list of
   normalized lines per chunk, plus a flat ordered concatenation.
4. Compare:
     missing-from-stitch:   source line has no fuzzy match >= threshold
                            in the stitched output (potential lost text).
     no-source-match:       stitched line has no fuzzy match in source
                            (potential corrupt OCR or extra content).
     duplicated-in-stitch:  stitched line of length >= 8 chars appears
                            two or more times across all chunks.

Writes per-category artifact files plus report.json into --out.

Outputs
-------
  source_lines.txt   one normalized source line per row
  stitch_lines.txt   "<chunk>\t<line>" rows in stitched order
  missing.txt        "[score=NN] SRC=... BEST_STITCH=..." per missing line
  corrupt.txt        "[score=NN] STITCH=... BEST_SRC=..." per unmatched stitch line
  duplicates.txt     "xN: <line>" for every line repeated in the stitch
  report.json        aggregate counts + recall / precision
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_io import (  # type: ignore  # noqa: E402
    _resolve_ffmpeg,
    close_proc,
    open_rgb_pipe,
    probe_video,
    read_frame,
)

from winsdk.windows.graphics.imaging import BitmapDecoder  # type: ignore
from winsdk.windows.media.ocr import OcrEngine  # type: ignore
from winsdk.windows.storage.streams import (  # type: ignore
    DataWriter,
    InMemoryRandomAccessStream,
)

try:
    from rapidfuzz import fuzz, process

    USE_RF = True
except ImportError:  # pragma: no cover
    USE_RF = False
    from difflib import SequenceMatcher

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = OcrEngine.try_create_from_user_profile_languages()
        if _engine is None:
            raise RuntimeError(
                "No Windows OCR engine available for user-profile languages."
            )
    return _engine


async def ocr_image_bytes(png_bytes: bytes) -> list[str]:
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream.get_output_stream_at(0))
    writer.write_bytes(png_bytes)
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = get_engine()
    result = await engine.recognize_async(bitmap)
    return [line.text for line in result.lines]


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def normalize_line(s: str) -> str:
    return " ".join(s.split()).strip()


def fuzz_score(a: str, b: str) -> int:
    if USE_RF:
        return int(fuzz.token_set_ratio(a, b))
    return int(SequenceMatcher(None, a, b).ratio() * 100)


def best_match(needle: str, haystack: list[str]) -> tuple[str, int]:
    if not haystack:
        return ("", 0)
    if USE_RF:
        res = process.extractOne(needle, haystack, scorer=fuzz.token_set_ratio)
        if res is None:
            return ("", 0)
        return (res[0], int(res[1]))
    best = ("", 0)
    for h in haystack:
        s = fuzz_score(needle, h)
        if s > best[1]:
            best = (h, s)
    return best


async def ocr_many(images: list[tuple[str, bytes]], label: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    t0 = time.time()
    for i, (name, data) in enumerate(images, 1):
        lines = await ocr_image_bytes(data)
        out[name] = lines
        if i % 5 == 0 or i == len(images):
            print(f"  [{label}] {i}/{len(images)}  ({time.time()-t0:.1f}s)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="Source video.")
    ap.add_argument("--keyframes", required=True, type=Path,
                    help="Path to keyframes.json from detect_pauses.")
    ap.add_argument("--stitch-dir", required=True, type=Path,
                    help="Directory containing keyframe_chunk_*.png.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for OCR artifacts and report.")
    ap.add_argument("--min-line-len", type=int, default=3,
                    help="Drop normalized OCR lines shorter than this.")
    ap.add_argument("--similarity", type=int, default=85,
                    help="rapidfuzz token_set_ratio threshold (0-100).")
    ap.add_argument("--min-dup-len", type=int, default=8,
                    help="Min line length to count as a duplication candidate.")
    ap.add_argument("--source-crop-top", type=int, default=320,
                    help="When OCRing source frames, skip this many rows at "
                         "the top to avoid the iOS status bar/chat header "
                         "(which would duplicate across every keyframe).")
    ap.add_argument("--source-crop-bottom", type=int, default=1200,
                    help="When OCRing source frames, skip this many rows at "
                         "the bottom to avoid the keyboard/composer chrome.")
    ap.add_argument("--token-min-len", type=int, default=4,
                    help="Min token length for the token-level comparison "
                         "(short tokens like 'I', 'a', 'is' add noise).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[load] keyframes from {args.keyframes}")
    with open(args.keyframes) as f:
        kf = json.load(f)
    keyframe_indices = sorted(set(int(k["i"]) for k in kf.get("keyframes", [])))
    if not keyframe_indices:
        print("[fatal] no keyframes in JSON", file=sys.stderr)
        return 2
    print(f"[load] {len(keyframe_indices)} keyframes")

    ffmpeg, ffprobe = _resolve_ffmpeg()
    info = probe_video(ffprobe, args.input)
    W, H = info["width"], info["height"]
    print(f"[probe] {W}x{H}  nb_frames~{info['nb_frames']}")

    frame_bytes = W * H * 3
    proc = open_rgb_pipe(ffmpeg, args.input)

    kf_set = set(keyframe_indices)
    sampled: dict[int, np.ndarray] = {}
    n_read = 0
    t0 = time.time()
    try:
        while True:
            buf = read_frame(proc, frame_bytes)
            if buf is None:
                break
            if n_read in kf_set:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).copy()
                # Crop out the static header (always identical across keyframes
                # so it would dominate dup counts) and keyboard chrome (which
                # OCRs as noise) and any partial lines at the very top/bottom
                # of the dynamic band.
                y0 = max(0, args.source_crop_top)
                y1 = min(H, H - args.source_crop_bottom)
                if y1 > y0:
                    arr = arr[y0:y1]
                sampled[n_read] = arr
            n_read += 1
            if n_read % 500 == 0:
                print(f"  [decode] scanned {n_read} frames, "
                      f"captured {len(sampled)}/{len(keyframe_indices)}")
    finally:
        close_proc(proc)
    print(f"[decode] read {n_read} frames, captured {len(sampled)} "
          f"keyframes in {time.time()-t0:.2f}s")

    # ---- OCR source keyframes ----
    print(f"[ocr-source] OCRing {len(sampled)} keyframes")
    source_images = [
        (f"kf_{idx:05d}", numpy_to_png_bytes(sampled[idx]))
        for idx in sorted(sampled)
    ]
    raw_source = asyncio.run(ocr_many(source_images, "ocr-source"))

    source_lines_per_kf: dict[str, list[str]] = {}
    for name, lines in raw_source.items():
        kept = [
            normalize_line(s)
            for s in lines
            if len(normalize_line(s)) >= args.min_line_len
        ]
        source_lines_per_kf[name] = kept

    source_seen: set[str] = set()
    source_lines_ordered: list[str] = []
    for name in sorted(source_lines_per_kf):
        for line in source_lines_per_kf[name]:
            if line not in source_seen:
                source_seen.add(line)
                source_lines_ordered.append(line)
    print(f"[ocr-source] {len(source_lines_ordered)} unique normalized lines "
          f"across {len(sampled)} keyframes")

    # ---- OCR stitched chunks ----
    chunk_paths = sorted(args.stitch_dir.glob("keyframe_chunk_*.png"))
    print(f"[ocr-stitch] OCRing {len(chunk_paths)} chunks")
    stitch_images = [(p.name, p.read_bytes()) for p in chunk_paths]
    raw_stitch = asyncio.run(ocr_many(stitch_images, "ocr-stitch"))

    stitch_lines_per_chunk: dict[str, list[str]] = {}
    for name, lines in raw_stitch.items():
        kept = [
            normalize_line(s)
            for s in lines
            if len(normalize_line(s)) >= args.min_line_len
        ]
        stitch_lines_per_chunk[name] = kept

    stitch_lines_all: list[tuple[str, str]] = []
    for p in chunk_paths:
        for line in stitch_lines_per_chunk[p.name]:
            stitch_lines_all.append((p.name, line))

    stitch_seen: set[str] = set()
    stitch_lines_unique: list[str] = []
    for _name, line in stitch_lines_all:
        if line not in stitch_seen:
            stitch_seen.add(line)
            stitch_lines_unique.append(line)
    print(f"[ocr-stitch] {len(stitch_lines_all)} total / "
          f"{len(stitch_lines_unique)} unique normalized lines")

    # ---- Compare ----
    print(f"[compare] {'rapidfuzz' if USE_RF else 'difflib'} "
          f"threshold={args.similarity}")

    missing = []
    for s in source_lines_ordered:
        m, score = best_match(s, stitch_lines_unique)
        if score < args.similarity:
            missing.append({"source": s, "best_stitch": m, "score": score})
    print(f"[compare] missing-from-stitch: "
          f"{len(missing)} / {len(source_lines_ordered)}")

    corrupt = []
    for s in stitch_lines_unique:
        m, score = best_match(s, source_lines_ordered)
        if score < args.similarity:
            corrupt.append({"stitch": s, "best_source": m, "score": score})
    print(f"[compare] stitch-with-no-source-match: "
          f"{len(corrupt)} / {len(stitch_lines_unique)}")

    counts = Counter(line for _name, line in stitch_lines_all)
    duplicated = [
        {"line": line, "count": c}
        for line, c in counts.most_common()
        if c >= 2 and len(line) >= args.min_dup_len
    ]
    print(f"[compare] lines appearing 2+ times in stitch "
          f"(len>={args.min_dup_len}): {len(duplicated)}")

    # ---- Token-level comparison (more robust to OCR line-break noise) ----
    import re

    def tokenize(text: str) -> list[str]:
        return [
            t.lower()
            for t in re.findall(r"[A-Za-z][A-Za-z']+", text)
            if len(t) >= args.token_min_len
        ]

    source_blob = " ".join(source_lines_ordered)
    stitch_blob = " ".join(line for _n, line in stitch_lines_all)
    src_tokens = Counter(tokenize(source_blob))
    sti_tokens = Counter(tokenize(stitch_blob))

    src_vocab = set(src_tokens)
    sti_vocab = set(sti_tokens)
    only_in_source = sorted(src_vocab - sti_vocab)
    only_in_stitch = sorted(sti_vocab - src_vocab)

    # Words appearing far more often in stitch than source -> potential
    # duplication signal. Filter to tokens that appear at least 3x in stitch
    # and at least 2x more than in source.
    over_repeated = sorted(
        (
            {"token": t, "source": src_tokens[t], "stitch": sti_tokens[t]}
            for t in sti_vocab
            if sti_tokens[t] >= 3 and sti_tokens[t] >= 2 * max(1, src_tokens[t])
        ),
        key=lambda r: -(r["stitch"] - r["source"]),
    )

    total_src_tokens = sum(src_tokens.values())
    total_sti_tokens = sum(sti_tokens.values())
    shared_vocab = src_vocab & sti_vocab
    token_recall = (
        100.0 * len(shared_vocab) / max(1, len(src_vocab))
    )
    token_precision = (
        100.0 * len(shared_vocab) / max(1, len(sti_vocab))
    )
    print(f"[tokens] source vocab {len(src_vocab)} "
          f"({total_src_tokens} occurrences), "
          f"stitch vocab {len(sti_vocab)} ({total_sti_tokens} occurrences)")
    print(f"[tokens] shared vocab: {len(shared_vocab)}  "
          f"only-source: {len(only_in_source)}  "
          f"only-stitch: {len(only_in_stitch)}")
    print(f"[tokens] vocab recall: {token_recall:.2f}%  "
          f"precision: {token_precision:.2f}%")
    print(f"[tokens] over-repeated in stitch (potential dup): "
          f"{len(over_repeated)}")

    # ---- Write artifacts ----
    (args.out / "source_lines.txt").write_text(
        "\n".join(source_lines_ordered), encoding="utf-8")
    (args.out / "stitch_lines.txt").write_text(
        "\n".join(f"{name}\t{line}" for name, line in stitch_lines_all),
        encoding="utf-8")
    (args.out / "missing.txt").write_text(
        "\n".join(
            f"[score={r['score']:3d}] SRC={r['source']!r}  "
            f"BEST_STITCH={r['best_stitch']!r}"
            for r in missing
        ),
        encoding="utf-8",
    )
    (args.out / "corrupt.txt").write_text(
        "\n".join(
            f"[score={r['score']:3d}] STITCH={r['stitch']!r}  "
            f"BEST_SRC={r['best_source']!r}"
            for r in corrupt
        ),
        encoding="utf-8",
    )
    (args.out / "duplicates.txt").write_text(
        "\n".join(f"x{r['count']:2d}: {r['line']}" for r in duplicated),
        encoding="utf-8",
    )
    (args.out / "tokens_only_source.txt").write_text(
        "\n".join(f"{src_tokens[t]:3d}  {t}" for t in only_in_source),
        encoding="utf-8",
    )
    (args.out / "tokens_only_stitch.txt").write_text(
        "\n".join(f"{sti_tokens[t]:3d}  {t}" for t in only_in_stitch),
        encoding="utf-8",
    )
    (args.out / "tokens_over_repeated.txt").write_text(
        "\n".join(
            f"src={r['source']:3d} stitch={r['stitch']:3d}  {r['token']}"
            for r in over_repeated
        ),
        encoding="utf-8",
    )

    src_n = max(1, len(source_lines_ordered))
    sti_n = max(1, len(stitch_lines_unique))
    report = {
        "source": {
            "keyframes_sampled": len(sampled),
            "unique_lines": len(source_lines_ordered),
        },
        "stitch": {
            "chunks": len(chunk_paths),
            "total_lines": len(stitch_lines_all),
            "unique_lines": len(stitch_lines_unique),
        },
        "params": {
            "min_line_len": args.min_line_len,
            "similarity_threshold": args.similarity,
            "min_dup_len": args.min_dup_len,
            "fuzzy_engine": "rapidfuzz" if USE_RF else "difflib",
        },
        "comparison": {
            "missing_from_stitch": len(missing),
            "no_source_match": len(corrupt),
            "duplicated_in_stitch": len(duplicated),
            "recall_pct": round(100.0 * (1.0 - len(missing) / src_n), 2),
            "precision_pct": round(100.0 * (1.0 - len(corrupt) / sti_n), 2),
        },
        "tokens": {
            "source_vocab": len(src_vocab),
            "stitch_vocab": len(sti_vocab),
            "source_occurrences": total_src_tokens,
            "stitch_occurrences": total_sti_tokens,
            "shared_vocab": len(shared_vocab),
            "only_in_source": len(only_in_source),
            "only_in_stitch": len(only_in_stitch),
            "vocab_recall_pct": round(token_recall, 2),
            "vocab_precision_pct": round(token_precision, 2),
            "over_repeated_count": len(over_repeated),
        },
    }
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("[summary]")
    print(f"  Source:    {len(source_lines_ordered):5d} unique lines "
          f"from {len(sampled)} keyframes")
    print(f"  Stitch:    {len(stitch_lines_all):5d} total / "
          f"{len(stitch_lines_unique)} unique lines "
          f"from {len(chunk_paths)} chunks")
    print(f"  Recall:    {report['comparison']['recall_pct']:6.2f}%  "
          f"(source-line found in stitch)")
    print(f"  Precision: {report['comparison']['precision_pct']:6.2f}%  "
          f"(stitch-line found in source)")
    print(f"  Duplicates: {len(duplicated):5d}  "
          f"(stitch lines repeated 2+ times)")
    print(f"  Token-level vocab recall:    {token_recall:6.2f}%")
    print(f"  Token-level vocab precision: {token_precision:6.2f}%")
    print(f"  Over-repeated tokens:        {len(over_repeated):5d}")
    print(f"  Artifacts in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
