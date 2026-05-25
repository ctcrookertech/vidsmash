"""First-pass diagnostic: detect pause regions and extract key frames.

Premise (user-supplied): during recording, the user paused momentarily at
every unique state — even when those states overlapped previous or subsequent
content. Therefore the video has clear "no-change" runs that bracket each
unique view of the conversation. Frames inside those runs are the most
reliable representations of the underlying content (no motion blur, no
partial scroll, no compression artifacts from rapid motion).

Detection strategy
------------------
1. Hash pass (cheap): for every frame, hash the dynamic-band pixel bytes
   (blake2b digest, 8 bytes). Consecutive frames with identical hashes are
   GUARANTEED to be pixel-identical -> exact pause.
2. Coalesce pass (optional, on by default): for short transition gaps
   between exact pause groups, compute MAD-to-neighbors; merge frames into
   the adjacent pause group when MAD < --coalesce-mad-threshold. This
   recovers pauses where HEVC compression introduced a sub-luma-unit delta
   between otherwise-identical frames.

This tool performs at most TWO passes (hash always, coalesce only when
ambiguity exists) and produces:

  out/keyframes.json
    {
      "video": { w, h, dyn_top, dyn_bot, dyn_h, frames_decoded, fps },
      "params": { ... },
      "pauses": [
        { "start": i0, "end": i1, "length": n, "mid": iMid,
          "hash": "<hex>", "coalesced_count": <int> },
        ...
      ],
      "keyframes": [
        { "i": iMid, "pause_index": k, "pause_length": n, "hash": "<hex>" },
        ...
      ],
      "between_runs": [
        { "from_pause": k, "to_pause": k+1, "gap_frames": g }, ...
      ],
      "hash_summary": {
        "n_frames": <int>,
        "n_unique_hashes": <int>,
        "exact_pause_groups": <int>,
        "frames_in_exact_pauses": <int>
      }
    }

  out/keyframes/kf_NNN_f<idx>.png  (only with --save-frames)

Run:
  python tools/detect_pauses.py --input lexiconv.mp4 --out out
  python tools/detect_pauses.py --input lexiconv.mp4 --out out --save-frames
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_scroll import (  # type: ignore  # noqa: E402
    _resolve_ffmpeg,
    close_proc,
    detect_static_bands,
    open_rgb_pipe,
    probe_video,
    read_frame,
)


def hash_bytes(b: bytes) -> bytes:
    return hashlib.blake2b(b, digest_size=8).digest()


def find_equal_runs(hashes: list[bytes], min_len: int) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) runs of consecutive equal hashes >= min_len."""
    runs: list[tuple[int, int]] = []
    n = len(hashes)
    if n == 0:
        return runs
    i = 0
    while i < n:
        j = i + 1
        while j < n and hashes[j] == hashes[i]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-pause-len", type=int, default=2,
        help="Minimum consecutive identical frames to register as a pause",
    )
    ap.add_argument(
        "--coalesce-mad-threshold", type=float, default=0.3,
        help="If >0, merge inter-pause frames whose MAD to the nearest pause "
             "frame is below this threshold (set 0 to disable coalesce pass).",
    )
    ap.add_argument(
        "--save-frames", action="store_true",
        help="Write each selected key frame as PNG under out/keyframes/",
    )
    # Static-band detection (reused from stitcher).
    ap.add_argument("--std-threshold", type=float, default=2.0)
    ap.add_argument("--min-static-run", type=int, default=30)
    ap.add_argument("--dynamic-top", type=int, default=-1)
    ap.add_argument("--dynamic-bottom", type=int, default=-1)
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg, ffprobe = _resolve_ffmpeg()
    vinfo = probe_video(ffprobe, args.input)
    W, H = vinfo["width"], vinfo["height"]
    print(f"[probe] {W}x{H}, {vinfo['nb_frames']} frames, {vinfo['fps']:.3f} fps")

    if args.dynamic_top >= 0 and args.dynamic_bottom >= 0:
        top, bot = args.dynamic_top, args.dynamic_bottom
        print(f"[ui] dynamic band overridden: {top}..{bot}")
    else:
        print("[ui] detecting static bands...")
        top, bot = detect_static_bands(
            ffmpeg=ffmpeg,
            path=args.input,
            W=W, H=H,
            n_samples=32,
            std_threshold=args.std_threshold,
            min_run=args.min_static_run,
        )
        print(f"[ui] static_top=0..{top}  dynamic={top}..{bot}  static_bot={bot}..{H}")
    dyn_h = bot - top
    fbytes = W * H * 3

    # ----- Pass 1: hash every frame's dyn band -----
    proc = open_rgb_pipe(ffmpeg, args.input)
    hashes: list[bytes] = []
    # Keep a small cache of representative dyn-band bytes by hash for the
    # coalesce pass (avoid a second video pass for MAD comparisons).
    # Memory: one dyn-band per unique hash. Worst case = all frames unique
    # (~10 GB at 1126x969x3) — but in practice the user paused frequently,
    # so unique hash count should be modest.
    rep_dyn: dict[bytes, np.ndarray] = {}
    n_frames = 0
    try:
        while True:
            buf = read_frame(proc, fbytes)
            if buf is None:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            dyn = frame[top:bot]
            db = dyn.tobytes()
            h = hash_bytes(db)
            hashes.append(h)
            # Keep one rep per unique hash for coalesce step.
            if h not in rep_dyn:
                rep_dyn[h] = dyn.copy()
            n_frames += 1
            if n_frames % 200 == 0:
                print(f"[pass1] frame {n_frames}  unique_hashes={len(rep_dyn)}")
    finally:
        close_proc(proc)

    print(f"[pass1] decoded {n_frames} frames  unique_hashes={len(rep_dyn)}")

    # ----- Find exact-hash pause runs -----
    exact_runs = find_equal_runs(hashes, args.min_pause_len)
    exact_frames_in_pause = sum(e - s for s, e in exact_runs)
    print(
        f"[pause] exact-hash pauses: {len(exact_runs)} runs covering "
        f"{exact_frames_in_pause} frames"
    )

    # ----- Coalesce pass: merge near-identical isolated frames into adjacent pauses -----
    coalesced_counts: list[int] = [e - s for s, e in exact_runs]
    if args.coalesce_mad_threshold > 0 and exact_runs:
        # Walk between consecutive exact pauses; for each inter-pause frame,
        # compare to the LAST frame of the preceding pause and the FIRST
        # frame of the following pause. If either MAD < threshold, attach
        # the frame to that pause.
        # Implementation: we walk the original frame index order, growing
        # pause boundaries as we go.
        new_runs: list[list[int]] = [[s, e] for s, e in exact_runs]
        for k in range(len(new_runs)):
            s, e = new_runs[k]
            # Extend forward: pull in following frames close to last frame of this pause
            last_dyn = rep_dyn[hashes[e - 1]].astype(np.float32)
            j = e
            stop = new_runs[k + 1][0] if k + 1 < len(new_runs) else n_frames
            while j < stop:
                cand_dyn = rep_dyn[hashes[j]].astype(np.float32)
                mad = float(np.abs(cand_dyn - last_dyn).mean())
                if mad >= args.coalesce_mad_threshold:
                    break
                j += 1
            new_runs[k][1] = j
        # Extend backward: pull in preceding frames close to first frame of this pause
        for k in range(len(new_runs)):
            s, e = new_runs[k]
            first_dyn = rep_dyn[hashes[s]].astype(np.float32)
            stop = new_runs[k - 1][1] if k > 0 else 0
            j = s - 1
            while j >= stop:
                cand_dyn = rep_dyn[hashes[j]].astype(np.float32)
                mad = float(np.abs(cand_dyn - first_dyn).mean())
                if mad >= args.coalesce_mad_threshold:
                    break
                j -= 1
            new_runs[k][0] = j + 1
        # Re-sort & merge any overlapping ranges (shouldn't usually occur)
        new_runs.sort()
        merged: list[list[int]] = []
        for r in new_runs:
            if merged and r[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], r[1])
            else:
                merged.append(r[:])
        # Recompute coalesced_counts as the count of frames absorbed beyond
        # the original exact-pause length (informational only).
        coalesced_counts = []
        # Map merged back to exact (k-th merged corresponds to one or more exact runs)
        # For simplicity, just report total merged size per group.
        for r in merged:
            coalesced_counts.append(r[1] - r[0])
        final_runs = [(s, e) for s, e in merged]
    else:
        final_runs = exact_runs

    final_frames_in_pause = sum(e - s for s, e in final_runs)
    print(
        f"[coalesce] final pauses: {len(final_runs)} runs covering "
        f"{final_frames_in_pause} frames "
        f"(+{final_frames_in_pause - exact_frames_in_pause} via coalesce)"
    )

    # ----- Build pauses, keyframes, between_runs -----
    pauses = []
    keyframes = []
    for k, (s, e) in enumerate(final_runs):
        length = e - s
        mid = s + length // 2
        # Use hash of original midpoint frame
        hh = hashes[mid].hex()
        pauses.append({
            "start": int(s),
            "end": int(e),
            "length": int(length),
            "mid": int(mid),
            "hash": hh,
            "coalesced_count": int(length),
        })
        keyframes.append({
            "i": int(mid),
            "pause_index": k,
            "pause_length": int(length),
            "hash": hh,
        })

    between = []
    for k in range(len(pauses) - 1):
        a_end = pauses[k]["end"]
        b_start = pauses[k + 1]["start"]
        gap = b_start - a_end
        if gap > 0:
            between.append({
                "from_pause": k,
                "to_pause": k + 1,
                "gap_frames": int(gap),
            })

    result = {
        "video": {
            "w": W, "h": H,
            "dyn_top": int(top), "dyn_bot": int(bot), "dyn_h": int(dyn_h),
            "frames_decoded": int(n_frames),
            "fps": vinfo["fps"],
        },
        "params": {
            "min_pause_len": args.min_pause_len,
            "coalesce_mad_threshold": args.coalesce_mad_threshold,
        },
        "pauses": pauses,
        "keyframes": keyframes,
        "between_runs": between,
        "hash_summary": {
            "n_frames": n_frames,
            "n_unique_hashes": len(rep_dyn),
            "exact_pause_groups": len(exact_runs),
            "frames_in_exact_pauses": exact_frames_in_pause,
            "final_pause_groups": len(final_runs),
            "frames_in_final_pauses": final_frames_in_pause,
        },
    }

    (out_dir / "keyframes.json").write_text(json.dumps(result, indent=2))
    print(f"[write] {out_dir / 'keyframes.json'}")

    if args.save_frames and keyframes:
        kf_dir = out_dir / "keyframes"
        kf_dir.mkdir(exist_ok=True)
        wanted = {kf["i"]: kf["pause_index"] for kf in keyframes}
        proc = open_rgb_pipe(ffmpeg, args.input)
        i = 0
        saved = 0
        try:
            while True:
                buf = read_frame(proc, fbytes)
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

    # Brief summary
    print()
    print("=== Pause / keyframe summary ===")
    print(f"frames_decoded         : {n_frames}")
    print(f"unique_hashes          : {len(rep_dyn)}")
    print(f"exact pause groups     : {len(exact_runs)}  ({exact_frames_in_pause} frames)")
    print(f"final pause groups     : {len(final_runs)}  ({final_frames_in_pause} frames)")
    if pauses:
        ll = [p["length"] for p in pauses]
        print(f"  pause_lengths: min={min(ll)} median={int(np.median(ll))} mean={np.mean(ll):.1f} max={max(ll)}")
    print(f"transitions (gaps)     : {len(between)}")
    if between:
        gap_lengths = [b["gap_frames"] for b in between]
        print(f"  gap_frames: min={min(gap_lengths)} median={int(np.median(gap_lengths))} max={max(gap_lengths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
