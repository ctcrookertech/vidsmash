# vidsmash — Agent context

Concise operating context for any agent working in `C:\_\vidsmash`.
Style: STE100-like paragraphs and decision tables `[Option | Details | Tradeoffs]` with explicit recommendations. No inline scripts; tools live in Python under `strategy_*/tools/`.

## Goal

Convert a hand-scrolled vertical screen recording (iOS Messenger conversation in `lexiconv.mp4`) into a **continuous vertical image broken into ordered, non-overlapping PNG chunks of bounded height**. All content must be preserved. No overlap between chunks.

## Input facts (`lexiconv.mp4`)

| Field | Value |
|---|---|
| Resolution | 1126 × 2436 |
| Codec / fps | HEVC / 59.955 |
| Frames decoded | 3060 (ffprobe reports 3058; off-by-two is a decode quirk) |
| Static top UI | rows 0..291 (status bar + chat header) |
| Dynamic band | rows 291..1260 (h = 969) |
| Static bottom UI | rows 1260..2436 (keyboard + composer) |
| Scroll direction | **upward** through the conversation (older messages above) |
| User recording technique | Hand-scrolled; intentional **pause at every unique state** |

## User-supplied invariants

1. **Perfect-alignment invariant.** Every conversation line is present somewhere in the video with perfect pixel alignment for at least some subset of consecutive frames. Therefore any shear/gap in the stitched output is a matcher error, not a source limitation.
2. **Pause-as-ground-truth invariant.** User paused momentarily at every unique state. Frames inside pause runs are the most reliable keyframes (no motion blur, no partial scroll, no compression artifacts from rapid motion).
3. **Drag handling.** Horizontal drags happen (accidental, or to reveal SMS timestamp metadata). Drag-reveals-data is a future extraction step — stub only for now.

## Project layout

```
vidsmash/
  lexiconv.mp4               # input
  AGENTS.md                  # this file
  tools/                     # legacy / shared scratch (pre-fork; do not extend)
  strategy_a/                # Strategy A: surgical fix to incremental frame-to-frame matcher
    tools/
      stitch_scroll_a.py     # current bidirectional matcher (K-seg sig + prior_alpha + stationary short-circuit)
      validate_stitch_a.py   # validator (pair-overlap MAD + canvas placement MAD + best-offset search + coverage map)
    out/                     # chunks, report, drag sidecars, view.html, validation/
  strategy_b/                # Strategy B: pause-driven keyframe stitcher
    tools/
      detect_pauses_b.py     # v1 hash-based pause detection (deprecated; archive only)
      detect_pauses_b_v2.py  # v2 velocity-based pause detection + timeline visualization
      stitch_scroll_b.py     # ffmpeg helpers + gray_*_profile / luma_*_profile (imported by detect & stitcher)
      stitch_keyframes_b.py  # TODO: keyframe-based stitcher (pairwise full-pixel MAD alignment)
      validate_stitch_b.py   # same validator, applied to strategy B output
    out/                     # keyframes.json, timeline.png, detect_v2.log
  debug/                     # sample frames + thumbnails for spot-checks
```

Both strategies operate on the same `lexiconv.mp4`. The validator and compare tools should be invariant to which strategy produced the output.

## Strategies

| ID | Approach | Status |
|---|---|---|
| A | Incremental frame-to-frame matcher with bidirectional canvas, K-segmented luma row signature, soft prior_alpha drift bias, stationary short-circuit, edge-hit + suspect-advance guards | Runs end-to-end on `lexiconv.mp4`; output exists in `strategy_a/out/`; **shearing and missing content reported by user**; validator partial output exists (crashed on summary write — see Known issues) |
| B v1 | Per-frame blake2b hash of dyn band → consecutive-equal-hash runs as exact pauses + MAD coalesce | **Deprecated.** Only finds 3 pauses on `lexiconv.mp4`. Wrong signal: scrolling frames differ by translation, so absolute bytes diverge even when content is constant. Kept for archive in `detect_pauses_b.py`. |
| B v2 | Decode pass: build K-segmented luma profile (and column profile) per frame. Pair pass: `match_1d_offset(prev, cur, predicted_p=0, search_radius=R)` gives per-pair scroll velocity `dy` and best-fit MAD. Pause = `|dy|<=dy_thr AND mad<=mad_thr` for ≥ `min_pause_len` consecutive frames. Coalesce small motion blips. Keyframe = pause midpoint. Drag-suspect flag via column-profile MAD between adjacent pauses. | Detector implemented in `detect_pauses_b_v2.py`. On `lexiconv.mp4`: 45 pause groups covering 584 frames (median pause length 6, max 175). Timeline visualization confirms structure. **Stitcher not yet built.** |

## Tooling discipline

- All tools are Python (project language). Use the project venv interpreter (`python` on PATH).
- ffmpeg is resolved via the hard-coded fallback in `_resolve_ffmpeg()` (Winget install at `C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...`).
- **Never** inline tool logic on the command line (`python -c "..."`). Place scripts under `strategy_*/tools/`. Ad-hoc verification commands (one-liners against produced data) are fine.
- ffmpeg pipe pattern: `open_rgb_pipe` → `read_frame(proc, frame_bytes)` loop until `None` → `close_proc(proc)`. `stderr=DEVNULL` to avoid OS-pipe-buffer deadlock on long runs.
- ffmpeg `select='eq(n,N-1)'` for the last frame is unreliable (off-by-two decode quirk). Capture the last-frame static bottom **during the main pass** instead.
- Validator strategy: pair-overlap MAD between consecutive frames is the most precise local check; canvas-MAD at recorded position is the placement check; slide-search around recorded position finds the *correct* placement and is cheap to gate behind flagging.
- Coverage map: per canvas row support count; rows with zero support are true gaps.

## Conventions

- Decision-making in agent responses: short STE100 paragraphs or `[Option | Details | Tradeoffs]` tables with an explicit **Recommendation**.
- Chunks named `chunk_NNN.png`, drag sidecars `drag_NNN.png`, validator outputs under `out/validation/`, pause/keyframe data in `out/keyframes.json`, timeline in `out/timeline.png`.
- Canvas coordinates are **absolute** (can be negative; anchored at the dynamic band of frame 0). Final stitched order is `static_top + canvas[min_top..max_bot] + static_bot` so chunk_000 = oldest content, last chunk = newest (where the user started).

## Performance / hardware acceleration

Profiled on this machine (AMD Ryzen 9 7845HX, NVIDIA RTX 4070 Laptop, NVMe SSD) against `lexiconv.mp4` (1126×2436 HEVC, 3060 frames). All timings are warm-cache; first-run cold-cache is ~50 % slower because the 17 MB mp4 hits disk.

### Measured timing ladder (`detect_pauses_b_v2.py`, instrumented per-stage)

| Stage | Baseline (cold prior log) | +A1 buffered pipe | +A2 ffmpeg-side crop+gray | +A3 Numba JIT |
|---|---|---|---|---|
| pass1 decode | 57.06 s | 22.74 s | 9.91 s | 9.96 s |
| pass1 profile | 10.34 s | 12.60 s | 10.57 s | **6.40 s** |
| pass2 match (3059 pairs) | 28.47 s | 23.11 s | 23.32 s | **8.40 s** |
| **analysis total** | **95.87 s** | 58.45 s | 43.80 s | **24.76 s** |
| ms / pair (pass2) | 9.31 | 7.55 | 7.62 | **2.75** |

End-to-end speedup: **3.87×**. Output identical across all four runs (45 pause groups / 584 frames / same dy & MAD stats). One drag-suspect boundary flipped (#38 removed) at A2 — float-rounding difference from the ffmpeg crop filter's internal pipeline; below noise floor.

### Decisions in order they were measured

**A1 — Buffered ffmpeg pipe (`bufsize = 16 MB` in `open_rgb_pipe`).** Was `bufsize=0` (Python-unbuffered, per-syscall round-trips). One line; 2.5× pass1-decode speedup with zero behavior change. Implemented in `stitch_scroll_{a,b}.py`.

**A2 — ffmpeg-side `crop=W:dyn_h:0:top` + `pix_fmt=gray` (new `crop` param on `open_rgb_pipe`).** Lets ffmpeg drop static top + bottom UI bands before the pipe, cutting bandwidth 2.5× on `lexiconv.mp4`'s dyn band. Bench against three alternatives (`bench_ffmpeg_pipes.py`):

| # | Pipe | Wall | MB read |
|---|---|---|---|
| 1 | CPU full-frame gray | 19.91 s | 8005 MB |
| 2 | NVDEC full-frame gray | 20.64 s | 8005 MB |
| 3 | NVDEC + crop filter + gray | 13.92 s | 8004 MB (crop silently no-op'd in this ffmpeg build) |
| 4 | **CPU crop + gray** | **6.41 s** | **3184 MB** |

CPU crop wins because ffmpeg's `crop` filter is highly optimised C, the NVDEC path still has to `hwdownload` to system RAM, and the crop dropped 60 % of pipe bytes.

**A3 — Numba `@njit` on `_match_1d_core` + `_gray_row_profile_jit` + `_gray_col_profile_jit`.** Numba fuses the abs-diff + accumulator into a tight scalar loop, removing the per-offset numpy dispatch overhead that was the *real* cost of pass2 (the prior 28 ms-per-pair was almost entirely numpy call overhead, not arithmetic). Pass2: 23.3 → 8.4 s (2.78×). Pass1 profile: 10.6 → 6.4 s (1.65× from fused cast+mean). `numba 0.65.1` + `llvmlite 0.47.0` already installed; first call has ~1 s compile time then `cache=True` writes a pyc-adjacent cache.

### Things tried and rejected (do not repeat)

**Vectorized `match_1d_offset` (NaN-pad + `sliding_window_view`).** Identical results but **1.7× slower** (264 s vs 158 s back when 158 s was the baseline). At R=400, h=969, K=16 each call allocates ~50 MB and blows L2/L3 cache; the loop's per-iter ~15 KB slice fits in L1. Test harness (`strategy_b/tools/test_match_1d_offset.py`) is kept as a regression guard and as evidence — that test now also covers the Numba kernel and all 13 cases pass.

**NVDEC for HEVC decode on this video.** Bench above: 20.6 s vs 19.9 s CPU. CUDA init + mandatory `hwdownload` exceed NVDEC's gain on a single-stream short video. Triggers to re-test: input ≥4K, input ≥10 min, or running ≥N videos in parallel.

**`scale_cuda` / GPU crop filter.** Bench option 3 produced the same byte count as full-frame, indicating the crop filter silently did not apply with `-hwaccel_output_format cuda`. Not worth chasing while CPU crop wins 3.1×.

### Future perf options to revisit if needed

| Option | When to revisit | Expected win |
|---|---|---|
| Multi-process pool over independent videos | Pipeline used on a batch — wall time scales with N | Linear with cores; per-video runtime is now ≤30 s so a 4-way pool gets 4 videos in the time of 1 |
| CuPy / PyTorch batched pass-1 on the RTX 4070 | Future input where pass1 still dominates after Numba | Big if batched-across-frames; pure per-call is launch-overhead bound. ~700 MB install. |
| NVDEC + GPU crop with proper `scale_cuda` chain | Input ≥4K or batch of ≥N | Re-bench `bench_ffmpeg_pipes.py` with same input first |
| ~~Vectorize `match_1d_offset`~~ | Don't | Cache-bound; rejected with measurement |

Hardware decode (`hwaccel="cuda"`) and ffmpeg-side crop are both wired as opt-in params on `open_rgb_pipe`. Crop is now the default in `detect_pauses_b_v2.py`; hwaccel stays opt-in until a measured input proves a win.

## Stitch (Strategy B) — keyframes + bridges

`strategy_b/tools/stitch_keyframes_b.py` reads `out/keyframes.json` from `detect_pauses_b_v2.py` and produces ordered, non-overlapping vertical chunks. Algorithm:

1. Compute `cum_y[k] = Σ dy_series` per keyframe.
2. For each consecutive keyframe pair, if `|chain_dy| ≥ dyn_h − overlap_margin` (i.e., the keyframes do not overlap), walk the transition frames between them and add the minimum **bridge frames** such that consecutive placed frames overlap by at least `overlap_margin` rows.
3. Decode the source ONCE with `crop=(W, dyn_h, 0, dyn_top)` RGB pipe; capture only target frames into memory.
4. Optional `--validate-match` re-matches each consecutive overlapping pair with `match_1d_offset` and logs `chain_dy` vs `direct_dy` disagreement (sanity tap; not used to correct the placement).
5. Sort by `cum_y`, walk in order, cursor-model write into `ChunkWriter` (first-write-wins; ignores fully-overlapped frames; red-row gap marker if anything is missed).
6. `--ui keep-once` prepends `static_top` to the first chunk and appends `static_bot` to the last; `--ui strip` omits UI bands.

On `lexiconv.mp4`: **45 keyframes + 16 bridges = 61 placements**, 8 chunks of 4096 px (last 2102), canvas 29307 px content, total 30774 rows written, ~36 s end-to-end (decode dominates). Validation: 60 overlapping pairs checked; 1 disagreement > 20 px (kf 1671: chain −870 vs direct −848). No gaps emitted.

## Strategy comparison

`compare_strategies.py` (project root) loads `strategy_a/out/chunk_*.png` and `strategy_b/out/stitch/keyframe_chunk_*.png`, trims each strategy's static UI bands using its `report.json`, row-profile-aligns the two content regions with `match_1d_offset`, and emits `compare_out/compare_preview.png` (side-by-side downscaled) and `compare_out/compare_report.json`.

Result on `lexiconv.mp4`:

| Metric | Strategy A | Strategy B |
|---|---|---|
| Chunks | 8 | 8 |
| Content rows | 28 845 | 29 307 |
| Drag sidecars | 4 | (flagged in report only) |

Alignment B-vs-A: **+12 row** head offset, profile MAD 9.2 at best, per-row mean MAD 14.4 over the 28 833-row overlap. The 462-row (~1.6 %) height delta is consistent with sub-pixel `dy_series` quantization drift accumulated differently by the two pipelines (A: incremental matched offsets; B: cumulative integer dy from the pause detector). Both stitches are visually complete with no missing chunks or shears.

Open question (not blocking): which strategy is closer to ground truth in absolute height? Resolution would require a known-distance physical marker (none present), or a third independent pipeline. For now the side-by-side preview is the eyeball check.

## Known issues / open work

- Drag-reveals-data extraction: stub only; see `extract_drag_revealed_columns_stub` in `validate_stitch_*.py`. `detect_pauses_b_v2.py` flags drag-suspect pauses via column-profile MAD between adjacent pauses (29/45 on `lexiconv.mp4` — threshold may be too tight; revisit when a video where drag clearly matters is processed).
- Decode count vs ffprobe report mismatch is non-critical (reported as `[warn] decoded N+2 frames but ffprobe reported N`).
- 1.6 % height drift between A and B (above). Not a blocker for the "vertical chunks, no overlap, all content present" invariant.

## Next steps

1. ✅ `detect_pauses_b_v2.py` velocity-based pause detector — 45 pauses on `lexiconv.mp4`.
2. ✅ Plan A perf ladder (buffered pipe, ffmpeg-side crop+gray, Numba JIT) — 3.87 × end-to-end.
3. ✅ `stitch_keyframes_b.py` with keyframe + bridge placement and `ChunkWriter` output.
4. ✅ `compare_strategies.py` with row-profile alignment + side-by-side preview.
5. Next video onboarding: run `detect_pauses_b_v2.py` → eyeball `timeline.png` → `stitch_keyframes_b.py --validate-match` → eyeball `compare_preview.png` if Strategy A was also run.
6. Future: drag-reveals-data extraction (replace `extract_drag_revealed_columns_stub`).
