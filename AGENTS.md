# vidsmash — Agent context

Concise operating context for any agent working in `C:\_\vidsmash`.
Style: STE100-like paragraphs and decision tables `[Option | Details | Tradeoffs]` with explicit **Recommendation**. No inline scripts (`python -c "..."`). All tools are Python under `tools/`.

## Goal

Convert a hand-scrolled vertical screen recording (iOS Messenger conversation) into a **continuous vertical image broken into ordered, non-overlapping PNG chunks of bounded height**. All content must be preserved. No overlap between chunks.

Sample input shipped with the repo: `lexiconv.mp4` (not committed — `.mp4` is ignored).

## Project layout

```
vidsmash/
  lexiconv.mp4              # sample input (gitignored)
  AGENTS.md                 # this file
  .gitignore
  bench_ffmpeg_pipes.py     # ffmpeg pipe perf bench (kept as a perf reference)
  tools/
    video_io.py             # ffmpeg pipe wrappers, Numba JIT row/col profiles,
                            #   match_1d_offset, ChunkWriter, static-band detection
    detect_pauses.py        # velocity-based pause detector → out/keyframes.json
    detect_overlays.py      # agreement-fraction overlay-mask detector
                            #   (iOS scrollbar / static gutter etc.)
    detect_overlay_circles.py # HoughCircles-based screen-fixed circular
                              #   overlay detector (iOS scroll-to-latest button)
                              #   discover_persistent_circles +
                              #   detect_circle_in_band
    detect_bubble_extents.py # cv2.Canny + morph-close per-row leftmost/rightmost
                             #   bubble-edge detection (library + PoC CLI)
    stitch_keyframes.py     # keyframe+bridge stitcher with multi-pass overlay-aware
                            #   canvas fill + extent-based gutter clear +
                            #   per-band detected-circle masks
                            #   → out/stitch/keyframe_chunk_NNN.png
    validate_stitch.py      # pair-overlap MAD + canvas placement MAD + coverage
    inspect_keyframe_chain.py
    inspect_brightness.py   # per-region pixel-colour histogram inspector (debug)
    crop_chunk.py           # PNG region cropper (debug)
    extract_frame.py        # single-frame ffmpeg extractor (debug)
    find_overlays.py        # single-frame colour-heuristic overlay sanity tool
    draw_circle.py          # annotate a frame with a known (cx,cy,r) (debug)
    smoke_circle_detector.py # smoke test for detect_overlay_circles
    diff_circle_regions.py  # side-by-side patch diff: baseline vs circle-masked stitch
    make_view_html.py       # emit view.html that vertically renders chunk PNGs
  ALTERNATIVE.md            # deferred OCR+HTML hybrid approach (with switch triggers)
  out/
    keyframes.json          # detect_pauses output (timeline + per-pause meta)
    timeline.png            # detect_pauses visualisation
    stitch/
      keyframe_chunk_NNN.png   # ordered, non-overlapping vertical chunks
      report.json              # placements, mask stats, gaps, params
      view.html                # generated viewer
      run.log
  debug_overlays/           # ad-hoc crops; gitignored
```

## Pipeline (one command per stage)

```
python tools/detect_pauses.py    --input lexiconv.mp4 --out out
python tools/stitch_keyframes.py --input lexiconv.mp4 --keyframes out/keyframes.json --out out/stitch
python tools/validate_stitch.py  --input lexiconv.mp4 --out out/stitch
python tools/make_view_html.py   --dir out/stitch --glob "keyframe_chunk_*.png" --title "vidsmash"
```

## Input invariants (per sample video `lexiconv.mp4`; new inputs may differ)

| Field | Value |
|---|---|
| Resolution | 1126 × 2436 |
| Codec / fps | HEVC / 59.955 |
| Frames | 3060 |
| Static top UI | rows 0..291 (status bar + chat header) |
| Dynamic band | rows 291..1260 (h = 969) |
| Static bottom UI | rows 1260..2436 (keyboard + composer) |
| Scroll direction | upward through the conversation (older messages above) |
| User recording technique | hand-scrolled; **pause at every unique state** |

User-supplied invariants that hold across inputs:

1. **Perfect-alignment invariant.** Every conversation line is present somewhere in the video with perfect pixel alignment for at least some subset of consecutive frames. Any shear/gap in the stitched output is a matcher error, not a source limitation.
2. **Pause-as-ground-truth invariant.** The user paused momentarily at every unique state. Frames inside pause runs are the most reliable keyframes (no motion blur, no partial scroll, no compression artefacts).
3. **Drag handling.** Horizontal drags happen (accidental, or to reveal SMS timestamp metadata). Drag-reveals-data is a future extraction step — stub only for now.

## Detect pauses (`tools/detect_pauses.py`)

Two-pass. Pass 1 (decode): for every frame build a K-segmented luma row profile and a column profile of the dynamic band; ffmpeg-side crop + gray and a 16 MB pipe buffer keep this near disk-read speed. Pass 2 (match): `match_1d_offset(prev, cur, predicted_p=0, search_radius=R)` per consecutive pair gives scroll velocity `dy` and best-fit MAD. A frame is a pause if `|dy| ≤ dy_thr AND mad ≤ mad_thr` for ≥ `min_pause_len` consecutive frames; small motion blips are coalesced. Each pause's midpoint becomes a keyframe.

Output: `out/keyframes.json` carries `dy_series`, `keyframes`, `pauses`, `between_runs`, `drag_suspect` flags, and video meta (`dyn_top`, `dyn_bot`, `dyn_h`, `w`). On `lexiconv.mp4`: 45 pause groups covering 584 frames (median pause length 6, max 175).

## Detect overlays (`tools/detect_overlays.py`)

iOS overlays (right-edge scrollbar, scroll-to-latest circular button) are **conditionally** present (the button only shows when not already at the bottom). A variance/range threshold across sample bands misses conditional overlays. Instead this module uses an **agreement-fraction** metric:

```
n_agree[y, x]   = count(|luma_band[i, y, x] − median_y_x| ≤ tol)
mask[y, x]      = n_agree[y, x] / N_bands ≥ frac
```

with defaults `frac=0.6`, `tol=12`. Connected components are dilated by 2 px and components smaller than `min_area=30` are dropped. Entry: `detect_overlay_mask_from_bands(bands, *, agreement_frac=0.6, agreement_tol=12, dilate=2, min_area=30) → (mask, report)`. Report `components[*].bbox` is `[x0, y0, x1, y1]`.

On `lexiconv.mp4`: 426 components, 42.8 % of one band masked; dominant components are the left gutter (x = 0..355) and the right-edge area (x = 856..1126) containing the scrollbar. The scroll-to-latest button is **not** reliably caught here (it is conditionally visible and centered far inside the dynamic band where agreement-fraction sometimes falls below threshold). It is handled separately by `detect_overlay_circles.py`.

## Detect overlay circles (`tools/detect_overlay_circles.py`)

The iOS scroll-to-latest button is a screen-fixed gray-filled circle (~120 px diameter) with a downward chevron. It is **opaque** (not semi-transparent), only shown when not at the bottom, and lands at the same screen position in every frame that contains it. Agreement-fraction (above) misses it intermittently because the dark chevron + thin outline produce sharp variance vs the disc fill, and because conditional visibility splits the prevalence below `frac=0.6` in some band sets. Result in the baseline pipeline: pass-1 averages partially blend the disc, pass-2 median over all covering bands picks the dark chevron/outline → visible dark half-disc "ghost" on the stitched canvas everywhere a band carrying the button was placed.

This module uses **OpenCV HoughCircles** in two stages:

1. **Discovery** — `discover_persistent_circles(bands, *, min_prevalence=0.4, r_min=20, r_max=100, bin_px=8, param1=100, param2=30, min_dist=60, dp=1.2, median_blur_ksize=5) → (list[CircleSpec], report)`. Runs HoughCircles on each band, bins detections by `(cx_bin, cy_bin)` with `bin_px=8`, dedupes one vote per band per bin, and promotes any bin appearing in `≥ min_prevalence` fraction of bands. Returns one `CircleSpec` per promoted bin (cx, cy, r, prevalence, r_min, r_max, n_detected, n_total).
2. **Per-band detection** — `detect_circle_in_band(band, *, expected_cx, expected_cy, expected_r, slack_xy=10, slack_r=4, pad=4, param1=100, param2=30, dp=1.2, median_blur_ksize=5, min_dist=60) → (mask, BandDetection)`. Crops a tight `(2*(r+slack_xy))²` ROI around the expected center, runs HoughCircles with `[r-slack_r, r+slack_r]` radius sweep, and returns the closest hit's filled disk mask (radius `detected_r + pad` to cover the AA outline). `~5-10 ms / band`.

On `lexiconv.mp4`: discovery finds 1 spec at `(cx=564, cy=876, r=60)` with 63.9 % prevalence; per-band detector hits 42 / 61 placed bands; total mask area ≈ 540 k px (summed across bands).

## Stitch keyframes (`tools/stitch_keyframes.py`)

1. **Placement.** Compute `cum_y[k] = Σ dy_series` per keyframe. Where consecutive keyframes overlap by less than `overlap_margin`, walk the inter-keyframe frames and add the minimum **bridge** frames so consecutive placements overlap by ≥ `overlap_margin` rows.
2. **Decode once.** Single ffmpeg RGB pipe with `crop=(W, dyn_h, 0, dyn_top)`; only target frames (keyframes ∪ bridges) are kept in memory.
3. **Auto overlay mask.** Sample N evenly-spaced bands → `detect_overlay_mask_from_bands` → seed `clean_mask` (per-band per-pixel "this candidate is trustworthy"). Manual `--mask-circle x y r` and `--mask-right-strip-from X` arguments UNION with the auto mask.
4. **4-pass canvas fill** (motivated by the observation that average canvas coverage is ~2 bands per pixel; the median-of-bands trick that suppresses overlays only works where coverage > 2):

   | Pass | Pixels handled | Strategy | Why |
   |---|---|---|---|
   | 1 | `clean_count ≥ 1` | mean of all CLEAN candidates per canvas pixel | Trust-weighted: ignores known-overlay positions entirely |
   | 2 | `clean_count == 0 ∧ total_count ≥ 1` | median across ALL covering bands | Recovers uniform gutter colour where every band reported the position as "overlay" but the underlying pixel really is just background |
   | 3 | columns where > 90 % of in-band rows are masked AND contiguous run ≤ `--overlay-max-strip-width` (default 50) | horizontal inpaint from nearest non-dense column | Removes thin vertical scrollbars where pass 2 still produced the overlay colour. Wide runs (gutter) are left alone — they are correctly handled by pass 2. |
   | 4 | `total_count == 0` | `scipy.ndimage.distance_transform_edt` spatial inpaint from CLEAN pixels | Edge cases at canvas top/bottom where no band covers |

5. **Chunk write.** Sort placed frames by `cum_y`, walk in order, cursor-model write into `ChunkWriter` (first-write-wins). `--ui keep-once` prepends `static_top` to the first chunk and appends `static_bot` to the last; `--ui strip` omits UI bands.

On `lexiconv.mp4`: 45 keyframes + 16 bridges = 61 placements, 8 chunks of 4096 px (last 2102), canvas 29 307 px content, 30 774 rows written, ~36 s end-to-end (decode dominates).

**Known artefact.** The scroll-to-latest circular button can still leave one faint ghost on canvas rows where coverage = 1 (only one band covers the pixel, and that band has the overlay at this in-band position). The median-of-bands trick degenerates to identity at k = 1. Mitigations to revisit if it matters for downstream OCR: lower `--overlay-agreement-frac` (more aggressive masking), or extend pass 3 to also handle small isolated bbox regions (not just narrow vertical strips).

## Tooling discipline

- All tools are Python. Project venv on PATH; no managed environments.
- ffmpeg is resolved via `_resolve_ffmpeg()` in `tools/video_io.py` (Winget install at `C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_...`).
- **Never** inline tool logic on the command line (`python -c "..."`, `bash -c`, etc.). Place scripts under `tools/`. Ad-hoc verification commands against produced data (e.g., `jq`, `Get-Content`, `Select-Object`) are fine.
- ffmpeg pipe pattern: `open_rgb_pipe` → `read_frame(proc, frame_bytes)` loop until `None` → `close_proc(proc)`. `stderr=DEVNULL` to avoid OS-pipe-buffer deadlock on long runs.
- ffmpeg `select='eq(n,N-1)'` for the last frame is unreliable (off-by-two decode quirk). Capture the last-frame static bottom **during the main pass** instead.
- Canvas coordinates are **absolute** (anchored at the dynamic band of frame 0; can be negative). Final stitched order is `static_top + canvas[min_top..max_bot] + static_bot` so `chunk_000` = oldest content, last chunk = newest (where the user started).

## Performance — `detect_pauses.py`

Profiled on AMD Ryzen 9 7845HX, NVIDIA RTX 4070 Laptop, NVMe SSD against `lexiconv.mp4` (1126 × 2436 HEVC, 3060 frames). Warm-cache.

| Stage | Baseline | +A1 buffered pipe | +A2 ffmpeg-side crop+gray | +A3 Numba JIT |
|---|---|---|---|---|
| pass1 decode | 57.06 s | 22.74 s | 9.91 s | 9.96 s |
| pass1 profile | 10.34 s | 12.60 s | 10.57 s | **6.40 s** |
| pass2 match (3059 pairs) | 28.47 s | 23.11 s | 23.32 s | **8.40 s** |
| **analysis total** | **95.87 s** | 58.45 s | 43.80 s | **24.76 s** |
| ms / pair (pass2) | 9.31 | 7.55 | 7.62 | **2.75** |

End-to-end **3.87 ×**. Output identical across all four runs (same 45 pause groups, same dy & MAD stats).

### Decisions in order they were measured

**A1 — Buffered ffmpeg pipe** (`bufsize = 16 MB` in `open_rgb_pipe`). Was `bufsize=0` (per-syscall round-trips). 2.5 × pass1-decode speedup with zero behaviour change.

**A2 — ffmpeg-side `crop=W:dyn_h:0:top` + `pix_fmt=gray`** (`crop` param on `open_rgb_pipe`). Drops static top + bottom UI bands before the pipe, cutting bandwidth 2.5 ×. Bench against three alternatives (`bench_ffmpeg_pipes.py`):

| # | Pipe | Wall | MB read |
|---|---|---|---|
| 1 | CPU full-frame gray | 19.91 s | 8005 MB |
| 2 | NVDEC full-frame gray | 20.64 s | 8005 MB |
| 3 | NVDEC + crop filter + gray | 13.92 s | 8004 MB (crop silently no-op'd in this ffmpeg build) |
| 4 | **CPU crop + gray** | **6.41 s** | **3184 MB** |

CPU crop wins because ffmpeg's `crop` filter is highly optimised C, the NVDEC path still has to `hwdownload` to system RAM, and the crop dropped 60 % of pipe bytes.

**A3 — Numba `@njit`** on `_match_1d_core`, `_gray_row_profile_jit`, `_gray_col_profile_jit`. Numba fuses the abs-diff + accumulator into a tight scalar loop, removing the per-offset numpy dispatch overhead that dominated pass 2. Pass 2: 23.3 → 8.4 s (2.78 ×). Pass 1 profile: 10.6 → 6.4 s (1.65 ×). `numba 0.65.1` + `llvmlite 0.47.0` already installed; first call has ~1 s compile time, then `cache=True` reuses a pyc-adjacent cache.

### Rejected (do not repeat without measurement)

- **Vectorized `match_1d_offset` (NaN-pad + `sliding_window_view`).** Identical results but **1.7 × slower**. At R = 400, h = 969, K = 16 each call allocates ~50 MB and blows L2/L3 cache; the scalar loop's per-iter ~15 KB slice fits in L1. The dropped regression test (`test_match_1d_offset.py`) covered both implementations — re-add it before retrying.
- **NVDEC for HEVC decode on this video.** 20.6 s vs 19.9 s CPU. CUDA init + mandatory `hwdownload` exceed NVDEC's gain on a single-stream short video. Triggers to re-test: input ≥ 4K, input ≥ 10 min, or running ≥ N videos in parallel.
- **`scale_cuda` / GPU crop filter.** Bench option 3 produced the same byte count as full-frame, indicating the crop filter silently did not apply with `-hwaccel_output_format cuda`. Not worth chasing while CPU crop wins 3.1 ×.

### Future perf to revisit when justified

| Option | When | Expected win |
|---|---|---|
| Multi-process pool across videos | Batch input | Linear with cores (per-video runtime ≤ 30 s now) |
| CuPy / PyTorch batched pass 1 on the 4070 | Pass 1 still dominates after Numba on some future input | Big only if batched-across-frames; ~700 MB install |
| NVDEC + working GPU crop | Input ≥ 4K or large batch | Re-bench with `bench_ffmpeg_pipes.py` first |

Hardware decode (`hwaccel="cuda"`) and ffmpeg-side crop are both opt-in params on `open_rgb_pipe`. Crop is default in `detect_pauses.py`; hwaccel stays opt-in until a measured input proves a win.

## Conventions

- Decision-making in agent responses: short STE100 paragraphs or `[Option | Details | Tradeoffs]` tables with an explicit **Recommendation**.
- Chunks named `keyframe_chunk_NNN.png`. Reports in `out/stitch/report.json`. Pause/keyframe data in `out/keyframes.json`. Timeline in `out/timeline.png`.
- `.gitignore` excludes `*.mp4` and PNG outputs; commit the JSON/HTML/PY only. Outputs are regeneratable.

## Known issues / open work

- **Faint scroll-to-latest button ghost** on coverage-1 canvas rows (see Stitch §). Acceptable for current outputs; revisit if it bites OCR.
- **Drag-reveals-data extraction.** Stub only. `detect_pauses.py` flags drag-suspect pauses via column-profile MAD between adjacent pauses (29/45 on `lexiconv.mp4` — threshold may be too tight).
- **Decode-count vs ffprobe-report mismatch** is non-critical (`[warn] decoded N+2 frames but ffprobe reported N`).
