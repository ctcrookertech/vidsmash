# ALTERNATIVE — Hybrid OCR + HTML reconstruction

Status: **deferred, not rejected.** Captured here so we can revisit on demand
without re-deriving the analysis.

## TL;DR

Rebuild the conversation as **HTML** from OCR + per-frame classification,
sandwiched between the **real header/footer images** we already extract. The
body becomes structured DOM (bubbles, timestamps, attachments) instead of a
stitched bitmap. This sidesteps every artifact class we are currently chasing
(scrollbar leaks, button ghosts, pass-2 gutter fabrication, sub-pixel scroll
seams) because the body is no longer a pixel composite.

The earlier off-hand framing of this as "v2 / much larger effort" was
overstated. The honest cost is comparable to finishing the bitmap stitcher
robustly, with a strictly better steady-state output.

## What stays vs what changes

| Component | Today (bitmap stitch) | Hybrid (OCR + HTML body) |
|---|---|---|
| Static header (status bar, chat header w/ profile + name + nav) | Cropped from frame 0, prepended | **Same** — cropped from frame 0 |
| Static footer (composer + keyboard) | Cropped from last frame, appended | **Same** — cropped from last frame |
| Dynamic body | Pixel-stitched canvas of N keyframes + bridges, with overlay masking + multi-pass fill | **HTML/CSS** reconstruction from per-keyframe OCR + element classifier |
| Output container | One tall PNG split into chunks | One HTML doc with `<header><img></header>` + `<main>...bubbles...</main>` + `<footer><img></footer>`, optionally rendered to PNG with headless Chromium for parity with current chunk format |

## Why the effort is not as large as I implied

| Reason | Detail |
|---|---|
| **Pause-keyframes already give us the work units** | `detect_pauses.py` collapses 3060 frames → ~45 unique screen states. OCR runs ~45 times, not 3060. Each frame is one settled iOS screen. |
| **Layout is highly constrained** | iOS Messenger has a small vocabulary: incoming bubble (left, light grey, rounded), outgoing bubble (right, blue, rounded), reply quote (small bubble stacked on parent), centered timestamp, "Delivered/Read" indicator, attachment thumbnail, system notice ("Image deleted"). A classifier with ~10 categories handles essentially everything. |
| **OpenCV does most of the heavy lifting we already need** | `cv2.findContours` on a bubble-colour mask → bubble bboxes. `cv2.connectedComponents` → element grouping. We were going to add OpenCV for edge-based gutter detection anyway; the marginal cost for body classification is small. |
| **OCR engines are commodity** | Tesseract via `pytesseract` is free, local, and good enough for SF Pro at 3x. PaddleOCR is a drop-in upgrade. macOS users could shell out to Apple Vision via `shortcuts` for near-perfect quality. No model training needed. |
| **De-duplication is free** | Adjacent keyframes share most messages. Matching by `(bubble_bbox_normalised, ocr_text)` collapses duplicates trivially — far simpler than pixel-aligning overlapping bands. |
| **No more 4-pass canvas fill, no more clean_mask, no more bridge frames** | Entire `stitch_keyframes.py` body-canvas logic (~400 LOC) deletes. Replaced by ~200 LOC of "OCR keyframe → emit bubble JSON → merge → render HTML". |

## Honest cost estimate

| Phase | What | Effort |
|---|---|---|
| 1 | `tools/extract_messages.py`: per keyframe, find bubble contours by colour (`cv2.inRange` for blue + grey), OCR each bubble, classify (incoming/outgoing/reply/system/timestamp), output JSON | 1 session |
| 2 | `tools/merge_messages.py`: walk keyframes in scroll order, de-dup by `(bbox, text)`, produce flat ordered message list | 0.5 session |
| 3 | `tools/render_messages.py`: emit HTML/CSS that visually matches iOS Messenger (CSS-only, no JS). Sandwich header.png and footer.png as `<img>`. | 1 session |
| 4 | Optional: render HTML → PNG via `playwright` headless for chunk-PNG parity | 0.25 session |
| 5 | Tune CSS to pixel-match per-bubble look (font, padding, corner radius, drop shadow) | 0.5–1 session |
| **Total** | | **3–4 sessions, comparable to remaining bitmap-stitch hardening** |

For comparison, remaining bitmap-stitch work is also non-trivial: pass-2
root-cause fix, per-frame button detection + masking, gutter clearing,
sub-pixel seam tuning, regression tests for new input videos. Easily 2–3
sessions.

## What the hybrid gains

| Gain | Detail |
|---|---|
| **Zero artifact class** | No scrollbar, no button ghost, no pass-2 fabrication, no overlap seam, no sub-pixel scroll drift — none of these exist in the HTML body. |
| **Searchable / copy-pasteable text** | The whole conversation becomes selectable text. Major UX win for any downstream review use case. |
| **Compact output** | A 600-message conversation is ~50 KB HTML vs ~30 MB of PNG chunks. |
| **Trivially diffable / versionable** | HTML diffs cleanly; PNGs do not. |
| **Re-stylable** | Render in dark mode, larger text, etc., without re-processing video. |
| **Robust to new input videos** | Different phones, iOS versions, languages — OCR + colour classification adapts; the bitmap stitcher would need re-tuning per-device. |

## What the hybrid loses

| Loss | Mitigation |
|---|---|
| **Pixel-exact visual fidelity** | If user wants a screenshot-perfect deliverable, ship the rendered HTML → PNG. Visually indistinguishable for ≥99% of content. |
| **Inline images / GIFs / stickers** | Detect by "non-bubble-colour rectangle inside bubble area" → crop pixels from source frame → embed as `<img>` in HTML. Same trick for reactions. |
| **Custom emoji / Memoji** | Same: crop from source as image. |
| **Animations** | Lost (a single hand-scrolled recording loses these in either approach). |
| **OCR errors on code / unusual scripts** | Real but bounded — Tesseract on SF Pro 3x is ~99% accurate on English. PaddleOCR is better. Surfacing low-confidence regions for human review is straightforward. |
| **Layout edge cases** | Group chats with sender names, voice messages, polls, Apple Pay, link previews — each needs a classifier branch. Long tail. |

## When to switch

Switch to the hybrid path if **any** of these become true:

1. Bitmap-stitch artifact whack-a-mole continues past 2 more sessions without
   a robust general fix.
2. End use requires searchable/copy-pasteable text or structured export
   (JSON / CSV / Markdown of the conversation).
3. We need to process inputs from a non-iOS-3x device (Android, iPad,
   different iPhone resolution) and re-tuning per-device is unattractive.
4. Output size or HTML-diffability becomes a constraint.
5. Downstream consumers want any kind of semantic operation on the
   conversation (search, filter by sender, redact PII, summarise).

## When to stay with bitmap stitch

Stay if **all** of these are true:

1. Output is purely for human eyeballs to scroll once.
2. Pixel-perfect appearance (including any UI quirks) matters more than text
   utility.
3. The current artifact class is reducible to a small, finite fix list.
4. Input device / resolution will not vary.

## Suggested re-entry point

When we revisit, the cheapest validation is:

1. Pick one keyframe (e.g. `out/diag/src_2500.png`).
2. `cv2.inRange` for blue bubble colour → contours → bboxes.
3. OCR each bbox with Tesseract.
4. Render the result as standalone HTML.
5. Eyeball: how close to source? Identify the next-most-needed classifier
   branch.

If step 5 is "very close, just need timestamps and reply quotes", the path is
clear and short. If step 5 is "OCR garbled / bboxes wrong everywhere", we
re-evaluate.

## Decision

| Option | Details | Tradeoffs |
|---|---|---|
| **A. Continue bitmap stitch** | Finish edge-detection-based gutter clearing + per-frame button masking | Known artifact list; finite remaining work; no new deps beyond cv2 |
| **B. Switch to hybrid now** | Build OCR + HTML reconstruction; keep header/footer as images | Eliminates entire artifact class; better downstream utility; comparable effort |
| **C. Build both, A/B compare** | Ship A first; build B in parallel for evaluation | Highest cost; only worth it if downstream use case is unsettled |

**Recommendation: A for now (per user direction). Re-evaluate after the next
edge-detection iteration. If gutter-clearing + per-frame button masking
together do not produce a clean result, switch to B without further
deliberation.**
