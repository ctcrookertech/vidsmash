"""Stitch a vertical-scroll screen-recording into ordered, non-overlapping tall PNG chunks.

Algorithm
---------
1. Probe video dims via ffprobe.
2. Probe pass: sample N frames uniformly to detect the static top/bottom UI
   bands using per-row luma std across samples (std-based metric is robust
   to sparse outliers like a clock tick or recording-indicator blink).
3. Main pass: stream every frame as raw rgb24 from ffmpeg. For each frame:
     - Compute a K-segmented per-row luma signature (cur_row, shape (dyn_h, K))
       and a per-column luma profile (cur_col) of the dynamic band. The row
       signature preserves left/right structure so the matcher does not latch
       onto self-similar all-dark rows; it is also tolerant of small
       horizontal jitter inside a segment.
     - Detect incremental horizontal offset (dx_inc) against the previous
       frame's column profile; maintain a cumulative abs_dx with a zero-
       streak reset.
     - If |abs_dx| > drag-threshold the frame is flagged DRAG: canvas state
       is frozen. Drag events emit deepest-drag sidecar PNGs.
     - Else if MAD(cur_row, prev_row) < stationary_threshold the frame is a
       PAUSE: canvas state is frozen, last_dy reset to 0. This short-circuit
       eliminates drift during long stationary segments where the SAD
       landscape is noise-dominated.
     - Else match cur_row against the full canvas signature (allowing the
       canvas to grow in EITHER direction). The match offset is computed in
       absolute (frame-0-anchored) coordinates with a soft prior_alpha
       penalty toward the predicted position.
     - Edge-hit and suspect-advance guards (symmetric in either direction)
       freeze canvas state if the match is implausible.
     - Otherwise the canvas extends above (if y_top fell below min_top) or
       below (if cur_bottom rose above max_bot).
4. At end-of-video, sort all canvas blocks by absolute y, prepend static-top
   UI, append static-bot UI, and stream into chunk PNGs of --chunk-height.

Outputs
-------
out/chunk_NNN.png  : ordered, non-overlapping vertical chunks of the stitch.
out/drag_NNN.png   : deepest-drag full-frame for each detected drag event.
out/report.json    : full per-frame diagnostics, drag events, gaps, warnings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit
from PIL import Image


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe wrappers
# ---------------------------------------------------------------------------

def _resolve_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) executable paths."""
    candidates = [
        ("ffmpeg", "ffprobe"),
        (
            r"C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
            r"C:\Users\ccrook\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-full_build\bin\ffprobe.exe",
        ),
    ]
    for ff, fp in candidates:
        try:
            subprocess.run(
                [ff, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ff, fp
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("ffmpeg not found on PATH or known install location")


def probe_video(ffprobe: str, path: Path) -> dict:
    out = subprocess.check_output(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,avg_frame_rate,duration",
            "-of", "json",
            str(path),
        ]
    )
    info = json.loads(out)["streams"][0]
    num, den = (int(x) for x in info["avg_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "nb_frames": int(info.get("nb_frames", 0)),
        "fps": fps,
        "duration": float(info.get("duration", 0.0)),
    }


def open_rgb_pipe(
    ffmpeg: str, path: Path, select_expr: str | None = None,
    hwaccel: str | None = None,
    pix_fmt: str = "rgb24",
    crop: tuple[int, int, int, int] | None = None,
) -> subprocess.Popen:
    """Open an ffmpeg process producing raw frames on stdout.

    stderr is sent to DEVNULL to avoid OS-pipe-buffer deadlock on a long run.

    hwaccel: passed to ffmpeg `-hwaccel`. Defaults to None (CPU decode).
    Benchmark on lexi_iphone_messenger_all.mp4 (1126x2436 HEVC, 3060 frames)
    showed CPU decode ~8 s vs NVDEC ~10 s on a Ryzen 9 7845HX + RTX 4070
    Laptop -- NVDEC has GPU init + readback overhead that exceeds the
    dedicated HW decoder's gain on a single-stream short video. Pass "cuda"
    only if you confirm a measured win on your input.

    pix_fmt: "rgb24" (default, 3 B/px) or "gray" (1 B/px). Using "gray" cuts
    pipe bandwidth 3x and lets analysis skip the RGB->luma conversion. The
    stitcher writes color PNGs and so needs rgb24, but cheap diagnostic
    passes (pause detection, profile build) should use "gray".

    crop: (w, h, x, y) ffmpeg crop spec. When set, ffmpeg crops the frame
    server-side before piping, so callers receive only the dyn band (or
    whatever rectangle they asked for). Combined with pix_fmt="gray" this
    drops pipe bandwidth 2.5x on the 1126x969 dyn band of
    lexi_iphone_messenger_all.mp4 and measured 3.11x faster than the
    full-frame gray pipe in bench_ffmpeg_pipes.py option 4. Caller is
    responsible for sizing its read buffer / reshape to (h, w).
    """
    cmd = [ffmpeg, "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(path)]
    vf_parts: list[str] = []
    if select_expr:
        vf_parts.append(f"select='{select_expr}'")
    if crop is not None:
        cw, ch, cx, cy = crop
        vf_parts.append(f"crop={cw}:{ch}:{cx}:{cy}:exact=1")
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
        if select_expr:
            cmd += ["-vsync", "vfr"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    # bufsize: a healthy buffer avoids per-byte Python pipe round-trips.
    # Measured on lexi_iphone_messenger_all.mp4 (1126x2436 gray): bufsize=0
    # -> 57 s of pipe read; bufsize=~11 MB (4 frames) -> 20 s. 16 MB covers
    # 4K-class gray frames with margin. See AGENTS.md "Performance" ->
    # "buffered pipe".
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=16 * 1024 * 1024,
    )


def close_proc(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def read_frame(proc: subprocess.Popen, frame_bytes: int) -> bytes | None:
    """Read exactly frame_bytes from stdout, looping until satisfied or EOF."""
    chunks = []
    remaining = frame_bytes
    while remaining > 0:
        buf = proc.stdout.read(remaining)
        if not buf:
            return None
        chunks.append(buf)
        remaining -= len(buf)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


# ---------------------------------------------------------------------------
# Static UI detection
# ---------------------------------------------------------------------------

def detect_static_bands(
    ffmpeg: str,
    path: Path,
    width: int,
    height: int,
    nb_frames: int,
    n_samples: int = 60,
    std_threshold: float = 12.0,
    min_run: int = 16,
) -> tuple[int, int]:
    """Return (top_static_end, bottom_static_start) row indices.

    Uses per-row standard deviation of a luma projection across uniformly
    sampled frames. std is more robust than max-min range because occasional
    single-sample changes (clock digit tick, battery icon update, recording
    dot animation, dynamic-island content) inflate the range but barely move
    the std. Rows in the keyboard / status-bar UI bands have std ~ 0, rows
    in the conversation area show std typically > 20.

    After thresholding we take the widest contiguous run of "dynamic" rows of
    length >= min_run. If no run satisfies that, the whole frame is treated
    as dynamic.
    """
    n_samples = max(8, min(n_samples, max(2, nb_frames - 1)))
    step = max(1, nb_frames // n_samples)
    indices = list(range(0, nb_frames, step))[:n_samples]
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    proc = open_rgb_pipe(ffmpeg, path, select_expr=expr)
    fbytes = width * height * 3
    profiles = []
    try:
        while True:
            buf = read_frame(proc, fbytes)
            if buf is None:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            luma = (
                0.299 * arr[..., 0].astype(np.float32)
                + 0.587 * arr[..., 1].astype(np.float32)
                + 0.114 * arr[..., 2].astype(np.float32)
            )
            profiles.append(luma.mean(axis=1))
    finally:
        close_proc(proc)
    if len(profiles) < 2:
        return 0, height
    mat = np.stack(profiles, axis=0)  # (S, H)
    row_std = mat.std(axis=0)  # (H,)
    dynamic = row_std > std_threshold
    # Find the widest contiguous True run of length >= min_run.
    best_start, best_len = -1, 0
    i = 0
    H = dynamic.shape[0]
    while i < H:
        if dynamic[i]:
            j = i
            while j < H and dynamic[j]:
                j += 1
            run_len = j - i
            if run_len >= min_run and run_len > best_len:
                best_start, best_len = i, run_len
            i = j
        else:
            i += 1
    if best_len == 0:
        return 0, height
    top = best_start
    bot = best_start + best_len
    if bot - top < 32:
        return 0, height
    return top, bot


# ---------------------------------------------------------------------------
# Offset detection (against the rolling canvas tail — drift-free)
# ---------------------------------------------------------------------------

@dataclass
class OffsetResult:
    p: int           # row index in tail at which cur's dynamic top sits
    sad: float       # mean abs diff per element of best alignment
    confidence: float  # second_best_sad / best_sad
    overlap: int     # number of elements actually compared at best p


def gray_row_profile(gray_band: np.ndarray, hpad: int, n_segments: int = 8) -> np.ndarray:
    """K-segmented per-row signature from an already-gray (HxW uint8) band.

    Identical math to luma_row_profile but skips the RGB->luma conversion
    (which is the dominant cost when reading whole-frame uint8 arrays).
    Use when the upstream pipe is opened with pix_fmt="gray".
    """
    band = gray_band[:, hpad : gray_band.shape[1] - hpad]
    H, W = band.shape
    K = max(1, int(n_segments))
    seg_w = W // K
    if seg_w <= 0:
        return band.astype(np.float32).mean(axis=1, keepdims=True)
    return _gray_row_profile_jit(np.ascontiguousarray(band), K, seg_w)


@njit(cache=True, fastmath=True)
def _gray_row_profile_jit(band: np.ndarray, K: int, seg_w: int) -> np.ndarray:
    """Fused (cast->reshape->mean) loop, ~2-3x faster than the numpy form.

    Numba fuses the uint8->float cast with the segment-mean accumulator so
    no intermediate H x usable_W float32 buffer is allocated. Equivalent to
    band[:, :K*seg_w].astype(np.float32).reshape(H, K, seg_w).mean(axis=2)
    to within float-rounding (see test_match_1d_offset for invariants).
    """
    H = band.shape[0]
    out = np.zeros((H, K), dtype=np.float32)
    inv = np.float32(1.0) / np.float32(seg_w)
    for r in range(H):
        for k in range(K):
            s = np.float32(0.0)
            base = k * seg_w
            for c in range(seg_w):
                s += np.float32(band[r, base + c])
            out[r, k] = s * inv
    return out


def gray_col_profile(gray_band: np.ndarray, vpad: int) -> np.ndarray:
    """Per-column mean from an already-gray (HxW uint8) band. 1D shape (W,)."""
    band = gray_band[vpad : gray_band.shape[0] - vpad, :]
    return _gray_col_profile_jit(np.ascontiguousarray(band))


@njit(cache=True, fastmath=True)
def _gray_col_profile_jit(band: np.ndarray) -> np.ndarray:
    H, W = band.shape
    out = np.zeros(W, dtype=np.float32)
    inv = np.float32(1.0) / np.float32(H)
    for c in range(W):
        s = np.float32(0.0)
        for r in range(H):
            s += np.float32(band[r, c])
        out[c] = s * inv
    return out


def luma_row_profile(rgb_band: np.ndarray, hpad: int, n_segments: int = 8) -> np.ndarray:
    """Per-row K-segmented luma signature.

    Returns shape (H, K) float32 where K = n_segments. Each segment is the
    mean luma over W/K consecutive columns of the inner-padded band. Compared
    to a plain per-row mean (K=1), this preserves left/right structure: an
    avatar-on-left + bubble-on-right row no longer looks like a uniform-grey
    row of the same overall brightness. This is critical for matching across
    long self-similar chat regions where the row-mean signal is otherwise
    too ambiguous.

    The signature still tolerates small horizontal jitter (a few pixels of
    drift inside one segment do not change the segment mean meaningfully).
    Larger horizontal motion is handled by the separate drag detector, which
    freezes canvas state when |abs_dx| > drag_threshold.
    """
    band = rgb_band[:, hpad : rgb_band.shape[1] - hpad]
    luma = (
        0.299 * band[..., 0].astype(np.float32)
        + 0.587 * band[..., 1].astype(np.float32)
        + 0.114 * band[..., 2].astype(np.float32)
    )
    H, W = luma.shape
    K = max(1, int(n_segments))
    seg_w = W // K
    if seg_w <= 0:
        return luma.mean(axis=1, keepdims=True)
    # Drop a few right-edge cols if W not divisible; matches alignment.
    usable = seg_w * K
    cropped = luma[:, :usable]
    # Reshape to (H, K, seg_w) and mean over seg_w.
    return cropped.reshape(H, K, seg_w).mean(axis=2)


def luma_col_profile(rgb_band: np.ndarray, vpad: int) -> np.ndarray:
    """Per-column mean luma across rows (with vertical inner pad).

    1D shape (W,). Used to detect horizontal drag by aligning against the
    previous frame's column profile.
    """
    band = rgb_band[vpad : rgb_band.shape[0] - vpad, :]
    return (
        0.299 * band[..., 0].astype(np.float32)
        + 0.587 * band[..., 1].astype(np.float32)
        + 0.114 * band[..., 2].astype(np.float32)
    ).mean(axis=0)


def match_1d_offset(
    ref: np.ndarray,
    cur: np.ndarray,
    predicted_p: int,
    search_radius: int,
    min_overlap: int,
    prior_alpha: float = 0.0,
) -> OffsetResult:
    """Find integer p that minimizes (MAD + prior_alpha*|p - predicted_p|).

    Convention: cur[i] aligns with ref[i + p].
      p > 0 : cur starts BELOW/RIGHT of ref start.
      p < 0 : cur starts ABOVE/LEFT of ref start.

    ref and cur may be 1D (shape (N,)) or 2D (shape (N, K)). For 2D, MAD is
    averaged over both axes (per-element). Per-element MAD (so candidates
    with different overlap are comparable).

    prior_alpha > 0 adds a soft per-luma-unit penalty proportional to
    |p - predicted_p| in pixels. This biases the matcher toward the predicted
    position so that, when several locations are nearly tied (common in
    long, self-similar chat regions), the closest-to-prediction wins. Set
    prior_alpha = 0 to disable.

    The returned sad is the raw MAD (without the prior).

    Implementation: numeric core is a Numba @njit kernel. We tried a
    vectorized numpy variant (NaN-padded ref + sliding_window_view) -- it
    produced identical results but ran 1.7x SLOWER on
    lexi_iphone_messenger_all.mp4 because the 50 MB per-call candidate
    matrix blew the L2/L3 cache. Numba on the Python loop keeps the
    cache-friendly per-iteration shape AND removes numpy dispatch overhead.
    See AGENTS.md "Performance".
    """
    if ref.ndim == 1:
        ref2 = np.ascontiguousarray(ref, dtype=np.float32).reshape(-1, 1)
        cur2 = np.ascontiguousarray(cur, dtype=np.float32).reshape(-1, 1)
    else:
        ref2 = np.ascontiguousarray(ref, dtype=np.float32)
        cur2 = np.ascontiguousarray(cur, dtype=np.float32)
    lo = predicted_p - search_radius
    hi = predicted_p + search_radius
    sads, overlaps = _match_1d_core(ref2, cur2, lo, hi, int(min_overlap))
    ps = np.arange(lo, hi + 1)
    if prior_alpha > 0:
        penalty = (prior_alpha * np.abs(ps - predicted_p)).astype(np.float32)
        scored = sads + penalty
    else:
        scored = sads
    best_i = int(np.argmin(scored))
    best = float(sads[best_i])
    excl = 3
    mask = np.ones_like(scored, dtype=bool)
    s = max(0, best_i - excl)
    e = min(scored.shape[0], best_i + excl + 1)
    mask[s:e] = False
    valid = mask & np.isfinite(scored)
    second = float(scored[valid].min()) if valid.any() else float(scored[best_i])
    conf = (second / float(scored[best_i])) if float(scored[best_i]) > 1e-6 else float("inf")
    return OffsetResult(
        p=int(ps[best_i]),
        sad=best,
        confidence=conf,
        overlap=int(overlaps[best_i]),
    )


@njit(cache=True, fastmath=True)
def _match_1d_core(
    ref: np.ndarray, cur: np.ndarray, lo: int, hi: int, min_overlap: int,
):
    """SAD slide-search core. ref/cur are float32 2D (N, K_cols).

    Returns (sads, overlaps). sads is per-element MAD or +inf where overlap
    < min_overlap. Equivalent to the prior Python loop within float
    rounding. Numba fuses the inner abs-diff + accumulator so we get a
    tight scalar loop instead of one numpy dispatch per offset (~9-10x
    speedup on the lexi_iphone_messenger_all profile of search_radius=400,
    h=969, K=8).
    """
    K_ref = ref.shape[0]
    K_cols = ref.shape[1]
    h = cur.shape[0]
    n = hi - lo + 1
    sads = np.full(n, np.inf, dtype=np.float32)
    overlaps = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = lo + i
        c0 = -p if -p > 0 else 0
        c1_a = h
        c1_b = K_ref - p
        c1 = c1_a if c1_a < c1_b else c1_b
        ov = c1 - c0
        if ov < min_overlap:
            continue
        s = np.float32(0.0)
        for r in range(c0, c1):
            for k in range(K_cols):
                d = cur[r, k] - ref[r + p, k]
                if d < 0:
                    d = -d
                s += d
        sads[i] = s / np.float32(ov * K_cols)
        overlaps[i] = ov
    return sads, overlaps


# ---------------------------------------------------------------------------
# Bidirectional canvas
# ---------------------------------------------------------------------------

class Canvas:
    """Canvas that may grow at both ends.

    Coordinates are ABSOLUTE: frame 0's dyn-top is anchored at y=0. As the
    user scrolls in either direction, ``y_top`` for subsequent frames may go
    positive (newer content appears below) or negative (older content appears
    above). The canvas extends accordingly.

    Two stores are maintained:
      * ``sig_buf``: a single pre-allocated (K_sig, K) float32 buffer with the
        anchor placed at ``anchor`` (middle). Absolute y maps to index
        ``anchor + y``. The matcher consumes a contiguous view of this
        buffer covering [min_top, max_bot).
      * ``img_blocks``: list of ``(y_start, rows)`` tuples. New rows are
        recorded as they are added; at finalize time the list is sorted by
        ``y_start`` and concatenated for chunk emission.

    The same scheme is used for gap markers (red rows) inserted when the
    matcher reports an advance >= dyn_h, so absolute-y indexing stays
    consistent everywhere.
    """

    def __init__(self, W: int, K: int, dyn_h: int, sig_preallocate: int = 500_000):
        if sig_preallocate < 4 * dyn_h:
            sig_preallocate = max(sig_preallocate, 8 * dyn_h)
        self.W = W
        self.K = K
        self.dyn_h = dyn_h
        self.sig_capacity = sig_preallocate
        self.anchor = sig_preallocate // 2
        self.sig_buf = np.zeros((sig_preallocate, K), dtype=np.float32)
        self.img_blocks: list[tuple[int, np.ndarray]] = []
        self.gaps: list[dict] = []
        self.min_top = 0
        self.max_bot = 0
        self._initialized = False

    def _abs_to_idx(self, y: int) -> int:
        return self.anchor + y

    def init(self, dyn_rows: np.ndarray, sig: np.ndarray) -> None:
        assert not self._initialized
        h = dyn_rows.shape[0]
        if h != self.dyn_h:
            raise ValueError(f"dyn_rows height {h} != dyn_h {self.dyn_h}")
        if sig.shape[0] != h:
            raise ValueError("sig height mismatch")
        self.min_top = 0
        self.max_bot = h
        i = self._abs_to_idx(0)
        self.sig_buf[i : i + h] = sig
        self.img_blocks.append((0, dyn_rows.copy()))
        self._initialized = True

    def get_sig(self) -> tuple[np.ndarray, int]:
        """Returns (sig_view, top_y); sig_view covers absolute [top_y, max_bot)."""
        i0 = self._abs_to_idx(self.min_top)
        i1 = self._abs_to_idx(self.max_bot)
        return self.sig_buf[i0:i1], self.min_top

    def _check_capacity(self, lo_abs: int, hi_abs: int) -> None:
        if self._abs_to_idx(lo_abs) < 0 or self._abs_to_idx(hi_abs) > self.sig_capacity:
            raise RuntimeError(
                f"canvas signature buffer exhausted: need [{lo_abs}, {hi_abs}), "
                f"capacity {self.sig_capacity} anchored at {self.anchor}"
            )

    def extend_below(self, y_top_cur: int, dyn_rows: np.ndarray, sig: np.ndarray, frame_idx: int) -> int:
        """Extend canvas downward if cur reaches below max_bot. Returns advance in rows."""
        cur_bot = y_top_cur + self.dyn_h
        if cur_bot <= self.max_bot:
            return 0
        advance = cur_bot - self.max_bot
        self._check_capacity(self.min_top, cur_bot)
        if advance >= self.dyn_h:
            gap_h = advance - self.dyn_h
            if gap_h > 0:
                gap_img = np.zeros((gap_h, self.W, 3), dtype=np.uint8)
                gap_img[:, :, 0] = 255
                self.img_blocks.append((self.max_bot, gap_img))
                # Sig stays zero (pre-allocated).
                self.gaps.append({"i": frame_idx, "y_start": int(self.max_bot), "gap_rows": int(gap_h), "side": "below"})
            new_start = self.max_bot + gap_h
            self.img_blocks.append((new_start, dyn_rows.copy()))
            i = self._abs_to_idx(new_start)
            self.sig_buf[i : i + self.dyn_h] = sig
        else:
            slice_start = self.dyn_h - advance
            self.img_blocks.append((self.max_bot, dyn_rows[slice_start:].copy()))
            i = self._abs_to_idx(self.max_bot)
            self.sig_buf[i : i + advance] = sig[slice_start:]
        self.max_bot = cur_bot
        return advance

    def extend_above(self, y_top_cur: int, dyn_rows: np.ndarray, sig: np.ndarray, frame_idx: int) -> int:
        """Extend canvas upward if cur starts above min_top. Returns advance in rows."""
        if y_top_cur >= self.min_top:
            return 0
        advance = self.min_top - y_top_cur
        self._check_capacity(y_top_cur, self.max_bot)
        if advance >= self.dyn_h:
            # Layout from top: dyn_rows at [y_top_cur, y_top_cur+dyn_h), gap below.
            self.img_blocks.append((y_top_cur, dyn_rows.copy()))
            i = self._abs_to_idx(y_top_cur)
            self.sig_buf[i : i + self.dyn_h] = sig
            gap_h = advance - self.dyn_h
            if gap_h > 0:
                gap_img = np.zeros((gap_h, self.W, 3), dtype=np.uint8)
                gap_img[:, :, 0] = 255
                self.img_blocks.append((y_top_cur + self.dyn_h, gap_img))
                self.gaps.append({"i": frame_idx, "y_start": int(y_top_cur + self.dyn_h), "gap_rows": int(gap_h), "side": "above"})
        else:
            self.img_blocks.append((y_top_cur, dyn_rows[:advance].copy()))
            i = self._abs_to_idx(y_top_cur)
            self.sig_buf[i : i + advance] = sig[:advance]
        self.min_top = y_top_cur
        return advance

    def emit(self, writer: "ChunkWriter") -> None:
        """Sort blocks by y_start and append rows to writer in absolute-y order.

        If any blocks overlap (matcher revisited rows), the later block's
        overlapping prefix is discarded so each absolute row is written
        exactly once. Gaps between recorded blocks (not captured as red
        markers) are also reported.
        """
        if not self.img_blocks:
            return
        blocks = sorted(self.img_blocks, key=lambda b: b[0])
        cursor = blocks[0][0]
        for y_start, rows in blocks:
            R = rows.shape[0]
            y_end = y_start + R
            if y_end <= cursor:
                continue  # fully overlapped by earlier block
            if y_start < cursor:
                rows = rows[cursor - y_start :]
                y_start = cursor
            elif y_start > cursor:
                # Unexpected gap (extend_* should have inserted red markers).
                gap_h = y_start - cursor
                gap_img = np.zeros((gap_h, self.W, 3), dtype=np.uint8)
                gap_img[:, :, 0] = 255
                writer.append(gap_img)
            writer.append(rows)
            cursor = y_end


# ---------------------------------------------------------------------------
# Chunked canvas writer
# ---------------------------------------------------------------------------

@dataclass
class ChunkWriter:
    out_dir: Path
    width: int
    chunk_height: int
    prefix: str = "chunk"
    _buffers: list[np.ndarray] = field(default_factory=list)
    _buffer_rows: int = 0
    _chunk_index: int = 0
    _total_rows: int = 0
    _emitted: list[dict] = field(default_factory=list)

    def append(self, rows: np.ndarray) -> None:
        """Append rows (R, W, 3) uint8 to the active chunk, flushing as needed."""
        if rows.size == 0:
            return
        self._buffers.append(rows)
        self._buffer_rows += rows.shape[0]
        self._total_rows += rows.shape[0]
        while self._buffer_rows >= self.chunk_height:
            self._flush_one(self.chunk_height)

    def _flush_one(self, take: int) -> None:
        merged = np.concatenate(self._buffers, axis=0)
        head = merged[:take]
        tail = merged[take:]
        self._write(head)
        self._buffers = [tail] if tail.size else []
        self._buffer_rows = tail.shape[0] if tail.size else 0

    def _write(self, arr: np.ndarray) -> None:
        path = self.out_dir / f"{self.prefix}_{self._chunk_index:03d}.png"
        if arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        Image.fromarray(arr, mode=mode).save(path, optimize=False, compress_level=4)
        self._emitted.append({"index": self._chunk_index, "path": str(path), "height": int(arr.shape[0])})
        self._chunk_index += 1

    def finalize(self) -> list[dict]:
        if self._buffer_rows > 0:
            merged = np.concatenate(self._buffers, axis=0)
            self._write(merged)
            self._buffers = []
            self._buffer_rows = 0
        return self._emitted

    @property
    def total_rows(self) -> int:
        return self._total_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk-height", type=int, default=4096)
    ap.add_argument("--ui", choices=["keep-once", "strip", "keep-on-every-chunk"], default="keep-once")
    ap.add_argument("--max-dy", type=int, default=400, help="Max plausible per-frame vertical offset in pixels.")
    ap.add_argument("--max-dx", type=int, default=300, help="Max plausible per-frame horizontal offset in pixels.")
    ap.add_argument("--search-radius", type=int, default=200, help="dy search radius around predicted p (pixels). Hand-scrolling rarely exceeds 200 px between adjacent frames at 60 fps.")
    ap.add_argument("--n-segments", type=int, default=16, help="Number of horizontal segments for the per-row K-segmented luma signature (more = more discriminative, slower).")
    ap.add_argument("--dy-prior-alpha", type=float, default=0.02, help="Soft penalty per pixel of |p - predicted_p| added to SAD during dy matching. Breaks ties toward consistency. 0 disables.")
    ap.add_argument("--stationary-threshold", type=float, default=0.5, help="If MAD(cur_row, prev_row) < this, the frame is treated as a pause: y_top is held fixed and the match step is skipped. Eliminates drift during long static segments.")
    ap.add_argument("--drag-threshold", type=int, default=20, help="Frames with |cumulative dx| > this are treated as a drag and frozen.")
    ap.add_argument("--drag-reset-frames", type=int, default=3, help="Number of consecutive |incremental dx|==0 frames that resets cumulative dx to 0.")
    ap.add_argument("--inner-pad-frac", type=float, default=0.05, help="Fraction of dyn_h trimmed top/bot of matching strip (and same fraction of W trimmed left/right).")
    ap.add_argument("--std-threshold", type=float, default=12.0, help="Per-row luma std threshold for static-band detection.")
    ap.add_argument("--min-static-run", type=int, default=16, help="Min contiguous dynamic rows required.")
    ap.add_argument("--dynamic-top", type=int, default=-1, help="Override: explicit top row of the dynamic band (skip auto-detect).")
    ap.add_argument("--dynamic-bottom", type=int, default=-1, help="Override: explicit bottom row of the dynamic band (skip auto-detect).")
    ap.add_argument("--confidence-warn", type=float, default=1.15)
    ap.add_argument("--sad-floor", type=float, default=0.3, help="SAD below this is treated as 'essentially zero' for confidence purposes.")
    ap.add_argument("--max-advance-frac", type=float, default=1.0, help="If a single-frame advance exceeds this fraction of dyn_h, the match is treated as suspect: canvas state is frozen and a warning is logged. 1.0 means at most one full screen of new rows per frame (already a 60-fps physical upper bound).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = _resolve_ffmpeg()
    info = probe_video(ffprobe, args.input)
    W, H, N = info["width"], info["height"], info["nb_frames"]
    if N <= 0:
        raise RuntimeError("ffprobe reported zero frames; refusing to proceed.")
    print(f"[probe] {W}x{H}, {N} frames, {info['fps']:.3f} fps, {info['duration']:.2f}s", flush=True)

    print("[ui] detecting static bands...", flush=True)
    if args.dynamic_top >= 0 and args.dynamic_bottom > args.dynamic_top:
        top = max(0, min(args.dynamic_top, H - 1))
        bot = max(top + 1, min(args.dynamic_bottom, H))
        print(f"[ui] using overrides --dynamic-top={top} --dynamic-bottom={bot}", flush=True)
    else:
        top, bot = detect_static_bands(
            ffmpeg, args.input, W, H, N,
            std_threshold=args.std_threshold,
            min_run=args.min_static_run,
        )
    dyn_h = bot - top
    print(f"[ui] static_top=0..{top}  dynamic={top}..{bot} (h={dyn_h})  static_bot={bot}..{H}", flush=True)
    if dyn_h < 64:
        raise RuntimeError(f"Dynamic band too small (h={dyn_h}); aborting. Use --dynamic-top/--dynamic-bottom to override.")
    if dyn_h > args.chunk_height:
        print(f"[warn] dyn_h ({dyn_h}) > chunk_height ({args.chunk_height}); chunks may split a single frame.", flush=True)
    max_single_advance = max(1, int(round(args.max_advance_frac * dyn_h)))
    print(f"[guard] max-single-advance={max_single_advance} rows (max-advance-frac={args.max_advance_frac})", flush=True)

    vpad = max(4, int(args.inner_pad_frac * dyn_h))
    hpad = max(4, int(args.inner_pad_frac * W))
    print(f"[ui] inner pads: vpad={vpad}, hpad={hpad}", flush=True)

    writer = ChunkWriter(out_dir=args.out, width=W, chunk_height=args.chunk_height)
    canvas = Canvas(W=W, K=args.n_segments, dyn_h=dyn_h)

    # Previous frame's column profile, used to detect incremental dx.
    prev_col: np.ndarray | None = None
    prev_row: np.ndarray | None = None
    abs_dx = 0
    zero_dx_streak = 0

    y_top = 0
    last_dy = 0

    static_top_img: np.ndarray | None = None
    static_bot_img: np.ndarray | None = None
    last_frame_img: np.ndarray | None = None

    per_frame: list[dict] = []
    warnings_low: list[dict] = []
    drag_events: list[dict] = []
    in_drag = False
    cur_drag = None

    proc = open_rgb_pipe(ffmpeg, args.input)
    fbytes = W * H * 3

    frame_idx = 0
    progress_every = max(1, N // 50)
    min_overlap_rows = max(32, dyn_h // 8)
    min_overlap_cols = max(64, W // 6)

    try:
        while True:
            buf = read_frame(proc, fbytes)
            if buf is None:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            dyn = frame[top:bot]
            if bot < H:
                # Always capture the most recent frame's static-bottom band.
                # At end-of-video this gives us the keyboard/toolbar UI for
                # the final chunk without a second ffmpeg pass.
                last_frame_img = frame
            cur_row = luma_row_profile(dyn, hpad=hpad, n_segments=args.n_segments)
            cur_col = luma_col_profile(dyn, vpad=vpad)

            # ----- dx (horizontal drag) detection -----
            dx_inc = 0
            dx_sad = 0.0
            dx_conf = float("inf")
            if prev_col is not None:
                dxres = match_1d_offset(
                    ref=prev_col, cur=cur_col,
                    predicted_p=0,
                    search_radius=args.max_dx,
                    min_overlap=min_overlap_cols,
                )
                dx_inc = dxres.p
                dx_sad = dxres.sad
                dx_conf = dxres.confidence
                if dx_inc == 0:
                    zero_dx_streak += 1
                else:
                    zero_dx_streak = 0
                abs_dx += dx_inc
                if zero_dx_streak >= args.drag_reset_frames and abs(abs_dx) < args.drag_threshold:
                    abs_dx = 0
            is_drag = abs(abs_dx) > args.drag_threshold

            # ----- drag-event bookkeeping -----
            if is_drag:
                if not in_drag:
                    in_drag = True
                    cur_drag = {
                        "start": frame_idx,
                        "end": frame_idx,
                        "max_abs_dx": abs(abs_dx),
                        "deepest_frame_idx": frame_idx,
                        "deepest_full_frame": frame.copy(),
                    }
                else:
                    cur_drag["end"] = frame_idx
                    if abs(abs_dx) > cur_drag["max_abs_dx"]:
                        cur_drag["max_abs_dx"] = abs(abs_dx)
                        cur_drag["deepest_frame_idx"] = frame_idx
                        cur_drag["deepest_full_frame"] = frame.copy()
            else:
                if in_drag and cur_drag is not None:
                    idx_drag = len(drag_events)
                    sidecar_path = args.out / f"drag_{idx_drag:03d}.png"
                    Image.fromarray(cur_drag["deepest_full_frame"], mode="RGB").save(sidecar_path, compress_level=4)
                    drag_events.append({
                        "index": idx_drag,
                        "start_frame": cur_drag["start"],
                        "end_frame": cur_drag["end"],
                        "deepest_frame": cur_drag["deepest_frame_idx"],
                        "max_abs_dx": int(cur_drag["max_abs_dx"]),
                        "sidecar": str(sidecar_path),
                    })
                    in_drag = False
                    cur_drag = None

            # ----- dy / canvas advance -----
            if frame_idx == 0:
                canvas.init(dyn, cur_row)
                y_top = 0
                if top > 0:
                    static_top_img = frame[:top].copy()
                per_frame.append({
                    "i": 0, "p": 0, "y_top": 0, "dy": 0, "new_rows": int(dyn_h),
                    "sad": 0.0, "conf": float("inf"), "overlap": int(dyn_h),
                    "dx_inc": 0, "abs_dx": 0, "is_drag": False,
                    "dx_sad": 0.0, "dx_conf": float("inf"),
                    "stationary": False,
                })
            elif is_drag:
                # Freeze canvas state. Record frame but do nothing else.
                per_frame.append({
                    "i": frame_idx, "p": None, "y_top": int(y_top), "dy": 0,
                    "new_rows": 0, "sad": None, "conf": None, "overlap": None,
                    "dx_inc": int(dx_inc), "abs_dx": int(abs_dx), "is_drag": True,
                    "dx_sad": float(dx_sad), "dx_conf": float(dx_conf),
                    "stationary": False,
                })
            else:
                # ----- stationary-frame short-circuit -----
                # If the row signature is essentially unchanged from the
                # previous frame, the user is paused. Skip matching entirely
                # so noise in the SAD landscape can't induce drift.
                stationary_mad = None
                if prev_row is not None:
                    stationary_mad = float(np.abs(cur_row - prev_row).mean())
                if stationary_mad is not None and stationary_mad < args.stationary_threshold:
                    last_dy = 0
                    per_frame.append({
                        "i": frame_idx, "p": int(y_top), "y_top": int(y_top),
                        "dy": 0, "new_rows": 0,
                        "sad": float(stationary_mad), "conf": float("inf"),
                        "overlap": int(dyn_h),
                        "dx_inc": int(dx_inc), "abs_dx": int(abs_dx), "is_drag": False,
                        "dx_sad": float(dx_sad), "dx_conf": float(dx_conf),
                        "stationary": True,
                    })
                else:
                    canvas_sig, canvas_top_y = canvas.get_sig()
                    # Convert absolute predicted y to canvas-relative offset.
                    predicted_p_abs = y_top + last_dy
                    cp = predicted_p_abs - canvas_top_y
                    radius = args.max_dy if frame_idx == 1 else args.search_radius
                    res = match_1d_offset(
                        ref=canvas_sig, cur=cur_row,
                        predicted_p=cp,
                        search_radius=radius,
                        min_overlap=min_overlap_rows,
                        prior_alpha=args.dy_prior_alpha,
                    )
                    new_y_top = canvas_top_y + res.p
                    hit_edge = abs(res.p - cp) >= radius - 1
                    if hit_edge:
                        warnings_low.append({
                            "i": frame_idx, "dy": int(new_y_top - y_top), "p": int(new_y_top),
                            "sad": float(res.sad), "conf": float(res.confidence),
                            "overlap": int(res.overlap),
                            "edge_hit": True,
                        })
                        per_frame.append({
                            "i": frame_idx, "p": int(new_y_top), "y_top": int(y_top),
                            "dy": 0, "new_rows": 0,
                            "sad": float(res.sad), "conf": float(res.confidence),
                            "overlap": int(res.overlap),
                            "dx_inc": int(dx_inc), "abs_dx": int(abs_dx), "is_drag": False,
                            "dx_sad": float(dx_sad), "dx_conf": float(dx_conf),
                            "edge_hit": True, "stationary": False,
                        })
                        prev_col = cur_col
                        prev_row = cur_row
                        if frame_idx % progress_every == 0:
                            print(
                                f"[main] frame {frame_idx}/{N}  y_top={y_top}  "
                                f"canvas=[{canvas.min_top},{canvas.max_bot})  "
                                f"abs_dx={abs_dx}  EDGE_HIT",
                                flush=True,
                            )
                        frame_idx += 1
                        continue
                    dy = new_y_top - y_top
                    # ----- suspect-match guard (symmetric) -----
                    # Bound advance in EITHER direction. A frame can extend
                    # the canvas by at most one dyn_h above OR below per frame.
                    adv_above_proposed = max(0, canvas.min_top - new_y_top)
                    adv_below_proposed = max(0, (new_y_top + dyn_h) - canvas.max_bot)
                    if max(adv_above_proposed, adv_below_proposed) > max_single_advance:
                        warnings_low.append({
                            "i": frame_idx, "dy": int(dy), "p": int(new_y_top),
                            "sad": float(res.sad), "conf": float(res.confidence),
                            "overlap": int(res.overlap),
                            "suspect_advance": int(max(adv_above_proposed, adv_below_proposed)),
                        })
                        per_frame.append({
                            "i": frame_idx, "p": int(new_y_top), "y_top": int(y_top),
                            "dy": 0, "new_rows": 0,
                            "sad": float(res.sad), "conf": float(res.confidence),
                            "overlap": int(res.overlap),
                            "dx_inc": int(dx_inc), "abs_dx": int(abs_dx), "is_drag": False,
                            "dx_sad": float(dx_sad), "dx_conf": float(dx_conf),
                            "suspect": True,
                            "suspect_advance": int(max(adv_above_proposed, adv_below_proposed)),
                            "stationary": False,
                        })
                    else:
                        y_top = new_y_top
                        last_dy = dy
                        adv_below = canvas.extend_below(y_top, dyn, cur_row, frame_idx)
                        adv_above = canvas.extend_above(y_top, dyn, cur_row, frame_idx)
                        new_rows = adv_below + adv_above

                        is_low_conf = (
                            res.confidence < args.confidence_warn
                            and res.sad > args.sad_floor
                        )
                        if is_low_conf:
                            warnings_low.append({
                                "i": frame_idx, "dy": int(dy), "p": int(y_top),
                                "sad": float(res.sad), "conf": float(res.confidence),
                                "overlap": int(res.overlap),
                            })
                        per_frame.append({
                            "i": frame_idx, "p": int(y_top), "y_top": int(y_top),
                            "dy": int(dy), "new_rows": int(new_rows),
                            "adv_above": int(adv_above), "adv_below": int(adv_below),
                            "sad": float(res.sad), "conf": float(res.confidence),
                            "overlap": int(res.overlap),
                            "dx_inc": int(dx_inc), "abs_dx": int(abs_dx), "is_drag": False,
                            "dx_sad": float(dx_sad), "dx_conf": float(dx_conf),
                            "stationary": False,
                        })

            prev_col = cur_col
            prev_row = cur_row

            if frame_idx % progress_every == 0:
                print(
                    f"[main] frame {frame_idx}/{N}  y_top={y_top}  "
                    f"canvas=[{canvas.min_top},{canvas.max_bot}) "
                    f"({canvas.max_bot - canvas.min_top} rows)  "
                    f"abs_dx={abs_dx}{' DRAG' if is_drag else ''}",
                    flush=True,
                )
            frame_idx += 1
    finally:
        close_proc(proc)

    # Close out any open drag event at end-of-video.
    if in_drag and cur_drag is not None:
        idx_drag = len(drag_events)
        sidecar_path = args.out / f"drag_{idx_drag:03d}.png"
        Image.fromarray(cur_drag["deepest_full_frame"], mode="RGB").save(sidecar_path, compress_level=4)
        drag_events.append({
            "index": idx_drag,
            "start_frame": cur_drag["start"],
            "end_frame": cur_drag["end"],
            "deepest_frame": cur_drag["deepest_frame_idx"],
            "max_abs_dx": int(cur_drag["max_abs_dx"]),
            "sidecar": str(sidecar_path),
        })

    if frame_idx != N:
        print(f"[warn] decoded {frame_idx} frames but ffprobe reported {N}", flush=True)

    # Static bottom from the last decoded frame (captured during the main
    # pass to avoid a second ffmpeg decode and any select-filter edge cases).
    if args.ui != "strip" and bot < H and last_frame_img is not None:
        static_bot_img = last_frame_img[bot:H].copy()

    # Emit canvas in absolute-y order. With UI mode, prepend static-top to the
    # very first chunk and append static-bot to the very last chunk after
    # canvas rows are fully written.
    if args.ui == "keep-once" and static_top_img is not None:
        writer.append(static_top_img)
    canvas.emit(writer)
    if args.ui == "keep-once" and static_bot_img is not None:
        writer.append(static_bot_img)
    emitted = writer.finalize()

    if args.ui == "keep-on-every-chunk":
        for ch in emitted:
            with Image.open(ch["path"]) as im:
                img = np.array(im.convert("RGB"))
            parts = []
            if static_top_img is not None:
                parts.append(static_top_img)
            parts.append(img)
            if static_bot_img is not None:
                parts.append(static_bot_img)
            new = np.concatenate(parts, axis=0)
            Image.fromarray(new, mode="RGB").save(ch["path"], compress_level=4)
            ch["height"] = int(new.shape[0])

    report = {
        "input": str(args.input),
        "video": info,
        "static": {"top_end": int(top), "bottom_start": int(bot), "dynamic_h": int(dyn_h)},
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "result": {
            "total_dynamic_rows": int(canvas.max_bot - canvas.min_top),
            "canvas_extent": {"min_top": int(canvas.min_top), "max_bot": int(canvas.max_bot)},
            "frames_decoded": int(frame_idx),
            "frames_expected": int(N),
            "chunks": emitted,
            "gaps": canvas.gaps,
            "drag_events": drag_events,
            "low_confidence_pairs": warnings_low[:200],
            "low_confidence_count": len(warnings_low),
        },
        "frames": per_frame,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"[done] {len(emitted)} chunks, canvas=[{canvas.min_top},{canvas.max_bot}) "
        f"({canvas.max_bot - canvas.min_top} rows), "
        f"{len(drag_events)} drag events, "
        f"{len(warnings_low)} low-confidence pairs, {len(canvas.gaps)} gaps",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
