"""Correctness check for the vectorized match_1d_offset.

Reimplements the legacy loop-based MAD inline and asserts that the new
vectorized stitch_scroll_b.match_1d_offset returns identical (or float-tight)
results across a battery of synthetic inputs covering:
  * 1D and 2D ref / cur
  * partial overlap at left and right edges
  * predicted_p != 0
  * prior_alpha > 0
  * min_overlap exclusion
  * sub-overlap-only candidates
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_scroll_b import match_1d_offset, OffsetResult  # noqa: E402


def legacy_match(
    ref: np.ndarray,
    cur: np.ndarray,
    predicted_p: int,
    search_radius: int,
    min_overlap: int,
    prior_alpha: float = 0.0,
) -> OffsetResult:
    K_ref = ref.shape[0]
    h = cur.shape[0]
    lo = predicted_p - search_radius
    hi = predicted_p + search_radius
    ps = np.arange(lo, hi + 1)
    sads = np.full(ps.shape[0], np.inf, dtype=np.float32)
    overlaps = np.zeros(ps.shape[0], dtype=np.int32)
    for i, p in enumerate(ps):
        c0 = max(0, -p)
        c1 = min(h, K_ref - p)
        ov = c1 - c0
        if ov < min_overlap:
            continue
        a = cur[c0:c1].astype(np.float32)
        b = ref[c0 + p : c1 + p].astype(np.float32)
        sads[i] = np.abs(a - b).mean()
        overlaps[i] = ov
    if prior_alpha > 0:
        penalty = prior_alpha * np.abs(ps - predicted_p).astype(np.float32)
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


def _close(a: float, b: float, *, atol: float, rtol: float = 1e-5) -> bool:
    if np.isinf(a) and np.isinf(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def run_case(name: str, ref, cur, pp, R, mo, pa=0.0):
    new = match_1d_offset(ref, cur, pp, R, mo, pa)
    old = legacy_match(ref, cur, pp, R, mo, pa)
    ok_p = new.p == old.p
    # MAD/conf may differ in last ULPs due to sum order; use a tight tolerance.
    ok_sad = _close(new.sad, old.sad, atol=1e-4)
    ok_ov = new.overlap == old.overlap
    ok_conf = (
        (np.isinf(new.confidence) and np.isinf(old.confidence))
        or (np.isnan(new.confidence) and np.isnan(old.confidence))
        or _close(new.confidence, old.confidence, atol=1e-3, rtol=1e-3)
    )
    status = "OK" if (ok_p and ok_sad and ok_ov and ok_conf) else "FAIL"
    print(
        f"[{status}] {name:<40}  new={new}  old={old}  "
        f"p={ok_p} sad={ok_sad} ov={ok_ov} conf={ok_conf}"
    )
    assert ok_p and ok_sad and ok_ov and ok_conf, name


def main() -> int:
    rng = np.random.default_rng(1234)

    # 1D, identity inside search range, predicted at 0
    K = 200
    h = 200
    ref = rng.integers(0, 256, size=K, dtype=np.uint8).astype(np.float32)
    cur = ref.copy()
    run_case("1D identity", ref, cur, 0, 20, 50)

    # 1D, known shift
    shift = 7
    cur2 = np.roll(ref, -shift)  # cur[i] == ref[i+shift]
    cur2[K - shift :] = 0  # roll wraps; replace wrap with garbage
    run_case("1D shift=+7 (partial)", ref[:K], cur2[: K - shift].astype(np.float32), shift, 20, 50)

    # 1D, large negative shift, near-edge
    run_case("1D shift=-30 near edge", ref, np.concatenate([np.zeros(30, np.float32), ref[:-30]]), -30, 50, 50)

    # 1D, prior_alpha forces near-prediction
    run_case("1D prior_alpha=0.5", ref, ref.copy(), 0, 30, 50, pa=0.5)

    # 1D, search radius exceeds ref (forces NaN padding both sides)
    short = rng.integers(0, 256, size=80, dtype=np.uint8).astype(np.float32)
    run_case("1D radius>=ref", short, short.copy(), 0, 100, 20)

    # 1D, min_overlap excludes some candidates
    run_case("1D min_overlap filter", ref, ref.copy(), 0, 100, 150)

    # 2D, K=8 segments
    ref2d = rng.integers(0, 256, size=(K, 8), dtype=np.uint8).astype(np.float32)
    run_case("2D identity K=8", ref2d, ref2d.copy(), 0, 25, 50)

    # 2D, known shift
    sh = 12
    cur2d = np.zeros_like(ref2d)
    cur2d[: K - sh] = ref2d[sh:]
    run_case("2D shift=+12 (partial)", ref2d, cur2d[: K - sh].copy(), sh, 30, 40)

    # 2D, predicted_p != 0
    run_case("2D predicted_p=15 radius=20", ref2d, ref2d.copy(), 15, 20, 60, pa=0.0)
    run_case("2D predicted_p=15 prior=0.2", ref2d, ref2d.copy(), 15, 20, 60, pa=0.2)

    # 2D, big search that pushes window outside ref
    short2d = rng.integers(0, 256, size=(60, 8), dtype=np.uint8).astype(np.float32)
    run_case("2D radius>=ref", short2d, short2d.copy(), 0, 100, 20)

    # 2D, K=16 (matches detect_v2 default)
    big = rng.integers(0, 256, size=(969, 16), dtype=np.uint8).astype(np.float32)
    big_cur = np.roll(big, -6, axis=0)
    big_cur[-6:] = 0
    run_case("2D 969x16 shift=+6", big, big_cur[:-6].copy(), 6, 400, 100)

    # 2D, all-candidates-too-short (overlap < min_overlap) → check best is inf-safe
    run_case("2D all too short", ref2d, ref2d[:5].copy(), 0, 10, 50)

    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
