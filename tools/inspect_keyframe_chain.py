"""Quick inspector: prints per-keyframe chain_dy / cumulative_y from keyframes.json.

Run: python inspect_keyframe_chain.py <path_to_keyframes_json>
     (per-video convention: out/<video_basename>/keyframes.json)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: inspect_keyframe_chain.py <path_to_keyframes_json>",
              file=sys.stderr)
        return 2
    p = Path(sys.argv[1])
    d = json.loads(p.read_text())
    dys = d["dy_series"]
    kfs = d["keyframes"]
    dyn_h = d["video"]["dyn_h"]
    print(f"dyn_h={dyn_h}  n_keyframes={len(kfs)}")
    cum = 0
    prev_i = 0
    max_inter = 0
    for kf in kfs:
        i = kf["i"]
        inter = sum(dys[prev_i + 1 : i + 1])
        cum += inter
        overlap = dyn_h - abs(inter)
        flag = "GAP!" if abs(inter) > dyn_h else ""
        print(
            f"kf {kf['pause_index']:2d} i={i:4d} "
            f"chain_dy={inter:+5d} cum_y={cum:+6d} "
            f"overlap_with_prev={overlap:+5d} {flag} "
            f"drag={kf['drag_suspect']}"
        )
        if abs(inter) > max_inter:
            max_inter = abs(inter)
        prev_i = i
    print(f"max |chain_dy| = {max_inter}  vs dyn_h={dyn_h}  "
          f"(coverage: {'OK' if max_inter <= dyn_h else 'GAPS PRESENT'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
