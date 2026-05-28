#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-tap bisect for the Qwen 3.5 vision encoder EP-vs-CPU cosine gap.

REWRITE NOTE (handoff): the previous version recompiled the whole model
for EVERY probe (~30 s/probe, 8 probes hung ~4+ min in). This version
adds ALL probe taps to a SINGLE model copy, compiles ONCE, runs CPU
once and EP once, then compares cosine per tap. Total cost ~2 fresh
compiles instead of N.

Usage:
  python test/qwen3_5_9b/bisect_qwen_cosine.py                 # default: 16 spread taps
  python test/qwen3_5_9b/bisect_qwen_cosine.py --n 32          # more taps
  python test/qwen3_5_9b/bisect_qwen_cosine.py --range 0.4,0.6 # narrow band
  python test/qwen3_5_9b/bisect_qwen_cosine.py --ops Gemm,MatMul,Add

Empirical baseline (2026-05-27, HEAD 3bfada9):
  Final output (image_features) on grid_small=[2,8,8]: cos=0.46 vs CPU.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

# register_morphizen_ep lives in test/python/conftest.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
MODEL_DIR = REPO_ROOT / "models" / "Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu"
MODEL = str(MODEL_DIR / "vision.onnx")
GRID = [2, 8, 8]


def make_inputs():
    t, h, w = GRID
    n = t * h * w
    rng = np.random.default_rng(0)
    return {
        "pixel_values": (rng.standard_normal((n, 1536)) * 0.1).astype(np.float16),
        "image_grid_thw": np.array([GRID], dtype=np.int64),
    }


def compare(a, b):
    """Returns (cos, max_abs_diff, ep_norm, cpu_norm, n_finite). cos==-1
    means "both all-zero" (don't flag as BAD)."""
    af = a.astype(np.float32).flatten()
    bf = b.astype(np.float32).flatten()
    mask = np.isfinite(af) & np.isfinite(bf)
    af, bf = af[mask], bf[mask]
    if af.size == 0:
        return -2.0, 0.0, 0.0, 0.0, 0
    ep_norm = float(np.linalg.norm(af))
    cpu_norm = float(np.linalg.norm(bf))
    max_abs_diff = float(np.max(np.abs(af - bf))) if af.size else 0.0
    denom = ep_norm * cpu_norm
    if denom == 0.0:
        # If both norms are 0 the outputs are bit-identical zeros.
        if ep_norm == 0.0 and cpu_norm == 0.0:
            return -1.0, 0.0, 0.0, 0.0, af.size
        # One side is zero, the other isn't — that IS a divergence.
        return 0.0, max_abs_diff, ep_norm, cpu_norm, af.size
    return float(np.dot(af, bf) / denom), max_abs_diff, ep_norm, cpu_norm, af.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="number of taps")
    ap.add_argument(
        "--ops",
        default="Gemm,MatMul,Add,Mul,LayerNormalization",
        help="comma-separated op types to probe",
    )
    ap.add_argument(
        "--range",
        default=None,
        help='topo-fraction range "lo,hi" (e.g. "0.0,0.5" to probe early half)',
    )
    ap.add_argument(
        "--ep-only", action="store_true", help="skip CPU baseline (faster smoke test)"
    )
    ap.add_argument(
        "--taps",
        default=None,
        help="explicit comma-separated tap output names (overrides --n)",
    )
    args = ap.parse_args()

    ops = set(args.ops.split(","))
    t0 = time.time()

    print(
        f"[+{time.time() - t0:.1f}s] Loading {MODEL} (model proto only)...", flush=True
    )
    m = onnx.load(MODEL, load_external_data=False)
    vi_map = {
        v.name: v
        for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input)
    }

    candidates = []
    for n in m.graph.node:
        if n.op_type in ops:
            for o in n.output:
                if o and o in vi_map:
                    candidates.append((n.op_type, n.name, o))

    print(
        f"[+{time.time() - t0:.1f}s] Candidate tap points (ops={sorted(ops)}): {len(candidates)}",
        flush=True,
    )

    if args.range:
        lo, hi = (float(x) for x in args.range.split(","))
        cand_lo = int(len(candidates) * lo)
        cand_hi = int(len(candidates) * hi)
        candidates = candidates[cand_lo:cand_hi]
        print(
            f"[+{time.time() - t0:.1f}s] Restricted to [{lo}, {hi}]: {len(candidates)}",
            flush=True,
        )

    if args.taps:
        wanted = set(args.taps.split(","))
        taps = [c for c in candidates if c[2] in wanted]
        missing = wanted - {c[2] for c in taps}
        if missing:
            print(
                f"WARNING: --taps not found in candidate set: {sorted(missing)}",
                flush=True,
            )
    else:
        n_taps = min(args.n, len(candidates))
        if n_taps == 0:
            print("ERROR: no candidates after filtering", flush=True)
            return 1
        # Evenly spread taps across topo order.
        idxs = sorted(
            set(int(len(candidates) * (i + 0.5) / n_taps) for i in range(n_taps))
        )
        taps = [candidates[i] for i in idxs]
    print(f"[+{time.time() - t0:.1f}s] Selected {len(taps)} tap points:", flush=True)
    for op, name, o in taps:
        print(f"    [{op:18s}] {name} -> {o}", flush=True)

    # Build ONE probe model with ALL tap outputs appended.
    print(f"[+{time.time() - t0:.1f}s] Building combined probe model...", flush=True)
    existing = {o.name for o in m.graph.output}
    tap_names = []
    for op, name, o in taps:
        if o not in existing:
            m.graph.output.append(vi_map[o])
            existing.add(o)
        tap_names.append(o)
    # Always include the model's original outputs in the comparison so we
    # confirm the end-to-end cosine matches the standalone test number.
    orig_outputs = [o.name for o in m.graph.output if o.name not in tap_names]
    for orig in orig_outputs:
        if orig not in tap_names:
            tap_names.append(orig)
            taps.append(("OUTPUT", "model_output", orig))
    probe_path = str(MODEL_DIR / "vision_probe_all.onnx")
    onnx.save(m, probe_path, save_as_external_data=False)
    print(f"[+{time.time() - t0:.1f}s] Probe model saved: {probe_path}", flush=True)

    inputs = make_inputs()

    # ── CPU baseline (single run, all taps) ────────────────────────────────
    cpu_outputs = {}
    if not args.ep_only:
        print(
            f"[+{time.time() - t0:.1f}s] Starting CPU baseline (single run)...",
            flush=True,
        )
        cpu_sess = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
        cpu_out_names = [o.name for o in cpu_sess.get_outputs()]
        cpu_results = cpu_sess.run(cpu_out_names, inputs)
        for name, arr in zip(cpu_out_names, cpu_results):
            cpu_outputs[name] = arr
        del cpu_sess
        print(
            f"[+{time.time() - t0:.1f}s] CPU baseline done: {len(cpu_outputs)} outputs",
            flush=True,
        )

    # ── EP run (single compile, single run, all taps) ──────────────────────
    print(f"[+{time.time() - t0:.1f}s] Registering MorphiZen EP...", flush=True)
    from conftest import register_morphizen_ep

    devices = register_morphizen_ep(REPO_ROOT)
    so = ort.SessionOptions()
    so.add_provider_for_devices(devices, {})
    print(
        f"[+{time.time() - t0:.1f}s] Starting EP compile + run (single pass)...",
        flush=True,
    )
    ep_sess = ort.InferenceSession(probe_path, sess_options=so)
    ep_out_names = [o.name for o in ep_sess.get_outputs()]
    ep_results = ep_sess.run(ep_out_names, inputs)
    ep_outputs = {name: arr for name, arr in zip(ep_out_names, ep_results)}
    del ep_sess
    print(f"[+{time.time() - t0:.1f}s] EP run done", flush=True)

    # ── Compare per tap ────────────────────────────────────────────────────
    print("", flush=True)
    print("=== Summary (topo order) ===", flush=True)
    for tap, op, name in [(t, o, n) for o, n, t in taps]:
        ep_o = ep_outputs.get(tap)
        cpu_o = cpu_outputs.get(tap)
        if ep_o is None:
            print(f"  MISSING_EP  {op:18s} {tap}", flush=True)
            continue
        if cpu_o is None and not args.ep_only:
            print(f"  MISSING_CPU {op:18s} {tap}", flush=True)
            continue
        if args.ep_only:
            print(
                f"  EP shape={list(ep_o.shape)} dtype={ep_o.dtype} {op:18s} {tap}",
                flush=True,
            )
            continue
        if ep_o.shape != cpu_o.shape:
            print(
                f"  SHAPE_MISMATCH ep={list(ep_o.shape)} cpu={list(cpu_o.shape)} "
                f"{op:18s} {tap}",
                flush=True,
            )
            continue
        cos, mad, en, cn, nf = compare(ep_o, cpu_o)
        if cos == -1.0:
            tag = "OK0"  # both all-zero (treat as match)
        elif cos >= 0.999:
            tag = "OK "
        elif cos >= 0.5:
            tag = "LOW"
        else:
            tag = "BAD"
        print(
            f"  [{tag}] cos={cos:+.4f} max_abs={mad:.3e} "
            f"ep_norm={en:.3e} cpu_norm={cn:.3e} n={nf} shape={list(ep_o.shape)} "
            f"{op:18s} {tap}",
            flush=True,
        )

    try:
        os.remove(probe_path)
    except OSError:
        pass
    print(f"[+{time.time() - t0:.1f}s] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
