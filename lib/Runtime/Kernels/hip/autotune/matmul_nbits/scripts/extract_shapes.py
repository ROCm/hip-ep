#!/usr/bin/env python3
"""Enumerate every MatMulNBits shape in a set of ONNX models.

The exact-match tier of the LUT is only as good as its shape list, and the
shapes cannot be derived reliably from a HuggingFace config: whether QKV is
fused, how far a pruned lm_head was cut, and how MoE experts are laid out are
all decisions the exporter made, not the architecture. So read them out of the
graphs themselves.

MatMulNBits (com.microsoft) carries K, N, bits and block_size as node
attributes, so only the graph file is needed -- external weight files are never
opened.

QMoE (com.microsoft) has to be read differently, and it is read here because a
MoE model's expert GEMMs are invisible otherwise: its weights are 3-D
initialisers rather than node attributes, so a graph walk that keys on
MatMulNBits alone never sees them. They still reach hip_matmul_nbits at runtime
(wrap_qmoe calls it per expert), and because the LUT's nearest-neighbour search
has no distance cap, an absent shape does not miss -- it silently borrows a
config measured at some other (N, K). On a gpt-oss-120b proxy that cost ~10% of
prefill: every expert GEMM resolved Nearest from (5120, 2880) or (2816, 2112),
and tuning the real shapes online instead was worth -20 ms of 202 ms at 2048
tokens. Dims alone are needed, so external data still stays closed.

Usage:
    python extract_shapes.py --models <dir> --out shapes.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import onnx


ATTRS = ("K", "N", "bits", "block_size", "accuracy_level")

# QMoE input positions, as (weights, scales, zero_points) triples: fc1, fc2 and
# the fc3 that an unfused SwiGLU export splits the gate off into. Reading N and K
# from the *scales* tensor rather than the packed weights avoids depending on the
# nibble packing: scales are [num_experts, N, K / block_size].
QMOE_WEIGHT_INPUTS = ((2, 3, 11), (5, 6, 12), (8, 9, 13))


def _matmul_nbits_shapes(node):
    """Yield (K, N, bits, block_size, has_zp) for one MatMulNBits node."""
    a = {p.name: p.i for p in node.attribute if p.name in ATTRS}
    if "K" not in a or "N" not in a:
        return
    # inputs: A, B, scales, [zero_points], [g_idx], [bias]
    has_zp = len(node.input) > 3 and node.input[3] != ""
    yield (a["K"], a["N"], a.get("bits", 4), a.get("block_size", 128), has_zp)


def _qmoe_shapes(node, init_dims):
    """Yield (K, N, bits, block_size, has_zp) for one QMoE node's expert GEMMs."""
    a = {p.name: p.i for p in node.attribute}
    bits = a.get("expert_weight_bits", 4)
    block_size = a.get("block_size", 32)
    if block_size <= 0:
        return
    for w_idx, s_idx, zp_idx in QMOE_WEIGHT_INPUTS:
        if s_idx >= len(node.input) or not node.input[s_idx]:
            continue
        scales = init_dims.get(node.input[s_idx])
        if not scales or len(scales) != 3:
            continue
        _, n, groups = scales
        # One GEMM per expert at runtime, so the expert count is a launch-count
        # property and not part of the shape the tuner keys on.
        has_zp = (zp_idx < len(node.input) and bool(node.input[zp_idx])
                  and node.input[zp_idx] in init_dims)
        yield (groups * block_size, n, bits, block_size, has_zp)


def node_shapes(model_path: Path):
    """Yield (K, N, bits, block_size, has_zp) for each quantised GEMM shape."""
    # load_external_data=False keeps this to the graph file; the .onnx.data
    # blobs are hundreds of GB across the model set and carry nothing we need.
    # Initializer *dims* survive that, which is all QMoE needs.
    model = onnx.load(str(model_path), load_external_data=False)
    init_dims = {t.name: list(t.dims) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            yield from _matmul_nbits_shapes(node)
        elif node.op_type == "QMoE":
            yield from _qmoe_shapes(node, init_dims)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="directory holding one subdirectory per model")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--bits", type=int, default=None,
                    help="keep only this bit width (e.g. 4)")
    args = ap.parse_args()

    root = Path(args.models)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1

    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        graphs = sorted(model_dir.rglob("*.onnx"))
        if not graphs:
            continue
        seen = Counter()
        for g in graphs:
            try:
                for shape in node_shapes(g):
                    seen[shape] += 1
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  !! {g.name}: {exc}", file=sys.stderr)
        if not seen:
            print(f"{model_dir.name}: no MatMulNBits or QMoE nodes")
            continue
        kept = 0
        for (k, n, bits, bs, has_zp), count in sorted(seen.items()):
            if args.bits is not None and bits != args.bits:
                continue
            rows.append({
                "model": model_dir.name, "K": k, "N": n, "bits": bits,
                "block_size": bs, "has_zp": int(has_zp), "node_count": count,
            })
            kept += 1
        print(f"{model_dir.name}: {kept} distinct shapes "
              f"({sum(seen.values())} nodes)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "K", "N", "bits",
                                          "block_size", "has_zp",
                                          "node_count"])
        w.writeheader()
        w.writerows(rows)

    distinct = {(r["K"], r["N"], r["block_size"], r["has_zp"]) for r in rows}
    print(f"\n{len(rows)} rows, {len(distinct)} distinct (K,N,block_size,zp) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
