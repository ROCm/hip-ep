#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Check the narrow-N decode GEMV against an fp64 reference, in isolation.

A whole-model logit comparison cannot separate "my kernel indexes B wrongly"
from "my kernel sums in a different order than the GEMM library did". Both show
up as a small cosine drop, and for a router feeding a top-k the drop is
amplified by whichever token sits near an expert tie. So test the operation on
its own: build a single-MatMul graph at the router's shape, run it with the flag
off and with it on, and score both against the same fp64 reference.

The pass condition is relative, not absolute. Neither arm can beat fp16
accumulation into an fp16 output, so both will differ from fp64; what matters is
that the flagged path is no further away than the unflagged one. An indexing bug
misses by orders of magnitude. A reassociation lands on the same error scale.

The flag is latched on first read, so one process can only measure one state.
Run with --state off, then --state on, then --compare.

    python gemv_check.py --bin <arms>\\bin --state off
    python gemv_check.py --bin <arms>\\bin --state on
    python gemv_check.py --compare
"""

import argparse
import os

import numpy as np


def build_model(path, N, K):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    # Scaled by 1/sqrt(K) so the accumulator stays well inside fp16 range at the
    # output, which keeps the comparison about reduction order rather than
    # about overflow.
    w = (rng.standard_normal((K, N)) * (1.0 / np.sqrt(K))).astype(np.float16)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "W"], ["C"])],
        "router_gemv",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, K])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT16, [1, N])],
        [numpy_helper.from_array(w, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, path)
    return w


def measure(args):
    import onnxruntime as ort

    path = os.path.join(args.work, f"gemv_{args.n}x{args.k}.onnx")
    w = build_model(path, args.n, args.k)
    rng = np.random.default_rng(1)
    a = (rng.standard_normal((1, args.k)) * 0.5).astype(np.float16)

    if "AMDGPU" not in ort.get_available_providers():
        ort.register_execution_provider_library(
            "AMDGPU", os.path.join(args.bin, "amdgpu-ep.dll"))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "AMDGPU"]
    if not devices:
        raise SystemExit("AMDGPU EP registered but exposes no device")

    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    # A plugin EP is selected by device, not by the providers= list: that list
    # only knows the EPs built into the wheel, so naming "AMDGPU" there silently
    # falls back to CPU. "profile: hip" matches what the model exports request.
    so.add_provider_for_devices(devices, {"profile": "hip"})
    sess = ort.InferenceSession(path, so)
    got = np.array(sess.run(None, {"A": a})[0]).astype(np.float64).ravel()

    ref = (a.astype(np.float64) @ w.astype(np.float64)).ravel()
    np.savez(os.path.join(args.work, f"gemv_{args.state}.npz"), got=got, ref=ref)
    print(f"{args.state}: 1x{args.k} @ {args.k}x{args.n}, "
          f"max|err| = {np.abs(got - ref).max():.6f}")


def compare(work):
    off = np.load(os.path.join(work, "gemv_off.npz"))
    on = np.load(os.path.join(work, "gemv_on.npz"))
    ref = off["ref"]
    if not np.array_equal(ref, on["ref"]):
        raise SystemExit("the two runs used different inputs; rerun both")

    def score(v):
        err = np.abs(v - ref)
        cos = float(np.dot(v, ref) / (np.linalg.norm(v) * np.linalg.norm(ref)))
        return cos, err.max(), float(np.sqrt(np.mean(err ** 2)))

    c_off, m_off, r_off = score(off["got"])
    c_on, m_on, r_on = score(on["got"])
    # One ulp of fp16 at the output's own magnitude: the smallest difference the
    # output type can even express, and therefore the scale any correct
    # reassociation lands on.
    ulp = float(np.spacing(np.float16(np.abs(ref).max())))

    print(f"{'arm':>6} {'cosine vs fp64':>18} {'max abs err':>14} {'rms err':>12}")
    print(f"{'off':>6} {c_off:18.10f} {m_off:14.6f} {r_off:12.6f}")
    print(f"{'on':>6} {c_on:18.10f} {m_on:14.6f} {r_on:12.6f}")
    print(f"\nfp16 ulp at max output: {ulp:.6f}")
    print(f"arms differ by         : {np.abs(on['got'] - off['got']).max():.6f} "
          f"({np.abs(on['got'] - off['got']).max() / ulp:.1f} ulp)")

    # Both arms quantise to the same fp16 output, so the flagged path is correct
    # if it is no worse against fp64 than the library path -- allowing a couple
    # of ulp of slack for the different summation order.
    ok = m_on <= m_off + 4 * ulp
    print("\nGEMV OK: reassociation only, no indexing error" if ok
          else "\nGEMV FAIL: flagged path is materially further from the reference")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", help="directory holding amdgpu-ep.dll")
    ap.add_argument("--state", choices=("off", "on"))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--n", type=int, default=128, help="router N (num_experts)")
    ap.add_argument("--k", type=int, default=2048, help="router K (hidden)")
    ap.add_argument("--work", default=os.environ.get("TEMP", "."))
    args = ap.parse_args()

    if args.compare:
        raise SystemExit(compare(args.work))
    if not args.bin or not args.state:
        ap.error("--bin and --state are required unless --compare")
    measure(args)


if __name__ == "__main__":
    main()
