#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generate an ONNX model whose MatMuls all run on the NPU through
DynamicDispatch when compiled by hip-ep with HIPEP_USE_DYNAMIC_DISPATCH=1.

The model is a fan-out of float16 MatMuls, A [1, 1024] x W_i [1024, N], whose
results are concatenated into Y [1, count * N]. N must be a width DD's
combined_gemm has a kernel for (include/hip/dynamic_dispatch_support.h).
The Concat runs on the GPU.

Only standard ONNX operators are used, at an opset and IR version that the
CPU, DirectML and VitisAI execution providers load, so the same file can be
benchmarked with onnxruntime_perf_test on each of them.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

K = 1024
SUPPORTED_N = (192, 384)
OPSET = 17
IR_VERSION = 8


def build_model(count, n, seed):
    rng = np.random.default_rng(seed)
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, K])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, count * n])

    initializers, nodes, partials = [], [], []
    for i in range(count):
        # Scaled so each output stays well inside the float16 range.
        w = (rng.standard_normal((K, n)) / np.sqrt(K)).astype(np.float16)
        initializers.append(numpy_helper.from_array(w, f"W{i}"))
        partials.append(f"Y{i}")
        nodes.append(
            helper.make_node("MatMul", ["A", f"W{i}"], [f"Y{i}"], name=f"MatMul{i}")
        )
    nodes.append(helper.make_node("Concat", partials, ["Y"], name="Concat", axis=1))

    graph = helper.make_graph(nodes, "dd_matmul_fanout", [a], [y], initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="hip-ep make_dd_matmul_model",
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def verify_on_cpu(path, count, n):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    a = np.random.default_rng(1).random((1, K), dtype=np.float32)
    (y,) = sess.run(None, {"A": a.astype(np.float16)})
    assert y.shape == (1, count * n), y.shape
    assert np.isfinite(y).all(), "non-finite output"
    print(f"CPU check passed: Y {y.shape} {y.dtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dd_matmul_fanout.onnx")
    )
    parser.add_argument(
        "--count", type=int, default=8, help="number of MatMuls (default: 8)"
    )
    parser.add_argument(
        "-n",
        type=int,
        choices=SUPPORTED_N,
        default=384,
        help="MatMul output width (default: 384)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the onnxruntime CPU run"
    )
    args = parser.parse_args()

    model = build_model(args.count, args.n, args.seed)
    onnx.save(model, args.output)
    print(
        f"Wrote {args.output}: {args.count} x MatMul "
        f"[1, {K}] x [{K}, {args.n}] -> Concat [1, {args.count * args.n}], "
        f"opset {OPSET}, IR {IR_VERSION}"
    )
    if not args.no_verify:
        verify_on_cpu(args.output, args.count, args.n)


if __name__ == "__main__":
    main()
