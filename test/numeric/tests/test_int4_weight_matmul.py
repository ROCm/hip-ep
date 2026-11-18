#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""A MatMul whose B [K, N] is an int4 DequantizeLinear, as int4 QDQ LLMs export
their projections: per column (axis 1) or in blocks along K (axis 0).

With HIPDNN_EP_ROCMLIR=1 and autotuning, the rocMLIR GEMM kernel reads the
packed int4 weights itself, two values per byte, low nibble first; UINT4 is
zero-extended. The weights use every 4-bit value in both nibble positions, so a
swapped nibble order or a signed read of UINT4 fails the comparison.

The expected output is computed here, not by ORT CPU: ORT fuses a blocked
DequantizeLinear and MatMul into MatMulNBits, which quantizes the activations
and so differs from the exact (w - zp) * scale product.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_RNG = np.random.default_rng(13)
_ONNX_FLOAT = {np.float32: TensorProto.FLOAT, np.float16: TensorProto.FLOAT16}


def _int4_weights(k, n, form, dtype):
    """DequantizeLinear of int4 B [k, n], its initializers, and B in float64."""
    unsigned = form == "blk32u"
    lo, hi = (0, 16) if unsigned else (-8, 8)
    q = _RNG.integers(lo, hi, size=(k, n)).astype(np.int64)
    if form.startswith("col"):
        pshape, axis, block = [n], 1, 0
    else:
        pshape, axis, block = [k // 32, n], 0, 32
    scale = _RNG.uniform(0.002, 0.02, size=pshape).astype(dtype)
    with_zp = form in ("col", "blk32u")
    zp = _RNG.integers(lo, hi, size=pshape).astype(np.int64) if with_zp else None
    qtype = TensorProto.UINT4 if unsigned else TensorProto.INT4
    inits = [
        helper.make_tensor("w", qtype, [k, n], q.ravel().tolist()),
        numpy_helper.from_array(scale, "ws"),
    ]
    inputs = ["w", "ws"]
    if with_zp:
        inits.append(helper.make_tensor("wz", qtype, pshape, zp.ravel().tolist()))
        inputs.append("wz")
    attrs = {"axis": axis}
    if block:
        attrs["block_size"] = block
    node = helper.make_node("DequantizeLinear", inputs, ["wf"], **attrs)

    def expand(p):
        return np.repeat(p, block, axis=0) if block else p[None, :]

    zero = expand(zp) if with_zp else 0
    weights = (q - zero) * expand(scale.astype(np.float64))
    return node, inits, weights


def _model(m, k, n, form, dtype, bias):
    node, inits, weights = _int4_weights(k, n, form, dtype)
    nodes = [node, helper.make_node("MatMul", ["x", "wf"], ["mm" if bias else "y"])]
    b = None
    if bias:
        b = _RNG.uniform(-1.0, 1.0, size=(n,)).astype(dtype)
        inits.append(numpy_helper.from_array(b, "b"))
        nodes.append(helper.make_node("Add", ["mm", "b"], ["y"]))
    model = make_model_from_nodes(
        nodes,
        [helper.make_tensor_value_info("x", _ONNX_FLOAT[dtype], [m, k])],
        [helper.make_tensor_value_info("y", _ONNX_FLOAT[dtype], [m, n])],
        initializers=inits,
        opset=21,
    )
    return model, weights, b


class TestInt4WeightMatMul:
    @pytest.mark.parametrize("form", ["col", "col_nozp", "blk32", "blk32u"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
    @pytest.mark.parametrize(
        "m,k,n",
        [(4, 64, 32), (1, 896, 4864), (128, 896, 896)],
        ids=["small", "decode", "prefill"],
    )
    def test_matmul(self, model_runner, m, k, n, form, dtype):
        model, weights, _ = _model(m, k, n, form, dtype, bias=False)
        x = _RNG.uniform(-1.0, 1.0, size=(m, k)).astype(dtype)
        actual, _ = model_runner.run_sample(model, [x])
        expected = (x.astype(np.float64) @ weights).astype(dtype)
        tol = 1e-3 if dtype == np.float32 else 2e-2
        compare_outputs(actual, [expected], atol=tol, rtol=tol)

    @pytest.mark.parametrize("form", ["col", "blk32u"])
    def test_matmul_bias(self, model_runner, form):
        """MatMul + Add of a bias, which the importer makes one gemm."""
        model, weights, b = _model(1, 896, 896, form, np.float32, bias=True)
        x = _RNG.uniform(-1.0, 1.0, size=(1, 896)).astype(np.float32)
        actual, _ = model_runner.run_sample(model, [x])
        expected = (x.astype(np.float64) @ weights + b).astype(np.float32)
        compare_outputs(actual, [expected], atol=1e-3, rtol=1e-3)
