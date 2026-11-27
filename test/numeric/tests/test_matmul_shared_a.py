#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""MatMul with one A shared by a batched B: [M, K] x [batch, K, N].

ONNX MatMul broadcasts the batch dims, so the result is [batch, M, N]. The
runtime has to read the same A for every batch element and step B by K * N.
A rank-2 A and an A with a unit batch dim both take that path.

The batch is static here. With a dynamic batch, the output shape comes from
the destination that the ONNX conversion builds, which these cases do not
exercise; the hip-to-llvm lit tests cover a dynamic B at the IR level.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# (M, K, N): a 1x1-convolution-like shape, a tiny one, and extents that are
# not multiples of the tile sizes.
SHAPES = [(64, 32, 49), (8, 16, 8), (5, 7, 3), (128, 96, 196)]
# A batch above M as well: a result buffer sized from M would be too small.
BATCHES = [2, 5, 9]
A_FORMS = {"rank2": lambda m, k: [m, k], "unit": lambda m, k: [1, m, k]}


def _model(a_shape, b_shape, dtype, a_const, rng):
    tp = np_to_onnx_type(dtype)
    out_shape = [b_shape[0], a_shape[-2], b_shape[-1]]
    a = rng.uniform(-1, 1, a_shape).astype(dtype)
    inputs = [helper.make_tensor_value_info("B", tp, b_shape)]
    inits = []
    if a_const:
        inits.append(numpy_helper.from_array(a, name="A"))
    else:
        inputs.insert(0, helper.make_tensor_value_info("A", tp, a_shape))
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    model = make_model_from_nodes(
        [node],
        inputs,
        [helper.make_tensor_value_info("Y", tp, out_shape)],
        initializers=inits,
    )
    return model, a


def _bf16_models(a_shape, b_shape):
    """Shared-A bf16 MatMul behind f32 inputs/outputs, plus an f32 reference.

    ORT Python cannot feed bf16 arrays; ORT CPU has no bf16 MatMul.
    """
    f32, bf16 = TensorProto.FLOAT, TensorProto.BFLOAT16
    out_shape = [b_shape[0], a_shape[-2], b_shape[-1]]
    A = helper.make_tensor_value_info("A", f32, a_shape)
    B = helper.make_tensor_value_info("B", f32, b_shape)
    Y = helper.make_tensor_value_info("Y", f32, out_shape)
    to_bf16 = [
        helper.make_node("Cast", ["A"], ["Ab"], to=bf16),
        helper.make_node("Cast", ["B"], ["Bb"], to=bf16),
    ]
    model = make_model_from_nodes(
        to_bf16
        + [
            helper.make_node("MatMul", ["Ab", "Bb"], ["Yb"]),
            helper.make_node("Cast", ["Yb"], ["Y"], to=f32),
        ],
        [A, B],
        [Y],
    )
    reference = make_model_from_nodes(
        to_bf16
        + [
            helper.make_node("Cast", ["Ab"], ["A32"], to=f32),
            helper.make_node("Cast", ["Bb"], ["B32"], to=f32),
            helper.make_node("MatMul", ["A32", "B32"], ["Y32"]),
            helper.make_node("Cast", ["Y32"], ["Yr"], to=bf16),
            helper.make_node("Cast", ["Yr"], ["Y"], to=f32),
        ],
        [A, B],
        [Y],
    )
    return model, reference


class TestMatMulSharedA:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize("batch", BATCHES)
    @pytest.mark.parametrize("form", list(A_FORMS))
    @pytest.mark.parametrize("m, k, n", SHAPES)
    def test_shared_a_constant(self, model_runner, m, k, n, form, batch, dtype):
        rng = np.random.default_rng(11)
        model, _ = _model(A_FORMS[form](m, k), [batch, k, n], dtype, True, rng)
        b = rng.uniform(-1, 1, [batch, k, n]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [b], reference="cpu")
        tol = 2e-3 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize("form", list(A_FORMS))
    def test_shared_a_runtime_input(self, model_runner, form, dtype):
        """A is a graph input rather than an initializer."""
        rng = np.random.default_rng(13)
        m, k, n, batch = 16, 24, 20, 3
        model, a = _model(A_FORMS[form](m, k), [batch, k, n], dtype, False, rng)
        b = rng.uniform(-1, 1, [batch, k, n]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        tol = 2e-3 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("batch", [2, 5])
    @pytest.mark.parametrize("form", list(A_FORMS))
    @pytest.mark.parametrize("m, k, n", SHAPES[:3])
    def test_shared_a_bf16(self, model_runner, m, k, n, form, batch):
        a_shape = A_FORMS[form](m, k)
        b_shape = [batch, k, n]
        model, reference = _bf16_models(a_shape, b_shape)
        rng = np.random.default_rng(17)
        a = rng.uniform(-1, 1, a_shape).astype(np.float32)
        b = rng.uniform(-1, 1, b_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(
            model, [a, b], reference="cpu", reference_model=reference
        )
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2)
