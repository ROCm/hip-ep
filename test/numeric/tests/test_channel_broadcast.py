#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Per-channel broadcast Add / Mul / Min / Max: one operand has the output's
shape, the other varies along a single axis (a per-channel bias or scale).

hip_elementwise_binary_bcast has a path for exactly that form. Each thread
handles one 16-byte vector of output, so it applies when the extent below the
channel axis is a multiple of the vector width (8 for f16 and bf16, 4 for
f32); other extents run on the general kernel. The cases cover both, in both
operand orders (Min and Max of a -0 and a +0 depend on the order), channel
axes other than 1, 3-D and 4-D operands, and channel extents that are and
are not powers of two.

The bf16 cases take f32 inputs and produce an f32 output, with Casts to and
from bf16 around the op, because ORT Python cannot feed bf16 arrays. The
ONNX import currently maps bf16 to f32, so until it imports bf16 these
cases run the f32 kernel; the kernel unit test elementwise_bcast covers the
bf16 kernel directly.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

OPS = ["Add", "Mul", "Min", "Max"]
NP_OPS = {
    "Add": np.add,
    "Mul": np.multiply,
    "Min": np.minimum,
    "Max": np.maximum,
}

# (output shape, channel operand shape, why)
CASES = {
    # Extent below the channel axis: 784, 3136 -- whole vectors in f16 and f32.
    "nchw_28x28": ([2, 24, 28, 28], [1, 24, 1, 1]),
    "nchw_56x56": ([1, 64, 56, 56], [1, 64, 1, 1]),
    # 196: whole f32 vectors, not whole f16 vectors.
    "nchw_14x14": ([2, 40, 14, 14], [1, 40, 1, 1]),
    # 49 is not a whole vector in any type: general kernel.
    "nchw_7x7": ([1, 1000, 7, 7], [1, 1000, 1, 1]),
    # Rank 3: [N, C, P] with a [1, C, 1] channel operand.
    "ncp_3136": ([3, 40, 3136], [1, 40, 1]),
    # The channel axis is not axis 1: the extent below it is 16.
    "axis2": ([2, 8, 24, 16], [1, 1, 24, 1]),
    # Channel extent that is a power of two.
    "pow2_channels": ([2, 64, 8, 8], [1, 64, 1, 1]),
    # Channel extent 1 on a non-unit tensor is no broadcast at all.
    "extent1": ([4, 1, 32, 32], [1, 1, 1, 1]),
}


def _tol(dtype):
    return (2e-3, 2e-3) if dtype == np.float16 else (1e-5, 1e-5)


def _model(op, shape, cshape, dtype, channel_first):
    tp = np_to_onnx_type(dtype)
    A = helper.make_tensor_value_info("a", tp, shape)
    C = helper.make_tensor_value_info("c", tp, cshape)
    Y = helper.make_tensor_value_info("y", tp, shape)
    ins = ["c", "a"] if channel_first else ["a", "c"]
    return make_model_from_nodes([helper.make_node(op, ins, ["y"])], [A, C], [Y])


class TestChannelBroadcast:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize(
        "channel_first", [False, True], ids=["dense_lhs", "dense_rhs"]
    )
    @pytest.mark.parametrize("op", OPS)
    @pytest.mark.parametrize("name", list(CASES))
    def test_channel_broadcast(self, model_runner, name, op, channel_first, dtype):
        shape, cshape = CASES[name]
        model = _model(op, shape, cshape, dtype, channel_first)
        rng = np.random.default_rng(sum(shape) + len(op))
        a = rng.standard_normal(shape).astype(dtype)
        c = rng.standard_normal(cshape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, c], reference="cpu")
        atol, rtol = _tol(dtype)
        compare_outputs(actual, expected, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "channel_first", [False, True], ids=["dense_lhs", "dense_rhs"]
    )
    @pytest.mark.parametrize("op", OPS)
    @pytest.mark.parametrize("name", ["nchw_28x28", "nchw_14x14", "ncp_3136", "axis2"])
    def test_channel_broadcast_bf16(self, model_runner, name, op, channel_first):
        """bf16 behind f32 inputs and outputs (see the module docstring)."""
        shape, cshape = CASES[name]
        f32, bf16 = TensorProto.FLOAT, TensorProto.BFLOAT16
        A = helper.make_tensor_value_info("a", f32, shape)
        C = helper.make_tensor_value_info("c", f32, cshape)
        Y = helper.make_tensor_value_info("y", f32, shape)
        to_bf16 = [
            helper.make_node("Cast", ["a"], ["ab"], to=bf16),
            helper.make_node("Cast", ["c"], ["cb"], to=bf16),
        ]
        ins = ["cb", "ab"] if channel_first else ["ab", "cb"]
        model = make_model_from_nodes(
            to_bf16
            + [
                helper.make_node(op, ins, ["yb"]),
                helper.make_node("Cast", ["yb"], ["y"], to=f32),
            ],
            [A, C],
            [Y],
        )
        ref_ins = ["c32", "a32"] if channel_first else ["a32", "c32"]
        reference = make_model_from_nodes(
            to_bf16
            + [
                helper.make_node("Cast", ["ab"], ["a32"], to=f32),
                helper.make_node("Cast", ["cb"], ["c32"], to=f32),
                helper.make_node(op, ref_ins, ["y32"]),
                helper.make_node("Cast", ["y32"], ["yr"], to=bf16),
                helper.make_node("Cast", ["yr"], ["y"], to=f32),
            ],
            [A, C],
            [Y],
        )
        rng = np.random.default_rng(sum(shape) + len(op) + 7)
        # Exactly representable in bf16: the casts then do not round.
        a = rng.standard_normal(shape).astype(ml_dtypes.bfloat16).astype(np.float32)
        c = rng.standard_normal(cshape).astype(ml_dtypes.bfloat16).astype(np.float32)
        actual, expected = model_runner.run_sample(
            model, [a, c], reference="cpu", reference_model=reference
        )
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape,cshape",
        [([2, 8, 16, 16], [1, 8, 1, 1]), ([2, 8, 7, 7], [1, 8, 1, 1])],
        ids=["vectors", "general"],
    )
    @pytest.mark.parametrize(
        "channel_first", [False, True], ids=["dense_lhs", "dense_rhs"]
    )
    @pytest.mark.parametrize("op", OPS)
    def test_channel_broadcast_nonfinite(
        self, model_runner, op, channel_first, shape, cshape
    ):
        """Infinities in either operand, and NaN for Add and Mul: same result
        as the reference, NaN positions included. Min and Max get no NaN: the
        kernel's Min and Max (shared by every path) do not propagate a NaN the
        way the reference does."""
        model = _model(op, shape, cshape, np.float32, channel_first)
        rng = np.random.default_rng(5)
        a = rng.standard_normal(shape).astype(np.float32)
        c = rng.standard_normal(cshape).astype(np.float32)
        a[1, 2, 4, 4] = np.inf
        a[1, 5, 0, 0] = -np.inf
        c[0, 6, 0, 0] = np.inf
        if op in ("Add", "Mul"):
            a[0, 1, 3, 3] = np.nan
            c[0, 3, 0, 0] = np.nan
        actual, expected = model_runner.run_sample(model, [a, c], reference="cpu")
        got, want = np.asarray(actual[0]), np.asarray(expected[0])
        assert got.shape == want.shape
        assert np.array_equal(np.isnan(got), np.isnan(want)), "NaN positions differ"
        finite = ~np.isnan(want)
        assert np.array_equal(got[finite], want[finite])
