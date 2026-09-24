#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the fused Q(op(DQ(x))) runtime entry points.

Two fusions produce an op whose kernel does dequantize, the operation, and
requantize in one pass:

    QuantizeLinear(Sigmoid(DequantizeLinear(X)))          -> hip.qsigmoid
    QuantizeLinear(LpNormalization(DequantizeLinear(X)))  -> hip.qlpnormalization

The IR rewrites are covered by LIT. What needs a GPU is the kernel behind each
one, because folding the requantize into the operation changes where rounding
happens: the unfused chain rounds once at the quantize, while a fused kernel
that reassociates the scaling rounds somewhere else and drifts by an LSB or
more. The comparisons below are one output LSB, which is tight enough to catch
that.

Both patterns require UINT16 on each end, per-tensor splat scales, and
constant zero points, so every model here is built that way; anything else
keeps the unfused path and would test nothing.

LpNormalization reaches the fused op indirectly. Its p=2 trailing-axis case is
decomposed into an RMS normalization with epsilon 0 and a scale of 1/sqrt(N),
which is the same function, and the fusion matches that. The cases below vary
N across and around the kernel's block width so a reduction that mishandles a
partial tile is visible.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

UINT16_MAX = 65535
# Both kernels reduce or map with a 256-thread block. Sizes on either side of
# it, and one that is not a multiple of it, expose a broken tail.
BLOCK = 256


def _make_qsigmoid_model(shape):
    """QuantizeLinear(Sigmoid(DequantizeLinear(X))) over UINT16.

    The input scale spans [-8, 8], which covers sigmoid's whole transition:
    past that the function is flat to within an output LSB and a wrong kernel
    would compare equal. The output maps [0, 1) onto the full UINT16 range, so
    an LSB of the comparison is an LSB of the real quantization grid.
    """
    x = helper.make_tensor_value_info("X", TensorProto.UINT16, shape)
    y = helper.make_tensor_value_info("Y", TensorProto.UINT16, shape)

    inits = [
        numpy_helper.from_array(np.array(16.0 / UINT16_MAX, np.float32), "Xscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Xzp"),
        numpy_helper.from_array(np.array(1.0 / UINT16_MAX, np.float32), "Yscale"),
        numpy_helper.from_array(np.array(0, np.uint16), "Yzp"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["X", "Xscale", "Xzp"], ["dqx"]),
        helper.make_node("Sigmoid", ["dqx"], ["sig"]),
        helper.make_node("QuantizeLinear", ["sig", "Yscale", "Yzp"], ["Y"]),
    ]
    return make_model_from_nodes(nodes, [x], [y], initializers=inits, opset=21)


def _make_qlpnorm_model(outer, norm_size):
    """QuantizeLinear(LpNormalization(DequantizeLinear(X))) over UINT16.

    L2 normalization is scale-invariant, so the input scale cancels and only
    the zero point decides where the data sits; 32768 centres it. A normalized
    row of N roughly-gaussian elements has entries near 1/sqrt(N), so the
    output scale is sized to put +-4/sqrt(N) at the rails.
    """
    shape = [outer, norm_size]
    x = helper.make_tensor_value_info("X", TensorProto.UINT16, shape)
    y = helper.make_tensor_value_info("Y", TensorProto.UINT16, shape)

    y_span = 8.0 / np.sqrt(norm_size)
    inits = [
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "Xscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Xzp"),
        numpy_helper.from_array(np.array(y_span / UINT16_MAX, np.float32), "Yscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Yzp"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["X", "Xscale", "Xzp"], ["dqx"]),
        helper.make_node("LpNormalization", ["dqx"], ["lp"], axis=-1, p=2),
        helper.make_node("QuantizeLinear", ["lp", "Yscale", "Yzp"], ["Y"]),
    ]
    return make_model_from_nodes(nodes, [x], [y], initializers=inits, opset=21)


def _uniform_u16(shape, seed=99):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, shape, dtype=np.uint16)


def _gaussian_u16(shape, seed=99):
    """Centred gaussian codes, so a normalized row is not dominated by rails."""
    rng = np.random.default_rng(seed)
    v = rng.normal(32768.0, 8000.0, shape)
    return np.clip(v, 0, UINT16_MAX).astype(np.uint16)


def _assert_not_saturated(out):
    """A clamped output agrees with the reference no matter what the kernel did."""
    pinned = np.count_nonzero((out == 0) | (out == UINT16_MAX))
    assert pinned < 0.01 * out.size, (
        f"{pinned}/{out.size} outputs saturated -- the scales chose a range the "
        "comparison cannot see through"
    )


class TestQSigmoid:
    @pytest.mark.parametrize(
        "shape",
        [[1, 1], [1, BLOCK], [1, BLOCK + 1], [4, 1024], [2, 5120]],
        ids=["single", "one_block", "block_plus_one", "batched", "hidden"],
    )
    def test_qsigmoid(self, model_runner, shape):
        model = _make_qsigmoid_model(shape)
        x = _uniform_u16(shape)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)

    def test_qsigmoid_saturating_input(self, model_runner):
        """Codes at both rails drive sigmoid to its asymptotes.

        The fused kernel clamps after requantizing rather than before, so a
        missing clamp wraps instead of pinning and the two are far apart.
        """
        shape = [1, 512]
        model = _make_qsigmoid_model(shape)
        x = np.zeros(shape, np.uint16)
        x[:, : shape[1] // 2] = UINT16_MAX
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)


class TestQLpNormalization:
    @pytest.mark.parametrize(
        "norm_size",
        [64, BLOCK, BLOCK + 1, 1024, 5120],
        ids=["sub_block", "one_block", "block_plus_one", "multi_block", "hidden"],
    )
    def test_qlpnormalization(self, model_runner, norm_size):
        """Row width across and around the block, including a partial tile."""
        shape = [4, norm_size]
        model = _make_qlpnorm_model(*shape)
        x = _gaussian_u16(shape)
        actual, expected = model_runner.run_sample(model, [x])
        _assert_not_saturated(actual[0])
        compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)

    def test_qlpnormalization_many_rows(self, model_runner):
        """One block per row, so the grid rather than the reduction is wide."""
        shape = [512, 1024]
        model = _make_qlpnorm_model(*shape)
        x = _gaussian_u16(shape)
        actual, expected = model_runner.run_sample(model, [x])
        _assert_not_saturated(actual[0])
        compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)
