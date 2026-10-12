#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the quantized convolution, on both sides of the QConv fusion.

The fusion is W4A16: per-tensor UINT16 activations, weights quantized per
output channel at 4 bits, and a UINT16 requantize. It matches

    QuantizeLinear(Conv(DequantizeLinear(X), DequantizeLinear(W)))

only for a 1x1 window, and lowers it to hip_qconv. A wider window declines the
fusion and reaches the patch-embed rewrite instead, which permutes and flattens
the window into a GEMM over dequantized operands. A 1x1 window covers hip_qconv
on both sides of the spatial_size it dispatches on, and a 14x14 patch window
covers the GEMM path. All are compared against ORT CPU.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_qconv_model(
    cin: int,
    cout: int,
    spatial_shape: tuple,
    kernel: tuple = (1, 1),
    stride: tuple = None,
    seed: int = 42,
):
    """Build QuantizeLinear(Conv(DQ(X), DQ(W))) with random quantized weights.

    Scales are chosen so the requantized output lands mid-range: the
    accumulator is a sum of ``cin * kh * kw`` random-signed terms, so it grows
    as sqrt of that, and a fixed weight scale would saturate the wide
    reductions and flatten the narrow ones. A saturated output compares equal
    for the wrong reason, which is exactly what this suite is meant to catch.
    """
    kh, kw = kernel
    sh, sw = stride if stride is not None else (1, 1)
    h, w = spatial_shape
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    rng = np.random.default_rng(seed)

    lo, hi = -8, 8
    weights = rng.integers(lo, hi, [cout, cin, kh, kw], dtype=np.int64)
    weight_zp = rng.integers(lo, hi, [cout], dtype=np.int64)

    # Term magnitude is |X - z_x| * |W - z_w| ~ (hi - lo)/4 * 32768, and the sum
    # of k random-signed terms grows as sqrt(k), roughly normally. Target one
    # standard deviation at 6000 of the 32768 available either side of the zero
    # point: the rails then sit past 5 sigma, so nothing saturates, while the
    # spread is still four decimal digits wide and an atol of 1 on it is a tight
    # comparison.
    k = cin * kh * kw
    term = (hi - lo) / 4.0 * 32768.0
    scale_mag = 6000.0 / (np.sqrt(k) * term)
    weight_scale = rng.uniform(0.5 * scale_mag, 1.5 * scale_mag, [cout]).astype(
        np.float32
    )

    x = helper.make_tensor_value_info("X", TensorProto.UINT16, [1, cin, h, w])
    y = helper.make_tensor_value_info("Y", TensorProto.UINT16, [1, cout, out_h, out_w])

    inits = [
        helper.make_tensor(
            "W", TensorProto.INT4, [cout, cin, kh, kw], weights.ravel().tolist()
        ),
        helper.make_tensor("Wzp", TensorProto.INT4, [cout], weight_zp.tolist()),
        numpy_helper.from_array(weight_scale, "Wscale"),
        # x_scale / y_scale is the only ratio the kernel sees; keep it at 1 so
        # the weight scale alone controls the output range.
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "Xscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Xzp"),
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "Yscale"),
        numpy_helper.from_array(np.array(32768, np.uint16), "Yzp"),
    ]

    # No bias: the fusion matches a four-operand hip.conv only, so a biased
    # convolution never reaches hip_qconv. The kernel takes a bias pointer and
    # its epilogue honours one, but nothing can currently pass a non-null value.
    nodes = [
        helper.make_node("DequantizeLinear", ["X", "Xscale", "Xzp"], ["dqx"]),
        helper.make_node("DequantizeLinear", ["W", "Wscale", "Wzp"], ["dqw"], axis=0),
        helper.make_node(
            "Conv",
            ["dqx", "dqw"],
            ["conv"],
            kernel_shape=[kh, kw],
            strides=[sh, sw],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("QuantizeLinear", ["conv", "Yscale", "Yzp"], ["Y"]),
    ]

    return make_model_from_nodes(nodes, [x], [y], initializers=inits, opset=21)


def _assert_not_saturated(out: np.ndarray):
    """A clamped output agrees with the reference no matter what the kernel did."""
    pinned = np.count_nonzero((out == 0) | (out == 65535))
    assert pinned < 0.005 * out.size, (
        f"{pinned}/{out.size} outputs saturated -- the scales chose a range the "
        "comparison cannot see through"
    )


def _run(model_runner, cin, cout, spatial_shape, **kwargs):
    model = _make_qconv_model(cin, cout, spatial_shape, **kwargs)
    rng = np.random.default_rng(99)
    x = rng.integers(0, 65536, [1, cin, *spatial_shape], dtype=np.uint16)
    actual, expected = model_runner.run_sample(model, [x])
    _assert_not_saturated(actual[0])
    # Both sides accumulate the same integers exactly; the only divergence is
    # the float requantize at the end, which is well under one output LSB.
    compare_outputs(actual, expected, atol=1, rtol=0, cos_threshold=0.9999)


class TestQConv:
    @pytest.mark.parametrize("spatial", [1, 128])
    def test_qconv_unit_kernel(self, model_runner, spatial):
        """1x1 window over a decoder-width reduction: fuses to hip_qconv.

        hip_qconv dispatches on spatial_size, so 1 and 128 land on the two
        different kernels behind it. Cin is the hidden width of a 40-layer
        decoder: one term reaches ~1e6, so an int32 accumulator saturates after
        about two thousand of them, which this is well past.
        """
        _run(model_runner, 5120, 64, (1, spatial))

    def test_qconv_patch_kernel(self, model_runner):
        """14x14 patch window: declines the fusion and lowers to a GEMM.

        The window is wider than 1x1, so hip_qconv cannot express it and the
        patch-embed rewrite takes the convolution instead.
        """
        _run(model_runner, 3, 64, (224, 224), kernel=(14, 14), stride=(14, 14))
