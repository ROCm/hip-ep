#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Conv operation, focused on grouped / depthwise convolution.

The MIOpen weight descriptor must use the per-group input-channel count
(input_c / group), not input_c. For group == 1 the two are equal, but for
grouped (1 < group < C) and depthwise (group == C) convolutions, using input_c
describes the wrong filter shape and produces silently incorrect results. These
tests exercise all three regimes against the ORT CPU reference.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _np_dtype(dtype):
    return np.float32 if dtype == TensorProto.FLOAT else np.float16


def _make_conv_model(
    in_c: int,
    out_c: int,
    group: int,
    k: int = 3,
    spatial: "int | tuple[int, int]" = 8,
    seed: int = 7,
    dtype: int = TensorProto.FLOAT16,
    n: int = 1,
    stride: int = 1,
    pad: "int | None" = None,
    bias: bool = True,
):
    """Build a 2D Conv model with weight (and optional bias) initializers.

    input:   [n, in_c, h, w]
    weights: [out_c, in_c // group, k, k]  (initializer)
    bias:    [out_c]                       (initializer, omitted if bias=False)
    output:  [n, out_c, oh, ow]
    All tensors are `dtype` (TensorProto.FLOAT16 or FLOAT). `spatial` is the
    input height and width, or an (h, w) pair. `pad` defaults to k // 2, which
    keeps the size at stride 1.
    """
    assert in_c % group == 0 and out_c % group == 0
    h, w = (spatial, spatial) if isinstance(spatial, int) else spatial
    pad = k // 2 if pad is None else pad
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    np_dtype = _np_dtype(dtype)

    inp = helper.make_tensor_value_info("input", dtype, [n, in_c, h, w])
    out = helper.make_tensor_value_info("output", dtype, [n, out_c, oh, ow])

    rng = np.random.default_rng(seed)
    wt = rng.uniform(-0.1, 0.1, [out_c, in_c // group, k, k]).astype(np_dtype)
    inits = [numpy_helper.from_array(wt, name="weight")]
    names = ["input", "weight"]
    if bias:
        b = rng.uniform(-0.1, 0.1, [out_c]).astype(np_dtype)
        inits.append(numpy_helper.from_array(b, name="bias"))
        names.append("bias")

    node = helper.make_node(
        "Conv",
        names,
        ["output"],
        group=group,
        kernel_shape=[k, k],
        strides=[stride, stride],
        pads=[pad] * 4,
        dilations=[1, 1],
    )
    return make_model_from_nodes([node], [inp], [out], initializers=inits)


class TestConv:
    @pytest.mark.parametrize(
        "in_c,out_c,group",
        [
            (4, 8, 1),  # standard conv (group=1: input_c/group == input_c)
            (8, 16, 2),  # grouped conv (intermediate group)
            (8, 16, 4),  # grouped conv (intermediate group)
            (8, 8, 8),  # depthwise conv (group == channels)
            (16, 16, 16),  # depthwise conv
        ],
    )
    def test_conv_grouped(self, model_runner, in_c, out_c, group):
        """Conv across group regimes; weights are [out_c, in_c/group, k, k].

        Regression guard: using input_c instead of input_c/group for the MIOpen
        weight descriptor yields uncorrelated output (cosine ~0) for group > 1.
        """
        model = _make_conv_model(in_c, out_c, group)

        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, [1, in_c, 8, 8]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_conv3d_temporal(self, model_runner):
        """Rank-5 NCDHW conv: depth kernel 3, spatial kernel 1x1, pad 1 on D.

        Output depth stays 6:
        (6 + 1 + 1 - (3 - 1) - 1) / 1 + 1 = 6.
        """
        depth, height, width = 6, 4, 4
        channels = 4
        inp = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT16, [1, channels, depth, height, width]
        )
        out = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, [1, channels, depth, height, width]
        )
        rng = np.random.default_rng(7)
        w = rng.uniform(-0.1, 0.1, [channels, channels, 3, 1, 1]).astype(np.float16)
        b = rng.uniform(-0.1, 0.1, [channels]).astype(np.float16)
        node = helper.make_node(
            "Conv",
            ["input", "weight", "bias"],
            ["output"],
            group=1,
            kernel_shape=[3, 1, 1],
            strides=[1, 1, 1],
            pads=[1, 0, 0, 1, 0, 0],
            dilations=[1, 1, 1],
        )
        model = make_model_from_nodes(
            [node],
            [inp],
            [out],
            initializers=[
                numpy_helper.from_array(w, name="weight"),
                numpy_helper.from_array(b, name="bias"),
            ],
        )
        x = rng.uniform(-1.0, 1.0, [1, channels, depth, height, width]).astype(
            np.float16
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)


class TestConvSplitK:
    """Shapes whose tile grid is far short of the device: a long reduction over few
    output positions. The kernel cuts the reduction across blocks and sums the
    partial results in a second pass, so these check the partial sums, the bias
    (added once, in the reduce, or absent) and the ragged edges of the last slice.

    Every row has to split, or it only repeats the unsplit kernels' tests. They
    do on the 40-CU gfx1151 (checked from the plan arithmetic and from the
    kernels launched) and, for most rows, on smaller and larger devices too. The
    1x1 rows use a reduction that is not a multiple of 8, which keeps fp16 off
    its dedicated 1x1 kernel (that one never splits); the rows without a bias
    cover the reduce kernel's null-bias path in fp32 and fp16."""

    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
    @pytest.mark.parametrize(
        "n,in_c,hw,out_c,k,stride,group,bias",
        [
            (1, 512, 7, 512, 3, 1, 1, True),  # K = 4608 over 49 columns
            (2, 256, 14, 256, 3, 1, 1, True),  # batch > 1
            (1, 200, 9, 130, 3, 1, 1, True),  # ragged K and ragged M
            (1, 128, 14, 128, 3, 1, 2, False),  # grouped, no bias
            (1, 256, 28, 256, 3, 2, 1, True),  # stride 2
            (1, 1020, 14, 256, 1, 1, 1, False),  # 1x1, K % 8 != 0, no bias
            (1, 516, 7, 512, 1, 1, 1, True),  # 1x1, K % 8 != 0, narrow output
        ],
    )
    def test_conv_splitk(
        self, model_runner, dtype, n, in_c, hw, out_c, k, stride, group, bias
    ):
        model = _make_conv_model(
            in_c,
            out_c,
            group,
            k=k,
            spatial=hw,
            seed=11,
            dtype=dtype,
            n=n,
            stride=stride,
            bias=bias,
        )
        rng = np.random.default_rng(5)
        x = rng.uniform(-1.0, 1.0, [n, in_c, hw, hw]).astype(_np_dtype(dtype))
        actual, expected = model_runner.run_sample(model, [x])
        tol = 1e-3 if dtype == TensorProto.FLOAT else 2e-2
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.9999)


class TestConvScalarTiles:
    """fp32 shapes that run on the scalar tiles with a grid far short of the
    device, plus the narrow-Cout fallback. fp16 is not covered here: it takes
    the matrix cores, never the scalar tiles.

    Rows below 32 output channels per group used to fall back to the
    one-thread-per-output direct kernel; they now run on a 64x64 tile with dead
    rows, which must be zero-filled on load and skipped on store. The two rows
    at 32 and 40 channels make the block-count rule pick the 32x32 and the
    32x128 tile on any device with two or more units. The reduction
    K = in_c / group * k * k is deliberately not a multiple of the 16-tap slice
    in most rows, so the last slice is partial and exercises the masked filter
    loads and the invalid-K sentinel next to leading padding; K = 27 and 9 are
    shorter than two slices. The narrow rows are large enough (>= 1M
    multiply-accumulates) to leave the direct kernel behind."""

    @pytest.mark.parametrize(
        "n,in_c,hw,out_c,k,group",
        [
            (1, 3, 128, 3, 3, 1),  # 3 channels, K = 27
            (1, 256, 24, 16, 1, 1),  # 1x1, K = 256
            (1, 50, 32, 24, 3, 1),  # K = 450
            (2, 7, 28, 30, 3, 1),  # batch > 1, one row short of a tile, K = 63
            (1, 64, 40, 24, 3, 2),  # grouped: 12 channels per group, K = 288
            (1, 3, 9, 32, 3, 1),  # 32x32 tile, K = 27
            (1, 1, 8, 40, 3, 1),  # 32x128 tile, K = 9 (< one slice)
        ],
    )
    def test_conv_scalar_tiles(self, model_runner, n, in_c, hw, out_c, k, group):
        model = _make_conv_model(
            in_c,
            out_c,
            group,
            k=k,
            spatial=hw,
            seed=11,
            dtype=TensorProto.FLOAT,
            n=n,
        )
        rng = np.random.default_rng(9)
        x = rng.uniform(-1.0, 1.0, [n, in_c, hw, hw]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3, cos_threshold=0.9999)


class TestConv1x1Gemm:
    """fp16 1x1 convolutions with at least 128 input channels per group run as a
    GEMM with their own kernel. The accepted shapes cover each global-load width
    it uses (output plane a multiple of 8, of 4, or neither), a reduction that is
    not a multiple of its 64-tap step, ragged output channels, groups, batch and
    a missing bias. The last four rows sit just outside what it accepts (K below
    128, K not a multiple of 8, stride 2, padding 1) and run on the general
    kernel; they check that the neighbouring shapes are still computed
    correctly."""

    @pytest.mark.parametrize(
        "n,in_c,h,w,out_c,k,stride,pad,group,bias",
        [
            (1, 256, 16, 16, 64, 1, 1, 0, 1, True),  # plane 256: 16-byte loads
            (2, 192, 12, 12, 70, 1, 1, 0, 1, True),  # K = 192 = 3 steps, ragged M
            (1, 200, 8, 16, 130, 1, 1, 0, 1, True),  # K = 200, not a multiple of 64
            (1, 256, 10, 14, 128, 1, 1, 0, 1, True),  # plane 140: 8-byte loads
            (1, 512, 7, 7, 2048, 1, 1, 0, 1, True),  # plane 49: 2-byte loads
            (1, 256, 9, 9, 40, 1, 1, 0, 1, True),  # plane 81, ragged M
            (2, 256, 16, 8, 192, 1, 1, 0, 2, True),  # grouped
            (1, 128, 24, 24, 192, 1, 1, 0, 1, False),  # K at the minimum, no bias
            (1, 64, 16, 16, 128, 1, 1, 0, 1, True),  # K = 64 < 128: general kernel
            (1, 132, 16, 16, 64, 1, 1, 0, 1, True),  # K = 132, not a multiple of 8
            (1, 256, 16, 16, 64, 1, 2, 0, 1, True),  # stride 2
            (1, 256, 14, 14, 64, 1, 1, 1, 1, True),  # padding 1
        ],
    )
    def test_conv_1x1(
        self, model_runner, n, in_c, h, w, out_c, k, stride, pad, group, bias
    ):
        model = _make_conv_model(
            in_c,
            out_c,
            group,
            k=k,
            spatial=(h, w),
            seed=11,
            n=n,
            stride=stride,
            pad=pad,
            bias=bias,
        )
        rng = np.random.default_rng(3)
        x = rng.uniform(-1.0, 1.0, [n, in_c, h, w]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.9999)
