#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for the custom conv kernel that replaced MIOpen.

Forward Conv no longer calls MIOpen. `hip_conv` is the catch-all for everything
`ConvConversion.cpp` accepts, so it has to be right across the whole envelope,
not just on the shapes in the supported models. These tests pin it against the
ORT CPU reference.

Two things about the kernel drive what is worth testing:

* It dispatches between a tiled implicit-GEMM path and a one-thread-per-output
  direct path on the output-channel count per group. The two paths compute the
  same thing by different means, so both need coverage, and so does the
  boundary between them -- the tiled path is chosen at 64 output channels per
  group and at 24 for the narrow tile shape.
* The ABI carries only `pads_begin`. The trailing pad is supposed to be implied
  by the output extent, which is correct but not obvious, so asymmetric padding
  gets its own cases rather than riding on the symmetric ones.

The in-scope shapes from the supported models are here at full size (the audio
encoder 3x3 stride-2 pair) plus the Whisper 1D encoder front-end shapes, which
are the largest workload on this kernel and are not in any model in the
supported set.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_DTYPES = {
    TensorProto.FLOAT16: np.float16,
    TensorProto.FLOAT: np.float32,
}


def _conv_model(
    in_spatial,
    in_channels,
    out_channels,
    kernel,
    strides=None,
    pads=None,
    dilations=None,
    group=1,
    with_bias=True,
    batch=1,
    dtype=TensorProto.FLOAT16,
    seed=11,
):
    """A single Conv of arbitrary spatial rank, weights and bias as initializers.

    The output extents are derived from the attributes rather than passed in, so
    a wrong output shape cannot mask a wrong kernel.
    """
    np_dtype = _DTYPES[dtype]
    nd = len(in_spatial)
    strides = [1] * nd if strides is None else list(strides)
    pads = [0] * (2 * nd) if pads is None else list(pads)
    dilations = [1] * nd if dilations is None else list(dilations)

    out_spatial = [
        (in_spatial[i] + pads[i] + pads[i + nd] - (dilations[i] * (kernel[i] - 1) + 1))
        // strides[i]
        + 1
        for i in range(nd)
    ]
    assert all(o >= 1 for o in out_spatial), out_spatial

    X = helper.make_tensor_value_info("x", dtype, [batch, in_channels, *in_spatial])
    Y = helper.make_tensor_value_info("y", dtype, [batch, out_channels, *out_spatial])

    rng = np.random.default_rng(seed)
    # Scale by 1/sqrt(fan_in): the Whisper shapes contract over 3840 terms, and
    # a unit-variance fp16 reference would blow the tolerance for reasons that
    # have nothing to do with the kernel.
    fan_in = (in_channels // group) * int(np.prod(kernel))
    w = (
        rng.standard_normal((out_channels, in_channels // group, *kernel))
        / np.sqrt(fan_in)
    ).astype(np_dtype)

    inputs = ["x", "w"]
    initializers = [numpy_helper.from_array(w, name="w")]
    if with_bias:
        b = (rng.standard_normal((out_channels,)) * 0.1).astype(np_dtype)
        inputs.append("b")
        initializers.append(numpy_helper.from_array(b, name="b"))

    node = helper.make_node(
        "Conv",
        inputs,
        ["y"],
        kernel_shape=list(kernel),
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=initializers, opset=17)


def _sample(shape, seed, dtype=np.float16):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def _check(model_runner, model, x, dtype=np.float16):
    actual, expected = model_runner.run_sample(model, [x])
    if dtype is np.float32:
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4, cos_threshold=0.99999)
    else:
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)


class TestConvInScopeShapes:
    """The shapes this kernel exists to serve, at their real sizes."""

    def test_audio_conv0_1_to_128(self, model_runner):
        """gemma-4 E2B/E4B audio encoder, first conv: 1 -> 128, 3x3 stride 2.

        C_in == 1 makes the contraction 9 elements long, so this is bound by the
        12.3M-element output write rather than by arithmetic -- the opposite
        regime from every other in-scope shape.
        """
        model = _conv_model(
            in_spatial=[3000, 128],
            in_channels=1,
            out_channels=128,
            kernel=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        )
        _check(model_runner, model, _sample((1, 1, 3000, 128), 0xA00))

    def test_audio_conv1_128_to_32(self, model_runner):
        """gemma-4 E2B/E4B audio encoder, second conv: 128 -> 32, 3x3 stride 2.

        32 output channels per group is exactly the narrow-tile case: it is
        below the 64-row tile and would otherwise waste half of one.
        """
        model = _conv_model(
            in_spatial=[1500, 64],
            in_channels=128,
            out_channels=32,
            kernel=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        )
        _check(model_runner, model, _sample((1, 128, 1500, 64), 0xA01))

    def test_whisper_l0_128_to_1280(self, model_runner):
        """Whisper encoder front-end conv1: 128 -> 1280, k=3 stride 1.

        Rank-3, so it rides the NCL -> NC1L view. 2.9 GFLOP.
        """
        model = _conv_model(
            in_spatial=[3000],
            in_channels=128,
            out_channels=1280,
            kernel=[3],
            strides=[1],
            pads=[1, 1],
        )
        _check(model_runner, model, _sample((1, 128, 3000), 0xA02))

    def test_whisper_l1_1280_to_1280_stride2(self, model_runner):
        """Whisper encoder front-end conv2: 1280 -> 1280, k=3 stride 2.

        14.7 GFLOP and a 3840-element contraction -- the largest problem this
        kernel sees, and the one where fp32 accumulation actually matters.
        """
        model = _conv_model(
            in_spatial=[3000],
            in_channels=1280,
            out_channels=1280,
            kernel=[3],
            strides=[2],
            pads=[1, 1],
        )
        _check(model_runner, model, _sample((1, 1280, 3000), 0xA03))

    @pytest.mark.parametrize("out_c", [384, 512])
    def test_whisper_small_variants(self, model_runner, out_c):
        """The narrower Whisper front-ends: 80 -> 384/512, k=3 stride 1."""
        model = _conv_model(
            in_spatial=[3000],
            in_channels=80,
            out_channels=out_c,
            kernel=[3],
            strides=[1],
            pads=[1, 1],
        )
        _check(model_runner, model, _sample((1, 80, 3000), 0xA04 + out_c))


class TestConvPathSelection:
    """Both dispatch paths, and the boundary between them.

    The kernel picks the tiled implicit-GEMM at 64 output channels per group
    (and a narrower tile at 24), falling back to one-thread-per-output below
    that. A shape either side of each threshold has to give the same answer.
    """

    @pytest.mark.parametrize("out_c", [1, 4, 16, 23, 24, 32, 63, 64, 65, 128])
    def test_output_channel_sweep(self, model_runner, out_c):
        model = _conv_model(
            in_spatial=[24, 24],
            in_channels=16,
            out_channels=out_c,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
        )
        _check(model_runner, model, _sample((1, 16, 24, 24), 0xB00 + out_c))

    @pytest.mark.parametrize("n_out", [1, 7, 63, 64, 65, 127, 128, 129])
    def test_output_position_sweep(self, model_runner, n_out):
        """Output position counts around the tile width, where the column
        guards decide whether the tail of a tile is written or not."""
        model = _conv_model(
            in_spatial=[n_out + 2],
            in_channels=8,
            out_channels=64,
            kernel=[3],
        )
        _check(model_runner, model, _sample((1, 8, n_out + 2), 0xC00 + n_out))

    @pytest.mark.parametrize("k_in", [1, 15, 16, 17, 31, 33])
    def test_contraction_extent_sweep(self, model_runner, k_in):
        """Input channel counts around the BK=16 reduction slice: a contraction
        that is not a multiple of the slice must zero-fill the tail rather than
        read past K."""
        model = _conv_model(
            in_spatial=[40],
            in_channels=k_in,
            out_channels=64,
            kernel=[1],
        )
        _check(model_runner, model, _sample((1, k_in, 40), 0xD00 + k_in))


class TestConvPadding:
    """Only pads_begin is in the ABI; the trailing pad is carried by the output
    extent. These cases are what makes that claim testable."""

    @pytest.mark.parametrize(
        "pads",
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 0, 0, 2],  # asymmetric on both axes, opposite directions
            [0, 2, 3, 0],
            [3, 3, 0, 0],  # begin-only, i.e. causal in 2D
            [0, 0, 3, 3],  # end-only: pads_begin is all zero and the extra
            #               output positions come from out_d alone
        ],
    )
    def test_asymmetric_pads_2d(self, model_runner, pads):
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=8,
            out_channels=64,
            kernel=[4, 4],
            pads=pads,
        )
        _check(model_runner, model, _sample((1, 8, 16, 16), 0xE00 + sum(pads)))

    @pytest.mark.parametrize("pads", [[0, 0], [2, 2], [4, 0], [0, 4], [1, 3]])
    def test_asymmetric_pads_1d(self, model_runner, pads):
        model = _conv_model(
            in_spatial=[50],
            in_channels=8,
            out_channels=64,
            kernel=[5],
            pads=pads,
        )
        _check(model_runner, model, _sample((1, 8, 50), 0xE10 + sum(pads)))

    def test_pad_wider_than_input(self, model_runner):
        """Padding larger than the input extent, so some output positions read
        nothing but zeros."""
        model = _conv_model(
            in_spatial=[2],
            in_channels=8,
            out_channels=64,
            kernel=[5],
            pads=[4, 4],
        )
        _check(model_runner, model, _sample((1, 8, 2), 0xE20))


class TestConvGrouped:
    """Grouped and depthwise. Every group has its own contraction range, so a
    group index dropped anywhere -- input channel base, filter row base, output
    channel base -- shows up here and nowhere else."""

    @pytest.mark.parametrize(
        "in_c,out_c,group",
        [
            (16, 16, 16),  # depthwise
            (16, 32, 16),  # depthwise with a channel multiplier
            (64, 64, 2),   # wide groups: each group still fills a 32-row tile
            (128, 128, 2),
            (128, 256, 4),
            (12, 18, 3),   # group divides neither channel count evenly by 2
        ],
    )
    def test_grouped_2d(self, model_runner, in_c, out_c, group):
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=in_c,
            out_channels=out_c,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
            group=group,
        )
        _check(model_runner, model, _sample((1, in_c, 16, 16), 0xF00 + out_c))

    def test_depthwise_1d_symmetric(self, model_runner):
        """Rank-3 depthwise with symmetric pads. Causal pads would be claimed by
        the fused causal kernel instead, so this stays on the general path."""
        model = _conv_model(
            in_spatial=[128],
            in_channels=64,
            out_channels=64,
            kernel=[5],
            pads=[2, 2],
            group=64,
        )
        _check(model_runner, model, _sample((1, 64, 128), 0xF10))


class TestConvStrideDilation:
    @pytest.mark.parametrize("stride", [1, 2, 3, 5])
    def test_stride_1d(self, model_runner, stride):
        model = _conv_model(
            in_spatial=[97],
            in_channels=16,
            out_channels=64,
            kernel=[3],
            strides=[stride],
            pads=[1, 1],
        )
        _check(model_runner, model, _sample((1, 16, 97), 0x1100 + stride))

    @pytest.mark.parametrize("dil", [1, 2, 3])
    def test_dilation_1d(self, model_runner, dil):
        """Rank-3 dilation. The NCL -> NC1L view carries the 1D dilation onto
        the W axis; before this kernel that view forced dilation 1 and the
        conversion declined anything else outright.
        """
        model = _conv_model(
            in_spatial=[64],
            in_channels=16,
            out_channels=64,
            kernel=[3],
            dilations=[dil],
            pads=[dil, dil],
        )
        _check(model_runner, model, _sample((1, 16, 64), 0x1110 + dil))

    @pytest.mark.parametrize("dil", [[1, 2], [2, 1], [2, 3]])
    def test_dilation_2d(self, model_runner, dil):
        model = _conv_model(
            in_spatial=[20, 20],
            in_channels=16,
            out_channels=64,
            kernel=[3, 3],
            dilations=dil,
            pads=[dil[0], dil[1], dil[0], dil[1]],
        )
        _check(model_runner, model, _sample((1, 16, 20, 20), 0x1120 + sum(dil)))

    def test_stride_and_dilation_together(self, model_runner):
        model = _conv_model(
            in_spatial=[40, 40],
            in_channels=32,
            out_channels=64,
            kernel=[3, 3],
            strides=[2, 2],
            dilations=[2, 2],
            pads=[2, 2, 2, 2],
        )
        _check(model_runner, model, _sample((1, 32, 40, 40), 0x1130))

    def test_asymmetric_kernel(self, model_runner):
        """k_h != k_w with different strides per axis: catches a transposed
        spatial-axis mapping, which square kernels hide."""
        model = _conv_model(
            in_spatial=[24, 40],
            in_channels=16,
            out_channels=64,
            kernel=[3, 5],
            strides=[1, 2],
            pads=[1, 2, 1, 2],
        )
        _check(model_runner, model, _sample((1, 16, 24, 40), 0x1140))


class TestConvMisc:
    def test_batched(self, model_runner):
        """Batch is the outer grid axis; a dropped batch stride writes every
        image on top of the first."""
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=16,
            out_channels=64,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
            batch=4,
        )
        _check(model_runner, model, _sample((4, 16, 16, 16), 0x1200))

    def test_batched_grouped(self, model_runner):
        """Batch and group are packed into the same grid axis, so this is where
        an unpacking mistake shows."""
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=64,
            out_channels=128,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
            group=2,
            batch=3,
        )
        _check(model_runner, model, _sample((3, 64, 16, 16), 0x1201))

    @pytest.mark.parametrize("out_c", [16, 64])
    def test_no_bias(self, model_runner, out_c):
        """Bias is fused into the accumulator, so a null bias has to be handled
        inside the kernel rather than by skipping a second pass. Both dispatch
        paths do it, hence both channel counts."""
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=16,
            out_channels=out_c,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
            with_bias=False,
        )
        _check(model_runner, model, _sample((1, 16, 16, 16), 0x1210 + out_c))

    @pytest.mark.parametrize("out_c", [16, 64])
    def test_fp32(self, model_runner, out_c):
        model = _conv_model(
            in_spatial=[16, 16],
            in_channels=16,
            out_channels=out_c,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
            dtype=TensorProto.FLOAT,
        )
        _check(
            model_runner,
            model,
            _sample((1, 16, 16, 16), 0x1220 + out_c, dtype=np.float32),
            dtype=np.float32,
        )

    def test_1x1_conv(self, model_runner):
        """k=1 everywhere: a pure channel mix, and the degenerate case for the
        kernel-tap decode."""
        model = _conv_model(
            in_spatial=[32, 32],
            in_channels=64,
            out_channels=128,
            kernel=[1, 1],
        )
        _check(model_runner, model, _sample((1, 64, 32, 32), 0x1230))

    def test_kernel_covers_input(self, model_runner):
        """Output spatial extent 1 on both axes: the reduction is the whole
        input plane and there is exactly one output position per channel."""
        model = _conv_model(
            in_spatial=[7, 7],
            in_channels=32,
            out_channels=64,
            kernel=[7, 7],
        )
        _check(model_runner, model, _sample((1, 32, 7, 7), 0x1240))

    def test_deep_contraction(self, model_runner):
        """A 4608-element contraction in fp16. Accumulating in fp16 rather than
        float loses roughly a digit here, so this is the case that pins the
        accumulator type."""
        model = _conv_model(
            in_spatial=[12, 12],
            in_channels=512,
            out_channels=64,
            kernel=[3, 3],
            pads=[1, 1, 1, 1],
        )
        _check(model_runner, model, _sample((1, 512, 12, 12), 0x1250))
