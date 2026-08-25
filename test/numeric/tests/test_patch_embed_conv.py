#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for the patch-embed Conv -> Gemm route.

A Conv whose stride equals its kernel, with no padding, no dilation and
``group == 1``, partitions the input spatial volume into disjoint patches and
contracts each patch against every filter. ``PatchEmbedConvToGemm`` rewrites it
into Reshape/Transpose/Gemm so it lands on the existing hipBLASLt path and never
reaches a Conv runtime at all.

Two shapes in the supported model set take this route:

* Qwen3.5/3.6/3.8 ``vision.onnx``: rank-5, kernel == input spatial, so every
  output spatial dim is 1 and there is exactly one patch per batch element. The
  flatten is then a pure reshape and no transpose is emitted. This half is a
  regression test -- that route predates this change.
* gemma3-4b ``vision.onnx``: rank-4, 14x14 stride 14 over 896x896, so 64x64
  patches per image. 73% of all Conv FLOPs in the supported set. Multiple
  patches per batch element means the gather is a genuine permutation, which is
  where the row ordering can go wrong: the GEMM row for patch (oh, ow) has to
  be laid out channel-major then kernel-major to match the flattened weight.

The guard cases (overlapping stride, padding, ragged tiling, grouped) are not a
patch partition, so they must stay on the generic conv path. They are here so a
guard that wrongly lets one through fails loudly instead of silently returning
the wrong numbers.
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


def _patch_conv_model(
    in_spatial,
    kernel,
    out_channels,
    in_channels=3,
    strides=None,
    pads=None,
    dilations=None,
    group=1,
    with_bias=True,
    batch=1,
    dtype=TensorProto.FLOAT16,
    seed=7,
    auto_pad=None,
):
    """A single Conv of arbitrary spatial rank, with weight+bias initializers.

    ``strides`` defaults to ``kernel``, which is the patch-partition case. The
    output spatial extents are derived rather than passed so a bad guard cannot
    be masked by a hand-written output shape.
    """
    np_dtype = _DTYPES[dtype]
    nd = len(in_spatial)
    strides = list(kernel) if strides is None else list(strides)
    pads = [0] * (2 * nd) if pads is None else list(pads)
    dilations = [1] * nd if dilations is None else list(dilations)

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        out_spatial = [-(-in_spatial[i] // strides[i]) for i in range(nd)]
    else:
        # VALID and NOTSET-with-explicit-pads share the same formula; VALID just
        # forces every pad to 0, which is what `pads` already defaults to.
        eff_pads = [0] * (2 * nd) if auto_pad == "VALID" else pads
        out_spatial = [
            (
                in_spatial[i]
                + eff_pads[i]
                + eff_pads[i + nd]
                - (dilations[i] * (kernel[i] - 1) + 1)
            )
            // strides[i]
            + 1
            for i in range(nd)
        ]

    X = helper.make_tensor_value_info("x", dtype, [batch, in_channels, *in_spatial])
    Y = helper.make_tensor_value_info("y", dtype, [batch, out_channels, *out_spatial])

    rng = np.random.default_rng(seed)
    # Scale by 1/sqrt(K): the contraction extent here reaches 588, and fp16
    # accumulation of unit-variance products over that many terms would swamp
    # the tolerance for reasons that have nothing to do with the rewrite.
    fan_in = (in_channels // group) * int(np.prod(kernel))
    scale = 1.0 / np.sqrt(fan_in)
    w = (rng.standard_normal((out_channels, in_channels // group, *kernel)) * scale).astype(
        np_dtype
    )

    inputs = ["x", "w"]
    initializers = [numpy_helper.from_array(w, name="w")]
    if with_bias:
        b = (rng.standard_normal((out_channels,)) * 0.1).astype(np_dtype)
        inputs.append("b")
        initializers.append(numpy_helper.from_array(b, name="b"))

    attrs = dict(
        kernel_shape=list(kernel),
        strides=strides,
        dilations=dilations,
        group=group,
    )
    # auto_pad and pads are mutually exclusive in the ONNX spec, and the real
    # gemma3 export carries auto_pad="VALID" with no pads at all.
    if auto_pad is None:
        attrs["pads"] = pads
    else:
        attrs["auto_pad"] = auto_pad

    node = helper.make_node("Conv", inputs, ["y"], **attrs)
    return make_model_from_nodes([node], [X], [Y], initializers=initializers, opset=17)


def _sample(shape, seed, dtype=np.float16):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


class TestPatchEmbedSinglePatch:
    """kernel == input spatial: one patch per batch element, no transposes."""

    def test_qwen_rank5_patch_embed(self, model_runner):
        """Qwen3.5/3.6/3.8 vision patch embed, verbatim.

        Regression case: this shape already took the Gemm route before the
        stride==kernel generalization, and must keep taking it.
        """
        model = _patch_conv_model(
            in_spatial=[2, 16, 16], kernel=[2, 16, 16], out_channels=1152
        )
        x = _sample((1, 3, 2, 16, 16), 0x9E4)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_qwen_rank5_batched(self, model_runner):
        """Several images at once: the batch axis is the only GEMM row axis here."""
        model = _patch_conv_model(
            in_spatial=[2, 16, 16], kernel=[2, 16, 16], out_channels=256, batch=5
        )
        x = _sample((5, 3, 2, 16, 16), 0x9E5)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_rank4_single_patch(self, model_runner):
        """Rank-4 with kernel == input spatial, the degenerate 1x1 output grid."""
        model = _patch_conv_model(in_spatial=[14, 14], kernel=[14, 14], out_channels=192)
        x = _sample((1, 3, 14, 14), 0x9E6)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)


class TestPatchEmbedMultiPatch:
    """stride == kernel with a real patch grid: the gather permutation matters."""

    def test_gemma3_patch_embed(self, model_runner):
        """gemma3-4b vision patch embed, verbatim: 14x14 stride 14 over 896x896.

        4096 patches, K=588, M=1152. The single largest Conv in the supported
        model set.
        """
        model = _patch_conv_model(
            in_spatial=[896, 896], kernel=[14, 14], out_channels=1152
        )
        x = _sample((1, 3, 896, 896), 0x63A)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    @pytest.mark.parametrize("k", [2, 3, 4, 7])
    def test_square_patch_widths(self, model_runner, k):
        """Small square patches over a small image.

        The permutation formula is indexed off the spatial rank, not the
        extents, so an error in it is width-independent -- but an error in the
        *reshape* interleave (O_i, k_i) is not, and shows up here as soon as
        k_h != k_w or the grid is not square. Kept small so the fp16 reference
        comparison stays tight.
        """
        model = _patch_conv_model(
            in_spatial=[k * 5, k * 5], kernel=[k, k], out_channels=64
        )
        x = _sample((1, 3, k * 5, k * 5), 0x100 + k)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_asymmetric_kernel_and_grid(self, model_runner):
        """kernel_h != kernel_w and O_h != O_w.

        This is the case that catches a transposed patch grid: with square
        kernels and a square grid, swapping the two spatial axes in the
        permutation is invisible.
        """
        model = _patch_conv_model(
            in_spatial=[12, 20], kernel=[4, 5], out_channels=48
        )
        x = _sample((1, 3, 12, 20), 0x2A5)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_batched_multi_patch(self, model_runner):
        """Batch and patch both fold into the GEMM row axis; check the order."""
        model = _patch_conv_model(
            in_spatial=[16, 16], kernel=[4, 4], out_channels=32, batch=3
        )
        x = _sample((3, 3, 16, 16), 0x3B7)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_rank3_multi_patch(self, model_runner):
        """Rank-3 stride==kernel is below the pattern's rank>=4 floor.

        It stays on the generic conv path; asserted numerically so raising that
        floor later cannot silently change the answer.
        """
        model = _patch_conv_model(in_spatial=[64], kernel=[8], out_channels=32)
        x = _sample((1, 3, 64), 0x4C8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_rank5_multi_patch(self, model_runner):
        """Rank-5 with a real 3D patch grid: the gather intermediate is rank 8,
        exactly hip.transpose's ceiling."""
        model = _patch_conv_model(
            in_spatial=[4, 8, 8], kernel=[2, 4, 4], out_channels=32
        )
        x = _sample((1, 3, 4, 8, 8), 0x5D9)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_auto_pad_valid(self, model_runner):
        """auto_pad="VALID" with no pads attribute, as gemma3-4b exports it.

        VALID means zero padding, so this is the same arithmetic as the
        explicit-zero-pads case -- but it reaches the pattern through a
        different attribute, and reading it as an unknown padding mode is what
        kept the real model on MIOpen.
        """
        model = _patch_conv_model(
            in_spatial=[16, 16], kernel=[4, 4], out_channels=32, auto_pad="VALID"
        )
        x = _sample((1, 3, 16, 16), 0x6E9)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_no_bias(self, model_runner):
        """No bias input: the pattern synthesizes a zero C for the Gemm."""
        model = _patch_conv_model(
            in_spatial=[16, 16], kernel=[4, 4], out_channels=32, with_bias=False
        )
        x = _sample((1, 3, 16, 16), 0x6EA)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_fp32(self, model_runner):
        """fp32 patch embed: the Gemm path has to carry the dtype through."""
        model = _patch_conv_model(
            in_spatial=[16, 16],
            kernel=[4, 4],
            out_channels=32,
            dtype=TensorProto.FLOAT,
        )
        x = _sample((1, 3, 16, 16), 0x7FB, dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4, cos_threshold=0.99999)


class TestPatchEmbedGuards:
    """Not patch partitions. Must fall through to the generic conv path."""

    def test_overlapping_stride(self, model_runner):
        """stride < kernel: patches overlap, so no reshape expresses the gather."""
        model = _patch_conv_model(
            in_spatial=[16, 16], kernel=[4, 4], out_channels=32, strides=[2, 2]
        )
        x = _sample((1, 3, 16, 16), 0x8A1)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_padded(self, model_runner):
        """Nonzero pads: a padded patch is not a slice of the input."""
        model = _patch_conv_model(
            in_spatial=[16, 16],
            kernel=[4, 4],
            out_channels=32,
            pads=[1, 1, 1, 1],
        )
        x = _sample((1, 3, 16, 16), 0x8A2)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_ragged_tiling(self, model_runner):
        """Input not an exact multiple of the kernel: the trailing partial patch
        is dropped by Conv, which is a Slice and not a Reshape."""
        model = _patch_conv_model(in_spatial=[18, 18], kernel=[4, 4], out_channels=32)
        x = _sample((1, 3, 18, 18), 0x8A3)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_grouped(self, model_runner):
        """group > 1: each output channel contracts over C/group, not C, so one
        dense GEMM is the wrong arithmetic."""
        model = _patch_conv_model(
            in_spatial=[16, 16],
            kernel=[4, 4],
            in_channels=8,
            out_channels=32,
            group=4,
        )
        x = _sample((1, 8, 16, 16), 0x8A4)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_auto_pad_same_upper(self, model_runner):
        """auto_pad="SAME_UPPER": nonzero implicit padding, so the patches are
        not plain slices of the input."""
        model = _patch_conv_model(
            in_spatial=[15, 15],
            kernel=[4, 4],
            out_channels=32,
            auto_pad="SAME_UPPER",
        )
        x = _sample((1, 3, 15, 15), 0x8A6)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_dilated(self, model_runner):
        """Dilated: the patch is strided within the input, not contiguous."""
        model = _patch_conv_model(
            in_spatial=[16, 16],
            kernel=[2, 2],
            out_channels=32,
            strides=[2, 2],
            dilations=[2, 2],
        )
        x = _sample((1, 3, 16, 16), 0x8A5)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)
