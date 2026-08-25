#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for the depthwise causal rank-3 Conv route.

A rank-3 depthwise ``onnx.Conv`` with left-only padding of exactly ``k-1`` is a
causal convolution whose carry state is zero, so ``DepthwiseCausalConvToHip``
routes it to ``hip.causal_conv_with_state`` and the runtime serves it from the
fused ``hip_causal_conv_prefill`` kernel instead of MIOpen. These tests pin the
arithmetic of that route against the ORT CPU reference.

The shapes come from the gemma-4 E2B/E4B audio encoders, which contain 12 of
these each: ``W=[1024,1,5]``, ``pads=[4,0]``, ``group=1024``, fp16. In the real
graph every one is bracketed by a ``Transpose[0,2,1]`` pair, which the
canonicalizer absorbs into ``channels_last`` -- ``test_causal_transpose_pair``
exercises that folded form, since it dispatches to a different kernel
(``hip_causal_conv_prefill_nlc``) reading ``[B,L,C]`` directly.

The guard cases (symmetric pads, stride 2) must stay on the generic conv path;
they are here so a guard that wrongly lets them through fails numerically rather
than silently.
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


def _conv1d_nodes(channels, length, k, pads, stride=1, dtype=TensorProto.FLOAT16, seed=5):
    """A single rank-3 depthwise Conv with weight+bias initializers."""
    np_dtype = _DTYPES[dtype]
    lout = (length + pads[0] + pads[1] - k) // stride + 1

    X = helper.make_tensor_value_info("x", dtype, [1, channels, length])
    Y = helper.make_tensor_value_info("y", dtype, [1, channels, lout])

    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((channels, 1, k)) * 0.3).astype(np_dtype)
    b = (rng.standard_normal((channels,)) * 0.3).astype(np_dtype)

    node = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[k],
        strides=[stride],
        pads=list(pads),
        dilations=[1],
        group=channels,
    )
    initializers = [
        numpy_helper.from_array(w, name="w"),
        numpy_helper.from_array(b, name="b"),
    ]
    return make_model_from_nodes([node], [X], [Y], initializers=initializers, opset=17)


class TestDepthwiseCausalConv1d:
    """pads == [k-1, 0]: the shapes that take the fused causal kernel."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
    def test_causal_kernel_widths(self, model_runner, k):
        """Every kernel width the fused kernel instantiates.

        k is the whole causal contract: get the tap ordering or the zero-carry
        prefix wrong and only some widths break, so sweep all of them.
        """
        model = _conv1d_nodes(channels=64, length=97, k=k, pads=[k - 1, 0])
        rng = np.random.default_rng(k)
        x = rng.standard_normal((1, 64, 97)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_audio_encoder_shape(self, model_runner):
        """The gemma-4 audio encoder lconv1d: C=1024, k=5, L=750."""
        model = _conv1d_nodes(channels=1024, length=750, k=5, pads=[4, 0])
        rng = np.random.default_rng(0xA0D)
        x = rng.standard_normal((1, 1024, 750)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    @pytest.mark.parametrize("length", [1, 2, 4, 5, 6, 31])
    def test_short_sequences(self, model_runner, length):
        """L around and below k, where the zero-carry prefix dominates.

        At k=5 a length-1 input has four of its five taps in the zero prefix, so
        these lengths are where an off-by-one in the virtual-sequence indexing
        shows up as a large error rather than a small one. L=1 also crosses from
        the prefill kernel to the single-step decode kernel.
        """
        model = _conv1d_nodes(channels=32, length=length, k=5, pads=[4, 0])
        rng = np.random.default_rng(length)
        x = rng.standard_normal((1, 32, length)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_causal_fp32(self, model_runner):
        """fp32 is on the fused kernel's envelope (element_size_bytes == 4)."""
        model = _conv1d_nodes(
            channels=64, length=128, k=5, pads=[4, 0], dtype=TensorProto.FLOAT
        )
        rng = np.random.default_rng(1)
        x = rng.standard_normal((1, 64, 128)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_causal_no_bias(self, model_runner):
        """No bias operand: the fused kernel takes a null bias pointer."""
        rng = np.random.default_rng(2)
        w = (rng.standard_normal((64, 1, 5)) * 0.3).astype(np.float16)
        X = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 64, 128])
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 64, 128])
        node = helper.make_node(
            "Conv",
            ["x", "w"],
            ["y"],
            kernel_shape=[5],
            strides=[1],
            pads=[4, 0],
            dilations=[1],
            group=64,
        )
        model = make_model_from_nodes(
            [node],
            [X],
            [Y],
            initializers=[numpy_helper.from_array(w, name="w")],
            opset=17,
        )
        x = rng.standard_normal((1, 64, 128)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_causal_transpose_pair(self, model_runner):
        """The form the audio encoders actually export.

        Transpose[0,2,1] -> depthwise causal Conv -> Transpose[0,2,1]. The
        canonicalizer absorbs both transposes into `channels_last`, which routes
        to hip_causal_conv_prefill_nlc -- a different kernel reading [B,L,C]
        directly. Numerics must be unchanged by that layout switch.
        """
        channels, length, k = 1024, 200, 5
        rng = np.random.default_rng(0xC1C)
        w = (rng.standard_normal((channels, 1, k)) * 0.3).astype(np.float16)
        b = (rng.standard_normal((channels,)) * 0.3).astype(np.float16)

        X = helper.make_tensor_value_info(
            "x", TensorProto.FLOAT16, [1, length, channels]
        )
        Y = helper.make_tensor_value_info(
            "y", TensorProto.FLOAT16, [1, length, channels]
        )
        nodes = [
            helper.make_node("Transpose", ["x"], ["t0"], perm=[0, 2, 1]),
            helper.make_node(
                "Conv",
                ["t0", "w", "b"],
                ["c"],
                kernel_shape=[k],
                strides=[1],
                pads=[k - 1, 0],
                dilations=[1],
                group=channels,
            ),
            helper.make_node("Transpose", ["c"], ["y"], perm=[0, 2, 1]),
        ]
        model = make_model_from_nodes(
            nodes,
            [X],
            [Y],
            initializers=[
                numpy_helper.from_array(w, name="w"),
                numpy_helper.from_array(b, name="b"),
            ],
            opset=17,
        )
        x = rng.standard_normal((1, length, channels)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)


class TestDepthwiseNonCausalGuards:
    """Shapes the causal route must decline. Correct on the generic path."""

    def test_symmetric_pads(self, model_runner):
        """pads=[2,2] reads future taps, so it is not the causal operation."""
        model = _conv1d_nodes(channels=64, length=128, k=5, pads=[2, 2])
        rng = np.random.default_rng(3)
        x = rng.standard_normal((1, 64, 128)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_causal_pads_stride2(self, model_runner):
        """Causal pads but stride 2: the fused kernel emits one output per
        input position, so this must not be claimed."""
        model = _conv1d_nodes(channels=64, length=128, k=5, pads=[4, 0], stride=2)
        rng = np.random.default_rng(4)
        x = rng.standard_normal((1, 64, 128)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)

    def test_grouped_not_depthwise(self, model_runner):
        """group=32 with C=64: grouped, not depthwise, weight is [64,2,4]."""
        rng = np.random.default_rng(6)
        w = (rng.standard_normal((64, 2, 4)) * 0.3).astype(np.float16)
        b = (rng.standard_normal((64,)) * 0.3).astype(np.float16)
        X = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 64, 128])
        Y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 64, 128])
        node = helper.make_node(
            "Conv",
            ["x", "w", "b"],
            ["y"],
            kernel_shape=[4],
            strides=[1],
            pads=[3, 0],
            dilations=[1],
            group=32,
        )
        model = make_model_from_nodes(
            [node],
            [X],
            [Y],
            initializers=[
                numpy_helper.from_array(w, name="w"),
                numpy_helper.from_array(b, name="b"),
            ],
            opset=17,
        )
        x = rng.standard_normal((1, 64, 128)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2, cos_threshold=0.9999)
