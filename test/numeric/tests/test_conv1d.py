#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for 1D Conv (rank-3 ONNX `Conv`) -- whisper-large-v3 encoder
front-end shapes:

  Conv layer 0: Cin=128,  Cout=1280, K=3, stride=1, pad=1 -> [1, 1280, 3000]
  Conv layer 1: Cin=1280, Cout=1280, K=3, stride=2, pad=1 -> [1, 1280, 1500]

Both layers carry per-channel bias. The MorphiZen path lowers rank-3
onnx.Conv to the shared 2D hip.conv via a unit-H reshape (NCL -> NC1L ->
hip.conv -> NCL'), dispatching to MIOpen's 4D convolution
(wrap_miopenConvolutionForward, now dtype + bias + scratch-pool aware).
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


# Encoder front-end (n_mels, hidden) for the 80-mel variant family — tiny 80/384,
# base 80/512, small 80/768, medium 80/1024. large-v3 & turbo are 128/1280, which
# the dedicated layer0/layer1 tests above already cover. Each variant runs the
# same two conv layers: layer0 = n_mels->hidden k3 s1, layer1 = hidden->hidden
# k3 s2 (the only front-end shape that differs across variants).
VARIANT_ENC_80MEL = [(80, 384), (80, 512), (80, 768), (80, 1024)]


def _make_conv1d_model(cin, lin, cout, k, stride, pad, dtype=TensorProto.FLOAT16):
    """Single-op rank-3 Conv model with weight+bias as initializers."""
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32

    lout = (lin + 2 * pad - k) // stride + 1
    X = helper.make_tensor_value_info("x", dtype, [1, cin, lin])
    Y = helper.make_tensor_value_info("y", dtype, [1, cout, lout])

    rng = np.random.default_rng(0xC1D)
    w = (rng.standard_normal((cout, cin, k)) * 0.1).astype(np_dtype)
    b = (rng.standard_normal((cout,)) * 0.1).astype(np_dtype)
    initializers = [
        numpy_helper.from_array(w, name="w"),
        numpy_helper.from_array(b, name="b"),
    ]

    node = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[k],
        strides=[stride],
        pads=[pad, pad],
    )
    return make_model_from_nodes([node], [X], [Y], initializers=initializers, opset=17)


class TestWhisperConv1d:
    """The two whisper-large-v3 encoder front-end Conv shapes."""

    def test_layer0_cin128_cout1280_k3_s1(self, model_runner):
        """First encoder conv: mel[1,128,3000] -> [1,1280,3000]."""
        model = _make_conv1d_model(cin=128, lin=3000, cout=1280, k=3, stride=1, pad=1)
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((1, 128, 3000)) * 0.1).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2)

    def test_layer1_cin1280_cout1280_k3_s2(self, model_runner):
        """Second encoder conv: [1,1280,3000] -> [1,1280,1500]."""
        model = _make_conv1d_model(cin=1280, lin=3000, cout=1280, k=3, stride=2, pad=1)
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((1, 1280, 3000)) * 0.1).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2)

    @pytest.mark.parametrize(
        "cin,lin,cout,k,stride,pad",
        [
            # Tiny sanity cases to localise any kernel-shape / stride regression
            # independently of the full whisper memory footprint.
            (4, 16, 4, 3, 1, 1),
            (4, 16, 8, 3, 2, 1),
            (8, 32, 8, 1, 1, 0),  # 1x1 conv
        ],
    )
    def test_conv1d_tiny(self, model_runner, cin, lin, cout, k, stride, pad):
        model = _make_conv1d_model(cin, lin, cout, k, stride, pad)
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((1, cin, lin)) * 0.5).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-2)

    @pytest.mark.parametrize("n_mels,hidden", VARIANT_ENC_80MEL)
    def test_encoder_frontend_variant(self, model_runner, n_mels, hidden):
        """Both encoder conv layers for an 80-mel variant (layer0 + layer1)."""
        rng = np.random.default_rng(hidden)
        # layer0: n_mels -> hidden, k3 s1
        m0 = _make_conv1d_model(cin=n_mels, lin=3000, cout=hidden, k=3, stride=1, pad=1)
        x0 = (rng.standard_normal((1, n_mels, 3000)) * 0.1).astype(np.float16)
        a0, e0 = model_runner.run_sample(m0, [x0])
        compare_outputs(a0, e0, atol=5e-3, rtol=1e-2)
        # layer1: hidden -> hidden, k3 s2
        m1 = _make_conv1d_model(cin=hidden, lin=3000, cout=hidden, k=3, stride=2, pad=1)
        x1 = (rng.standard_normal((1, hidden, 3000)) * 0.1).astype(np.float16)
        a1, e1 = model_runner.run_sample(m1, [x1])
        compare_outputs(a1, e1, atol=5e-3, rtol=1e-2)


class TestWhisperConv1dFp32:
    """fp32 variants of the whisper encoder Conv shapes.

    Exercises the fp32 MIOpen descriptor path of the shared 2D conv kernel
    (element_size_bytes=4 -> miopenFloat) via the rank-3 reshape lowering.
    fp32 accumulation is tighter than fp16, so tolerances are much smaller and
    cosine should be ~1.0.
    """

    def test_layer0_cin128_cout1280_k3_s1_fp32(self, model_runner):
        """First encoder conv in fp32: mel[1,128,3000] -> [1,1280,3000]."""
        model = _make_conv1d_model(
            cin=128, lin=3000, cout=1280, k=3, stride=1, pad=1, dtype=TensorProto.FLOAT
        )
        rng = np.random.default_rng(0)
        x = (rng.standard_normal((1, 128, 3000)) * 0.1).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    def test_layer1_cin1280_cout1280_k3_s2_fp32(self, model_runner):
        """Second encoder conv in fp32: [1,1280,3000] -> [1,1280,1500]."""
        model = _make_conv1d_model(
            cin=1280, lin=3000, cout=1280, k=3, stride=2, pad=1, dtype=TensorProto.FLOAT
        )
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((1, 1280, 3000)) * 0.1).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("n_mels,hidden", VARIANT_ENC_80MEL)
    def test_encoder_frontend_variant_fp32(self, model_runner, n_mels, hidden):
        """Both encoder conv layers for an 80-mel variant in fp32."""
        rng = np.random.default_rng(hidden)
        m0 = _make_conv1d_model(
            cin=n_mels,
            lin=3000,
            cout=hidden,
            k=3,
            stride=1,
            pad=1,
            dtype=TensorProto.FLOAT,
        )
        x0 = (rng.standard_normal((1, n_mels, 3000)) * 0.1).astype(np.float32)
        a0, e0 = model_runner.run_sample(m0, [x0])
        compare_outputs(a0, e0, atol=1e-4, rtol=1e-4)
        m1 = _make_conv1d_model(
            cin=hidden,
            lin=3000,
            cout=hidden,
            k=3,
            stride=2,
            pad=1,
            dtype=TensorProto.FLOAT,
        )
        x1 = (rng.standard_normal((1, hidden, 3000)) * 0.1).astype(np.float32)
        a1, e1 = model_runner.run_sample(m1, [x1])
        compare_outputs(a1, e1, atol=1e-4, rtol=1e-4)
