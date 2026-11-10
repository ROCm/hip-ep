#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Stable Diffusion UNet building blocks as PyTorch exports them, against CPU.

Each graph is the primitive spelling the frontend rewrites into one operation,
so these check the rewritten paths end to end:

- scaled-dot-product attention as Transpose / Mul / MatMul / Softmax / MatMul
  (-> MultiHeadAttention and its flash kernel), self- and cross-attention at
  the UNet's head sizes 40, 80 and 160;
- GEGLU as two Slices, the erf GELU primitives and a Mul (-> hip.geglu);
- GroupNorm as Reshape / InstanceNormalization / Reshape / Mul / Add, with and
  without the trailing SiLU (-> GroupNorm, channels_last = 0);
- the 3x3 convolutions of the 8x8 and 16x16 levels, which take the split-K
  convolution path.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

F16 = TensorProto.FLOAT16


def _const(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_attention_model(batch, sq, skv, heads, head_dim):
    """SD-style attention: d^-1/4 on Q and on K^T, BSNH in and out."""
    hidden = heads * head_dim
    q = helper.make_tensor_value_info("q", F16, [batch, sq, hidden])
    k = helper.make_tensor_value_info("k", F16, [batch, skv, hidden])
    v = helper.make_tensor_value_info("v", F16, [batch, skv, hidden])
    out = helper.make_tensor_value_info("out", F16, [batch, sq, hidden])
    inits = [
        _const("split", np.array([batch, -1, heads, head_dim], np.int64)),
        _const("merge", np.array([batch, sq, hidden], np.int64)),
        _const("s", np.array([head_dim**-0.25], np.float16)),
    ]
    nodes = [
        helper.make_node("Reshape", ["q", "split"], ["qr"]),
        helper.make_node("Transpose", ["qr"], ["qt"], perm=[0, 2, 1, 3]),
        helper.make_node("Mul", ["qt", "s"], ["qs"]),
        helper.make_node("Reshape", ["k", "split"], ["kr"]),
        helper.make_node("Transpose", ["kr"], ["kt"], perm=[0, 2, 3, 1]),
        helper.make_node("Mul", ["kt", "s"], ["ks"]),
        helper.make_node("Reshape", ["v", "split"], ["vr"]),
        helper.make_node("Transpose", ["vr"], ["vt"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["qs", "ks"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "vt"], ["o"]),
        helper.make_node("Transpose", ["o"], ["ot"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["ot", "merge"], ["out"]),
    ]
    return make_model_from_nodes(nodes, [q, k, v], [out], initializers=inits)


class TestDecomposedAttention:
    @pytest.mark.parametrize(
        "sq,skv,head_dim",
        [
            (256, 256, 40),  # 64x64 level, self
            (256, 77, 40),  # 64x64 level, cross
            (128, 128, 80),  # 32x32 level
            (64, 64, 160),  # 16x16 / 8x8 levels
            (64, 77, 160),
        ],
    )
    def test_sd_attention(self, model_runner, sq, skv, head_dim):
        batch, heads = 2, 8
        model = _make_attention_model(batch, sq, skv, heads, head_dim)
        rng = np.random.default_rng(11)
        hidden = heads * head_dim
        q = rng.standard_normal((batch, sq, hidden)).astype(np.float16)
        k = rng.standard_normal((batch, skv, hidden)).astype(np.float16)
        v = rng.standard_normal((batch, skv, hidden)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [q, k, v])
        # The reference rounds Q*s, K*s, the scores and the probabilities to
        # fp16; the flash kernel keeps all of them in fp32.
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)


def _make_geglu_model(rows, f):
    p = helper.make_tensor_value_info("p", F16, [2, rows, 2 * f])
    y = helper.make_tensor_value_info("y", F16, [2, rows, f])
    inits = [
        _const("zero", np.array([0], np.int64)),
        _const("f", np.array([f], np.int64)),
        _const("two_f", np.array([2 * f], np.int64)),
        _const("axis", np.array([-1], np.int64)),
        _const("sqrt2", np.array(1.4140625, np.float16)),
        _const("one", np.array(1.0, np.float16)),
        _const("half", np.array(0.5, np.float16)),
    ]
    nodes = [
        helper.make_node("Slice", ["p", "zero", "f", "axis"], ["h"]),
        helper.make_node("Slice", ["p", "f", "two_f", "axis"], ["g"]),
        helper.make_node("Div", ["g", "sqrt2"], ["d"]),
        helper.make_node("Erf", ["d"], ["e"]),
        helper.make_node("Add", ["e", "one"], ["a"]),
        helper.make_node("Mul", ["g", "a"], ["m"]),
        helper.make_node("Mul", ["m", "half"], ["gelu"]),
        helper.make_node("Mul", ["h", "gelu"], ["y"]),
    ]
    return make_model_from_nodes(nodes, [p], [y], initializers=inits)


class TestGeglu:
    @pytest.mark.parametrize("rows,f", [(256, 320), (64, 640), (16, 1280)])
    def test_geglu(self, model_runner, rows, f):
        model = _make_geglu_model(rows, f)
        rng = np.random.default_rng(12)
        p = (rng.standard_normal((2, rows, 2 * f)) * 2).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [p])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)


def _make_group_norm_model(channels, size, groups, silu):
    shape = [2, channels, size, size]
    x = helper.make_tensor_value_info("x", F16, shape)
    y = helper.make_tensor_value_info("y", F16, shape)
    rng = np.random.default_rng(13)
    inits = [
        _const("view", np.array([0, groups, -1], np.int64)),
        _const("back", np.array(shape, np.int64)),
        _const("ones", np.ones(groups, np.float16)),
        _const("zeros", np.zeros(groups, np.float16)),
        _const("gamma", rng.uniform(0.5, 1.5, (channels, 1, 1)).astype(np.float16)),
        _const("beta", rng.uniform(-0.5, 0.5, (channels, 1, 1)).astype(np.float16)),
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "view"], ["r"]),
        helper.make_node(
            "InstanceNormalization", ["r", "ones", "zeros"], ["n"], epsilon=1e-5
        ),
        helper.make_node("Reshape", ["n", "back"], ["u"]),
        helper.make_node("Mul", ["u", "gamma"], ["m"]),
        helper.make_node("Add", ["m", "beta"], ["a" if silu else "y"]),
    ]
    if silu:
        nodes += [
            helper.make_node("Sigmoid", ["a"], ["s"]),
            helper.make_node("Mul", ["a", "s"], ["y"]),
        ]
    return make_model_from_nodes(nodes, [x], [y], initializers=inits)


class TestGroupNorm:
    @pytest.mark.parametrize(
        "channels,size,silu",
        [(320, 64, True), (640, 32, False), (1280, 16, True), (1280, 8, True)],
    )
    def test_group_norm(self, model_runner, channels, size, silu):
        model = _make_group_norm_model(channels, size, 32, silu)
        rng = np.random.default_rng(14)
        # An offset larger than the spread checks the variance for cancellation.
        x = (rng.standard_normal((2, channels, size, size)) + 3.0).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)


def _make_conv_model(cin, cout, size):
    x = helper.make_tensor_value_info("x", F16, [2, cin, size, size])
    y = helper.make_tensor_value_info("y", F16, [2, cout, size, size])
    rng = np.random.default_rng(15)
    w = (rng.standard_normal((cout, cin, 3, 3)) / np.sqrt(cin * 9)).astype(np.float16)
    b = rng.uniform(-0.1, 0.1, cout).astype(np.float16)
    node = helper.make_node(
        "Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
    )
    return make_model_from_nodes(
        [node], [x], [y], initializers=[_const("w", w), _const("b", b)]
    )


class TestSmallPlaneConv:
    @pytest.mark.parametrize(
        "cin,cout,size", [(1280, 1280, 8), (2560, 1280, 8), (640, 1280, 16)]
    )
    def test_conv3x3(self, model_runner, cin, cout, size):
        model = _make_conv_model(cin, cout, size)
        rng = np.random.default_rng(16)
        x = rng.standard_normal((2, cin, size, size)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)
