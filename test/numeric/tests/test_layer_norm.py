#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization,
and the full LayerNormalization (added by qwen-vision-kernels PR).

These ops are used in Llama-3.1-8B / Qwen / GPT-OSS:
- SimplifiedLayerNormalization is registered in ORT's default domain.
- SkipSimplifiedLayerNormalization is a com.microsoft contrib op.
- LayerNormalization is the full ONNX-17 op:
      y = (x - mean) * rsqrt(var + eps) * scale + bias
  with optional `bias` input and optional `mean`/`inv_std` outputs. The
  runtime implements a block-per-row HIP kernel with fp32 internal math.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]
HIDDEN = 4096


def _make_simplified_layer_norm_model(input_shape: list[int]):
    """Build a SimplifiedLayerNormalization ONNX model (f16).

    ORT registers this op in the default ONNX domain (not com.microsoft).
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(77)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    scale_init = numpy_helper.from_array(scale_data, name="scale")

    node = helper.make_node(
        "SimplifiedLayerNormalization",
        ["X", "scale"],
        ["Y"],
        axis=-1,
        epsilon=1e-5,
        stash_type=1,
    )
    model = make_model_from_nodes(
        [node], [X], [Y], initializers=[scale_init], extra_opsets=[]
    )
    return model


def _make_skip_simplified_layer_norm_model(input_shape: list[int]):
    """Build a SkipSimplifiedLayerNormalization ONNX model (f16).

    This op is in the com.microsoft domain. ORT CPU provider supports it.
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(88)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    scale_init = numpy_helper.from_array(scale_data, name="scale")

    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        ["X", "skip", "scale"],
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=[scale_init],
        extra_opsets=[ms_opset],
    )
    return model


class TestSimplifiedLayerNorm:
    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_simplified_layer_norm(self, model_runner, input_shape):
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_simplified_layer_norm_llama_shape(self, model_runner, seq_len):
        """SimplifiedLayerNorm with shapes matching Llama input_layernorm."""
        input_shape = [1, seq_len, HIDDEN]
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)


class TestSkipSimplifiedLayerNorm:
    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_skip_simplified_layer_norm(self, model_runner, input_shape):
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(55)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_skip_simplified_layer_norm_llama_shape(self, model_runner, seq_len):
        """SkipSimplifiedLayerNorm with shapes matching Llama post_attention_layernorm."""
        input_shape = [1, seq_len, HIDDEN]
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(55)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)


# ---------------------------------------------------------------------------
# Full LayerNormalization (added by qwen-vision-kernels PR)
# ---------------------------------------------------------------------------


def _make_layer_norm_model(
    input_shape: list[int],
    dtype: np.dtype = np.float16,
    with_bias: bool = False,
):
    """Build a full LayerNormalization ONNX model.

    Output is single (no `mean` / `inv_std`) -- the runtime supports those
    optional outputs but they are emitted only when the producing graph
    requests them; this signature is the most common in practice.
    """
    hidden = input_shape[-1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)

    rng = np.random.default_rng(99)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
    scale_init = numpy_helper.from_array(scale_data, name="scale")
    initializers = [scale_init]
    input_names = ["X", "scale"]

    if with_bias:
        bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(dtype)
        bias_init = numpy_helper.from_array(bias_data, name="bias")
        initializers.append(bias_init)
        input_names.append("bias")

    node = helper.make_node(
        "LayerNormalization",
        input_names,
        ["Y"],
        axis=-1,
        epsilon=1e-5,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=initializers)


class TestLayerNormalization:
    """Full LN: (x - mean) * rsqrt(var + eps) * scale [+ bias]."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
            [2, 4, 64],
        ],
    )
    def test_layer_norm_no_bias(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=False)
        rng = np.random.default_rng(1101)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        # Block-per-row reduction in fp32, fp16 round-trip on I/O -> ~1e-3.
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 8, 32],
            [2, 4, 64],
        ],
    )
    def test_layer_norm_with_bias(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=True)
        rng = np.random.default_rng(1102)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_layer_norm_llama_shape(self, model_runner, seq_len):
        """fp16 LN with bias on a Llama-style [1, S, 4096] tensor."""
        input_shape = [1, seq_len, HIDDEN]
        model = _make_layer_norm_model(input_shape, np.float16, with_bias=True)
        rng = np.random.default_rng(1103)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    def test_layer_norm_fp32(self, model_runner):
        """LN on f32 input with bias -- exercises the f32 dispatch path."""
        input_shape = [1, 8, 64]
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        rng = np.random.default_rng(1104)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("num_patches", [256, 1024])
    def test_layer_norm_qwen35b_vision_shape(self, model_runner, num_patches):
        """LN exactly as it appears 55x in Qwen3.5-35B vision.onnx:
        in:    f32 [num_patches, 1152]
        scale: f32 [1152]
        bias:  f32 [1152]
        axis=-1, epsilon=1e-6, stash_type=1
        """
        input_shape = [num_patches, 1152]
        hidden = input_shape[-1]
        tp = TensorProto.FLOAT
        X = helper.make_tensor_value_info("X", tp, input_shape)
        Y = helper.make_tensor_value_info("Y", tp, input_shape)
        rng = np.random.default_rng(1105)
        scale_init = numpy_helper.from_array(
            rng.uniform(0.5, 1.5, [hidden]).astype(np.float32), name="scale"
        )
        bias_init = numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [hidden]).astype(np.float32), name="bias"
        )
        node = helper.make_node(
            "LayerNormalization",
            ["X", "scale", "bias"],
            ["Y"],
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )
        model = make_model_from_nodes(
            [node], [X], [Y], initializers=[scale_init, bias_init]
        )
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)
