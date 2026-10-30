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
from framework.onnx_utils import fp64_normalize_last_axis, make_model_from_nodes

SEQ_LENS = [1, 128]
HIDDEN = 4096
# Odd row widths: the RMS-norm kernels only take their packed __half2 path
# when the row is even, so these pin down the scalar fp16 fallback. 33 also
# lands mid-block (block size 256), exercising the strided loop tail.
ODD_HIDDENS = [33, 4095]


_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_simplified_layer_norm_model(
    input_shape: list[int], dtype: np.dtype = np.float16
):
    """Build a SimplifiedLayerNormalization ONNX model.

    ORT registers this op in the default ONNX domain (not com.microsoft).
    """
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)

    rng = np.random.default_rng(77)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
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


def _make_skip_simplified_layer_norm_model(
    input_shape: list[int], dtype: np.dtype = np.float16, with_bias: bool = False
):
    """Build a SkipSimplifiedLayerNormalization ONNX model.

    This op is in the com.microsoft domain. ORT CPU provider supports it.
    """
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    skip = helper.make_tensor_value_info("skip", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", tp, input_shape)

    rng = np.random.default_rng(88)
    scale_data = rng.uniform(0.5, 1.5, [hidden]).astype(dtype)
    initializers = [numpy_helper.from_array(scale_data, name="scale")]
    inputs = ["X", "skip", "scale"]
    if with_bias:
        bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(dtype)
        initializers.append(numpy_helper.from_array(bias_data, name="bias"))
        inputs.append("bias")

    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs,
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=initializers,
        extra_opsets=[ms_opset],
    )
    return model


def _make_fp32_residual_skip_norm_model(input_shape: list[int]):
    """gpt-oss residual layout: fp32 residual, everything else fp16 with Casts.

    Y = Cast_f16(SSLN(res32, Cast_f32(skip16), Cast_f32(gamma16)).output)
    Y3 = the fp32 residual sum, which feeds the next layer's norm.
    """
    hidden = input_shape[-1]
    R = helper.make_tensor_value_info("R", TensorProto.FLOAT, input_shape)
    S = helper.make_tensor_value_info("S", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT, input_shape)

    rng = np.random.default_rng(89)
    gamma = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    nodes = [
        helper.make_node("Cast", ["S"], ["S32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["gamma"], ["G32"], to=TensorProto.FLOAT),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["R", "S32", "G32"],
            ["Y32", "", "", "Y3"],
            domain="com.microsoft",
            epsilon=1e-5,
        ),
        helper.make_node("Cast", ["Y32"], ["Y"], to=TensorProto.FLOAT16),
    ]
    return make_model_from_nodes(
        nodes,
        [R, S],
        [Y, Y3],
        initializers=[numpy_helper.from_array(gamma, name="gamma")],
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _make_skip_layer_norm_model(input_shape: list[int]):
    """Build a standard com.microsoft.SkipLayerNormalization ONNX model (f16).

    This is mean-subtracting LayerNorm WITH a skip/residual add and bias (beta),
    distinct from SkipSimplifiedLayerNormalization (RMS norm). 4-input form
    (input, skip, gamma, beta), 4-output (output, mean, inv_std, sum) with only
    output[0] and output[3] consumed -- exactly the Whisper encoder pattern.

    Semantics (MS spec, 4-input / no input-bias):
        sum    = input + skip
        output = LayerNorm(sum, gamma, beta, epsilon)
        output[3] (input_skip_bias_sum) = sum

    ORT CPU provider supports it. Validates the converter's hip.add +
    hip.layer_norm composition end-to-end.
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    # output[3] = input_skip_bias_sum (Whisper consumes this as next residual).
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(66)
    gamma_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    beta_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    gamma_init = numpy_helper.from_array(gamma_data, name="gamma")
    beta_init = numpy_helper.from_array(beta_data, name="beta")

    node = helper.make_node(
        "SkipLayerNormalization",
        ["X", "skip", "gamma", "beta"],
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=[gamma_init, beta_init],
        extra_opsets=[ms_opset],
    )
    return model


class TestSkipLayerNorm:
    """Standard SkipLayerNormalization (Whisper encoder, 4-in / output[0]+[3])."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 32],
        ],
    )
    def test_skip_layer_norm(self, model_runner, input_shape):
        model = _make_skip_layer_norm_model(input_shape)
        rng = np.random.default_rng(77)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        # output[0] = LayerNorm(x+skip); output[3] = x+skip. Block-per-row fp32
        # reduction + fp16 I/O round-trip -> ~2e-3.
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    def test_skip_layer_norm_whisper_shape(self, model_runner):
        """SkipLayerNorm at the real Whisper-large-v3 encoder hidden size 1280."""
        input_shape = [1, 16, 1280]
        model = _make_skip_layer_norm_model(input_shape)
        rng = np.random.default_rng(78)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _make_skip_layer_norm_bias_model(input_shape: list[int]):
    """5-input SkipLayerNormalization: bias is added to input before the skip.

    Stable Diffusion's text encoder uses this form (SkipLayerNorm_AddBias):
        sum    = input + bias + skip
        output = LayerNorm(sum, gamma, beta, epsilon)
        output[3] = sum
    """
    hidden = input_shape[-1]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT16, input_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, input_shape)
    Y3 = helper.make_tensor_value_info("Y3", TensorProto.FLOAT16, input_shape)

    rng = np.random.default_rng(66)
    gamma_data = rng.uniform(0.5, 1.5, [hidden]).astype(np.float16)
    beta_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    bias_data = rng.uniform(-0.5, 0.5, [hidden]).astype(np.float16)
    gamma_init = numpy_helper.from_array(gamma_data, name="gamma")
    beta_init = numpy_helper.from_array(beta_data, name="beta")
    bias_init = numpy_helper.from_array(bias_data, name="bias")

    node = helper.make_node(
        "SkipLayerNormalization",
        ["X", "skip", "gamma", "beta", "bias"],
        ["Y", "", "", "Y3"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [X, skip],
        [Y, Y3],
        initializers=[gamma_init, beta_init, bias_init],
        extra_opsets=[ms_opset],
    )


class TestSkipLayerNormInputBias:
    """5-input SkipLayerNormalization (Stable Diffusion text encoder)."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [1, 8, 1024],
        ],
    )
    def test_skip_layer_norm_input_bias(self, model_runner, input_shape):
        model = _make_skip_layer_norm_bias_model(input_shape)
        rng = np.random.default_rng(79)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


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
        """SimplifiedLayerNorm with shapes matching Llama input_layernorm.

        HIDDEN=4096 is also the case that would regress first if the kernel's
        sum-of-squares accumulator were ever narrowed from fp32 to fp16.
        """
        input_shape = [1, seq_len, HIDDEN]
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("hidden", ODD_HIDDENS)
    def test_simplified_layer_norm_odd_hidden(self, model_runner, hidden):
        """Odd hidden size forces the scalar fp16 body.

        The packed __half2 path needs an even row, so without this the scalar
        fp16 fallback is never exercised -- every other case here is even.
        """
        input_shape = [1, 4, hidden]
        model = _make_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(43)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("input_shape", [[1, 4, 16], [1, 8, 4096]])
    def test_simplified_layer_norm_fp32(self, model_runner, input_shape):
        model = _make_simplified_layer_norm_model(input_shape, np.float32)

        rng = np.random.default_rng(44)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5)


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

    @pytest.mark.parametrize("hidden", ODD_HIDDENS)
    def test_skip_simplified_layer_norm_odd_hidden(self, model_runner, hidden):
        """Odd hidden size forces the scalar fp16 body (see the RMS-norm twin)."""
        input_shape = [1, 4, hidden]
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(56)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("input_shape", [[1, 4, 16], [1, 8, 4096]])
    def test_skip_simplified_layer_norm_fp32(self, model_runner, input_shape):
        model = _make_skip_simplified_layer_norm_model(input_shape, np.float32)

        rng = np.random.default_rng(57)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_skip_simplified_layer_norm_gpt_oss_shape(
        self, model_runner, seq_len, dtype
    ):
        """gpt-oss residual norm: hidden 2880, run in fp32 by the export."""
        input_shape = [1, seq_len, 2880]
        model = _make_skip_simplified_layer_norm_model(input_shape, dtype)

        rng = np.random.default_rng(58)
        x = rng.uniform(-2, 2, input_shape).astype(dtype)
        skip_input = rng.uniform(-2, 2, input_shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=1e-5 if dtype == np.float32 else 2e-3)

    @pytest.mark.parametrize(
        "dtype,hidden",
        [(np.float16, 2880), (np.float32, 2880), (np.float16, 8200), (np.float16, 33)],
    )
    def test_skip_simplified_layer_norm_bias(self, model_runner, dtype, hidden):
        """4-input form, through the row kernel (2880) and both fallbacks."""
        input_shape = [1, 3, hidden]
        model = _make_skip_simplified_layer_norm_model(
            input_shape, dtype, with_bias=True
        )

        rng = np.random.default_rng(60)
        x = rng.uniform(-2, 2, input_shape).astype(dtype)
        skip_input = rng.uniform(-2, 2, input_shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=1e-5 if dtype == np.float32 else 2e-3)

    @pytest.mark.parametrize(
        "seq_len,hidden", [(1, 2880), (128, 2880), (3, 8200), (2, 33)]
    )
    def test_skip_simplified_layer_norm_fp32_residual(
        self, model_runner, seq_len, hidden
    ):
        """Cast -> fp32 SSLN -> Cast as gpt-oss exports it (Casts folded in)."""
        input_shape = [1, seq_len, hidden]
        model = _make_fp32_residual_skip_norm_model(input_shape)

        rng = np.random.default_rng(61)
        res = rng.uniform(-2, 2, input_shape).astype(np.float32)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [res, skip_input])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("hidden", [8192, 8200])
    def test_skip_simplified_layer_norm_wide_row(self, model_runner, hidden):
        """Either side of the register-resident row kernel's width limit."""
        input_shape = [1, 2, hidden]
        model = _make_skip_simplified_layer_norm_model(input_shape)

        rng = np.random.default_rng(59)
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

    # |mean| / stddev near 260 with sparse outliers: E[x^2] - mean^2 in fp32
    # misses the tolerance on these rows. The reference is fp64 for the same
    # reason as test_instance_norm_large_offset.
    @pytest.mark.parametrize("input_shape", [[4, 1152], [3, 4096], [2, 7, 4097]])
    def test_layer_norm_large_offset(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        rng = np.random.default_rng(1106)
        x = rng.uniform(-1, 1, input_shape) + 200.0
        x.flat[::61] += 4.0
        actual, expected = model_runner.run_sample(
            model,
            [x.astype(np.float32)],
            reference_model=_fp64_layer_norm_reference(model, input_shape),
        )
        compare_outputs(actual, expected, atol=2e-4, rtol=2e-4)


def _fp64_layer_norm_reference(model, input_shape: list[int]):
    """*model*'s fp32 last-axis LayerNormalization recomputed in fp64 from
    primitives, with the same scale and bias initializers."""
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    norm_nodes, norm_inits = fp64_normalize_last_axis("xd", "nd", 1e-5)
    nodes = [
        helper.make_node("Cast", ["X"], ["xd"], to=TensorProto.DOUBLE),
        *norm_nodes,
        helper.make_node("Mul", ["nd", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["yd"]),
        helper.make_node("Cast", ["yd"], ["Y"], to=TensorProto.FLOAT),
    ]
    initializers = [
        numpy_helper.from_array(inits["scale"].astype(np.float64), "scale"),
        numpy_helper.from_array(inits["bias"].astype(np.float64), "bias"),
        *norm_inits,
    ]
    return make_model_from_nodes(
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)],
        initializers=initializers,
    )


def _make_instance_norm_model(input_shape: list[int], dtype=np.float32):
    """Build an ONNX InstanceNormalization model (scale/B as initializers)."""
    channels = input_shape[1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(2026)
    scale_init = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="scale"
    )
    bias_init = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="B"
    )
    node = helper.make_node(
        "InstanceNormalization",
        ["X", "scale", "B"],
        ["Y"],
        epsilon=1e-5,
    )
    return make_model_from_nodes([node], [X], [Y], initializers=[scale_init, bias_init])


def _fp64_instance_norm_reference(model, input_shape: list[int]):
    """*model*'s fp32 InstanceNormalization recomputed in fp64 from primitives,
    with the same scale and B initializers."""
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    affine_shape = [input_shape[1]] + [1] * (len(input_shape) - 2)
    norm_nodes, norm_inits = fp64_normalize_last_axis("xg", "ng", 1e-5)
    nodes = [
        helper.make_node("Cast", ["X"], ["xd"], to=TensorProto.DOUBLE),
        helper.make_node("Reshape", ["xd", "rows"], ["xg"]),
        *norm_nodes,
        helper.make_node("Reshape", ["ng", "shape"], ["nr"]),
        helper.make_node("Mul", ["nr", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "B"], ["yd"]),
        helper.make_node("Cast", ["yd"], ["Y"], to=TensorProto.FLOAT),
    ]
    initializers = [
        numpy_helper.from_array(np.array([0, 0, -1], np.int64), "rows"),
        numpy_helper.from_array(np.array(input_shape, np.int64), "shape"),
        numpy_helper.from_array(
            inits["scale"].astype(np.float64).reshape(affine_shape), "scale"
        ),
        numpy_helper.from_array(
            inits["B"].astype(np.float64).reshape(affine_shape), "B"
        ),
        *norm_inits,
    ]
    return make_model_from_nodes(
        nodes,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)],
        initializers=initializers,
    )


class TestInstanceNormalization:
    """Per-(N, C) mean/var over spatial axes, then channel affine."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 8],
            [2, 3, 8, 8],
            [1, 3, 4, 5, 6],
        ],
    )
    def test_instance_norm_f32(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float32)
        rng = np.random.default_rng(7)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 16],
            [2, 3, 8, 8],
        ],
    )
    def test_instance_norm_f16(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float16)
        rng = np.random.default_rng(8)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    # Rows of 16K elements or more take GroupNorm's split reduction (one
    # channel per group), as in a Reshape(N, 32, -1) + InstanceNormalization
    # GroupNorm decomposition. The CPU reference sums long rows in fp32, so
    # the tolerance is looser than for the short rows above.
    @pytest.mark.parametrize(
        "input_shape, dtype",
        [
            ([1, 32, 36864], np.float32),
            ([2, 32, 20001], np.float32),
            ([1, 32, 147456], np.float16),
            ([1, 8, 64, 64], np.float16),
        ],
    )
    def test_instance_norm_long_rows(self, model_runner, input_shape, dtype):
        model = _make_instance_norm_model(input_shape, dtype)
        rng = np.random.default_rng(9)
        x = (rng.uniform(-1, 1, input_shape) + 2.0).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 1e-4 if dtype == np.float32 else 4e-3
        compare_outputs(actual, expected, atol=atol, rtol=atol)

    # |mean| / stddev near 260 with sparse outliers, on both the per-row
    # kernel (short rows) and the split reduction (long rows): E[x^2] - mean^2
    # in fp32 would miss the tolerance by about 7x on these inputs. The
    # reference is fp64: ORT CPU's fp32 mean of the long rows is itself off by
    # more than the tolerance.
    @pytest.mark.parametrize(
        "input_shape",
        [[2, 16, 4096], [1, 3, 17, 19], [1, 32, 36864]],
    )
    def test_instance_norm_large_offset(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float32)
        rng = np.random.default_rng(10)
        x = rng.uniform(-1, 1, input_shape) + 200.0
        x.flat[::61] += 4.0
        actual, expected = model_runner.run_sample(
            model,
            [x.astype(np.float32)],
            reference_model=_fp64_instance_norm_reference(model, input_shape),
        )
        compare_outputs(actual, expected, atol=2e-4, rtol=2e-4)


def _make_batch_norm_model(input_shape: list[int], dtype=np.float32):
    """Build an inference ONNX BatchNormalization model.

    Scale, bias, mean, and variance are per-channel initializers.
    """
    channels = input_shape[1]
    tp = {
        np.float16: TensorProto.FLOAT16,
        np.float32: TensorProto.FLOAT,
    }[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(2026)
    scale_init = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="scale"
    )
    bias_init = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="B"
    )
    mean_init = numpy_helper.from_array(
        rng.uniform(-1.0, 1.0, [channels]).astype(dtype), name="mean"
    )
    var_init = numpy_helper.from_array(
        rng.uniform(0.1, 1.5, [channels]).astype(dtype), name="var"
    )
    node = helper.make_node(
        "BatchNormalization",
        ["X", "scale", "B", "mean", "var"],
        ["Y"],
        epsilon=1e-5,
    )
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[scale_init, bias_init, mean_init, var_init],
    )


class TestBatchNormalization:
    """Inference BN: supplied per-channel mean and variance, one kernel."""

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 1, 1],
            [2, 3, 8],
            [1, 3, 4, 5],
            [1, 4, 2, 3, 3],
        ],
    )
    def test_batch_norm_f32(self, model_runner, input_shape):
        model = _make_batch_norm_model(input_shape, np.float32)
        rng = np.random.default_rng(9)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 4, 1, 1],
            [2, 3, 8, 8],
        ],
    )
    def test_batch_norm_f16(self, model_runner, input_shape):
        model = _make_batch_norm_model(input_shape, np.float16)
        rng = np.random.default_rng(10)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _plant_special_rows(x: np.ndarray, row_len: int) -> np.ndarray:
    """Row 1 gets a NaN, row 2 starts with +Inf (the kernels' shift value), and
    row 3 becomes one repeated large value."""
    rows = x.reshape(-1, row_len)
    rows[1, row_len // 2] = np.nan
    rows[2, 0] = np.inf
    rows[3, :] = 200.0
    return x


def _check_special_rows(actual, expected, row_len, constant_row_want, atol):
    """Rows 1 and 2 must be all NaN, row 3 exactly *constant_row_want* (the
    bias), and every other row must match the reference."""
    got = actual[0].reshape(-1, row_len)
    want = expected[0].reshape(-1, row_len)
    assert np.isnan(got[1]).all() and np.isnan(got[2]).all()
    np.testing.assert_allclose(got[3], constant_row_want, rtol=0, atol=1e-6)
    keep = np.r_[0, 4 : got.shape[0]]
    compare_outputs([got[keep]], [want[keep]], atol=atol, rtol=atol)


class TestNormSpecialValues:
    """NaN and Inf stay inside their normalization row, and a row without
    spread yields exactly the bias. ORT CPU's results on these rows are not
    used: its mean of a constant row is rounded, not exact."""

    @pytest.mark.parametrize("input_shape", [[2, 4, 64], [1, 32, 20000]])
    def test_instance_norm_special_values(self, model_runner, input_shape):
        model = _make_instance_norm_model(input_shape, np.float32)
        bias = numpy_helper.to_array(
            next(i for i in model.graph.initializer if i.name == "B")
        )
        rng = np.random.default_rng(11)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        x = _plant_special_rows(x, input_shape[-1])
        actual, expected = model_runner.run_sample(model, [x])
        _check_special_rows(
            actual, expected, input_shape[-1], bias[3 % input_shape[1]], 1e-4
        )

    @pytest.mark.parametrize("input_shape", [[4, 1152], [6, 4097]])
    def test_layer_norm_special_values(self, model_runner, input_shape):
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        bias = numpy_helper.to_array(
            next(i for i in model.graph.initializer if i.name == "bias")
        )
        rng = np.random.default_rng(12)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        x = _plant_special_rows(x, input_shape[-1])
        actual, expected = model_runner.run_sample(model, [x])
        _check_special_rows(actual, expected, input_shape[-1], bias, 1e-5)
