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


def _layer_norm_reference(model, x: np.ndarray, epsilon: float = 1e-5):
    """Last-axis LayerNorm in float64 with the model's scale and bias.

    ORT's CPU LayerNormalization takes the variance as E[x^2] - E[x]^2, which
    cancels for rows whose mean is large next to their spread, so it cannot
    referee those rows.
    """
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x64 - mean) / np.sqrt(var + epsilon) * weights["scale"].astype(np.float64)
    if "bias" in weights:
        y = y + weights["bias"].astype(np.float64)
    return [y.astype(x.dtype)]


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

    def test_layer_norm_fp32_large_offset(self, model_runner):
        """Rows with variance far below mean^2 (e.g. [10000, 10001]) must use a
        centered variance; E[x^2] - mean^2 cancels to 0 in fp32 and scales the
        outputs by rsqrt(epsilon)."""
        input_shape = [4, 2]
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        x = np.array(
            [[10000, 10001], [-10000, -10001], [4096, 4097], [1e5, 1e5 + 2]],
            dtype=np.float32,
        )
        actual, _ = model_runner.run_sample(model, [x])
        compare_outputs(actual, _layer_norm_reference(model, x), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("hidden", [2, 64, 1152])
    def test_layer_norm_large_mean_fp32(self, model_runner, hidden):
        """Rows whose mean is large next to their spread.

        A one-pass E[x^2] - E[x]^2 variance cancels here: x = [8192, 8193]
        gives 0 instead of 0.25 in fp32 and outputs of about +-158 instead of
        +-1. hidden=2 is exactly that row. One fp32 ulp at 8192 is ~1e-3, so
        fp32 rounding of the row costs a few 1e-3 after dividing by the ~0.3
        stddev.
        """
        input_shape = [3, hidden]
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        rng = np.random.default_rng(1106)
        x = (8192.0 + rng.uniform(-1, 1, input_shape)).astype(np.float32)
        if hidden == 2:
            x[0] = [8192.0, 8193.0]
        actual, _ = model_runner.run_sample(model, [x])
        compare_outputs(actual, _layer_norm_reference(model, x), atol=5e-3, rtol=1e-3)

    def test_layer_norm_large_magnitude_fp32(self, model_runner):
        """Rows whose plain sum of squares is finite but near the fp32 limit.

        Squaring x - x[0] overflows for x = [1e19, -1e19] (the second shifted
        value is -2e19), which would zero both outputs instead of giving about
        +-1 times scale.
        """
        input_shape = [2, 2]
        model = _make_layer_norm_model(input_shape, np.float32, with_bias=True)
        x = np.array([[1e19, -1e19], [-1e19, 1e19]], dtype=np.float32)
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


class TestLayerNormKernelSizes:
    """Row-kernel dispatch: one wave per row up to 256 16-byte chunks, wider
    groups above that, and the block-per-row fallback for ragged or too-wide
    rows. Odd row counts leave the last block partly empty."""

    @pytest.mark.parametrize(
        "rows,hidden",
        [(13, 768), (5, 1024), (7, 1280), (3, 2048), (2, 8192), (3, 1004)],
    )
    def test_layer_norm_sizes_fp16(self, model_runner, rows, hidden):
        model = _make_layer_norm_model([rows, hidden], np.float16, with_bias=True)
        rng = np.random.default_rng(1201)
        x = rng.uniform(-2, 2, [rows, hidden]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=4e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "rows,hidden",
        [(9, 768), (5, 1280), (3, 4096), (2, 16384), (2, 16388), (3, 1001)],
    )
    def test_layer_norm_sizes_fp32(self, model_runner, rows, hidden):
        model = _make_layer_norm_model([rows, hidden], np.float32, with_bias=True)
        rng = np.random.default_rng(1202)
        x = rng.uniform(-2, 2, [rows, hidden]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "dtype,hidden", [(np.float32, 768), (np.float16, 1280), (np.float32, 33)]
    )
    def test_layer_norm_mean_inv_std(self, model_runner, dtype, hidden):
        """Mean and InvStdDev outputs from the row kernel and the fallback."""
        rows = 6
        tp = _TENSOR_PROTO[dtype]
        X = helper.make_tensor_value_info("X", tp, [rows, hidden])
        Y = helper.make_tensor_value_info("Y", tp, [rows, hidden])
        M = helper.make_tensor_value_info("M", TensorProto.FLOAT, [rows, 1])
        R = helper.make_tensor_value_info("R", TensorProto.FLOAT, [rows, 1])
        rng = np.random.default_rng(1203)
        scale = numpy_helper.from_array(
            rng.uniform(0.5, 1.5, [hidden]).astype(dtype), name="scale"
        )
        node = helper.make_node(
            "LayerNormalization",
            ["X", "scale"],
            ["Y", "M", "R"],
            axis=-1,
            epsilon=1e-5,
            stash_type=1,
        )
        model = make_model_from_nodes([node], [X], [Y, M, R], initializers=[scale])
        x = (3.0 + rng.uniform(-2, 2, [rows, hidden])).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 1e-5 if dtype == np.float32 else 4e-3
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)


def _make_decomposed_layer_norm_model(
    input_shape: list[int],
    dtype: np.dtype,
    spelling: str,
    *,
    affine: str = "scale_bias",
    residual: bool = False,
):
    """LayerNorm over the last axis written as primitives.

    spelling picks the normalization step, with a = var + eps:
      div_sqrt   d / Sqrt(a)
      mul_recip  d * Reciprocal(Sqrt(a))
      mul_pow    d * Pow(a, -0.5)
      div_pow    d / Pow(a, 0.5)
    affine is "scale_bias", "scale" or "none". With residual=True the norm
    input is X + R and that sum is a second graph output.
    """
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    rng = np.random.default_rng(1301)
    graph_inputs = [helper.make_tensor_value_info("X", tp, input_shape)]
    outputs = [helper.make_tensor_value_info("Y", tp, input_shape)]
    inits = [
        numpy_helper.from_array(np.array(1e-5, dtype=dtype), name="eps"),
        numpy_helper.from_array(np.array(2.0, dtype=dtype), name="two"),
        numpy_helper.from_array(np.array(0.5, dtype=dtype), name="half"),
        numpy_helper.from_array(np.array(-0.5, dtype=dtype), name="neg_half"),
        numpy_helper.from_array(
            rng.uniform(0.5, 1.5, [hidden]).astype(dtype), name="gamma"
        ),
        numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [hidden]).astype(dtype), name="beta"
        ),
    ]
    nodes = []
    x = "X"
    if residual:
        graph_inputs.append(helper.make_tensor_value_info("R", tp, input_shape))
        outputs.append(helper.make_tensor_value_info("S", tp, input_shape))
        nodes.append(helper.make_node("Add", ["X", "R"], ["S"]))
        x = "S"
    nodes += [
        helper.make_node("ReduceMean", [x], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", [x, "mean"], ["d"]),
        helper.make_node("Pow", ["d", "two"], ["sq"]),
        helper.make_node("ReduceMean", ["sq"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["a"]),
    ]
    if spelling == "div_sqrt":
        nodes += [
            helper.make_node("Sqrt", ["a"], ["s"]),
            helper.make_node("Div", ["d", "s"], ["n"]),
        ]
    elif spelling == "mul_recip":
        nodes += [
            helper.make_node("Sqrt", ["a"], ["s"]),
            helper.make_node("Reciprocal", ["s"], ["r"]),
            helper.make_node("Mul", ["d", "r"], ["n"]),
        ]
    elif spelling == "mul_pow":
        nodes += [
            helper.make_node("Pow", ["a", "neg_half"], ["r"]),
            helper.make_node("Mul", ["r", "d"], ["n"]),
        ]
    elif spelling == "div_pow":
        nodes += [
            helper.make_node("Pow", ["a", "half"], ["s"]),
            helper.make_node("Div", ["d", "s"], ["n"]),
        ]
    else:
        raise ValueError(spelling)
    if affine == "none":
        nodes.append(helper.make_node("Identity", ["n"], ["Y"]))
    elif affine == "scale":
        nodes.append(helper.make_node("Mul", ["n", "gamma"], ["Y"]))
    else:
        nodes += [
            helper.make_node("Mul", ["n", "gamma"], ["m"]),
            helper.make_node("Add", ["m", "beta"], ["Y"]),
        ]
    return make_model_from_nodes(nodes, graph_inputs, outputs, initializers=inits)


class TestDecomposedLayerNorm:
    """Primitive LayerNorm chains fold into one hip.layer_norm (or
    hip.skip_layer_norm with a residual Add). The CPU reference runs the
    primitives op by op, so fp16 tolerances cover its per-op rounding."""

    @pytest.mark.parametrize(
        "spelling", ["div_sqrt", "mul_recip", "mul_pow", "div_pow"]
    )
    @pytest.mark.parametrize("affine", ["scale_bias", "scale", "none"])
    def test_spellings_fp32(self, model_runner, spelling, affine):
        shape = [2, 8, 768]
        model = _make_decomposed_layer_norm_model(
            shape, np.float32, spelling, affine=affine
        )
        rng = np.random.default_rng(1302)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("spelling", ["div_sqrt", "mul_recip", "mul_pow"])
    def test_spellings_fp16(self, model_runner, spelling):
        shape = [1, 16, 1280]
        model = _make_decomposed_layer_norm_model(shape, np.float16, spelling)
        rng = np.random.default_rng(1303)
        x = rng.uniform(-2, 2, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "dtype,hidden", [(np.float32, 768), (np.float16, 1024), (np.float32, 30)]
    )
    def test_residual_add(self, model_runner, dtype, hidden):
        """Residual Add + LayerNorm: both the norm and the sum are outputs."""
        shape = [1, 12, hidden]
        model = _make_decomposed_layer_norm_model(
            shape, dtype, "mul_recip", residual=True
        )
        rng = np.random.default_rng(1304)
        x = rng.uniform(-2, 2, shape).astype(dtype)
        r = rng.uniform(-2, 2, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, r])
        atol = 1e-4 if dtype == np.float32 else 1e-2
        compare_outputs(actual, expected, atol=atol, rtol=atol)


def _make_residual_layer_norm_model(input_shape: list[int], dtype: np.dtype):
    """Add(X, R) -> LayerNormalization, with the sum consumed afterwards."""
    hidden = input_shape[-1]
    tp = _TENSOR_PROTO[dtype]
    rng = np.random.default_rng(1401)
    nodes = [
        helper.make_node("Add", ["X", "R"], ["S"]),
        helper.make_node(
            "LayerNormalization", ["S", "gamma", "beta"], ["Y"], axis=-1, epsilon=1e-5
        ),
        helper.make_node("Add", ["S", "Y"], ["Z"]),
    ]
    return make_model_from_nodes(
        nodes,
        [
            helper.make_tensor_value_info("X", tp, input_shape),
            helper.make_tensor_value_info("R", tp, input_shape),
        ],
        [helper.make_tensor_value_info("Z", tp, input_shape)],
        initializers=[
            numpy_helper.from_array(
                rng.uniform(0.5, 1.5, [hidden]).astype(dtype), name="gamma"
            ),
            numpy_helper.from_array(
                rng.uniform(-0.5, 0.5, [hidden]).astype(dtype), name="beta"
            ),
        ],
    )


class TestResidualLayerNorm:
    """onnx.LayerNormalization fed by a residual Add -> hip.skip_layer_norm."""

    @pytest.mark.parametrize(
        "dtype,hidden",
        [(np.float16, 768), (np.float32, 1280), (np.float16, 1001)],
    )
    def test_residual_layer_normalization(self, model_runner, dtype, hidden):
        shape = [2, 7, hidden]
        model = _make_residual_layer_norm_model(shape, dtype)
        rng = np.random.default_rng(1402)
        x = rng.uniform(-2, 2, shape).astype(dtype)
        r = rng.uniform(-2, 2, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x, r])
        atol = 1e-5 if dtype == np.float32 else 4e-3
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)


class TestSkipLayerNormSizes:
    """Fused SkipLayerNormalization across the row kernel and the fallback."""

    @pytest.mark.parametrize("input_shape", [[1, 64, 768], [2, 9, 1280], [1, 3, 1004]])
    def test_skip_layer_norm_sizes(self, model_runner, input_shape):
        model = _make_skip_layer_norm_model(input_shape)
        rng = np.random.default_rng(1501)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        skip_input = rng.uniform(-2, 2, input_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x, skip_input])
        compare_outputs(actual, expected, atol=4e-3, rtol=1e-3)


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


def _make_group_norm_model(
    input_shape: list[int],
    groups: int,
    *,
    channels_last: int = 0,
    activation: int = 0,
    dtype=np.float32,
    epsilon: float = 1e-5,
):
    """Build a com.microsoft.GroupNorm model. Channel axis is 1, or -1 when
    channels_last is set."""
    channels = input_shape[-1] if channels_last else input_shape[1]
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Y = helper.make_tensor_value_info("Y", tp, input_shape)
    rng = np.random.default_rng(11)
    gamma = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [channels]).astype(dtype), name="gamma"
    )
    beta = numpy_helper.from_array(
        rng.uniform(-0.5, 0.5, [channels]).astype(dtype), name="beta"
    )
    node = helper.make_node(
        "GroupNorm",
        ["X", "gamma", "beta"],
        ["Y"],
        domain="com.microsoft",
        groups=groups,
        epsilon=epsilon,
        activation=activation,
        channels_last=channels_last,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[gamma, beta],
        extra_opsets=[ms_opset],
    )


class TestGroupNorm:
    """com.microsoft GroupNorm: per-group stats, optional SiLU."""

    def test_group_norm_nchw_f32(self, model_runner):
        shape = [1, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=4, activation=0)
        rng = np.random.default_rng(12)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nchw_silu_f32(self, model_runner):
        shape = [2, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=4, activation=1)
        rng = np.random.default_rng(13)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nhwc_f32(self, model_runner):
        shape = [1, 4, 4, 8]
        model = _make_group_norm_model(shape, groups=4, channels_last=1, activation=0)
        rng = np.random.default_rng(14)
        x = rng.uniform(-2, 2, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_group_norm_nchw_silu_f16(self, model_runner):
        shape = [1, 8, 4, 4]
        model = _make_group_norm_model(shape, groups=2, activation=1, dtype=np.float16)
        rng = np.random.default_rng(15)
        x = rng.uniform(-2, 2, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)


def _make_embed_layer_norm_model(
    batch: int,
    seq: int,
    hidden: int,
    *,
    with_position_ids: bool,
):
    """com.microsoft.EmbedLayerNormalization over a small f32 table.

    The CPU reference counts ones in the mask. Position ids are either an
    explicit input or omitted, in which case positions are 0 .. seq-1.
    """
    vocab = 32
    max_pos = seq + 2
    ids = helper.make_tensor_value_info("ids", TensorProto.INT32, [batch, seq])
    segments = helper.make_tensor_value_info(
        "segments", TensorProto.INT32, [batch, seq]
    )
    mask = helper.make_tensor_value_info("mask", TensorProto.INT32, [batch, seq])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch, seq, hidden]
    )
    mask_index = helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch])
    rng = np.random.default_rng(21)
    initializers = [
        numpy_helper.from_array(
            rng.uniform(-1, 1, [vocab, hidden]).astype(np.float32), name="word"
        ),
        numpy_helper.from_array(
            rng.uniform(-1, 1, [max_pos, hidden]).astype(np.float32),
            name="position",
        ),
        numpy_helper.from_array(
            rng.uniform(-1, 1, [2, hidden]).astype(np.float32), name="segment"
        ),
        numpy_helper.from_array(
            rng.uniform(0.5, 1.5, [hidden]).astype(np.float32), name="gamma"
        ),
        numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, [hidden]).astype(np.float32), name="beta"
        ),
    ]
    inputs = ["ids", "segments", "word", "position", "segment", "gamma", "beta", "mask"]
    graph_inputs = [ids, segments, mask]
    if with_position_ids:
        position_ids = helper.make_tensor_value_info(
            "position_ids", TensorProto.INT32, [1, seq]
        )
        inputs.append("position_ids")
        graph_inputs.append(position_ids)
    node = helper.make_node(
        "EmbedLayerNormalization",
        inputs,
        ["output", "mask_index"],
        domain="com.microsoft",
        epsilon=1e-12,
    )
    return make_model_from_nodes(
        [node],
        graph_inputs,
        [output, mask_index],
        initializers=initializers,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestEmbedLayerNorm:
    """Gather + add + LayerNormalization decomposition of the embedding fusion."""

    def test_embed_layer_norm_with_position_ids(self, model_runner):
        batch, seq, hidden = 2, 6, 8
        model = _make_embed_layer_norm_model(batch, seq, hidden, with_position_ids=True)
        rng = np.random.default_rng(22)
        ids = rng.integers(0, 32, size=[batch, seq], dtype=np.int32)
        segments = rng.integers(0, 2, size=[batch, seq], dtype=np.int32)
        mask = np.ones([batch, seq], dtype=np.int32)
        mask[:, 4:] = 0
        position_ids = np.array([[0, 2, 1, 3, 0, 0]], dtype=np.int32)
        actual, expected = model_runner.run_sample(
            model, [ids, segments, mask, position_ids]
        )
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    def test_embed_layer_norm_default_positions(self, model_runner):
        batch, seq, hidden = 1, 5, 8
        model = _make_embed_layer_norm_model(
            batch, seq, hidden, with_position_ids=False
        )
        rng = np.random.default_rng(23)
        ids = rng.integers(0, 32, size=[batch, seq], dtype=np.int32)
        segments = rng.integers(0, 2, size=[batch, seq], dtype=np.int32)
        mask = np.array([[1, 1, 1, 0, 0]], dtype=np.int32)
        actual, expected = model_runner.run_sample(model, [ids, segments, mask])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)
