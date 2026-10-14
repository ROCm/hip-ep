#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the MatMulNBits custom op (com.microsoft).

MatMulNBits performs matrix multiplication with N-bit quantized weights.
Weights are stored as uint8 (4-bit packed) with per-block f16 scales.

Two model families are covered:

GPT-OSS-20B (block_size = 32, accuracy_level = 4):
  hidden = 2880, attn_hidden = 4096 (num_heads * head_dim)
  qkv_out = 5120 (q: 4096, k: 512, v: 512 fused)
  num_experts = 32 (router output)
  vocab = 201088 (lm_head output)

  Scale magnitudes (from real model weights):
    qkv:     |scale| mean ~ 0.021,  max ~ 0.25
    o_proj:  |scale| mean ~ 0.002,  max ~ 0.39
    router:  |scale| mean ~ 0.008,  max ~ 0.03
    lm_head: |scale| mean ~ 0.002,  max ~ 0.01

Llama-3.1-8B AWQ-int4-g128 (block_size = 128, accuracy_level = 4):
  hidden = 4096, kv_hidden = 1024, head_dim = 128 (32 Q / 8 KV heads, GQA 4:1)
  intermediate = 14336 (SwiGLU MLP)
  vocab = 128256 (lm_head output)

  Shapes per decoder layer (K -> N):
    q_proj / o_proj      : 4096 ->  4096
    k_proj / v_proj      : 4096 ->  1024
    gate_proj / up_proj  : 4096 -> 14336
    down_proj            : 14336 -> 4096
    lm_head              : 4096 -> 128256
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# 1 is the row-major GEMV, 128 the WMMA path. 2/7/15 cover the tier between
# them, where M is below the 16-row WMMA fragment so every tile is a partial
# one -- the case that depends on the bounds-checked kernel variant.
SEQ_LENS = [1, 2, 7, 15, 128]

# --- GPT-OSS-20B constants ---
HIDDEN = 2880
ATTN_HIDDEN = 4096
QKV_OUT = 5120
NUM_EXPERTS = 32
VOCAB = 201088

BITS = 4
BLOCK_SIZE = 32

# --- Llama-3.1-8B AWQ-int4-g128 constants ---
# Source: Llama-3.1-8B-awq-g128-int4-onnx-directml/model.onnx
LLAMA_HIDDEN = 4096
LLAMA_KV_HIDDEN = 1024
LLAMA_INTERMEDIATE = 14336
LLAMA_VOCAB = 128256
LLAMA_BLOCK_SIZE = 128

# --- Qwen3.5 mixed bits=4 / bits=8 model constants ---
# Two real Qwen3.5 model variants ship with mixed-precision MatMulNBits
# graphs.  Both default to zero_point=128 (no explicit zp tensor) and
# block_size=128 for every 8-bit op, so a single shared block size and
# bits constant cover both models.  Per-model shape constants live with
# the corresponding test sections below.
QWEN_INT8_BLOCK_SIZE = 128
QWEN_INT8_BITS = 8

# Sequence lengths exercised for every int8 case below -- capped to the
# project-wide {1, 128} sweep used everywhere else in this suite.
QWEN_INT8_SEQ_LENS = [1, 128]

# --- Qwen3.5-35B-A3B int4_rtn_128gs (MoE) ---
# Source: Qwen3.5-35B-A3B_int4_rtn_128gs_cuda/text.onnx -- the model
# mixes bits=4 (241 ops) and bits=8 (150 ops) in the same graph.  The
# 8-bit ops cluster around four shapes (K, N, block_size):
#   (2048,   32, 128) x60  -- MoE router-style narrow-N projection
#   (2048, 8192, 128) x30  -- MoE expert up_proj-style fan-out
#   (2048, 4096, 128) x30  -- attention out / fan-in projection
#   (4096, 2048, 128) x30  -- attention QKV / fan-in projection
QWEN_35B_INT8_HIDDEN = 2048

# --- Qwen3.5-9B-rtn-int4-int8-128gs (dense) ---
# Source: Qwen3.5-9B-rtn-int4-int8-128gs-fp16-onnx-gpu/text.onnx -- the
# model leans even more heavily on bits=8 (192 of 249 MatMulNBits ops,
# 77%).  All bits=8 ops use block_size=128 and default zp=128.  Five
# unique shapes are observed; together they exercise both the very
# wide-N (12288 = 3 * intermediate) and very wide-K (12288, down_proj)
# accumulation paths the 35B suite doesn't reach:
#   (4096,    32, 128) x48  -- router-style narrow-N
#   (4096,  4096, 128) x48  -- attention projection (square)
#   (4096,  8192, 128) x24  -- gate / up_proj fan-out
#   (4096, 12288, 128) x48  -- combined gate_up fan-out
#   (12288, 4096, 128) x24  -- down_proj wide-K accumulation
QWEN_9B_INT8_HIDDEN = 4096
QWEN_9B_INT8_INTERMEDIATE = 12288


def _make_matmul_nbits_model(
    batch: int,
    seq_len: int,
    K: int,
    N: int,
    scale_range: tuple[float, float] = (-0.05, 0.05),
    bits: int = 4,
    block_size: int = 32,
    accuracy_level: int = 4,
    seed: int = 42,
    fp32: bool = False,
):
    """Build a MatMulNBits ONNX model with random quantized weights.

    For bits=4 the weight tensor is packed two-nibbles-per-byte:
        Weight   shape: [N, ceil(K / block_size), block_size // 2]
        Scales   shape: [N, ceil(K / block_size)]

    For bits=8 each weight is one byte (no packing):
        Weight   shape: [N, ceil(K / block_size), block_size]
        Scales   shape: [N, ceil(K / block_size)]

    ``scale_range`` controls the uniform distribution used to generate
    per-block quantization scales -- set it to match the target projection.

    ``fp32``: when True, X/Y and the scales initializer are fp32 instead of
    fp16 (ONNX MatMulNBits allows either dtype for scales, independent of the
    packed-weight dtype -- e.g. the PSU_MF_LORA lm_head submodel uses fp32
    activations with fp32 scales). Regression coverage for the bug where the
    HIP EP always read `scales` as raw fp16 regardless of its actual dtype,
    producing NaN/Inf logits for fp32-scale models.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    n_blocks = (K + block_size - 1) // block_size
    if bits == 4:
        weights_per_byte = 2
        weight_name = "weight_Q4"
    else:
        weights_per_byte = 1
        weight_name = "weight_Q8"
    bytes_per_block = block_size // weights_per_byte

    onnx_dtype = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    np_dtype = np.float32 if fp32 else np.float16

    x = helper.make_tensor_value_info(
        "X",
        onnx_dtype,
        [batch, seq_len, K],
    )
    y = helper.make_tensor_value_info(
        "Y",
        onnx_dtype,
        [batch, seq_len, N],
    )

    rng = np.random.default_rng(seed)
    q_weight = rng.integers(
        0,
        256,
        [N, n_blocks, bytes_per_block],
        dtype=np.uint8,
    )
    scales = rng.uniform(
        scale_range[0],
        scale_range[1],
        [N, n_blocks],
    ).astype(np_dtype)

    initializers = [
        numpy_helper.from_array(q_weight, name=weight_name),
        numpy_helper.from_array(scales, name="weight_scales"),
    ]

    node = helper.make_node(
        "MatMulNBits",
        ["X", weight_name, "weight_scales"],
        ["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        accuracy_level=accuracy_level,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [x],
        [y],
        initializers=initializers,
        opset=21,
        extra_opsets=[ms_opset],
    )


class TestMatMulNBits:
    @pytest.mark.parametrize(
        "K,N",
        [
            (64, 32),
            (128, 64),
        ],
    )
    def test_matmul_nbits(self, model_runner, K, N):
        """MatMulNBits with small shapes for fast sanity checking."""
        model = _make_matmul_nbits_model(1, 4, K, N, scale_range=(-0.05, 0.05))

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, 4, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [1, 8, 64])
    def test_matmul_nbits_fp32_activation_fp32_scale(self, model_runner, seq_len):
        """fp32 activation + fp32 scale (PSU_MF_LORA lm_head shape/dtype).

        Regression test: ONNX MatMulNBits allows `scales` to be fp16 or fp32
        independent of the packed-weight dtype. The HIP EP used to always
        read `scales` as raw fp16 regardless of its declared dtype, so a
        fp32-scale model (e.g. PSU_MF_LORA_lm_head.onnx: fp32 hidden states,
        fp32 scales, K=2048, N=200029, bits=4, block_size=32) produced
        NaN/Inf logits on GPU while the CPU EP was correct. Uses a smaller N
        than the real model to keep the test fast.
        """
        K, N = 64, 256
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            K,
            N,
            scale_range=(-0.01, 0.01),
            fp32=True,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x])
        assert np.isfinite(actual[0]).all(), "GPU EP produced NaN/Inf logits"
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_qkv_gpt_oss_shape(self, model_runner, seq_len):
        """QKV fused projection: [1, S, 2880] -> [1, S, 5120].

        Input: post-layernorm hidden states (~ normalised, range [-1, 1]).
        Scales: mean |s| ~ 0.021, max ~ 0.25 (widest among projections).
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            HIDDEN,
            QKV_OUT,
            scale_range=(-0.05, 0.05),
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_o_proj_gpt_oss_shape(self, model_runner, seq_len):
        """Output projection: [1, S, 4096] -> [1, S, 2880].

        Input: post-attention hidden states (range [-2, 2]).
        Scales: mean |s| ~ 0.002, but sparse outliers up to 0.39.
        Larger K (4096) means more accumulation error.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            ATTN_HIDDEN,
            HIDDEN,
            scale_range=(-0.01, 0.01),
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-2, 2, [1, seq_len, ATTN_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_router_gpt_oss_shape(self, model_runner, seq_len):
        """MoE router: [1, S, 2880] -> [1, S, 32].

        Input: post-layernorm hidden states (range [-1, 1]).
        Scales: small, mean |s| ~ 0.008, max ~ 0.03.
        Small N (32) so output is compact.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            HIDDEN,
            NUM_EXPERTS,
            scale_range=(-0.03, 0.03),
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_lm_head_gpt_oss_shape(self, model_runner, seq_len):
        """LM head (vocab projection): [1, S, 2880] -> [1, S, 201088].

        Input: post-final-layernorm hidden states (range [-1, 1]).
        Scales: very small, mean |s| ~ 0.002, max ~ 0.01.
        WARNING: ~310 MB of quantized weight initializers.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            HIDDEN,
            VOCAB,
            scale_range=(-0.01, 0.01),
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2, cos_threshold=0.999)

    # ------------------------------------------------------------------
    # Llama-3.1-8B AWQ-int4-g128 single-node shapes
    #
    # The full decoder model has 225 MatMulNBits nodes (32 layers x 7 per
    # layer + lm_head).  All use bits=4, block_size=128, accuracy_level=4.
    # block_size=128 is 4x the GPT-OSS g32 groups, so these tests exercise
    # a distinct code path in the WMMA / GEMV kernels.
    #
    # seq_len=1 hits the GEMV (decode) path; seq_len=128 crosses into the
    # WMMA (prefill / long-context) path.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_q_proj_llama_shape(self, model_runner, seq_len):
        """Llama q_proj: [1, S, 4096] -> [1, S, 4096], block_size=128.

        Same shape as o_proj; both drive attention-side GEMMs with K=4096
        accumulation.  AWQ scales are generally tight -- use (-0.01, 0.01).
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_HIDDEN,
            LLAMA_HIDDEN,
            scale_range=(-0.01, 0.01),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_kv_proj_llama_shape(self, model_runner, seq_len):
        """Llama k_proj / v_proj: [1, S, 4096] -> [1, S, 1024], block_size=128.

        Narrow-N GQA KV projection (num_key_value_heads=8, head_dim=128).
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            scale_range=(-0.01, 0.01),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_o_proj_llama_shape(self, model_runner, seq_len):
        """Llama o_proj: [1, S, 4096] -> [1, S, 4096], block_size=128.

        Consumes post-attention hidden states whose magnitude is larger
        than normalized LN output -- use range [-2, 2] to stress the
        accumulator.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_HIDDEN,
            LLAMA_HIDDEN,
            scale_range=(-0.01, 0.01),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-2, 2, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_gate_up_proj_llama_shape(self, model_runner, seq_len):
        """Llama gate_proj / up_proj: [1, S, 4096] -> [1, S, 14336], block_size=128.

        Wide-N SwiGLU MLP projection; covers both gate and up since they
        share shape and scale layout.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_HIDDEN,
            LLAMA_INTERMEDIATE,
            scale_range=(-0.01, 0.01),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_down_proj_llama_shape(self, model_runner, seq_len):
        """Llama down_proj: [1, S, 14336] -> [1, S, 4096], block_size=128.

        Largest-K projection in the model (K=14336); stresses in-kernel
        FP16 accumulation.  Scale range tightened to keep outputs in-range.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_INTERMEDIATE,
            LLAMA_HIDDEN,
            scale_range=(-0.005, 0.005),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_INTERMEDIATE]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_nbits_lm_head_llama_shape(self, model_runner, seq_len):
        """Llama lm_head: [1, S, 4096] -> [1, S, 128256], block_size=128.

        Vocab projection.  ~260 MB of quantized weights; runs fine at
        seq_len=1 and 128 but is the heaviest non-slow test.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            LLAMA_HIDDEN,
            LLAMA_VOCAB,
            scale_range=(-0.005, 0.005),
            block_size=LLAMA_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2, cos_threshold=0.999)

    # ------------------------------------------------------------------
    # Qwen3.5-35B-A3B int4_rtn_128gs single-node shapes (bits=8 ops)
    #
    # The model mixes bits=4 and bits=8 MatMulNBits in the same graph.
    # Until we land 8-bit support every bits=8 op fails at runtime, so
    # these tests double as a gate: they compile and run a single 8-bit
    # MatMulNBits node and compare against the CPU reference.  All four
    # canonical shapes are covered at decode (S=1) and prefill-ish
    # (S=128) so we exercise both the small-M and wide-M dispatch paths.
    #
    # Tolerance notes:
    # The int8 path accumulates K up to 4096 partials of (uint8-128) *
    # f16 input * f16 scale in fp32.  Outputs reach |y| ~ 20-40 so an
    # absolute atol of ~0.2 is needed to cover f16 round-off; rtol stays
    # at 1e-2 (~0.4% relative error) and cosine >= 0.9995 catches any
    # real correctness regression.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen35b_router_shape_int8(self, model_runner, seq_len):
        """Qwen-35B int8 router-style: [1, S, 2048] -> [1, S, 32], block_size=128."""
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_35B_INT8_HIDDEN,
            32,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_35B_INT8_HIDDEN]).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen35b_expert_up_shape_int8(self, model_runner, seq_len):
        """Qwen-35B int8 expert up_proj-style: [1, S, 2048] -> [1, S, 8192], block_size=128.

        Wide-N fan-out -- exercises the int8 naive kernel's grid scaling
        in the N dimension (largest unique 8-bit shape in the 35B model).
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_35B_INT8_HIDDEN,
            8192,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_35B_INT8_HIDDEN]).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen35b_attn_out_shape_int8(self, model_runner, seq_len):
        """Qwen-35B int8 attn-out-style: [1, S, 2048] -> [1, S, 4096], block_size=128."""
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_35B_INT8_HIDDEN,
            4096,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_35B_INT8_HIDDEN]).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen35b_attn_qkv_shape_int8(self, model_runner, seq_len):
        """Qwen-35B int8 attn-qkv-style: [1, S, 4096] -> [1, S, 2048], block_size=128.

        Wide-K accumulation over K=4096 with int8 weights and zp=128 default.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            4096,
            QWEN_35B_INT8_HIDDEN,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, 4096]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    # ------------------------------------------------------------------
    # Qwen3.5-9B-rtn-int4-int8-128gs single-node shapes (bits=8 ops)
    #
    # The 9B dense variant leans even more heavily on bits=8 (192 of
    # 249 MatMulNBits ops, 77%).  All bits=8 ops use block_size=128 and
    # default zp=128.  Compared with the 35B section above, these tests
    # add coverage for two regimes the 35B suite doesn't reach:
    #   * very wide-N fan-out (N=12288, the combined gate_up width)
    #   * very wide-K accumulation (K=12288, down_proj)
    #
    # Tolerance notes:
    # K up to 12288 means roughly 3x the round-off budget of the 35B
    # cases.  Outputs reach |y| ~ 60-70, max absolute diffs ~0.28; the
    # relative error stays well under 0.5% (rtol=1e-2) and cosine
    # similarity remains >= 0.999992 across every shape, so we widen
    # atol to 0.3 only for the K=12288 down_proj case.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen9b_router_shape_int8(self, model_runner, seq_len):
        """Qwen-9B int8 router-style: [1, S, 4096] -> [1, S, 32], block_size=128."""
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_9B_INT8_HIDDEN,
            32,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_9B_INT8_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen9b_attn_proj_shape_int8(self, model_runner, seq_len):
        """Qwen-9B int8 attn projection (square): [1, S, 4096] -> [1, S, 4096], block_size=128."""
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_9B_INT8_HIDDEN,
            QWEN_9B_INT8_HIDDEN,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_9B_INT8_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen9b_up_proj_shape_int8(self, model_runner, seq_len):
        """Qwen-9B int8 up_proj-style: [1, S, 4096] -> [1, S, 8192], block_size=128."""
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_9B_INT8_HIDDEN,
            8192,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_9B_INT8_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen9b_gate_up_shape_int8(self, model_runner, seq_len):
        """Qwen-9B int8 combined gate_up: [1, S, 4096] -> [1, S, 12288], block_size=128.

        Widest-N case in the model (3x intermediate fan-out) -- stresses
        the int8 naive kernel's per-N parallelism on the GPU grid.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_9B_INT8_HIDDEN,
            QWEN_9B_INT8_INTERMEDIATE,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_9B_INT8_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-1, rtol=1e-2, cos_threshold=0.9995)

    @pytest.mark.parametrize("seq_len", QWEN_INT8_SEQ_LENS)
    def test_matmul_nbits_qwen9b_down_proj_shape_int8(self, model_runner, seq_len):
        """Qwen-9B int8 down_proj-style: [1, S, 12288] -> [1, S, 4096], block_size=128.

        Widest-K case in the model (3x hidden accumulation).  fp32
        accumulation over K=12288 partials of (uint8-128)*f16*f16
        scale produces |y| ~ 65 and max abs-diffs near 0.28, so this
        case uses a slightly looser atol than the rest of the suite.
        Cosine similarity stays at 0.999992 -- the comparator's strict
        guardrail confirms correctness.
        """
        model = _make_matmul_nbits_model(
            1,
            seq_len,
            QWEN_9B_INT8_INTERMEDIATE,
            QWEN_9B_INT8_HIDDEN,
            scale_range=(-0.01, 0.01),
            bits=QWEN_INT8_BITS,
            block_size=QWEN_INT8_BLOCK_SIZE,
        )

        rng = np.random.default_rng(99)
        x = rng.uniform(-0.5, 0.5, [1, seq_len, QWEN_9B_INT8_INTERMEDIATE]).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=3e-1, rtol=1e-2, cos_threshold=0.9995)


# --- Projections sharing one activation ---
# Separate q/k/v and gate/up MatMulNBits nodes reading one activation, the MLP
# combined as Mul(Mul(gate, Sigmoid(gate)), up). Shapes from the Llama-2-13B
# and Phi-3.5-mini int4 decoders; Llama-3.1-8B AWQ adds asymmetric weights
# (packed uint8 zero points, block_size 128).
LLAMA2_HIDDEN = 5120
LLAMA2_INTERMEDIATE = 13824
PHI35_HIDDEN = 3072
PHI35_INTERMEDIATE = 8192

# 1 and 7 are on the W4A8 dp4a path (HIPDNN_EP_MATMUL_DP4A_MAX_M defaults to
# 7) and read the shared int8 activation; 128 takes the prefill kernels and
# reads the fp16 activation, with the int8 copy written but unused.
DP4A_MAX_M = 7
SHARED_ACT_SEQ_LENS = [1, DP4A_MAX_M, 128]


def _q4_initializers(
    rng,
    prefix: str,
    K: int,
    N: int,
    scale_range=(-0.02, 0.02),
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
):
    n_blocks = K // block_size
    q_weight = rng.integers(0, 256, [N, n_blocks, block_size // 2], dtype=np.uint8)
    scales = rng.uniform(scale_range[0], scale_range[1], [N, n_blocks]).astype(
        np.float16
    )
    inits = [
        numpy_helper.from_array(q_weight, name=f"{prefix}_Q4"),
        numpy_helper.from_array(scales, name=f"{prefix}_scales"),
    ]
    if asym:
        # Two 4-bit zero points per byte, each row padded to whole bytes.
        zp = rng.integers(0, 256, [N * ((n_blocks + 1) // 2)], dtype=np.uint8)
        inits.append(numpy_helper.from_array(zp, name=f"{prefix}_zp"))
    return inits


def _q4_node(
    prefix: str,
    x: str,
    y: str,
    K: int,
    N: int,
    block_size: int = BLOCK_SIZE,
    asym: bool = False,
):
    inputs = [x, f"{prefix}_Q4", f"{prefix}_scales"]
    if asym:
        inputs.append(f"{prefix}_zp")
    return helper.make_node(
        "MatMulNBits",
        inputs,
        [y],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=BITS,
        block_size=block_size,
        accuracy_level=4,
    )


def _qkv_nodes(rng, x: str, K: int, n_q: int, n_kv: int, seq_len: int, **q4):
    nodes, inits, outs = [], [], []
    for name, n in (("q", n_q), ("k", n_kv), ("v", n_kv)):
        inits += _q4_initializers(rng, name, K, n, **q4)
        nodes.append(_q4_node(name, x, name.upper(), K, n, **q4))
        outs.append(
            helper.make_tensor_value_info(
                name.upper(), TensorProto.FLOAT16, [1, seq_len, n]
            )
        )
    return nodes, inits, outs


def _gate_up_silu_nodes(rng, x: str, K: int, N: int, seq_len: int, **q4):
    inits = _q4_initializers(rng, "gate", K, N, **q4) + _q4_initializers(
        rng, "up", K, N, **q4
    )
    nodes = [
        _q4_node("gate", x, "G", K, N, **q4),
        _q4_node("up", x, "U", K, N, **q4),
        helper.make_node("Sigmoid", ["G"], ["S"]),
        helper.make_node("Mul", ["G", "S"], ["GS"]),
        helper.make_node("Mul", ["GS", "U"], ["Y"]),
    ]
    outs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, seq_len, N])]
    return nodes, inits, outs


def _make_shared_act_model(
    seq_len: int, K: int, kind: str, seed: int, norm: bool = False, **dims
):
    """Projections of kind "qkv" or "gate_up" reading one activation.

    With norm=True the activation is the output of a
    SkipSimplifiedLayerNormalization over inputs X and SKIP, and the model
    also returns that output (NORM) and the residual sum (SUM).
    """
    rng = np.random.default_rng(seed)
    shape = [1, seq_len, K]
    graph_inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT16, shape)]
    pre_nodes, pre_inits, pre_outs = [], [], []
    act = "X"
    if norm:
        graph_inputs.append(
            helper.make_tensor_value_info("SKIP", TensorProto.FLOAT16, shape)
        )
        # Own generator, so the weights match the norm=False model's.
        gamma_rng = np.random.default_rng(seed + 1)
        gamma = gamma_rng.uniform(0.5, 1.5, [K]).astype(np.float16)
        pre_inits.append(numpy_helper.from_array(gamma, name="gamma"))
        pre_nodes.append(
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                ["X", "SKIP", "gamma"],
                ["NORM", "", "", "SUM"],
                domain="com.microsoft",
                epsilon=1e-5,
            )
        )
        pre_outs = [
            helper.make_tensor_value_info(n, TensorProto.FLOAT16, shape)
            for n in ("NORM", "SUM")
        ]
        act = "NORM"
    q4 = dict(block_size=dims.pop("block_size", BLOCK_SIZE), asym=dims.pop("asym"))
    if kind == "qkv":
        nodes, inits, outs = _qkv_nodes(rng, act, K, seq_len=seq_len, **dims, **q4)
    else:
        nodes, inits, outs = _gate_up_silu_nodes(
            rng, act, K, seq_len=seq_len, **dims, **q4
        )
    return make_model_from_nodes(
        pre_nodes + nodes,
        graph_inputs,
        outs + pre_outs,
        initializers=pre_inits + inits,
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestMatMulNBitsSharedActivation:
    """MatMulNBits nodes reading one activation, which the compiler quantizes
    once (hip.quantize_act) for every int4 projection that reads it."""

    @pytest.mark.parametrize("seq_len", SHARED_ACT_SEQ_LENS)
    @pytest.mark.parametrize(
        "K,n_q,n_kv,block_size,asym",
        [
            (LLAMA2_HIDDEN, LLAMA2_HIDDEN, LLAMA2_HIDDEN, BLOCK_SIZE, False),
            (PHI35_HIDDEN, PHI35_HIDDEN, PHI35_HIDDEN, BLOCK_SIZE, False),
            (LLAMA_HIDDEN, LLAMA_HIDDEN, LLAMA_KV_HIDDEN, LLAMA_BLOCK_SIZE, True),
            (512, 512, 128, BLOCK_SIZE, True),
        ],
        ids=["llama2", "phi35", "llama31_awq", "gqa_bs32_asym"],
    )
    def test_qkv(self, model_runner, seq_len, K, n_q, n_kv, block_size, asym):
        model = _make_shared_act_model(
            seq_len, K, "qkv", 7, n_q=n_q, n_kv=n_kv, block_size=block_size, asym=asym
        )
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=5e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SHARED_ACT_SEQ_LENS)
    @pytest.mark.parametrize(
        "K,N,block_size,asym",
        [
            (LLAMA2_HIDDEN, LLAMA2_INTERMEDIATE, BLOCK_SIZE, False),
            (PHI35_HIDDEN, PHI35_INTERMEDIATE, BLOCK_SIZE, False),
            (LLAMA_HIDDEN, LLAMA_INTERMEDIATE, LLAMA_BLOCK_SIZE, True),
        ],
        ids=["llama2", "phi35", "llama31_awq"],
    )
    def test_gate_up_silu(self, model_runner, seq_len, K, N, block_size, asym):
        model = _make_shared_act_model(
            seq_len, K, "gate_up", 11, N=N, block_size=block_size, asym=asym
        )
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        # silu(gate) * up scales gate's fp16 GEMM error by |up|, so absolute
        # error is larger than for a single projection; cosine stays tight.
        compare_outputs(actual, expected, atol=5e-1, rtol=1e-2, cos_threshold=0.9999)


class TestMatMulNBitsNormFedActivation:
    """Projections reading a SkipSimplifiedLayerNormalization output: the norm
    kernel writes the shared int8 activation itself (quant_outs).

    Bit-identity with hip.quantize_act is checked end to end: the projection
    outputs of the norm-fed model must equal, bit for bit, those of a model
    that takes the norm output as a graph input and quantizes it standalone.
    The dp4a GEMV is deterministic, so any difference in the int8 activation
    or its scales shows up in the outputs at seq_len 1 and 7.
    """

    @staticmethod
    def _run_and_check(model_runner, request, seq_len, K, kind, seed, tol, **dims):
        fused = _make_shared_act_model(seq_len, K, kind, seed, norm=True, **dims)
        split = _make_shared_act_model(seq_len, K, kind, seed, **dims)
        rng = np.random.default_rng(99)
        x = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)
        skip = rng.uniform(-1, 1, [1, seq_len, K]).astype(np.float16)

        # ORT CPU computes accuracy_level=4 with an int8 activation. That
        # matches the GPU's dp4a path; past DP4A_MAX_M rows the GPU computes
        # in fp16, and on the norm's heavier-tailed output the CPU's int8
        # rounding alone exceeds the tolerance, so compare that path with an
        # fp32-compute reference instead.
        reference = None
        if seq_len > DP4A_MAX_M:
            reference = onnx.ModelProto()
            reference.CopyFrom(fused)
            for node in reference.graph.node:
                for attr in node.attribute:
                    if node.op_type == "MatMulNBits" and attr.name == "accuracy_level":
                        attr.i = 0

        tag = request.node.name
        actual, expected = model_runner.run_sample(
            fused, [x, skip], name=f"{tag}_fused", reference_model=reference
        )
        compare_outputs(actual, expected, **tol)

        num_proj = len(actual) - 2
        norm_out = actual[num_proj]
        split_actual, _ = model_runner.run_sample(
            split, [norm_out], name=f"{tag}_split"
        )
        for i in range(num_proj):
            assert np.array_equal(
                actual[i].view(np.uint16), split_actual[i].view(np.uint16)
            ), f"projection {i}: norm-fused int8 activation differs from quantize_act"

    @pytest.mark.parametrize("seq_len", SHARED_ACT_SEQ_LENS)
    @pytest.mark.parametrize(
        "K,n_q,n_kv,block_size,asym",
        [
            (LLAMA2_HIDDEN, LLAMA2_HIDDEN, LLAMA2_HIDDEN, BLOCK_SIZE, False),
            (LLAMA_HIDDEN, LLAMA_HIDDEN, LLAMA_KV_HIDDEN, LLAMA_BLOCK_SIZE, True),
            (512, 512, 128, 64, True),
        ],
        ids=["llama2", "llama31_awq", "gqa_bs64_asym"],
    )
    def test_qkv(self, model_runner, request, seq_len, K, n_q, n_kv, block_size, asym):
        self._run_and_check(
            model_runner,
            request,
            seq_len,
            K,
            "qkv",
            7,
            dict(atol=5e-2, rtol=1e-2, cos_threshold=0.999),
            n_q=n_q,
            n_kv=n_kv,
            block_size=block_size,
            asym=asym,
        )

    @pytest.mark.parametrize("seq_len", SHARED_ACT_SEQ_LENS)
    @pytest.mark.parametrize(
        "K,N,block_size,asym",
        [
            (PHI35_HIDDEN, PHI35_INTERMEDIATE, BLOCK_SIZE, False),
            (LLAMA_HIDDEN, LLAMA_INTERMEDIATE, LLAMA_BLOCK_SIZE, True),
        ],
        ids=["phi35", "llama31_awq"],
    )
    def test_gate_up_silu(self, model_runner, request, seq_len, K, N, block_size, asym):
        self._run_and_check(
            model_runner,
            request,
            seq_len,
            K,
            "gate_up",
            11,
            dict(atol=5e-1, rtol=1e-2, cos_threshold=0.9999),
            N=N,
            block_size=block_size,
            asym=asym,
        )
