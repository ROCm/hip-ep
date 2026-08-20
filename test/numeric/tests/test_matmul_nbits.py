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

    x = helper.make_tensor_value_info(
        "X",
        TensorProto.FLOAT16,
        [batch, seq_len, K],
    )
    y = helper.make_tensor_value_info(
        "Y",
        TensorProto.FLOAT16,
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
    ).astype(np.float16)

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
