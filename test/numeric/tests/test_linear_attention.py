#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the LinearAttention custom op (com.microsoft).

Unified linear-attention operator for autoregressive decoding (T = 1)
and prefill (T > 1).  All inputs use 3D packed format
[B, T, H * D]; the recurrent state is always 4D [B, H_kv, d_k, d_v].

Update rule observed in chunk_opt / Qwen3.5:

  S_t = exp(g_t) * S_{t-1}
        + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T k_t)
  o_t = scale * q_t^T S_t

(``gated_delta`` rule).  ``g_t`` is the decay in log-space, ``beta_t``
is the per-token update rate (sigmoid output, in [0, 1]), and ``(x)``
denotes outer product over the (d_k, d_v) plane.

Shape footprint observed in the chunk_opt prefill / decode models
(30 occurrences per graph each):

  prefill -- S = 128 (sweep cap):
    Q     fp16 [1,   S, 2048]
    K     fp16 [1,   S, 2048]
    V     fp16 [1,   S, 4096]
    state fp16 [1,  32,  128, 128]
    decay fp16 [1,   S,   32]
    beta  fp16 [1,   S,   32]
    output fp16[1,   S, 4096]
    attrs : kv_num_heads=32, q_num_heads=16, scale=1.0,
            update_rule='gated_delta'

  decode -- S = 1: same shapes with the time axis collapsed.

Note on the inputs' head-dim asymmetry:
  Q's hidden = 2048 with q_num_heads=16   -> d_k_q = 128 per Q head.
  K's hidden = 2048 with kv_num_heads=32  -> d_k_k = 64 per spec
  V's hidden = 4096 with kv_num_heads=32  -> d_v   = 128 per V head.
  state shape [B, 32, 128, 128]           -> internal d_k = 128, d_v = 128.
ORT accepts this layout as-is (the runtime resolves the broadcast /
head-grouping internally), so the test mirrors the model contract
exactly rather than forcing a strictly self-consistent breakdown.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]

# Chunk-opt model footprint (Qwen3.5 GatedDeltaNet)
CHUNK_OPT_Q_DIM = 2048
CHUNK_OPT_K_DIM = 2048
CHUNK_OPT_V_DIM = 4096
CHUNK_OPT_Q_HEADS = 16
CHUNK_OPT_KV_HEADS = 32
CHUNK_OPT_STATE_DK = 128
CHUNK_OPT_STATE_DV = 128


def _make_linear_attention_model(
    batch: int,
    seq_len: int,
    q_num_heads: int,
    kv_num_heads: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    state_dk: int,
    state_dv: int,
    scale: float = 1.0,
    update_rule: str = "gated_delta",
):
    """Build a LinearAttention ONNX model with all six runtime inputs.

    The output ``Y`` has shape [B, T, H_q * d_v_out].  When the model
    follows the 'aligned' contract, that equals [B, T, V dim].  In the
    chunk_opt layout (q_num_heads != kv_num_heads with V's hidden
    derived from kv_num_heads * d_v), the ORT runtime broadcasts the
    KV side per H_q / H_kv group, so the output dim equals V's hidden
    dim.  The reference (CPU EP) and the test backend agree on this
    convention; we just declare the output shape that ORT actually
    produces.
    """
    tp = TensorProto.FLOAT16

    # Output hidden-dim follows the V tensor's hidden in the layouts we
    # exercise (chunk_opt and the consistent breakdown).
    out_dim = v_dim

    Q = helper.make_tensor_value_info("query", tp, [batch, seq_len, q_dim])
    K = helper.make_tensor_value_info("key", tp, [batch, seq_len, k_dim])
    V = helper.make_tensor_value_info("value", tp, [batch, seq_len, v_dim])
    State = helper.make_tensor_value_info(
        "past_state",
        tp,
        [batch, kv_num_heads, state_dk, state_dv],
    )
    Decay = helper.make_tensor_value_info(
        "decay",
        tp,
        [batch, seq_len, kv_num_heads],
    )
    Beta = helper.make_tensor_value_info(
        "beta",
        tp,
        [batch, seq_len, kv_num_heads],
    )

    Y = helper.make_tensor_value_info(
        "output",
        tp,
        [batch, seq_len, out_dim],
    )
    Present = helper.make_tensor_value_info(
        "present_state",
        tp,
        [batch, kv_num_heads, state_dk, state_dv],
    )

    node = helper.make_node(
        "LinearAttention",
        ["query", "key", "value", "past_state", "decay", "beta"],
        ["output", "present_state"],
        domain="com.microsoft",
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        scale=scale,
        update_rule=update_rule,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [Q, K, V, State, Decay, Beta],
        [Y, Present],
        opset=17,
        extra_opsets=[ms_opset],
    )


def _make_inputs(
    rng: np.random.Generator,
    batch: int,
    seq_len: int,
    kv_num_heads: int,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    state_dk: int,
    state_dv: int,
    *,
    state_scale: float = 0.1,
    qkv_scale: float = 0.5,
):
    """Build a coherent input set whose magnitudes won't blow up fp16.

    - Q/K are typically L2-normalised in the runtime -- we approximate
      that by drawing N(0, qkv_scale^2).
    - Decay is read in log-space; ORT applies exp(g) so we keep it in
      [-1, 0] to stay <= 1 multiplicatively per token.
    - Beta is the sigmoid output of a separate gate -- in (0, 1).
    """
    q = (rng.standard_normal((batch, seq_len, q_dim)) * qkv_scale).astype(np.float16)
    k = (rng.standard_normal((batch, seq_len, k_dim)) * qkv_scale).astype(np.float16)
    v = (rng.standard_normal((batch, seq_len, v_dim)) * qkv_scale).astype(np.float16)
    state = (
        rng.standard_normal((batch, kv_num_heads, state_dk, state_dv)) * state_scale
    ).astype(np.float16)
    decay = rng.uniform(-1.0, 0.0, (batch, seq_len, kv_num_heads)).astype(np.float16)
    beta = rng.uniform(0.0, 1.0, (batch, seq_len, kv_num_heads)).astype(np.float16)
    return [q, k, v, state, decay, beta]


class TestLinearAttention:
    """A tiny 'consistent breakdown' case (q_num_heads == kv_num_heads,
    Q/K/V hidden derived from a single d_k/d_v) gives clean coverage of
    the gated_delta math; the chunk_opt cases then verify the asymmetric
    layout the actual model uses."""

    @pytest.mark.parametrize(
        "batch,seq_len,qh,kvh,d_k,d_v",
        [
            (1, 8, 4, 4, 16, 16),
            (1, 1, 4, 4, 16, 16),
        ],
    )
    def test_la_tiny_gated_delta(
        self,
        model_runner,
        batch,
        seq_len,
        qh,
        kvh,
        d_k,
        d_v,
    ):
        """Small-shape sanity coverage of the gated_delta recurrence.

        Restricted to ``q_num_heads == kv_num_heads`` (no GQA-style head
        sharing) so the output-dim convention is unambiguous.  Asymmetric
        head-grouping is exercised separately by ``test_la_chunk_opt_shape``
        with the 16/32 q/kv layout used by the real Qwen3.5 chunk_opt
        model -- the only asymmetric case the EP must support."""
        q_dim = qh * d_k
        k_dim = kvh * d_k
        v_dim = kvh * d_v
        model = _make_linear_attention_model(
            batch,
            seq_len,
            qh,
            kvh,
            q_dim,
            k_dim,
            v_dim,
            d_k,
            d_v,
            scale=1.0,
            update_rule="gated_delta",
        )

        rng = np.random.default_rng(31)
        inputs = _make_inputs(
            rng,
            batch,
            seq_len,
            kvh,
            q_dim,
            k_dim,
            v_dim,
            d_k,
            d_v,
        )

        actual, expected = model_runner.run_sample(model, inputs)
        # gated_delta accumulates an outer-product update over T tokens
        # in fp32 internally; fp16 round-trip on the I/O still leaves a
        # tight tolerance.
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_la_chunk_opt_shape(self, model_runner, seq_len):
        """LinearAttention with the chunk_opt prefill/decode footprint:

        Q     fp16 [1, S, 2048]   K     fp16 [1, S, 2048]
        V     fp16 [1, S, 4096]   state fp16 [1, 32, 128, 128]
        decay fp16 [1, S, 32]     beta  fp16 [1, S, 32]
        kv_num_heads=32, q_num_heads=16, scale=1.0,
        update_rule='gated_delta'.

        S = 1   -> decode regime (matches chunk_opt decode model)
        S = 128 -> moderate-prefill regime within the project-wide sweep cap.
        """
        model = _make_linear_attention_model(
            1,
            seq_len,
            CHUNK_OPT_Q_HEADS,
            CHUNK_OPT_KV_HEADS,
            CHUNK_OPT_Q_DIM,
            CHUNK_OPT_K_DIM,
            CHUNK_OPT_V_DIM,
            CHUNK_OPT_STATE_DK,
            CHUNK_OPT_STATE_DV,
            scale=1.0,
            update_rule="gated_delta",
        )

        rng = np.random.default_rng(37)
        inputs = _make_inputs(
            rng,
            1,
            seq_len,
            CHUNK_OPT_KV_HEADS,
            CHUNK_OPT_Q_DIM,
            CHUNK_OPT_K_DIM,
            CHUNK_OPT_V_DIM,
            CHUNK_OPT_STATE_DK,
            CHUNK_OPT_STATE_DV,
            # Long-T runs let the recurrent state accumulate; bring the
            # initial-state magnitude down a touch so fp16 doesn't
            # saturate at S=128.
            state_scale=0.05,
            qkv_scale=0.4,
        )

        actual, expected = model_runner.run_sample(model, inputs)
        # Two outputs: attention output (B, S, 4096) and updated state
        # (B, 32, 128, 128).  Both accumulate in fp32 inside the kernel
        # but are stored in fp16; long-S gated_delta drift means we keep
        # atol relaxed and rely on rtol + cosine for correctness.
        compare_outputs(
            actual,
            expected,
            atol=2e-1,
            rtol=1e-2,
            cos_threshold=0.999,
        )
