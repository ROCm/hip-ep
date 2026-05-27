#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the CausalConvWithState custom op (com.microsoft).

CausalConvWithState fuses the (Concat + Conv + Slice) preprocessing
pattern used by Gated DeltaNet (Qwen3.5) and Mamba/Jamba into a single
op.  The convolution is causal on the last spatial axis and depthwise
on the channel dimension.

Inputs (channels-first):
  input        -- (batch, channels, L)        (1D case)
  weight       -- (channels, 1, K)            depthwise kernel
  bias         -- (channels,)                 optional
  past_state   -- (batch, channels, K - 1)    optional carry state

Outputs:
  output       -- (batch, channels, L)
  present_state -- (batch, channels, K - 1)

Shape footprint observed in the chunk_opt prefill / decode models
(30 occurrences per graph each):
  prefill: input  fp16[1, 8192, 128] , state fp16[1, 8192, 3], K = 4
  decode : input  fp16[1, 8192,   1] , state fp16[1, 8192, 3], K = 4
  attrs  : activation='silu', ndim=1
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Exercise prefill + decode shape regimes with one helper -- capped to
# the project-wide {1, 128} sweep.
SEQ_LENS = [1, 128]

# Chunk-opt model footprint.
CHUNK_OPT_CHANNELS = 8192
CHUNK_OPT_K = 4  # kernel size; carry state size = K - 1 = 3


def _make_causal_conv_with_state_model(
    batch: int,
    channels: int,
    seq_len: int,
    kernel_size: int,
    activation: str = "silu",
    with_bias: bool = True,
):
    """Build a 1D CausalConvWithState ONNX model with weight/bias as
    initializers and runtime inputs for the data tensor and carry state.
    """
    tp = TensorProto.FLOAT16
    state_len = kernel_size - 1

    X = helper.make_tensor_value_info("input", tp, [batch, channels, seq_len])
    State = helper.make_tensor_value_info(
        "past_state", tp, [batch, channels, state_len]
    )
    Y = helper.make_tensor_value_info("output", tp, [batch, channels, seq_len])
    Present = helper.make_tensor_value_info(
        "present_state", tp, [batch, channels, state_len]
    )

    # Weight / bias are realistic small magnitudes; keeping them tight
    # avoids fp16 overflow in long-S prefill cases when SiLU saturates.
    rng = np.random.default_rng(0xC0DE)
    weight = (rng.standard_normal((channels, 1, kernel_size)) * 0.1).astype(np.float16)
    initializers = [numpy_helper.from_array(weight, name="weight")]
    inputs_w = ["input", "weight"]
    if with_bias:
        bias = (rng.standard_normal((channels,)) * 0.05).astype(np.float16)
        initializers.append(numpy_helper.from_array(bias, name="bias"))
        inputs_w.append("bias")
    else:
        inputs_w.append("")  # explicit empty optional
    inputs_w.append("past_state")

    node = helper.make_node(
        "CausalConvWithState",
        inputs_w,
        ["output", "present_state"],
        domain="com.microsoft",
        activation=activation,
        ndim=1,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [X, State],
        [Y, Present],
        initializers=initializers,
        opset=17,
        extra_opsets=[ms_opset],
    )


class TestCausalConvWithState:
    """SiLU is the only activation observed in chunk_opt; we additionally
    cover 'none' on a tiny shape so a regression in the activation branch
    surfaces independently of the SiLU code path."""

    @pytest.mark.parametrize(
        "batch,channels,seq_len,kernel_size,activation",
        [
            (1, 32, 16, 4, "silu"),
            (1, 32, 16, 4, "none"),
            (1, 32, 1, 4, "silu"),
            # seq_len=128 on tiny channels: exercises the MIOpen prefill
            # path at moderate sequence length without the 8192-channel
            # memory footprint of test_ccws_chunk_opt_shape[128]. Covers
            # both activation branches so a regression in either surfaces
            # here independently.
            (1, 32, 128, 4, "silu"),
            (1, 32, 128, 4, "none"),
        ],
    )
    def test_ccws_tiny(
        self,
        model_runner,
        batch,
        channels,
        seq_len,
        kernel_size,
        activation,
    ):
        """Small-shape sanity coverage (prefill + decode regimes)."""
        model = _make_causal_conv_with_state_model(
            batch,
            channels,
            seq_len,
            kernel_size,
            activation=activation,
        )

        rng = np.random.default_rng(1)
        x = (rng.standard_normal((batch, channels, seq_len)) * 0.5).astype(np.float16)
        state = (rng.standard_normal((batch, channels, kernel_size - 1)) * 0.5).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x, state])
        # Two outputs: depthwise fp16 conv; tight tolerance is fine.
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_ccws_chunk_opt_shape(self, model_runner, seq_len):
        """CausalConvWithState with chunk_opt prefill/decode shapes:

        input fp16[1, 8192, S], past_state fp16[1, 8192, 3], K=4,
        activation='silu', ndim=1.

        S = 1   -> decode regime  (matches chunk_opt decode model)
        S = 128 -> moderate prefill within the project-wide sweep cap.
        """
        channels = CHUNK_OPT_CHANNELS
        kernel_size = CHUNK_OPT_K
        model = _make_causal_conv_with_state_model(
            1,
            channels,
            seq_len,
            kernel_size,
            activation="silu",
        )

        rng = np.random.default_rng(2)
        # Inputs sit close to a normalised activation distribution; the
        # small std avoids fp16 overflow when SiLU is applied to the conv
        # output across 8192 channels.
        x = (rng.standard_normal((1, channels, seq_len)) * 0.5).astype(np.float16)
        state = (rng.standard_normal((1, channels, kernel_size - 1)) * 0.5).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x, state])
        # Per-channel depthwise conv with K=4 and SiLU; fp16 noise from
        # the activation dominates, so allow a slightly looser absolute
        # tolerance than the tiny-shape variant above.
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2)
