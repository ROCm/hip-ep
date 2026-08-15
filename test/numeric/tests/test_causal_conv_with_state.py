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


_NP_TO_TP = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_causal_conv_with_state_model(
    batch: int,
    channels: int,
    seq_len: int,
    kernel_size: int,
    activation: str = "silu",
    with_bias: bool = True,
    dtype: np.dtype = np.float16,
):
    """Build a 1D CausalConvWithState ONNX model with weight/bias as
    initializers and runtime inputs for the data tensor and carry state.

    `dtype` defaults to fp16 (the chunk_opt footprint); pass np.float32 to
    exercise the kernel's f32 path.
    """
    tp = _NP_TO_TP[dtype]
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
    weight = (rng.standard_normal((channels, 1, kernel_size)) * 0.1).astype(dtype)
    initializers = [numpy_helper.from_array(weight, name="weight")]
    inputs_w = ["input", "weight"]
    if with_bias:
        bias = (rng.standard_normal((channels,)) * 0.05).astype(dtype)
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


def _make_channels_last_causal_conv_model(
    batch: int,
    channels: int,
    seq_len: int,
    kernel_size: int,
    activation: str = "silu",
    dtype: np.dtype = np.float16,
):
    """Build the shape the exporter actually emits: a channels-last graph with
    the convolution bracketed by a Transpose pair.

    ONNX Conv is channels-first, so a channels-last graph has to permute into
    the op and back out again. Those two transposes are what the
    `channels_last` fold removes, so this is the model that exercises it --
    building the op with a channels-last input directly would not, since
    nothing in the ONNX op says which layout it was given.

    The carry state is channels-first in both graphs: only input and output are
    permuted.
    """
    tp = _NP_TO_TP[dtype]
    state_len = kernel_size - 1

    X = helper.make_tensor_value_info("input_nlc", tp, [batch, seq_len, channels])
    State = helper.make_tensor_value_info(
        "past_state", tp, [batch, channels, state_len]
    )
    Y = helper.make_tensor_value_info("output_nlc", tp, [batch, seq_len, channels])
    Present = helper.make_tensor_value_info(
        "present_state", tp, [batch, channels, state_len]
    )

    rng = np.random.default_rng(0xC0DE)
    weight = (rng.standard_normal((channels, 1, kernel_size)) * 0.1).astype(dtype)
    bias = (rng.standard_normal((channels,)) * 0.05).astype(dtype)
    initializers = [
        numpy_helper.from_array(weight, name="weight"),
        numpy_helper.from_array(bias, name="bias"),
    ]

    nodes = [
        helper.make_node(
            "Transpose", ["input_nlc"], ["input_ncl"], perm=[0, 2, 1], name="pre_t"
        ),
        helper.make_node(
            "CausalConvWithState",
            ["input_ncl", "weight", "bias", "past_state"],
            ["output_ncl", "present_state"],
            domain="com.microsoft",
            activation=activation,
            ndim=1,
        ),
        helper.make_node(
            "Transpose", ["output_ncl"], ["output_nlc"], perm=[0, 2, 1], name="post_t"
        ),
    ]

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        nodes,
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

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_ccws_prefill_dtype_bias(self, model_runner, dtype, with_bias):
        """Prefill (seq_len>1) across the fused kernel's dtype and bias
        branches: f32 in addition to fp16, and the no-bias (HAS_BIAS=0) path
        that the chunk_opt footprint (always-bias) never exercises.
        """
        batch, channels, seq_len, kernel_size = 1, 32, 128, 4
        model = _make_causal_conv_with_state_model(
            batch,
            channels,
            seq_len,
            kernel_size,
            activation="silu",
            with_bias=with_bias,
            dtype=dtype,
        )

        rng = np.random.default_rng(3)
        x = (rng.standard_normal((batch, channels, seq_len)) * 0.5).astype(dtype)
        state = (rng.standard_normal((batch, channels, kernel_size - 1)) * 0.5).astype(
            dtype
        )

        actual, expected = model_runner.run_sample(model, [x, state])
        if dtype == np.float32:
            compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)
        else:
            compare_outputs(actual, expected, atol=2e-3, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [257, 600])
    def test_ccws_prefill_multi_tile(self, model_runner, seq_len):
        """Sequences longer than the prefill kernel's 256-output tile.

        The fused prefill kernel gives each block one 256-long run of outputs
        and stages that run's window in LDS, so everything specific to tiling
        only happens above 256: a block whose window starts inside `input`
        rather than in `past_state`, and the single block that owns
        `present_state` because it covers the tail. Both 128 and 1 (the sweep
        the other cases use) are single-tile, which leaves that logic
        unexercised. 257 puts one output in the second tile; 600 spans three.
        """
        batch, channels, kernel_size = 1, 32, 4
        model = _make_causal_conv_with_state_model(
            batch,
            channels,
            seq_len,
            kernel_size,
            activation="silu",
        )

        rng = np.random.default_rng(4)
        x = (rng.standard_normal((batch, channels, seq_len)) * 0.5).astype(np.float16)
        state = (rng.standard_normal((batch, channels, kernel_size - 1)) * 0.5).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x, state])
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

    # ---- channels-last -----------------------------------------------------

    @pytest.mark.parametrize("seq_len", [1, 128, 257])
    def test_ccws_channels_last_matches_reference(self, model_runner, seq_len):
        """The Transpose/Conv/Transpose graph, which the compiler folds into one
        channels-last convolution, still agrees with the reference.

        seq_len=1 is the decode regime, where the fold's premise is that the
        permuted axis has extent 1 and the layouts coincide. 257 crosses the
        prefill kernel's output tile so the channels-last kernel's tail handling
        (the tile that owns present_state) is exercised, not just its first tile.
        """
        batch, channels, kernel_size = 1, 32, 4
        model = _make_channels_last_causal_conv_model(
            batch, channels, seq_len, kernel_size, activation="silu"
        )

        rng = np.random.default_rng(5)
        x = (rng.standard_normal((batch, seq_len, channels)) * 0.5).astype(np.float16)
        state = (rng.standard_normal((batch, channels, kernel_size - 1)) * 0.5).astype(
            np.float16
        )

        actual, expected = model_runner.run_sample(model, [x, state])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", [128, 257])
    def test_ccws_channels_last_bit_identical(self, model_runner, seq_len):
        """The folded path is bit-identical to the channels-first path.

        Both kernels accumulate the same K taps in fp32 in the same order, so
        the layout change must not perturb a single bit. Comparing against the
        CPU reference (the test above) cannot show that -- it has its own error
        -- so this compares the two GPU results to each other and asserts exact
        equality. A tolerance here would hide precisely the class of bug this
        exists to catch: a reordered accumulation that is merely close.
        """
        batch, channels, kernel_size = 1, 32, 4
        rng = np.random.default_rng(6)
        x_ncl = (rng.standard_normal((batch, channels, seq_len)) * 0.5).astype(
            np.float16
        )
        state = (rng.standard_normal((batch, channels, kernel_size - 1)) * 0.5).astype(
            np.float16
        )

        cf_model = _make_causal_conv_with_state_model(
            batch, channels, seq_len, kernel_size, activation="silu"
        )
        nlc_model = _make_channels_last_causal_conv_model(
            batch, channels, seq_len, kernel_size, activation="silu"
        )

        cf_actual, _ = model_runner.run_sample(
            cf_model, [x_ncl, state], name=f"ccws_bitident_cf_{seq_len}"
        )
        nlc_actual, _ = model_runner.run_sample(
            nlc_model,
            [np.ascontiguousarray(x_ncl.transpose(0, 2, 1)), state],
            name=f"ccws_bitident_nlc_{seq_len}",
        )

        # Both helpers seed weight and bias from the same generator, so the two
        # graphs hold identical initializers.
        np.testing.assert_array_equal(
            nlc_actual[0].transpose(0, 2, 1),
            cf_actual[0],
            err_msg="channels-last output differs from channels-first",
        )
        np.testing.assert_array_equal(
            nlc_actual[1],
            cf_actual[1],
            err_msg="channels-last present_state differs from channels-first",
        )
