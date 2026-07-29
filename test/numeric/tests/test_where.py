#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX Where operator (element-wise select).

Where(condition, X, Y) returns elements from X where condition is True
and from Y otherwise; bidirectional NumPy-style broadcasting applies
across all three operands.

Real-model footprint -- Qwen3.5-9B (text.onnx) uses 4 Where ops in
the rotary-embedding mask preprocessing.  All four share the same
shape signature:

    cond : bool [32]                   (initializer mask)
    X    : fp32 [B, S, 32]             (gathered cos/sin half)
    Y    : fp32 [B, S, 32]             (gathered cos/sin half)
    out  : fp32 [B, S, 32]

Two cascaded pairs realise the per-axis interleave used by RoPE:
    cos/w/Where -> cos/h/Where  and  sin/w/Where -> sin/h/Where

These tests pin that exact shape and dtype and add a handful of
generic shapes (same-shape, scalar broadcast, fp16) to guard against
silent regressions on broadcasting / dtype paths.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


# Sequence lengths chosen to cover:
#   * S=1   : decode (e.g. chunk_opt/decode_p512m16384.onnx, 4 Where ops)
#   * S=128 : moderate prefill / smoke
SEQ_LENS = [1, 128]

# Qwen3.5-9B RoPE mask: half-dim split is 32 (head_dim 256 / 2 / 4 = 32
# is the per-side stride for the four-way interleave).
QWEN9B_ROPE_HALF = 32


def _make_where_model(
    cond_shape, x_shape, y_shape, out_shape, dtype, cond_init: np.ndarray | None = None
):
    """Build a Where node.  If ``cond_init`` is provided, ``condition`` is
    embedded as a graph initializer (matches the Qwen3.5-9B model where
    the RoPE mask is a constant); otherwise it is a graph input."""
    tp = np_to_onnx_type(dtype)
    inputs = []
    initializers = []

    if cond_init is not None:
        initializers.append(numpy_helper.from_array(cond_init, name="condition"))
    else:
        inputs.append(
            helper.make_tensor_value_info(
                "condition",
                helper.TensorProto.BOOL,
                list(cond_shape),
            )
        )
    inputs.append(helper.make_tensor_value_info("X", tp, list(x_shape)))
    inputs.append(helper.make_tensor_value_info("Y", tp, list(y_shape)))
    output = helper.make_tensor_value_info("Z", tp, list(out_shape))

    node = helper.make_node("Where", ["condition", "X", "Y"], ["Z"])
    return make_model_from_nodes(
        [node],
        inputs,
        [output],
        initializers=initializers,
    )


class TestWhere:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_where_same_shape(self, model_runner, dtype):
        """Same-shape Where: no broadcasting."""
        shape = [4, 8]
        rng = np.random.default_rng(42)
        cond = rng.integers(0, 2, shape, dtype=np.bool_)
        x = rng.uniform(-1, 1, shape).astype(dtype)
        y = rng.uniform(-1, 1, shape).astype(dtype)

        model = _make_where_model(shape, shape, shape, shape, dtype)
        actual, expected = model_runner.run_sample(model, [cond, x, y])
        atol = 0 if np.issubdtype(dtype, np.integer) else 1e-4
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_where_scalar_condition(self, model_runner, dtype):
        """Where with a 0-D bool condition broadcasting across X / Y."""
        x_shape = [2, 16]
        rng = np.random.default_rng(43)
        cond = np.array(True, dtype=np.bool_)
        x = rng.uniform(-1, 1, x_shape).astype(dtype)
        y = rng.uniform(-1, 1, x_shape).astype(dtype)

        model = _make_where_model([], x_shape, x_shape, x_shape, dtype)
        actual, expected = model_runner.run_sample(model, [cond, x, y])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_where_qwen9b_rope_mask_shape(self, model_runner, seq_len):
        """RoPE mask Where as it appears in Qwen3.5-9B text.onnx.

        cond  bool[32]                (constant mask, embedded as initializer)
        X / Y fp32[1, S, 32]          (gathered cos/sin halves)
        out   fp32[1, S, 32]
        """
        x_shape = [1, seq_len, QWEN9B_ROPE_HALF]
        rng = np.random.default_rng(44)
        # Match the model: alternating mask of length 32 (half True / half False).
        cond_init = (np.arange(QWEN9B_ROPE_HALF) % 2 == 0).astype(np.bool_)
        x = rng.uniform(-1, 1, x_shape).astype(np.float32)
        y = rng.uniform(-1, 1, x_shape).astype(np.float32)

        model = _make_where_model(
            [QWEN9B_ROPE_HALF],
            x_shape,
            x_shape,
            x_shape,
            np.float32,
            cond_init=cond_init,
        )
        actual, expected = model_runner.run_sample(model, [x, y])
        compare_outputs(actual, expected, atol=1e-5)
