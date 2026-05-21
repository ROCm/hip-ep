#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX Transpose operator.

Real-model footprint -- Qwen3.5-9B (text.onnx) uses 80 Transpose ops,
all on fp16 with one of two permutations:

    perm=[0, 2, 1]    (3-D)  fp16 [B, S, 8192] <-> fp16 [B, 8192, S]   (x48)
    perm=[0, 2, 1, 3] (4-D)  fp16 [B, S, H, D] <-> fp16 [B, H, S, D]   (x32)

The 4-D case is the standard "interleave heads" reshape that flips
(seq, heads) for attention with H in {4, 16} and D=256.

The 3-D `[B, S, 8192] <-> [B, 8192, S]` cases (count=48, both
directions) cover the GEMM-friendly layout swap around projection
boundaries.

These tests pin both directions of both permutations at the model's
exact dtype (fp16), plus a tiny generic 2-D / 3-D smoke pair.
"""

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


# Sequence lengths chosen to cover:
#   * S=1   : decode (e.g. chunk_opt/decode_p512m16384.onnx, 100 Transpose ops)
#   * S=128 : moderate prefill / smoke
SEQ_LENS = [1, 128]

# Qwen3.5-9B attention shape: hidden=4096, num_heads=16, head_dim=256
# (and a secondary path with H=4 KV heads -- both head counts appear
# in the Transpose footprint above).  The chunk_opt prefill/decode
# models exercise the same family with H ∈ {2, 16}, so we add H=2
# as a third KV-head sweep value (Qwen3.5-35B GQA 8:1 grouping).
QWEN9B_HIDDEN_FAT = 8192  # 2x intermediate fan-out / fan-in
QWEN9B_HEAD_DIM = 256


def _make_transpose_model(input_shape, perm, dtype, output_shape=None):
    """Build a single-node Transpose model."""
    if output_shape is None:
        output_shape = [input_shape[p] for p in perm]
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    Y = helper.make_tensor_value_info("Y", tp, list(output_shape))
    node = helper.make_node("Transpose", ["X"], ["Y"], perm=list(perm))
    return make_model_from_nodes([node], [X], [Y])


class TestTranspose:
    @pytest.mark.parametrize(
        "shape, perm",
        [
            ([4, 8], [1, 0]),
            ([2, 3, 5], [2, 0, 1]),
        ],
        ids=["2d_swap", "3d_rotate"],
    )
    def test_transpose_smoke(self, model_runner, shape, perm):
        """Generic small-shape smoke check (any-rank, any perm)."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, shape).astype(np.float16)

        model = _make_transpose_model(shape, perm, np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    # ------------------------------------------------------------------
    # Qwen3.5-9B 3-D layout-swap perms: [0, 2, 1] in both directions
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_transpose_qwen9b_3d_bsh_to_bhs(self, model_runner, seq_len):
        """fp16 [1, S, 8192] -- perm=[0,2,1] --> [1, 8192, S]  (x24 in model)."""
        shape = [1, seq_len, QWEN9B_HIDDEN_FAT]
        rng = np.random.default_rng(43)
        x = rng.uniform(-1, 1, shape).astype(np.float16)

        model = _make_transpose_model(shape, [0, 2, 1], np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_transpose_qwen9b_3d_bhs_to_bsh(self, model_runner, seq_len):
        """fp16 [1, 8192, S] -- perm=[0,2,1] --> [1, S, 8192]  (x24 in model)."""
        shape = [1, QWEN9B_HIDDEN_FAT, seq_len]
        rng = np.random.default_rng(44)
        x = rng.uniform(-1, 1, shape).astype(np.float16)

        model = _make_transpose_model(shape, [0, 2, 1], np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    # ------------------------------------------------------------------
    # 4-D head-interleave: perm=[0, 2, 1, 3] both directions
    #   H = 2  (chunk_opt KV heads, Qwen3.5-35B GQA 8:1)
    #   H = 4  (Qwen3.5-9B KV heads)
    #   H = 16 (Qwen3.5-9B / chunk_opt Q heads)
    #   D = 256
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "num_heads",
        [2, 4, 16],
        ids=["kv_heads_2", "kv_heads_4", "q_heads_16"],
    )
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_transpose_qwen9b_4d_bshd_to_bhsd(
        self,
        model_runner,
        seq_len,
        num_heads,
    ):
        """fp16 [1, S, H, 256] -- perm=[0,2,1,3] --> [1, H, S, 256]  (x16+ in model)."""
        shape = [1, seq_len, num_heads, QWEN9B_HEAD_DIM]
        rng = np.random.default_rng(45 + num_heads)
        x = rng.uniform(-1, 1, shape).astype(np.float16)

        model = _make_transpose_model(shape, [0, 2, 1, 3], np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "num_heads",
        [2, 4, 16],
        ids=["kv_heads_2", "kv_heads_4", "q_heads_16"],
    )
    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_transpose_qwen9b_4d_bhsd_to_bshd(
        self,
        model_runner,
        seq_len,
        num_heads,
    ):
        """fp16 [1, H, S, 256] -- perm=[0,2,1,3] --> [1, S, H, 256]  (x16+ in model)."""
        shape = [1, num_heads, seq_len, QWEN9B_HEAD_DIM]
        rng = np.random.default_rng(46 + num_heads)
        x = rng.uniform(-1, 1, shape).astype(np.float16)

        model = _make_transpose_model(shape, [0, 2, 1, 3], np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
