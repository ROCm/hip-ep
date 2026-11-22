#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Frame-masked vision attention softmax: Softmax(Where(Equal(ids, ids^T),
Mul(scores, scale), fill)).

Qwen-VL style vision encoders mask each video frame's patches to attend only
within the frame, by comparing per-patch frame ids. hip-fusion-transform folds
the chain into one hip.scaled_masked_softmax; these cases pin its numerics for
one frame (mask all true, the single-image case), two frames (block diagonal),
rows wider than the register-resident kernel, fp32, a length-1 id vector
broadcast across its axis, rank-4 scores, and scores produced in-graph.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_model(dtype, heads: int, seq: int, scale: float, fill: float):
    tp = np_to_onnx_type(dtype)
    scores = helper.make_tensor_value_info("scores", tp, [heads, seq, seq])
    ids = helper.make_tensor_value_info("ids", np_to_onnx_type(np.int64), [seq])
    probs = helper.make_tensor_value_info("probs", tp, [heads, seq, seq])
    initializers = [
        numpy_helper.from_array(np.array(scale, dtype=dtype), "scale"),
        numpy_helper.from_array(np.array(fill, dtype=dtype), "fill"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
    ]
    nodes = [
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Unsqueeze", ["ids", "axis1"], ["row_ids"]),
        helper.make_node("Unsqueeze", ["ids", "axis0"], ["col_ids"]),
        helper.make_node("Equal", ["row_ids", "col_ids"], ["mask"]),
        helper.make_node("Where", ["mask", "scaled", "fill"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
    ]
    return make_model_from_nodes(nodes, [scores, ids], [probs], initializers)


def _make_split_ids_model(dtype, scores_shape, row_len: int, col_len: int):
    """Row and column ids from separate inputs, so either may be 1 long and
    broadcast across its scores axis, as Where allows."""
    tp = np_to_onnx_type(dtype)
    i64 = np_to_onnx_type(np.int64)
    scores = helper.make_tensor_value_info("scores", tp, scores_shape)
    rows = helper.make_tensor_value_info("rows", i64, [row_len])
    cols = helper.make_tensor_value_info("cols", i64, [col_len])
    probs = helper.make_tensor_value_info("probs", tp, scores_shape)
    initializers = [
        numpy_helper.from_array(np.array(0.125, dtype=dtype), "scale"),
        numpy_helper.from_array(np.array(-65504.0, dtype=dtype), "fill"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
    ]
    nodes = [
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Unsqueeze", ["rows", "axis1"], ["row_ids"]),
        helper.make_node("Unsqueeze", ["cols", "axis0"], ["col_ids"]),
        helper.make_node("Equal", ["row_ids", "col_ids"], ["mask"]),
        helper.make_node("Where", ["mask", "scaled", "fill"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
    ]
    return make_model_from_nodes(nodes, [scores, rows, cols], [probs], initializers)


def _make_qk_model(heads: int, seq: int, dim: int):
    """Scores from an in-graph Q @ K^T, as in the vision encoder, rather than
    a graph input."""
    tp = np_to_onnx_type(np.float16)
    q = helper.make_tensor_value_info("q", tp, [heads, seq, dim])
    kt = helper.make_tensor_value_info("kt", tp, [heads, dim, seq])
    ids = helper.make_tensor_value_info("ids", np_to_onnx_type(np.int64), [seq])
    probs = helper.make_tensor_value_info("probs", tp, [heads, seq, seq])
    initializers = [
        numpy_helper.from_array(np.array(dim**-0.5, dtype=np.float16), "scale"),
        numpy_helper.from_array(np.array(-65504.0, dtype=np.float16), "fill"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis0"),
    ]
    nodes = [
        helper.make_node("MatMul", ["q", "kt"], ["scores"]),
        helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
        helper.make_node("Unsqueeze", ["ids", "axis1"], ["row_ids"]),
        helper.make_node("Unsqueeze", ["ids", "axis0"], ["col_ids"]),
        helper.make_node("Equal", ["row_ids", "col_ids"], ["mask"]),
        helper.make_node("Where", ["mask", "scaled", "fill"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["probs"], axis=-1),
    ]
    return make_model_from_nodes(nodes, [q, kt, ids], [probs], initializers)


def _frame_ids(seq: int, frames: int) -> np.ndarray:
    return (np.arange(seq, dtype=np.int64) * frames) // seq


class TestScaledMaskedSoftmax:
    @pytest.mark.parametrize(
        "heads,seq,frames",
        [
            (16, 96, 1),  # one image: the mask keeps everything
            (16, 96, 2),  # two frames: block-diagonal mask
            (4, 300, 2),  # cols not a multiple of the block
            (2, 1024, 3),  # several elements per thread, uneven frames
            (2, 7296, 1),  # the vision encoder's row width, one image
            (1, 7296, 2),
            (2, 1500, 2),  # not a multiple of 8: per-element form
        ],
    )
    def test_fp16(self, model_runner, heads, seq, frames):
        model = _make_model(np.float16, heads, seq, 0.1178, -65504.0)
        rng = np.random.default_rng(7)
        scores = rng.uniform(-40, 40, [heads, seq, seq]).astype(np.float16)
        actual, expected = model_runner.run_sample(
            model, [scores, _frame_ids(seq, frames)]
        )
        compare_outputs(actual, expected, atol=2e-3)

    def test_fp16_wide_rows(self, model_runner):
        # Wider than the register-resident form: the strided kernel runs.
        heads, seq = 1, 8448
        model = _make_model(np.float16, heads, seq, 0.125, -65504.0)
        rng = np.random.default_rng(11)
        scores = rng.uniform(-20, 20, [heads, seq, seq]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [scores, _frame_ids(seq, 2)])
        compare_outputs(actual, expected, atol=2e-3)

    @pytest.mark.parametrize("frames", [1, 2])
    def test_fp32(self, model_runner, frames):
        heads, seq = 4, 160
        model = _make_model(np.float32, heads, seq, 0.25, -1e9)
        rng = np.random.default_rng(3)
        scores = rng.uniform(-20, 20, [heads, seq, seq]).astype(np.float32)
        actual, expected = model_runner.run_sample(
            model, [scores, _frame_ids(seq, frames)]
        )
        compare_outputs(actual, expected, atol=1e-5)

    @pytest.mark.parametrize(
        "row_len,col_len",
        [
            (1, 96),  # one row id against every column: whole rows kept or masked
            (96, 1),  # one column id: each row is all kept or all masked
        ],
    )
    def test_broadcast_ids(self, model_runner, row_len, col_len):
        heads, seq = 4, 96
        model = _make_split_ids_model(np.float16, [heads, seq, seq], row_len, col_len)
        rng = np.random.default_rng(5)
        scores = rng.uniform(-20, 20, [heads, seq, seq]).astype(np.float16)
        ids = _frame_ids(seq, 2)
        rows = ids[:row_len] if row_len > 1 else np.array([1], dtype=np.int64)
        cols = ids[:col_len] if col_len > 1 else np.array([1], dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [scores, rows, cols])
        compare_outputs(actual, expected, atol=2e-3)

    def test_rank4(self, model_runner):
        # [batch, heads, R, C]: outer folds both leading axes.
        shape = [2, 3, 128, 128]
        model = _make_split_ids_model(np.float16, shape, 128, 128)
        rng = np.random.default_rng(9)
        scores = rng.uniform(-20, 20, shape).astype(np.float16)
        ids = _frame_ids(128, 2)
        actual, expected = model_runner.run_sample(model, [scores, ids, ids])
        compare_outputs(actual, expected, atol=2e-3)

    def test_scores_from_matmul(self, model_runner):
        heads, seq, dim = 4, 256, 64
        model = _make_qk_model(heads, seq, dim)
        rng = np.random.default_rng(13)
        q = rng.uniform(-2, 2, [heads, seq, dim]).astype(np.float16)
        kt = rng.uniform(-2, 2, [heads, dim, seq]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [q, kt, _frame_ids(seq, 2)])
        compare_outputs(actual, expected, atol=2e-3)
