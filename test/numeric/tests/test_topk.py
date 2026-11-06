#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX TopK op (values + indices outputs).

K is supplied as a constant initializer so the ONNX shape-inference resolves
the output extent (k) statically -- the runtime still reads k from the K
buffer at run time (real/top_k.cpp does a D2H copy of k), so the GPU kernel is
exercised. `largest` and `sorted` are both covered.

Outputs are compared bit-exact against the ORT CPU reference (atol=0).
K above 256 runs on the GPU for full contiguous last-axis sorts and on the host
otherwise; both are covered. The GPU full-sort NaN case checks the documented
GPU policy directly because ONNX does not define NaN ordering.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_topk_model(
    x_shape: list[int],
    k: int,
    axis: int,
    largest: int,
    sorted_: int,
    dtype=np.float32,
):
    out_shape = list(x_shape)
    out_shape[axis] = k
    X = helper.make_tensor_value_info("x", np_to_onnx_type(dtype), list(x_shape))
    values = helper.make_tensor_value_info("values", np_to_onnx_type(dtype), out_shape)
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(np.int64), out_shape
    )
    k_init = numpy_helper.from_array(np.array([k], dtype=np.int64), name="k")
    node = helper.make_node(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        axis=axis,
        largest=largest,
        sorted=sorted_,
    )
    return make_model_from_nodes([node], [X], [values, indices], initializers=[k_init])


def _distinct_rows(rng, rows: int, cols: int) -> np.ndarray:
    """Each row is a shuffled arange -> globally distinct within a row, so
    TopK has no ties to break."""
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        out[r] = rng.permutation(cols).astype(np.float32) + rng.uniform(
            -0.25, 0.25, cols
        ).astype(np.float32)
    return out


class TestTopK:
    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize("sorted_", [1])
    def test_topk_axis1(self, model_runner, largest, sorted_):
        x_shape = [3, 8]
        k = 3
        model = _make_topk_model(x_shape, k, axis=1, largest=largest, sorted_=sorted_)
        rng = np.random.default_rng(601)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_large_full_sort_f16(self, model_runner):
        """f16 values are represented exactly by the f32 workspace keys."""
        cols = 1024
        model = _make_topk_model(
            [1, cols], cols, axis=1, largest=0, sorted_=1, dtype=np.float16
        )
        x = (
            np.random.default_rng(605)
            .permutation(cols)
            .astype(np.float16)[np.newaxis, :]
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize(
        "rows,cols,k",
        [
            (2, 32000, 1),  # argmax / argmin over a vocab row
            (3, 1000, 16),  # one block per row
            (2, 20000, 32),  # rows past 8192 take the chunked two-pass path
            (2, 248320, 128),  # 32 < k <= 256 stays on the two-pass path
            (1, 100000, 256),
        ],
    )
    def test_topk_last_axis_vocab(self, model_runner, rows, cols, k, largest):
        model = _make_topk_model([rows, cols], k, axis=1, largest=largest, sorted_=1)
        rng = np.random.default_rng(603)
        x = _distinct_rows(rng, rows, cols)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_full_k(self, model_runner):
        """k == axis length -> a full sort of each row."""
        x_shape = [4, 5]
        k = 5
        model = _make_topk_model(x_shape, k, axis=1, largest=1, sorted_=1)
        rng = np.random.default_rng(602)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize(
        "axis_size,k",
        [
            pytest.param(300, 257, id="partial-k257"),
            pytest.param(257, 257, id="full-axis"),
        ],
    )
    def test_topk_host_non_last_axis_with_ties(
        self, model_runner, axis_size, k, largest
    ):
        model = _make_topk_model([axis_size, 3], k, axis=0, largest=largest, sorted_=1)
        tied = (np.arange(axis_size, dtype=np.float32) // 2).astype(np.float32)
        x = np.stack((tied, np.roll(tied, 7), -tied), axis=1)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_host_f64_preserves_comparison_precision(self, model_runner):
        x = np.concatenate(
            (
                np.array([1.0, 1.0 + 2.0**-30], dtype=np.float64),
                np.full(256, 2.0, dtype=np.float64),
            )
        )
        model = _make_topk_model(
            [x.size], 257, axis=0, largest=1, sorted_=1, dtype=np.float64
        )

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize("cols", [256, 257, 1024, 9460])
    def test_topk_large_full_sort(self, model_runner, cols, largest):
        """Full sorts cover the K=256/257 dispatch boundary and large K."""
        x_shape = [2, cols]
        model = _make_topk_model(x_shape, cols, axis=1, largest=largest, sorted_=1)
        rng = np.random.default_rng(604)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("largest", [1, 0])
    def test_topk_large_full_sort_nan_policy(self, model_runner, largest):
        """Finite values precede NaNs; ties, including NaNs, use lower indices."""
        cols = 512
        model = _make_topk_model([1, cols], cols, axis=1, largest=largest, sorted_=1)
        x = (np.arange(cols, dtype=np.float32) % 17)[np.newaxis, :]
        x[0, [3, 100, 400]] = np.nan
        actual, _ = model_runner.run_sample(model, [x])

        expected_indices = sorted(
            range(cols),
            key=lambda i: (
                bool(np.isnan(x[0, i])),
                0.0
                if np.isnan(x[0, i])
                else (-float(x[0, i]) if largest else float(x[0, i])),
                i,
            ),
        )
        np.testing.assert_array_equal(
            actual[1][0], np.asarray(expected_indices, dtype=np.int64)
        )
        finite_count = cols - 3
        assert not np.isnan(actual[0][0, :finite_count]).any()
        assert np.isnan(actual[0][0, finite_count:]).all()
