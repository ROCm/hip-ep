#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for CumSum (added by qwen-vision-kernels PR).

ONNX-14 CumSum signature:
    y = CumSum(x, axis) {exclusive, reverse}

`axis` is a 0-D scalar (i32 or i64) tensor; both flags default to 0.
The runtime supports f16, f32, i32, i64 inputs.

Real-model footprint: attention-mask position-id computation in
Llama / Qwen graphs uses a single CumSum on int64 [1, S] along axis=1
(exclusive=0, reverse=0).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

SEQ_LENS = [1, 128]


def _make_cumsum_model(
    dtype,
    input_shape: list[int],
    axis: int,
    exclusive: int = 0,
    reverse: int = 0,
):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    # CumSum's `axis` is a 0-D scalar tensor (initializer here).
    axis_init = numpy_helper.from_array(np.array(axis, dtype=np.int64), name="axis")
    Y = helper.make_tensor_value_info("Y", tp, list(input_shape))
    attrs = {}
    if exclusive:
        attrs["exclusive"] = exclusive
    if reverse:
        attrs["reverse"] = reverse
    node = helper.make_node("CumSum", ["X", "axis"], ["Y"], **attrs)
    return make_model_from_nodes([node], [X], [Y], initializers=[axis_init])


class TestCumSum:
    @pytest.mark.parametrize(
        "dtype,shape,axis",
        [
            (np.float32, [4, 8], 1),
            (np.float32, [4, 8], 0),
            (np.float16, [2, 16], -1),
            (np.int32, [3, 5], 1),
            (np.int64, [3, 5], 0),
        ],
    )
    def test_cumsum(self, model_runner, dtype, shape, axis):
        model = _make_cumsum_model(dtype, shape, axis)
        rng = np.random.default_rng(501)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-5, 5, shape, dtype=dtype)
        else:
            # Bounded so fp16 cumulative sums of up to ~16 partials stay
            # well within range.
            x = rng.uniform(-0.5, 0.5, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        if np.issubdtype(dtype, np.integer):
            atol = 0
        elif dtype == np.float16:
            atol = 1e-2
        else:
            atol = 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    @pytest.mark.parametrize("exclusive", [0, 1])
    @pytest.mark.parametrize("reverse", [0, 1])
    def test_cumsum_flag_combinations(self, model_runner, exclusive, reverse):
        """All four (exclusive, reverse) combinations on i64 [3, 6]."""
        shape = [3, 6]
        model = _make_cumsum_model(
            np.int64,
            shape,
            axis=1,
            exclusive=exclusive,
            reverse=reverse,
        )
        rng = np.random.default_rng(502)
        x = rng.integers(-5, 5, shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_cumsum_position_ids_shape(self, model_runner, seq_len):
        """i64 [1, S], axis=1 -- the canonical attention-mask -> position
        ids pattern."""
        shape = [1, seq_len]
        model = _make_cumsum_model(np.int64, shape, axis=1)
        rng = np.random.default_rng(503)
        x = rng.integers(0, 2, shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
