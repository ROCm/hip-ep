#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX TensorScatter op (opset 24).

output = copy(past_cache); then for every batch b and update row t,
output[b, ..., w, ...] = update[b, ..., t, ...] along `axis`, with
w = write_indices[b] + t (mode "linear") or that modulo the cache length
(mode "circular"). write_indices is optional and defaults to zeros. Every
write position is distinct, so the result is bit-exact (atol=0).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_model(cache_shape, update_shape, dtype, with_indices, **attrs):
    elem = np_to_onnx_type(dtype)
    inputs = [
        helper.make_tensor_value_info("past", elem, list(cache_shape)),
        helper.make_tensor_value_info("update", elem, list(update_shape)),
    ]
    node_inputs = ["past", "update"]
    if with_indices:
        inputs.append(
            helper.make_tensor_value_info(
                "write_indices", np_to_onnx_type(np.int64), [cache_shape[0]]
            )
        )
        node_inputs.append("write_indices")
    out = helper.make_tensor_value_info("present", elem, list(cache_shape))
    node = helper.make_node("TensorScatter", node_inputs, ["present"], **attrs)
    model = make_model_from_nodes([node], inputs, [out], opset=24)
    # Opset 24 needs IR 11; newer onnx packages stamp a version ORT may not
    # load yet.
    model.ir_version = 11
    return model


class TestTensorScatter:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_kv_cache_default_axis(self, model_runner, dtype):
        """[B, heads, max_seq, head_dim] cache, axis -2, per-batch positions."""
        cache_shape, update_shape = [2, 4, 16, 8], [2, 4, 3, 8]
        model = _make_model(cache_shape, update_shape, dtype, True)
        rng = np.random.default_rng(611)
        past = rng.standard_normal(cache_shape).astype(dtype)
        update = rng.standard_normal(update_shape).astype(dtype)
        write_indices = np.array([0, 13], np.int64)
        actual, expected = model_runner.run_sample(
            model, [past, update, write_indices]
        )
        compare_outputs(actual, expected, atol=0)

    def test_axis1_rank3(self, model_runner):
        """[B, max_seq, hidden] cache scattered along axis 1."""
        cache_shape, update_shape = [2, 10, 12], [2, 3, 12]
        model = _make_model(cache_shape, update_shape, np.float32, True, axis=1)
        rng = np.random.default_rng(612)
        past = rng.standard_normal(cache_shape).astype(np.float32)
        update = rng.standard_normal(update_shape).astype(np.float32)
        write_indices = np.array([7, 2], np.int64)
        actual, expected = model_runner.run_sample(
            model, [past, update, write_indices]
        )
        compare_outputs(actual, expected, atol=0)

    def test_circular_wraps(self, model_runner):
        """Rows written past the end wrap to the start of the cache."""
        cache_shape, update_shape = [1, 2, 8, 4], [1, 2, 4, 4]
        model = _make_model(
            cache_shape, update_shape, np.float16, True, mode="circular"
        )
        rng = np.random.default_rng(613)
        past = rng.standard_normal(cache_shape).astype(np.float16)
        update = rng.standard_normal(update_shape).astype(np.float16)
        write_indices = np.array([6], np.int64)
        actual, expected = model_runner.run_sample(
            model, [past, update, write_indices]
        )
        compare_outputs(actual, expected, atol=0)

    def test_no_write_indices(self, model_runner):
        """Without write_indices every batch writes from row 0."""
        cache_shape, update_shape = [2, 1, 6, 5], [2, 1, 2, 5]
        model = _make_model(cache_shape, update_shape, np.float16, False)
        rng = np.random.default_rng(614)
        past = rng.standard_normal(cache_shape).astype(np.float16)
        update = rng.standard_normal(update_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [past, update])
        compare_outputs(actual, expected, atol=0)
