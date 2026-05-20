#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for the Sigmoid activation op."""

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# Llama MLP gate width: SwiGLU's sigmoid arm sees [1, S, 14336] f16 at every
# decoder layer, so this shape exercises the production kernel layout.
LLAMA_GATE_WIDTH = 14336
SEQ_LENS = [1, 128]


def _make_sigmoid_model(dtype, shape: list[int]):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, shape)
    Y = helper.make_tensor_value_info("Y", tp, shape)
    node = helper.make_node("Sigmoid", ["X"], ["Y"])
    return make_model_from_nodes([node], [X], [Y])


class TestSigmoid:
    def test_sigmoid_small(self, model_runner):
        """Trivial [1, 10] fp16 sanity check (CPU cost negligible)."""
        shape = [1, 10]
        model = _make_sigmoid_model(np.float16, shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_sigmoid_llama_gate_shape(self, model_runner, seq_len):
        """Sigmoid on the Llama MLP SwiGLU gate shape: [1, S, 14336] fp16.

        First invocation pays the ORT CPU cost; subsequent runs hit the
        cache. This shape is the canonical case where caching matters --
        14336 elements per token is still cheap on CPU, but the same
        pattern composes with much heavier ops in the matmul suite.
        """
        shape = [1, seq_len, LLAMA_GATE_WIDTH]
        model = _make_sigmoid_model(np.float16, shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)
