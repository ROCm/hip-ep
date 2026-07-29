#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for the MatMul op (fp16, constant weight).

Weight shapes mirror Llama-3.1-8B projections:
  q_proj / o_proj       : [4096, 4096]
  gate_proj / up_proj   : [4096, 14336]

The 4096x4096 and 4096x14336 CPU MatMuls are seconds each on a single
core, which is exactly where ``reference="cache"`` pays for itself:
first run computes once, every subsequent run loads from disk.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]

LLAMA_HIDDEN = 4096
LLAMA_INTERMEDIATE = 14336


def _make_matmul_model(input_shape: list[int], weight_shape: list[int]):
    """Build a MatMul ONNX model with fp16 input X and constant fp16 weight W."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, input_shape)
    output_shape = input_shape[:-1] + [weight_shape[-1]]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, output_shape)
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])

    rng = np.random.default_rng(123)
    w_data = rng.uniform(-0.5, 0.5, weight_shape).astype(np.float16)
    w_init = numpy_helper.from_array(w_data, name="W")
    return make_model_from_nodes([node], [X], [Y], initializers=[w_init])


class TestMatMul:
    @pytest.mark.parametrize(
        "input_shape,weight_shape",
        [
            ([1, 4, 16], [16, 8]),
            ([1, 8, 32], [32, 16]),
        ],
    )
    def test_matmul_small(self, model_runner, input_shape, weight_shape):
        """Small MatMul sanity. Trivial CPU cost, so reference='cpu' is fine."""
        model = _make_matmul_model(input_shape, weight_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(
            model,
            [x],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_qo_proj_llama_shape(self, model_runner, seq_len):
        """Llama Q/O projection: [1, S, 4096] @ [4096, 4096] -> [1, S, 4096]."""
        model = _make_matmul_model(
            [1, seq_len, LLAMA_HIDDEN],
            [LLAMA_HIDDEN, LLAMA_HIDDEN],
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matmul_gate_up_proj_llama_shape(self, model_runner, seq_len):
        """Llama gate/up projection: [1, S, 4096] @ [4096, 14336] -> [1, S, 14336]."""
        model = _make_matmul_model(
            [1, seq_len, LLAMA_HIDDEN],
            [LLAMA_HIDDEN, LLAMA_INTERMEDIATE],
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)
