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
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_bf16_cross_axis_broadcast(self, model_runner, dynamic):
        """BF16's two-byte ABI remains distinct from fp16 across batch axes."""
        rng = np.random.default_rng(520)
        a_shape, b_shape = [2, 1, 19, 37], [1, 3, 37, 21]
        a = rng.uniform(-0.25, 0.25, a_shape).astype(np.float32)
        b = rng.uniform(-0.25, 0.25, b_shape).astype(np.float32)
        a_dims = ["a", 1, 19, 37] if dynamic else a_shape
        b_dims = [1, "b", 37, 21] if dynamic else b_shape
        y_dims = ["a", "b", 19, 21] if dynamic else [2, 3, 19, 21]
        inputs = [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, a_dims),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, b_dims),
        ]
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_dims)]
        casts = [
            helper.make_node("Cast", ["A"], ["aq"], to=TensorProto.BFLOAT16),
            helper.make_node("Cast", ["B"], ["bq"], to=TensorProto.BFLOAT16),
        ]
        model = make_model_from_nodes(
            casts
            + [
                helper.make_node("MatMul", ["aq", "bq"], ["yq"]),
                helper.make_node("Cast", ["yq"], ["Y"], to=TensorProto.FLOAT),
            ],
            inputs,
            outputs,
        )
        # The CPU lacks a BF16 MatMul implementation. Exact operand widening,
        # fp32 multiplication and BF16 output rounding provide its reference.
        reference = make_model_from_nodes(
            casts
            + [
                helper.make_node("Cast", ["aq"], ["af"], to=TensorProto.FLOAT),
                helper.make_node("Cast", ["bq"], ["bf"], to=TensorProto.FLOAT),
                helper.make_node("MatMul", ["af", "bf"], ["yf"]),
                helper.make_node("Cast", ["yf"], ["yq"], to=TensorProto.BFLOAT16),
                helper.make_node("Cast", ["yq"], ["Y"], to=TensorProto.FLOAT),
            ],
            inputs,
            outputs,
        )
        actual, expected = model_runner.run_sample(
            model, [a, b], reference="cpu", reference_model=reference
        )
        compare_outputs(actual, expected, atol=1e-2, rtol=2e-2)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [
            ([19, 37], [37, 21]),
            ([2, 19, 37], [37, 21]),
            ([2, 19, 37], [2, 37, 21]),
            ([2, 3, 19, 37], [1, 3, 37, 21]),
            ([2, 1, 19, 37], [1, 3, 37, 21]),
            ([19, 37], [3, 37, 21]),
            ([37], [3, 37, 21]),
            ([2, 19, 37], [37]),
            ([2, 1] + [1] * 9 + [19, 37], [1, 3] + [1] * 9 + [37, 21]),
        ],
    )
    def test_dense_batch_broadcast(self, model_runner, dtype, a_shape, b_shape):
        """Each batch uses the right source matrix, including cross-axis reuse."""
        rng = np.random.default_rng(517)
        a = rng.uniform(-0.25, 0.25, a_shape).astype(dtype)
        b = rng.uniform(-0.25, 0.25, b_shape).astype(dtype)
        output_shape = list(np.matmul(a, b).shape)
        elem_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
        model = make_model_from_nodes(
            [helper.make_node("MatMul", ["A", "B"], ["Y"])],
            [
                helper.make_tensor_value_info("A", elem_type, a_shape),
                helper.make_tensor_value_info("B", elem_type, b_shape),
            ],
            [helper.make_tensor_value_info("Y", elem_type, output_shape)],
        )
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        compare_outputs(actual, expected, atol=1e-3 if dtype == np.float16 else 1e-5)

    @pytest.mark.parametrize("b_batch", [1, 2])
    def test_dynamic_batch_broadcast(self, model_runner, b_batch):
        """Runtime leading dimensions distinguish shared from per-batch B."""
        rng = np.random.default_rng(518)
        a = rng.uniform(-0.25, 0.25, [2, 19, 37]).astype(np.float32)
        b = rng.uniform(-0.25, 0.25, [b_batch, 37, 21]).astype(np.float32)
        model = make_model_from_nodes(
            [helper.make_node("MatMul", ["A", "B"], ["Y"])],
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, ["a", 19, 37]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, ["b", 37, 21]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["out", 19, 21])],
        )
        actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
        compare_outputs(actual, expected, atol=1e-5)

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

    @pytest.mark.parametrize(
        "input_shape,weight_shape",
        [
            ([1, 1, 2880], [2880, 128]),  # gpt-oss-120b router
            ([1, 1, 2880], [2880, 32]),  # gpt-oss-20b router
            ([1, 520], [520, 8]),
            ([1, 1, 2048], [2048, 256]),
            ([1, 1, 2048], [2048, 512]),
            ([1, 1, 2048], [2048, 520]),  # above the one-row GEMV N cap
            ([1, 8192], [8192, 128]),
            ([1, 8200], [8200, 128]),  # above the one-row GEMV K cap
        ],
    )
    def test_matmul_one_row(self, model_runner, input_shape, weight_shape):
        """One-row MatMul with a narrow weight, the MoE router's shape."""
        model = _make_matmul_model(input_shape, weight_shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, input_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x], reference="cpu")
        compare_outputs(actual, expected, atol=1e-3)

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
