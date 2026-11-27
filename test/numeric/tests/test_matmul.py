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

    @staticmethod
    def _model_b_dynamic_batch(a_shape, k, n):
        """MatMul of a constant A and an input B whose batch dim is dynamic."""
        tp = TensorProto.FLOAT16
        rng = np.random.default_rng(654)
        a = numpy_helper.from_array(
            rng.uniform(-0.5, 0.5, a_shape).astype(np.float16), name="A"
        )
        B = helper.make_tensor_value_info("B", tp, ["batch", k, n])
        Y = helper.make_tensor_value_info("Y", tp, ["batch", a_shape[-2], n])
        model = make_model_from_nodes(
            [helper.make_node("MatMul", ["A", "B"], ["Y"])], [B], [Y], initializers=[a]
        )
        return model, rng

    @pytest.mark.parametrize("a_shape", [[24, 16], [1, 24, 16]], ids=["rank2", "unit"])
    def test_matmul_b_dynamic_batch_one(self, model_runner, a_shape):
        """A has no batch (or a unit one) and B's dynamic batch is 1 at run time.

        The output must be [1, M, N]. The destination used to take its batch
        from A's first dim, which is M for a rank-2 A.
        """
        model, rng = self._model_b_dynamic_batch(a_shape, 16, 8)
        b = rng.uniform(-1, 1, [1, 16, 8]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [b], reference="cpu")
        assert actual[0].shape == (1, 24, 8)
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("batch", [2, 3])
    @pytest.mark.parametrize("a_shape", [[24, 16], [1, 24, 16]], ids=["rank2", "unit"])
    def test_matmul_b_dynamic_batch_shape(self, model_runner, a_shape, batch):
        """The output shape follows B's batch, whatever its size.

        Only the shape is compared: the values of a matmul with a shared A
        and a batch above 1 depend on the matmul lowering, not on the shape
        of the destination.
        """
        model, rng = self._model_b_dynamic_batch(a_shape, 16, 8)
        b = rng.uniform(-1, 1, [batch, 16, 8]).astype(np.float16)
        actual, _ = model_runner.run_sample(model, [b], reference="cpu")
        assert actual[0].shape == (batch, 24, 8)
