#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for ONNX Gemm, including transpose-aware K."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_gemm_model(a_metadata_shape, b_metadata_shape, trans_a, trans_b):
    a_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, a_metadata_shape)
    b_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, b_metadata_shape)
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
    node = helper.make_node(
        "Gemm",
        ["A", "B"],
        ["Y"],
        transA=trans_a,
        transB=trans_b,
        alpha=1.0,
        beta=0.0,
    )
    return make_model_from_nodes([node], [a_info, b_info], [y_info])


@pytest.mark.parametrize(
    "a_shape,b_shape,trans_a,trans_b,output_shape",
    [
        ([3, 4], [4, 5], 0, 0, [3, 5]),
        ([4, 3], [5, 4], 1, 1, [3, 5]),
    ],
)
def test_gemm_small(model_runner, a_shape, b_shape, trans_a, trans_b, output_shape):
    assert output_shape == [3, 5]
    model = _make_gemm_model(a_shape, b_shape, trans_a, trans_b)

    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, a_shape).astype(np.float32)
    b = rng.uniform(-1, 1, b_shape).astype(np.float32)
    actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
    compare_outputs(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    "a_metadata_shape,b_metadata_shape,a_shape,b_shape,trans_a,trans_b",
    [
        ([3, "K"], ["K", 5], [3, 4], [4, 5], 0, 0),
        ([3, "K_a"], [4, 5], [3, 4], [4, 5], 0, 0),
        ([3, 4], ["K_b", 5], [3, 4], [4, 5], 0, 0),
        (["K", 3], [5, "K"], [4, 3], [5, 4], 1, 1),
    ],
)
def test_gemm_dynamic_k_matching(
    model_runner,
    a_metadata_shape,
    b_metadata_shape,
    a_shape,
    b_shape,
    trans_a,
    trans_b,
):
    """Dynamic and one-sided-dynamic K metadata with matching runtime K."""
    model = _make_gemm_model(a_metadata_shape, b_metadata_shape, trans_a, trans_b)
    rng = np.random.default_rng(43)
    a = rng.uniform(-1, 1, a_shape).astype(np.float32)
    b = rng.uniform(-1, 1, b_shape).astype(np.float32)
    actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
    compare_outputs(actual, expected, atol=1e-5)
