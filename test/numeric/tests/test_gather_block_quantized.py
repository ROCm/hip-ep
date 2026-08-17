#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for com.microsoft.GatherBlockQuantized.

The current Microsoft-domain schema admits UINT8 but not INT8 for T1, so ORT
CPU can provide a strict reference only for unsigned 8-bit storage. Signed
byte interpretation is covered by the shared host/device runtime utility test.
"""

import numpy as np
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_uint8_gather_block_quantized_model():
    data = helper.make_tensor_value_info("data", TensorProto.UINT8, [4, 32])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2])
    scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 32])
    node = helper.make_node(
        "GatherBlockQuantized",
        ["data", "indices", "scales"],
        ["output"],
        domain="com.microsoft",
        bits=8,
        block_size=16,
        gather_axis=0,
        quantize_axis=1,
    )
    return make_model_from_nodes(
        [node],
        [data, indices, scales],
        [output],
        opset=21,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def test_gather_block_quantized_uint8_default_zero_point(model_runner):
    model = _make_uint8_gather_block_quantized_model()
    data = np.array(
        [
            np.arange(0, 32),
            np.arange(96, 128),
            np.arange(128, 160),
            np.arange(224, 256),
        ],
        dtype=np.uint8,
    )
    indices = np.array([3, 1], dtype=np.int64)
    scales = np.array(
        [[0.25, 0.5], [1.0, 2.0], [0.125, 0.25], [0.5, 1.5]],
        dtype=np.float32,
    )

    actual, expected = model_runner.run_sample(
        model, [data, indices, scales], reference="cpu"
    )
    compare_outputs(actual, expected, atol=0, rtol=0)
