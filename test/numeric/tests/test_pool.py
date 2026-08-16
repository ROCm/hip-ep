#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""GPU-vs-CPU correctness for ONNX Pool ceil-mode window geometry."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_maxpool_model(
    input_shape,
    output_shape,
    *,
    kernel,
    stride,
    pads,
    dilation,
    with_indices,
):
    model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    outputs = [helper.make_tensor_value_info("values", TensorProto.FLOAT, output_shape)]
    output_names = ["values"]
    if with_indices:
        outputs.append(
            helper.make_tensor_value_info("indices", TensorProto.INT64, output_shape)
        )
        output_names.append("indices")

    node = helper.make_node(
        "MaxPool",
        ["input"],
        output_names,
        kernel_shape=kernel,
        strides=stride,
        pads=pads,
        dilations=dilation,
        ceil_mode=1,
    )
    return make_model_from_nodes([node], [model_input], outputs)


class TestPool:
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_maxpool_ceil_trailing_window_with_indices(self, model_runner, dynamic):
        """The raw extent 3 is corrected to 2 for pads=[0, 2]."""
        input_shape = [None, None, None] if dynamic else [1, 1, 4]
        output_shape = [None, None, None] if dynamic else [1, 1, 2]
        model = _make_maxpool_model(
            input_shape,
            output_shape,
            kernel=[3],
            stride=[2],
            pads=[0, 2],
            dilation=[1],
            with_indices=True,
        )
        x = np.array([[[1.0, 4.0, 2.0, 3.0]]], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected)

    def test_maxpool_ceil_trailing_window_with_dilation(self, model_runner):
        """Effective kernel 3 participates before the trailing correction."""
        model = _make_maxpool_model(
            [1, 1, 3],
            [1, 1, 1],
            kernel=[2],
            stride=[3],
            pads=[0, 1],
            dilation=[2],
            with_indices=False,
        )
        x = np.array([[[1.0, 2.0, 4.0]]], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected)
