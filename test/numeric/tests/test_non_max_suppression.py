#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""NonMaxSuppression: selected rows, not the candidate capacity.

Six overlapping boxes with max_output_boxes_per_class = 1 must return one
row, not a [6, 3] buffer. Zero boxes must return [0, 3] without indexing an
empty score buffer.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _nms_model(num_boxes: int):
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, num_boxes, 4])
    scores = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [1, 1, num_boxes]
    )
    selected = helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3])
    initializers = [
        numpy_helper.from_array(np.array(1, dtype=np.int64), name="max_out"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="iou"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="score"),
    ]
    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_out", "iou", "score"],
        ["selected"],
        center_point_box=0,
    )
    return make_model_from_nodes(
        [node], [boxes, scores], [selected], initializers=initializers
    )


class TestNonMaxSuppression:
    def test_one_of_six_overlapping_boxes(self, model_runner):
        boxes = np.array(
            [[[i * 0.1, 0.0, 1.0 + i * 0.1, 1.0] for i in range(6)]],
            dtype=np.float32,
        )
        scores = np.array([[[0.9, 0.2, 0.4, 0.1, 0.3, 0.05]]], dtype=np.float32)
        actual, expected = model_runner.run_sample(_nms_model(6), [boxes, scores])
        compare_outputs(actual, expected, atol=0)

    def test_zero_boxes(self, model_runner):
        boxes = np.zeros((1, 0, 4), dtype=np.float32)
        scores = np.zeros((1, 1, 0), dtype=np.float32)
        actual, expected = model_runner.run_sample(_nms_model(0), [boxes, scores])
        assert actual[0].shape == (0, 3)
        assert expected[0].shape == (0, 3)
