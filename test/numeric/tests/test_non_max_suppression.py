#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for ONNX NonMaxSuppression."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_nms_model(
    batch,
    num_classes,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold=None,
    center_point_box=0,
):
    boxes = helper.make_tensor_value_info(
        "boxes", TensorProto.FLOAT, [batch, "num_boxes", 4]
    )
    scores = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [batch, num_classes, "num_boxes"]
    )
    selected = helper.make_tensor_value_info(
        "selected_indices", TensorProto.INT64, ["num_selected", 3]
    )

    def scalar(name, tp, value):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name + "_v", tp, [1], [value]),
        )

    nodes = [
        scalar("max_out", TensorProto.INT64, max_output_boxes_per_class),
        scalar("iou", TensorProto.FLOAT, iou_threshold),
    ]
    nms_inputs = ["boxes", "scores", "max_out", "iou"]
    if score_threshold is not None:
        nodes.append(scalar("score_thr", TensorProto.FLOAT, score_threshold))
        nms_inputs.append("score_thr")
    nodes.append(
        helper.make_node(
            "NonMaxSuppression",
            nms_inputs,
            ["selected_indices"],
            center_point_box=center_point_box,
        )
    )
    return make_model_from_nodes(nodes, [boxes, scores], [selected], opset=11)


def _random_corner_boxes(rng, batch, num_boxes, flip_corners=False):
    y1 = rng.uniform(0, 80, (batch, num_boxes))
    x1 = rng.uniform(0, 80, (batch, num_boxes))
    h = rng.uniform(4, 30, (batch, num_boxes))
    w = rng.uniform(4, 30, (batch, num_boxes))
    boxes = np.stack([y1, x1, y1 + h, x1 + w], axis=-1)
    if flip_corners:
        boxes = boxes[..., [2, 3, 0, 1]]
    return boxes.astype(np.float32)


def _random_center_boxes(rng, batch, num_boxes):
    xc = rng.uniform(10, 90, (batch, num_boxes))
    yc = rng.uniform(10, 90, (batch, num_boxes))
    w = rng.uniform(4, 30, (batch, num_boxes))
    h = rng.uniform(4, 30, (batch, num_boxes))
    return np.stack([xc, yc, w, h], axis=-1).astype(np.float32)


class TestNonMaxSuppression:
    @pytest.mark.parametrize(
        "batch,num_classes,num_boxes,max_out,iou",
        [
            (1, 1, 64, 2000, 0.7),
            (1, 1, 300, 50, 0.5),
            (2, 3, 40, 10, 0.4),
        ],
    )
    def test_nms_corners(
        self, model_runner, batch, num_classes, num_boxes, max_out, iou
    ):
        model = _make_nms_model(batch, num_classes, max_out, iou)
        rng = np.random.default_rng(31)
        boxes = _random_corner_boxes(rng, batch, num_boxes)
        scores = rng.uniform(0, 1, (batch, num_classes, num_boxes)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [boxes, scores])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_nms_score_threshold_flipped_corners(self, model_runner):
        model = _make_nms_model(1, 2, 20, 0.5, score_threshold=0.3)
        rng = np.random.default_rng(32)
        boxes = _random_corner_boxes(rng, 1, 50, flip_corners=True)
        scores = rng.uniform(0, 1, (1, 2, 50)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [boxes, scores])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_nms_center_point_box(self, model_runner):
        model = _make_nms_model(1, 1, 30, 0.45, center_point_box=1)
        rng = np.random.default_rng(33)
        boxes = _random_center_boxes(rng, 1, 60)
        scores = rng.uniform(0, 1, (1, 1, 60)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [boxes, scores])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_nms_tied_scores(self, model_runner):
        model = _make_nms_model(1, 1, 10, 0.5)
        rng = np.random.default_rng(34)
        boxes = _random_corner_boxes(rng, 1, 16)
        scores = np.round(rng.uniform(0, 1, (1, 1, 16)), 1).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [boxes, scores])
        compare_outputs(actual, expected, atol=0, rtol=0)
