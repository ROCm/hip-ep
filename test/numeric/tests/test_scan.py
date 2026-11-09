#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for forward, axis-0 ONNX Scan."""

import numpy as np
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_scan_without_state_model(sequence_length: int):
    slice_info = helper.make_tensor_value_info("slice_in", TensorProto.FLOAT, [4])
    value_info = helper.make_tensor_value_info("value_out", TensorProto.FLOAT, [4])
    body = helper.make_graph(
        [helper.make_node("Identity", ["slice_in"], ["value_out"])],
        "scan_body",
        [slice_info],
        [value_info],
    )

    sequence = helper.make_tensor_value_info(
        "sequence", TensorProto.FLOAT, [sequence_length, 4]
    )
    stacked = helper.make_tensor_value_info(
        "stacked", TensorProto.FLOAT, [sequence_length, 4]
    )
    scan = helper.make_node(
        "Scan",
        ["sequence"],
        ["stacked"],
        body=body,
        num_scan_inputs=1,
    )
    return make_model_from_nodes([scan], [sequence], [stacked])


def _make_stateful_scan_model(sequence_length: int):
    state_in = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [4])
    slice_in = helper.make_tensor_value_info("slice_in", TensorProto.FLOAT, [4])
    state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [4])
    value_out = helper.make_tensor_value_info("value_out", TensorProto.FLOAT, [4])
    body = helper.make_graph(
        [
            helper.make_node("Add", ["state_in", "slice_in"], ["state_out"]),
            helper.make_node("Add", ["state_out", "bias_capture"], ["value_out"]),
        ],
        "scan_body",
        [state_in, slice_in],
        [state_out, value_out],
    )

    initial_state = helper.make_tensor_value_info(
        "initial_state", TensorProto.FLOAT, [4]
    )
    sequence = helper.make_tensor_value_info(
        "sequence", TensorProto.FLOAT, [sequence_length, 4]
    )
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4])
    bias_extra = helper.make_tensor_value_info("bias_extra", TensorProto.FLOAT, [4])
    final_state = helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [4])
    stacked = helper.make_tensor_value_info(
        "stacked", TensorProto.FLOAT, [sequence_length, 4]
    )
    scan = helper.make_node(
        "Scan",
        ["initial_state", "sequence"],
        ["final_state", "stacked"],
        body=body,
        num_scan_inputs=1,
    )
    capture_producer = helper.make_node("Add", ["bias", "bias_extra"], ["bias_capture"])
    return make_model_from_nodes(
        [capture_producer, scan],
        [initial_state, sequence, bias, bias_extra],
        [final_state, stacked],
    )


class TestScan:
    def test_scan_without_state(self, model_runner):
        model = _make_scan_without_state_model(sequence_length=3)
        sequence = (
            np.random.default_rng(1401).standard_normal((3, 4)).astype(np.float32)
        )

        actual, expected = model_runner.run_sample(model, [sequence], reference="cpu")
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_state_capture_and_stacked_output(self, model_runner):
        model = _make_stateful_scan_model(sequence_length=3)
        rng = np.random.default_rng(1402)
        initial_state = rng.standard_normal(4).astype(np.float32)
        sequence = rng.standard_normal((3, 4)).astype(np.float32)
        bias = rng.standard_normal(4).astype(np.float32)
        bias_extra = rng.standard_normal(4).astype(np.float32)

        actual, expected = model_runner.run_sample(
            model,
            [initial_state, sequence, bias, bias_extra],
            reference="cpu",
        )
        compare_outputs(actual, expected, atol=1e-6, rtol=1e-6)
