#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""
Generate a two-layer convolution ONNX model for MLIR integration testing.

This script creates a simple Conv → ReLU → Conv → ReLU pipeline:
- Input: [1, 3, 224, 224] (batch=1, RGB channels, 224x224 image)
- Conv1: stride=1 (preserves spatial dimensions) → [1, 64, 224, 224]
- ReLU1
- Conv2: stride=2 (halves spatial dimensions) → [1, 64, 112, 112]
- ReLU2
- Output: [1, 64, 112, 112]

Usage:
    python gen_two_layer_conv_model.py --output models/two_layer_conv.onnx
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto


def create_two_layer_conv_model():
    """Create a two-layer convolution model with ReLU activations."""

    # Input: [1, 3, 224, 224]
    input_shape = [1, 3, 224, 224]

    # Conv1: 3 → 64 channels, kernel=3x3, stride=1, padding=1
    conv1_weight_shape = [64, 3, 3, 3]  # [out_channels, in_channels, kH, kW]
    conv1_bias_shape = [64]
    # conv1_output_shape = [1, 64, 224, 224]  # stride=1, padding=1 preserves size

    # Conv2: 64 → 64 channels, kernel=3x3, stride=2, padding=1
    conv2_weight_shape = [64, 64, 3, 3]
    conv2_bias_shape = [64]
    conv2_output_shape = [1, 64, 112, 112]  # stride=2 halves spatial dimensions

    # Random weight initialization for testing
    np.random.seed(42)
    conv1_weight = np.random.randn(*conv1_weight_shape).astype(np.float32) * 0.01
    conv1_bias = np.random.randn(*conv1_bias_shape).astype(np.float32) * 0.01
    conv2_weight = np.random.randn(*conv2_weight_shape).astype(np.float32) * 0.01
    conv2_bias = np.random.randn(*conv2_bias_shape).astype(np.float32) * 0.01

    # Create nodes
    nodes = [
        # Conv1
        helper.make_node(
            "Conv",
            inputs=["input", "conv1_weight", "conv1_bias"],
            outputs=["conv1_output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],  # ONNX uses [top, left, bottom, right]
            name="conv1",
        ),
        # ReLU1
        helper.make_node(
            "Relu", inputs=["conv1_output"], outputs=["relu1_output"], name="relu1"
        ),
        # Conv2
        helper.make_node(
            "Conv",
            inputs=["relu1_output", "conv2_weight", "conv2_bias"],
            outputs=["conv2_output"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            name="conv2",
        ),
        # ReLU2
        helper.make_node(
            "Relu", inputs=["conv2_output"], outputs=["output"], name="relu2"
        ),
    ]

    # Create graph inputs
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)]

    # Create graph outputs
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, conv2_output_shape)
    ]

    # Create initializers (weights and biases)
    initializers = [
        helper.make_tensor(
            "conv1_weight",
            TensorProto.FLOAT,
            conv1_weight_shape,
            conv1_weight.flatten(),
        ),
        helper.make_tensor(
            "conv1_bias", TensorProto.FLOAT, conv1_bias_shape, conv1_bias.flatten()
        ),
        helper.make_tensor(
            "conv2_weight",
            TensorProto.FLOAT,
            conv2_weight_shape,
            conv2_weight.flatten(),
        ),
        helper.make_tensor(
            "conv2_bias", TensorProto.FLOAT, conv2_bias_shape, conv2_bias.flatten()
        ),
    ]

    # Create the graph
    graph = helper.make_graph(nodes, "two_layer_conv", inputs, outputs, initializers)

    # Create the model
    model = helper.make_model(graph, producer_name="onnx-hipdnn-ep-test")
    model.opset_import[0].version = 13

    # Check model validity
    onnx.checker.check_model(model)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate two-layer convolution ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the ONNX model (e.g., models/two_layer_conv.onnx)",
    )
    args = parser.parse_args()

    print("Generating two-layer convolution model...")
    model = create_two_layer_conv_model()

    print(f"Saving model to {args.output}...")
    onnx.save(model, args.output)

    print("Model saved successfully!")
    print("\nModel structure:")
    print("  Input: [1, 3, 224, 224]")
    print("  Conv1 (stride=1, padding=1) -> [1, 64, 224, 224]")
    print("  ReLU1")
    print("  Conv2 (stride=2, padding=1) -> [1, 64, 112, 112]")
    print("  ReLU2")
    print("  Output: [1, 64, 112, 112]")


if __name__ == "__main__":
    main()
