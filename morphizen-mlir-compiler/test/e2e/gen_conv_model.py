#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generate a simple Conv ONNX model for testing."""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_conv_model(
    batch_size=1,
    in_channels=3,
    height=8,
    width=8,
    out_channels=16,
    kernel_size=3,
    padding=1,
    stride=1,
    has_bias=False,
    output_path="conv_model.onnx",
):
    """Create a simple Conv model."""

    # Input tensor
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch_size, in_channels, height, width])

    # Weight tensor (initializer)
    W_shape = [out_channels, in_channels, kernel_size, kernel_size]
    W_data = np.random.randn(*W_shape).astype(np.float32) * 0.1
    W = helper.make_tensor("W", TensorProto.FLOAT, W_shape, W_data.flatten())

    # Compute output shape (same padding assumed)
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch_size, out_channels, out_height, out_width])

    # Build inputs and initializers
    inputs = ["X", "W"]
    initializers = [W]

    if has_bias:
        B_data = np.random.randn(out_channels).astype(np.float32) * 0.1
        B = helper.make_tensor("B", TensorProto.FLOAT, [out_channels], B_data.flatten())
        inputs.append("B")
        initializers.append(B)

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=inputs,
        outputs=["Y"],
        name="conv",
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[stride, stride],
        dilations=[1, 1],
    )

    # Graph
    graph = helper.make_graph([conv_node], "conv_graph", [X], [Y], initializers)

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # Save
    onnx.save(model, output_path)
    print(f"Saved model to {output_path}")
    print(f"  Input: X [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  Weight: W {W_shape}")
    print(f"  Output: Y [{batch_size}, {out_channels}, {out_height}, {out_width}]")
    print(f"  Has bias: {has_bias}")

    return output_path


def create_two_layer_conv_model(
    batch_size=1,
    in_channels=3,
    height=224,
    width=224,
    mid_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    output_path="conv_model.onnx",
):
    """
    Create a two-layer Conv model similar to demo_two_layer_conv.mlir.

    Pipeline:
    Input [batch, in_channels, height, width]
      -> Conv1 (stride=1, same size) -> [batch, mid_channels, height, width]
      -> ReLU
      -> Conv2 (stride=2, half size) -> [batch, out_channels, height//2, width//2]
      -> ReLU
    """

    # Input tensor
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch_size, in_channels, height, width])

    # First Conv layer weights and bias
    W1_shape = [mid_channels, in_channels, kernel_size, kernel_size]
    W1_data = np.random.randn(*W1_shape).astype(np.float32) * 0.1
    W1 = helper.make_tensor("W1", TensorProto.FLOAT, W1_shape, W1_data.flatten())

    B1_data = np.random.randn(mid_channels).astype(np.float32) * 0.01
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, [mid_channels], B1_data.flatten())

    # Second Conv layer weights and bias
    W2_shape = [out_channels, mid_channels, kernel_size, kernel_size]
    W2_data = np.random.randn(*W2_shape).astype(np.float32) * 0.1
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, W2_shape, W2_data.flatten())

    B2_data = np.random.randn(out_channels).astype(np.float32) * 0.01
    B2 = helper.make_tensor("B2", TensorProto.FLOAT, [out_channels], B2_data.flatten())

    # First Conv (stride=1, same spatial dimensions)
    conv1_node = helper.make_node(
        "Conv",
        inputs=["X", "W1", "B1"],
        outputs=["conv1_out"],
        name="conv1",
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[1, 1],
        dilations=[1, 1],
    )

    # First ReLU
    relu1_node = helper.make_node(
        "Relu",
        inputs=["conv1_out"],
        outputs=["relu1_out"],
        name="relu1",
    )

    # Second Conv (stride=2, halves spatial dimensions)
    conv2_node = helper.make_node(
        "Conv",
        inputs=["relu1_out", "W2", "B2"],
        outputs=["conv2_out"],
        name="conv2",
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[2, 2],
        dilations=[1, 1],
    )

    # Second ReLU
    relu2_node = helper.make_node(
        "Relu",
        inputs=["conv2_out"],
        outputs=["Y"],
        name="relu2",
    )

    # Compute output shape
    out_height = height // 2
    out_width = width // 2

    # Output tensor
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch_size, out_channels, out_height, out_width])

    # Graph
    graph = helper.make_graph(
        [conv1_node, relu1_node, conv2_node, relu2_node],
        "two_layer_conv_graph",
        [X],
        [Y],
        [W1, B1, W2, B2],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model.doc_string = "Two-layer Conv model similar to demo_two_layer_conv.mlir"

    # Save
    onnx.save(model, output_path)
    print(f"Saved two-layer Conv model to {output_path}")
    print("\nModel Pipeline:")
    print(f"  Input X: [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  -> Conv1 (kernel={kernel_size}x{kernel_size}, stride=1, pad={padding})")
    print(f"  -> Conv1 output: [{batch_size}, {mid_channels}, {height}, {width}]")
    print("  -> ReLU1")
    print(f"  -> Conv2 (kernel={kernel_size}x{kernel_size}, stride=2, pad={padding})")
    print(f"  -> Conv2 output: [{batch_size}, {out_channels}, {out_height}, {out_width}]")
    print("  -> ReLU2")
    print(f"  -> Output Y: [{batch_size}, {out_channels}, {out_height}, {out_width}]")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Conv ONNX model")
    parser.add_argument(
        "--two-layer",
        action="store_true",
        help="Generate two-layer Conv model (similar to demo_two_layer_conv.mlir)",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--out-channels", type=int, default=16, help="Output channels")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Padding")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--bias", action="store_true", help="Add bias")
    parser.add_argument("--output", type=str, default="conv_model.onnx", help="Output file")

    args = parser.parse_args()

    if args.two_layer:
        # For two-layer model, use defaults matching demo_two_layer_conv.mlir
        # unless explicitly overridden
        height = args.height if args.height != 8 else 224
        width = args.width if args.width != 8 else 224
        out_channels = args.out_channels if args.out_channels != 16 else 64

        create_two_layer_conv_model(
            batch_size=args.batch,
            in_channels=args.in_channels,
            height=height,
            width=width,
            mid_channels=64,  # Fixed to match demo
            out_channels=out_channels,
            kernel_size=args.kernel,
            padding=args.padding,
            output_path=args.output,
        )
    else:
        create_conv_model(
            batch_size=args.batch,
            in_channels=args.in_channels,
            height=args.height,
            width=args.width,
            out_channels=args.out_channels,
            kernel_size=args.kernel,
            padding=args.padding,
            stride=args.stride,
            has_bias=args.bias,
            output_path=args.output,
        )
