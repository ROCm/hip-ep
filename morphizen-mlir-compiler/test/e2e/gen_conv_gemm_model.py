#!/usr/bin/env python3
#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generate a combined Conv + Gemm ONNX model for testing."""

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_conv_gemm_model(
    # Conv parameters
    batch_size=1,
    in_channels=3,
    height=8,
    width=8,
    out_channels=16,
    kernel_size=3,
    padding=1,
    stride=1,
    # Gemm parameters
    gemm_n=32,
    output_path="conv_gemm_model.onnx",
):
    """
    Create a model with Conv followed by Gemm.

    Pipeline:
    Input [batch, in_channels, height, width]
      -> Conv -> [batch, out_channels, out_h, out_w]
      -> Flatten -> [batch, out_channels * out_h * out_w]
      -> Gemm -> [batch, gemm_n]
    """

    # Input tensor
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch_size, in_channels, height, width])

    # Conv weight (initializer)
    conv_W_shape = [out_channels, in_channels, kernel_size, kernel_size]
    conv_W_data = np.random.randn(*conv_W_shape).astype(np.float32) * 0.1
    conv_W = helper.make_tensor("conv_W", TensorProto.FLOAT, conv_W_shape, conv_W_data.flatten())

    # Conv bias (optional)
    conv_B_data = np.random.randn(out_channels).astype(np.float32) * 0.01
    conv_B = helper.make_tensor("conv_B", TensorProto.FLOAT, [out_channels], conv_B_data.flatten())

    # Compute conv output shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "conv_W", "conv_B"],
        outputs=["conv_out"],
        name="conv",
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[stride, stride],
        dilations=[1, 1],
    )

    # Flatten node (reshape conv output for gemm)
    flatten_node = helper.make_node("Flatten", inputs=["conv_out"], outputs=["flatten_out"], name="flatten", axis=1)

    # Gemm weight (initializer)
    gemm_k = out_channels * out_height * out_width
    gemm_W_shape = [gemm_k, gemm_n]
    gemm_W_data = np.random.randn(*gemm_W_shape).astype(np.float32) * 0.1
    gemm_W = helper.make_tensor("gemm_W", TensorProto.FLOAT, gemm_W_shape, gemm_W_data.flatten())

    # Gemm bias (optional)
    gemm_B_data = np.random.randn(gemm_n).astype(np.float32) * 0.01
    gemm_B = helper.make_tensor("gemm_B", TensorProto.FLOAT, [gemm_n], gemm_B_data.flatten())

    # Gemm node
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["flatten_out", "gemm_W", "gemm_B"],
        outputs=["Y"],
        name="gemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=0,
    )

    # Output tensor
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch_size, gemm_n])

    # Graph
    graph = helper.make_graph(
        [conv_node, flatten_node, gemm_node],
        "conv_gemm_graph",
        [X],
        [Y],
        [conv_W, conv_B, gemm_W, gemm_B],
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # Add metadata
    model.doc_string = "Combined Conv + Gemm model for ROCm testing"

    # Save
    onnx.save(model, output_path)
    print(f"Saved model to {output_path}")
    print("\nModel Pipeline:")
    print(f"  Input X: [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  -> Conv (kernel={kernel_size}x{kernel_size}, pad={padding}, stride={stride})")
    print(f"  -> Conv output: [{batch_size}, {out_channels}, {out_height}, {out_width}]")
    print("  -> Flatten")
    print(f"  -> Flatten output: [{batch_size}, {gemm_k}]")
    print(f"  -> Gemm (K={gemm_k}, N={gemm_n})")
    print(f"  -> Output Y: [{batch_size}, {gemm_n}]")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Conv + Gemm ONNX model")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--out-channels", type=int, default=16, help="Conv output channels")
    parser.add_argument("--kernel", type=int, default=3, help="Conv kernel size")
    parser.add_argument("--padding", type=int, default=1, help="Conv padding")
    parser.add_argument("--stride", type=int, default=1, help="Conv stride")
    parser.add_argument("--gemm-n", type=int, default=32, help="Gemm output dimension N")
    parser.add_argument("--output", type=str, default="conv_gemm_model.onnx", help="Output file")

    args = parser.parse_args()

    create_conv_gemm_model(
        batch_size=args.batch,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        out_channels=args.out_channels,
        kernel_size=args.kernel,
        padding=args.padding,
        stride=args.stride,
        gemm_n=args.gemm_n,
        output_path=args.output,
    )
