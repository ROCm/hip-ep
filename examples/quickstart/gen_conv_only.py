#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Generate a single-Conv ONNX model for testing.

Trimmed-down variant of gen_conv_model.py -- only produces the plain Conv
graph (no bias / NHWC / MatMul+Sigmoid variants).  Useful for isolating the
Conv lowering path (hipDNN graph vs. hip.conv / MIOpen) without additional
ops in the graph.

Defaults reproduce the YOLOv8x backbone 3x3 stage-4 conv:
    input   : [1, 512, 40, 40]   (fp32)   ~  3.2 MB activation
    weights : [1024, 512, 3, 3]  (fp32)   ~ 18.9 MB  (4.72M params)
    output  : [1, 1024, 40, 40]  (fp32)   ~  6.4 MB activation
    pad=1, stride=1                       ~  7.5 GFLOPs per inference

This is one of the larger convs in real CV workloads, big enough to make
the hipDNN-graph vs MIOpen dispatch choice visible in perf measurements,
yet small enough to finish compilation and inference in seconds on Strix
Halo (gfx1151).  Override via CLI flags for smaller/larger variants.
"""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_conv_model(
    batch=1,
    in_channels=512,
    out_channels=1024,
    height=40,
    width=40,
    kernel_h=3,
    kernel_w=3,
    pad_h=1,
    pad_w=1,
    stride_h=1,
    stride_w=1,
    output_file="conv_yolov8x_backbone.onnx",
):
    """Create a simple Conv model."""

    # Input
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch, in_channels, height, width]
    )

    # Weight (as initializer with random values)
    W_shape = [out_channels, in_channels, kernel_h, kernel_w]
    W_data = np.random.randn(*W_shape).astype(np.float32)
    W = helper.make_tensor("W", TensorProto.FLOAT, W_shape, W_data.flatten().tolist())

    # Output shape
    out_h = (height + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * pad_w - kernel_w) // stride_w + 1
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, out_channels, out_h, out_w]
    )

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[kernel_h, kernel_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )

    # Graph
    graph = helper.make_graph(
        [conv_node],
        "conv_test",
        [X],  # inputs
        [Y],  # outputs
        [W],  # initializers
    )

    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_file)
    print(f"Saved model to {output_file}")
    print(f"  Input shape: [{batch}, {in_channels}, {height}, {width}]")
    print(f"  Weight shape: {W_shape}")
    print(f"  Output shape: [{batch}, {out_channels}, {out_h}, {out_w}]")

    # Also save weights for reference comparison
    np.save(output_file.replace(".onnx", "_weights.npy"), W_data)
    print(f"Saved weights to {output_file.replace('.onnx', '_weights.npy')}")

    return model, W_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a single-Conv ONNX model. "
        "Defaults = YOLOv8x backbone 3x3 stage-4 (512->1024, 40x40)."
    )
    parser.add_argument("--output", "-o", default="conv_yolov8x_backbone.onnx")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=512)
    parser.add_argument("--out-channels", type=int, default=1024)
    parser.add_argument("--height", type=int, default=40)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    create_conv_model(
        batch=args.batch,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel_h=args.kernel,
        kernel_w=args.kernel,
        pad_h=args.pad,
        pad_w=args.pad,
        stride_h=args.stride,
        stride_w=args.stride,
        output_file=args.output,
    )
