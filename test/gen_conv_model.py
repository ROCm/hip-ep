#!/usr/bin/env python3
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

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
    output_path="conv_model.onnx"
):
    """Create a simple Conv model."""
    
    # Input tensor
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, 
                                       [batch_size, in_channels, height, width])
    
    # Weight tensor (initializer)
    W_shape = [out_channels, in_channels, kernel_size, kernel_size]
    W_data = np.random.randn(*W_shape).astype(np.float32) * 0.1
    W = helper.make_tensor('W', TensorProto.FLOAT, W_shape, W_data.flatten())
    
    # Compute output shape (same padding assumed)
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    
    # Output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                       [batch_size, out_channels, out_height, out_width])
    
    # Build inputs and initializers
    inputs = ['X', 'W']
    initializers = [W]
    
    if has_bias:
        B_data = np.random.randn(out_channels).astype(np.float32) * 0.1
        B = helper.make_tensor('B', TensorProto.FLOAT, [out_channels], B_data.flatten())
        inputs.append('B')
        initializers.append(B)
    
    # Conv node
    conv_node = helper.make_node(
        'Conv',
        inputs=inputs,
        outputs=['Y'],
        name='conv',
        kernel_shape=[kernel_size, kernel_size],
        pads=[padding, padding, padding, padding],
        strides=[stride, stride],
        dilations=[1, 1],
    )
    
    # Graph
    graph = helper.make_graph(
        [conv_node],
        'conv_graph',
        [X],
        [Y],
        initializers
    )
    
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Conv ONNX model")
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
        output_path=args.output
    )
