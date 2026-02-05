#!/usr/bin/env python3
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""Generate a Conv (no bias) -> Conv (with bias) -> Reshape -> Gemm -> Softmax ONNX model for testing."""

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_conv_conv_reshape_gemm_softmax_model(
    batch_size=1,
    in_channels=3,
    height=8,
    width=8,
    output_path="sample.onnx"
):
    """
    Create a model with: Conv (no bias) -> Conv (with bias) -> Reshape -> Gemm -> Softmax
    
    Pipeline:
    Input [1, 3, 8, 8]
      -> Conv1 (no bias) -> [1, 16, 8, 8]
      -> Conv2 (with bias) -> [1, 32, 8, 8]
      -> Reshape -> [1, 2048]
      -> Gemm -> [1, 64]
      -> Softmax -> [1, 64]
    """
    
    # Input tensor
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, 
                                       [batch_size, in_channels, height, width])
    
    # ========== Conv1 (no bias) ==========
    conv1_out_channels = 16
    conv1_kernel = 3
    conv1_pad = 1
    
    conv1_W_shape = [conv1_out_channels, in_channels, conv1_kernel, conv1_kernel]
    conv1_W_data = np.random.randn(*conv1_W_shape).astype(np.float32) * 0.1
    conv1_W = helper.make_tensor('conv1_W', TensorProto.FLOAT, conv1_W_shape, conv1_W_data.flatten())
    
    conv1_out_h = (height + 2 * conv1_pad - conv1_kernel) + 1
    conv1_out_w = (width + 2 * conv1_pad - conv1_kernel) + 1
    
    # Conv1 without bias (only 2 inputs: X and W)
    conv1_node = helper.make_node(
        'Conv',
        inputs=['X', 'conv1_W'],
        outputs=['conv1_out'],
        name='conv1',
        kernel_shape=[conv1_kernel, conv1_kernel],
        pads=[conv1_pad, conv1_pad, conv1_pad, conv1_pad],
        strides=[1, 1],
    )
    
    # ========== Conv2 (with bias) ==========
    conv2_in_channels = conv1_out_channels
    conv2_out_channels = 32
    conv2_kernel = 3
    conv2_pad = 1
    
    conv2_W_shape = [conv2_out_channels, conv2_in_channels, conv2_kernel, conv2_kernel]
    conv2_W_data = np.random.randn(*conv2_W_shape).astype(np.float32) * 0.1
    conv2_W = helper.make_tensor('conv2_W', TensorProto.FLOAT, conv2_W_shape, conv2_W_data.flatten())
    
    conv2_B_data = np.random.randn(conv2_out_channels).astype(np.float32) * 0.01
    conv2_B = helper.make_tensor('conv2_B', TensorProto.FLOAT, [conv2_out_channels], conv2_B_data.flatten())
    
    conv2_out_h = (conv1_out_h + 2 * conv2_pad - conv2_kernel) + 1
    conv2_out_w = (conv1_out_w + 2 * conv2_pad - conv2_kernel) + 1
    
    # Conv2 with bias (3 inputs: X, W, B)
    conv2_node = helper.make_node(
        'Conv',
        inputs=['conv1_out', 'conv2_W', 'conv2_B'],
        outputs=['conv2_out'],
        name='conv2',
        kernel_shape=[conv2_kernel, conv2_kernel],
        pads=[conv2_pad, conv2_pad, conv2_pad, conv2_pad],
        strides=[1, 1],
    )
    
    # ========== Reshape ==========
    # Reshape [1, 32, 8, 8] -> [1, 2048] for Gemm
    flatten_size = conv2_out_channels * conv2_out_h * conv2_out_w
    reshape_shape = np.array([batch_size, flatten_size], dtype=np.int64)
    reshape_shape_tensor = helper.make_tensor('reshape_shape', TensorProto.INT64, 
                                               [2], reshape_shape.flatten())
    
    reshape_node = helper.make_node(
        'Reshape',
        inputs=['conv2_out', 'reshape_shape'],
        outputs=['reshape_out'],
        name='reshape'
    )
    
    # ========== Gemm ==========
    gemm_k = flatten_size
    gemm_n = 64
    
    gemm_W_shape = [gemm_k, gemm_n]
    gemm_W_data = np.random.randn(*gemm_W_shape).astype(np.float32) * 0.1
    gemm_W = helper.make_tensor('gemm_W', TensorProto.FLOAT, gemm_W_shape, gemm_W_data.flatten())
    
    gemm_B_data = np.random.randn(gemm_n).astype(np.float32) * 0.01
    gemm_B = helper.make_tensor('gemm_B', TensorProto.FLOAT, [gemm_n], gemm_B_data.flatten())
    
    gemm_node = helper.make_node(
        'Gemm',
        inputs=['reshape_out', 'gemm_W', 'gemm_B'],
        outputs=['gemm_out'],
        name='gemm',
        alpha=1.0,
        beta=1.0,
    )
    
    # ========== Softmax ==========
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['gemm_out'],
        outputs=['Y'],
        name='softmax',
        axis=1
    )
    
    # Output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [batch_size, gemm_n])
    
    # Graph
    graph = helper.make_graph(
        [conv1_node, conv2_node, reshape_node, gemm_node, softmax_node],
        'conv_conv_reshape_gemm_softmax_graph',
        [X],
        [Y],
        [conv1_W, conv2_W, conv2_B, reshape_shape_tensor, gemm_W, gemm_B]
    )
    
    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    # Add metadata
    model.doc_string = "Conv(no bias)->Conv(with bias)->Reshape->Gemm->Softmax model for ROCm testing"
    
    # Save
    onnx.save(model, output_path)
    print(f"Saved model to {output_path}")
    print(f"\nModel Pipeline:")
    print(f"  Input X: [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  -> Conv1 (3x3, pad=1, NO bias)")
    print(f"  -> Conv1 output: [{batch_size}, {conv1_out_channels}, {conv1_out_h}, {conv1_out_w}]")
    print(f"  -> Conv2 (3x3, pad=1, WITH bias)")
    print(f"  -> Conv2 output: [{batch_size}, {conv2_out_channels}, {conv2_out_h}, {conv2_out_w}]")
    print(f"  -> Reshape")
    print(f"  -> Reshape output: [{batch_size}, {flatten_size}]")
    print(f"  -> Gemm (K={gemm_k}, N={gemm_n})")
    print(f"  -> Gemm output: [{batch_size}, {gemm_n}]")
    print(f"  -> Softmax (axis=1)")
    print(f"  -> Output Y: [{batch_size}, {gemm_n}]")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Conv(no bias)->Conv(with bias)->Reshape->Gemm->Softmax ONNX model")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--output", type=str, default="sample.onnx", 
                        help="Output file")
    
    args = parser.parse_args()
    
    create_conv_conv_reshape_gemm_softmax_model(
        batch_size=args.batch,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        output_path=args.output
    )
