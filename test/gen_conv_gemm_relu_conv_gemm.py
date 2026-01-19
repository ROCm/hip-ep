#!/usr/bin/env python3
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""Generate a complex Conv -> Gemm -> ReLU -> Conv -> Gemm ONNX model for testing."""

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_conv_gemm_relu_conv_gemm_model(
    batch_size=1,
    in_channels=3,
    height=8,
    width=8,
    output_path="conv_gemm_relu_conv_gemm_model.onnx"
):
    """
    Create a complex model with: Conv -> Gemm -> ReLU -> Conv -> Gemm
    
    Pipeline:
    Input [1, 3, 8, 8]
      -> Conv1 -> [1, 16, 8, 8]
      -> Flatten -> [1, 1024]
      -> Gemm1 -> [1, 64]
      -> ReLU -> [1, 64]
      -> Reshape -> [1, 8, 8, 1] (prepare for conv2)
      -> Conv2 -> [1, 16, 8, 8]
      -> Flatten -> [1, 1024]
      -> Gemm2 -> [1, 32]
    """
    
    # Input tensor
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, 
                                       [batch_size, in_channels, height, width])
    
    # ========== Conv1 ==========
    conv1_out_channels = 16
    conv1_kernel = 3
    conv1_pad = 1
    
    conv1_W_shape = [conv1_out_channels, in_channels, conv1_kernel, conv1_kernel]
    conv1_W_data = np.random.randn(*conv1_W_shape).astype(np.float32) * 0.1
    conv1_W = helper.make_tensor('conv1_W', TensorProto.FLOAT, conv1_W_shape, conv1_W_data.flatten())
    
    conv1_B_data = np.random.randn(conv1_out_channels).astype(np.float32) * 0.01
    conv1_B = helper.make_tensor('conv1_B', TensorProto.FLOAT, [conv1_out_channels], conv1_B_data.flatten())
    
    conv1_out_h = (height + 2 * conv1_pad - conv1_kernel) + 1
    conv1_out_w = (width + 2 * conv1_pad - conv1_kernel) + 1
    
    conv1_node = helper.make_node(
        'Conv',
        inputs=['X', 'conv1_W', 'conv1_B'],
        outputs=['conv1_out'],
        name='conv1',
        kernel_shape=[conv1_kernel, conv1_kernel],
        pads=[conv1_pad, conv1_pad, conv1_pad, conv1_pad],
        strides=[1, 1],
    )
    
    # ========== Flatten1 ==========
    flatten1_node = helper.make_node(
        'Flatten',
        inputs=['conv1_out'],
        outputs=['flatten1_out'],
        name='flatten1',
        axis=1
    )
    
    # ========== Gemm1 ==========
    gemm1_k = conv1_out_channels * conv1_out_h * conv1_out_w
    gemm1_n = 64
    
    gemm1_W_shape = [gemm1_k, gemm1_n]
    gemm1_W_data = np.random.randn(*gemm1_W_shape).astype(np.float32) * 0.1
    gemm1_W = helper.make_tensor('gemm1_W', TensorProto.FLOAT, gemm1_W_shape, gemm1_W_data.flatten())
    
    gemm1_B_data = np.random.randn(gemm1_n).astype(np.float32) * 0.01
    gemm1_B = helper.make_tensor('gemm1_B', TensorProto.FLOAT, [gemm1_n], gemm1_B_data.flatten())
    
    gemm1_node = helper.make_node(
        'Gemm',
        inputs=['flatten1_out', 'gemm1_W', 'gemm1_B'],
        outputs=['gemm1_out'],
        name='gemm1',
        alpha=1.0,
        beta=1.0,
    )
    
    # ========== ReLU ==========
    relu_node = helper.make_node(
        'Relu',
        inputs=['gemm1_out'],
        outputs=['relu_out'],
        name='relu'
    )
    
    # ========== Reshape (prepare for Conv2) ==========
    # Reshape [1, 64] -> [1, 8, 8, 1] for next conv
    reshape_shape = np.array([batch_size, 8, 8, 1], dtype=np.int64)
    reshape_shape_tensor = helper.make_tensor('reshape_shape', TensorProto.INT64, 
                                               [4], reshape_shape.flatten())
    
    reshape_node = helper.make_node(
        'Reshape',
        inputs=['relu_out', 'reshape_shape'],
        outputs=['reshape_out'],
        name='reshape'
    )
    
    # ========== Conv2 ==========
    conv2_in_channels = 1
    conv2_out_channels = 16
    conv2_kernel = 3
    conv2_pad = 1
    
    conv2_W_shape = [conv2_out_channels, conv2_in_channels, conv2_kernel, conv2_kernel]
    conv2_W_data = np.random.randn(*conv2_W_shape).astype(np.float32) * 0.1
    conv2_W = helper.make_tensor('conv2_W', TensorProto.FLOAT, conv2_W_shape, conv2_W_data.flatten())
    
    conv2_B_data = np.random.randn(conv2_out_channels).astype(np.float32) * 0.01
    conv2_B = helper.make_tensor('conv2_B', TensorProto.FLOAT, [conv2_out_channels], conv2_B_data.flatten())
    
    conv2_node = helper.make_node(
        'Conv',
        inputs=['reshape_out', 'conv2_W', 'conv2_B'],
        outputs=['conv2_out'],
        name='conv2',
        kernel_shape=[conv2_kernel, conv2_kernel],
        pads=[conv2_pad, conv2_pad, conv2_pad, conv2_pad],
        strides=[1, 1],
    )
    
    # ========== Flatten2 ==========
    flatten2_node = helper.make_node(
        'Flatten',
        inputs=['conv2_out'],
        outputs=['flatten2_out'],
        name='flatten2',
        axis=1
    )
    
    # ========== Gemm2 ==========
    gemm2_k = conv2_out_channels * 8 * 8
    gemm2_n = 32
    
    gemm2_W_shape = [gemm2_k, gemm2_n]
    gemm2_W_data = np.random.randn(*gemm2_W_shape).astype(np.float32) * 0.1
    gemm2_W = helper.make_tensor('gemm2_W', TensorProto.FLOAT, gemm2_W_shape, gemm2_W_data.flatten())
    
    gemm2_B_data = np.random.randn(gemm2_n).astype(np.float32) * 0.01
    gemm2_B = helper.make_tensor('gemm2_B', TensorProto.FLOAT, [gemm2_n], gemm2_B_data.flatten())
    
    gemm2_node = helper.make_node(
        'Gemm',
        inputs=['flatten2_out', 'gemm2_W', 'gemm2_B'],
        outputs=['Y'],
        name='gemm2',
        alpha=1.0,
        beta=1.0,
    )
    
    # Output tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [batch_size, gemm2_n])
    
    # Graph
    graph = helper.make_graph(
        [conv1_node, flatten1_node, gemm1_node, relu_node, reshape_node, 
         conv2_node, flatten2_node, gemm2_node],
        'conv_gemm_relu_conv_gemm_graph',
        [X],
        [Y],
        [conv1_W, conv1_B, gemm1_W, gemm1_B, reshape_shape_tensor, 
         conv2_W, conv2_B, gemm2_W, gemm2_B]
    )
    
    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    # Add metadata
    model.doc_string = "Complex Conv->Gemm->ReLU->Conv->Gemm model for ROCm testing"
    
    # Save
    onnx.save(model, output_path)
    print(f"Saved model to {output_path}")
    print(f"\nModel Pipeline:")
    print(f"  Input X: [{batch_size}, {in_channels}, {height}, {width}]")
    print(f"  -> Conv1 (3x3, pad=1)")
    print(f"  -> Conv1 output: [{batch_size}, {conv1_out_channels}, {conv1_out_h}, {conv1_out_w}]")
    print(f"  -> Flatten1")
    print(f"  -> Flatten1 output: [{batch_size}, {gemm1_k}]")
    print(f"  -> Gemm1 (K={gemm1_k}, N={gemm1_n})")
    print(f"  -> Gemm1 output: [{batch_size}, {gemm1_n}]")
    print(f"  -> ReLU")
    print(f"  -> Reshape to [{batch_size}, 8, 8, 1]")
    print(f"  -> Conv2 (3x3, pad=1)")
    print(f"  -> Conv2 output: [{batch_size}, {conv2_out_channels}, 8, 8]")
    print(f"  -> Flatten2")
    print(f"  -> Flatten2 output: [{batch_size}, {gemm2_k}]")
    print(f"  -> Gemm2 (K={gemm2_k}, N={gemm2_n})")
    print(f"  -> Output Y: [{batch_size}, {gemm2_n}]")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Conv->Gemm->ReLU->Conv->Gemm ONNX model")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels")
    parser.add_argument("--height", type=int, default=8, help="Input height")
    parser.add_argument("--width", type=int, default=8, help="Input width")
    parser.add_argument("--output", type=str, default="conv_gemm_relu_conv_gemm_model.onnx", 
                        help="Output file")
    
    args = parser.parse_args()
    
    create_conv_gemm_relu_conv_gemm_model(
        batch_size=args.batch,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        output_path=args.output
    )
