#!/usr/bin/env python3
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

"""Generate a simple Gemm ONNX model for testing."""

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_gemm_model(
    m=64,
    k=48,
    n=32,
    alpha=1.0,
    beta=0.0,
    trans_a=False,
    trans_b=False,
    has_bias=False,
    output_path="gemm_model.onnx"
):
    """Create a simple Gemm model: Y = alpha * A @ B + beta * C"""
    
    # Input tensor A
    A_shape = [m, k] if not trans_a else [k, m]
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, A_shape)
    
    # Weight tensor B (initializer)
    B_shape = [k, n] if not trans_b else [n, k]
    B_data = np.random.randn(*B_shape).astype(np.float32) * 0.1
    B = helper.make_tensor('B', TensorProto.FLOAT, B_shape, B_data.flatten())
    
    # Output tensor Y
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [m, n])
    
    # Build inputs and initializers
    inputs = ['A', 'B']
    initializers = [B]
    
    if has_bias:
        C_data = np.random.randn(n).astype(np.float32) * 0.1  # 1D bias broadcast
        C = helper.make_tensor('C', TensorProto.FLOAT, [n], C_data.flatten())
        inputs.append('C')
        initializers.append(C)
    
    # Gemm node
    gemm_node = helper.make_node(
        'Gemm',
        inputs=inputs,
        outputs=['Y'],
        name='gemm',
        alpha=alpha,
        beta=beta,
        transA=1 if trans_a else 0,
        transB=1 if trans_b else 0,
    )
    
    # Graph
    graph = helper.make_graph(
        [gemm_node],
        'gemm_graph',
        [A],
        [Y],
        initializers
    )
    
    # Model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    
    # Save
    onnx.save(model, output_path)
    print(f"Saved model to {output_path}")
    print(f"  Input A: {A_shape}")
    print(f"  Weight B: {B_shape}")
    print(f"  Output Y: [{m}, {n}]")
    print(f"  alpha={alpha}, beta={beta}")
    print(f"  Has bias: {has_bias}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Gemm ONNX model")
    parser.add_argument("--m", type=int, default=64, help="M dimension")
    parser.add_argument("--k", type=int, default=48, help="K dimension")
    parser.add_argument("--n", type=int, default=32, help="N dimension")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha scalar")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta scalar")
    parser.add_argument("--trans-a", action="store_true", help="Transpose A")
    parser.add_argument("--trans-b", action="store_true", help="Transpose B")
    parser.add_argument("--bias", action="store_true", help="Add bias (C)")
    parser.add_argument("--output", type=str, default="gemm_model.onnx", help="Output file")
    
    args = parser.parse_args()
    
    create_gemm_model(
        m=args.m,
        k=args.k,
        n=args.n,
        alpha=args.alpha,
        beta=args.beta,
        trans_a=args.trans_a,
        trans_b=args.trans_b,
        has_bias=args.bias,
        output_path=args.output
    )
