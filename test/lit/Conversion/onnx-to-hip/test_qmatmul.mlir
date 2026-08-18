// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// EXAMPLE: QDQ MatMul Fusion Pattern
//
// This shows the INTENDED fusion pattern (currently stubbed out in code).
// Input:  QuantizeLinear -> MatMul -> DequantizeLinear
// Output: hip.qmatmul (single fused operation)
// ============================================================================

// Input MLIR (before fusion):
// =============================
//
// module {
//   func.func @qdq_matmul(%input: tensor<4x128xf32>, %weight: tensor<128x256xf32>) -> tensor<4x256xf32> {
//     // Quantization parameters
//     %lhs_scale = onnx.Constant dense<0.1> : tensor<f32>
//     %lhs_zp = onnx.Constant dense<0> : tensor<i8>
//     %output_scale = onnx.Constant dense<0.2> : tensor<f32>
//     %output_zp = onnx.Constant dense<0> : tensor<i8>
//
//     // QDQ pattern: Quantize -> MatMul -> Dequantize
//     %quantized = "onnx.QuantizeLinear"(%input, %lhs_scale, %lhs_zp)
//                  : (tensor<4x128xf32>, tensor<f32>, tensor<i8>) -> tensor<4x128xi8>
//
//     %matmul_out = "onnx.MatMul"(%quantized, %weight)
//                   : (tensor<4x128xi8>, tensor<128x256xf32>) -> tensor<4x256xi32>
//
//     %result = "onnx.DequantizeLinear"(%matmul_out, %output_scale, %output_zp)
//               : (tensor<4x256xi32>, tensor<f32>, tensor<i8>) -> tensor<4x256xf32>
//
//     return %result : tensor<4x256xf32>
//   }
// }

// Expected Output (after fusion):
// ================================
//
// module {
//   func.func @qdq_matmul(%arg0: !hip.context, %arg1: tensor<4x128xf32>, %arg2: tensor<128x256xf32>) -> tensor<4x256xf32> {
//     %0 = tensor.empty() : tensor<4x256xf32>
//
//     // Fused operation - single kernel call!
//     %1 = hip.qmatmul(%arg0)
//            ins(%arg1, %arg2 : tensor<4x128xf32>, tensor<128x256xf32>)
//            outs(%0 : tensor<4x256xf32>)
//            {lhs_scale = 1.000000e-01 : f32,   // extracted from onnx.Constant
//             rhs_scale = 1.000000e-00 : f32,   // default or extracted
//             output_scale = 2.000000e-01 : f32} // extracted from onnx.Constant
//            : tensor<4x256xf32>
//
//     return %1 : tensor<4x256xf32>
//   }
// }

// Benefits of fusion:
// ===================
// - Single kernel launch instead of 3 separate operations
// - No intermediate quantized tensors materialized
// - Scales are compile-time constants (attributes)
// - Enables specialized INT8 GEMM kernels

// Current test - verifies the TARGET operation works
// RUN: hip-mlir-opt %s | FileCheck %s

module {
  func.func @qmatmul_target(%ctx: !hip.context, %lhs: tensor<4x128xf32>, %rhs: tensor<128x256xf32>) -> tensor<4x256xf32> {
    %output = tensor.empty() : tensor<4x256xf32>
    %result = hip.qmatmul(%ctx) ins(%lhs, %rhs : tensor<4x128xf32>, tensor<128x256xf32>)
                           outs(%output : tensor<4x256xf32>)
                           {lhs_scale = 0.1 : f32, rhs_scale = 1.0 : f32, output_scale = 0.2 : f32}
                           : tensor<4x256xf32>
    return %result : tensor<4x256xf32>
  }
}

// CHECK-LABEL: func.func @qmatmul_target
// CHECK: hip.qmatmul
// CHECK-SAME: lhs_scale = 1.000000e-01
// CHECK-SAME: rhs_scale = 1.000000e+00
// CHECK-SAME: output_scale = 2.000000e-01
