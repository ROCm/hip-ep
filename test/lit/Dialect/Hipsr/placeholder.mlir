// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for hipsr.placeholder:
//   - round-trips dynamic, static, and rank-0 tensor forms
//   - initializes a hipsr destination-style operation
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file %s | FileCheck %s

// -----
// Single result, dynamic dims.
// CHECK-LABEL: func.func @single_dynamic
// CHECK: %[[INIT:.*]] = hipsr.placeholder : tensor<?x?xf16>
// CHECK: hipsr.cast
// CHECK-SAME: outs(%[[INIT]] : tensor<?x?xf16>)
func.func @single_dynamic(%ctx: !hipsr.context,
                          %input: tensor<?x?xf32>) -> tensor<?x?xf16> {
  %init = hipsr.placeholder : tensor<?x?xf16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
  return %result : tensor<?x?xf16>
}

// -----
// Single result, static dims.
// CHECK-LABEL: func.func @single_static
// CHECK: %[[INIT:.*]] = hipsr.placeholder : tensor<4x8xi64>
// CHECK: hipsr.cast
// CHECK-SAME: outs(%[[INIT]] : tensor<4x8xi64>)
func.func @single_static(%ctx: !hipsr.context,
                         %input: tensor<4x8xf32>) -> tensor<4x8xi64> {
  %init = hipsr.placeholder : tensor<4x8xi64>
  %result = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xi64>) : tensor<4x8xi64>
  return %result : tensor<4x8xi64>
}

// -----
// Rank-0 (scalar) result.
// CHECK-LABEL: func.func @rank0
// CHECK: %[[INIT:.*]] = hipsr.placeholder : tensor<f16>
// CHECK: hipsr.cast
// CHECK-SAME: outs(%[[INIT]] : tensor<f16>)
func.func @rank0(%ctx: !hipsr.context,
                 %input: tensor<f32>) -> tensor<f16> {
  %init = hipsr.placeholder : tensor<f16>
  %result = hipsr.cast(%ctx) ins(%input : tensor<f32>)
      outs(%init : tensor<f16>) : tensor<f16>
  return %result : tensor<f16>
}
