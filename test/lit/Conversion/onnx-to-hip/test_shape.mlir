// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Shape is correctly lowered to arith.constant (static shapes) or
// tensor.from_elements (dynamic shapes). Shape is a zero-cost metadata
// operation that extracts tensor dimensions.
//
// This test validates:
// - Static shapes folded to arith.constant
// - Dynamic dimensions extracted via tensor.dim + index_cast
// - Partial shape extraction via start/end attributes
// - Negative start/end indices
// - Edge cases: rank-0 scalars, rank-1 vectors
// - Different input dtypes (Shape is dtype-agnostic)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
    return %arg0 : tensor<2x3x4xf32>
  }

  // CHECK-LABEL: func @shape_static_full_range
  func.func @shape_static_full_range(%arg1: tensor<2x3x4xf32>) -> tensor<3xi64> {
    // Full shape extraction: [2, 3, 4]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[2, 3, 4]> : tensor<3xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) : (tensor<2x3x4xf32>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }

  // CHECK-LABEL: func @shape_static_with_start
  func.func @shape_static_with_start(%arg1: tensor<2x3x4xf32>) -> tensor<2xi64> {
    // Partial shape extraction starting at index 1: [3, 4]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[3, 4]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) {start = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_static_with_start_and_end
  func.func @shape_static_with_start_and_end(%arg1: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
    // Partial shape extraction [start=1, end=3): [3, 4]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[3, 4]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) {start = 1 : si64, end = 3 : si64} : (tensor<2x3x4x5xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_static_negative_start
  func.func @shape_static_negative_start(%arg1: tensor<2x3x4xf32>) -> tensor<2xi64> {
    // Negative start index: start=-2 means last 2 dims: [3, 4]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[3, 4]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) {start = -2 : si64} : (tensor<2x3x4xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_static_negative_end
  func.func @shape_static_negative_end(%arg1: tensor<2x3x4xf32>) -> tensor<2xi64> {
    // Negative end index: end=-1 means all but last dim: [2, 3]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[2, 3]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) {end = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_static_rank0_scalar
  func.func @shape_static_rank0_scalar(%arg1: tensor<f32>) -> tensor<0xi64> {
    // Rank-0 scalar: shape is empty tensor
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<> : tensor<0xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) : (tensor<f32>) -> tensor<0xi64>
    return %0 : tensor<0xi64>
  }

  // CHECK-LABEL: func @shape_static_rank1_vector
  func.func @shape_static_rank1_vector(%arg1: tensor<5xf32>) -> tensor<1xi64> {
    // Rank-1 vector: shape is [5]
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<5> : tensor<1xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) : (tensor<5xf32>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }

  // CHECK-LABEL: func @shape_static_different_dtype_f16
  func.func @shape_static_different_dtype_f16(%arg1: tensor<2x3xf16>) -> tensor<2xi64> {
    // Shape is dtype-agnostic: output is always i64
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[2, 3]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) : (tensor<2x3xf16>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_static_different_dtype_i32
  func.func @shape_static_different_dtype_i32(%arg1: tensor<4x5xi32>) -> tensor<2xi64> {
    // Shape is dtype-agnostic: output is always i64
    // CHECK-NOT: onnx.Shape
    // CHECK: %[[CONST:.*]] = arith.constant dense<[4, 5]> : tensor<2xi64>
    // CHECK: return %[[CONST]]
    %0 = "onnx.Shape"(%arg1) : (tensor<4x5xi32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_dynamic_batch_dim
  func.func @shape_dynamic_batch_dim(%arg1: tensor<?x3x4xf32>) -> tensor<3xi64> {
    // Dynamic batch dimension: extract using tensor.dim, static dims are constants
    // CHECK-NOT: onnx.Shape
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x3x4xf32>
    // CHECK-DAG: %[[DIM0_I64:.*]] = arith.index_cast %[[DIM0]] : index to i64
    // CHECK-DAG: %[[DIM1:.*]] = arith.constant 3 : i64
    // CHECK-DAG: %[[DIM2:.*]] = arith.constant 4 : i64
    // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[DIM0_I64]], %[[DIM1]], %[[DIM2]] : tensor<3xi64>
    // CHECK: return %[[RESULT]]
    %0 = "onnx.Shape"(%arg1) : (tensor<?x3x4xf32>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }

  // CHECK-LABEL: func @shape_multiple_dynamic_dims
  func.func @shape_multiple_dynamic_dims(%arg1: tensor<?x?x4xf32>) -> tensor<3xi64> {
    // Multiple dynamic dimensions
    // CHECK-NOT: onnx.Shape
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x?x4xf32>
    // CHECK-DAG: %[[DIM0_I64:.*]] = arith.index_cast %[[DIM0]] : index to i64
    // CHECK-DAG: %[[DIM1:.*]] = tensor.dim %arg1, %[[C1]] : tensor<?x?x4xf32>
    // CHECK-DAG: %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
    // CHECK-DAG: %[[DIM2:.*]] = arith.constant 4 : i64
    // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[DIM0_I64]], %[[DIM1_I64]], %[[DIM2]] : tensor<3xi64>
    // CHECK: return %[[RESULT]]
    %0 = "onnx.Shape"(%arg1) : (tensor<?x?x4xf32>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }

  // CHECK-LABEL: func @shape_dynamic_with_start
  func.func @shape_dynamic_with_start(%arg1: tensor<?x3x?xf32>) -> tensor<2xi64> {
    // Partial shape on dynamic tensor: start=1, end=3 (default) -> [3, ?]
    // CHECK-NOT: onnx.Shape
    // CHECK-DAG: %[[DIM1:.*]] = arith.constant 3 : i64
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[DIM2:.*]] = tensor.dim %arg1, %[[C2]] : tensor<?x3x?xf32>
    // CHECK-DAG: %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
    // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[DIM1]], %[[DIM2_I64]] : tensor<2xi64>
    // CHECK: return %[[RESULT]]
    %0 = "onnx.Shape"(%arg1) {start = 1 : si64} : (tensor<?x3x?xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_dynamic_with_start_and_end
  func.func @shape_dynamic_with_start_and_end(%arg1: tensor<?x3x?x5xf32>) -> tensor<2xi64> {
    // Partial shape [start=1, end=3): [3, ?]
    // CHECK-NOT: onnx.Shape
    // CHECK-DAG: %[[DIM1:.*]] = arith.constant 3 : i64
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[DIM2:.*]] = tensor.dim %arg1, %[[C2]] : tensor<?x3x?x5xf32>
    // CHECK-DAG: %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
    // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[DIM1]], %[[DIM2_I64]] : tensor<2xi64>
    // CHECK: return %[[RESULT]]
    %0 = "onnx.Shape"(%arg1) {start = 1 : si64, end = 3 : si64} : (tensor<?x3x?x5xf32>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // CHECK-LABEL: func @shape_all_dynamic_dims
  func.func @shape_all_dynamic_dims(%arg1: tensor<?x?x?xf32>) -> tensor<3xi64> {
    // All dynamic dimensions
    // CHECK-NOT: onnx.Shape
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %arg1, %[[C0]] : tensor<?x?x?xf32>
    // CHECK-DAG: %[[DIM0_I64:.*]] = arith.index_cast %[[DIM0]] : index to i64
    // CHECK-DAG: %[[DIM1:.*]] = tensor.dim %arg1, %[[C1]] : tensor<?x?x?xf32>
    // CHECK-DAG: %[[DIM1_I64:.*]] = arith.index_cast %[[DIM1]] : index to i64
    // CHECK-DAG: %[[DIM2:.*]] = tensor.dim %arg1, %[[C2]] : tensor<?x?x?xf32>
    // CHECK-DAG: %[[DIM2_I64:.*]] = arith.index_cast %[[DIM2]] : index to i64
    // CHECK: %[[RESULT:.*]] = tensor.from_elements %[[DIM0_I64]], %[[DIM1_I64]], %[[DIM2_I64]] : tensor<3xi64>
    // CHECK: return %[[RESULT]]
    %0 = "onnx.Shape"(%arg1) : (tensor<?x?x?xf32>) -> tensor<3xi64>
    return %0 : tensor<3xi64>
  }
}
