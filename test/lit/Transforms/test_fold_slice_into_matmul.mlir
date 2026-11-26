// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a static last-dim tensor.extract_slice of a hip.matmul / hip.gemm
// result is folded into a pre-sliced constant weight (and bias), and that the
// cases where the fold would be wrong are left alone.
//
// Test cases:
// 1. matmul_slice   - weight columns [1, 3) are kept
// 2. matmul_transb  - transB weight rows are sliced instead
// 3. gemm_bias      - weight and bias are both sliced
// 4. multi_use      - matmul result has a second consumer, slice survives
// 5. inner_dim      - slice of a non-N dim, slice survives
// 6. dynamic_weight - non-constant weight, slice survives
// ============================================================================

// RUN: hip-mlir-opt %s --hip-fold-slice-into-matmul --canonicalize | FileCheck %s

module {
  // 1. W = [[1,2,3,4],[5,6,7,8]], columns 1..2 -> [[2,3],[6,7]].
  func.func @matmul_slice(%ctx: !hip.context, %x: tensor<3x2xf32>)
      -> tensor<3x2xf32> {
    %w = hip.constant {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : tensor<2x4xf32>
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.matmul(%ctx) ins(%x, %w : tensor<3x2xf32>, tensor<2x4xf32>)
        outs(%e : tensor<3x4xf32>) : tensor<3x4xf32>
    %s = tensor.extract_slice %y[0, 1] [3, 2] [1, 1]
        : tensor<3x4xf32> to tensor<3x2xf32>
    return %s : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @matmul_slice
  // CHECK: %[[W:.*]] = hip.constant {value = dense<{{\[}}[2.000000e+00, 3.000000e+00], [6.000000e+00, 7.000000e+00]]> : tensor<2x2xf32>}
  // CHECK: %[[Y:.*]] = hip.matmul(%{{.*}}) ins(%{{.*}}, %[[W]] : tensor<3x2xf32>, tensor<2x2xf32>)
  // CHECK-NOT: tensor.extract_slice
  // CHECK: return %[[Y]]

  // 2. W^T rows are the output columns: rows 2..3 of the 4x2 weight.
  func.func @matmul_transb(%ctx: !hip.context, %x: tensor<3x2xf32>)
      -> tensor<3x2xf32> {
    %w = hip.constant {value = dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>} : tensor<4x2xf32>
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.matmul(%ctx) ins(%x, %w : tensor<3x2xf32>, tensor<4x2xf32>)
        outs(%e : tensor<3x4xf32>) {transB = 1 : i64} : tensor<3x4xf32>
    %s = tensor.extract_slice %y[0, 2] [3, 2] [1, 1]
        : tensor<3x4xf32> to tensor<3x2xf32>
    return %s : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @matmul_transb
  // CHECK: %[[W:.*]] = hip.constant {value = dense<{{\[}}[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf32>}
  // CHECK: hip.matmul(%{{.*}}) ins(%{{.*}}, %[[W]] : tensor<3x2xf32>, tensor<2x2xf32>)
  // CHECK-SAME: transB = 1
  // CHECK-NOT: tensor.extract_slice

  // 3. Gemm bias [10,20,30,40] -> [20,30] alongside the weight.
  func.func @gemm_bias(%ctx: !hip.context, %x: tensor<3x2xf32>)
      -> tensor<3x2xf32> {
    %w = hip.constant {value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : tensor<2x4xf32>
    %b = hip.constant {value = dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>} : tensor<4xf32>
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.gemm(%ctx) ins(%x, %w, %b : tensor<3x2xf32>, tensor<2x4xf32>, tensor<4xf32>)
        outs(%e : tensor<3x4xf32>) : tensor<3x4xf32>
    %s = tensor.extract_slice %y[0, 1] [3, 2] [1, 1]
        : tensor<3x4xf32> to tensor<3x2xf32>
    return %s : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @gemm_bias
  // CHECK-DAG: %[[W:.*]] = hip.constant {value = dense<{{\[}}[2.000000e+00, 3.000000e+00], [6.000000e+00, 7.000000e+00]]> : tensor<2x2xf32>}
  // CHECK-DAG: %[[B:.*]] = hip.constant {value = dense<[2.000000e+01, 3.000000e+01]> : tensor<2xf32>}
  // CHECK: hip.gemm(%{{.*}}) ins(%{{.*}}, %[[W]], %[[B]] : tensor<3x2xf32>, tensor<2x2xf32>, tensor<2xf32>)
  // CHECK-NOT: tensor.extract_slice

  // 4. The full product is still used elsewhere.
  func.func @multi_use(%ctx: !hip.context, %x: tensor<3x2xf32>)
      -> (tensor<3x2xf32>, tensor<3x4xf32>) {
    %w = hip.constant {value = dense<1.0> : tensor<2x4xf32>} : tensor<2x4xf32>
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.matmul(%ctx) ins(%x, %w : tensor<3x2xf32>, tensor<2x4xf32>)
        outs(%e : tensor<3x4xf32>) : tensor<3x4xf32>
    %s = tensor.extract_slice %y[0, 1] [3, 2] [1, 1]
        : tensor<3x4xf32> to tensor<3x2xf32>
    return %s, %y : tensor<3x2xf32>, tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @multi_use
  // CHECK: tensor<2x4xf32>
  // CHECK: tensor.extract_slice

  // 5. Slicing M rather than N must not touch the weight.
  func.func @inner_dim(%ctx: !hip.context, %x: tensor<3x2xf32>)
      -> tensor<2x4xf32> {
    %w = hip.constant {value = dense<1.0> : tensor<2x4xf32>} : tensor<2x4xf32>
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.matmul(%ctx) ins(%x, %w : tensor<3x2xf32>, tensor<2x4xf32>)
        outs(%e : tensor<3x4xf32>) : tensor<3x4xf32>
    %s = tensor.extract_slice %y[1, 0] [2, 4] [1, 1]
        : tensor<3x4xf32> to tensor<2x4xf32>
    return %s : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @inner_dim
  // CHECK: tensor.extract_slice

  // 6. Runtime weight: nothing to pre-slice.
  func.func @dynamic_weight(%ctx: !hip.context, %x: tensor<3x2xf32>,
                            %w: tensor<2x4xf32>) -> tensor<3x2xf32> {
    %e = tensor.empty() : tensor<3x4xf32>
    %y = hip.matmul(%ctx) ins(%x, %w : tensor<3x2xf32>, tensor<2x4xf32>)
        outs(%e : tensor<3x4xf32>) : tensor<3x4xf32>
    %s = tensor.extract_slice %y[0, 1] [3, 2] [1, 1]
        : tensor<3x4xf32> to tensor<3x2xf32>
    return %s : tensor<3x2xf32>
  }

  // CHECK-LABEL: func.func @dynamic_weight
  // CHECK: tensor.extract_slice
}
