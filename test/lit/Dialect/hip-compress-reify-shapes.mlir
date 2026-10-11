// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// Compress's selected count is payload-dependent. Reification must preserve
// the logical extent chosen by the destination instead of reporting the
// condition capacity (3).
// CHECK-LABEL: func.func @selected_count_from_outs
// CHECK: return %[[SELECTED:.*]] : index
func.func @selected_count_from_outs(
    %ctx: !hip.context, %input: tensor<3x2xf32>,
    %condition: tensor<3xi1>, %selected: index) -> index {
  %init = tensor.empty(%selected) : tensor<?x2xf32>
  %result = hip.compress(%ctx)
      ins(%input, %condition : tensor<3x2xf32>, tensor<3xi1>)
      outs(%init : tensor<?x2xf32>)
      {axis = 0 : i64, flatten = false}
      : tensor<?x2xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %result, %c0 : tensor<?x2xf32>
  return %dim : index
}
