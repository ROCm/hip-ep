// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --resolve-shaped-type-result-dims %s | FileCheck %s

// An endpoint difference larger than INT64_MAX cannot become an index
// attribute. Reification fails cleanly and lifts the destination dimension.
// CHECK-LABEL: func.func @count_overflow_falls_back
// CHECK-SAME: (%{{[^:]+}}: !hip.context, %[[INIT:[^:]+]]: tensor<?xi64>)
// CHECK: tensor.dim %[[INIT]]
func.func @count_overflow_falls_back(
    %ctx: !hip.context, %init: tensor<?xi64>) -> index {
  %start = arith.constant dense<-9223372036854775808> : tensor<i64>
  %limit = arith.constant dense<9223372036854775807> : tensor<i64>
  %delta = arith.constant dense<1> : tensor<i64>
  %result = hip.range(%ctx)
    ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
    outs(%init : tensor<?xi64>)
    : tensor<?xi64>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xi64>
  return %d0 : index
}

// -----

// CHECK-LABEL: func.func @count_i64_boundary
// CHECK: %[[MAX:.*]] = arith.constant 9223372036854775807 : index
// CHECK: return %[[MAX]] : index
func.func @count_i64_boundary(
    %ctx: !hip.context, %init: tensor<?xi64>) -> index {
  %start = arith.constant dense<0> : tensor<i64>
  %limit = arith.constant dense<9223372036854775807> : tensor<i64>
  %delta = arith.constant dense<1> : tensor<i64>
  %result = hip.range(%ctx)
    ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
    outs(%init : tensor<?xi64>)
    : tensor<?xi64>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xi64>
  return %d0 : index
}

// -----

// INT64_MIN is a valid negative delta. Its magnitude is computed in wide
// arithmetic instead of negating the signed C++ value.
// CHECK-LABEL: func.func @minimum_delta
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: return %[[ONE]] : index
func.func @minimum_delta(
    %ctx: !hip.context, %init: tensor<?xi64>) -> index {
  %start = arith.constant dense<9223372036854775807> : tensor<i64>
  %limit = arith.constant dense<-1> : tensor<i64>
  %delta = arith.constant dense<-9223372036854775808> : tensor<i64>
  %result = hip.range(%ctx)
    ins(%start, %limit, %delta : tensor<i64>, tensor<i64>, tensor<i64>)
    outs(%init : tensor<?xi64>)
    : tensor<?xi64>
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %result, %c0 : tensor<?xi64>
  return %d0 : index
}
