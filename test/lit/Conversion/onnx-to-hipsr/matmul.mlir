// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// REQUIRES: hipsr
//
//===----------------------------------------------------------------------===//
// Converts onnx.MatMul to hipsr.matmul (placeholder init, empty shape region).
// Two cases: a plain 2-D matmul and a 1-D-operand matmul (the rank-reducing
// ONNX case).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// 2-D x 2-D (dynamic M).
// CHECK-LABEL: func.func @matmul_2d(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x1024xf16>
// CHECK-NEXT:    hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x1024xf16>) : tensor<?x1024xf16>
// CHECK-NOT:     shape_region
func.func @matmul_2d(%ctx: !hipsr.context, %a: tensor<?x4096xf16>,
                     %b: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// 1-D B: (M,K) @ (K) -> (M), a rank-reducing result mirrored as-is.
// CHECK-LABEL: func.func @matmul_1d_rhs(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096xf16>) -> tensor<?xf16> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?xf16>
// CHECK-NEXT:    hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?xf16>) : tensor<?xf16>
// CHECK-NOT:     shape_region
func.func @matmul_1d_rhs(%ctx: !hipsr.context, %a: tensor<?x4096xf16>,
                         %b: tensor<4096xf16>) -> tensor<?xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096xf16>)
      -> tensor<?xf16>
  return %0 : tensor<?xf16>
}
