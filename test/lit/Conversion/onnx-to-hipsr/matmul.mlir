// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.MatMul to hipsr.matmul. The conversion mirrors the result type
// into a hipsr.placeholder init, threads the !hipsr.context from function arg 0
// (added in the ONNX phase) onto the op, and leaves the shape region empty (a
// later pass fills it). It does not branch on rank/shape, so two cases cover
// it: a plain 2-D matmul and a 1-D-operand matmul (the rank-reducing ONNX case).
//
// Positive CHECKs were seeded with mlir/utils/generate-test-checks.py (piped
// from the -convert-onnx-to-hipsr output) and then refined: captures renamed,
// the matmul operands tied back to the function arguments, and terminator
// boilerplate dropped.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// 2-D x 2-D (dynamic M).
// CHECK-LABEL: func.func @matmul_2d(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?x1024xf16>
// CHECK:         hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x1024xf16>) : tensor<?x1024xf16>
// CHECK-NOT:     shape_region
func.func @matmul_2d(%ctx: !hipsr.context, %a: tensor<?x4096xf16>,
                     %b: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// 1-D B: (M,K) @ (K) -> (M), a rank-reducing result. The conversion mirrors the
// result type as-is, so the placeholder and matmul still carry no shape math.
// CHECK-LABEL: func.func @matmul_1d_rhs(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096xf16>) -> tensor<?xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?xf16>
// CHECK:         hipsr.matmul(%[[CTX]]) ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?xf16>) : tensor<?xf16>
// CHECK-NOT:     shape_region
func.func @matmul_1d_rhs(%ctx: !hipsr.context, %a: tensor<?x4096xf16>,
                         %b: tensor<4096xf16>) -> tensor<?xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096xf16>)
      -> tensor<?xf16>
  return %0 : tensor<?xf16>
}
