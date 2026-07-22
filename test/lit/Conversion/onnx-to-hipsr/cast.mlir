// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.Cast to hipsr.cast. The shape region is left empty (zero
// blocks); a later dedicated pass fills in the shape computation, so this stage
// does not populate it.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// CHECK-LABEL: func.func @cast(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[IN:.*]]: tensor<?x8xf32>) -> tensor<?x8xf16> {
func.func @cast(%ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  // The DPS init is a hipsr.placeholder that mirrors the result type, so no
  // output-shape computation (tensor.empty / tensor.dim) is emitted here.
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<?x8xf16>
  // The !hipsr.context (function arg 0) is threaded onto the op, and the shape
  // region is left empty (zero blocks) here; a later pass populates it. The
  // optional region group prints nothing when the region is empty.
  // CHECK: hipsr.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: shape_region
  %0 = "onnx.Cast"(%input) : (tensor<?x8xf32>) -> tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}
