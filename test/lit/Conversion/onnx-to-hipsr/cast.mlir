// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.Cast to hipsr.cast, leaving the shape region empty (a later
// pass populates it).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// DPS init is a placeholder, shape region left empty.
// CHECK-LABEL: func.func @cast(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[IN:.*]]: tensor<?x8xf32>) -> tensor<?x8xf16> {
// CHECK:     %[[INIT:.+]] = hipsr.placeholder : tensor<?x8xf16>
// CHECK:     hipsr.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NOT: shape_region
func.func @cast(%ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  %0 = "onnx.Cast"(%input) : (tensor<?x8xf32>) -> tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}
