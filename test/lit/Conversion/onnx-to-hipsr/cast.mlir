// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.Cast to hipsr.cast, leaving the shape region empty (a later
// pass populates it).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The second placeholder follows the shape graph through the first
// placeholder, while the second cast follows the data graph.
// CHECK-LABEL: func.func @cast_chain(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[IN:.*]]: tensor<?x8xf32>) -> tensor<?x8xf32> {
// CHECK-NEXT: %[[FIRST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16>
// CHECK-NEXT: %[[FIRST:.+]] = hipsr.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32>) outs(%[[FIRST_INIT]] : tensor<?x8xf16>) : tensor<?x8xf16>
// CHECK-NEXT: %[[SECOND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[FIRST_INIT]] : tensor<?x8xf16>) {type = #hipsr.placeholder_type<normal>} : tensor<?x8xf32>
// CHECK-NEXT: %[[SECOND:.+]] = hipsr.cast(%[[CTX]]) ins(%[[FIRST]] : tensor<?x8xf16>) outs(%[[SECOND_INIT]] : tensor<?x8xf32>) : tensor<?x8xf32>
// CHECK-NOT: shape_region
func.func @cast_chain(
    %ctx: !hipsr.context, %input: tensor<?x8xf32>) -> tensor<?x8xf32> {
  %0 = "onnx.Cast"(%input) : (tensor<?x8xf32>) -> tensor<?x8xf16>
  %1 = "onnx.Cast"(%0) : (tensor<?x8xf16>) -> tensor<?x8xf32>
  return %1 : tensor<?x8xf32>
}
