// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.Cast to hipsr.cast using Scheme-defined patterns.
// This test verifies that the Scheme implementation produces identical output
// to the C++ implementation (test/lit/Conversion/onnx-to-hipsr/cast.mlir).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled -allow-unregistered-dialect --conversion-in-scheme="module=passes/onnx-to-hipsr" | FileCheck %s

// The second placeholder follows the shape graph through the first
// placeholder, while the second cast follows the data graph.
// CHECK-LABEL: func.func @cast_chain(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[IN:.*]]: tensor<?x8xf32, #hipsr.mem<device>>) -> tensor<?x8xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[FIRST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[FIRST:.+]] = hipsr.cast(%[[CTX]]) ins(%[[IN]] : tensor<?x8xf32, #hipsr.mem<device>>) outs(%[[FIRST_INIT]] : tensor<?x8xf16, #hipsr.mem<device>>) : tensor<?x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[SECOND_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[FIRST_INIT]] : tensor<?x8xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x8xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[SECOND:.+]] = hipsr.cast(%[[CTX]]) ins(%[[FIRST]] : tensor<?x8xf16, #hipsr.mem<device>>) outs(%[[SECOND_INIT]] : tensor<?x8xf32, #hipsr.mem<device>>) : tensor<?x8xf32, #hipsr.mem<device>>
// CHECK-NOT: shape_region
func.func @cast_chain(%arg0: !hipsr.context, %arg1: tensor<?x8xf32, #hipsr.mem<device>>) -> tensor<?x8xf32, #hipsr.mem<device>> {
  %0 = "onnx.Cast"(%arg1) {to = f16} : (tensor<?x8xf32, #hipsr.mem<device>>) -> tensor<?x8xf16, #hipsr.mem<device>>
  %1 = "onnx.Cast"(%0) {to = f32} : (tensor<?x8xf16, #hipsr.mem<device>>) -> tensor<?x8xf32, #hipsr.mem<device>>
  "onnx.Return"(%1) : (tensor<?x8xf32, #hipsr.mem<device>>) -> ()
}
