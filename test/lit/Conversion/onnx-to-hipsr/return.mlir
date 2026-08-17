// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// onnx.Return becomes func.return. Rejected forms live in return-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --convert-onnx-to-hipsr --split-input-file %s | FileCheck %s

// The shape an ONNX importer produces, with the return carrying the graph's
// result.
// CHECK-LABEL: func.func @imported_graph(
// CHECK-SAME:    %{{.*}}: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x4096xf16, #hipsr.mem<device>>) -> tensor<?x4096xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    return %[[INPUT]] : tensor<?x4096xf16, #hipsr.mem<device>>
func.func @imported_graph(%ctx: !hipsr.context, %input: tensor<?x4096xf16>)
    -> tensor<?x4096xf16> {
  "onnx.Return"(%input) : (tensor<?x4096xf16>) -> ()
}

// -----

// A graph with no results returns nothing.
// CHECK-LABEL: func.func @return_no_operands(
// CHECK-NEXT:    return
func.func @return_no_operands(%ctx: !hipsr.context) {
  "onnx.Return"() : () -> ()
}
