// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// bc,d->bcd has no contracted index, so it is a broadcast multiply rather
// than a GEMM. Both operands are expanded to the output rank first.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Einsum
module {
  func.func @main_graph(
      %arg0: tensor<1x3xf16> {onnx.name = "A"},
      %arg1: tensor<10xf16> {onnx.name = "B"})
      -> (tensor<1x3x10xf16> {onnx.name = "Y"}) {
    %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "bc,d->bcd",
         onnx_node_name = "einsum_outer"}
        : (tensor<1x3xf16>, tensor<10xf16>) -> tensor<1x3x10xf16>
    "onnx.Return"(%0) : (tensor<1x3x10xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
