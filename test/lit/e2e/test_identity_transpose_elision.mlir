// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// test_identity_transpose_elision.mlir
//
// Phase 4 kernel-launch count regression for the slot-buffer-coalesce
// design (docs/design/slot-buffer-coalesce.md).
//
// An ONNX graph carrying TWO identity `Transpose` ops (perm = [0, 1])
// in series. Without the IdentityPropagatorRebindPass each transpose
// emits a `llvm.call @wrap_transpose`. After the pass both transpose
// ops are erased and the downstream consumer reads directly from the
// upstream input -- so the only `wrap_transpose` call in the emitted
// LLVM IR is the count from any non-identity transpose. Here there
// are none, so we assert there is NO call.

// RUN: hip-mlir-opt %s --hipdnn-pipeline 2>&1 | FileCheck %s

// CHECK: module attributes {
// CHECK-NOT: llvm.call @wrap_transpose
// CHECK: return
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"})
      -> (tensor<3x4xf32> {onnx.name = "output"}) {
    %t1 = "onnx.Transpose"(%arg0) {onnx_node_name = "id_t1", perm = [0, 1]}
        : (tensor<3x4xf32>) -> tensor<3x4xf32>
    %t2 = "onnx.Transpose"(%t1) {onnx_node_name = "id_t2", perm = [0, 1]}
        : (tensor<3x4xf32>) -> tensor<3x4xf32>
    "onnx.Return"(%t2) : (tensor<3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
