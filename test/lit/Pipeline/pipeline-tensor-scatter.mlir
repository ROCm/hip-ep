// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify how bufferization and the output allocator place hip.tensor_scatter
// buffers. `past` stays the function argument it reads. A scatter result that
// is a function result writes into the caller's output buffer with no copy;
// a scatter result that is only an intermediate gets its own buffer, so
// later readers of `past` still see the unmodified argument.
// ============================================================================

// RUN: hip-mlir-opt %s --split-input-file --hipdnn-pipeline --mlir-print-ir-after=hip-use-output-allocator -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[PAST:[^:]*]]: memref<1x1x16x8xf16> {onnx.name = "past"}
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
// CHECK: %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64}
// CHECK: hip.tensor_scatter(%[[CTX]]) ins(%[[PAST]], %{{.*}} : {{.*}}) write_indices(%{{.*}}) outs(%[[OUT]] : memref<1x1x16x8xf16>)
// CHECK: return %[[OUT]]
module {
  func.func @main_graph(%past: tensor<1x1x16x8xf16> {onnx.name = "past"},
                        %update: tensor<1x1x?x8xf16> {onnx.name = "update"},
                        %wi: tensor<1xi64> {onnx.name = "write_indices"})
      -> (tensor<1x1x16x8xf16> {onnx.name = "present"}) {
    %0 = "onnx.TensorScatter"(%past, %update, %wi) {axis = 2 : si64}
        : (tensor<1x1x16x8xf16>, tensor<1x1x?x8xf16>, tensor<1xi64>)
          -> tensor<1x1x16x8xf16>
    "onnx.Return"(%0) : (tensor<1x1x16x8xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[PAST:[^:]*]]: memref<1x1x16x8xf16> {onnx.name = "past"}
// CHECK: %[[TMP:.*]] = memref.alloc()
// CHECK: hip.tensor_scatter(%[[CTX]]) ins(%[[PAST]], %{{.*}} : {{.*}}) write_indices(%{{.*}}) outs(%[[TMP]] : memref<1x1x16x8xf16>)
// CHECK: %[[OUT:.*]] = hip.alloc_output(%[[CTX]]) {out_idx = 0 : i64}
// CHECK: hip.add(%[[CTX]]) ins(%[[TMP]], %[[PAST]] : {{.*}}) outs(%[[OUT]] : {{.*}})
// CHECK: return %[[OUT]]
module {
  func.func @main_graph(%past: tensor<1x1x16x8xf16> {onnx.name = "past"},
                        %update: tensor<1x1x?x8xf16> {onnx.name = "update"},
                        %wi: tensor<1xi64> {onnx.name = "write_indices"})
      -> (tensor<1x1x16x8xf16> {onnx.name = "sum"}) {
    %0 = "onnx.TensorScatter"(%past, %update, %wi) {axis = 2 : si64}
        : (tensor<1x1x16x8xf16>, tensor<1x1x?x8xf16>, tensor<1xi64>)
          -> tensor<1x1x16x8xf16>
    %1 = "onnx.Add"(%0, %past)
        : (tensor<1x1x16x8xf16>, tensor<1x1x16x8xf16>) -> tensor<1x1x16x8xf16>
    "onnx.Return"(%1) : (tensor<1x1x16x8xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
