// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --onnx-to-hip-pipeline %s | FileCheck %s
// RUN: hip-mlir-opt --hipdnn-pipeline %s | FileCheck %s --check-prefix=LLVM

// An exact [1,0,D] Slice is a valid borrowed loop seed. Body allocations use
// the parent invocation's carrier bank with the exact growing descriptor, and
// the graph output is copied from the final descriptor before frame destroy.
// CHECK-LABEL: func.func @main_graph
// CHECK: memref.view
// CHECK-SAME: memref<1x0x3xf32>
// CHECK: memref.cast
// CHECK-SAME: memref<1x0x3xf32> to memref<1x?x3xf32>
// CHECK: %[[LOOP:.*]]:2 = hip.loop
// CHECK-SAME: -> (memref<1x?x3xf32>, !hip.loop_frame)
// CHECK: hip.alloc_output
// CHECK: hip.copy_output
// CHECK: hip.loop_frame_destroy
// CHECK-LABEL: func.func private @main_graph_loop_body_n0
// CHECK-SAME: %[[FRAME:[^ ,]+]]: !hip.loop_frame
// CHECK: hip.loop_alloc(%[[FRAME]],
// CHECK-SAME: memref<1x?x3xf32>
// LLVM-LABEL: llvm.func private @main_graph(
// LLVM: llvm.call @hipdnn_ep_run_counted_loop
// LLVM: llvm.call @hipdnn_ep_alloc_output
// LLVM: llvm.call @hipdnn_ep_loop_frame_destroy
module {
  func.func @main_graph(%row: tensor<1x1x3xf32> {onnx.name = "row"})
      -> (tensor<1x?x3xf32> {onnx.name = "output"}) {
    %starts = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %ends = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %axes = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %steps = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %seed = "onnx.Slice"(%row, %starts, %ends, %axes, %steps)
        : (tensor<1x1x3xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<1x0x3xf32>
    %m = "onnx.Constant"() {value = dense<2> : tensor<i64>}
        : () -> tensor<i64>
    %cond = "onnx.Constant"() {value = dense<true> : tensor<i1>}
        : () -> tensor<i1>
    %result = "onnx.Loop"(%m, %cond, %seed) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>,
         %acc: tensor<1x?x3xf32>):
      %next = "onnx.Concat"(%acc, %row) {axis = 1 : si64}
          : (tensor<1x?x3xf32>, tensor<1x1x3xf32>)
            -> tensor<1x?x3xf32>
      "onnx.Yield"(%cond_in, %next)
          : (tensor<i1>, tensor<1x?x3xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<1x0x3xf32>)
         -> tensor<1x?x3xf32>
    "onnx.Return"(%result) : (tensor<1x?x3xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
