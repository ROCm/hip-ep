// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --onnx-to-hip-pipeline --split-input-file %s | FileCheck %s
// RUN: hip-mlir-opt --hipdnn-pipeline --split-input-file %s | FileCheck %s --check-prefix=LLVM

// Full-pipeline nested loop: the inner loop must use the outer body's frame as
// parent ownership, and only the top-level frame is explicitly destroyed.
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: %[[OUTER:.*]]:2 = hip.loop
  // CHECK: hip.alloc_output
  // CHECK: hip.copy_output
  // CHECK: hip.loop_frame_destroy
  // CHECK-LABEL: func.func private @main_graph_loop_body_n1
  // CHECK-SAME: %[[PARENT:[^ ,]+]]: !hip.loop_frame
  // CHECK: hip.loop
  // CHECK-SAME: parent(%[[PARENT]])
  // CHECK-SAME: -> (memref<16xf32>, !hip.loop_frame)
  // CHECK: hip.loop_alloc(%[[PARENT]])
  // CHECK: hip.copy_output
  // LLVM-LABEL: llvm.func private @main_graph(
  // LLVM: llvm.call @hipdnn_ep_run_counted_loop
  // LLVM: llvm.call @hipdnn_ep_loop_frame_destroy
  func.func @main_graph(%seed: tensor<16xf32>) -> tensor<16xf32> {
    %outer_m = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %outer_c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %result = "onnx.Loop"(%outer_m, %outer_c, %seed) ({
    ^bb0(%oi: tensor<i64>, %oc: tensor<i1>, %outer: tensor<16xf32>):
      %inner_m = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
      %inner_c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
      %inner = "onnx.Loop"(%inner_m, %inner_c, %outer) ({
      ^bb1(%ii: tensor<i64>, %ic: tensor<i1>, %current: tensor<16xf32>):
        %next = "onnx.Add"(%current, %current) :
            (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
        "onnx.Yield"(%ic, %next) : (tensor<i1>, tensor<16xf32>) -> ()
      }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
      "onnx.Yield"(%oc, %inner) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %result : tensor<16xf32>
  }
}

// Dynamic condition takes the slow runtime path and exits after the body's
// first false condition publication.
// -----
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: hip.loop
  // CHECK-NOT: cond_is_passthrough
  // LLVM-LABEL: llvm.func private @main_graph(
  // LLVM: llvm.call @hipdnn_ep_run_loop
  func.func @main_graph(%seed: tensor<4xf32>) -> tensor<4xf32> {
    %m = "onnx.Constant"() {value = dense<8> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %result = "onnx.Loop"(%m, %c, %seed) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>, %current: tensor<4xf32>):
      %next = "onnx.Add"(%current, %current) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %stop = "onnx.Not"(%cond) : (tensor<i1>) -> tensor<i1>
      "onnx.Yield"(%stop, %next) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// Multi-carrier body returns status plus two independent ranked descriptors;
// LLVM uses the upstream packed multi-result struct ABI.
// -----
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: %{{.*}}:3 = hip.loop
  // LLVM-LABEL: llvm.func private @main_graph(
  // LLVM: llvm.call @hipdnn_ep_run_counted_loop
  // LLVM-LABEL: llvm.func @main_graph_loop_body_n0
  // LLVM-SAME: -> !llvm.struct<(i32, struct<{{.*}}>, struct<{{.*}}>)>
  func.func @main_graph(%a: tensor<4xf32>, %b: tensor<8xf32>)
      -> (tensor<4xf32>, tensor<8xf32>) {
    %m = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    %ra, %rb = "onnx.Loop"(%m, %c, %a, %b) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>,
         %ca: tensor<4xf32>, %cb: tensor<8xf32>):
      %na = "onnx.Add"(%ca, %ca) :
          (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %nb = "onnx.Add"(%cb, %cb) :
          (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
      "onnx.Yield"(%cond, %na, %nb) :
          (tensor<i1>, tensor<4xf32>, tensor<8xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4xf32>, tensor<8xf32>)
        -> (tensor<4xf32>, tensor<8xf32>)
    return %ra, %rb : tensor<4xf32>, tensor<8xf32>
  }
}

// Zero-carrier Loop uses status-only body ABI and a real frame token, never an
// empty LLVM/source aggregate.
// -----
module {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: %[[FRAME:.*]] = hip.loop
  // CHECK-SAME: -> (!hip.loop_frame)
  // CHECK: hip.loop_frame_destroy
  // CHECK-LABEL: func.func private @main_graph_loop_body_n0
  // CHECK-SAME: -> i32
  // LLVM-LABEL: llvm.func private @main_graph(
  // LLVM-NOT: !llvm.struct<()>
  // LLVM: llvm.call @hipdnn_ep_run_counted_loop
  // LLVM: llvm.call @hipdnn_ep_loop_frame_destroy
  func.func @main_graph() {
    %m = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    "onnx.Loop"(%m, %c) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>):
      "onnx.Yield"(%cond) : (tensor<i1>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> ()
    return
  }
}
