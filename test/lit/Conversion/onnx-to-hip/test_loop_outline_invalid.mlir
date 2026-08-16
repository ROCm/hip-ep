// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify nested onnx.Loop constructs are outlined in post-order now that every
// invocation owns independent iter/condition storage and carrier banks.
//
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-loop-outline --split-input-file %s | FileCheck %s

// -----

module {
  // CHECK-LABEL: func.func @nested_loop
  // CHECK: hip.loop{{.*}}body @nested_loop_loop_body_n1
  // CHECK-LABEL: func.func private @nested_loop_loop_body_n0
  // CHECK-SAME: !hip.loop_frame
  // CHECK-LABEL: func.func private @nested_loop_loop_body_n1
  // CHECK-SAME: !hip.loop_frame
  // CHECK: hip.loop{{.*}}body @nested_loop_loop_body_n0
  func.func @nested_loop(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %M_outer = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %c_outer = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %v_outer = "onnx.Loop"(%M_outer, %c_outer, %arg0) ({
    ^bb0(%iter_o: tensor<i64>, %cond_in_o: tensor<i1>, %v_in_o: tensor<16xf32>):
      %M_inner = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
      %c_inner = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
      %v_inner = "onnx.Loop"(%M_inner, %c_inner, %v_in_o) ({
      ^bb1(%iter_i: tensor<i64>, %cond_in_i: tensor<i1>, %v_in_i: tensor<16xf32>):
        %v_out_i = "onnx.Add"(%v_in_i, %v_in_i) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
        "onnx.Yield"(%cond_in_i, %v_out_i) : (tensor<i1>, tensor<16xf32>) -> ()
      }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
      "onnx.Yield"(%cond_in_o, %v_inner) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %v_outer : tensor<16xf32>
  }
}
