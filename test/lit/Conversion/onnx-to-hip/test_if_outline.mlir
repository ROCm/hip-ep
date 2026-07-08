// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify `--onnx-if-outline` rewrites `onnx.If` into `hip.if` plus two
// outlined branch funcs. Minimal constant-then / constant-else case from the
// ONNX spec example.

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-if-outline --split-input-file %s | FileCheck %s

// -----

module {
  func.func @main_graph(%cond: tensor<i1>) -> tensor<5xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%then_out) : (tensor<5xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Constant"() {value = dense<[5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%else_out) : (tensor<5xf32>) -> ()
    }) : (tensor<i1>) -> tensor<5xf32>
    return %res : tensor<5xf32>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<i1>)
  // CHECK-DAG: %[[OUT_INIT:.*]] = tensor.empty() : tensor<5xf32>
  // CHECK: %[[RES:.*]] = hip.if(%[[CTX]], %{{.*}})
  // CHECK-SAME: outs(%[[OUT_INIT]] : tensor<5xf32>)
  // CHECK-SAME: -> (tensor<5xf32>)
  // CHECK-SAME: then @main_graph_if_then_n0 else @main_graph_if_else_n1
  // CHECK-SAME: num_outputs = 1 : i32
  // CHECK: return %[[RES]] : tensor<5xf32>

  // CHECK-LABEL: func.func private @main_graph_if_then_n0
  // CHECK-SAME: (%{{.*}}: !hip.context) -> tensor<5xf32>
  // CHECK: return %{{.*}} : tensor<5xf32>

  // CHECK-LABEL: func.func private @main_graph_if_else_n1
  // CHECK-SAME: (%{{.*}}: !hip.context) -> tensor<5xf32>
  // CHECK: return %{{.*}} : tensor<5xf32>
}

// -----

// Capture wiring: each branch reads an outer value (%data).
module {
  func.func @main_graph_capture(%cond: tensor<i1>, %data: tensor<4xf32>) -> tensor<4xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Identity"(%data) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%then_out) : (tensor<4xf32>) -> ()
    }, {
    ^bb0():
      %zero = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
      %else_out = "onnx.Add"(%data, %zero) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%else_out) : (tensor<4xf32>) -> ()
    }) : (tensor<i1>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @main_graph_capture
  // CHECK: hip.if(%{{.*}}, %{{.*}})
  // CHECK-SAME: captures(%[[DATA:.*]] : tensor<4xf32>)
  // CHECK-LABEL: func.func private @main_graph_capture_if_then_n0
  // CHECK-SAME: (%{{.*}}, %[[DATA_ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: "onnx.Identity"(%[[DATA_ARG]])
}
