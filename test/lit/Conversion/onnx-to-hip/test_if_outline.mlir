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
  // Non-greedy operand match: hip.if now prints a `-> (tensor<...>)` result
  // clause, so a greedy `%{{.*}}` would consume past `outs(...)` to the last
  // `)` on the line. `[^)]*` stops at the closing paren of the operand list.
  // CHECK: %[[RES:.*]] = hip.if(%[[CTX]], %{{[^)]*}})
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
  // Non-greedy operand match (see note above) so the scan stops at the
  // operand-list paren and `captures(...)` is still reachable on the line.
  // CHECK: hip.if(%{{[^)]*}}, %{{[^)]*}})
  // CHECK-SAME: captures(%[[DATA:.*]] : tensor<4xf32>)
  // CHECK-LABEL: func.func private @main_graph_capture_if_then_n0
  // CHECK-SAME: (%{{.*}}, %[[DATA_ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: "onnx.Identity"(%[[DATA_ARG]])
}

// -----

// Each branch reads a different outer value. Both outlined funcs still take
// the union, in the same order hip.if passes as captures.
module {
  func.func @main_graph_asymmetric(%cond: tensor<i1>, %a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Identity"(%a) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%then_out) : (tensor<4xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Identity"(%b) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%else_out) : (tensor<4xf32>) -> ()
    }) : (tensor<i1>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @main_graph_asymmetric
  // CHECK: hip.if(%{{[^)]*}}, %{{[^)]*}})
  // CHECK-SAME: captures(%[[A:.*]], %[[B:.*]] : tensor<4xf32>, tensor<4xf32>)
  // CHECK-LABEL: func.func private @main_graph_asymmetric_if_then_n0
  // CHECK-SAME: (%{{.*}}: !hip.context, %[[A_ARG:.*]]: tensor<4xf32>, %{{.*}}: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: "onnx.Identity"(%[[A_ARG]])
  // CHECK-LABEL: func.func private @main_graph_asymmetric_if_else_n1
  // CHECK-SAME: (%{{.*}}: !hip.context, %{{.*}}: tensor<4xf32>, %[[B_ARG:.*]]: tensor<4xf32>) -> tensor<4xf32>
  // CHECK: "onnx.Identity"(%[[B_ARG]])
}

// -----

// Greater of tensor<1xf32> and a scalar broadcasts to tensor<1xui8>. ONNX If
// accepts that single element. Collapse it and read it back before hip.if.
module {
  func.func @main_graph_unit_cond(%cond: tensor<1xui8>) -> tensor<5xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %then_out = "onnx.Constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%then_out) : (tensor<5xf32>) -> ()
    }, {
    ^bb0():
      %else_out = "onnx.Constant"() {value = dense<[5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]> : tensor<5xf32>} : () -> tensor<5xf32>
      "onnx.Yield"(%else_out) : (tensor<5xf32>) -> ()
    }) : (tensor<1xui8>) -> tensor<5xf32>
    return %res : tensor<5xf32>
  }

  // CHECK-LABEL: func.func @main_graph_unit_cond
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[COND:.*]]: tensor<1xui8>)
  // CHECK: %[[SCALAR:.*]] = tensor.collapse_shape %[[COND]] [] : tensor<1xui8> into tensor<ui8>
  // CHECK: hip.readback_scalar(%[[CTX]], %[[SCALAR]] : tensor<ui8>) -> ui8
  // CHECK: arith.cmpi ne
  // CHECK: hip.if(%[[CTX]], {{.*}})
  // CHECK-NOT: onnx.If
}

// -----

// Dynamic output: the then branch grows the cache, the else branch starts
// one, so the size differs per branch. The o_init is sized by a
// hip.if_result_dim placeholder over the hip.if captures.
module {
  func.func @main_graph_dynamic(%cond: tensor<i1>, %x: tensor<?x4xf32>,
                                %past: tensor<?x4xf32>) -> tensor<?x4xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %cat = "onnx.Concat"(%past, %x) {axis = 0 : si64}
          : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
      "onnx.Yield"(%cat) : (tensor<?x4xf32>) -> ()
    }, {
    ^bb0():
      %r = "onnx.Relu"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      "onnx.Yield"(%r) : (tensor<?x4xf32>) -> ()
    }) : (tensor<i1>) -> tensor<?x4xf32>
    return %res : tensor<?x4xf32>
  }

  // CHECK-LABEL: func.func @main_graph_dynamic
  // CHECK: %[[D0:.*]] = hip.if_result_dim(%{{[^)]*}})
  // CHECK-SAME: captures(%{{.*}} : tensor<?x4xf32>, tensor<?x4xf32>)
  // CHECK-SAME: then @main_graph_dynamic_if_then_n0 else @main_graph_dynamic_if_else_n1
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]]) : tensor<?x4xf32>
  // CHECK: hip.if(%{{[^)]*}}, %{{[^)]*}})
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x4xf32>)
}

// -----

// A small outer constant (the LayerNorm Pow exponent in merged decoders) is
// copied into each branch rather than captured, so converters that need a
// constant operand still see one. The large one stays a capture.
module {
  func.func @main_graph_sink(%cond: tensor<i1>, %x: tensor<4xf32>) -> tensor<4xf32> {
    %two = "onnx.Constant"() {value = dense<2.0> : tensor<f32>} : () -> tensor<f32>
    %w = "onnx.Constant"() {value = dense<1.0> : tensor<128xf32>} : () -> tensor<128xf32>
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %p = "onnx.Pow"(%x, %two) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
      "onnx.Yield"(%p) : (tensor<4xf32>) -> ()
    }, {
    ^bb0():
      %s = "onnx.ReduceSum"(%w) {keepdims = 1 : si64} : (tensor<128xf32>) -> tensor<1xf32>
      %a = "onnx.Add"(%x, %s) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
      "onnx.Yield"(%a) : (tensor<4xf32>) -> ()
    }) : (tensor<i1>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @main_graph_sink
  // CHECK: hip.if(%{{[^)]*}}, %{{[^)]*}})
  // CHECK-SAME: captures(%{{[^:]*}} : tensor<4xf32>, tensor<128xf32>)
  // CHECK-LABEL: func.func private @main_graph_sink_if_then_n0
  // CHECK-SAME: (%{{[^,]*}}, %[[X:[^:]*]]: tensor<4xf32>, %{{[^:]*}}: tensor<128xf32>)
  // CHECK: %[[TWO:.*]] = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<f32>}
  // CHECK: "onnx.Pow"(%[[X]], %[[TWO]])
}
