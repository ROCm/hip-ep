// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip %s | FileCheck %s

// One caller-controlled binding for N. Both internal Cast results carry N, so
// Add destination construction reuses one contributor without a runtime merge.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["N"]},
    {scope = "main_graph", value_name = "c", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-NOT: hipdnn.onnx_dim_params_v1
  // CHECK-NOT: arith.select
  // CHECK: hip.cast
  // CHECK: hip.cast
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: hip.add{{.*}}outs(%[[INIT]] :
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> tensor<?xf32> {
    %b = "onnx.Cast"(%a) {node.outputs = ["b"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %c = "onnx.Cast"(%a) {node.outputs = ["c"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %out = "onnx.Add"(%b, %c) {node.outputs = ["out"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

// A successor-block argument is not a function argument. Its local argument
// number must not inherit onnx.name from the function entry block.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "c", dimensions = ["N"]},
    {scope = "main_graph", value_name = "cfg_out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: ^bb1
  // CHECK: arith.select
  // CHECK: hip.add
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}, %independent: tensor<?xf32>)
      -> tensor<?xf32> {
    cf.br ^bb1(%independent : tensor<?xf32>)
  ^bb1(%block_arg: tensor<?xf32>):
    %c = "onnx.Cast"(%a) {node.outputs = ["c"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %cfg_out = "onnx.Add"(%block_arg, %c) {node.outputs = ["cfg_out"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%cfg_out) : (tensor<?xf32>) -> ()
  }
}

// -----

// V1 metadata is scoped to operations directly in main_graph. Values inside a
// nested region cannot borrow main-graph names to prove independent extents.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "x", dimensions = ["N"]},
    {scope = "main_graph", value_name = "y", dimensions = ["N"]},
    {scope = "main_graph", value_name = "nested_out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: scf.if
  // CHECK: arith.select
  // CHECK: hip.add
  func.func @main_graph(
      %p: tensor<?xf16>, %q: tensor<?xf16>) -> tensor<?xf32> {
    %condition = arith.constant true
    %result = scf.if %condition -> tensor<?xf32> {
      %x = "onnx.Cast"(%p) {node.outputs = ["x"], to = 1 : i64}
          : (tensor<?xf16>) -> tensor<?xf32>
      %y = "onnx.Cast"(%q) {node.outputs = ["y"], to = 1 : i64}
          : (tensor<?xf16>) -> tensor<?xf32>
      %nested_out = "onnx.Add"(%x, %y) {node.outputs = ["nested_out"]}
          : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
      scf.yield %nested_out : tensor<?xf32>
    } else {
      %fallback = "onnx.Cast"(%p) {to = 1 : i64}
          : (tensor<?xf16>) -> tensor<?xf32>
      scf.yield %fallback : tensor<?xf32>
    }
    "onnx.Return"(%result) : (tensor<?xf32>) -> ()
  }
}

// -----

// Greater lowers with reversed HIP operands. The positional plan is remapped
// through actual operand Values, so equal symbols still avoid the merge.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["N"]},
    {scope = "main_graph", value_name = "c", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: %[[B:.*]] = hip.cast
  // CHECK: %[[C:.*]] = hip.cast
  // CHECK: tensor.dim %[[B]]
  // CHECK-NOT: tensor.dim %[[C]]
  // CHECK-NOT: arith.select
  // CHECK: hip.less{{.*}}ins(%[[C]], %[[B]]
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> tensor<?xi1> {
    %b = "onnx.Cast"(%a) {node.outputs = ["b"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %c = "onnx.Cast"(%a) {node.outputs = ["c"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %out = "onnx.Greater"(%b, %c) {node.outputs = ["out"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    "onnx.Return"(%out) : (tensor<?xi1>) -> ()
  }
}

// -----

// Where consumes all three original contributors. One graph-input binding for
// N makes the complete contributor class eligible.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "cond", dimensions = ["N"]},
    {scope = "main_graph", value_name = "x", dimensions = ["N"]},
    {scope = "main_graph", value_name = "y", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK-NOT: arith.select
  // CHECK: hip.where
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> tensor<?xf32> {
    %cond = "onnx.Cast"(%a) {node.outputs = ["cond"], to = 9 : i64}
        : (tensor<?xf16>) -> tensor<?xi1>
    %x = "onnx.Cast"(%a) {node.outputs = ["x"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %y = "onnx.Cast"(%a) {node.outputs = ["y"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %out = "onnx.Where"(%cond, %x, %y) {node.outputs = ["out"]}
        : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

// Where requires one complete equality class across condition, X, and Y.
// A different condition symbol retains exact three-way broadcast.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "cond", dimensions = ["M"]},
    {scope = "main_graph", value_name = "x", dimensions = ["N"]},
    {scope = "main_graph", value_name = "y", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: arith.select
  // CHECK: hip.where
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> tensor<?xf32> {
    %cond = "onnx.Cast"(%a) {node.outputs = ["cond"], to = 9 : i64}
        : (tensor<?xf16>) -> tensor<?xi1>
    %x = "onnx.Cast"(%a) {node.outputs = ["x"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %y = "onnx.Cast"(%a) {node.outputs = ["y"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %out = "onnx.Where"(%cond, %x, %y) {node.outputs = ["out"]}
        : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

// Variadic Max uses a pairwise HIP chain. Intermediate accumulators have no
// ONNX tensor identity, so v1 deliberately keeps exact runtime selects.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["N"]},
    {scope = "main_graph", value_name = "c", dimensions = ["N"]},
    {scope = "main_graph", value_name = "d", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: arith.select
  // CHECK: hip.max
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> tensor<?xf32> {
    %b = "onnx.Cast"(%a) {node.outputs = ["b"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %c = "onnx.Cast"(%a) {node.outputs = ["c"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %d = "onnx.Cast"(%a) {node.outputs = ["d"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %out = "onnx.Max"(%b, %c, %d) {node.outputs = ["out"]}
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

// Distinct symbols retain exact dynamic broadcast.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["M"]},
    {scope = "main_graph", value_name = "out", dimensions = ["K"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: arith.cmpi eq
  // CHECK: arith.select
  // CHECK: tensor.empty
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"},
      %b: tensor<?xf32> {onnx.name = "b"}) -> tensor<?xf32> {
    %out = "onnx.Add"(%a, %b) {node.outputs = ["out"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

// The same symbol on two independently caller-controlled axes is conservative:
// ORT does not enforce their runtime equality, so retain exact broadcast.
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["N"]},
    {scope = "main_graph", value_name = "out", dimensions = ["N"]}
  ]
} {
  // CHECK-LABEL: func.func @main_graph
  // CHECK: arith.cmpi eq
  // CHECK: arith.select
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"},
      %b: tensor<?xf32> {onnx.name = "b"}) -> tensor<?xf32> {
    %out = "onnx.Add"(%a, %b) {node.outputs = ["out"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}
