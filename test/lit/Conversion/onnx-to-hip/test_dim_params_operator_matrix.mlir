// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s \
// RUN:   | FileCheck --implicit-check-not=arith.select %s

module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "b", dimensions = ["N"]},
    {scope = "main_graph", value_name = "c", dimensions = ["N"]},
    {scope = "main_graph", value_name = "add", dimensions = ["N"]},
    {scope = "main_graph", value_name = "and", dimensions = ["N"]},
    {scope = "main_graph", value_name = "div", dimensions = ["N"]},
    {scope = "main_graph", value_name = "eq", dimensions = ["N"]},
    {scope = "main_graph", value_name = "ge", dimensions = ["N"]},
    {scope = "main_graph", value_name = "gt", dimensions = ["N"]},
    {scope = "main_graph", value_name = "le", dimensions = ["N"]},
    {scope = "main_graph", value_name = "lt", dimensions = ["N"]},
    {scope = "main_graph", value_name = "mod", dimensions = ["N"]},
    {scope = "main_graph", value_name = "mul", dimensions = ["N"]},
    {scope = "main_graph", value_name = "or", dimensions = ["N"]},
    {scope = "main_graph", value_name = "sub", dimensions = ["N"]},
    {scope = "main_graph", value_name = "where", dimensions = ["N"]}
  ]
} {
  // Every supported ONNX broadcast converter must consume the operation-local
  // symbolic plan. A missed converter emits an arith.select for this dynamic
  // equal-symbol axis.
  // CHECK-LABEL: func.func @main_graph
  // CHECK-DAG: hip.add
  // CHECK-DAG: hip.sub
  // CHECK-DAG: hip.mul
  // CHECK-DAG: hip.div
  // CHECK-DAG: hip.mod
  // CHECK-DAG: hip.equal
  // CHECK-DAG: hip.less
  // CHECK-DAG: hip.and
  // CHECK-DAG: hip.or
  // CHECK-DAG: hip.where
  func.func @main_graph(
      %a: tensor<?xf16> {onnx.name = "a"}) -> (
        tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
        tensor<?xf32>, tensor<?xi1>, tensor<?xi1>, tensor<?xi1>,
        tensor<?xi1>, tensor<?xi1>, tensor<?xi1>, tensor<?xi1>,
        tensor<?xf32>) {
    %b = "onnx.Cast"(%a) {node.outputs = ["b"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %c = "onnx.Cast"(%a) {node.outputs = ["c"], to = 1 : i64}
        : (tensor<?xf16>) -> tensor<?xf32>
    %add = "onnx.Add"(%b, %c) {node.outputs = ["add"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %sub = "onnx.Sub"(%b, %c) {node.outputs = ["sub"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %mul = "onnx.Mul"(%b, %c) {node.outputs = ["mul"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %div = "onnx.Div"(%b, %c) {node.outputs = ["div"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %mod = "onnx.Mod"(%b, %c) {
      fmod = 1 : si64, node.outputs = ["mod"]
    } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %eq = "onnx.Equal"(%b, %c) {node.outputs = ["eq"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    %gt = "onnx.Greater"(%b, %c) {node.outputs = ["gt"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    %ge = "onnx.GreaterOrEqual"(%b, %c) {node.outputs = ["ge"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    %lt = "onnx.Less"(%b, %c) {node.outputs = ["lt"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    %le = "onnx.LessOrEqual"(%b, %c) {node.outputs = ["le"]}
        : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    %and = "onnx.And"(%eq, %lt) {node.outputs = ["and"]}
        : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
    %or = "onnx.Or"(%eq, %lt) {node.outputs = ["or"]}
        : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
    %where = "onnx.Where"(%eq, %add, %sub) {node.outputs = ["where"]}
        : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%add, %sub, %mul, %div, %mod, %eq, %gt, %ge, %lt, %le,
                  %and, %or, %where)
        : (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>,
           tensor<?xf32>, tensor<?xi1>, tensor<?xi1>, tensor<?xi1>,
           tensor<?xi1>, tensor<?xi1>, tensor<?xi1>, tensor<?xi1>,
           tensor<?xf32>) -> ()
  }
}
