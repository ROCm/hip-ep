// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics \
// RUN:   --hip-add-context-arg --convert-onnx-to-hip %s

module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N", "M"]}
  ]
} {
  // expected-error@+1 {{symbolic dimension rank mismatch for a}}
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"}) -> tensor<?xf32> {
    "onnx.Return"(%a) : (tensor<?xf32>) -> ()
  }
}

// -----

// expected-error@+1 {{duplicate symbolic dimension value record: a}}
module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]},
    {scope = "main_graph", value_name = "a", dimensions = ["N"]}
  ]
} {
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"}) -> tensor<?xf32> {
    "onnx.Return"(%a) : (tensor<?xf32>) -> ()
  }
}

// -----

// expected-error@+1 {{symbolic dimension record must be a dictionary}}
module attributes {
  hipdnn.onnx_dim_params_v1 = ["not-a-dictionary"]
} {
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"}) -> tensor<?xf32> {
    "onnx.Return"(%a) : (tensor<?xf32>) -> ()
  }
}

// -----

module {
  func.func @main_graph(
      %a: tensor<?xf32> {onnx.name = "a"},
      %b: tensor<?xf32> {onnx.name = "b"}) -> tensor<?xf32> {
    // expected-error@+1 {{pre-existing broadcast dimension-source plan is not trusted compiler metadata}}
    %out = "onnx.Add"(%a, %b) {
      hipdnn.broadcast_dim_sources = array<i64: 0>,
      node.outputs = ["out"]
    } : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "onnx.Return"(%out) : (tensor<?xf32>) -> ()
  }
}

// -----

module attributes {
  hipdnn.onnx_dim_params_v1 = [
    {scope = "main_graph", value_name = "a", dimensions = ["N"]}
  ]
} {
  // NoneType is the only non-ranked exception. A named unranked internal
  // tensor still fails closed because its metadata rank cannot be validated.
  // expected-error@+1 {{symbolic dimension metadata names a non-ranked tensor value: a}}
  func.func @main_graph(
      %seed: tensor<?xf32> {onnx.name = "seed"}) -> tensor<?xf32> {
    %a = "onnx.Identity"(%seed) {node.outputs = ["a"]}
        : (tensor<?xf32>) -> tensor<*xf32>
    "onnx.Return"(%seed) : (tensor<?xf32>) -> ()
  }
}
