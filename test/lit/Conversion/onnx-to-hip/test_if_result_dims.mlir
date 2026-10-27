// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify `--hip-resolve-if-result-dims` sizes dynamic hip.if outputs from
// each branch's host shape arithmetic, selected on the If condition. The
// shape is a merged decoder's: the cache branch concatenates the past with
// the new rows and returns a static dummy for the second output; the no-cache
// branch computes both from the new rows.

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-if-outline --convert-onnx-to-hip --canonicalize --hip-resolve-if-result-dims --cse --split-input-file %s | FileCheck %s

module {
  func.func @main_graph(%cond: tensor<i1>, %x: tensor<?x4xf32>,
                        %past: tensor<?x4xf32>)
      -> (tensor<?x4xf32>, tensor<?x4xf32>) {
    %res:2 = "onnx.If"(%cond) ({
    ^bb0():
      %cat = "onnx.Concat"(%past, %x) {axis = 0 : si64}
          : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
      %dummy = "onnx.Constant"() {value = dense<0.0> : tensor<0x4xf32>}
          : () -> tensor<0x4xf32>
      "onnx.Yield"(%cat, %dummy) : (tensor<?x4xf32>, tensor<0x4xf32>) -> ()
    }, {
    ^bb0():
      %r = "onnx.Relu"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      %n = "onnx.Neg"(%x) : (tensor<?x4xf32>) -> tensor<?x4xf32>
      "onnx.Yield"(%r, %n) : (tensor<?x4xf32>, tensor<?x4xf32>) -> ()
    }) : (tensor<i1>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
    return %res#0, %res#1 : tensor<?x4xf32>, tensor<?x4xf32>
  }

  // CHECK-LABEL: func.func @main_graph(
  // CHECK-SAME: %{{.*}}: !hip.context, %{{.*}}: tensor<i1>, %[[X:.*]]: tensor<?x4xf32>, %[[PAST:.*]]: tensor<?x4xf32>)
  // CHECK-NOT: hip.if_result_dim
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[P:.*]] = tensor.dim %[[PAST]], %[[C0]] : tensor<?x4xf32>
  // CHECK-DAG: %[[S:.*]] = tensor.dim %[[X]], %[[C0]] : tensor<?x4xf32>
  // CHECK: %[[T:.*]] = arith.addi %[[P]], %[[S]] : index
  // CHECK: %[[D0:.*]] = arith.select %[[COND:.*]], %[[T]], %[[S]] : index
  // CHECK: %[[INIT0:.*]] = tensor.empty(%[[D0]]) : tensor<?x4xf32>
  // CHECK: %[[D1:.*]] = arith.select %[[COND]], %[[C0]], %{{.*}} : index
  // CHECK: %[[INIT1:.*]] = tensor.empty(%[[D1]]) : tensor<?x4xf32>
  // CHECK: hip.if(%{{[^)]*}}, %[[COND]])
  // CHECK-SAME: outs(%[[INIT0]], %[[INIT1]] : tensor<?x4xf32>, tensor<?x4xf32>)
  // CHECK-NOT: hip.if_result_dim
}

// -----

// Head split as decoders export it: the cache branch reshapes to a target
// built from Shape/Gather/Concat, the other uses Reshape's 0 ("copy") and -1
// ("infer") entries. The sizes must come out as host arithmetic on the input
// dims. The readback spelling the full decoders take is covered by
// test/lit/Dialect/hip-resolve-if-result-dims.mlir.
module {
  func.func @main_graph(%cond: tensor<i1>, %x: tensor<?x?x512xf32>)
      -> tensor<?x?x8x64xf32> {
    %res = "onnx.If"(%cond) ({
    ^bb0():
      %shape = "onnx.Shape"(%x) : (tensor<?x?x512xf32>) -> tensor<3xi64>
      %k0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
      %k1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
      %b = "onnx.Gather"(%shape, %k0) {axis = 0 : si64}
          : (tensor<3xi64>, tensor<1xi64>) -> tensor<1xi64>
      %s = "onnx.Gather"(%shape, %k1) {axis = 0 : si64}
          : (tensor<3xi64>, tensor<1xi64>) -> tensor<1xi64>
      %hd = "onnx.Constant"() {value = dense<[8, 64]> : tensor<2xi64>} : () -> tensor<2xi64>
      %target = "onnx.Concat"(%b, %s, %hd) {axis = 0 : si64}
          : (tensor<1xi64>, tensor<1xi64>, tensor<2xi64>) -> tensor<4xi64>
      %r = "onnx.Reshape"(%x, %target)
          : (tensor<?x?x512xf32>, tensor<4xi64>) -> tensor<?x?x8x64xf32>
      "onnx.Yield"(%r) : (tensor<?x?x8x64xf32>) -> ()
    }, {
    ^bb0():
      %target = "onnx.Constant"() {value = dense<[0, -1, 8, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
      %r = "onnx.Reshape"(%x, %target)
          : (tensor<?x?x512xf32>, tensor<4xi64>) -> tensor<?x?x8x64xf32>
      "onnx.Yield"(%r) : (tensor<?x?x8x64xf32>) -> ()
    }) : (tensor<i1>) -> tensor<?x?x8x64xf32>
    return %res : tensor<?x?x8x64xf32>
  }

  // CHECK-LABEL: func.func @main_graph(%{{.*}}: !hip.context, %{{.*}}: tensor<i1>, %{{.*}}: tensor<?x?x512xf32>)
  // CHECK-NOT: hip.if_result_dim
  // CHECK-DAG: %[[B:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?x512xf32>
  // CHECK-DAG: %[[S:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?x?x512xf32>
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x8x64xf32>
  // CHECK: hip.if
  // CHECK-NOT: hip.if_result_dim
}
