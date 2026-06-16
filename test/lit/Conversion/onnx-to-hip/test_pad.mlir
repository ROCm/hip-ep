// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Constant pad (default mode), no constant_value, no axes.
  // Default mode is elided from attr-dict, so just check the op shape.
  func.func @pad_constant_default(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_default
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<3x4xf32>, %[[P:.*]]: tensor<4xi64>)
  // CHECK: tensor.empty() : tensor<5x6xf32>
  // CHECK: hip.pad(%[[CTX]]) ins(%[[D]], %[[P]] : tensor<3x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<5x6xf32>)

  // Constant pad with explicit constant_value scalar.
  func.func @pad_constant_with_cval(%data: tensor<3x4xf32>, %pads: tensor<4xi64>, %cval: tensor<f32>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %cval, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_with_cval
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) cval({{.*}} : tensor<f32>) outs({{.*}} : tensor<5x6xf32>)

  // Reflect mode is non-default, so it stays in the attr-dict.
  func.func @pad_reflect(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "reflect"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_reflect
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<5x6xf32>) {mode = "reflect"}

  func.func @pad_edge(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "edge"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_edge
  // CHECK: hip.pad({{.*}}) {{.*}} {mode = "edge"}

  // With axes input.
  func.func @pad_axes(%data: tensor<3x4xf32>, %pads: tensor<2xi64>, %axes: tensor<1xi64>) -> tensor<3x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %axes) {mode = "constant"} : (tensor<3x4xf32>, tensor<2xi64>, none, tensor<1xi64>) -> tensor<3x6xf32>
    return %r : tensor<3x6xf32>
  }

  // CHECK-LABEL: func.func @pad_axes
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<2xi64>) axes({{.*}} : tensor<1xi64>) outs({{.*}} : tensor<3x6xf32>)

  // Dynamic output dims with a compile-time constant `pads`: the
  // pattern resolves each padded axis's output extent to
  //   data_dim[i] + pads[i] + pads[i + N]
  // entirely at IR-build time. data dim 0 is dynamic so its
  // contribution is a tensor.dim; both pad amounts are arith.constants
  // sourced from the constant `pads` vector.
  func.func @pad_dyn_output_const_pads(%data: tensor<?x4xf32>) -> tensor<?x6xf32> {
    %pads = arith.constant dense<[1, 1, 2, 1]> : tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<?x4xf32>, tensor<4xi64>, none, none) -> tensor<?x6xf32>
    return %r : tensor<?x6xf32>
  }

  // CHECK-LABEL: func.func @pad_dyn_output_const_pads
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x4xf32>)
  // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[D0:.*]] = tensor.dim %[[D]], %[[A0]] : tensor<?x4xf32>
  // CHECK-DAG: %[[B0:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[E0:.*]] = arith.constant 2 : index
  // CHECK: %[[S0:.*]] = arith.addi %[[D0]], %[[B0]] : index
  // CHECK: %[[OUT0:.*]] = arith.addi %[[S0]], %[[E0]] : index
  // CHECK: tensor.empty(%[[OUT0]]) : tensor<?x6xf32>
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<?x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<?x6xf32>)

  // Dynamic output with `pads` supplied as an onnx.Constant -- the form that
  // lowerOnnxConstants may externalize (replacing the inline value with a NULL
  // memref.global). The pre-lowering PadShapeFold stamps the constant onto the
  // op BEFORE externalization, so the output shape folds to arith.constants
  // here: no hip.readback_scalar, no device read of the pads buffer. This is
  // the path-1 fix for the externalized-pads host-read SEGV.
  func.func @pad_dyn_output_onnx_const_pads(%data: tensor<?x4xf32>) -> tensor<?x6xf32> {
    %pads = "onnx.Constant"() {value = dense<[1, 1, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<?x4xf32>, tensor<4xi64>, none, none) -> tensor<?x6xf32>
    return %r : tensor<?x6xf32>
  }

  // CHECK-LABEL: func.func @pad_dyn_output_onnx_const_pads
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x4xf32>)
  // No readback / extract_slice anywhere in the body: the pad amounts were
  // folded at compile time, so the only operand feeding tensor.empty is the
  // tensor.dim of the data plus arith.constant pad amounts.
  // CHECK-NOT: hip.readback_scalar
  // CHECK-NOT: tensor.extract_slice
  // CHECK: tensor.empty({{.*}}) : tensor<?x6xf32>
  // CHECK: hip.pad({{.*}}) ins({{.*}} : tensor<?x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<?x6xf32>)

  // Dynamic output dims with a non-constant `pads` (function arg): the
  // per-axis padding amounts are read on the HOST through a synchronized
  // hip.readback_scalar (extract_slice -> collapse_shape -> readback_scalar ->
  // index_cast), NOT a bare tensor.extract. A bare extract bufferizes to an
  // unsynchronized host load of the device-resident pads buffer and SEGVs when
  // `pads` is an externalized constant on a true-device-memory target -- see
  // PadConversion.cpp's extractAsIndex.
  func.func @pad_dyn_output_dyn_pads(%data: tensor<?x?xf32>, %pads: tensor<4xi64>) -> tensor<?x?xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<?x?xf32>, tensor<4xi64>, none, none) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @pad_dyn_output_dyn_pads
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x?xf32>, %[[P:.*]]: tensor<4xi64>)
  // Each padded axis's two pad entries are read via the synchronized readback
  // path; the index_cast results feed the addi chain into tensor.empty.
  // CHECK: tensor.extract_slice %[[P]]
  // CHECK: tensor.collapse_shape
  // CHECK: hip.readback_scalar(%[[CTX]], %{{.*}} : tensor<i64>) -> i64
  // CHECK: arith.index_cast %{{.*}} : i64 to index
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?xf32>
  // CHECK: hip.pad
}
