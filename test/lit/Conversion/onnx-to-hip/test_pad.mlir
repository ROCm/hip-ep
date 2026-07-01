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
  // pads are brought to host via the explicit transfer mechanism (emitted before
  // the output init, since the dynamic-shape path reads pad amounts from
  // this same host copy); hip.pad reads the host copy (wrap_pad consumes pads
  // CPU-side). The output init is a DEVICE-space bufferization.alloc_tensor (the
  // pad kernel writes device memory), so hip.pad's output buffer is device-typed
  // after bufferization.
  // CHECK: %[[PH:.*]] = hip.transfer(%[[CTX]], %[[P]] : tensor<4xi64>) to <host> -> tensor<4xi64>
  // CHECK: bufferization.alloc_tensor() {memory_space = #hip.mem<device>} : tensor<5x6xf32>
  // CHECK: hip.pad(%[[CTX]]) ins(%[[D]], %[[PH]] : tensor<3x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<5x6xf32>)

  // Constant pad with a COMPILE-TIME-CONSTANT fill value (the common case, e.g.
  // 0.0). hip.pad takes the fill value BY VALUE (a scalar `f32`, no buffer):
  // the converter folds the constant straight to an arith.constant scalar --
  // zero device traffic, NO transfer and NO readback for the cval.
  func.func @pad_constant_cval_const(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %cval = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %cval, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_cval_const
  // The fill value is folded to a scalar constant: no transfer of the cval and
  // crucially no hip.readback_scalar for it.
  // CHECK-NOT: hip.readback_scalar
  // CHECK: %[[CV:.*]] = arith.constant {{.*}} : f32
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) cval(%[[CV]] : f32) outs({{.*}} : tensor<5x6xf32>)

  // Constant pad with a RUNTIME (non-constant) constant_value scalar (here a
  // function arg). hip.pad still takes it BY VALUE: the converter brings the
  // runtime scalar to the host via the SAME explicit transfer mechanism used
  // for pads/axes (hip.transfer -> stack #hip.mem<host> buffer + memcpy_d2h +
  // sync at bufferization), then reads it by value with tensor.extract.
  func.func @pad_constant_with_cval(%data: tensor<3x4xf32>, %pads: tensor<4xi64>, %cval: tensor<f32>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %cval, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, tensor<f32>, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_with_cval
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<3x4xf32>, %[[P:.*]]: tensor<4xi64>, %[[CVT:.*]]: tensor<f32>)
  // CHECK: %[[CVH:.*]] = hip.transfer(%[[CTX]], %[[CVT]] : tensor<f32>) to <host> -> tensor<f32>
  // CHECK: %[[CV:.*]] = tensor.extract %[[CVH]][] : tensor<f32>
  // CHECK: hip.transfer(%[[CTX]], %{{.*}} : tensor<4xi64>) to <host> -> tensor<4xi64>
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) cval(%[[CV]] : f32) outs({{.*}} : tensor<5x6xf32>)

  // A producer may emit constant_value as a redundant single-element 1-D
  // tensor. For a RUNTIME value the converter collapses it to rank-0, then
  // brings it to host via transfer + tensor.extract (here a function arg).
  func.func @pad_constant_cval_1d(%data: tensor<3x4xf32>, %pads: tensor<4xi64>, %cval: tensor<1xf32>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %cval, %none) {mode = "constant"} : (tensor<3x4xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_constant_cval_1d
  // CHECK: %[[CV0:.*]] = tensor.collapse_shape %{{.*}} [] : tensor<1xf32> into tensor<f32>
  // CHECK: %[[CVH:.*]] = hip.transfer({{.*}}, %[[CV0]] : tensor<f32>) to <host> -> tensor<f32>
  // CHECK: %[[CV:.*]] = tensor.extract %[[CVH]][] : tensor<f32>
  // CHECK: hip.transfer({{.*}}, %{{.*}} : tensor<4xi64>) to <host> -> tensor<4xi64>
  // CHECK: hip.pad({{.*}}) ins({{.*}}, {{.*}} : tensor<3x4xf32>, tensor<4xi64>) cval(%[[CV]] : f32) outs({{.*}} : tensor<5x6xf32>)

  // Reflect mode is non-default, so it stays in the attr-dict.
  func.func @pad_reflect(%data: tensor<3x4xf32>, %pads: tensor<4xi64>) -> tensor<5x6xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "reflect"} : (tensor<3x4xf32>, tensor<4xi64>, none, none) -> tensor<5x6xf32>
    return %r : tensor<5x6xf32>
  }

  // CHECK-LABEL: func.func @pad_reflect
  // CHECK: hip.transfer({{.*}}, %{{.*}} : tensor<4xi64>) to <host> -> tensor<4xi64>
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
  // Both pads AND axes are transferred to host (both consumed CPU-side by wrap_pad).
  // CHECK-DAG: hip.transfer({{.*}}, %{{.*}} : tensor<2xi64>) to <host> -> tensor<2xi64>
  // CHECK-DAG: hip.transfer({{.*}}, %{{.*}} : tensor<1xi64>) to <host> -> tensor<1xi64>
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
  // CHECK: bufferization.alloc_tensor(%[[OUT0]]) {memory_space = #hip.mem<device>} : tensor<?x6xf32>
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
  // CHECK: bufferization.alloc_tensor({{.*}}) {memory_space = #hip.mem<device>} : tensor<?x6xf32>
  // CHECK: hip.pad({{.*}}) ins({{.*}} : tensor<?x4xf32>, tensor<4xi64>) outs({{.*}} : tensor<?x6xf32>)

  // Dynamic output dims with a non-constant `pads` (function arg): `pads` is
  // brought to host ONCE via hip.transfer (the same host copy wrap_pad consumes
  // CPU-side), and each padded axis's two pad entries are read from THAT host
  // copy with a synchronized tensor.extract -- NOT a bare tensor.extract of the
  // device `pads` and NOT hip.readback_scalar. A bare device extract bufferizes
  // to an unsynchronized host load of the device-resident pads buffer and SEGVs
  // when `pads` is an externalized constant on a true-device-memory target;
  // extracting from the post-sync host transfer copy is safe. See
  // PadConversion.cpp's extractAsIndex.
  func.func @pad_dyn_output_dyn_pads(%data: tensor<?x?xf32>, %pads: tensor<4xi64>) -> tensor<?x?xf32> {
    %none = "onnx.NoValue"() {value} : () -> none
    %r = "onnx.Pad"(%data, %pads, %none, %none) {mode = "constant"} : (tensor<?x?xf32>, tensor<4xi64>, none, none) -> tensor<?x?xf32>
    return %r : tensor<?x?xf32>
  }

  // CHECK-LABEL: func.func @pad_dyn_output_dyn_pads
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[D:.*]]: tensor<?x?xf32>, %[[P:.*]]: tensor<4xi64>)
  // `pads` is transferred to host once; the per-axis pad entries are read from
  // that host copy via tensor.extract (no readback, no extract_slice). The
  // index_cast results feed the addi chain into tensor.empty.
  // CHECK: %[[PH:.*]] = hip.transfer(%[[CTX]], %[[P]] : tensor<4xi64>) to <host> -> tensor<4xi64>
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.extract %[[PH]]{{\[}}%{{.*}}] : tensor<4xi64>
  // CHECK: arith.index_cast %{{.*}} : i64 to index
  // CHECK: bufferization.alloc_tensor(%{{.*}}, %{{.*}}) {memory_space = #hip.mem<device>} : tensor<?x?xf32>
  // CHECK: hip.pad
}
