// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the exported GroupNorm decomposition
//   Reshape(x, [0, G, -1]) -> InstanceNormalization(1, 0) -> Reshape(_, Shape(x))
//   -> [Mul gamma] -> [Add beta] -> [x * Sigmoid(x)]
// fuses to one hip.group_norm (channels_last = 0).
//
// 1. Full chain with SiLU, dynamic batch and spatial extents
// 2. Static shapes, second reshape by constant, no SiLU
// 3. Add only (gamma becomes ones), rank-4 [1,C,1,1] beta
// 4. InstanceNormalization scale is not 1: stays unfused
// 5. Second reshape is not x's shape: stays unfused
// 6. An intermediate has a second use: stays unfused

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  func.func @decomposed_silu_dynamic(%x: tensor<?x64x?x?xf16>) -> tensor<?x64x?x?xf16> {
    %g = "onnx.Constant"() {value = dense<[0, 32, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<32xf16>} : () -> tensor<32xf16>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<32xf16>} : () -> tensor<32xf16>
    %gamma = "onnx.Constant"() {value = dense<1.5> : tensor<64x1x1xf16>} : () -> tensor<64x1x1xf16>
    %beta = "onnx.Constant"() {value = dense<0.25> : tensor<64x1x1xf16>} : () -> tensor<64x1x1xf16>
    %r = "onnx.Reshape"(%x, %g) {allowzero = 0 : si64} : (tensor<?x64x?x?xf16>, tensor<3xi64>) -> tensor<?x32x?xf16>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) {epsilon = 9.99999974E-6 : f32} : (tensor<?x32x?xf16>, tensor<32xf16>, tensor<32xf16>) -> tensor<?x32x?xf16>
    %s = "onnx.Shape"(%x) : (tensor<?x64x?x?xf16>) -> tensor<4xi64>
    %b = "onnx.Reshape"(%n, %s) {allowzero = 0 : si64} : (tensor<?x32x?xf16>, tensor<4xi64>) -> tensor<?x64x?x?xf16>
    %m = "onnx.Mul"(%b, %gamma) : (tensor<?x64x?x?xf16>, tensor<64x1x1xf16>) -> tensor<?x64x?x?xf16>
    %a = "onnx.Add"(%beta, %m) : (tensor<64x1x1xf16>, tensor<?x64x?x?xf16>) -> tensor<?x64x?x?xf16>
    %sg = "onnx.Sigmoid"(%a) : (tensor<?x64x?x?xf16>) -> tensor<?x64x?x?xf16>
    %y = "onnx.Mul"(%sg, %a) : (tensor<?x64x?x?xf16>, tensor<?x64x?x?xf16>) -> tensor<?x64x?x?xf16>
    return %y : tensor<?x64x?x?xf16>
  }

  // CHECK-LABEL: func.func @decomposed_silu_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x64x?x?xf16>)
  // CHECK-DAG: %[[GAMMA:.*]] = hip.constant {{.*}}dense<1.500000e+00> : tensor<64xf16>
  // CHECK-DAG: %[[BETA:.*]] = hip.constant {{.*}}dense<2.500000e-01> : tensor<64xf16>
  // CHECK-NOT: hip.instance_norm
  // CHECK-NOT: hip.sigmoid
  // CHECK: %[[Y:.*]] = hip.group_norm(%[[CTX]]) ins(%[[X]], %[[GAMMA]], %[[BETA]] :
  // CHECK-SAME: activation = 1
  // CHECK-SAME: channels_last = 0
  // CHECK-SAME: groups = 32
  // CHECK-NOT: hip.instance_norm
  // CHECK-NOT: hip.mul
  // CHECK: return %[[Y]]

  func.func @decomposed_static(%x: tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32> {
    %g = "onnx.Constant"() {value = dense<[2, 4, 32]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %gamma = "onnx.Constant"() {value = dense<[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]], [[7.0]], [[8.0]]]> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %beta = "onnx.Constant"() {value = dense<0.5> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %back = "onnx.Constant"() {value = dense<[0, 8, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %r = "onnx.Reshape"(%x, %g) : (tensor<2x8x4x4xf32>, tensor<3xi64>) -> tensor<2x4x32xf32>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) : (tensor<2x4x32xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4x32xf32>
    %b = "onnx.Reshape"(%n, %back) : (tensor<2x4x32xf32>, tensor<4xi64>) -> tensor<2x8x4x4xf32>
    %m = "onnx.Mul"(%gamma, %b) : (tensor<8x1x1xf32>, tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
    %y = "onnx.Add"(%m, %beta) : (tensor<2x8x4x4xf32>, tensor<8x1x1xf32>) -> tensor<2x8x4x4xf32>
    return %y : tensor<2x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @decomposed_static
  // CHECK-DAG: hip.constant {{.*}}dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<8xf32>
  // CHECK-DAG: hip.constant {{.*}}dense<5.000000e-01> : tensor<8xf32>
  // CHECK-NOT: hip.instance_norm
  // CHECK: hip.group_norm
  // CHECK-SAME: channels_last = 0
  // CHECK-SAME: groups = 4
  // CHECK-NOT: hip.add

  func.func @decomposed_add_only(%x: tensor<1x8x16xf32>) -> tensor<1x8x16xf32> {
    %g = "onnx.Constant"() {value = dense<[0, 2, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<2xf32>} : () -> tensor<2xf32>
    %beta = "onnx.Constant"() {value = dense<0.5> : tensor<1x8x1xf32>} : () -> tensor<1x8x1xf32>
    %r = "onnx.Reshape"(%x, %g) : (tensor<1x8x16xf32>, tensor<3xi64>) -> tensor<1x2x64xf32>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) : (tensor<1x2x64xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x64xf32>
    %s = "onnx.Shape"(%x) : (tensor<1x8x16xf32>) -> tensor<3xi64>
    %b = "onnx.Reshape"(%n, %s) : (tensor<1x2x64xf32>, tensor<3xi64>) -> tensor<1x8x16xf32>
    %y = "onnx.Add"(%b, %beta) : (tensor<1x8x16xf32>, tensor<1x8x1xf32>) -> tensor<1x8x16xf32>
    return %y : tensor<1x8x16xf32>
  }

  // CHECK-LABEL: func.func @decomposed_add_only
  // CHECK-DAG: hip.constant {{.*}}dense<1.000000e+00> : tensor<8xf32>
  // CHECK-DAG: hip.constant {{.*}}dense<5.000000e-01> : tensor<8xf32>
  // CHECK-NOT: hip.instance_norm
  // CHECK: hip.group_norm
  // CHECK-SAME: groups = 2

  func.func @decomposed_scaled_instance_norm(%x: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %g = "onnx.Constant"() {value = dense<[0, 4, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %twos = "onnx.Constant"() {value = dense<2.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %r = "onnx.Reshape"(%x, %g) : (tensor<1x8x4x4xf32>, tensor<3xi64>) -> tensor<1x4x32xf32>
    %n = "onnx.InstanceNormalization"(%r, %twos, %zeros) : (tensor<1x4x32xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x32xf32>
    %s = "onnx.Shape"(%x) : (tensor<1x8x4x4xf32>) -> tensor<4xi64>
    %y = "onnx.Reshape"(%n, %s) : (tensor<1x4x32xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
    return %y : tensor<1x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @decomposed_scaled_instance_norm
  // CHECK: hip.instance_norm
  // CHECK-NOT: hip.group_norm

  // An externally backed scale cannot be inspected at compile time, so the
  // chain is left as InstanceNorm rather than assumed to be ones.
  func.func @decomposed_external_scale(%x: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
    %g = "onnx.Constant"() {value = dense<[0, 4, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {location = "weights.bin", offset = 0 : i64, size = 16 : i64} : () -> tensor<4xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %r = "onnx.Reshape"(%x, %g) : (tensor<1x8x4x4xf32>, tensor<3xi64>) -> tensor<1x4x32xf32>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) : (tensor<1x4x32xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x32xf32>
    %s = "onnx.Shape"(%x) : (tensor<1x8x4x4xf32>) -> tensor<4xi64>
    %y = "onnx.Reshape"(%n, %s) : (tensor<1x4x32xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
    return %y : tensor<1x8x4x4xf32>
  }

  // CHECK-LABEL: func.func @decomposed_external_scale
  // CHECK: hip.instance_norm
  // CHECK-NOT: hip.group_norm

  func.func @decomposed_wrong_restore(%x: tensor<1x8x4x4xf32>) -> tensor<1x8x16xf32> {
    %g = "onnx.Constant"() {value = dense<[0, 4, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %back = "onnx.Constant"() {value = dense<[1, 8, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %r = "onnx.Reshape"(%x, %g) : (tensor<1x8x4x4xf32>, tensor<3xi64>) -> tensor<1x4x32xf32>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) : (tensor<1x4x32xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x32xf32>
    %y = "onnx.Reshape"(%n, %back) : (tensor<1x4x32xf32>, tensor<3xi64>) -> tensor<1x8x16xf32>
    return %y : tensor<1x8x16xf32>
  }

  // CHECK-LABEL: func.func @decomposed_wrong_restore
  // CHECK: hip.instance_norm
  // CHECK-NOT: hip.group_norm

  func.func @decomposed_shared_intermediate(%x: tensor<1x8x4x4xf32>) -> (tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>) {
    %g = "onnx.Constant"() {value = dense<[0, 4, -1]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ones = "onnx.Constant"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %zeros = "onnx.Constant"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
    %beta = "onnx.Constant"() {value = dense<0.5> : tensor<8x1x1xf32>} : () -> tensor<8x1x1xf32>
    %r = "onnx.Reshape"(%x, %g) : (tensor<1x8x4x4xf32>, tensor<3xi64>) -> tensor<1x4x32xf32>
    %n = "onnx.InstanceNormalization"(%r, %ones, %zeros) : (tensor<1x4x32xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x32xf32>
    %s = "onnx.Shape"(%x) : (tensor<1x8x4x4xf32>) -> tensor<4xi64>
    %b = "onnx.Reshape"(%n, %s) : (tensor<1x4x32xf32>, tensor<4xi64>) -> tensor<1x8x4x4xf32>
    %y = "onnx.Add"(%b, %beta) : (tensor<1x8x4x4xf32>, tensor<8x1x1xf32>) -> tensor<1x8x4x4xf32>
    return %y, %b : tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>
  }

  // The Add is not fused (the restored tensor has a second use), but the
  // chain up to that tensor still is.
  // CHECK-LABEL: func.func @decomposed_shared_intermediate
  // CHECK-NOT: hip.instance_norm
  // CHECK: %[[N:.*]] = hip.group_norm
  // CHECK-SAME: groups = 4
  // CHECK: hip.add(%{{.*}}) ins(%[[N]]
}
