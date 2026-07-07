// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    return %arg0 : tensor<1x128x4096xf16>
  }

  func.func @rms_norm_static(
      %input: tensor<1x128x4096xf16>, %scale: tensor<4096xf16>)
      -> tensor<1x128x4096xf16> {
    %output = "onnx.RMSNormalization"(%input, %scale) {
      axis = -1 : si64,
      epsilon = 9.99999974E-6 : f32,
      stash_type = 1 : si64
    } : (tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> tensor<1x128x4096xf16>
    return %output : tensor<1x128x4096xf16>
  }

  // CHECK-LABEL: func.func @rms_norm_static
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context,
  // CHECK-SAME: %[[INPUT:.*]]: tensor<1x128x4096xf16>,
  // CHECK-SAME: %[[SCALE:.*]]: tensor<4096xf16>)
  // CHECK: tensor.empty() : tensor<1x128x4096xf16>
  // CHECK: hip.rms_norm(%[[CTX]]) ins(%[[INPUT]], %[[SCALE]] : tensor<1x128x4096xf16>, tensor<4096xf16>) outs({{.*}} : tensor<1x128x4096xf16>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}
  // CHECK-NOT: onnx.RMSNormalization

  func.func @rms_norm_defaults(
      %input: tensor<2x3xf32>, %scale: tensor<3xf32>) -> tensor<2x3xf32> {
    %output = "onnx.RMSNormalization"(%input, %scale)
        : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
    return %output : tensor<2x3xf32>
  }

  // CHECK-LABEL: func.func @rms_norm_defaults
  // CHECK: hip.rms_norm(%{{.*}}) ins({{.*}} : tensor<2x3xf32>, tensor<3xf32>) outs({{.*}} : tensor<2x3xf32>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64}

  func.func @rms_norm_dynamic(
      %input: tensor<?x?xf16>, %scale: tensor<?xf16>) -> tensor<?x?xf16> {
    %output = "onnx.RMSNormalization"(%input, %scale) {
      axis = -1 : si64,
      epsilon = 1.0e-05 : f32,
      stash_type = 1 : si64
    } : (tensor<?x?xf16>, tensor<?xf16>) -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }

  // CHECK-LABEL: func.func @rms_norm_dynamic
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[IN2:.*]]: tensor<?x?xf16>, %[[SC2:.*]]: tensor<?xf16>)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %{{.*}} = tensor.dim %[[IN2]], %[[C0]] : tensor<?x?xf16>
  // CHECK: %{{.*}} = tensor.dim %[[IN2]], %[[C1]] : tensor<?x?xf16>
  // CHECK: tensor.empty({{.*}}, {{.*}}) : tensor<?x?xf16>
  // CHECK: hip.rms_norm(%[[CTX2]]) ins(%[[IN2]], %[[SC2]] : tensor<?x?xf16>, tensor<?xf16>) outs({{.*}} : tensor<?x?xf16>) {axis = -1 : i64, epsilon = 1.000000e-05 : f32, stash_type = 1 : i64}
  // CHECK-NOT: onnx.RMSNormalization
}
