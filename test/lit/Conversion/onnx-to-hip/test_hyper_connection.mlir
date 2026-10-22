// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf16>) -> tensor<4xf16> {
    return %arg0 : tensor<4xf16>
  }

  // Flattened streams, sigmoid gate already applied, scale 1/4.
  // CHECK-LABEL: func.func @pre_mix_flattened_feature
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.hyper_connection_pre_mix({{.*}}) ins({{.*}} : tensor<?x?x10240xf16>, tensor<?x?x10240xf16>) outs({{.*}} : tensor<?x?x2560xf16>)
  // CHECK-SAME: branches = 4 : i64
  // CHECK-SAME: flattened = 1 : i64
  // CHECK-SAME: gate_layout = "feature"
  // CHECK-SAME: hidden = 2560 : i64
  // CHECK-SAME: reduction_scale = 2.500000e-01 : f32
  func.func @pre_mix_flattened_feature(%x: tensor<?x?x10240xf16>, %g: tensor<?x?x10240xf16>) -> tensor<?x?x2560xf16> {
    %y = "onnx.Custom"(%x, %g) {
      function_name = "HyperConnectionPreMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64,
      reduction_scale = 2.500000e-01 : f32
    } : (tensor<?x?x10240xf16>, tensor<?x?x10240xf16>) -> tensor<?x?x2560xf16>
    return %y : tensor<?x?x2560xf16>
  }

  // CHECK-LABEL: func.func @pre_mix_branch_gate
  // CHECK: hip.hyper_connection_pre_mix({{.*}}) ins({{.*}} : tensor<2x3x20xf32>, tensor<2x3x4xf32>)
  // CHECK-SAME: gate_layout = "branch"
  func.func @pre_mix_branch_gate(%x: tensor<2x3x20xf32>, %g: tensor<2x3x4xf32>) -> tensor<2x3x5xf32> {
    %y = "onnx.Custom"(%x, %g) {
      function_name = "HyperConnectionPreMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64
    } : (tensor<2x3x20xf32>, tensor<2x3x4xf32>) -> tensor<2x3x5xf32>
    return %y : tensor<2x3x5xf32>
  }

  // CHECK-LABEL: func.func @post_mix_flattened_branch
  // CHECK-NOT: onnx.Custom
  // CHECK: hip.hyper_connection_post_mix({{.*}}) ins({{.*}} : tensor<?x?x10240xf16>, tensor<?x?x2560xf16>, tensor<?x?x4xf16>) outs({{.*}} : tensor<?x?x10240xf16>)
  // CHECK-SAME: branches = 4 : i64
  // CHECK-SAME: gate_layout = "branch"
  // CHECK-SAME: hidden = 2560 : i64
  func.func @post_mix_flattened_branch(%x: tensor<?x?x10240xf16>, %b: tensor<?x?x2560xf16>, %g: tensor<?x?x4xf16>) -> tensor<?x?x10240xf16> {
    %y = "onnx.Custom"(%x, %b, %g) {
      function_name = "HyperConnectionPostMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64
    } : (tensor<?x?x10240xf16>, tensor<?x?x2560xf16>, tensor<?x?x4xf16>) -> tensor<?x?x10240xf16>
    return %y : tensor<?x?x10240xf16>
  }

  // CHECK-LABEL: func.func @post_mix_stream_mix
  // CHECK: hip.hyper_connection_post_mix({{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x3x4x8xf32>, tensor<2x3x8xf32>, tensor<2x3x4x8xf32>, tensor<2x3x4x4xf32>)
  // CHECK-SAME: flattened = 0 : i64
  // CHECK-SAME: gate_layout = "feature"
  func.func @post_mix_stream_mix(%x: tensor<2x3x4x8xf32>, %b: tensor<2x3x8xf32>, %g: tensor<2x3x4x8xf32>, %m: tensor<2x3x4x4xf32>) -> tensor<2x3x4x8xf32> {
    %y = "onnx.Custom"(%x, %b, %g, %m) {
      function_name = "HyperConnectionPostMix",
      domain_name = "com.microsoft"
    } : (tensor<2x3x4x8xf32>, tensor<2x3x8xf32>, tensor<2x3x4x8xf32>, tensor<2x3x4x4xf32>) -> tensor<2x3x4x8xf32>
    return %y : tensor<2x3x4x8xf32>
  }

  // CHECK-LABEL: func.func @pre_mix_sigmoid
  // CHECK-NOT: onnx.Sigmoid
  // CHECK: hip.hyper_connection_pre_mix({{.*}}) ins({{.*}} : tensor<?x?x10240xf16>, tensor<?x?x10240xf16>)
  // CHECK-SAME: gate_sigmoid = 1
  func.func @pre_mix_sigmoid(%x: tensor<?x?x10240xf16>, %g: tensor<?x?x10240xf16>) -> tensor<?x?x2560xf16> {
    %s = "onnx.Sigmoid"(%g) : (tensor<?x?x10240xf16>) -> tensor<?x?x10240xf16>
    %y = "onnx.Custom"(%x, %s) {
      function_name = "HyperConnectionPreMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64,
      reduction_scale = 2.500000e-01 : f32
    } : (tensor<?x?x10240xf16>, tensor<?x?x10240xf16>) -> tensor<?x?x2560xf16>
    return %y : tensor<?x?x2560xf16>
  }

  // CHECK-LABEL: func.func @post_mix_inject_norm
  // CHECK-NOT: onnx.Sigmoid
  // CHECK: hip.hyper_connection_post_mix_norm
  // CHECK-SAME: epsilon = 9.99999997E-7
  // CHECK: gate_activation = 1
  // CHECK: gate_layout = "branch"
  func.func @post_mix_inject_norm(%x: tensor<1x2x10240xf16>, %b: tensor<1x2x2560xf16>, %g: tensor<1x2x4xf16>, %scale: tensor<10240xf16>) -> (tensor<1x2x10240xf16>, tensor<1x2x10240xf16>) {
    %four = "onnx.Constant"() {value = dense<4.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %two = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %div = "onnx.Div"(%g, %four) : (tensor<1x2x4xf16>, tensor<f16>) -> tensor<1x2x4xf16>
    %sig = "onnx.Sigmoid"(%div) : (tensor<1x2x4xf16>) -> tensor<1x2x4xf16>
    %mul = "onnx.Mul"(%sig, %two) : (tensor<1x2x4xf16>, tensor<f16>) -> tensor<1x2x4xf16>
    %y = "onnx.Custom"(%x, %b, %mul) {
      function_name = "HyperConnectionPostMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64
    } : (tensor<1x2x10240xf16>, tensor<1x2x2560xf16>, tensor<1x2x4xf16>) -> tensor<1x2x10240xf16>
    %n = "onnx.Custom"(%y, %scale) {
      function_name = "BranchwiseRMSNorm",
      domain_name = "com.microsoft",
      epsilon = 9.99999997E-7 : f32,
      num_branches = 4 : si64
    } : (tensor<1x2x10240xf16>, tensor<10240xf16>) -> tensor<1x2x10240xf16>
    return %y, %n : tensor<1x2x10240xf16>, tensor<1x2x10240xf16>
  }

  // Grouped streams, feature gate already applied, plus stream_mix and the norm.
  // CHECK-LABEL: func.func @post_mix_feature_stream_mix_norm
  // CHECK: hip.hyper_connection_post_mix_norm
  // CHECK-SAME: tensor<2x3x4x4xf32>
  // CHECK-NOT: gate_activation
  // CHECK: gate_layout = "feature"
  func.func @post_mix_feature_stream_mix_norm(%x: tensor<2x3x4x8xf32>, %b: tensor<2x3x8xf32>, %g: tensor<2x3x4x8xf32>, %m: tensor<2x3x4x4xf32>, %scale: tensor<32xf32>) -> (tensor<2x3x4x8xf32>, tensor<2x3x4x8xf32>) {
    %y = "onnx.Custom"(%x, %b, %g, %m) {
      function_name = "HyperConnectionPostMix",
      domain_name = "com.microsoft"
    } : (tensor<2x3x4x8xf32>, tensor<2x3x8xf32>, tensor<2x3x4x8xf32>, tensor<2x3x4x4xf32>) -> tensor<2x3x4x8xf32>
    %n = "onnx.Custom"(%y, %scale) {
      function_name = "BranchwiseRMSNorm",
      domain_name = "com.microsoft",
      epsilon = 1.000000e-06 : f32,
      num_branches = 4 : si64
    } : (tensor<2x3x4x8xf32>, tensor<32xf32>) -> tensor<2x3x4x8xf32>
    return %y, %n : tensor<2x3x4x8xf32>, tensor<2x3x4x8xf32>
  }

  // Flattened streams, 2*sigmoid(x/4), stream_mix, and the norm.
  // CHECK-LABEL: func.func @post_mix_stream_mix_inject_norm
  // CHECK-NOT: onnx.Sigmoid
  // CHECK: hip.hyper_connection_post_mix_norm
  // CHECK-SAME: tensor<1x2x4x4xf16>
  // CHECK: gate_activation = 1
  // CHECK: gate_layout = "branch"
  func.func @post_mix_stream_mix_inject_norm(%x: tensor<1x2x10240xf16>, %b: tensor<1x2x2560xf16>, %g: tensor<1x2x4xf16>, %m: tensor<1x2x4x4xf16>, %scale: tensor<10240xf16>) -> (tensor<1x2x10240xf16>, tensor<1x2x10240xf16>) {
    %four = "onnx.Constant"() {value = dense<4.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %two = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %div = "onnx.Div"(%g, %four) : (tensor<1x2x4xf16>, tensor<f16>) -> tensor<1x2x4xf16>
    %sig = "onnx.Sigmoid"(%div) : (tensor<1x2x4xf16>) -> tensor<1x2x4xf16>
    %mul = "onnx.Mul"(%sig, %two) : (tensor<1x2x4xf16>, tensor<f16>) -> tensor<1x2x4xf16>
    %y = "onnx.Custom"(%x, %b, %mul, %m) {
      function_name = "HyperConnectionPostMix",
      domain_name = "com.microsoft",
      num_branches = 4 : si64
    } : (tensor<1x2x10240xf16>, tensor<1x2x2560xf16>, tensor<1x2x4xf16>, tensor<1x2x4x4xf16>) -> tensor<1x2x10240xf16>
    %n = "onnx.Custom"(%y, %scale) {
      function_name = "BranchwiseRMSNorm",
      domain_name = "com.microsoft",
      epsilon = 9.99999997E-7 : f32,
      num_branches = 4 : si64
    } : (tensor<1x2x10240xf16>, tensor<10240xf16>) -> tensor<1x2x10240xf16>
    return %y, %n : tensor<1x2x10240xf16>, tensor<1x2x10240xf16>
  }
}
