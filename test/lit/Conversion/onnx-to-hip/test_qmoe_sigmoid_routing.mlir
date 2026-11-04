// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// A sigmoid + correction-bias router built outside com.microsoft QMoE
// (TopK over sigmoid + bias, the weights s / sum * c, and two ScatterElements
// that feed router_probs / router_weights) folds into
// hip.qmoe {routing_type = "sigmoid_bias"} over the fp32 logits, and the
// routing chain is erased. Any deviation from that exact chain declines the
// fold and keeps the plain router_weights conversion.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%x: tensor<1x8x128xf16>) -> tensor<1x8x128xf16> {
    return %x : tensor<1x8x128xf16>
  }

  // ===== Test 1: the full chain folds =====
  func.func @fold(%x: tensor<1x8x128xf16>, %logits: tensor<1x8x16xf32>) -> tensor<1x8x128xf16> {
    %s = "onnx.Sigmoid"(%logits) : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %bias = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16xf32>
    %sel = "onnx.Add"(%s, %bias) : (tensor<1x8x16xf32>, tensor<16xf32>) -> tensor<1x8x16xf32>
    %k = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %v, %idx = "onnx.TopK"(%sel, %k) {axis = -1 : si64, largest = 1 : si64, sorted = 1 : si64}
        : (tensor<1x8x16xf32>, tensor<1xi64>) -> (tensor<1x8x4xf32>, tensor<1x8x4xi64>)
    %g = "onnx.GatherElements"(%s, %idx) {axis = -1 : si64}
        : (tensor<1x8x16xf32>, tensor<1x8x4xi64>) -> tensor<1x8x4xf32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sum = "onnx.ReduceSum"(%g, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<1x8x4xf32>, tensor<1xi64>) -> tensor<1x8x1xf32>
    %d = "onnx.Div"(%g, %sum) : (tensor<1x8x4xf32>, tensor<1x8x1xf32>) -> tensor<1x8x4xf32>
    %c = "onnx.Constant"() {value = dense<3.125000e-01> : tensor<f32>} : () -> tensor<f32>
    %m = "onnx.Mul"(%d, %c) : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<1x8x4xf32>
    %w = "onnx.Cast"(%m) {saturate = 1 : si64, to = f16} : (tensor<1x8x4xf32>) -> tensor<1x8x4xf16>
    %ninf = "onnx.Constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %zero = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %shp = "onnx.Constant"() {value = dense<[1, 8, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ishp = "onnx.Shape"(%idx) : (tensor<1x8x4xi64>) -> tensor<3xi64>
    %pfill = "onnx.Expand"(%ninf, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %pupd = "onnx.Expand"(%one, %ishp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x4xf16>
    %rp = "onnx.ScatterElements"(%pfill, %idx, %pupd) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %wfill = "onnx.Expand"(%zero, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %rw = "onnx.ScatterElements"(%wfill, %idx, %w) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %rs = "onnx.Constant"() {value = dense<[-1, 16]> : tensor<2xi64>} : () -> tensor<2xi64>
    %rp2 = "onnx.Reshape"(%rp, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %rw2 = "onnx.Reshape"(%rw, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x64xui8>} : () -> tensor<16x128x64xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x32xui8>} : () -> tensor<16x128x32xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Custom"(%x, %rp2, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %rw2) {
      function_name = "QMoE", domain_name = "com.microsoft",
      activation_type = "swiglu", block_size = 128 : si64,
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 0 : si64, swiglu_fusion = 1 : si64,
      weights_prepacked = 0 : si64, onnx_node_name = "QMoE_sigmoid"
    } : (tensor<1x8x128xf16>, tensor<8x16xf16>,
         tensor<16x128x64xui8>, tensor<16x128x1xf16>, none,
         tensor<16x128x32xui8>, tensor<16x128x1xf16>,
         none, none, none, none, none, none, none, tensor<8x16xf16>) -> tensor<1x8x128xf16>
    return %y : tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @fold
  // CHECK-SAME: %[[LOGITS:[^:]+]]: tensor<1x8x16xf32>
  // CHECK-NOT: onnx.TopK
  // CHECK-NOT: onnx.ScatterElements
  // CHECK: hip.qmoe(%{{.*}}) ins(%{{[^,]+}}, %[[LOGITS]],
  // CHECK-SAME: tensor<1x8x128xf16>, tensor<1x8x16xf32>,
  // CHECK-NOT: router_weights
  // CHECK-SAME: correction_bias(%{{.*}} : tensor<16xf32>)
  // CHECK-SAME: k = 4
  // CHECK-SAME: normalize_routing_weights = 1
  // CHECK-SAME: routed_scaling_factor = 3.125000e-01 : f32
  // CHECK-SAME: routing_type = "sigmoid_bias"
  // CHECK-NOT: onnx.TopK
  // CHECK-NOT: onnx.ScatterElements
  // CHECK: return

  // ===== Test 2: TopK k differs from the QMoE k: no fold =====
  func.func @k_mismatch(%x: tensor<1x8x128xf16>, %logits: tensor<1x8x16xf32>) -> tensor<1x8x128xf16> {
    %s = "onnx.Sigmoid"(%logits) : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %bias = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16xf32>
    %sel = "onnx.Add"(%s, %bias) : (tensor<1x8x16xf32>, tensor<16xf32>) -> tensor<1x8x16xf32>
    %k = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
    %v, %idx = "onnx.TopK"(%sel, %k) {axis = -1 : si64, largest = 1 : si64, sorted = 1 : si64}
        : (tensor<1x8x16xf32>, tensor<1xi64>) -> (tensor<1x8x3xf32>, tensor<1x8x3xi64>)
    %g = "onnx.GatherElements"(%s, %idx) {axis = -1 : si64}
        : (tensor<1x8x16xf32>, tensor<1x8x3xi64>) -> tensor<1x8x3xf32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sum = "onnx.ReduceSum"(%g, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<1x8x3xf32>, tensor<1xi64>) -> tensor<1x8x1xf32>
    %d = "onnx.Div"(%g, %sum) : (tensor<1x8x3xf32>, tensor<1x8x1xf32>) -> tensor<1x8x3xf32>
    %c = "onnx.Constant"() {value = dense<3.125000e-01> : tensor<f32>} : () -> tensor<f32>
    %m = "onnx.Mul"(%d, %c) : (tensor<1x8x3xf32>, tensor<f32>) -> tensor<1x8x3xf32>
    %w = "onnx.Cast"(%m) {saturate = 1 : si64, to = f16} : (tensor<1x8x3xf32>) -> tensor<1x8x3xf16>
    %ninf = "onnx.Constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %zero = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %shp = "onnx.Constant"() {value = dense<[1, 8, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ishp = "onnx.Constant"() {value = dense<[1, 8, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
    %pfill = "onnx.Expand"(%ninf, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %pupd = "onnx.Expand"(%one, %ishp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x3xf16>
    %rp = "onnx.ScatterElements"(%pfill, %idx, %pupd) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x3xi64>, tensor<1x8x3xf16>) -> tensor<1x8x16xf16>
    %wfill = "onnx.Expand"(%zero, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %rw = "onnx.ScatterElements"(%wfill, %idx, %w) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x3xi64>, tensor<1x8x3xf16>) -> tensor<1x8x16xf16>
    %rs = "onnx.Constant"() {value = dense<[-1, 16]> : tensor<2xi64>} : () -> tensor<2xi64>
    %rp2 = "onnx.Reshape"(%rp, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %rw2 = "onnx.Reshape"(%rw, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x64xui8>} : () -> tensor<16x128x64xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x32xui8>} : () -> tensor<16x128x32xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Custom"(%x, %rp2, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %rw2) {
      function_name = "QMoE", domain_name = "com.microsoft",
      activation_type = "swiglu", block_size = 128 : si64,
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 0 : si64, swiglu_fusion = 1 : si64,
      weights_prepacked = 0 : si64, onnx_node_name = "QMoE_k_mismatch"
    } : (tensor<1x8x128xf16>, tensor<8x16xf16>,
         tensor<16x128x64xui8>, tensor<16x128x1xf16>, none,
         tensor<16x128x32xui8>, tensor<16x128x1xf16>,
         none, none, none, none, none, none, none, tensor<8x16xf16>) -> tensor<1x8x128xf16>
    return %y : tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @k_mismatch
  // CHECK: hip.qmoe
  // CHECK-SAME: tensor<8x16xf16>
  // CHECK-SAME: router_weights(%{{.*}} : tensor<8x16xf16>)
  // CHECK-NOT: sigmoid_bias
  // CHECK: return

  // ===== Test 3: weights gathered from the biased score: no fold =====
  func.func @biased_weights(%x: tensor<1x8x128xf16>, %logits: tensor<1x8x16xf32>) -> tensor<1x8x128xf16> {
    %s = "onnx.Sigmoid"(%logits) : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %bias = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16xf32>
    %sel = "onnx.Add"(%s, %bias) : (tensor<1x8x16xf32>, tensor<16xf32>) -> tensor<1x8x16xf32>
    %k = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %v, %idx = "onnx.TopK"(%sel, %k) {axis = -1 : si64, largest = 1 : si64, sorted = 1 : si64}
        : (tensor<1x8x16xf32>, tensor<1xi64>) -> (tensor<1x8x4xf32>, tensor<1x8x4xi64>)
    %g = "onnx.GatherElements"(%sel, %idx) {axis = -1 : si64}
        : (tensor<1x8x16xf32>, tensor<1x8x4xi64>) -> tensor<1x8x4xf32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sum = "onnx.ReduceSum"(%g, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<1x8x4xf32>, tensor<1xi64>) -> tensor<1x8x1xf32>
    %d = "onnx.Div"(%g, %sum) : (tensor<1x8x4xf32>, tensor<1x8x1xf32>) -> tensor<1x8x4xf32>
    %c = "onnx.Constant"() {value = dense<3.125000e-01> : tensor<f32>} : () -> tensor<f32>
    %m = "onnx.Mul"(%d, %c) : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<1x8x4xf32>
    %w = "onnx.Cast"(%m) {saturate = 1 : si64, to = f16} : (tensor<1x8x4xf32>) -> tensor<1x8x4xf16>
    %ninf = "onnx.Constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %zero = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %shp = "onnx.Constant"() {value = dense<[1, 8, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ishp = "onnx.Shape"(%idx) : (tensor<1x8x4xi64>) -> tensor<3xi64>
    %pfill = "onnx.Expand"(%ninf, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %pupd = "onnx.Expand"(%one, %ishp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x4xf16>
    %rp = "onnx.ScatterElements"(%pfill, %idx, %pupd) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %wfill = "onnx.Expand"(%zero, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %rw = "onnx.ScatterElements"(%wfill, %idx, %w) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %rs = "onnx.Constant"() {value = dense<[-1, 16]> : tensor<2xi64>} : () -> tensor<2xi64>
    %rp2 = "onnx.Reshape"(%rp, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %rw2 = "onnx.Reshape"(%rw, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x64xui8>} : () -> tensor<16x128x64xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x32xui8>} : () -> tensor<16x128x32xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Custom"(%x, %rp2, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %rw2) {
      function_name = "QMoE", domain_name = "com.microsoft",
      activation_type = "swiglu", block_size = 128 : si64,
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 0 : si64, swiglu_fusion = 1 : si64,
      weights_prepacked = 0 : si64, onnx_node_name = "QMoE_biased_weights"
    } : (tensor<1x8x128xf16>, tensor<8x16xf16>,
         tensor<16x128x64xui8>, tensor<16x128x1xf16>, none,
         tensor<16x128x32xui8>, tensor<16x128x1xf16>,
         none, none, none, none, none, none, none, tensor<8x16xf16>) -> tensor<1x8x128xf16>
    return %y : tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @biased_weights
  // CHECK: hip.qmoe
  // CHECK-SAME: router_weights(%{{.*}} : tensor<8x16xf16>)
  // CHECK-NOT: sigmoid_bias
  // CHECK: return

  // ===== Test 4: normalize_routing_weights = 1 renormalizes: no fold =====
  func.func @normalized(%x: tensor<1x8x128xf16>, %logits: tensor<1x8x16xf32>) -> tensor<1x8x128xf16> {
    %s = "onnx.Sigmoid"(%logits) : (tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
    %bias = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<16xf32>} : () -> tensor<16xf32>
    %sel = "onnx.Add"(%s, %bias) : (tensor<1x8x16xf32>, tensor<16xf32>) -> tensor<1x8x16xf32>
    %k = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %v, %idx = "onnx.TopK"(%sel, %k) {axis = -1 : si64, largest = 1 : si64, sorted = 1 : si64}
        : (tensor<1x8x16xf32>, tensor<1xi64>) -> (tensor<1x8x4xf32>, tensor<1x8x4xi64>)
    %g = "onnx.GatherElements"(%s, %idx) {axis = -1 : si64}
        : (tensor<1x8x16xf32>, tensor<1x8x4xi64>) -> tensor<1x8x4xf32>
    %axes = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %sum = "onnx.ReduceSum"(%g, %axes) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
        : (tensor<1x8x4xf32>, tensor<1xi64>) -> tensor<1x8x1xf32>
    %d = "onnx.Div"(%g, %sum) : (tensor<1x8x4xf32>, tensor<1x8x1xf32>) -> tensor<1x8x4xf32>
    %c = "onnx.Constant"() {value = dense<3.125000e-01> : tensor<f32>} : () -> tensor<f32>
    %m = "onnx.Mul"(%d, %c) : (tensor<1x8x4xf32>, tensor<f32>) -> tensor<1x8x4xf32>
    %w = "onnx.Cast"(%m) {saturate = 1 : si64, to = f16} : (tensor<1x8x4xf32>) -> tensor<1x8x4xf16>
    %ninf = "onnx.Constant"() {value = dense<0xFC00> : tensor<f16>} : () -> tensor<f16>
    %one = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %zero = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f16>} : () -> tensor<f16>
    %shp = "onnx.Constant"() {value = dense<[1, 8, 16]> : tensor<3xi64>} : () -> tensor<3xi64>
    %ishp = "onnx.Shape"(%idx) : (tensor<1x8x4xi64>) -> tensor<3xi64>
    %pfill = "onnx.Expand"(%ninf, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %pupd = "onnx.Expand"(%one, %ishp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x4xf16>
    %rp = "onnx.ScatterElements"(%pfill, %idx, %pupd) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %wfill = "onnx.Expand"(%zero, %shp) : (tensor<f16>, tensor<3xi64>) -> tensor<1x8x16xf16>
    %rw = "onnx.ScatterElements"(%wfill, %idx, %w) {axis = -1 : si64}
        : (tensor<1x8x16xf16>, tensor<1x8x4xi64>, tensor<1x8x4xf16>) -> tensor<1x8x16xf16>
    %rs = "onnx.Constant"() {value = dense<[-1, 16]> : tensor<2xi64>} : () -> tensor<2xi64>
    %rp2 = "onnx.Reshape"(%rp, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %rw2 = "onnx.Reshape"(%rw, %rs) {allowzero = 0 : si64} : (tensor<1x8x16xf16>, tensor<2xi64>) -> tensor<8x16xf16>
    %fc1_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x64xui8>} : () -> tensor<16x128x64xui8>
    %fc1_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %fc2_w = "onnx.Constant"() {value = dense<1> : tensor<16x128x32xui8>} : () -> tensor<16x128x32xui8>
    %fc2_s = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x128x1xf16>} : () -> tensor<16x128x1xf16>
    %none = "onnx.NoValue"() {value} : () -> none
    %y = "onnx.Custom"(%x, %rp2, %fc1_w, %fc1_s, %none, %fc2_w, %fc2_s,
                       %none, %none, %none, %none, %none, %none, %none, %rw2) {
      function_name = "QMoE", domain_name = "com.microsoft",
      activation_type = "swiglu", block_size = 128 : si64,
      expert_weight_bits = 4 : si64, k = 4 : si64,
      normalize_routing_weights = 1 : si64, swiglu_fusion = 1 : si64,
      weights_prepacked = 0 : si64, onnx_node_name = "QMoE_normalized"
    } : (tensor<1x8x128xf16>, tensor<8x16xf16>,
         tensor<16x128x64xui8>, tensor<16x128x1xf16>, none,
         tensor<16x128x32xui8>, tensor<16x128x1xf16>,
         none, none, none, none, none, none, none, tensor<8x16xf16>) -> tensor<1x8x128xf16>
    return %y : tensor<1x8x128xf16>
  }

  // CHECK-LABEL: func.func @normalized
  // CHECK: hip.qmoe
  // CHECK-SAME: router_weights(%{{.*}} : tensor<8x16xf16>)
  // CHECK-NOT: sigmoid_bias
  // CHECK: return
}
