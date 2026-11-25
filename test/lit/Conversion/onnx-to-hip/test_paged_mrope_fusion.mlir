// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.mrotary_embedding
// CHECK: hip.paged_mrope_attention
// CHECK-SAME: rotary(
// CHECK-SAME: tensor<8192x32xf32>, tensor<8192x32xf32>, tensor<3x?xi64>
// CHECK-SAME: mrope_layout = 1
// CHECK-SAME: mrope_section = [11, 11, 10]
// CHECK-SAME: rotary_dim = 64
// CHECK-NOT: hip.mrotary_embedding
// CHECK-LABEL: func.func @model_static
// CHECK-NOT: hip.mrotary_embedding
// CHECK: hip.paged_mrope_attention
// Layout 0 is the elided default.
// CHECK-SAME: rotary_dim = 64
// CHECK-NOT: mrope_layout = 1
// CHECK-LABEL: func.func @extra_use
// CHECK: hip.mrotary_embedding
// CHECK: hip.paged_attention
// CHECK-NOT: hip.paged_mrope_attention
// CHECK-LABEL: func.func @different_positions
// CHECK: hip.mrotary_embedding
// CHECK: hip.paged_attention
// CHECK-NOT: hip.paged_mrope_attention
module {
  func.func @main_graph(%q: tensor<?x6144xf32>, %k: tensor<?x1024xf32>, %v: tensor<?x1024xf16>, %kp: tensor<?x?x4x256xf16>, %vp: tensor<?x?x4x256xf16>, %cum: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>, %co: tensor<8192x32xf32>, %si: tensor<8192x32xf32>, %pos: tensor<3x?xi64>, %other: tensor<3x?xi64>, %metadata: tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %axis0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %qin = "onnx.Unsqueeze"(%q, %axis0) : (tensor<?x6144xf32>, tensor<1xi64>) -> tensor<1x?x6144xf32>
    %qpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %qmr = "onnx.Custom"(%qin, %qpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 24 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x6144xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x6144xf32>
    %qflat = "onnx.Squeeze"(%qmr, %axis0) : (tensor<1x?x6144xf32>, tensor<1xi64>) -> tensor<?x6144xf32>
    %q16 = "onnx.Cast"(%qflat) {to = 10 : si64} : (tensor<?x6144xf32>) -> tensor<?x6144xf16>
    %kin = "onnx.Unsqueeze"(%k, %axis0) : (tensor<?x1024xf32>, tensor<1xi64>) -> tensor<1x?x1024xf32>
    %kpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %kmr = "onnx.Custom"(%kin, %kpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 4 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x1024xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x1024xf32>
    %kflat = "onnx.Squeeze"(%kmr, %axis0) : (tensor<1x?x1024xf32>, tensor<1xi64>) -> tensor<?x1024xf32>
    %k16 = "onnx.Cast"(%kflat) {to = 10 : si64} : (tensor<?x1024xf32>) -> tensor<?x1024xf16>
    %y, %pk, %pv = "onnx.Custom"(%q16, %k16, %v, %kp, %vp, %cum, %past, %table, %none, %none, %none, %none, %none, %none, %none, %none, %metadata) {function_name = "PagedAttention", domain_name = "com.microsoft", num_heads = 24 : si64, kv_num_heads = 4 : si64, scale = 0.0625 : f32, do_rotary = 0 : si64, rotary_interleaved = 0 : si64, local_window_size = -1 : si64, softcap = 0.0 : f32} : (tensor<?x6144xf16>, tensor<?x1024xf16>, tensor<?x1024xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, none, none, none, none, none, none, none, none, tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>)
    "onnx.Return"(%y, %pk, %pv) : (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) -> ()
  }
  func.func @model_static(%q: tensor<4x6144xf32>, %k: tensor<4x1024xf32>, %v: tensor<4x1024xf16>, %kp: tensor<?x?x4x256xf16>, %vp: tensor<?x?x4x256xf16>, %cum: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>, %co: tensor<8192x32xf32>, %si: tensor<8192x32xf32>, %pos: tensor<3x4xi64>, %other: tensor<3x4xi64>, %metadata: tensor<3xi32>) -> (tensor<4x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %axis0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %qin = "onnx.Unsqueeze"(%q, %axis0) : (tensor<4x6144xf32>, tensor<1xi64>) -> tensor<1x4x6144xf32>
    %qpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x4xi64>, tensor<1xi64>) -> tensor<3x1x4xi64>
    %qmr = "onnx.Custom"(%qin, %qpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 24 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 0 : si64, scale = 1.0 : f32} : (tensor<1x4x6144xf32>, tensor<3x1x4xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x4x6144xf32>
    %qflat = "onnx.Squeeze"(%qmr, %axis0) : (tensor<1x4x6144xf32>, tensor<1xi64>) -> tensor<4x6144xf32>
    %q16 = "onnx.Cast"(%qflat) {to = 10 : si64} : (tensor<4x6144xf32>) -> tensor<4x6144xf16>
    %kin = "onnx.Unsqueeze"(%k, %axis0) : (tensor<4x1024xf32>, tensor<1xi64>) -> tensor<1x4x1024xf32>
    %kpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x4xi64>, tensor<1xi64>) -> tensor<3x1x4xi64>
    %kmr = "onnx.Custom"(%kin, %kpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 4 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 0 : si64, scale = 1.0 : f32} : (tensor<1x4x1024xf32>, tensor<3x1x4xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x4x1024xf32>
    %kflat = "onnx.Squeeze"(%kmr, %axis0) : (tensor<1x4x1024xf32>, tensor<1xi64>) -> tensor<4x1024xf32>
    %k16 = "onnx.Cast"(%kflat) {to = 10 : si64} : (tensor<4x1024xf32>) -> tensor<4x1024xf16>
    %y, %pk, %pv = "onnx.Custom"(%q16, %k16, %v, %kp, %vp, %cum, %past, %table, %none, %none, %none, %none, %none, %none, %none, %none, %metadata) {function_name = "PagedAttention", domain_name = "com.microsoft", num_heads = 24 : si64, kv_num_heads = 4 : si64, scale = 0.0625 : f32, do_rotary = 0 : si64, rotary_interleaved = 0 : si64, local_window_size = -1 : si64, softcap = 0.0 : f32} : (tensor<4x6144xf16>, tensor<4x1024xf16>, tensor<4x1024xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, none, none, none, none, none, none, none, none, tensor<3xi32>) -> (tensor<4x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>)
    "onnx.Return"(%y, %pk, %pv) : (tensor<4x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) -> ()
  }
  func.func @extra_use(%q: tensor<?x6144xf32>, %k: tensor<?x1024xf32>, %v: tensor<?x1024xf16>, %kp: tensor<?x?x4x256xf16>, %vp: tensor<?x?x4x256xf16>, %cum: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>, %co: tensor<8192x32xf32>, %si: tensor<8192x32xf32>, %pos: tensor<3x?xi64>, %other: tensor<3x?xi64>, %metadata: tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?x6144xf32>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %axis0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %qin = "onnx.Unsqueeze"(%q, %axis0) : (tensor<?x6144xf32>, tensor<1xi64>) -> tensor<1x?x6144xf32>
    %qpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %qmr = "onnx.Custom"(%qin, %qpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 24 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x6144xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x6144xf32>
    %qflat = "onnx.Squeeze"(%qmr, %axis0) : (tensor<1x?x6144xf32>, tensor<1xi64>) -> tensor<?x6144xf32>
    %q16 = "onnx.Cast"(%qflat) {to = 10 : si64} : (tensor<?x6144xf32>) -> tensor<?x6144xf16>
    %kin = "onnx.Unsqueeze"(%k, %axis0) : (tensor<?x1024xf32>, tensor<1xi64>) -> tensor<1x?x1024xf32>
    %kpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %kmr = "onnx.Custom"(%kin, %kpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 4 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x1024xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x1024xf32>
    %kflat = "onnx.Squeeze"(%kmr, %axis0) : (tensor<1x?x1024xf32>, tensor<1xi64>) -> tensor<?x1024xf32>
    %k16 = "onnx.Cast"(%kflat) {to = 10 : si64} : (tensor<?x1024xf32>) -> tensor<?x1024xf16>
    %y, %pk, %pv = "onnx.Custom"(%q16, %k16, %v, %kp, %vp, %cum, %past, %table, %none, %none, %none, %none, %none, %none, %none, %none, %metadata) {function_name = "PagedAttention", domain_name = "com.microsoft", num_heads = 24 : si64, kv_num_heads = 4 : si64, scale = 0.0625 : f32, do_rotary = 0 : si64, rotary_interleaved = 0 : si64, local_window_size = -1 : si64, softcap = 0.0 : f32} : (tensor<?x6144xf16>, tensor<?x1024xf16>, tensor<?x1024xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, none, none, none, none, none, none, none, none, tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>)
    "onnx.Return"(%y, %pk, %pv, %qflat) : (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?x6144xf32>) -> ()
  }
  func.func @different_positions(%q: tensor<?x6144xf32>, %k: tensor<?x1024xf32>, %v: tensor<?x1024xf16>, %kp: tensor<?x?x4x256xf16>, %vp: tensor<?x?x4x256xf16>, %cum: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>, %co: tensor<8192x32xf32>, %si: tensor<8192x32xf32>, %pos: tensor<3x?xi64>, %other: tensor<3x?xi64>, %metadata: tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %axis0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %axis1 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %qin = "onnx.Unsqueeze"(%q, %axis0) : (tensor<?x6144xf32>, tensor<1xi64>) -> tensor<1x?x6144xf32>
    %qpos = "onnx.Unsqueeze"(%pos, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %qmr = "onnx.Custom"(%qin, %qpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 24 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x6144xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x6144xf32>
    %qflat = "onnx.Squeeze"(%qmr, %axis0) : (tensor<1x?x6144xf32>, tensor<1xi64>) -> tensor<?x6144xf32>
    %q16 = "onnx.Cast"(%qflat) {to = 10 : si64} : (tensor<?x6144xf32>) -> tensor<?x6144xf16>
    %kin = "onnx.Unsqueeze"(%k, %axis0) : (tensor<?x1024xf32>, tensor<1xi64>) -> tensor<1x?x1024xf32>
    %kpos = "onnx.Unsqueeze"(%other, %axis1) : (tensor<3x?xi64>, tensor<1xi64>) -> tensor<3x1x?xi64>
    %kmr = "onnx.Custom"(%kin, %kpos, %co, %si) {function_name = "MRotaryEmbedding", domain_name = "com.microsoft", num_heads = 4 : si64, rotary_embedding_dim = 64 : si64, interleaved = 0 : si64, is_packed_batching = 1 : si64, mrope_section = [11 : si64, 11 : si64, 10 : si64], mrope_layout = 1 : si64, scale = 1.0 : f32} : (tensor<1x?x1024xf32>, tensor<3x1x?xi64>, tensor<8192x32xf32>, tensor<8192x32xf32>) -> tensor<1x?x1024xf32>
    %kflat = "onnx.Squeeze"(%kmr, %axis0) : (tensor<1x?x1024xf32>, tensor<1xi64>) -> tensor<?x1024xf32>
    %k16 = "onnx.Cast"(%kflat) {to = 10 : si64} : (tensor<?x1024xf32>) -> tensor<?x1024xf16>
    %y, %pk, %pv = "onnx.Custom"(%q16, %k16, %v, %kp, %vp, %cum, %past, %table, %none, %none, %none, %none, %none, %none, %none, %none, %metadata) {function_name = "PagedAttention", domain_name = "com.microsoft", num_heads = 24 : si64, kv_num_heads = 4 : si64, scale = 0.0625 : f32, do_rotary = 0 : si64, rotary_interleaved = 0 : si64, local_window_size = -1 : si64, softcap = 0.0 : f32} : (tensor<?x6144xf16>, tensor<?x1024xf16>, tensor<?x1024xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>, none, none, none, none, none, none, none, none, tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>)
    "onnx.Return"(%y, %pk, %pv) : (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) -> ()
  }
}
