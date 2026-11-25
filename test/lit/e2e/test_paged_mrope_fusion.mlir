// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s
// CHECK-DAG: llvm.func @wrap_paged_attention_mrope
// CHECK-DAG: llvm.func @hipdnn_ep_op_state_construct_paged_attention
// CHECK-LABEL: llvm.func private @main_graph_internal
// CHECK-NOT: llvm.call @wrap_mrotary_embedding
// CHECK-NOT: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_bind_output
// CHECK: llvm.call @hipdnn_ep_bind_output
// CHECK: llvm.call @wrap_paged_attention_mrope
// CHECK-NOT: llvm.call @hipdnn_ep_memcpy
// CHECK: llvm.return
// CHECK: llvm.func @inference_compute
// CHECK: llvm.call @hipdnn_ep_state_read_and_clear_error_flag
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
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
