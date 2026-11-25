// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s --check-prefix=HIP
// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s --check-prefix=LLVM
// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s --check-prefix=INTERFACE

// HIP-LABEL: func.func @main_graph
// HIP: hip.paged_attention
// HIP-SAME: metadata(
// HIP-SAME: outs(%{{.*}}, %{{.*}}, %{{.*}}
// LLVM-DAG: llvm.func @hipdnn_ep_bind_output
// LLVM-DAG: llvm.func @wrap_paged_attention_v2
// LLVM-LABEL: llvm.func private @main_graph_internal
// LLVM: llvm.call @hipdnn_ep_bind_output
// LLVM: llvm.call @hipdnn_ep_bind_output
// LLVM: llvm.call @wrap_paged_attention_v2
// LLVM-NOT: llvm.call @hipdnn_ep_memcpy
// LLVM: llvm.return

// Full inference interface follows the existing GQA/MatMulNBits/QMoE path.
// INTERFACE: module attributes {
// INTERFACE-SAME: hipdnn.input_count = 9
// INTERFACE-SAME: hipdnn.output_count = 3
// INTERFACE-DAG: llvm.call @hipdnn_ep_op_states_alloc(
// INTERFACE-DAG: llvm.func @hipdnn_ep_op_states_init_fn(
// INTERFACE-DAG: llvm.call @hipdnn_ep_op_state_construct_paged_attention(
// INTERFACE-DAG: llvm.func @inference_init(
// INTERFACE-DAG: llvm.call @hipdnn_ep_op_states_init_fn(
// INTERFACE-DAG: llvm.func @inference_compute(
// INTERFACE-DAG: llvm.call @hipdnn_ep_state_reset_error_flag(
// INTERFACE-DAG: llvm.call @main_graph(
// INTERFACE-DAG: llvm.call @main_graph_internal(
// INTERFACE-DAG: llvm.call @hipdnn_ep_state_read_and_clear_error_flag(
// INTERFACE-DAG: llvm.func @inference_cleanup(
// INTERFACE-DAG: llvm.call @hipdnn_ep_state_cleanup(
// INTERFACE-DAG: llvm.func @inference_get_metadata_json(

module {
  func.func @main_graph(
      %q: tensor<?x6144xf16>, %k: tensor<?x1024xf16>, %v: tensor<?x1024xf16>,
      %kp: tensor<?x?x4x256xf16>, %vp: tensor<?x?x4x256xf16>,
      %cum: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>,
      %metadata: tensor<3xi32>) -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %y, %pk, %pv = "onnx.Custom"(%q, %k, %v, %kp, %vp, %cum, %past, %table,
        %none, %none, %none, %none, %none, %none, %none, %none, %metadata) {
      function_name = "PagedAttention", domain_name = "com.microsoft",
      num_heads = 24 : si64, kv_num_heads = 4 : si64, scale = 0.0625 : f32,
      local_window_size = -1 : si64, softcap = 0.0 : f32,
      do_rotary = 0 : si64, rotary_interleaved = 0 : si64,
      k_cache_dtype = "\00", v_cache_dtype = "\00"
    } : (tensor<?x6144xf16>, tensor<?x1024xf16>, tensor<?x1024xf16>,
      tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>, tensor<?xi32>, tensor<?xi32>,
      tensor<?x?xi32>, none, none, none, none, none, none, none, none, tensor<3xi32>)
      -> (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>)
    "onnx.Return"(%y, %pk, %pv) : (tensor<?x6144xf16>, tensor<?x?x4x256xf16>, tensor<?x?x4x256xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
