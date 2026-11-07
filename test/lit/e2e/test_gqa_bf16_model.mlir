// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | hip-mlir-opt --cse | FileCheck %s

// The test_gqa_model.mlir decode step with bf16 Q/K/V, KV cache and output.
// A bf16 GroupQueryAttention lowers through the full pipeline to the original
// GQA entry (never the fp16-only Ring entry) and passes the query element type
// last, data_type=2 (HIPDNN_EP_DATATYPE_BFLOAT16): element_size_bytes is 2 for
// fp16 too, and the runtime picks bf16 GEMMs and softmax from it. --cse leaves
// one constant per value, so the first constant 2 is the operand.

// CHECK: llvm.func @wrap_group_query_attention_host_length({{.*}}, !llvm.ptr, i64, i64, i64) -> i32
// CHECK: llvm.func private @main_graph(
// CHECK: %[[DTYPE:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[DTYPE]]) :
// CHECK-NOT: wrap_group_query_attention_ring_f16
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x1x4096xbf16> {onnx.name = "/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0"}, %arg1: tensor<1x1x1024xbf16> {onnx.name = "/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0"}, %arg2: tensor<1x1x1024xbf16> {onnx.name = "/model/layers.0/attn/v_proj/MatMul/output_0"}, %arg3: tensor<1x8x127x128xbf16> {onnx.name = "past_key_values.0.key"}, %arg4: tensor<1x8x127x128xbf16> {onnx.name = "past_key_values.0.value"}, %arg5: tensor<1x1xi32> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0"}, %arg6: tensor<i32> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"}) -> (tensor<1x1x4096xbf16> {onnx.name = "/model/layers.0/attn/GroupQueryAttention/output_0"}, tensor<1x8x128x128xbf16> {onnx.name = "present.0.key"}, tensor<1x8x128x128xbf16> {onnx.name = "present.0.value"}) {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.NoValue"() {value} : () -> none
    %2:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %1) {function_name = "GroupQueryAttention", do_rotary = 0 : si64, domain_name = "com.microsoft", kv_num_heads = 8 : si64, num_heads = 32 : si64, onnx_node_name = "/model/layers.0/attn/GroupQueryAttention", rotary_interleaved = 0 : si64, scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32} : (tensor<1x1x4096xbf16>, tensor<1x1x1024xbf16>, tensor<1x1x1024xbf16>, tensor<1x8x127x128xbf16>, tensor<1x8x127x128xbf16>, tensor<1x1xi32>, tensor<i32>, none, none) -> (tensor<1x1x4096xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>)
    "onnx.Return"(%2#0, %2#1, %2#2) : (tensor<1x1x4096xbf16>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
