// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test GroupQueryAttention E2E full pipeline from real Llama-3.1-8B decode step
// This IR was imported from GroupQueryAttention_fix_decode.onnx via onnx-mlir
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: ONNX Custom op → hip.group_query_attention
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool buffers into single allocation (if any)
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 7
// CHECK-SAME: hipdnn.output_count = 3
// CHECK: llvm.func @wrap_group_query_attention
// CHECK: llvm.func private @main_graph(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// CHECK: llvm.call @wrap_group_query_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
// CHECK-NOT: onnx.NoValue
module {
  func.func @main_graph(%arg0: tensor<1x1x4096xf16> {onnx.name = "/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0"}, %arg1: tensor<1x1x1024xf16> {onnx.name = "/model/layers.0/attn/k_rotary/RotaryEmbedding/output_0"}, %arg2: tensor<1x1x1024xf16> {onnx.name = "/model/layers.0/attn/v_proj/MatMul/output_0"}, %arg3: tensor<1x8x127x128xf16> {onnx.name = "past_key_values.0.key"}, %arg4: tensor<1x8x127x128xf16> {onnx.name = "past_key_values.0.value"}, %arg5: tensor<1x1xi32> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0"}, %arg6: tensor<i32> {onnx.name = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"}) -> (tensor<1x1x4096xf16> {onnx.name = "/model/layers.0/attn/GroupQueryAttention/output_0"}, tensor<1x8x128x128xf16> {onnx.name = "present.0.key"}, tensor<1x8x128x128xf16> {onnx.name = "present.0.value"}) {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.NoValue"() {value} : () -> none
    %2:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %0, %1) {function_name = "GroupQueryAttention", do_rotary = 0 : si64, domain_name = "com.microsoft", kv_num_heads = 8 : si64, num_heads = 32 : si64, onnx_node_name = "/model/layers.0/attn/GroupQueryAttention", rotary_interleaved = 0 : si64, scale = 0.0883883461 : f32, softcap = 0.000000e+00 : f32} : (tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>, tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>, tensor<1x1xi32>, tensor<i32>, none, none) -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)
    "onnx.Return"(%2#0, %2#1, %2#2) : (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
