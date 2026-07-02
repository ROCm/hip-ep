// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test LinearAttention E2E full pipeline
// This IR represents a com.microsoft.LinearAttention operation imported via
// onnx-mlir as an onnx.Custom op. Used by Gated DeltaNet (Qwen3.5) and other
// linear-attention / state-space models that support multiple update rules
// ("linear", "gated", "delta", "gated_delta").
//
// The model has one onnx.Custom with f16 tensors (all 6 inputs + 2 outputs):
//   inputs:
//     query       (B=1, T=128, H_q*d_k=4096)   -- H_q=32, d_k=128
//     key         (1, 128, H_kv*d_k=1024)      -- H_kv=8
//     value       (1, 128, H_kv*d_v=1024)      -- d_v=128
//     past_state  (1, H_kv=8, d_k=128, d_v=128)
//     decay       (1, 128, H_kv*d_v=1024)
//     beta        (1, 128, H_kv=8)
//   outputs:
//     output        (1, 128, 4096)
//     present_state (1, 8, 128, 128)
//   attrs:
//     q_num_heads = 32, kv_num_heads = 8,
//     scale = 1/sqrt(128) ~= 0.0883883461,
//     update_rule = "gated_delta"
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Custom (LinearAttention)
//                         -> hip.linear_attention
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output and present_state buffers into single alloc
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls (wrap_linear_attention)
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 6
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_linear_attention
// CHECK: llvm.func private @main_graph(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// CHECK: llvm.call @wrap_linear_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
// CHECK-NOT: onnx.NoValue
module {
  func.func @main_graph(
      %arg0: tensor<1x128x4096xf16> {onnx.name = "query"},
      %arg1: tensor<1x128x1024xf16> {onnx.name = "key"},
      %arg2: tensor<1x128x1024xf16> {onnx.name = "value"},
      %arg3: tensor<1x8x128x128xf16> {onnx.name = "past_state"},
      %arg4: tensor<1x128x1024xf16> {onnx.name = "decay"},
      %arg5: tensor<1x128x8xf16> {onnx.name = "beta"})
      -> (tensor<1x128x4096xf16> {onnx.name = "output"},
          tensor<1x8x128x128xf16> {onnx.name = "present_state"}) {
    %0:2 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {
      function_name = "LinearAttention",
      domain_name = "com.microsoft",
      onnx_node_name = "/model/layers.0/linear_attn/LinearAttention",
      q_num_heads = 32 : si64,
      kv_num_heads = 8 : si64,
      scale = 0.0883883461 : f32,
      chunk_size = 64 : si64,
      update_rule = "gated_delta"
    } : (tensor<1x128x4096xf16>, tensor<1x128x1024xf16>,
         tensor<1x128x1024xf16>, tensor<1x8x128x128xf16>,
         tensor<1x128x1024xf16>, tensor<1x128x8xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>)
    "onnx.Return"(%0#0, %0#1)
        : (tensor<1x128x4096xf16>, tensor<1x8x128x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
