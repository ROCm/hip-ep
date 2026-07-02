// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test MultiHeadAttention E2E full pipeline (com.microsoft.MultiHeadAttention).
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: ONNX Custom op -> hip.multi_head_attention
// 2. canonicalize: simplify redundant operations
// 3. memory-pooling: pool buffers into a single allocation (if any)
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: emit inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_multi_head_attention
// CHECK: llvm.func private @main_graph(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// CHECK: llvm.call @wrap_multi_head_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<1x128x4096xf16> {onnx.name = "query"},
      %arg1: tensor<1x128x4096xf16> {onnx.name = "key"},
      %arg2: tensor<1x128x4096xf16> {onnx.name = "value"})
      -> (tensor<1x128x4096xf16> {onnx.name = "output"}) {
    %0 = "onnx.Custom"(%arg0, %arg1, %arg2)
        {function_name = "MultiHeadAttention",
         domain_name = "com.microsoft",
         onnx_node_name = "/model/layers.0/attn/MultiHeadAttention",
         num_heads = 32 : si64,
         scale = 0.0883883461 : f32,
         mask_filter_value = -1.000000e+04 : f32,
         unidirectional = 0 : si64}
        : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
        -> tensor<1x128x4096xf16>
    "onnx.Return"(%0) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
