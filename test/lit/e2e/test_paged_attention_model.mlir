// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// E2E: com.microsoft.PagedAttention (onnx.Custom) through full hipdnn-pipeline.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 8
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_paged_attention
// CHECK: llvm.call @wrap_paged_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
// CHECK-NOT: onnx.NoValue
module {
  func.func @main_graph(
      %arg0: tensor<4x128xf16> {onnx.name = "query"},
      %arg1: tensor<4x128xf16> {onnx.name = "key"},
      %arg2: tensor<4x128xf16> {onnx.name = "value"},
      %arg3: tensor<8x16x2x16xf16> {onnx.name = "key_cache"},
      %arg4: tensor<8x16x2x16xf16> {onnx.name = "value_cache"},
      %arg5: tensor<2xi32> {onnx.name = "cumulative_sequence_length"},
      %arg6: tensor<1xi32> {onnx.name = "past_seqlens"},
      %arg7: tensor<1x8xi32> {onnx.name = "block_table"})
      -> (tensor<4x128xf16> {onnx.name = "output"}) {
    %0 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         onnx_node_name = "paged_attn_0"}
        : (tensor<4x128xf16>, tensor<4x128xf16>, tensor<4x128xf16>,
           tensor<8x16x2x16xf16>, tensor<8x16x2x16xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x128xf16>
    "onnx.Return"(%0) : (tensor<4x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
