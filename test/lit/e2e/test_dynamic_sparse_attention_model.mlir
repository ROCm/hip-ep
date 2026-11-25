// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 9
// CHECK-SAME: hipdnn.output_count = 3
// CHECK: llvm.func @wrap_dynamic_sparse_attention
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<?x?x6144xf16> {onnx.name = "query"},
      %arg1: tensor<?x?x512xf16> {onnx.name = "key"},
      %arg2: tensor<?x?x512xf16> {onnx.name = "value"},
      %arg3: tensor<?x2x?x256xf16> {onnx.name = "past_key"},
      %arg4: tensor<?x2x?x256xf16> {onnx.name = "past_value"},
      %arg5: tensor<?x2051xi32> {onnx.name = "selected_indices"},
      %arg6: tensor<?xi32> {onnx.name = "selected_counts"},
      %arg7: tensor<?xi32> {onnx.name = "seqlens_k"},
      %arg8: tensor<i32> {onnx.name = "total_sequence_length"})
      -> (tensor<?x?x6144xf16> {onnx.name = "output"},
          tensor<?x2x?x256xf16> {onnx.name = "present_key"},
          tensor<?x2x?x256xf16> {onnx.name = "present_value"}) {
    %none = "onnx.NoValue"() {value} : () -> none
    %0:3 = "onnx.Custom"(%arg0, %arg1, %arg2, %arg3, %arg4, %none, %none, %arg5, %arg6, %arg7, %arg8, %none, %none, %none, %none, %none, %none) {
      function_name = "DynamicSparseAttention",
      domain_name = "com.microsoft",
      onnx_node_name = "dsa_node",
      num_heads = 24 : si64,
      kv_num_heads = 2 : si64,
      scale = 6.250000e-02 : f32,
      is_causal = 1 : si64,
      attention_mode = "selected_only",
      selected_kv_source = "main"
    } : (tensor<?x?x6144xf16>, tensor<?x?x512xf16>, tensor<?x?x512xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>, none, none, tensor<?x2051xi32>, tensor<?xi32>, tensor<?xi32>, tensor<i32>, none, none, none, none, none, none)
      -> (tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>)
    "onnx.Return"(%0#0, %0#1, %0#2) : (tensor<?x?x6144xf16>, tensor<?x2x?x256xf16>, tensor<?x2x?x256xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
