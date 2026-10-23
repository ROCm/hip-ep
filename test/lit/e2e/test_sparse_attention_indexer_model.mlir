// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 7
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_sparse_attention_indexer
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<?x?x640xf16> {onnx.name = "query"},
      %arg1: tensor<128xf16> {onnx.name = "query_norm_weight"},
      %arg2: tensor<128xf16> {onnx.name = "key_norm_weight"},
      %arg3: tensor<?x?x64xf16> {onnx.name = "cos_cache"},
      %arg4: tensor<?x?x64xf16> {onnx.name = "sin_cache"},
      %arg5: tensor<?x?xi64> {onnx.name = "attention_mask"},
      %arg6: tensor<?x?x128xf16> {onnx.name = "past_key"})
      -> (tensor<?x?x2051xi32> {onnx.name = "selected_indices"},
          tensor<?x?x128xf16> {onnx.name = "present_key"}) {
    %none = "onnx.NoValue"() {value} : () -> none
    %0:2 = "onnx.Custom"(%arg0, %none, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {
      function_name = "SparseAttentionIndexer",
      domain_name = "com.microsoft",
      onnx_node_name = "sai_node",
      policy_mode = "qsa",
      compress_ratio = 4 : si64,
      token_budget = 2048 : si64,
      epsilon = 9.99999997E-7 : f32,
      scale = 0.0883883461 : f32
    } : (tensor<?x?x640xf16>, none, tensor<128xf16>, tensor<128xf16>, tensor<?x?x64xf16>, tensor<?x?x64xf16>, tensor<?x?xi64>, tensor<?x?x128xf16>)
      -> (tensor<?x?x2051xi32>, tensor<?x?x128xf16>)
    "onnx.Return"(%0#0, %0#1) : (tensor<?x?x2051xi32>, tensor<?x?x128xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
