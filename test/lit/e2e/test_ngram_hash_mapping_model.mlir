// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_ngram_hash_mapping
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<?x?xi64> {onnx.name = "input_ids"},
      %arg1: tensor<?x2xi64> {onnx.name = "past.1.ple_tokens"})
      -> (tensor<?x?x16xi64> {onnx.name = "hash_ids"},
          tensor<?x2xi64> {onnx.name = "present.1.ple_tokens"}) {
    %mult = "onnx.Constant"() {value = dense<[23703573157769, 20109073645365, 8052911324071]> : tensor<3xi64>} : () -> tensor<3xi64>
    %vocab = "onnx.Constant"() {value = dense<[20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]> : tensor<16xi64>} : () -> tensor<16xi64>
    %off = "onnx.Constant"() {value = dense<[0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]> : tensor<16xi64>} : () -> tensor<16xi64>
    %eos = "onnx.Constant"() {value = dense<248044> : tensor<i64>} : () -> tensor<i64>
    %0:2 = "onnx.Custom"(%arg0, %mult, %vocab, %arg1, %off, %eos) {
      function_name = "NGramHashMapping",
      domain_name = "com.microsoft",
      onnx_node_name = "ngram_node",
      max_ngram_size = 3 : si64,
      n_head_per_ngram = 8 : si64,
      pad_id = 248044 : si64,
      reset_on_eos = 1 : si64
    } : (tensor<?x?xi64>, tensor<3xi64>, tensor<16xi64>, tensor<?x2xi64>, tensor<16xi64>, tensor<i64>)
      -> (tensor<?x?x16xi64>, tensor<?x2xi64>)
    "onnx.Return"(%0#0, %0#1) : (tensor<?x?x16xi64>, tensor<?x2xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
