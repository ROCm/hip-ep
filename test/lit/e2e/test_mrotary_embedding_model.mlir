// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_mrotary_embedding
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(
      %arg0: tensor<?x?x6144xf16> {onnx.name = "query"},
      %arg1: tensor<3x?x?xi64> {onnx.name = "position_ids"})
      -> (tensor<?x?x6144xf16> {onnx.name = "output"}) {
    %cos = "onnx.Constant"() {value = dense<0.5> : tensor<4096x32xf16>} : () -> tensor<4096x32xf16>
    %sin = "onnx.Constant"() {value = dense<0.25> : tensor<4096x32xf16>} : () -> tensor<4096x32xf16>
    %0 = "onnx.Custom"(%arg0, %arg1, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      onnx_node_name = "mrope_node",
      interleaved = 0 : si64,
      is_packed_batching = 0 : si64,
      mrope_layout = 1 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      num_heads = 24 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<?x?x6144xf16>, tensor<3x?x?xi64>, tensor<4096x32xf16>, tensor<4096x32xf16>) -> tensor<?x?x6144xf16>
    "onnx.Return"(%0) : (tensor<?x?x6144xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
