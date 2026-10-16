// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test MatMulNBits grouping through the full hipdnn-pipeline: three decode
// projections of one activation (q/k/v shaped) become a single
// wrap_matmul_nbits_group call and no per-projection wrap_matmul_nbits call.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.output_count = 3
// CHECK-NOT: llvm.call @wrap_matmul_nbits(
// CHECK: llvm.call @wrap_matmul_nbits_group(
// CHECK-NOT: llvm.call @wrap_matmul_nbits(
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x1x2048xf16> {onnx.name = "A"}) -> (tensor<1x1x8192xf16> {onnx.name = "Q"}, tensor<1x1x512xf16> {onnx.name = "K"}, tensor<1x1x512xf16> {onnx.name = "V"}) {
    %Bq = "onnx.Constant"() {value = dense<1> : tensor<8192x64x16xui8>} : () -> tensor<8192x64x16xui8>
    %Sq = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<8192x64xf16>} : () -> tensor<8192x64xf16>
    %Bk = "onnx.Constant"() {value = dense<2> : tensor<512x64x16xui8>} : () -> tensor<512x64x16xui8>
    %Sk = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<512x64xf16>} : () -> tensor<512x64xf16>
    %Bv = "onnx.Constant"() {value = dense<3> : tensor<512x64x16xui8>} : () -> tensor<512x64x16xui8>
    %Sv = "onnx.Constant"() {value = dense<3.000000e+00> : tensor<512x64xf16>} : () -> tensor<512x64xf16>
    %Q = "onnx.Custom"(%arg0, %Bq, %Sq) {function_name = "MatMulNBits", K = 2048 : si64, N = 8192 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft", onnx_node_name = "q_proj"} : (tensor<1x1x2048xf16>, tensor<8192x64x16xui8>, tensor<8192x64xf16>) -> tensor<1x1x8192xf16>
    %K = "onnx.Custom"(%arg0, %Bk, %Sk) {function_name = "MatMulNBits", K = 2048 : si64, N = 512 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft", onnx_node_name = "k_proj"} : (tensor<1x1x2048xf16>, tensor<512x64x16xui8>, tensor<512x64xf16>) -> tensor<1x1x512xf16>
    %V = "onnx.Custom"(%arg0, %Bv, %Sv) {function_name = "MatMulNBits", K = 2048 : si64, N = 512 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft", onnx_node_name = "v_proj"} : (tensor<1x1x2048xf16>, tensor<512x64x16xui8>, tensor<512x64xf16>) -> tensor<1x1x512xf16>
    "onnx.Return"(%Q, %K, %V) : (tensor<1x1x8192xf16>, tensor<1x1x512xf16>, tensor<1x1x512xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
