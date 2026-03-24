// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test MatMulNBits E2E full pipeline
// Verifies the complete hipdnn-pipeline for 4-bit quantized matmul:
// 1. convert-onnx-to-hip: ONNX Custom op → hip.matmul_nbits
// 2. convert-hip-to-llvm: HIP op → wrap_matmul_nbits runtime call
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_matmul_nbits
// CHECK: llvm.call @wrap_matmul_nbits
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
module {
  func.func @main_graph(%arg0: tensor<1x128x2880xf16> {onnx.name = "A"}) -> (tensor<1x128x5120xf16> {onnx.name = "Y"}) {
    %B = "onnx.Constant"() {value = dense<1> : tensor<5120x90x16xui8>} : () -> tensor<5120x90x16xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<5120x90xf16>} : () -> tensor<5120x90xf16>
    %Y = "onnx.Custom"(%arg0, %B, %scales) {function_name = "MatMulNBits", K = 2880 : si64, N = 5120 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft", onnx_node_name = "MatMulNBits_0"} : (tensor<1x128x2880xf16>, tensor<5120x90x16xui8>, tensor<5120x90xf16>) -> tensor<1x128x5120xf16>
    "onnx.Return"(%Y) : (tensor<1x128x5120xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
