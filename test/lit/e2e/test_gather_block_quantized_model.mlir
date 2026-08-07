// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test com.microsoft.GatherBlockQuantized E2E full pipeline.
// Verifies the complete hipdnn-pipeline for the gather-then-block-dequant op:
// 1. convert-onnx-to-hip: onnx.Custom (com.microsoft.GatherBlockQuantized)
//                         → hip.gather_block_quantized
// 2. convert-hip-to-llvm: HIP op → wrap_gather_block_quantized runtime call
// 3. generate-interface: Create inference_init / compute / cleanup / metadata
//
// Sanity-checks that the original onnx.Custom marker is fully consumed (so
// nothing falls through to ORT CPU silently) and that the runtime symbol
// makes it all the way to the final LLVM module.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_gather_block_quantized
// CHECK: llvm.call @wrap_gather_block_quantized
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Custom
// CHECK-NOT: onnx.NoValue

module {
  func.func @main_graph(%indices: tensor<8xi64> {onnx.name = "indices"}) -> (tensor<8x192xf16> {onnx.name = "output"}) {
    %data = "onnx.Constant"() {value = dense<1> : tensor<2048x96xui8>} : () -> tensor<2048x96xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %zp = "onnx.Constant"() {value = dense<8> : tensor<2048x12xui8>} : () -> tensor<2048x12xui8>
    %out = "onnx.Custom"(%data, %indices, %scales, %zp) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_0"
    } : (tensor<2048x96xui8>, tensor<8xi64>, tensor<2048x12xf16>, tensor<2048x12xui8>) -> tensor<8x192xf16>
    "onnx.Return"(%out) : (tensor<8x192xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
