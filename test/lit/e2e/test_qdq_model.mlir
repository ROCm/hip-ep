// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test QuantizeLinear / DequantizeLinear E2E full pipeline on a QDQ pair.
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.QuantizeLinear   → hip.quantize_linear
//                         onnx.DequantizeLinear → hip.dequantize_linear
// 2. convert-hip-to-llvm: HIP ops → wrap_quantize_linear /
//                         wrap_dequantize_linear runtime calls
// 3. generate-interface: create inference_init / compute / cleanup / metadata
//
// Sanity-checks that both ONNX markers are fully consumed (so nothing falls
// through to ORT CPU silently) and that both runtime symbols make it to the
// final LLVM module.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_quantize_linear
// CHECK-DAG: llvm.func @wrap_dequantize_linear
// CHECK: llvm.call @wrap_quantize_linear
// CHECK: llvm.call @wrap_dequantize_linear
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.QuantizeLinear
// CHECK-NOT: onnx.DequantizeLinear

module {
  func.func @main_graph(%x: tensor<1x64x32xf32> {onnx.name = "x"})
      -> (tensor<1x64x32xf32> {onnx.name = "output"}) {
    %scale = "onnx.Constant"() {value = dense<1.000000e-01> : tensor<f32>} : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %q = "onnx.QuantizeLinear"(%x, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      saturate = 1 : si64,
      onnx_node_name = "QuantizeLinear_0"
    } : (tensor<1x64x32xf32>, tensor<f32>, tensor<i8>) -> tensor<1x64x32xi8>
    %dq = "onnx.DequantizeLinear"(%q, %scale, %zp) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "DequantizeLinear_0"
    } : (tensor<1x64x32xi8>, tensor<f32>, tensor<i8>) -> tensor<1x64x32xf32>
    "onnx.Return"(%dq) : (tensor<1x64x32xf32>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
