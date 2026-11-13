// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// ImageScaler with a non-zero per-channel bias:
//   output = scale * (input + bias)
// bias broadcasts as tensor<1x3x1x1xf32> through hip.add, then hip.mul
// applies the scalar scale. Both lower to wrap_elementwise.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.ImageScaler
module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32> {onnx.name = "image"}) -> (tensor<1x3x8x8xf32> {onnx.name = "scaled"}) {
    %0 = "onnx.ImageScaler"(%arg0) {bias = [0.000000e+00 : f32, 1.000000e-01 : f32, -1.000000e-01 : f32], onnx_node_name = "scaler", scale = 2.000000e+00 : f32} : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    "onnx.Return"(%0) : (tensor<1x3x8x8xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
