// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Reshape E2E full pipeline from real Reshape_seq128.onnx
// Rank-reducing reshape: 3D -> 2D (merge leading dims with -1 inference)
// Input:  tensor<1x128x32xf16>
// Output: tensor<128x32xf16>
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Reshape → tensor.collapse_shape
// 2. bufferize: tensor.collapse_shape → memref.collapse_shape
// 3. convert-hip-to-llvm: full lowering to LLVM IR
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK-NOT: memref.collapse_shape
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x128x32xf16> {onnx.name = "input_0"}) -> (tensor<128x32xf16> {onnx.name = "output_0"}) {
    %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64, onnx_node_name = "Reshape_0"} : (tensor<1x128x32xf16>, tensor<2xi64>) -> tensor<128x32xf16>
    "onnx.Return"(%result) : (tensor<128x32xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
