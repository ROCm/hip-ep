// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Shape E2E full pipeline.
// onnx.Shape is folded to a compile-time arith.constant carrying the
// (statically known) dimensions of the input tensor.  No runtime symbol
// is emitted because no compute happens at run time.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Shape -> arith.constant (then externalized
//    by the constant lowering path the same way ordinary constants are).
// 2. generate-interface: Create inference_init/compute/cleanup/metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Shape
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32> {onnx.name = "input"})
      -> (tensor<3xi64> {onnx.name = "output"}) {
    %0 = "onnx.Shape"(%arg0) {onnx_node_name = "shape_node"}
        : (tensor<2x3x4xf32>) -> tensor<3xi64>
    "onnx.Return"(%0) : (tensor<3xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
