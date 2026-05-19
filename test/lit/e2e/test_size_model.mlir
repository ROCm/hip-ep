// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Size E2E full pipeline.
// onnx.Size folds to a rank-0 int64 arith.constant during
// convert-onnx-to-hip because the EP requires static input shapes — so the
// element count is always a compile-time integer. No HIP op is materialised
// and no runtime symbol is required.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Size -> arith.constant
// 2. generate-interface: emit inference_init/compute/cleanup/metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Size
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32> {onnx.name = "input"})
      -> (tensor<i64> {onnx.name = "size"}) {
    %0 = "onnx.Size"(%arg0) {onnx_node_name = "size_node"}
        : (tensor<2x3x4xf32>) -> tensor<i64>
    "onnx.Return"(%0) : (tensor<i64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
