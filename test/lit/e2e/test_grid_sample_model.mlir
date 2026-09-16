// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test GridSample E2E full pipeline.
//
// 1. convert-onnx-to-hip: onnx.GridSample → hip.grid_sample
// 2. convert-hip-to-llvm: hip.grid_sample → llvm.call @wrap_grid_sample
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_grid_sample
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.GridSample
module {
  func.func @main_graph(%arg0: tensor<1x3x8x8xf32> {onnx.name = "X"}, %arg1: tensor<1x4x4x2xf32> {onnx.name = "grid"}) -> (tensor<1x3x4x4xf32> {onnx.name = "Y"}) {
    %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "bilinear", padding_mode = "zeros", onnx_node_name = "gridsample_node"} : (tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>) -> tensor<1x3x4x4xf32>
    "onnx.Return"(%0) : (tensor<1x3x4x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
