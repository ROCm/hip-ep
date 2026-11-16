// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test RoiAlign E2E full pipeline with a dynamic number of regions.
//
// 1. convert-onnx-to-hip: onnx.RoiAlign -> hip.roi_align
// 2. convert-hip-to-llvm: hip.roi_align -> llvm.call @wrap_roi_align
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_roi_align
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.RoiAlign
module {
  func.func @main_graph(%arg0: tensor<1x8x32x32xf32> {onnx.name = "X"}, %arg1: tensor<?x4xf32> {onnx.name = "rois"}, %arg2: tensor<?xi64> {onnx.name = "batch_indices"}) -> (tensor<?x8x7x7xf32> {onnx.name = "Y"}) {
    %0 = "onnx.RoiAlign"(%arg0, %arg1, %arg2) {mode = "avg", output_height = 7 : si64, output_width = 7 : si64, sampling_ratio = 2 : si64, spatial_scale = 2.500000e-01 : f32, onnx_node_name = "roialign_node"} : (tensor<1x8x32x32xf32>, tensor<?x4xf32>, tensor<?xi64>) -> tensor<?x8x7x7xf32>
    "onnx.Return"(%0) : (tensor<?x8x7x7xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
