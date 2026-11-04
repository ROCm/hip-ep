// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test NonMaxSuppression E2E full pipeline with a dynamic box count.
//
// 1. convert-onnx-to-hip: onnx.NonMaxSuppression -> hip.non_max_suppression
//    + hip.readback_dim + tensor.extract_slice
// 2. convert-hip-to-llvm: hip.non_max_suppression ->
//    llvm.call @wrap_non_max_suppression
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_non_max_suppression
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.NonMaxSuppression
module {
  func.func @main_graph(%arg0: tensor<1x?x4xf32> {onnx.name = "boxes"}, %arg1: tensor<1x1x?xf32> {onnx.name = "scores"}) -> (tensor<?x3xi64> {onnx.name = "selected_indices"}) {
    %max = "onnx.Constant"() {value = dense<2000> : tensor<1xi64>} : () -> tensor<1xi64>
    %iou = "onnx.Constant"() {value = dense<0.699999988> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %max, %iou) {center_point_box = 0 : si64, onnx_node_name = "nms_node"} : (tensor<1x?x4xf32>, tensor<1x1x?xf32>, tensor<1xi64>, tensor<1xf32>) -> tensor<?x3xi64>
    "onnx.Return"(%0) : (tensor<?x3xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
