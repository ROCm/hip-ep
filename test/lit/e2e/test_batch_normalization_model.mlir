// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test BatchNormalization inference through the full pipeline.
//
// 1. convert-onnx-to-hip: onnx.BatchNormalization → hip.batch_norm
// 2. convert-hip-to-llvm: hip.batch_norm → llvm.call @wrap_batch_normalization
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 5
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_batch_normalization
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.BatchNormalization
module {
  func.func @main_graph(%arg0: tensor<1x4x1x1xf16> {onnx.name = "X"}, %arg1: tensor<4xf16> {onnx.name = "scale"}, %arg2: tensor<4xf16> {onnx.name = "B"}, %arg3: tensor<4xf16> {onnx.name = "mean"}, %arg4: tensor<4xf16> {onnx.name = "var"}) -> (tensor<1x4x1x1xf16> {onnx.name = "Y"}) {
    %0 = "onnx.BatchNormalization"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 1.001000e-05 : f32, momentum = 0.899999976 : f32, onnx_node_name = "batchnorm_node"} : (tensor<1x4x1x1xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>, tensor<4xf16>) -> tensor<1x4x1x1xf16>
    "onnx.Return"(%0) : (tensor<1x4x1x1xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
