// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test InstanceNormalization E2E full pipeline.
//
// 1. convert-onnx-to-hip: onnx.InstanceNormalization → hip.instance_norm
// 2. convert-hip-to-llvm: hip.instance_norm → llvm.call @wrap_instance_normalization
//
// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_instance_normalization
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.InstanceNormalization
module {
  func.func @main_graph(%arg0: tensor<2x3x8x8xf32> {onnx.name = "X"}, %arg1: tensor<3xf32> {onnx.name = "scale"}, %arg2: tensor<3xf32> {onnx.name = "B"}) -> (tensor<2x3x8x8xf32> {onnx.name = "Y"}) {
    %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, onnx_node_name = "instancenorm_node"} : (tensor<2x3x8x8xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x8x8xf32>
    "onnx.Return"(%0) : (tensor<2x3x8x8xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
