// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test LayerNormalization E2E full pipeline
// Standard ONNX LayerNormalization (opset 17+) with bias.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.LayerNormalization → hip.layer_norm
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: hip.layer_norm → llvm.call @wrap_layer_normalization
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_layer_normalization
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.LayerNormalization
module {
  func.func @main_graph(%arg0: tensor<2x3x4xf32> {onnx.name = "X"}, %arg1: tensor<4xf32> {onnx.name = "Scale"}, %arg2: tensor<4xf32> {onnx.name = "B"}) -> (tensor<2x3x4xf32> {onnx.name = "Y"}) {
    %0 = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64, onnx_node_name = "layer_norm_node"} : (tensor<2x3x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    "onnx.Return"(%0) : (tensor<2x3x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
