// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Min E2E full pipeline
// onnx.Min reuses the MIOpen miopenOpTensor path with tensor_op = Min.
// No new runtime function needed — goes through wrap_miopenOpTensor.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Min → hip.min
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: hip.min → llvm.call @wrap_miopenOpTensor
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenOpTensor
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Min
module {
  func.func @main_graph(%arg0: tensor<3xf32> {onnx.name = "data_0"}, %arg1: tensor<3xf32> {onnx.name = "data_1"}) -> (tensor<3xf32> {onnx.name = "min"}) {
    %0 = "onnx.Min"(%arg0, %arg1) {onnx_node_name = "min_node"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    "onnx.Return"(%0) : (tensor<3xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// Integer Min model. MIOpen's miopenOpTensor doesn't support integer
// element types, so this lowering exercises the int fallback path in
// wrap_miopenOpTensor (hip_elementwise_min). The lit check only verifies
// the lowering still terminates in wrap_miopenOpTensor + the standard
// entry points; runtime correctness is covered by the numeric tests.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenOpTensor
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Min
module {
  func.func @main_graph(%arg0: tensor<32xi32> {onnx.name = "data_0"}, %arg1: tensor<32xi32> {onnx.name = "data_1"}) -> (tensor<32xi32> {onnx.name = "min"}) {
    %0 = "onnx.Min"(%arg0, %arg1) {onnx_node_name = "min_node"} : (tensor<32xi32>, tensor<32xi32>) -> tensor<32xi32>
    "onnx.Return"(%0) : (tensor<32xi32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
