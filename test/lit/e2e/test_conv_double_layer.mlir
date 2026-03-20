// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test double Conv+Relu E2E pipeline with constant weights.
// This model has two Conv+Relu layers with onnx.Constant weight tensors.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Conv → hip.conv, onnx.Relu → hip.relu
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.call @wrap_miopenConvolutionForward
// CHECK: llvm.call @wrap_miopenActivationForward_relu
// CHECK: llvm.call @wrap_miopenConvolutionForward
// CHECK: llvm.call @wrap_miopenActivationForward_relu
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Conv
// CHECK-NOT: onnx.Relu
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32> {onnx.name = "input"}) -> (tensor<1x64x112x112xf32> {onnx.name = "output"}) attributes {onnx.graph.name = "resent50_by_morphizen"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<64x3x3x3xf32>} : () -> tensor<64x3x3x3xf32>
    %2 = "onnx.Constant"() {value = dense<5.000000e-01> : tensor<64xf32>} : () -> tensor<64xf32>
    %3 = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<64x64x3x3xf32>} : () -> tensor<64x64x3x3xf32>
    %4 = "onnx.Constant"() {value = dense<1.000000e-01> : tensor<64xf32>} : () -> tensor<64xf32>
    %5 = "onnx.Conv"(%arg0, %1, %2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1], node.outputs = ["conv1_out"], onnx_node_name = ""} : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>
    %6 = "onnx.Relu"(%5) {node.outputs = ["relu1_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %7 = "onnx.Conv"(%6, %3, %4) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2], node.outputs = ["conv2_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %8 = "onnx.Relu"(%7) {node.outputs = ["output"], onnx_node_name = ""} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %8 : tensor<1x64x112x112xf32>
  }
}
