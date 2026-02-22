// RUN: hip-opt %s | FileCheck %s

// This test verifies that the parsed MLIR from the E2E test can be loaded
// and verified by hip-opt without crashing.

// CHECK-LABEL: func.func @main_graph
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32> {onnx.name = "input"}) -> (tensor<1x64x112x112xf32> {onnx.name = "output"}) attributes {onnx.graph.name = "resent50_by_morphizen"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "arith.constant"() {value = dense<1.000000e+00> : tensor<64x3x3x3xf32>, node.outputs = ["weights1"]} : () -> tensor<64x3x3x3xf32>
    %2 = "arith.constant"() {value = dense<5.000000e-01> : tensor<64xf32>, node.outputs = ["bias1"]} : () -> tensor<64xf32>
    %3 = "arith.constant"() {value = dense<2.000000e+00> : tensor<64x64x3x3xf32>, node.outputs = ["weights2"]} : () -> tensor<64x64x3x3xf32>
    %4 = "arith.constant"() {value = dense<1.000000e-01> : tensor<64xf32>, node.outputs = ["bias2"]} : () -> tensor<64xf32>
    %5 = "onnx.Conv"(%arg0, %1, %2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1], node.outputs = ["conv1_out"], onnx_node_name = ""} : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>
    %6 = "onnx.Relu"(%5) {node.outputs = ["relu1_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %7 = "onnx.Conv"(%6, %3, %4) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2], node.outputs = ["conv2_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %8 = "onnx.Relu"(%7) {node.outputs = ["output"], onnx_node_name = ""} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    func.return %8 : tensor<1x64x112x112xf32>
  }
}
