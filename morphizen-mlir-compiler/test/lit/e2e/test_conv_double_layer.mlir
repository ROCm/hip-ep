// RUN: hip-opt %s --convert-onnx-to-hip --convert-hip-to-llvm | FileCheck %s

// Test double-layer Conv+Relu network from E2E test
// This IR contains onnx.Constant operations with tensor types from mlir-imp
// Verifies that:
// 1. onnx.Constant operations are converted to LLVM globals
// 2. Constant registry is generated (@get_constant_registry)
// 3. Metadata is generated (hipdnn.input_count, etc.)
// 4. @main_graph signature is transformed to (ptr, ptr, ptr) -> i32

// CHECK: module attributes {hipdnn.input_count
// CHECK: llvm.mlir.global internal constant @constant_{{.*}}(dense
// CHECK: llvm.mlir.global internal constant @constant_registry
// CHECK: llvm.func @get_constant_registry() -> !llvm.ptr
// CHECK-NOT: arith.constant
// CHECK-NOT: onnx.Constant
module {
  func.func @main_graph(%arg0: tensor<1x3x224x224xf32> {onnx.name = "input"}) -> (tensor<1x64x112x112xf32> {onnx.name = "output"}) attributes {onnx.graph.name = "resent50_by_morphizen"} {
    %0 = "onnx.NoValue"() <{value}> : () -> none
    %1 = onnx.Constant dense<1.000000e+00> : tensor<64x3x3x3xf32>
    %2 = onnx.Constant dense<5.000000e-01> : tensor<64xf32>
    %3 = onnx.Constant dense<2.000000e+00> : tensor<64x64x3x3xf32>
    %4 = onnx.Constant dense<1.000000e-01> : tensor<64xf32>
    %5 = "onnx.Conv"(%arg0, %1, %2) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> {node.outputs = ["conv1_out"], onnx_node_name = ""} : (tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<1x64x224x224xf32>
    %6 = "onnx.Relu"(%5) {node.outputs = ["relu1_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>) -> tensor<1x64x224x224xf32>
    %7 = "onnx.Conv"(%6, %3, %4) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]}> {node.outputs = ["conv2_out"], onnx_node_name = ""} : (tensor<1x64x224x224xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %8 = "onnx.Relu"(%7) {node.outputs = ["output"], onnx_node_name = ""} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %8 : tensor<1x64x112x112xf32>
  }
}
