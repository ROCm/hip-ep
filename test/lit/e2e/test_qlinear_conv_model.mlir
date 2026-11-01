// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.QLinearConv -> hip.qlinear_conv -> wrap_qlinear_conv.
// 1x1 window so the constant weight stays a single element.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.QLinearConv
// CHECK: llvm.func @wrap_qlinear_conv
// CHECK: llvm.call @wrap_qlinear_conv
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json

module {
  func.func @main_graph(%x: tensor<1x1x2x2xui8> {onnx.name = "input"})
      -> (tensor<1x1x2x2xui8> {onnx.name = "output"}) {
    %xs = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %xz = "onnx.Constant"() {value = dense<1> : tensor<ui8>} : () -> tensor<ui8>
    %w = "onnx.Constant"() {value = dense<2> : tensor<1x1x1x1xi8>} : () -> tensor<1x1x1x1xi8>
    %ws = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %wz = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %ys = "onnx.Constant"() {value = dense<0.125> : tensor<f32>} : () -> tensor<f32>
    %yz = "onnx.Constant"() {value = dense<2> : tensor<ui8>} : () -> tensor<ui8>
    %b = "onnx.Constant"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
    %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1],
      onnx_node_name = "qlinear_conv"
    } : (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>, tensor<1x1x1x1xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>, tensor<1xi32>)
      -> tensor<1x1x2x2xui8>
    "onnx.Return"(%y) : (tensor<1x1x2x2xui8>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
