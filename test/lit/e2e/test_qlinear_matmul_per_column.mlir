// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.QLinearMatMul with one B scale and zero point per output column
// lowers through hip.qlinear_matmul and bufferizes to wrap_qlinear_matmul.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.QLinearMatMul
// CHECK: llvm.func @wrap_qlinear_matmul
// CHECK: llvm.call @wrap_qlinear_matmul
// CHECK: llvm.func @inference_compute

module {
  func.func @main_graph(%a: tensor<1x2xui8> {onnx.name = "input"})
      -> (tensor<1x2xui8> {onnx.name = "output"}) {
    %as = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %az = "onnx.Constant"() {value = dense<0> : tensor<ui8>} : () -> tensor<ui8>
    %b = "onnx.Constant"() {value = dense<[[1, 2], [3, 0]]> : tensor<2x2xui8>} : () -> tensor<2x2xui8>
    %bs = "onnx.Constant"() {value = dense<[0.5, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
    %bz = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xui8>} : () -> tensor<2xui8>
    %ys = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %yz = "onnx.Constant"() {value = dense<0> : tensor<ui8>} : () -> tensor<ui8>
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz) {
      onnx_node_name = "qlinear_matmul"
    } : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>, tensor<2x2xui8>,
         tensor<2xf32>, tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
    "onnx.Return"(%y) : (tensor<1x2xui8>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
