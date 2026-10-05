// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.QLinearMatMul -> hip.qlinear_matmul -> wrap_qlinear_matmul.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.QLinearMatMul
// CHECK: llvm.func @wrap_qlinear_matmul
// CHECK: llvm.call @wrap_qlinear_matmul
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json

module {
  func.func @main_graph(%a: tensor<1x2xui8> {onnx.name = "input"})
      -> (tensor<1x2xui8> {onnx.name = "output"}) {
    %as = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %az = "onnx.Constant"() {value = dense<1> : tensor<ui8>} : () -> tensor<ui8>
    %b = "onnx.Constant"() {value = dense<[[1, 0], [-1, 1]]> : tensor<2x2xi8>} : () -> tensor<2x2xi8>
    %bs = "onnx.Constant"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %bz = "onnx.Constant"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
    %ys = "onnx.Constant"() {value = dense<0.25> : tensor<f32>} : () -> tensor<f32>
    %yz = "onnx.Constant"() {value = dense<2> : tensor<ui8>} : () -> tensor<ui8>
    %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz) {
      onnx_node_name = "qlinear_matmul"
    } : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>, tensor<2x2xi8>,
         tensor<f32>, tensor<i8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
    "onnx.Return"(%y) : (tensor<1x2xui8>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
