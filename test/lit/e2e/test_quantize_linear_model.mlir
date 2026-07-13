// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_quantize_linear
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.QuantizeLinear

module {
  func.func @main_graph(%x: tensor<4x8xf32> {onnx.name = "x"}, %scale: tensor<8xf32> {onnx.name = "scale"}, %zp: tensor<8xui8> {onnx.name = "zp"}) -> (tensor<4x8xui8> {onnx.name = "y"}) {
    %y = "onnx.QuantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, onnx_node_name = "quant_node"} : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xui8>) -> tensor<4x8xui8>
    "onnx.Return"(%y) : (tensor<4x8xui8>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
