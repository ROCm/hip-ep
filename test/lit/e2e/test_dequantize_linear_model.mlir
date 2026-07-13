// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_dequantize_linear
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.DequantizeLinear

module {
  func.func @main_graph(%x: tensor<4x8xi8> {onnx.name = "x"}, %scale: tensor<8xf32> {onnx.name = "scale"}, %zp: tensor<8xi8> {onnx.name = "zp"}) -> (tensor<4x8xf32> {onnx.name = "y"}) {
    %y = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, onnx_node_name = "dequant_node"} : (tensor<4x8xi8>, tensor<8xf32>, tensor<8xi8>) -> tensor<4x8xf32>
    "onnx.Return"(%y) : (tensor<4x8xf32>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
