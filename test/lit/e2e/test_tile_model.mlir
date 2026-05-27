// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_tile
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Tile
module {
  func.func @main_graph(%arg0: tensor<2x3xf32> {onnx.name = "input"}, %arg1: tensor<2xi64> {onnx.name = "repeats"}) -> (tensor<4x9xf32> {onnx.name = "output"}) {
    %0 = "onnx.Tile"(%arg0, %arg1) {onnx_node_name = "tile_node"} : (tensor<2x3xf32>, tensor<2xi64>) -> tensor<4x9xf32>
    "onnx.Return"(%0) : (tensor<4x9xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
