// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// onnx.Upsample -> hip.resize -> wrap_resize.
// Schema 9 linear upsample uses asymmetric coordinates.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Upsample
// CHECK: llvm.func @wrap_resize
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x256x1x1xf32> {onnx.name = "input"})
      -> (tensor<1x256x65x65xf32> {onnx.name = "output"}) {
    %scales = "onnx.Constant"() {value = dense<[1.0, 1.0, 65.0, 65.0]> : tensor<4xf32>}
        : () -> tensor<4xf32>
    %0 = "onnx.Upsample"(%arg0, %scales) {mode = "linear", onnx_node_name = "upsample_node"}
        : (tensor<1x256x1x1xf32>, tensor<4xf32>) -> tensor<1x256x65x65xf32>
    "onnx.Return"(%0) : (tensor<1x256x65x65xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
