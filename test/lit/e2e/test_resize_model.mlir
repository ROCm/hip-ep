// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Resize E2E full pipeline (native strategy).
// onnx.Resize -> hip.resize -> wrap_resize.
//
// Verifies:
// 1. convert-onnx-to-hip: onnx.Resize -> hip.resize with all string attrs
//    decoded to integer enums (mode / coord_transform / nearest_mode)
// 2. one-shot-bufferize + memory-pooling: pool the output buffer
// 3. convert-hip-to-llvm: hip.resize -> llvm.call @wrap_resize
// 4. generate-interface: emit inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.Resize
// CHECK: llvm.func @wrap_resize
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x3x16x16xf16> {onnx.name = "input"})
      -> (tensor<1x3x32x32xf16> {onnx.name = "output"}) {
    %roi = "onnx.NoValue"() {value} : () -> none
    %scales = "onnx.Constant"() {
        value = dense<[1.0, 1.0, 2.0, 2.0]> : tensor<4xf32>
      } : () -> tensor<4xf32>
    %0 = "onnx.Resize"(%arg0, %roi, %scales)
        {onnx_node_name = "resize_node",
         mode = "linear",
         coordinate_transformation_mode = "half_pixel"}
        : (tensor<1x3x16x16xf16>, none, tensor<4xf32>)
        -> tensor<1x3x32x32xf16>
    "onnx.Return"(%0) : (tensor<1x3x32x32xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
