// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test MaxPool E2E full pipeline (native strategy).
// onnx.MaxPool -> hip.pool -> wrap_pool.
//
// Verifies:
// 1. convert-onnx-to-hip: onnx.MaxPool -> hip.pool with explicit pads /
//    dilations / strides resolved at compile time
// 2. one-shot-bufferize + memory-pooling: pool the output buffer
// 3. convert-hip-to-llvm: hip.pool -> llvm.call @wrap_pool
// 4. generate-interface: emit inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.MaxPool
// CHECK: llvm.func @wrap_pool
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x3x32x32xf16> {onnx.name = "input"})
      -> (tensor<1x3x16x16xf16> {onnx.name = "output"}) {
    %0 = "onnx.MaxPool"(%arg0)
        {onnx_node_name = "maxpool_node",
         kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
        : (tensor<1x3x32x32xf16>) -> tensor<1x3x16x16xf16>
    "onnx.Return"(%0) : (tensor<1x3x16x16xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
