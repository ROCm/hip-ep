// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ScatterND E2E full pipeline (native path — runtime is a logging
// stub today but the IR must still flow through every stage cleanly).
// Verifies:
//   1. convert-onnx-to-hip:  onnx.ScatterND -> hip.scatter_nd
//   2. bufferize:            tensor -> memref (DPS init is honoured)
//   3. convert-hip-to-llvm:  hip.scatter_nd -> wrap_scatter_nd call
//   4. generate-interface:   inference_init / compute / cleanup / metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.ScatterND
// CHECK-NOT: hip.scatter_nd
// CHECK: llvm.func @wrap_scatter_nd
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%data: tensor<4x4x4xf32>     {onnx.name = "data"},
                        %indices: tensor<2x1xi64>    {onnx.name = "indices"},
                        %updates: tensor<2x4x4xf32>  {onnx.name = "updates"})
      -> (tensor<4x4x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.ScatterND"(%data, %indices, %updates)
        {onnx_node_name = "scatter_node", reduction = "none"}
        : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>)
          -> tensor<4x4x4xf32>
    "onnx.Return"(%0) : (tensor<4x4x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
