// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test TopK E2E pipeline.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.TopK -> hip.top_k
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffers into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 2
// CHECK: llvm.func @wrap_top_k
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.TopK
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "x"},
                          %arg1: tensor<i64> {onnx.name = "k"})
      -> (tensor<3x2xf32> {onnx.name = "values"},
          tensor<3x2xi64> {onnx.name = "indices"}) {
    %values, %indices = "onnx.TopK"(%arg0, %arg1) {
      axis = 1 : si64,
      largest = 1 : si64,
      onnx_node_name = "topk_node",
      sorted = 1 : si64
    } : (tensor<3x4xf32>, tensor<i64>) -> (tensor<3x2xf32>, tensor<3x2xi64>)
    "onnx.Return"(%values, %indices)
        : (tensor<3x2xf32>, tensor<3x2xi64>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
