// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test global-pool E2E full pipeline (native strategy) — three flavors share
// one HIP op + one runtime symbol:
//   onnx.GlobalAveragePool / GlobalMaxPool / GlobalLpPool
//     -> hip.global_pool {mode, p}
//     -> wrap_global_pool
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: all three ONNX ops -> hip.global_pool
// 2. one-shot-bufferize + memory-pooling: pool the output buffer
// 3. convert-hip-to-llvm: hip.global_pool -> llvm.call @wrap_global_pool
// 4. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 3
// CHECK-NOT: onnx.GlobalAveragePool
// CHECK-NOT: onnx.GlobalMaxPool
// CHECK-NOT: onnx.GlobalLpPool
// CHECK: llvm.func @wrap_global_pool
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%avg_in: tensor<1x3x5x5xf32> {onnx.name = "avg_in"},
                        %max_in: tensor<1x3x5x5xf32> {onnx.name = "max_in"},
                        %lp_in:  tensor<1x3x5x5xf32> {onnx.name = "lp_in"})
      -> (tensor<1x3x1x1xf32> {onnx.name = "avg_out"},
          tensor<1x3x1x1xf32> {onnx.name = "max_out"},
          tensor<1x3x1x1xf32> {onnx.name = "lp_out"}) {
    %avg = "onnx.GlobalAveragePool"(%avg_in) {onnx_node_name = "gap_node"}
        : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    %max = "onnx.GlobalMaxPool"(%max_in) {onnx_node_name = "gmp_node"}
        : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    %lp = "onnx.GlobalLpPool"(%lp_in) {p = 3 : i64, onnx_node_name = "glp_node"}
        : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    "onnx.Return"(%avg, %max, %lp)
        : (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
