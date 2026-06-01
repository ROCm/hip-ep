// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test onnx.LpNormalization E2E full pipeline.
// This IR represents an ONNX LpNormalization operation as it would be
// imported via onnx-mlir / morphizen mlir-imp.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.LpNormalization decomposes into
//    Mul / ReduceSum / Sqrt / Div, each handled by its own converter.
//    The broadcasting Div is rewritten by BroadcastDivToMulReciprocal
//    into Mul(x, Reciprocal(norm)). No new HIP op or runtime symbol is
//    introduced — only existing primitives are exercised.
// 2. canonicalize: simplify redundant operations
// 3. memory-pooling: pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops -> LLVM runtime calls
//    (wrap_miopenOpTensor for Mul, wrap_reduce_sum for ReduceSum,
//     wrap_power for Sqrt + Reciprocal)
// 5. generate-interface: create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_miopenOpTensor
// CHECK-DAG: llvm.func @wrap_reduce_sum
// CHECK-DAG: llvm.func @wrap_power
// CHECK-DAG: llvm.func @inference_init
// CHECK-DAG: llvm.func @inference_compute
// CHECK-DAG: llvm.func @inference_cleanup
// CHECK-DAG: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.LpNormalization
module {
  func.func @main_graph(%arg0: tensor<2x3xf32> {onnx.name = "input"}) -> (tensor<2x3xf32> {onnx.name = "output"}) {
    %0 = "onnx.LpNormalization"(%arg0) {axis = -1 : si64, p = 2 : si64, onnx_node_name = "lpnorm_node"} : (tensor<2x3xf32>) -> tensor<2x3xf32>
    "onnx.Return"(%0) : (tensor<2x3xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
