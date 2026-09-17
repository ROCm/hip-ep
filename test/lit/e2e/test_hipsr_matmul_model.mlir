// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// The one model that goes end to end through the hipsr pipeline: a MatMul
// against a small constant weight. test/e2e/CMakeLists.txt compiles and runs
// it twice, once per pipeline (E2E_{Compile,Execute}_test_hipsr_matmul_model
// through hip, E2E_{Compile,Execute}_hipsr_matmul through hipsr), so the two
// lowerings are held to the same artifact.
//
// The weight is deliberately not a splat: hipsr.constant rejects splat values,
// because the ONNX importer never emits one.
//
// Everything below is what the hipsr pipeline is expected to leave standing.
// The four entry points come from generate-interface and are the artifact's
// whole public surface; @main_graph is the (state, inputs) wrapper
// hipsr-main-graph-abi wraps around the lowered graph.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 48>
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 2, 4>]
// CHECK-SAME: hipdnn.num_op_state_slots = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 2, 3>]
// CHECK-DAG: llvm.func @wrap_hipblasLtMatmul
// CHECK-DAG: llvm.func @hipdnn_ep_op_state_construct_matmul
// CHECK-DAG: llvm.func @hipdnn_ep_alloc_output
// CHECK-DAG: llvm.func private @main_graph(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @inference_init
// CHECK-DAG: llvm.func @inference_compute
// CHECK-DAG: llvm.func @inference_cleanup
// CHECK-DAG: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<2x4xf32> {onnx.name = "input_0"}) -> (tensor<2x3xf32> {onnx.name = "output_0"}) {
    %weight = "onnx.Constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 0.0, 1.0]]> : tensor<4x3xf32>} : () -> tensor<4x3xf32>
    %result = "onnx.MatMul"(%arg0, %weight) {onnx_node_name = "MatMul_0"} : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    "onnx.Return"(%result) : (tensor<2x3xf32>) -> ()
  }
}
