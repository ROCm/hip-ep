// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test ConstantOfShape E2E full pipeline.
// onnx.ConstantOfShape is folded to a splat arith.constant at compile time
// because its shape input is itself a compile-time constant (typical of
// transformer KV/mask initialisation).  The fold is then absorbed by the
// ordinary constant lowering path (inlined or externalised depending on
// size), so no dedicated runtime symbol is required.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 0
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.ConstantOfShape
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph() -> (tensor<2x3xf32> {onnx.name = "output"}) {
    %shape = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
    %0 = "onnx.ConstantOfShape"(%shape) {
      onnx_node_name = "cofs_node",
      value = dense<1.000000e+00> : tensor<1xf32>
    } : (tensor<2xi64>) -> tensor<2x3xf32>
    "onnx.Return"(%0) : (tensor<2x3xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
