// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test CastLike E2E full pipeline
// CastLike casts the first input to the element type of the second input.
// The second operand is a *type donor* whose data is never read; the target
// dtype is statically known from the result type. convert-onnx-to-hip runs a
// pre-metadata simplification that rewrites onnx.CastLike → onnx.Cast and
// drops the now-dead function argument, so the compiled DLL has only one
// real input. The downstream onnx.Cast → hip.cast → wrap_cast path is
// shared with plain ONNX Cast.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.CastLike → onnx.Cast (+ dead-arg drop) → hip.cast
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_cast
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.CastLike
module {
  func.func @main_graph(%arg0: tensor<3x4xf32> {onnx.name = "input"}, %arg1: tensor<0xf16> {onnx.name = "target_type"}) -> (tensor<3x4xf16> {onnx.name = "output"}) {
    %0 = "onnx.CastLike"(%arg0, %arg1) {onnx_node_name = "castlike_node"} : (tensor<3x4xf32>, tensor<0xf16>) -> tensor<3x4xf16>
    "onnx.Return"(%0) : (tensor<3x4xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
