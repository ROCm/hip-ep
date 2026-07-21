// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test onnx.LpNormalization E2E full pipeline.
// This IR represents an ONNX LpNormalization operation as it would be
// imported via onnx-mlir / morphizen mlir-imp.
//
// The input has a static trailing extent (N=3) and p=2 over the trailing
// axis, so it takes the fused fast path: LpNormalization -> one
// SimplifiedLayerNormalization (scale=1/sqrt(N), epsilon=0) ->
// hip.rms_norm -> wrap_miopenT5LayerNormForward. No Mul/ReduceSum/Sqrt
// decomposition is emitted for this shape.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.LpNormalization (p=2, trailing, static N)
//    fuses into a single hip.rms_norm.
// 2. canonicalize / memory-pooling: pool output buffer into single allocation
// 3. convert-hip-to-llvm: hip.rms_norm -> wrap_miopenT5LayerNormForward
// 4. generate-interface: create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-DAG: llvm.func @wrap_miopenT5LayerNormForward
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
