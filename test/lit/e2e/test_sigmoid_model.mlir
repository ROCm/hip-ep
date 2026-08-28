// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test Sigmoid E2E full pipeline from real Llama-3.1-8B SiLU activation
// This IR was imported from Sigmoid_fix.onnx via onnx-mlir
// The model has one onnx.Sigmoid with f16 tensors
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Sigmoid → hip.sigmoid
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_sigmoid
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Sigmoid
module {
  func.func @main_graph(%arg0: tensor<1x128x14336xf16> {onnx.name = "/model/layers.0/mlp/gate_proj/MatMul/output_0"}) -> (tensor<1x128x14336xf16> {onnx.name = "/model/layers.0/mlp/act_fn/Sigmoid/output_0"}) {
    %0 = "onnx.Sigmoid"(%arg0) {onnx_node_name = "/model/layers.0/mlp/act_fn/Sigmoid"} : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    "onnx.Return"(%0) : (tensor<1x128x14336xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
