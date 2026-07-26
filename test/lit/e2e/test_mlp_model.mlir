// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s
// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline | FileCheck %s --check-prefix=POOLED
// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: hip-mlir-opt %s --onnx-to-hip-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT-BUFFERIZE
// RUN: env HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE=1 hip-mlir-opt %s --onnx-to-hip-pipeline --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=COPY-BEFORE-WRITE

// Test MLP E2E pipeline from real Llama-3.1-8B MLP subgraph
// The model has MatMul + Sigmoid + Mul (SiLU gate) ops with constant weights.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx ops → hip ops
// 2. canonicalize: Simplify redundant operations
// 3. memory-pooling: Pool output buffer into single allocation
// 4. convert-hip-to-llvm: HIP ops → LLVM runtime calls
// 5. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_hipblasLtMatmul
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.MatMul
// CHECK-NOT: onnx.Sigmoid
// CHECK-NOT: onnx.Mul

// Production pipeline schedule: ownership-based deallocation must not return.
// PIPELINE-LABEL: Pass Manager with
// PIPELINE-NOT: buffer-deallocation
// PIPELINE: hip-loop-body-to-out-params
// PIPELINE-NOT: buffer-deallocation
// PIPELINE: hip-use-output-allocator
// PIPELINE-NOT: buffer-deallocation
// PIPELINE: hip-pool-allocs
// PIPELINE-NOT: buffer-deallocation

// Multi-intermediate production IR: transients are pooled and the graph output
// is runtime-owned, with no per-buffer allocation/free or deallocation clone.
// POOLED-LABEL: func.func @main_graph
// POOLED-NOT: memref.alloc
// POOLED-NOT: memref.dealloc
// POOLED-NOT: bufferization.dealloc
// POOLED-NOT: bufferization.clone
// POOLED-NOT: hip.alloc(
// POOLED-NOT: hip.free
// POOLED: hip.get_pool
// POOLED-NOT: memref.alloc
// POOLED-NOT: bufferization.dealloc
// POOLED-NOT: bufferization.clone
// POOLED: hip.alloc_output
// POOLED-NOT: memref.alloc
// POOLED-NOT: memref.dealloc
// POOLED-NOT: bufferization.dealloc
// POOLED-NOT: bufferization.clone
// POOLED-NOT: hip.alloc(
// POOLED-NOT: hip.free
// POOLED: return

// The huge-graph escape hatch is opt-in: default retains One-Shot analysis;
// setting the process environment enables copy-before-write.
// DEFAULT-BUFFERIZE: one-shot-bufferize{{.*}}copy-before-write=false
// COPY-BEFORE-WRITE: one-shot-bufferize{{.*}}copy-before-write=true
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_0"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/mlp/down_proj/MatMul/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<5.000000e-03> : tensor<4096x14336xf16>} : () -> tensor<4096x14336xf16>
    %1 = "onnx.Constant"() {value = dense<5.000000e-03> : tensor<4096x14336xf16>} : () -> tensor<4096x14336xf16>
    %2 = "onnx.Constant"() {value = dense<5.000000e-03> : tensor<14336x4096xf16>} : () -> tensor<14336x4096xf16>
    %3 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "/model/layers.0/mlp/gate_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x128x14336xf16>
    %4 = "onnx.MatMul"(%arg0, %1) {onnx_node_name = "/model/layers.0/mlp/up_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x128x14336xf16>
    %5 = "onnx.Sigmoid"(%3) {onnx_node_name = "/model/layers.0/mlp/act_fn/Sigmoid"} : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %6 = "onnx.Mul"(%3, %5) {onnx_node_name = "/model/layers.0/mlp/act_fn/Mul"} : (tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %7 = "onnx.Mul"(%6, %4) {onnx_node_name = "/model/layers.0/mlp/Mul"} : (tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %8 = "onnx.MatMul"(%7, %2) {onnx_node_name = "/model/layers.0/mlp/down_proj/MatMul"} : (tensor<1x128x14336xf16>, tensor<14336x4096xf16>) -> tensor<1x128x4096xf16>
    "onnx.Return"(%8) : (tensor<1x128x4096xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
