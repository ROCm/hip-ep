// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 Add / Mul E2E full pipeline.
// Covers the bf16 elementwise ops used by bf16 LLMs (e.g. Qwen
// torch.aten.add.Tensor / torch.aten.mul.Tensor): a same-shape rank-3 Add and
// a rank-4 Mul broadcasting over the head axis. Each op reads its own graph
// inputs so canonicalize cannot fold anything away.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Add / onnx.Mul -> hip.add / hip.mul
// 2. convert-hip-to-llvm: hip.add / hip.mul -> wrap_elementwise with bf16
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 4
// CHECK-SAME: hipdnn.output_count = 2
// data_type is HIPDNN_EP_DATATYPE_BFLOAT16 (2), followed by the tensor op.
// CHECK: llvm.func @wrap_elementwise
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[DT0:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[OP0:%[0-9]+]] = llvm.mlir.constant({{[0-9]+}} : i64) : i64
// CHECK-NEXT: llvm.call @wrap_elementwise({{.*}}, [[DT0]], [[OP0]])
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[DT1:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[OP1:%[0-9]+]] = llvm.mlir.constant({{[0-9]+}} : i64) : i64
// CHECK-NEXT: llvm.call @wrap_elementwise({{.*}}, [[DT1]], [[OP1]])
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.Mul
module {
  func.func @main_graph(%arg0: tensor<1x512x896xbf16> {onnx.name = "add_a"},
                        %arg1: tensor<1x512x896xbf16> {onnx.name = "add_b"},
                        %arg2: tensor<1x14x512x512xbf16> {onnx.name = "mul_a"},
                        %arg3: tensor<1x1x512x512xbf16> {onnx.name = "mul_b"})
      -> (tensor<1x512x896xbf16> {onnx.name = "add_out"},
          tensor<1x14x512x512xbf16> {onnx.name = "mul_bcast_out"}) {
    %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "add_same_shape"} : (tensor<1x512x896xbf16>, tensor<1x512x896xbf16>) -> tensor<1x512x896xbf16>
    %1 = "onnx.Mul"(%arg2, %arg3) {onnx_node_name = "mul_bcast"} : (tensor<1x14x512x512xbf16>, tensor<1x1x512x512xbf16>) -> tensor<1x14x512x512xbf16>
    "onnx.Return"(%0, %1) : (tensor<1x512x896xbf16>, tensor<1x14x512x512xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
