// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bf16 Cast E2E full pipeline.
// Covers the cast pairs used by bf16 LLMs (e.g. Qwen torch.aten.to.dtype
// f32 <-> bf16) plus bf16 <-> f16 and int64 -> bf16. Each cast reads its own
// graph input so canonicalize cannot fold a cast chain away.
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.Cast -> hip.cast (to = 16 for bf16)
// 2. convert-hip-to-llvm: hip.cast -> wrap_cast with bf16 element types
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 4
// CHECK-SAME: hipdnn.output_count = 4
// CHECK: llvm.func @wrap_cast
//
// Each cast passes (src, dst) as HIPDNN_EP_DATATYPE_* constants emitted right
// before the call: FLOAT=0, HALF=1, BFLOAT16=2, INT64=4. Anchor on the output
// allocation of each cast so the constants cannot match an earlier value.
//
// f32 -> bf16
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[S0:%[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: [[D0:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_cast({{.*}}, [[S0]], [[D0]])
// bf16 -> f32
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[S1:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[D1:%[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_cast({{.*}}, [[S1]], [[D1]])
// i64 -> bf16
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[S2:%[0-9]+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT: [[D2:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_cast({{.*}}, [[S2]], [[D2]])
// bf16 -> f16
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[S3:%[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[D3:%[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_cast({{.*}}, [[S3]], [[D3]])
// CHECK-NOT: llvm.call @wrap_cast
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.Cast
module {
  func.func @main_graph(%arg0: tensor<1x512x64xf32> {onnx.name = "f32_in"},
                        %arg1: tensor<1x512x896xbf16> {onnx.name = "bf16_in"},
                        %arg2: tensor<1x512xi64> {onnx.name = "i64_in"},
                        %arg3: tensor<1x512x64xbf16> {onnx.name = "bf16_in2"})
      -> (tensor<1x512x64xbf16> {onnx.name = "f32_to_bf16"},
          tensor<1x512x896xf32> {onnx.name = "bf16_to_f32"},
          tensor<1x512xbf16> {onnx.name = "i64_to_bf16"},
          tensor<1x512x64xf16> {onnx.name = "bf16_to_f16"}) {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16, onnx_node_name = "cast_f32_to_bf16"} : (tensor<1x512x64xf32>) -> tensor<1x512x64xbf16>
    %1 = "onnx.Cast"(%arg1) {saturate = 1 : si64, to = f32, onnx_node_name = "cast_bf16_to_f32"} : (tensor<1x512x896xbf16>) -> tensor<1x512x896xf32>
    %2 = "onnx.Cast"(%arg2) {saturate = 1 : si64, to = bf16, onnx_node_name = "cast_i64_to_bf16"} : (tensor<1x512xi64>) -> tensor<1x512xbf16>
    %3 = "onnx.Cast"(%arg3) {saturate = 1 : si64, to = f16, onnx_node_name = "cast_bf16_to_f16"} : (tensor<1x512x64xbf16>) -> tensor<1x512x64xf16>
    "onnx.Return"(%0, %1, %2, %3) : (tensor<1x512x64xbf16>, tensor<1x512x896xf32>, tensor<1x512xbf16>, tensor<1x512x64xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
