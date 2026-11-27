// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// End-to-end test for a bf16 MatMul.
//
// Pipeline: onnx.MatMul -> hipsr.matmul -> llvm.call @wrap_hipblasLtMatmul.
//
// The runtime picks the hipBLASLt type from the data_type argument. bf16 and
// f16 are both 2 bytes, so the call must carry HIPDNN_EP_DATATYPE_BFLOAT16 (2),
// not an element size. The checks anchor on hipdnn_ep_alloc_output and then
// follow the constants up to the call, so the output dtype of alloc_output or a
// shape dim of 2 cannot satisfy the check.

// CHECK: llvm.func @wrap_hipblasLtMatmul
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: %[[DT:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_hipblasLtMatmul({{.*}}, %[[DT]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) :
// CHECK-NOT: onnx.MatMul

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xbf16> {onnx.name = "input"}) -> (tensor<1x128x1024xbf16> {onnx.name = "output"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x1024xbf16>} : () -> tensor<4096x1024xbf16>
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "/k_proj/MatMul"} : (tensor<1x128x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<1x128x1024xbf16>
    "onnx.Return"(%1) : (tensor<1x128x1024xbf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
