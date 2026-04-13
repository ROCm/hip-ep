// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch RoPE (Rotary Position Embeddings) E2E via decomposed ops
// RoPE decomposes into: slice, mul, sub, add, cat
// Input: tensor<1x4x128xf16>  (batch x seq x hidden)
// Cos:   tensor<1x4x64xf16>   (batch x seq x half_hidden)
// Sin:   tensor<1x4x64xf16>   (batch x seq x half_hidden)
// Output: tensor<1x4x128xf16>
//
// Verifies all decomposed RoPE ops lower through the pipeline:
// 1. torch.aten.slice.Tensor → tensor.extract_slice
// 2. torch.aten.mul.Tensor → hip.mul
// 3. torch.aten.sub.Tensor → hip.sub
// 4. torch.aten.add.Tensor → hip.add
// 5. torch.aten.cat.default → tensor.insert_slice

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 3
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: torch.aten.slice
// CHECK-NOT: torch.aten.cat
module {
  func.func @main_graph(%arg0: tensor<1x4x128xf16>, %arg1: tensor<1x4x64xf16>, %arg2: tensor<1x4x64xf16>) -> tensor<1x4x128xf16> {
    // Slice: x1 = x[..., :64], x2 = x[..., 64:]
    %int2 = "torch.constant.int"() {value = 2 : i64} : () -> !torch.int
    %int0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
    %int64 = "torch.constant.int"() {value = 64 : i64} : () -> !torch.int
    %int_max = "torch.constant.int"() {value = 9223372036854775807 : i64} : () -> !torch.int
    %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %int_neg1 = "torch.constant.int"() {value = -1 : i64} : () -> !torch.int

    %x1 = "torch.aten.slice.Tensor"(%arg0, %int2, %int0, %int64, %int1) : (tensor<1x4x128xf16>, !torch.int, !torch.int, !torch.int, !torch.int) -> tensor<1x4x64xf16>
    %x2 = "torch.aten.slice.Tensor"(%arg0, %int2, %int64, %int_max, %int1) : (tensor<1x4x128xf16>, !torch.int, !torch.int, !torch.int, !torch.int) -> tensor<1x4x64xf16>

    // x1*cos - x2*sin
    %mul0 = "torch.aten.mul.Tensor"(%x1, %arg1) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
    %mul1 = "torch.aten.mul.Tensor"(%x2, %arg2) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
    %part1 = "torch.aten.sub.Tensor"(%mul0, %mul1, %int1) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>, !torch.int) -> tensor<1x4x64xf16>

    // x1*sin + x2*cos
    %mul2 = "torch.aten.mul.Tensor"(%x1, %arg2) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
    %mul3 = "torch.aten.mul.Tensor"(%x2, %arg1) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>) -> tensor<1x4x64xf16>
    %part2 = "torch.aten.add.Tensor"(%mul2, %mul3, %int1) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>, !torch.int) -> tensor<1x4x64xf16>

    // cat([part1, part2], dim=-1)
    %list = "torch.prim.ListConstruct"(%part1, %part2) : (tensor<1x4x64xf16>, tensor<1x4x64xf16>) -> !torch.list<tensor>
    %result = "torch.aten.cat.default"(%list, %int_neg1) : (!torch.list<tensor>, !torch.int) -> tensor<1x4x128xf16>

    return %result : tensor<1x4x128xf16>
  }
}
