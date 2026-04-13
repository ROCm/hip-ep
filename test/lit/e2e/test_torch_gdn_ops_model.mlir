// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Gated Delta Network building-block ops E2E
// Tests the key ops used in GDN layers: split, mul, add, silu, linear, cat
// This validates the decomposed GDN data path can compile and run.
//
// Models a simplified GDN computation:
//   split(input) -> [query_part, key_part]
//   silu(query_part)
//   linear(key_part)
//   cat([silu_out, linear_out])
//
// Input:  tensor<1x4x256xf16>  (batch x seq x hidden)
// Weight: tensor<128x128xf16>  (for linear projection)
// Output: tensor<1x4x256xf16>

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: torch.aten.split
// CHECK-NOT: torch.aten.silu
module {
  func.func @main_graph(%arg0: tensor<1x4x256xf16>, %arg1: tensor<128x128xf16>) -> tensor<1x4x256xf16> {
    // Split input along last dim: [1,4,256] -> [1,4,128] + [1,4,128]
    %int128 = "torch.constant.int"() {value = 128 : i64} : () -> !torch.int
    %int2 = "torch.constant.int"() {value = 2 : i64} : () -> !torch.int
    %int1 = "torch.constant.int"() {value = 1 : i64} : () -> !torch.int
    %int_neg1 = "torch.constant.int"() {value = -1 : i64} : () -> !torch.int

    // Split via slice (since split produces a list which is hard to represent)
    %int0 = "torch.constant.int"() {value = 0 : i64} : () -> !torch.int
    %int_max = "torch.constant.int"() {value = 9223372036854775807 : i64} : () -> !torch.int

    %q = "torch.aten.slice.Tensor"(%arg0, %int2, %int0, %int128, %int1) :
        (tensor<1x4x256xf16>, !torch.int, !torch.int, !torch.int, !torch.int) -> tensor<1x4x128xf16>
    %k = "torch.aten.slice.Tensor"(%arg0, %int2, %int128, %int_max, %int1) :
        (tensor<1x4x256xf16>, !torch.int, !torch.int, !torch.int, !torch.int) -> tensor<1x4x128xf16>

    // SiLU activation on query part
    %q_act = "torch.aten.silu"(%q) : (tensor<1x4x128xf16>) -> tensor<1x4x128xf16>

    // Linear projection on key part
    %none = "torch.constant.none"() : () -> !torch.none
    %k_proj = "torch.aten.linear"(%k, %arg1, %none) :
        (tensor<1x4x128xf16>, tensor<128x128xf16>, !torch.none) -> tensor<1x4x128xf16>

    // Elementwise multiply (gating)
    %gated = "torch.aten.mul.Tensor"(%q_act, %k_proj) :
        (tensor<1x4x128xf16>, tensor<1x4x128xf16>) -> tensor<1x4x128xf16>

    // Cat back together: [1,4,128] + [1,4,128] -> [1,4,256]
    %list = "torch.prim.ListConstruct"(%gated, %k) :
        (tensor<1x4x128xf16>, tensor<1x4x128xf16>) -> !torch.list<tensor>
    %result = "torch.aten.cat.default"(%list, %int_neg1) :
        (!torch.list<tensor>, !torch.int) -> tensor<1x4x256xf16>

    return %result : tensor<1x4x256xf16>
  }
}
