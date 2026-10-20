// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify --hip-group-matmul-nbits folds symmetric int4 hip.matmul_nbits ops
// that share an activation into hip.matmul_nbits_group (at most 4 members),
// gives every member its own destination, and leaves ineligible ops alone.
// ============================================================================

// RUN: hip-mlir-opt %s --simplify-onnx --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip --hip-infer-shapes --hip-group-matmul-nbits \
// RUN:   --split-input-file | FileCheck %s

// Three projections of one activation -> one group, results in member order.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.matmul_nbits(
// CHECK: %[[G:.*]]:3 = hip.matmul_nbits_group(%{{.*}}) ins(%{{.*}} : tensor<?x?x256xf16>)
// CHECK-SAME: Ns = array<i64: 64, 32, 32>
// CHECK-NOT: hip.matmul_nbits(
// CHECK: return %[[G]]#0, %[[G]]#1, %[[G]]#2
module {
  func.func @main_graph(%arg0: tensor<?x?x256xf16> {onnx.name = "A"}) -> (tensor<?x?x64xf16> {onnx.name = "Y0"}, tensor<?x?x32xf16> {onnx.name = "Y1"}, tensor<?x?x32xf16> {onnx.name = "Y2"}) {
    %B0 = "onnx.Constant"() {value = dense<1> : tensor<64x8x16xui8>} : () -> tensor<64x8x16xui8>
    %S0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<64x8xf16>} : () -> tensor<64x8xf16>
    %B1 = "onnx.Constant"() {value = dense<2> : tensor<32x8x16xui8>} : () -> tensor<32x8x16xui8>
    %S1 = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<32x8xf16>} : () -> tensor<32x8xf16>
    %B2 = "onnx.Constant"() {value = dense<3> : tensor<32x8x16xui8>} : () -> tensor<32x8x16xui8>
    %S2 = "onnx.Constant"() {value = dense<3.000000e+00> : tensor<32x8xf16>} : () -> tensor<32x8xf16>
    %Y0 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 64 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<?x?x256xf16>, tensor<64x8x16xui8>, tensor<64x8xf16>) -> tensor<?x?x64xf16>
    %Y1 = "onnx.Custom"(%arg0, %B1, %S1) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<?x?x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<?x?x32xf16>
    %Y2 = "onnx.Custom"(%arg0, %B2, %S2) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<?x?x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<?x?x32xf16>
    "onnx.Return"(%Y0, %Y1, %Y2) : (tensor<?x?x64xf16>, tensor<?x?x32xf16>, tensor<?x?x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// -----

// Different activations never share a group.
// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.matmul_nbits_group
// CHECK-COUNT-2: hip.matmul_nbits(
module {
  func.func @main_graph(%arg0: tensor<1x1x256xf16> {onnx.name = "A0"}, %arg1: tensor<1x1x256xf16> {onnx.name = "A1"}) -> (tensor<1x1x32xf16> {onnx.name = "Y0"}, tensor<1x1x32xf16> {onnx.name = "Y1"}) {
    %B0 = "onnx.Constant"() {value = dense<1> : tensor<32x8x16xui8>} : () -> tensor<32x8x16xui8>
    %S0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x8xf16>} : () -> tensor<32x8xf16>
    %Y0 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y1 = "onnx.Custom"(%arg1, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    "onnx.Return"(%Y0, %Y1) : (tensor<1x1x32xf16>, tensor<1x1x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// -----

// A member with a bias is not something the grouped kernel runs; the other
// two still group.
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.matmul_nbits_group
// CHECK-SAME: Ns = array<i64: 32, 32>
// CHECK: hip.matmul_nbits(
// CHECK-NOT: hip.matmul_nbits_group
module {
  func.func @main_graph(%arg0: tensor<1x1x256xf16> {onnx.name = "A"}) -> (tensor<1x1x32xf16> {onnx.name = "Y0"}, tensor<1x1x32xf16> {onnx.name = "Y1"}, tensor<1x1x32xf16> {onnx.name = "Y2"}) {
    %B0 = "onnx.Constant"() {value = dense<1> : tensor<32x8x16xui8>} : () -> tensor<32x8x16xui8>
    %S0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x8xf16>} : () -> tensor<32x8xf16>
    %zp = "onnx.NoValue"() {value} : () -> none
    %bias = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32xf16>} : () -> tensor<32xf16>
    %Y0 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y1 = "onnx.Custom"(%arg0, %B0, %S0, %zp, %zp, %bias) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>, none, none, tensor<32xf16>) -> tensor<1x1x32xf16>
    %Y2 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    "onnx.Return"(%Y0, %Y1, %Y2) : (tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// -----

// Projections over 64 MiB of weights (here 32768x8192, 128 MiB) stay single
// ops; the small ones next to them still group.
// CHECK-LABEL: func.func @main_graph
// CHECK-COUNT-2: hip.matmul_nbits(
// CHECK: hip.matmul_nbits_group
// CHECK-SAME: Ns = array<i64: 32, 32>
// CHECK-NOT: hip.matmul_nbits
module {
  func.func @main_graph(%arg0: tensor<1x1x8192xf16> {onnx.name = "A"}) -> (tensor<1x1x32768xf16> {onnx.name = "Y0"}, tensor<1x1x32768xf16> {onnx.name = "Y1"}, tensor<1x1x32xf16> {onnx.name = "Y2"}, tensor<1x1x32xf16> {onnx.name = "Y3"}) {
    %BL = "onnx.Constant"() {value = dense<1> : tensor<32768x64x64xui8>} : () -> tensor<32768x64x64xui8>
    %SL = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32768x64xf16>} : () -> tensor<32768x64xf16>
    %BS = "onnx.Constant"() {value = dense<2> : tensor<32x64x64xui8>} : () -> tensor<32x64x64xui8>
    %SS = "onnx.Constant"() {value = dense<2.000000e+00> : tensor<32x64xf16>} : () -> tensor<32x64xf16>
    %Y0 = "onnx.Custom"(%arg0, %BL, %SL) {function_name = "MatMulNBits", K = 8192 : si64, N = 32768 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 128 : si64, domain_name = "com.microsoft"} : (tensor<1x1x8192xf16>, tensor<32768x64x64xui8>, tensor<32768x64xf16>) -> tensor<1x1x32768xf16>
    %Y1 = "onnx.Custom"(%arg0, %BL, %SL) {function_name = "MatMulNBits", K = 8192 : si64, N = 32768 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 128 : si64, domain_name = "com.microsoft"} : (tensor<1x1x8192xf16>, tensor<32768x64x64xui8>, tensor<32768x64xf16>) -> tensor<1x1x32768xf16>
    %Y2 = "onnx.Custom"(%arg0, %BS, %SS) {function_name = "MatMulNBits", K = 8192 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 128 : si64, domain_name = "com.microsoft"} : (tensor<1x1x8192xf16>, tensor<32x64x64xui8>, tensor<32x64xf16>) -> tensor<1x1x32xf16>
    %Y3 = "onnx.Custom"(%arg0, %BS, %SS) {function_name = "MatMulNBits", K = 8192 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 128 : si64, domain_name = "com.microsoft"} : (tensor<1x1x8192xf16>, tensor<32x64x64xui8>, tensor<32x64xf16>) -> tensor<1x1x32xf16>
    "onnx.Return"(%Y0, %Y1, %Y2, %Y3) : (tensor<1x1x32768xf16>, tensor<1x1x32768xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// -----

// Five candidates: one group of four, the fifth stays a plain op.
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.matmul_nbits_group
// CHECK-SAME: Ns = array<i64: 32, 32, 32, 32>
// CHECK: hip.matmul_nbits(
// CHECK-NOT: hip.matmul_nbits
module {
  func.func @main_graph(%arg0: tensor<1x1x256xf16> {onnx.name = "A"}) -> (tensor<1x1x32xf16> {onnx.name = "Y0"}, tensor<1x1x32xf16> {onnx.name = "Y1"}, tensor<1x1x32xf16> {onnx.name = "Y2"}, tensor<1x1x32xf16> {onnx.name = "Y3"}, tensor<1x1x32xf16> {onnx.name = "Y4"}) {
    %B0 = "onnx.Constant"() {value = dense<1> : tensor<32x8x16xui8>} : () -> tensor<32x8x16xui8>
    %S0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<32x8xf16>} : () -> tensor<32x8xf16>
    %Y0 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y1 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y2 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y3 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    %Y4 = "onnx.Custom"(%arg0, %B0, %S0) {function_name = "MatMulNBits", K = 256 : si64, N = 32 : si64, accuracy_level = 4 : si64, bits = 4 : si64, block_size = 32 : si64, domain_name = "com.microsoft"} : (tensor<1x1x256xf16>, tensor<32x8x16xui8>, tensor<32x8xf16>) -> tensor<1x1x32xf16>
    "onnx.Return"(%Y0, %Y1, %Y2, %Y3, %Y4) : (tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>, tensor<1x1x32xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
