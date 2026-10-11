// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Axis-0 scalar Gather with a compile-time index lowers to tensor.extract_slice
// (zero-copy subview). Negative cases fall through to hip.gather when the index
// is dynamic, the axis is not 0, or the index tensor is not scalar/len-1.
//
// Coverage:
//   - externalized scalar indices tagged with hip.compile_time_scalar
//   - inline onnx.Constant scalar indices
//   - negative axis normalized to 0 on rank-1 data
//   - dynamic / untagged extern index -> hip.gather
//   - axis != 0, multi-index, and rank>1 index vectors -> hip.gather

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  memref.global "private" @hip_ext_constant_q_idx : memref<i64> {hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 8 : i64}, hip.compile_time_scalar = 0 : i64}
  memref.global "private" @hip_ext_constant_k_idx : memref<i64> {hip.external_data = {index = 1 : i64, offset = 8 : i64, size = 8 : i64}, hip.compile_time_scalar = 1 : i64}
  memref.global "private" @hip_ext_constant_dyn_idx : memref<i64> {hip.external_data = {index = 2 : i64, offset = 16 : i64, size = 8 : i64}}

  func.func @main_graph(%arg0: !hip.context, %data: tensor<3x4x2xf16>) -> (tensor<4x2xf16>, tensor<4x2xf16>) {
    %q_buf = memref.get_global @hip_ext_constant_q_idx : memref<i64>
    %k_buf = memref.get_global @hip_ext_constant_k_idx : memref<i64>
    %q_idx = bufferization.to_tensor %q_buf restrict : memref<i64> to tensor<i64>
    %k_idx = bufferization.to_tensor %k_buf restrict : memref<i64> to tensor<i64>

    %q = "onnx.Gather"(%data, %q_idx) {axis = 0 : si64} : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>
    %k = "onnx.Gather"(%data, %k_idx) {axis = 0 : si64} : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>

    // CHECK-LABEL: func.func @main_graph
    // CHECK: tensor.extract_slice {{.*}}[0, 0, 0] [1, 4, 2] [1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}}[1, 0, 0] [1, 4, 2] [1, 1, 1]
    // CHECK-NOT: hip.gather
    // CHECK-NOT: tensor.extract
    // CHECK-NOT: memref.load

    return %q, %k : tensor<4x2xf16>, tensor<4x2xf16>
  }

  // Inline scalar constant index.
  func.func @test_inline_scalar_idx(%data: tensor<2x3xf16>) -> tensor<3xf16> {
    %idx = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<2x3xf16>, tensor<i64>) -> tensor<3xf16>
    // CHECK-LABEL: func.func @test_inline_scalar_idx
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [1, 3] [1, 1]
    // CHECK-NOT: hip.gather
    return %out : tensor<3xf16>
  }

  // Negative: len-1 index vector with squeezed output rank does not match the
  // axis-0 scalar Gather shape contract, so the fold stays on hip.gather.
  func.func @test_no_fold_len1_idx(%data: tensor<2x3xf16>) -> tensor<3xf16> {
    %idx = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>} : () -> tensor<1xi64>
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<2x3xf16>, tensor<1xi64>) -> tensor<3xf16>
    // CHECK-LABEL: func.func @test_no_fold_len1_idx
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<3xf16>
  }

  // Negative axis on rank-1 data normalizes to axis 0.
  func.func @test_neg_axis_rank1(%data: tensor<4xf16>) -> tensor<f16> {
    %c = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %out = "onnx.Gather"(%data, %c) {axis = -1 : si64}
        : (tensor<4xf16>, tensor<i64>) -> tensor<f16>
    // CHECK-LABEL: func.func @test_neg_axis_rank1
    // CHECK: tensor.extract_slice {{.*}}[2] [1] [1]
    // CHECK-NOT: hip.gather
    return %out : tensor<f16>
  }

  // Negative: extern scalar without hip.compile_time_scalar stays dynamic.
  func.func @test_no_fold_untagged_extern(%data: tensor<3x4x2xf16>) -> tensor<4x2xf16> {
    %buf = memref.get_global @hip_ext_constant_dyn_idx : memref<i64>
    %idx = bufferization.to_tensor %buf restrict : memref<i64> to tensor<i64>
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>
    // CHECK-LABEL: func.func @test_no_fold_untagged_extern
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<4x2xf16>
  }

  // Negative: runtime index argument.
  func.func @test_no_fold_dynamic_idx(%data: tensor<3x4x2xf16>, %idx: tensor<i64>) -> tensor<4x2xf16> {
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>
    // CHECK-LABEL: func.func @test_no_fold_dynamic_idx
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<4x2xf16>
  }

  // Negative: axis != 0 even with a compile-time index.
  func.func @test_no_fold_nonzero_axis(%data: tensor<3x4x2xf16>) -> tensor<3x2xf16> {
    %idx = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %out = "onnx.Gather"(%data, %idx) {axis = 1 : si64}
        : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<3x2xf16>
    // CHECK-LABEL: func.func @test_no_fold_nonzero_axis
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<3x2xf16>
  }

  // Negative: len>1 index vector on axis 0.
  func.func @test_no_fold_multi_index(%data: tensor<3x4x2xf16>) -> tensor<2x4x2xf16> {
    %idx = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<3x4x2xf16>, tensor<2xi64>) -> tensor<2x4x2xf16>
    // CHECK-LABEL: func.func @test_no_fold_multi_index
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<2x4x2xf16>
  }

  // Negative: rank>1 index tensor with static dim != 1.
  func.func @test_no_fold_rank2_indices(%data: tensor<3x4x2xf16>) -> tensor<2x2x4x2xf16> {
    %idx = "onnx.Constant"() {value = dense<[[0, 1], [2, 0]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
    %out = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<3x4x2xf16>, tensor<2x2xi64>) -> tensor<2x2x4x2xf16>
    // CHECK-LABEL: func.func @test_no_fold_rank2_indices
    // CHECK: hip.gather
    // CHECK-NOT: tensor.extract_slice
    return %out : tensor<2x2x4x2xf16>
  }
}
