// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Axis-0 scalar Gather with a compile-time index lowers to tensor.extract_slice
// (zero-copy subview), including when the index was externalized to a global.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  memref.global "private" @hip_ext_constant_q_idx : memref<i64> {hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 8 : i64}, hip.compile_time_scalar = 0 : i64}
  memref.global "private" @hip_ext_constant_k_idx : memref<i64> {hip.external_data = {index = 1 : i64, offset = 8 : i64, size = 8 : i64}, hip.compile_time_scalar = 1 : i64}

  func.func @main_graph(%arg0: !hip.context, %data: tensor<3x4x2xf16>) -> (tensor<4x2xf16>, tensor<4x2xf16>) {
    %q_idx = bufferization.to_tensor @hip_ext_constant_q_idx restrict : memref<i64> to tensor<i64>
    %k_idx = bufferization.to_tensor @hip_ext_constant_k_idx restrict : memref<i64> to tensor<i64>

    %q = "onnx.Gather"(%data, %q_idx) {axis = 0 : si64} : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>
    %k = "onnx.Gather"(%data, %k_idx) {axis = 0 : si64} : (tensor<3x4x2xf16>, tensor<i64>) -> tensor<4x2xf16>

    // CHECK: tensor.extract_slice {{.*}}[0, 0, 0] [1, 4, 2] [1, 1, 1]
    // CHECK: tensor.extract_slice {{.*}}[1, 0, 0] [1, 4, 2] [1, 1, 1]
    // CHECK-NOT: hip.gather
    // CHECK-NOT: tensor.extract
    // CHECK-NOT: memref.load

    return %q, %k : tensor<4x2xf16>, tensor<4x2xf16>
  }
}
