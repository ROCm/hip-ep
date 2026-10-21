// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --canonicalize | FileCheck %s

module {
  func.func @main_graph(%x: tensor<f32>) -> tensor<f32> {
    return %x : tensor<f32>
  }

  // Nemotron expands eight key heads to 64 without changing batch or sequence.
  func.func @inline_expand(%x: tensor<?x?x8x1x128xf32>) -> tensor<?x?x8x8x128xf32> {
    %shape = hip.constant {value = dense<[1, 1, 1, 8, 1]> : tensor<5xi64>} : tensor<5xi64>
    %out = "onnx.Expand"(%x, %shape) : (tensor<?x?x8x1x128xf32>, tensor<5xi64>) -> tensor<?x?x8x8x128xf32>
    return %out : tensor<?x?x8x8x128xf32>
  }
  // CHECK-LABEL: func.func @inline_expand
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.dim
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.dim
  // CHECK-NOT: hip.readback_scalar
  // CHECK: tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x8x8x128xf32>
  // CHECK: hip.expand
  // CHECK-NOT: hip.readback_scalar
  // CHECK: return

  // A dynamic reshape must resolve the inline -1 without a device read.
  func.func @inline_reshape(%x: tensor<?x?x4096xf16>) -> tensor<?x8x512xf16> {
    %shape = hip.constant {value = dense<[-1, 8, 512]> : tensor<3xi64>} : tensor<3xi64>
    %out = "onnx.Reshape"(%x, %shape) : (tensor<?x?x4096xf16>, tensor<3xi64>) -> tensor<?x8x512xf16>
    return %out : tensor<?x8x512xf16>
  }
  // CHECK-LABEL: func.func @inline_reshape
  // CHECK-NOT: hip.readback_scalar
  // CHECK: return

  // Rank-zero integer and float values use the same helper.
  func.func @inline_scalar_i64(%limit: tensor<i64>) -> tensor<?xi64> {
    %start = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
    %delta = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
    %out = "onnx.Range"(%start, %limit, %delta) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    return %out : tensor<?xi64>
  }
  // CHECK-LABEL: func.func @inline_scalar_i64
  // CHECK-NOT: hip.readback_scalar
  // CHECK: hip.readback_scalar(%{{.*}}, %arg1 : tensor<i64>) -> i64
  // CHECK-NOT: hip.readback_scalar
  // CHECK: hip.range
  // CHECK: return

  func.func @inline_scalar_f32(%limit: tensor<f32>) -> tensor<?xf32> {
    %start = hip.constant {value = dense<0.0> : tensor<f32>} : tensor<f32>
    %delta = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
    %out = "onnx.Range"(%start, %limit, %delta) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @inline_scalar_f32
  // CHECK-NOT: hip.readback_scalar
  // CHECK: hip.readback_scalar(%{{.*}}, %arg1 : tensor<f32>) -> f32
  // CHECK-NOT: hip.readback_scalar
  // CHECK: hip.range
  // CHECK: return

  // Do not read file-backed or process-memory constants during conversion.
  func.func @external_expand(%x: tensor<1xf32>) -> tensor<?xf32> {
    %shape = hip.constant {location = "shape.bin", offset = 0 : i64, size = 8 : i64} : tensor<1xi64>
    %out = "onnx.Expand"(%x, %shape) : (tensor<1xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @external_expand
  // CHECK: hip.readback_scalar
  // CHECK: hip.expand

  func.func @memory_expand(%x: tensor<1xf32>) -> tensor<?xf32> {
    %shape = hip.constant {memory_address = 4096 : i64, size = 8 : i64} : tensor<1xi64>
    %out = "onnx.Expand"(%x, %shape) : (tensor<1xf32>, tensor<1xi64>) -> tensor<?xf32>
    return %out : tensor<?xf32>
  }
  // CHECK-LABEL: func.func @memory_expand
  // CHECK: hip.readback_scalar
  // CHECK: hip.expand
}
