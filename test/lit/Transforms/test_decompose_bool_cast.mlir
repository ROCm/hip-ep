// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify a non-bool hip.cast to bool becomes hip.not(hip.equal(x, 0)), since
// the cast kernels convert by value and a cast to bool means x != 0. A cast to
// bool has an i1 result or, as ONNX imports BOOL as ui8, `to = 9`.
//
// Test cases:
// 1. i64          - integer source, static shape
// 2. ui8_dyn      - unsigned byte source, dynamic dim sized from the input
// 3. f32          - float source gets a float zero
// 4. bf16         - widened to f32 first (hip.equal has no bf16 kernel)
// 5. i1           - already bool, cast survives
// 6. to_i64       - not a cast to bool, cast survives
// 7. f64          - no hip.equal kernel, cast survives
// 8. reshape      - tensor<1xi64> -> tensor<i1>: mask in the input shape, then
//                   an i1 -> i1 reshaping cast
// 9. onnx_bool    - i64 -> ui8 with to = 9 (ONNX BOOL) is rewritten
// 10. onnx_ui8    - ui8 -> ui8 with to = 9 is normalised too (5 -> 1)
// 11. onnx_uint8  - i64 -> ui8 with to = 3 is a by-value cast, survives
// 12. onnx_reshape - ui8 reshaping cast with to = 9 keeps a UINT8 (to = 2) cast
// ============================================================================

// RUN: hip-mlir-opt %s --pass-pipeline='builtin.module(func.func(hip-decompose-bool-cast))' | FileCheck %s

module {
  // CHECK-LABEL: func.func @i64
  // CHECK-SAME: (%[[CTX:arg[0-9]+]]: !hip.context, %[[X:arg[0-9]+]]: tensor<4xi64>)
  // CHECK: %[[ZERO:.*]] = hip.constant {value = dense<0> : tensor<i64>} : tensor<i64>
  // CHECK: %[[EQ:.*]] = hip.equal(%[[CTX]]) ins(%[[X]], %[[ZERO]] : tensor<4xi64>, tensor<i64>) outs(%{{.*}} : tensor<4xi1>)
  // CHECK: %[[NOT:.*]] = hip.not(%[[CTX]]) ins(%[[EQ]] : tensor<4xi1>) outs(%{{.*}} : tensor<4xi1>)
  // CHECK-NOT: hip.cast
  // CHECK: return %[[NOT]]
  func.func @i64(%ctx: !hip.context, %x: tensor<4xi64>) -> tensor<4xi1> {
    %e = tensor.empty() : tensor<4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<4xi64>) outs(%e : tensor<4xi1>) {to = 9 : i64} : tensor<4xi1>
    return %y : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @ui8_dyn
  // CHECK-SAME: %[[X:arg[0-9]+]]: tensor<?x4xui8>, %[[N:arg[0-9]+]]: index
  // CHECK: %[[E:.*]] = tensor.empty(%[[N]]) : tensor<?x4xi1>
  // CHECK: hip.constant {value = dense<0> : tensor<ui8>} : tensor<ui8>
  // CHECK: %[[D:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x4xui8>
  // CHECK: %[[E0:.*]] = tensor.empty(%[[D]]) : tensor<?x4xi1>
  // CHECK: %[[EQ:.*]] = hip.equal({{.*}}) ins(%[[X]], %{{.*}} : tensor<?x4xui8>, tensor<ui8>) outs(%[[E0]] : tensor<?x4xi1>)
  // CHECK: hip.not({{.*}}) ins(%[[EQ]] : tensor<?x4xi1>) outs(%[[E]] : tensor<?x4xi1>)
  // CHECK-NOT: hip.cast
  func.func @ui8_dyn(%ctx: !hip.context, %x: tensor<?x4xui8>, %n: index) -> tensor<?x4xi1> {
    %e = tensor.empty(%n) : tensor<?x4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<?x4xui8>) outs(%e : tensor<?x4xi1>) {to = 9 : i64} : tensor<?x4xi1>
    return %y : tensor<?x4xi1>
  }

  // CHECK-LABEL: func.func @f32
  // CHECK: hip.constant {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
  // CHECK: hip.equal
  // CHECK: hip.not
  // CHECK-NOT: hip.cast
  func.func @f32(%ctx: !hip.context, %x: tensor<4xf32>) -> tensor<4xi1> {
    %e = tensor.empty() : tensor<4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<4xf32>) outs(%e : tensor<4xi1>) {to = 9 : i64} : tensor<4xi1>
    return %y : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @bf16
  // CHECK: %[[W:.*]] = hip.cast({{.*}}) ins(%{{.*}} : tensor<4xbf16>) outs(%{{.*}} : tensor<4xf32>) {to = 1 : i64}
  // CHECK: %[[ZERO:.*]] = hip.constant {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
  // CHECK: hip.equal({{.*}}) ins(%[[W]], %[[ZERO]] : tensor<4xf32>, tensor<f32>)
  // CHECK: hip.not
  // CHECK-NOT: hip.cast
  func.func @bf16(%ctx: !hip.context, %x: tensor<4xbf16>) -> tensor<4xi1> {
    %e = tensor.empty() : tensor<4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<4xbf16>) outs(%e : tensor<4xi1>) {to = 9 : i64} : tensor<4xi1>
    return %y : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @i1
  // CHECK: hip.cast
  // CHECK-NOT: hip.equal
  func.func @i1(%ctx: !hip.context, %x: tensor<4xi1>) -> tensor<4xi1> {
    %e = tensor.empty() : tensor<4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<4xi1>) outs(%e : tensor<4xi1>) {to = 9 : i64} : tensor<4xi1>
    return %y : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @to_i64
  // CHECK: hip.cast
  // CHECK-NOT: hip.equal
  func.func @to_i64(%ctx: !hip.context, %x: tensor<4xi32>) -> tensor<4xi64> {
    %e = tensor.empty() : tensor<4xi64>
    %y = hip.cast(%ctx) ins(%x : tensor<4xi32>) outs(%e : tensor<4xi64>) {to = 7 : i64} : tensor<4xi64>
    return %y : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @f64
  // CHECK: hip.cast
  // CHECK-NOT: hip.equal
  func.func @f64(%ctx: !hip.context, %x: tensor<4xf64>) -> tensor<4xi1> {
    %e = tensor.empty() : tensor<4xi1>
    %y = hip.cast(%ctx) ins(%x : tensor<4xf64>) outs(%e : tensor<4xi1>) {to = 9 : i64} : tensor<4xi1>
    return %y : tensor<4xi1>
  }

  // CHECK-LABEL: func.func @reshape
  // CHECK: %[[EQ:.*]] = hip.equal({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<1xi64>, tensor<i64>) outs(%{{.*}} : tensor<1xi1>)
  // CHECK: %[[NOT:.*]] = hip.not({{.*}}) ins(%[[EQ]] : tensor<1xi1>) outs(%{{.*}} : tensor<1xi1>)
  // CHECK: hip.cast({{.*}}) ins(%[[NOT]] : tensor<1xi1>) outs(%{{.*}} : tensor<i1>) {to = 9 : i64}
  func.func @reshape(%ctx: !hip.context, %x: tensor<1xi64>) -> tensor<i1> {
    %e = tensor.empty() : tensor<i1>
    %y = hip.cast(%ctx) ins(%x : tensor<1xi64>) outs(%e : tensor<i1>) {to = 9 : i64} : tensor<i1>
    return %y : tensor<i1>
  }

  // CHECK-LABEL: func.func @onnx_bool
  // CHECK: %[[EQ:.*]] = hip.equal({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4xi64>, tensor<i64>) outs(%{{.*}} : tensor<4xui8>)
  // CHECK: hip.not({{.*}}) ins(%[[EQ]] : tensor<4xui8>) outs(%{{.*}} : tensor<4xui8>)
  // CHECK-NOT: hip.cast
  func.func @onnx_bool(%ctx: !hip.context, %x: tensor<4xi64>) -> tensor<4xui8> {
    %e = tensor.empty() : tensor<4xui8>
    %y = hip.cast(%ctx) ins(%x : tensor<4xi64>) outs(%e : tensor<4xui8>) {to = 9 : i64} : tensor<4xui8>
    return %y : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @onnx_ui8
  // CHECK: hip.equal({{.*}}) ins(%{{.*}}, %{{.*}} : tensor<4xui8>, tensor<ui8>)
  // CHECK: hip.not
  // CHECK-NOT: hip.cast
  func.func @onnx_ui8(%ctx: !hip.context, %x: tensor<4xui8>) -> tensor<4xui8> {
    %e = tensor.empty() : tensor<4xui8>
    %y = hip.cast(%ctx) ins(%x : tensor<4xui8>) outs(%e : tensor<4xui8>) {to = 9 : i64} : tensor<4xui8>
    return %y : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @onnx_uint8
  // CHECK: hip.cast({{.*}}) ins(%{{.*}} : tensor<4xi64>) outs(%{{.*}} : tensor<4xui8>) {to = 3 : i64}
  // CHECK-NOT: hip.equal
  func.func @onnx_uint8(%ctx: !hip.context, %x: tensor<4xi64>) -> tensor<4xui8> {
    %e = tensor.empty() : tensor<4xui8>
    %y = hip.cast(%ctx) ins(%x : tensor<4xi64>) outs(%e : tensor<4xui8>) {to = 3 : i64} : tensor<4xui8>
    return %y : tensor<4xui8>
  }

  // CHECK-LABEL: func.func @onnx_reshape
  // CHECK: %[[NOT:.*]] = hip.not({{.*}}) ins(%{{.*}} : tensor<1xui8>) outs(%{{.*}} : tensor<1xui8>)
  // CHECK: hip.cast({{.*}}) ins(%[[NOT]] : tensor<1xui8>) outs(%{{.*}} : tensor<ui8>) {to = 2 : i64}
  // CHECK-NOT: hip.equal
  func.func @onnx_reshape(%ctx: !hip.context, %x: tensor<1xi64>) -> tensor<ui8> {
    %e = tensor.empty() : tensor<ui8>
    %y = hip.cast(%ctx) ins(%x : tensor<1xi64>) outs(%e : tensor<ui8>) {to = 9 : i64} : tensor<ui8>
    return %y : tensor<ui8>
  }
}
