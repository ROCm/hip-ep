// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// A byte cast converts by value, so i64 5, f32 2.5 or ui8 5 would be stored in
// the mask unchanged instead of as 1. Only an i1 source may target i1; the
// pipeline rewrites the others first (hip-decompose-bool-cast).

// RUN: not hip-mlir-opt %s --split-input-file --convert-hip-to-llvm 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'hip.cast'
// CHECK-SAME: memref<8xi64, 1>, memref<8xi1, 1>
func.func @cast_i64_to_i1(
    %ctx: !hip.context,
    %input: memref<8xi64, 1>,
    %output: memref<8xi1, 1>) {
  hip.cast(%ctx) ins(%input : memref<8xi64, 1>)
                 outs(%output : memref<8xi1, 1>) {to = 9 : i64}
  return
}

// -----

// CHECK: failed to legalize operation 'hip.cast'
// CHECK-SAME: memref<8xui8, 1>, memref<8xi1, 1>
func.func @cast_ui8_to_i1(
    %ctx: !hip.context,
    %input: memref<8xui8, 1>,
    %output: memref<8xi1, 1>) {
  hip.cast(%ctx) ins(%input : memref<8xui8, 1>)
                 outs(%output : memref<8xi1, 1>) {to = 9 : i64}
  return
}

// -----

// CHECK: failed to legalize operation 'hip.cast'
// CHECK-SAME: memref<8xf32, 1>, memref<8xi1, 1>
func.func @cast_f32_to_i1(
    %ctx: !hip.context,
    %input: memref<8xf32, 1>,
    %output: memref<8xi1, 1>) {
  hip.cast(%ctx) ins(%input : memref<8xf32, 1>)
                 outs(%output : memref<8xi1, 1>) {to = 9 : i64}
  return
}

// -----

// CHECK: failed to legalize operation 'hip.cast'
// CHECK-SAME: memref<8xf64, 1>, memref<8xi1, 1>
func.func @cast_f64_to_i1(
    %ctx: !hip.context,
    %input: memref<8xf64, 1>,
    %output: memref<8xi1, 1>) {
  hip.cast(%ctx) ins(%input : memref<8xf64, 1>)
                 outs(%output : memref<8xi1, 1>) {to = 9 : i64}
  return
}

// -----

// ONNX BOOL is a ui8 result with `to = 9`; the pass leaves f64, and the
// runtime has no f64 -> ui8 kernel, so the cast must not reach it.
// CHECK: failed to legalize operation 'hip.cast'
// CHECK-SAME: memref<8xf64, 1>, memref<8xui8, 1>
func.func @cast_f64_to_ui8_bool(
    %ctx: !hip.context,
    %input: memref<8xf64, 1>,
    %output: memref<8xui8, 1>) {
  hip.cast(%ctx) ins(%input : memref<8xf64, 1>)
                 outs(%output : memref<8xui8, 1>) {to = 9 : i64}
  return
}
