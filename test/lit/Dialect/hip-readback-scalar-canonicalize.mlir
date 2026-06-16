// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --canonicalize --split-input-file %s | FileCheck %s

// What this file tests:
// The readback_scalar look-through canonicalization in
// lib/Dialect/IR/HipDialect.cpp: a
// `hip.readback_scalar` whose source rank-0 scalar traces back through
// `collapse_shape` / `extract_slice` to a host-built `tensor.from_elements`
// is redundant (the value never lived on the device), so it is replaced by the
// corresponding `from_elements` operand and the D2H copy + stream sync is
// dropped. Guards: tensor form only (no-op on bufferized memref operands), and
// non-`from_elements` sources are left untouched.

// CHECK-LABEL: func.func @fold_extract_slice
// CHECK-NOT:   hip.readback_scalar
// CHECK:       return %arg2 : i64
func.func @fold_extract_slice(%ctx: !hip.context, %a: i64, %b: i64, %c: i64) -> i64 {
  %fe = tensor.from_elements %a, %b, %c : tensor<3xi64>
  %es = tensor.extract_slice %fe[1] [1] [1] : tensor<3xi64> to tensor<1xi64>
  %s = tensor.collapse_shape %es [] : tensor<1xi64> into tensor<i64>
  %v = hip.readback_scalar(%ctx, %s : tensor<i64>) -> i64
  return %v : i64
}

// -----

// Direct collapse of a single-element from_elements (no extract_slice).
// CHECK-LABEL: func.func @fold_direct
// CHECK-NOT:   hip.readback_scalar
// CHECK:       return %arg1 : i64
func.func @fold_direct(%ctx: !hip.context, %a: i64) -> i64 {
  %fe = tensor.from_elements %a : tensor<1xi64>
  %s = tensor.collapse_shape %fe [] : tensor<1xi64> into tensor<i64>
  %v = hip.readback_scalar(%ctx, %s : tensor<i64>) -> i64
  return %v : i64
}

// -----

// Negative: source is a function-arg vector, not a from_elements. The readback
// reads a genuine (possibly device-resident) buffer and must be preserved.
// CHECK-LABEL: func.func @no_fold_argsrc
// CHECK:       hip.readback_scalar
func.func @no_fold_argsrc(%ctx: !hip.context, %vec: tensor<1xi64>) -> i64 {
  %s = tensor.collapse_shape %vec [] : tensor<1xi64> into tensor<i64>
  %v = hip.readback_scalar(%ctx, %s : tensor<i64>) -> i64
  return %v : i64
}

// -----

// Negative: bufferized memref operand — the host origin is gone, so the fold
// must not fire.
// CHECK-LABEL: func.func @no_fold_memref
// CHECK:       hip.readback_scalar
func.func @no_fold_memref(%ctx: !hip.context, %m: memref<i64>) -> i64 {
  %v = hip.readback_scalar(%ctx, %m : memref<i64>) -> i64
  return %v : i64
}
