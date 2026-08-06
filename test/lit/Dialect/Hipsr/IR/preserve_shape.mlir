// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_form(
// CHECK-SAME: %[[D0:.+]]: index) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @tensor_form(%d0: index) {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf16>
  return
}

// -----

// The memref form: same shape operand, data operand now the kind of space-less
// memref one-shot-bufferize emits.
// CHECK-LABEL: func.func @memref_form(
// CHECK-SAME: %[[D0:.+]]: index) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) : memref<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @memref_form(%d0: index) {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %alloc = memref.alloc(%d0) : memref<?x2048xf16>
  hipsr.preserve_shape %shape, %alloc : memref<?x2048xf16>
  return
}

// -----

// A memref that does name its space is equally acceptable.
// CHECK-LABEL: func.func @device_memref_form(
// CHECK-SAME: %[[DATA:.+]]: memref<4x8xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C4:.+]] = arith.constant 4 : index
// CHECK-NEXT: %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[C4]], %[[C8]] : index, index
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : memref<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @device_memref_form(%data: memref<4x8xf16, #hipsr.mem<device>>) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %shape = shape.from_extents %c4, %c8 : index, index
  hipsr.preserve_shape %shape, %data : memref<4x8xf16, #hipsr.mem<device>>
  return
}

// -----

// A scalar has one shape with an empty extent list, matching hipsr.shape_yield.
// The from_extents line stops at the op name because an empty extent list prints
// an empty type list after the colon, same as in shape_yield.mlir.

// CHECK-LABEL: func.func @scalar() {
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty() : tensor<f16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<f16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @scalar() {
  %shape = "shape.from_extents"() : () -> !shape.shape
  %init = tensor.empty() : tensor<f16>
  hipsr.preserve_shape %shape, %init : tensor<f16>
  return
}

// -----

// !shape.shape is rank erased, so the rank agreement the op documents is only
// checkable when the extent list is in the IR. Here it is, and it disagrees.
func.func @extent_count_mismatch(%d0: index) {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048, %c2048 : index, index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  // expected-error @+1 {{shape has 3 extents but data has rank 2}}
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf16>
  return
}

// -----

// A shape that does not come from shape.from_extents carries no rank, so the
// verifier has nothing to compare and accepts it. This is the case the intended
// producer builds, where the shape comes out of an scf.execute_region.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE:.+]]: !shape.shape) {
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: !shape.shape) {
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf16>
  return
}

// -----

// The data operand must be ranked: an unranked tensor has no dimensions for a
// shape to describe.
func.func @unranked_data(%data: tensor<*xf16>) {
  %shape = "shape.from_extents"() : () -> !shape.shape
  // expected-error @+1 {{operand #1 must be ranked tensor or memref}}
  hipsr.preserve_shape %shape, %data : tensor<*xf16>
  return
}
