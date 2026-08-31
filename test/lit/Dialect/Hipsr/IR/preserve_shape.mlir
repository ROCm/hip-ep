// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// The assembly format prints both types, so the shape operand carries its
// length and the verifier can compare it with the data's rank.
// CHECK-LABEL: func.func @tensor_form(
// CHECK-SAME: %[[D0:.+]]: index) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = tensor.from_elements %[[D0]], %[[C2048]] : tensor<2xindex>
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<2xindex>, tensor<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @tensor_form(%d0: index) {
  %c2048 = arith.constant 2048 : index
  %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  hipsr.preserve_shape %shape, %init : tensor<2xindex>, tensor<?x2048xf16>
  return
}

// -----

// The same tensor form, but device-spaced: preserve_shape's data operand is
// unconstrained on memory space, so a #hipsr.mem<device> tensor is equally
// accepted.
// CHECK-LABEL: func.func @tensor_form_device(
// CHECK-SAME: %[[D0:.+]]: index) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = tensor.from_elements %[[D0]], %[[C2048]] : tensor<2xindex>
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<2xindex>, tensor<?x2048xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @tensor_form_device(%d0: index) {
  %c2048 = arith.constant 2048 : index
  %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
  %init = tensor.empty(%d0) : tensor<?x2048xf16, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %init : tensor<2xindex>, tensor<?x2048xf16, #hipsr.mem<device>>
  return
}

// -----

// The memref form, as one-shot-bufferize leaves it: the data operand is a bare
// memref and the shape reaches the op through bufferization.to_tensor. An
// extent tensor stays a tensor, because preserve_shape names it without
// reading through it.
// CHECK-LABEL: func.func @memref_form(
// CHECK-SAME: %[[D0:.+]]: index, %[[EXTENTS:.+]]: memref<2xindex>) {
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[EXTENTS]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) : memref<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : tensor<2xindex>, memref<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @memref_form(%d0: index, %extents: memref<2xindex>) {
  %shape = bufferization.to_tensor %extents : memref<2xindex> to tensor<2xindex>
  %alloc = memref.alloc(%d0) : memref<?x2048xf16>
  hipsr.preserve_shape %shape, %alloc : tensor<2xindex>, memref<?x2048xf16>
  return
}

// -----

// A memref that does name its space is equally acceptable.
// CHECK-LABEL: func.func @device_memref_form(
// CHECK-SAME: %[[EXTENTS:.+]]: memref<2xindex>,
// CHECK-SAME: %[[DATA:.+]]: memref<4x8xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[EXTENTS]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : tensor<2xindex>, memref<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @device_memref_form(%extents: memref<2xindex>,
                              %data: memref<4x8xf16, #hipsr.mem<device>>) {
  %shape = bufferization.to_tensor %extents : memref<2xindex> to tensor<2xindex>
  hipsr.preserve_shape %shape, %data : tensor<2xindex>, memref<4x8xf16, #hipsr.mem<device>>
  return
}

// -----

// A tensor<?xindex> shape states no length, so the verifier has nothing to
// compare and accepts it. No pass builds one, but the type is still legal.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE:.+]]: tensor<?xindex>) {
// CHECK-NEXT: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x2048xf16>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[INIT]] : tensor<?xindex>, tensor<?x2048xf16>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: tensor<?xindex>) {
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  hipsr.preserve_shape %shape, %init : tensor<?xindex>, tensor<?x2048xf16>
  return
}

// -----

func.func @rank_mismatch(%d0: index, %shape: tensor<3xindex>) {
  %init = tensor.empty(%d0) : tensor<?x2048xf16>
  // expected-error @+1 {{shape holds 3 extents but data has rank 2}}
  hipsr.preserve_shape %shape, %init : tensor<3xindex>, tensor<?x2048xf16>
  return
}

// -----

// The data operand must be ranked: an unranked tensor has no dimensions for a
// shape to describe.
func.func @unranked_data(%shape: tensor<?xindex>, %data: tensor<*xf16>) {
  // expected-error @+1 {{operand #1 must be ranked tensor or memref}}
  hipsr.preserve_shape %shape, %data : tensor<?xindex>, tensor<*xf16>
  return
}
