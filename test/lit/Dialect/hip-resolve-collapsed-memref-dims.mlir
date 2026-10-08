// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-resolve-memref-dims --verify-each %s | FileCheck %s
// RUN: hip-mlir-opt --hip-resolve-memref-dims %s -o %t.once
// RUN: hip-mlir-opt --hip-resolve-memref-dims %t.once -o %t.twice
// RUN: diff %t.once %t.twice

// The leading axis is unchanged, even though the view has non-identity strides.
// CHECK-LABEL: func.func @singleton_group(
// CHECK-SAME: %[[SRC:.*]]: memref<?x1x4x8xf32, strided<[96, 32, 8, 1], offset: 32>>
// CHECK: %[[VIEW:.*]] = memref.collapse_shape %[[SRC]]
// CHECK: %[[N:.*]] = memref.dim %[[SRC]], %c0
// CHECK-NOT: memref.dim %[[VIEW]]
// CHECK: return %[[VIEW]], %[[N]]
func.func @singleton_group(
    %src: memref<?x1x4x8xf32, strided<[96, 32, 8, 1], offset: 32>>)
    -> (memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>, index) {
  %c0 = arith.constant 0 : index
  %view = memref.collapse_shape %src [[0], [1, 2], [3]]
      : memref<?x1x4x8xf32, strided<[96, 32, 8, 1], offset: 32>>
      into memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>
  %n = memref.dim %view, %c0
      : memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>
  return %view, %n : memref<?x4x8xf32, strided<[96, 8, 1], offset: 32>>, index
}

// An arbitrary result axis can combine several dynamic source dimensions.
// CHECK-LABEL: func.func @product(
// CHECK-SAME: %[[SRC:.*]]: memref<2x?x?xf32>
// CHECK: %[[M:.*]] = memref.dim %[[SRC]], %c1
// CHECK: %[[N:.*]] = memref.dim %[[SRC]], %c2
// CHECK: %[[SIZE:.*]] = arith.muli %[[M]], %[[N]] : index
// CHECK: return %[[SIZE]]
func.func @product(%src: memref<2x?x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %view = memref.collapse_shape %src [[0], [1, 2]]
      : memref<2x?x?xf32> into memref<2x?xf32>
  %n = memref.dim %view, %c1 : memref<2x?xf32>
  return %n : index
}

// Unit factors disappear without adding an allocation or payload read.
// CHECK-LABEL: func.func @unit_factors(
// CHECK-SAME: %[[SRC:.*]]: memref<1x?x1xf32>
// CHECK: %[[N:.*]] = memref.dim %[[SRC]], %c1
// CHECK-NOT: arith.muli
// CHECK: return %[[N]]
func.func @unit_factors(%src: memref<1x?x1xf32>) -> index {
  %c0 = arith.constant 0 : index
  %view = memref.collapse_shape %src [[0, 1, 2]]
      : memref<1x?x1xf32> into memref<?xf32>
  %n = memref.dim %view, %c0 : memref<?xf32>
  return %n : index
}

// CHECK-LABEL: func.func @zero_extent(
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NOT: memref.dim
// CHECK: return %[[ZERO]]
func.func @zero_extent(%src: memref<?x0xf32>) -> index {
  %c0 = arith.constant 0 : index
  %view = memref.collapse_shape %src [[0, 1]]
      : memref<?x0xf32> into memref<?xf32>
  %n = memref.dim %view, %c0 : memref<?xf32>
  return %n : index
}

// Reassociation composes through nested collapse operations.
// CHECK-LABEL: func.func @nested(
// CHECK-SAME: %[[SRC:.*]]: memref<?x?x?xf32>
// CHECK: %[[M:.*]] = memref.dim %[[SRC]], %c0
// CHECK: %[[N:.*]] = memref.dim %[[SRC]], %c1
// CHECK: %[[MN:.*]] = arith.muli %[[M]], %[[N]] : index
// CHECK: %[[K:.*]] = memref.dim %[[SRC]], %c2
// CHECK: %[[SIZE:.*]] = arith.muli %[[MN]], %[[K]] : index
// CHECK: return %[[SIZE]]
func.func @nested(%src: memref<?x?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %first = memref.collapse_shape %src [[0, 1], [2]]
      : memref<?x?x?xf32> into memref<?x?xf32>
  %second = memref.collapse_shape %first [[0, 1]]
      : memref<?x?xf32> into memref<?xf32>
  %n = memref.dim %second, %c0 : memref<?xf32>
  return %n : index
}

// Slice size, not the root buffer extent, determines this dimension. The
// subview drops one unit axis; its remaining strides and offset must survive.
// CHECK-LABEL: func.func @rank_reduced_slice(
// CHECK-SAME: %[[SRC:.*]]: memref<?x1x4x8xf32>, %[[N:.*]]: index
// CHECK: %[[SLICE:.*]] = memref.subview %[[SRC]][1, 0, 0, 0] [%[[N]], 1, 4, 8]
// CHECK: %[[VIEW:.*]] = memref.collapse_shape %[[SLICE]]
// CHECK-NOT: memref.dim
// CHECK: return %[[VIEW]], %[[N]]
func.func @rank_reduced_slice(%src: memref<?x1x4x8xf32>, %n: index)
    -> (memref<?x32xf32, strided<[32, 1], offset: 32>>, index) {
  %c0 = arith.constant 0 : index
  %slice = memref.subview %src[1, 0, 0, 0] [%n, 1, 4, 8] [1, 1, 1, 1]
      : memref<?x1x4x8xf32> to memref<?x4x8xf32, strided<[32, 8, 1], offset: 32>>
  %view = memref.collapse_shape %slice [[0], [1, 2]]
      : memref<?x4x8xf32, strided<[32, 8, 1], offset: 32>>
      into memref<?x32xf32, strided<[32, 1], offset: 32>>
  %dim = memref.dim %view, %c0 : memref<?x32xf32, strided<[32, 1], offset: 32>>
  return %view, %dim : memref<?x32xf32, strided<[32, 1], offset: 32>>, index
}

// A dynamic axis does not identify a reassociation group. Leave it unchanged.
// CHECK-LABEL: func.func @dynamic_axis(
// CHECK-SAME: %[[SRC:.*]]: memref<?x?x4xf32>, %[[AXIS:.*]]: index
// CHECK-NEXT: %[[VIEW:.*]] = memref.collapse_shape %[[SRC]]
// CHECK-NEXT: %[[N:.*]] = memref.dim %[[VIEW]], %[[AXIS]]
// CHECK-NEXT: return %[[N]]
func.func @dynamic_axis(%src: memref<?x?x4xf32>, %axis: index) -> index {
  %view = memref.collapse_shape %src [[0], [1, 2]]
      : memref<?x?x4xf32> into memref<?x?xf32>
  %n = memref.dim %view, %axis : memref<?x?xf32>
  return %n : index
}

// Out-of-bounds queries have undefined behavior. Do not index the
// reassociation groups or introduce operations for these queries.
// CHECK-LABEL: func.func @negative_axis(
// CHECK: %[[AXIS:.*]] = arith.constant -1 : index
// CHECK-NEXT: %[[VIEW:.*]] = memref.collapse_shape
// CHECK-NEXT: %[[N:.*]] = memref.dim %[[VIEW]], %[[AXIS]]
// CHECK-NEXT: return %[[N]]
func.func @negative_axis(%src: memref<?x4xf32>) -> index {
  %axis = arith.constant -1 : index
  %view = memref.collapse_shape %src [[0, 1]]
      : memref<?x4xf32> into memref<?xf32>
  %n = memref.dim %view, %axis : memref<?xf32>
  return %n : index
}

// CHECK-LABEL: func.func @axis_equal_to_rank(
// CHECK: %[[AXIS:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[VIEW:.*]] = memref.collapse_shape
// CHECK-NEXT: %[[N:.*]] = memref.dim %[[VIEW]], %[[AXIS]]
// CHECK-NEXT: return %[[N]]
func.func @axis_equal_to_rank(%src: memref<?x4xf32>) -> index {
  %axis = arith.constant 1 : index
  %view = memref.collapse_shape %src [[0, 1]]
      : memref<?x4xf32> into memref<?xf32>
  %n = memref.dim %view, %axis : memref<?xf32>
  return %n : index
}
