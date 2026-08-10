// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy


// CHECK-LABEL: func.func @dps_init(
// CHECK-SAME: %[[CTX:.+]]: !hip.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
// CHECK-NEXT: hip.matmul(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32>)
func.func @dps_init(%ctx: !hip.context, %d0: index, %lhs: tensor<?x64xf32>,
                    %rhs: tensor<64x2048xf32>) -> tensor<?x2048xf32> {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf32>
  %out = hip.matmul(%ctx) ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x2048xf32>)
                          outs(%init : tensor<?x2048xf32>) : tensor<?x2048xf32>
  return %out : tensor<?x2048xf32>
}

// -----

// CHECK-LABEL: func.func @dps_result(
// CHECK-SAME: %[[CTX:.+]]: !hip.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hip.matmul(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32>)
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
func.func @dps_result(%ctx: !hip.context, %d0: index, %lhs: tensor<?x64xf32>,
                      %rhs: tensor<64x2048xf32>) -> tensor<?x2048xf32> {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  %out = hip.matmul(%ctx) ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x2048xf32>)
                          outs(%init : tensor<?x2048xf32>) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %out : tensor<?x2048xf32>
  return %out : tensor<?x2048xf32>
}

// -----

// A tensor block argument has no producer to bufferize
// CHECK-LABEL: func.func @func_arg(
// CHECK-SAME: %[[D0:.+]]: index,
// CHECK-SAME: %[[DATA:.+]]: memref<?x2048xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : memref<?x2048xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @func_arg(%d0: index, %data: tensor<?x2048xf32>) {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  hipsr.preserve_shape %shape, %data : tensor<?x2048xf32>
  return
}

// -----

// An opaque shape is not a tensor either, so bufferization carries it over
// untouched just like the from_extents form. This is the shape the intended
// producer builds, out of an scf.execute_region.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE:.+]]: !shape.shape) {
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: !shape.shape) {
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf32>
  return
}
