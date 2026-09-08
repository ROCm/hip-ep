// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy


// CHECK-LABEL: func.func @dps_init(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : !shape.shape, memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32, #hipsr.mem<device>>)
func.func @dps_init(%ctx: !hipsr.context, %d0: index,
                    %in: tensor<?x2048xf16, #hipsr.mem<device>>) -> tensor<?x2048xf32, #hipsr.mem<device>> {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %init : !shape.shape, tensor<?x2048xf32, #hipsr.mem<device>>
  %out = hipsr.cast(%ctx) ins(%in : tensor<?x2048xf16, #hipsr.mem<device>>)
                          outs(%init : tensor<?x2048xf32, #hipsr.mem<device>>) : tensor<?x2048xf32, #hipsr.mem<device>>
  return %out : tensor<?x2048xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @dps_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : !shape.shape, memref<?x2048xf32, #hipsr.mem<device>>
func.func @dps_result(%ctx: !hipsr.context, %d0: index,
                      %in: tensor<?x2048xf16, #hipsr.mem<device>>) -> tensor<?x2048xf32, #hipsr.mem<device>> {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  %out = hipsr.cast(%ctx) ins(%in : tensor<?x2048xf16, #hipsr.mem<device>>)
                          outs(%init : tensor<?x2048xf32, #hipsr.mem<device>>) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %out : !shape.shape, tensor<?x2048xf32, #hipsr.mem<device>>
  return %out : tensor<?x2048xf32, #hipsr.mem<device>>
}

// -----

// A tensor block argument has no producer to bufferize
// CHECK-LABEL: func.func @func_arg(
// CHECK-SAME: %[[D0:.+]]: index,
// CHECK-SAME: %[[DATA:.+]]: memref<?x2048xf32, strided<[?, ?], offset: ?>, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : !shape.shape, memref<?x2048xf32, strided<[?, ?], offset: ?>, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @func_arg(%d0: index, %data: tensor<?x2048xf32, #hipsr.mem<device>>) {
  %c2048 = arith.constant 2048 : index
  %shape = shape.from_extents %d0, %c2048 : index, index
  hipsr.preserve_shape %shape, %data : !shape.shape, tensor<?x2048xf32, #hipsr.mem<device>>
  return
}

// -----

// An opaque shape is not a tensor either, so bufferization carries it over
// untouched just like the from_extents form. This is the shape the intended
// producer builds, out of an scf.execute_region.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE:.+]]: !shape.shape) {
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : !shape.shape, memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: !shape.shape) {
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %init : !shape.shape, tensor<?x2048xf32, #hipsr.mem<device>>
  return
}
