// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.


// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries use-encoding-for-memory-space" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy

// The shape operand is an extent tensor, so bufferization descends into the
// shape computation too:
//
//   - tensor.from_elements becomes a host memref.alloc, one store per extent;
//   - preserve_shape reads that buffer back with bufferization.to_tensor;
//   - only the data operand becomes a bare memref, because preserve_shape
//     names its shape instead of reading through it.

// CHECK-LABEL: func.func @dps_init(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[EXTENTS:.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: memref.store %[[D0]], %[[EXTENTS]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[C2048]], %[[EXTENTS]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[EXTENTS]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : tensor<2xindex>, memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32, #hipsr.mem<device>>)
func.func @dps_init(%ctx: !hipsr.context, %d0: index,
                    %in: tensor<?x2048xf16, #hipsr.mem<device>>) -> tensor<?x2048xf32, #hipsr.mem<device>> {
  %c2048 = arith.constant 2048 : index
  %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %init : tensor<2xindex>, tensor<?x2048xf32, #hipsr.mem<device>>
  %out = hipsr.cast(%ctx) ins(%in : tensor<?x2048xf16, #hipsr.mem<device>>)
                          outs(%init : tensor<?x2048xf32, #hipsr.mem<device>>) : tensor<?x2048xf32, #hipsr.mem<device>>
  return %out : tensor<?x2048xf32, #hipsr.mem<device>>
}

// -----

// CHECK-LABEL: func.func @dps_result(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[EXTENTS:.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: memref.store %[[D0]], %[[EXTENTS]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[C2048]], %[[EXTENTS]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[EXTENTS]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : tensor<2xindex>, memref<?x2048xf32, #hipsr.mem<device>>
func.func @dps_result(%ctx: !hipsr.context, %d0: index,
                      %in: tensor<?x2048xf16, #hipsr.mem<device>>) -> tensor<?x2048xf32, #hipsr.mem<device>> {
  %c2048 = arith.constant 2048 : index
  %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  %out = hipsr.cast(%ctx) ins(%in : tensor<?x2048xf16, #hipsr.mem<device>>)
                          outs(%init : tensor<?x2048xf32, #hipsr.mem<device>>) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %out : tensor<2xindex>, tensor<?x2048xf32, #hipsr.mem<device>>
  return %out : tensor<?x2048xf32, #hipsr.mem<device>>
}

// -----

// A tensor block argument has no producer to bufferize
// CHECK-LABEL: func.func @func_arg(
// CHECK-SAME: %[[D0:.+]]: index,
// CHECK-SAME: %[[DATA:.+]]: memref<?x2048xf32, strided<[?, ?], offset: ?>, #hipsr.mem<device>>) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[EXTENTS:.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT: memref.store %[[D0]], %[[EXTENTS]]{{\[}}%[[C0]]] : memref<2xindex>
// CHECK-NEXT: memref.store %[[C2048]], %[[EXTENTS]]{{\[}}%[[C1]]] : memref<2xindex>
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[EXTENTS]] : memref<2xindex> to tensor<2xindex>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : tensor<2xindex>, memref<?x2048xf32, strided<[?, ?], offset: ?>, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @func_arg(%d0: index, %data: tensor<?x2048xf32, #hipsr.mem<device>>) {
  %c2048 = arith.constant 2048 : index
  %shape = tensor.from_elements %d0, %c2048 : tensor<2xindex>
  hipsr.preserve_shape %shape, %data : tensor<2xindex>, tensor<?x2048xf32, #hipsr.mem<device>>
  return
}

// -----

// This function builds no extent list, so there is nothing to materialize.
// Bufferization only turns the argument into a memref and reads it back, giving
// preserve_shape the same to_tensor shape operand as above.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE_BUF:.+]]: memref<?xindex, strided<[?], offset: ?>>) {
// CHECK-NEXT: %[[SHAPE:.+]] = bufferization.to_tensor %[[SHAPE_BUF]] : memref<?xindex, strided<[?], offset: ?>> to tensor<?xindex>
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : tensor<?xindex>, memref<?x2048xf32, #hipsr.mem<device>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: tensor<?xindex>) {
  %init = tensor.empty(%d0) : tensor<?x2048xf32, #hipsr.mem<device>>
  hipsr.preserve_shape %shape, %init : tensor<?xindex>, tensor<?x2048xf32, #hipsr.mem<device>>
  return
}
