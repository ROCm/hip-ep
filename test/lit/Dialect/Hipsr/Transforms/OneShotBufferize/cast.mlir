// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The #hipsr.mem<device> tensor encoding is what supplies the memory space that
// hipsr operands require: use-encoding-for-memory-space carries the encoding of
// a ranked tensor onto the memref it becomes, so a tensor.empty init allocates
// in device space. Function boundaries use the identity layout map, as the
// production pipeline does, because HIP runtime calls consume contiguous
// buffers.

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// CHECK-LABEL: func.func @cast_static(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[IN:.+]]: memref<4x8xf32, #hipsr.mem<device>>) -> memref<4x8xf16, #hipsr.mem<device>> {
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc() {{.*}}: memref<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[IN]] : memref<4x8xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<4x8xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]] : memref<4x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @cast_static(%ctx: !hipsr.context,
                       %in: tensor<4x8xf32, #hipsr.mem<device>>)
    -> tensor<4x8xf16, #hipsr.mem<device>> {
  %init = tensor.empty() : tensor<4x8xf16, #hipsr.mem<device>>
  %0 = hipsr.cast(%ctx) ins(%in : tensor<4x8xf32, #hipsr.mem<device>>)
      outs(%init : tensor<4x8xf16, #hipsr.mem<device>>)
      : tensor<4x8xf16, #hipsr.mem<device>>
  return %0 : tensor<4x8xf16, #hipsr.mem<device>>
}

// -----

// The first cast's tensor result becomes the buffer of its init, so the second
// cast reads %[[MID]].
// CHECK-LABEL: func.func @cast_chain_dynamic(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[IN:.+]]: memref<?x8xf16, #hipsr.mem<device>>) -> memref<?x8xi32, #hipsr.mem<device>> {
// CHECK-NEXT: %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT: %[[D0:.+]] = memref.dim %[[IN]], %[[C0]] : memref<?x8xf16, #hipsr.mem<device>>
// CHECK-NEXT: %[[MID:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x8xf32, #hipsr.mem<device>>
// CHECK-NEXT: %[[OUT:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x8xi32, #hipsr.mem<device>>
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[IN]] : memref<?x8xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[MID]] : memref<?x8xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[MID]] : memref<?x8xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<?x8xi32, #hipsr.mem<device>>)
// CHECK-NEXT: return %[[OUT]] : memref<?x8xi32, #hipsr.mem<device>>
// CHECK-NEXT: }
func.func @cast_chain_dynamic(%ctx: !hipsr.context,
                              %in: tensor<?x8xf16, #hipsr.mem<device>>)
    -> tensor<?x8xi32, #hipsr.mem<device>> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %in, %c0 : tensor<?x8xf16, #hipsr.mem<device>>
  %mid_init = tensor.empty(%d0) : tensor<?x8xf32, #hipsr.mem<device>>
  %out_init = tensor.empty(%d0) : tensor<?x8xi32, #hipsr.mem<device>>
  %0 = hipsr.cast(%ctx) ins(%in : tensor<?x8xf16, #hipsr.mem<device>>)
      outs(%mid_init : tensor<?x8xf32, #hipsr.mem<device>>)
      : tensor<?x8xf32, #hipsr.mem<device>>
  %1 = hipsr.cast(%ctx) ins(%0 : tensor<?x8xf32, #hipsr.mem<device>>)
      outs(%out_init : tensor<?x8xi32, #hipsr.mem<device>>)
      : tensor<?x8xi32, #hipsr.mem<device>>
  return %1 : tensor<?x8xi32, #hipsr.mem<device>>
}
