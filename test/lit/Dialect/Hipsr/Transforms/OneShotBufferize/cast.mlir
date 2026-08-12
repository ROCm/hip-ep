// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Positive coverage for one-shot-bufferize on hipsr.cast, the first hipsr DPS
// op with a bufferization model.
//
// Both cases hand the pass device buffers through bufferization.to_tensor, so
// the bufferized cast keeps the #hipsr.mem<device> space its operands require.
// A tensor.empty init bufferizes to a space-less memref instead, which is an
// error today and lives in invalid.mlir.
//
// Each case spells out its whole function, so the CHECK block reads as the
// expected output and the absences are load bearing without a CHECK-NOT: an
// inserted memref.copy, a surviving bufferization.to_tensor, or a result left
// on the cast all break the CHECK-NEXT chain.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize %s | FileCheck %s

// The init buffer is writable, so the cast is rewritten in place: it takes the
// buffers the to_tensor ops came from and keeps no result.
// CHECK-LABEL: func.func @cast_static(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[IN:.+]]: memref<4x8xf32, #hipsr.mem<device>>,
// CHECK-SAME: %[[OUT:.+]]: memref<4x8xf16, #hipsr.mem<device>>) {
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[IN]] : memref<4x8xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<4x8xf16, #hipsr.mem<device>>)
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @cast_static(%ctx: !hipsr.context,
                       %in: memref<4x8xf32, #hipsr.mem<device>>,
                       %out: memref<4x8xf16, #hipsr.mem<device>>) {
  %input = bufferization.to_tensor %in restrict
      : memref<4x8xf32, #hipsr.mem<device>> to tensor<4x8xf32>
  %init = bufferization.to_tensor %out restrict writable
      : memref<4x8xf16, #hipsr.mem<device>> to tensor<4x8xf16>
  %0 = hipsr.cast(%ctx) ins(%input : tensor<4x8xf32>)
      outs(%init : tensor<4x8xf16>) : tensor<4x8xf16>
  return
}

// -----

// The first cast's tensor result is replaced by the buffer of its init, so the
// second cast reads %[[MID]] and the chain runs on the three device buffers.
// The dynamic dimension needs nothing extra, since every buffer already exists.
// CHECK-LABEL: func.func @cast_chain_dynamic(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME: %[[IN:.+]]: memref<?x8xf16, #hipsr.mem<device>>,
// CHECK-SAME: %[[MID:.+]]: memref<?x8xf32, #hipsr.mem<device>>,
// CHECK-SAME: %[[OUT:.+]]: memref<?x8xi32, #hipsr.mem<device>>) {
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[IN]] : memref<?x8xf16, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[MID]] : memref<?x8xf32, #hipsr.mem<device>>)
// CHECK-NEXT: hipsr.cast(%[[CTX]]) ins(%[[MID]] : memref<?x8xf32, #hipsr.mem<device>>)
// CHECK-SAME: outs(%[[OUT]] : memref<?x8xi32, #hipsr.mem<device>>)
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @cast_chain_dynamic(%ctx: !hipsr.context,
                              %in: memref<?x8xf16, #hipsr.mem<device>>,
                              %mid: memref<?x8xf32, #hipsr.mem<device>>,
                              %out: memref<?x8xi32, #hipsr.mem<device>>) {
  %input = bufferization.to_tensor %in restrict
      : memref<?x8xf16, #hipsr.mem<device>> to tensor<?x8xf16>
  %mid_init = bufferization.to_tensor %mid restrict writable
      : memref<?x8xf32, #hipsr.mem<device>> to tensor<?x8xf32>
  %out_init = bufferization.to_tensor %out restrict writable
      : memref<?x8xi32, #hipsr.mem<device>> to tensor<?x8xi32>
  %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf16>)
      outs(%mid_init : tensor<?x8xf32>) : tensor<?x8xf32>
  %1 = hipsr.cast(%ctx) ins(%0 : tensor<?x8xf32>)
      outs(%out_init : tensor<?x8xi32>) : tensor<?x8xi32>
  return
}
