// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s | FileCheck %s

// A dynamic graph output round-trips: one Index per dynamic result dim plus
// the out_idx attribute.
// CHECK-LABEL: func.func @alloc_output_dynamic(
// CHECK-SAME: %[[CTX:.+]]: !hipsr.context, %[[M:.+]]: index, %[[N:.+]]: index
// CHECK-NEXT: %[[OUT:.+]] = hipsr.alloc_output(%[[CTX]], %[[M]], %[[N]]) {out_idx = 0 : i64} : memref<?x?xf16, #hipsr.mem<device>>
// CHECK-NEXT: return %[[OUT]] : memref<?x?xf16, #hipsr.mem<device>>
func.func @alloc_output_dynamic(
    %ctx: !hipsr.context, %m: index, %n: index)
    -> memref<?x?xf16, #hipsr.mem<device>> {
  %out = hipsr.alloc_output(%ctx, %m, %n) {out_idx = 0 : i64}
      : memref<?x?xf16, #hipsr.mem<device>>
  return %out : memref<?x?xf16, #hipsr.mem<device>>
}
