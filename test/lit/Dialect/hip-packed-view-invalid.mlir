// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// No operand carries the result's extents, so a dynamic result has no size
// to lower.
func.func @dynamic_result(%src: memref<16xf32>) -> memref<?xf32> {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{requires a statically shaped result}}
  %0 = hip.packed_view %src[%c0] : memref<16xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func.func @span_out_of_bounds(%src: memref<16xf32>) -> memref<8xf32> {
  %c12 = arith.constant 12 : index
  // expected-error @+1 {{span [12, 20) is outside the source's 16 elements}}
  %0 = hip.packed_view %src[%c12] : memref<16xf32> to memref<8xf32>
  return %0 : memref<8xf32>
}
