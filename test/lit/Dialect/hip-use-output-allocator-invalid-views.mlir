// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Unsupported returned views must fail deterministically before the pass
// rewrites any output allocation. The first direct output is intentionally
// valid: the failure dump proves it remains a memref.alloc when later outputs
// fail validation.
//
// RUN: hip-mlir-opt --hip-use-output-allocator --verify-diagnostics %s
// RUN: not hip-mlir-opt --hip-use-output-allocator --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s --check-prefix=FAILURE-IR

// FAILURE-IR-LABEL: func.func @unsupported_output_views
// FAILURE-IR-COUNT-4: memref.alloc
// FAILURE-IR-NOT: hip.alloc_output
func.func @unsupported_output_views(%ctx: !hip.context)
    -> (memref<4xf32>,
        memref<2x4xf32, strided<[8, 1]>>,
        memref<2x8xf32, strided<[8, 1], offset: 8>>,
        memref<?x8xf32>) {
  %direct = memref.alloc() : memref<4xf32>

  %smaller_root = memref.alloc() : memref<4x8xf32>
  %smaller = memref.subview %smaller_root[0, 0] [2, 4] [1, 1]
      : memref<4x8xf32> to memref<2x4xf32, strided<[8, 1]>>

  %offset_root = memref.alloc() : memref<4x8xf32>
  %offset = memref.subview %offset_root[1, 0] [2, 8] [1, 1]
      : memref<4x8xf32> to memref<2x8xf32, strided<[8, 1], offset: 8>>

  %mixed_root = memref.alloc() : memref<2x4x8xf32>
  %collapsed = memref.collapse_shape %mixed_root [[0, 1, 2]]
      : memref<2x4x8xf32> into memref<64xf32>
  %expanded = memref.expand_shape %collapsed [[0, 1]] output_shape [8, 8]
      : memref<64xf32> into memref<8x8xf32>
  %mixed = memref.cast %expanded : memref<8x8xf32> to memref<?x8xf32>

  // expected-error @+2 {{output #1 does not have an identity-layout return type; exact-output copying cannot represent its offset or strides}}
  // expected-error @+1 {{output #2 does not have an identity-layout return type; exact-output copying cannot represent its offset or strides}}
  return %direct, %smaller, %offset, %mixed
      : memref<4xf32>,
        memref<2x4xf32, strided<[8, 1]>>,
        memref<2x8xf32, strided<[8, 1], offset: 8>>,
        memref<?x8xf32>
}
