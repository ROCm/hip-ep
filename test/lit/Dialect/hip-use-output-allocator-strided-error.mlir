// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Negative test for --hip-use-output-allocator.
//
// A returned view with static non-identity strides that must be copied (here a
// subview of a graph input) cannot be backed by an output-allocator buffer,
// which always has the identity layout. The pass reports an error before it
// rewrites anything instead of allocating a mis-laid-out output.
//===----------------------------------------------------------------------===//

// RUN: not hip-mlir-opt --hip-use-output-allocator %s 2>&1 | FileCheck %s

// CHECK: error: cannot return output 0 of type 'memref<4x4xf32, strided<[8, 1]>>' from an identity-layout output allocation
func.func @strided_passthrough(%ctx: !hip.context, %x: memref<8x8xf32>)
    -> memref<4x4xf32, strided<[8, 1]>> {
  %v = memref.subview %x[0, 0] [4, 4] [1, 1]
     : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1]>>
  return %v : memref<4x4xf32, strided<[8, 1]>>
}
