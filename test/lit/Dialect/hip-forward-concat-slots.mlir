// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-forward-concat-slots %s | FileCheck %s

// A rocMLIR kernel that is copied into a batch-1 channel slot writes the slot.
// Parent [1, 8, 2, 2] strides [32, 4, 2, 1]. The second slot starts at
// channel 4, offset 16, and is packed because the leading size is 1.

// CHECK-LABEL: func.func @rocmlir_channel_slot
// CHECK-NOT: memref.copy
// CHECK: %[[CAT:.*]] = memref.alloc() : memref<1x8x2x2xf32>
// CHECK: %[[SLOT:.*]] = memref.subview %[[CAT]][0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
// CHECK: hip.rocmlir
// CHECK-SAME: outs(%[[SLOT]] :
// CHECK-NOT: memref.copy
func.func @rocmlir_channel_slot(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>, %w: memref<4x4x1x1xf32>,
    %b: memref<4xf32>) -> memref<1x8x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in, %w, %b : memref<1x4x2x2xf32>,
      memref<4x4x1x1xf32>, memref<4xf32>) outs(%out : memref<1x4x2x2xf32>)
  %slot = memref.subview %cat[0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  return %cat : memref<1x8x2x2xf32>
}

// -----

// hip.resize uses the same slot.

// CHECK-LABEL: func.func @resize_channel_slot
// CHECK-NOT: memref.copy
// CHECK: hip.resize
// CHECK-SAME: outs(%{{.*}} : memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>)
// CHECK-NOT: memref.copy
func.func @resize_channel_slot(
    %ctx: !hip.context, %in: memref<1x4x1x1xf32>) -> memref<1x8x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.resize(%ctx) ins(%in : memref<1x4x1x1xf32>) outs(%out : memref<1x4x2x2xf32>)
      {mode = 0 : i64, coord_transform = 1 : i64, nearest_mode = 2 : i64}
  %slot = memref.subview %cat[0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  return %cat : memref<1x8x2x2xf32>
}

// -----

// The inner concat buffer is a prefix of the outer one, so the kernel writes
// the outer slot and neither copy survives.
// inner [1, 4, 2, 2] strides [16, 4, 2, 1]
// outer [1, 8, 2, 2] strides [32, 4, 2, 1], prefix offset 0

// CHECK-LABEL: func.func @chained_prefix
// CHECK-NOT: memref.copy
// CHECK: %[[OUTER:.*]] = memref.alloc() : memref<1x8x2x2xf32>
// CHECK: memref.subview %[[OUTER]][0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
// CHECK: hip.rocmlir
// CHECK-SAME: outs(%{{.*}} : memref<1x4x2x2xf32, strided<[32, 4, 2, 1]]>
// CHECK-NOT: memref.copy
func.func @chained_prefix(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>, %w: memref<4x4x1x1xf32>,
    %b: memref<4xf32>) -> memref<1x8x2x2xf32> {
  %outer = memref.alloc() : memref<1x8x2x2xf32>
  %inner = memref.alloc() : memref<1x4x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in, %w, %b : memref<1x4x2x2xf32>,
      memref<4x4x1x1xf32>, memref<4xf32>) outs(%out : memref<1x4x2x2xf32>)
  %full = memref.subview %inner[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[16, 4, 2, 1]>>
  memref.copy %out, %full : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[16, 4, 2, 1]>>
  %prefix = memref.subview %outer[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %inner, %prefix : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  return %outer : memref<1x8x2x2xf32>
}

// -----

// Batch 2 makes the channel slot strided. The copy stays.

// CHECK-LABEL: func.func @strided_batch_stays
// CHECK: memref.alloc() : memref<2x4x2x2xf32>
// CHECK: memref.copy
func.func @strided_batch_stays(
    %ctx: !hip.context, %in: memref<2x4x2x2xf32>, %w: memref<4x4x1x1xf32>,
    %b: memref<4xf32>) -> memref<2x8x2x2xf32> {
  %cat = memref.alloc() : memref<2x8x2x2xf32>
  %out = memref.alloc() : memref<2x4x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in, %w, %b : memref<2x4x2x2xf32>,
      memref<4x4x1x1xf32>, memref<4xf32>) outs(%out : memref<2x4x2x2xf32>)
  %slot = memref.subview %cat[0, 0, 0, 0] [2, 4, 2, 2] [1, 1, 1, 1]
      : memref<2x8x2x2xf32> to memref<2x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %out, %slot : memref<2x4x2x2xf32> to memref<2x4x2x2xf32, strided<[32, 4, 2, 1]>>
  return %cat : memref<2x8x2x2xf32>
}

// -----

// A block argument is not a rocMLIR or resize destination, so its copy stays.

// CHECK-LABEL: func.func @block_arg_stays
// CHECK: memref.copy
func.func @block_arg_stays(
    %in: memref<1x4x2x2xf32>) -> memref<1x8x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %slot = memref.subview %cat[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %in, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  return %cat : memref<1x8x2x2xf32>
}
