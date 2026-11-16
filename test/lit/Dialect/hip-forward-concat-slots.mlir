// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --hip-forward-concat-slots %s | FileCheck %s

// A rocMLIR kernel that is copied into a batch-1 channel slot writes the slot.
// Parent [1, 8, 2, 2]. The second slot starts at channel 4, element offset
// 16, and is packed because the leading size is 1. The identity view is
// shifted by that offset, so the lowering's bare pointer is the first
// element of the slot.

// CHECK-LABEL: func.func @rocmlir_channel_slot
// CHECK-NOT: memref.copy
// CHECK: %[[CAT:.*]] = memref.alloc() : memref<1x8x2x2xf32>
// CHECK: %[[SLOT:.*]] = hip.packed_view %[[CAT]][%{{.*}}] : memref<1x8x2x2xf32> to memref<1x4x2x2xf32>
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
// CHECK-SAME: outs(%{{.*}} : memref<1x4x2x2xf32>)
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
// CHECK: hip.packed_view %[[OUTER]][%{{.*}}] : memref<1x8x2x2xf32> to memref<1x4x2x2xf32>
// CHECK: hip.rocmlir
// CHECK-SAME: outs(%{{.*}} : memref<1x4x2x2xf32>)
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

// Nothing writes the block argument, so its copy stays.

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

// -----

// The same buffer feeds the concat and a later add. Both use the slot, so
// the copy is not replaced by a copy in front of the add.

// CHECK-LABEL: func.func @skip_reader
// CHECK-NOT: memref.copy
// CHECK: %[[SLOT:.*]] = hip.packed_view %{{.*}}[%{{.*}}] : memref<1x8x2x2xf32> to memref<1x4x2x2xf32>
// CHECK: hip.rocmlir
// CHECK-SAME: outs(%[[SLOT]] :
// CHECK: hip.add
// CHECK-SAME: ins(%[[SLOT]], %[[SLOT]]
// CHECK-NOT: memref.copy
func.func @skip_reader(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>, %w: memref<4x4x1x1xf32>,
    %b: memref<4xf32>) -> memref<1x8x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  %sum = memref.alloc() : memref<1x4x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in, %w, %b : memref<1x4x2x2xf32>,
      memref<4x4x1x1xf32>, memref<4xf32>) outs(%out : memref<1x4x2x2xf32>)
  hip.add(%ctx) ins(%out, %out : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%sum : memref<1x4x2x2xf32>)
  %slot = memref.subview %cat[0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  return %cat : memref<1x8x2x2xf32>
}

// -----

// An elementwise writer fills the slot directly. The pass is not limited to
// hip.rocmlir and hip.resize.

// CHECK-LABEL: func.func @elementwise_writer
// CHECK-NOT: memref.copy
// CHECK: %[[SLOT:.*]] = hip.packed_view %{{.*}}[%{{.*}}] : memref<1x8x2x2xf32> to memref<1x4x2x2xf32>
// CHECK: hip.add
// CHECK-SAME: outs(%[[SLOT]] :
// CHECK-NOT: memref.copy
func.func @elementwise_writer(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>) -> memref<1x8x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.add(%ctx) ins(%in, %in : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%out : memref<1x4x2x2xf32>)
  %slot = memref.subview %cat[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  return %cat : memref<1x8x2x2xf32>
}

// -----

// One buffer copied into two concats can live in only one of them. The other
// copy stays.

// CHECK-LABEL: func.func @two_concats_keep_one_copy
// CHECK: hip.packed_view
// CHECK: hip.rocmlir
// CHECK: memref.copy
// CHECK-NOT: memref.copy
func.func @two_concats_keep_one_copy(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>, %w: memref<4x4x1x1xf32>,
    %b: memref<4xf32>) -> (memref<1x8x2x2xf32>, memref<1x8x2x2xf32>) {
  %cat0 = memref.alloc() : memref<1x8x2x2xf32>
  %cat1 = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in, %w, %b : memref<1x4x2x2xf32>,
      memref<4x4x1x1xf32>, memref<4xf32>) outs(%out : memref<1x4x2x2xf32>)
  %slot0 = memref.subview %cat0[0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  memref.copy %out, %slot0 : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  %slot1 = memref.subview %cat1[0, 4, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  memref.copy %out, %slot1 : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1], offset: 16>>
  return %cat0, %cat1 : memref<1x8x2x2xf32>, memref<1x8x2x2xf32>
}

// -----

// The concat buffer is overwritten in place before the add reads the source.
// Aliasing would hand the add the overwritten values, so the copy stays.

// CHECK-LABEL: func.func @concat_overwritten_stays
// CHECK-NOT: hip.packed_view
// CHECK: memref.copy
func.func @concat_overwritten_stays(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>) -> memref<1x4x2x2xf32> {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  %sum = memref.alloc() : memref<1x4x2x2xf32>
  hip.add(%ctx) ins(%in, %in : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%out : memref<1x4x2x2xf32>)
  %slot = memref.subview %cat[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  hip.add(%ctx) ins(%cat, %cat : memref<1x8x2x2xf32>, memref<1x8x2x2xf32>)
      outs(%cat : memref<1x8x2x2xf32>)
  hip.add(%ctx) ins(%out, %out : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%sum : memref<1x4x2x2xf32>)
  return %sum : memref<1x4x2x2xf32>
}

// -----

// The source is written again after the copy. Aliasing would move that
// write into the concat result, so the copy stays.

// CHECK-LABEL: func.func @source_rewritten_stays
// CHECK-NOT: hip.packed_view
// CHECK: memref.copy
func.func @source_rewritten_stays(
    %ctx: !hip.context, %in: memref<1x4x2x2xf32>)
    -> (memref<1x8x2x2xf32>, memref<1x4x2x2xf32>) {
  %cat = memref.alloc() : memref<1x8x2x2xf32>
  %out = memref.alloc() : memref<1x4x2x2xf32>
  hip.add(%ctx) ins(%in, %in : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%out : memref<1x4x2x2xf32>)
  %slot = memref.subview %cat[0, 0, 0, 0] [1, 4, 2, 2] [1, 1, 1, 1]
      : memref<1x8x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  memref.copy %out, %slot : memref<1x4x2x2xf32> to memref<1x4x2x2xf32, strided<[32, 4, 2, 1]>>
  hip.add(%ctx) ins(%in, %in : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%out : memref<1x4x2x2xf32>)
  %sum = memref.alloc() : memref<1x4x2x2xf32>
  hip.add(%ctx) ins(%out, %out : memref<1x4x2x2xf32>, memref<1x4x2x2xf32>)
      outs(%sum : memref<1x4x2x2xf32>)
  return %cat, %sum : memref<1x8x2x2xf32>, memref<1x4x2x2xf32>
}

// -----

// A dense block: each concat repeats the previous one's inputs and appends
// one more. Every concat is a prefix of the last one, so the block shares
// one buffer and every input is written into it once.
// cat1 = [x, c1]         [1, 3, 2, 2]
// cat2 = [x, c1, c2]     [1, 4, 2, 2]
// cat3 = [x, c1, c2, c3] [1, 5, 2, 2]

// CHECK-LABEL: func.func @dense_block_one_buffer
// CHECK-NOT: memref.alloc() : memref<1x3x2x2xf32>
// CHECK-NOT: memref.alloc() : memref<1x4x2x2xf32>
// CHECK: %[[CAT:.*]] = memref.alloc() : memref<1x5x2x2xf32>
// CHECK-NOT: memref.alloc() : memref<1x{{[34]}}x2x2xf32>
// CHECK-NOT: memref.copy
// CHECK: return %[[CAT]]
func.func @dense_block_one_buffer(
    %ctx: !hip.context, %in: memref<1x2x2x2xf32>) -> memref<1x5x2x2xf32> {
  %x = memref.alloc() : memref<1x2x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in : memref<1x2x2x2xf32>)
      outs(%x : memref<1x2x2x2xf32>)
  %c1 = memref.alloc() : memref<1x1x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%x : memref<1x2x2x2xf32>)
      outs(%c1 : memref<1x1x2x2xf32>)
  %cat1 = memref.alloc() : memref<1x3x2x2xf32>
  %cat1_x = memref.subview %cat1[0, 0, 0, 0] [1, 2, 2, 2] [1, 1, 1, 1]
      : memref<1x3x2x2xf32> to memref<1x2x2x2xf32, strided<[12, 4, 2, 1]>>
  memref.copy %x, %cat1_x : memref<1x2x2x2xf32> to memref<1x2x2x2xf32, strided<[12, 4, 2, 1]>>
  %cat1_c1 = memref.subview %cat1[0, 2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x3x2x2xf32> to memref<1x1x2x2xf32, strided<[12, 4, 2, 1], offset: 8>>
  memref.copy %c1, %cat1_c1 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[12, 4, 2, 1], offset: 8>>
  %c2 = memref.alloc() : memref<1x1x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%cat1 : memref<1x3x2x2xf32>)
      outs(%c2 : memref<1x1x2x2xf32>)
  %cat2 = memref.alloc() : memref<1x4x2x2xf32>
  %cat2_x = memref.subview %cat2[0, 0, 0, 0] [1, 2, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x2x2x2xf32, strided<[16, 4, 2, 1]>>
  memref.copy %x, %cat2_x : memref<1x2x2x2xf32> to memref<1x2x2x2xf32, strided<[16, 4, 2, 1]>>
  %cat2_c1 = memref.subview %cat2[0, 2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 8>>
  memref.copy %c1, %cat2_c1 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 8>>
  %cat2_c2 = memref.subview %cat2[0, 3, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 12>>
  memref.copy %c2, %cat2_c2 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 12>>
  %c3 = memref.alloc() : memref<1x1x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%cat2 : memref<1x4x2x2xf32>)
      outs(%c3 : memref<1x1x2x2xf32>)
  %cat3 = memref.alloc() : memref<1x5x2x2xf32>
  %cat3_x = memref.subview %cat3[0, 0, 0, 0] [1, 2, 2, 2] [1, 1, 1, 1]
      : memref<1x5x2x2xf32> to memref<1x2x2x2xf32, strided<[20, 4, 2, 1]>>
  memref.copy %x, %cat3_x : memref<1x2x2x2xf32> to memref<1x2x2x2xf32, strided<[20, 4, 2, 1]>>
  %cat3_c1 = memref.subview %cat3[0, 2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x5x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 8>>
  memref.copy %c1, %cat3_c1 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 8>>
  %cat3_c2 = memref.subview %cat3[0, 3, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x5x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 12>>
  memref.copy %c2, %cat3_c2 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 12>>
  %cat3_c3 = memref.subview %cat3[0, 4, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x5x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 16>>
  memref.copy %c3, %cat3_c3 : memref<1x1x2x2xf32> to memref<1x1x2x2xf32, strided<[20, 4, 2, 1], offset: 16>>
  return %cat3 : memref<1x5x2x2xf32>
}

// -----

// The source changes between the two concats, so the prefix would hold the
// old value where the larger concat needs the new one. Both buffers stay.

// CHECK-LABEL: func.func @prefix_source_changes_stays
// CHECK-DAG: memref.alloc() : memref<1x4x2x2xf32>
// CHECK-DAG: %[[CAT1:.*]] = memref.alloc() : memref<1x3x2x2xf32>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<1x3x2x2xf32> to memref<1x3x2x2xf32, strided<[12, 4, 2, 1]>>
// CHECK: hip.add(%{{.*}}) ins(%[[CAT1]], %[[CAT1]]
func.func @prefix_source_changes_stays(
    %ctx: !hip.context, %in: memref<1x3x2x2xf32>)
    -> (memref<1x3x2x2xf32>, memref<1x4x2x2xf32>) {
  %x = memref.alloc() : memref<1x3x2x2xf32>
  hip.rocmlir(%ctx) @kern ins(%in : memref<1x3x2x2xf32>)
      outs(%x : memref<1x3x2x2xf32>)
  %cat1 = memref.alloc() : memref<1x3x2x2xf32>
  %cat1_x = memref.subview %cat1[0, 0, 0, 0] [1, 3, 2, 2] [1, 1, 1, 1]
      : memref<1x3x2x2xf32> to memref<1x3x2x2xf32, strided<[12, 4, 2, 1]>>
  memref.copy %x, %cat1_x : memref<1x3x2x2xf32> to memref<1x3x2x2xf32, strided<[12, 4, 2, 1]>>
  %out1 = memref.alloc() : memref<1x3x2x2xf32>
  hip.add(%ctx) ins(%cat1, %cat1 : memref<1x3x2x2xf32>, memref<1x3x2x2xf32>)
      outs(%out1 : memref<1x3x2x2xf32>)
  hip.rocmlir(%ctx) @kern ins(%in : memref<1x3x2x2xf32>)
      outs(%x : memref<1x3x2x2xf32>)
  %cat2 = memref.alloc() : memref<1x4x2x2xf32>
  %cat2_x = memref.subview %cat2[0, 0, 0, 0] [1, 3, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x3x2x2xf32, strided<[16, 4, 2, 1]>>
  memref.copy %x, %cat2_x : memref<1x3x2x2xf32> to memref<1x3x2x2xf32, strided<[16, 4, 2, 1]>>
  %cat2_y = memref.subview %cat2[0, 3, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x4x2x2xf32> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 12>>
  %y = memref.subview %in[0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
      : memref<1x3x2x2xf32> to memref<1x1x2x2xf32, strided<[12, 4, 2, 1]>>
  memref.copy %y, %cat2_y : memref<1x1x2x2xf32, strided<[12, 4, 2, 1]>> to memref<1x1x2x2xf32, strided<[16, 4, 2, 1], offset: 12>>
  return %out1, %cat2 : memref<1x3x2x2xf32>, memref<1x4x2x2xf32>
}
