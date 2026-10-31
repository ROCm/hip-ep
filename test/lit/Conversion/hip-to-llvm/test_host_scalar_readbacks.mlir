// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// Preserve bits and the unsigned i32-to-index conversion. Keep the stores.
// CHECK-LABEL: llvm.func @host_values(
// CHECK: llvm.call @hipdnn_ep_get_host_scratch_base
// CHECK: llvm.store %arg1,
// CHECK: llvm.store %arg2,
// CHECK: llvm.store %arg3,
// CHECK: llvm.store %arg4,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.zext %arg1 : i32 to i64
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return
func.func @host_values(%ctx: !hip.context, %n: i32, %wide: i64,
                      %f: f32, %index: index) -> (i32, index, i64, f32, index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c96 = arith.constant 96 : index
  %c112 = arith.constant 112 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %a = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %b = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi64>
  %c = memref.view %scratch[%c96][] : memref<?xi8> to memref<1xf32>
  %e = memref.view %scratch[%c112][] : memref<?xi8> to memref<1xindex>
  memref.store %n, %a[%c0] : memref<1xi32>
  memref.store %wide, %b[%c0] : memref<1xi64>
  memref.store %f, %c[%c0] : memref<1xf32>
  memref.store %index, %e[%c0] : memref<1xindex>
  %r = hip.readback_scalar(%ctx, %a : memref<1xi32>) -> i32
  %d = hip.readback_dim(%ctx, %a : memref<1xi32>) -> index
  %w = hip.readback_scalar(%ctx, %b : memref<1xi64>) -> i64
  %v = hip.readback_scalar(%ctx, %c : memref<1xf32>) -> f32
  %i = hip.readback_scalar(%ctx, %e : memref<1xindex>) -> index
  return %r, %d, %w, %v, %i : i32, index, i64, f32, index
}

// -----

// A write to a different region does not kill the known length.
// CHECK-LABEL: llvm.func @mixed_regions(
// CHECK: llvm.call @wrap_cast
// CHECK-NOT: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.call @hipdnn_ep_readback_i32
// CHECK-NOT: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @mixed_regions(%ctx: !hip.context, %n: i32,
                        %source: memref<1xi64>) -> (i32, index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %length = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %count = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %length[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%count : memref<1xi32>) {to = 6 : i64}
  %r = hip.readback_scalar(%ctx, %length : memref<1xi32>) -> i32
  %d = hip.readback_dim(%ctx, %count : memref<1xi32>) -> index
  return %r, %d : i32, index
}

// -----

// Keep the first GPU result read. Reuse it for both kinds of later readback.
// CHECK-LABEL: llvm.func @first_device_read(
// CHECK: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: %[[FIRST:.*]] = llvm.load {{.*}} : !llvm.ptr -> i32
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.zext %[[FIRST]] : i32 to i64
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return
func.func @first_device_read(%ctx: !hip.context,
                            %source: memref<1xi64>) -> (i32, i32, index) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %count = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%count : memref<1xi32>) {to = 6 : i64}
  %a = hip.readback_scalar(%ctx, %count : memref<1xi32>) -> i32
  %b = hip.readback_scalar(%ctx, %count : memref<1xi32>) -> i32
  %d = hip.readback_dim(%ctx, %count : memref<1xi32>) -> index
  return %a, %b, %d : i32, i32, index
}

// -----

// Removing both waits would expose the host load to a pending device write.
// CHECK-LABEL: llvm.func @joint_removal(
// CHECK: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.load
// CHECK: llvm.return
func.func @joint_removal(%ctx: !hip.context, %n: i32,
                        %source: memref<1xi64>) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %length = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %other = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %length[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%other : memref<1xi32>) {to = 6 : i64}
  %a = hip.readback_scalar(%ctx, %length : memref<1xi32>) -> i32
  %b = hip.readback_scalar(%ctx, %length : memref<1xi32>) -> i32
  %x = memref.load %other[%c0] : memref<1xi32>
  return %a, %b, %x : i32, i32, i32
}

// -----

// A retained readback completes earlier work before the host store.
// CHECK-LABEL: llvm.func @retained_barrier(
// CHECK: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_readback_i32
// CHECK: llvm.store %arg1,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return
func.func @retained_barrier(%ctx: !hip.context, %n: i32,
                           %source: memref<1xi64>) -> (index, i32) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %count = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%count : memref<1xi32>) {to = 6 : i64}
  %first = hip.readback_dim(%ctx, %count : memref<1xi32>) -> index
  memref.store %n, %count[%c0] : memref<1xi32>
  %again = hip.readback_scalar(%ctx, %count : memref<1xi32>) -> i32
  return %first, %again : index, i32
}

// -----

// Partial writes kill an exact value. Do not combine smaller stores.
// CHECK-LABEL: llvm.func @partial_write(
// CHECK: llvm.store
// CHECK: llvm.store
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @partial_write(%ctx: !hip.context, %n: i32, %half: i16) -> i32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %whole = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %upper = memref.view %scratch[%c2][] : memref<?xi8> to memref<1xi16>
  memref.store %n, %whole[%c0] : memref<1xi32>
  memref.store %half, %upper[%c0] : memref<1xi16>
  %r = hip.readback_scalar(%ctx, %whole : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// Equal widths do not make integer and float storage interchangeable.
// CHECK-LABEL: llvm.func @different_type(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @different_type(%ctx: !hip.context, %n: i32) -> f32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %integer = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %float = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xf32>
  memref.store %n, %integer[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %float : memref<1xf32>) -> f32
  return %r : f32
}

// -----

// Reinterpretation replaces the offset. Metadata preserves the aligned base.
// CHECK-LABEL: llvm.func @offset_reset(
// CHECK: llvm.store %arg1,
// CHECK: llvm.store %arg2,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i64
func.func @offset_reset(%ctx: !hip.context, %a: i64, %b: i64) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<2xi64>
  memref.store %a, %shape[%c0] : memref<2xi64>
  memref.store %b, %shape[%c1] : memref<2xi64>
  %part = memref.reinterpret_cast %shape to offset: [1], sizes: [1], strides: [1] :
    memref<2xi64> to memref<1xi64, strided<[1], offset: 1>>
  %base, %off, %n, %stride = memref.extract_strided_metadata %part :
    memref<1xi64, strided<[1], offset: 1>> -> memref<i64>, index, index, index
  %reset = memref.reinterpret_cast %base to offset: [0], sizes: [1], strides: [1] :
    memref<i64> to memref<1xi64>
  %r = hip.readback_scalar(%ctx, %reset : memref<1xi64>) -> i64
  return %r : i64
}

// -----

// An unknown offset cannot establish byte identity.
// CHECK-LABEL: llvm.func @unknown_offset(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @unknown_offset(%ctx: !hip.context, %n: i32, %offset: index) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %slot = memref.view %scratch[%offset][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %slot[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %slot : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// The witness can itself use an earlier forwarded result.
// CHECK-LABEL: llvm.func @dependent_replacements(
// CHECK: llvm.store %arg1,
// CHECK: llvm.store %arg1,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i32
func.func @dependent_replacements(%ctx: !hip.context, %n: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %a = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %b = memref.view %scratch[%c32][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %a[%c0] : memref<1xi32>
  %first = hip.readback_scalar(%ctx, %a : memref<1xi32>) -> i32
  memref.store %first, %b[%c0] : memref<1xi32>
  %second = hip.readback_scalar(%ctx, %b : memref<1xi32>) -> i32
  return %second : i32
}

// -----

// The known scalar's wait protects a later overwrite from a pending reader.
// CHECK-LABEL: llvm.func @pending_reader(
// CHECK: llvm.call @wrap_hipMemcpyAsync
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.store %arg2,
// CHECK: llvm.return
func.func @pending_reader(%ctx: !hip.context, %a: i32, %b: i32,
                         %target: memref<1xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %slot = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %slot[%c0] : memref<1xi32>
  memref.copy %slot, %target : memref<1xi32> to memref<1xi32>
  %r = hip.readback_scalar(%ctx, %slot : memref<1xi32>) -> i32
  memref.store %b, %slot[%c0] : memref<1xi32>
  return %r : i32
}

// -----

// An alias can reach another slot. The last store supplies its value.
// CHECK-LABEL: llvm.func @cross_region_store(
// CHECK: llvm.store %arg1,
// CHECK: llvm.store %arg2,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg2 : i32
func.func @cross_region_store(%ctx: !hip.context, %a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %first = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi32>
  %other = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %first[%c0] : memref<1xi32>
  %back = memref.reinterpret_cast %other to offset: [16], sizes: [1], strides: [1] :
    memref<1xi32> to memref<1xi32, strided<[1], offset: 16>>
  memref.store %b, %back[%c0] : memref<1xi32, strided<[1], offset: 16>>
  %r = hip.readback_scalar(%ctx, %first : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// Checked range arithmetic rejects an offset that overflows in bytes.
// CHECK-LABEL: llvm.func @overflowing_offset(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @overflowing_offset(%ctx: !hip.context, %n: i64) -> i64 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %slot = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi64>
  memref.store %n, %slot[%c0] : memref<1xi64>
  %far = memref.reinterpret_cast %slot to offset: [2305843009213693952], sizes: [1], strides: [1] :
    memref<1xi64> to memref<1xi64, strided<[1], offset: 2305843009213693952>>
  %r = hip.readback_scalar(%ctx, %far : memref<1xi64, strided<[1], offset: 2305843009213693952>>) -> i64
  return %r : i64
}

// -----

// A GPU overwrite invalidates the stored value immediately at enqueue.
// CHECK-LABEL: llvm.func @device_overwrite(
// CHECK: llvm.store %arg1,
// CHECK: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @device_overwrite(%ctx: !hip.context, %n: i32,
                           %source: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %slot = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %slot[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%slot : memref<1xi32>) {to = 6 : i64}
  %r = hip.readback_scalar(%ctx, %slot : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// TopK reads K before its context wait. Keep the earlier explicit wait.
// CHECK-LABEL: llvm.func @topk_pending_k(
// CHECK: llvm.call @wrap_cast
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.call @wrap_top_k
// CHECK: llvm.return
func.func @topk_pending_k(%ctx: !hip.context, %n: i32, %source: memref<1xi32>,
                        %data: memref<4xf32>, %values: memref<1xf32>,
                        %indices: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %length = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %k = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi64>
  memref.store %n, %length[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi32>)
    outs(%k : memref<1xi64>) {to = 7 : i64}
  %r = hip.readback_scalar(%ctx, %length : memref<1xi32>) -> i32
  hip.top_k(%ctx) ins(%data, %k : memref<4xf32>, memref<1xi64>)
    outs(%values, %indices : memref<1xf32>, memref<1xi64>) {axis = 0 : i64}
  return %r : i32
}

// -----

// A host-written K has no pending producer. Do not add a readback for TopK.
// CHECK-LABEL: llvm.func @topk_host_k(
// CHECK: llvm.store %arg1,
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.call @wrap_top_k
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i64
func.func @topk_host_k(%ctx: !hip.context, %n: i64, %data: memref<4xf32>,
                     %values: memref<1xf32>, %indices: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %k = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi64>
  memref.store %n, %k[%c0] : memref<1xi64>
  %r = hip.readback_scalar(%ctx, %k : memref<1xi64>) -> i64
  hip.top_k(%ctx) ins(%data, %k : memref<4xf32>, memref<1xi64>)
    outs(%values, %indices : memref<1xf32>, memref<1xi64>) {axis = 0 : i64}
  return %r : i64
}
