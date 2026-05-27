// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-materialize-host-scalars.
//
// Verifies that the pass redirects tiny host-fed memref.alloc ops (the
// pattern emitted by bufferized tensor.from_elements for shape arithmetic)
// from the GPU pool to a runtime-owned host-mapped scratch buffer, while
// leaving GPU-consumed and large allocs alone.
//
// This is the Part-2 fix for the gfx1151 dynseqlen regression where rank-0
// i64 staging buffers landed in the GPU pool and were written from host code.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-materialize-host-scalars %s 2>&1 | FileCheck %s

// --- Rank-0 i64 alloc + host store + host load: classic Shape->Sub->Cast
//     pattern after bufferization. Should be redirected to host scratch. ---
// CHECK-LABEL: func.func @rank0_i64_host_scalar
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK-NOT:     memref.alloc()
// CHECK:         %[[SCRATCH:.*]] = hip.get_host_scratch(%[[CTX]],
// CHECK-SAME:                            : memref<?xi8>
// CHECK:         %[[V:.*]] = memref.view %[[SCRATCH]]{{.*}} : memref<?xi8> to memref<i64>
// CHECK:         memref.store {{.*}}, %[[V]][] : memref<i64>
// CHECK:         memref.load %[[V]][] : memref<i64>
func.func @rank0_i64_host_scalar(%ctx: !hip.context, %x: i64) -> i64 {
  %a = memref.alloc() : memref<i64>
  memref.store %x, %a[] : memref<i64>
  %v = memref.load %a[] : memref<i64>
  memref.dealloc %a : memref<i64>
  return %v : i64
}

// --- 1xi64 (e.g., gathered shape index) — also a candidate. ---
// CHECK-LABEL: func.func @rank1_1xi64_host_scalar
// CHECK-NOT:     memref.alloc()
// CHECK:         hip.get_host_scratch
// CHECK:         memref.view {{.*}} : memref<?xi8> to memref<1xi64>
func.func @rank1_1xi64_host_scalar(%ctx: !hip.context, %x: i64) -> i64 {
  %a = memref.alloc() : memref<1xi64>
  %c0 = arith.constant 0 : index
  memref.store %x, %a[%c0] : memref<1xi64>
  %v = memref.load %a[%c0] : memref<1xi64>
  memref.dealloc %a : memref<1xi64>
  return %v : i64
}

// --- Multiple candidates share a single get_host_scratch. ---
// CHECK-LABEL: func.func @two_scalars_one_scratch
// CHECK-COUNT-1: hip.get_host_scratch
// CHECK-COUNT-2: memref.view
func.func @two_scalars_one_scratch(%ctx: !hip.context, %x: i64, %y: i32) -> (i64, i32) {
  %a = memref.alloc() : memref<i64>
  %b = memref.alloc() : memref<i32>
  memref.store %x, %a[] : memref<i64>
  memref.store %y, %b[] : memref<i32>
  %va = memref.load %a[] : memref<i64>
  %vb = memref.load %b[] : memref<i32>
  memref.dealloc %a : memref<i64>
  memref.dealloc %b : memref<i32>
  return %va, %vb : i64, i32
}

// --- Float scalar with host I/O: candidate (canonical ONNX Range trip-count
//     pattern uses `arith.divf` on f32 scalars loaded from a memref<f32> that
//     was produced by a preceding hip.cast(i64) -> f32). Without host-mapped
//     backing the host arith dereferences a GPU pointer on some targets. ---
// CHECK-LABEL: func.func @rank0_f32_host_scalar
// CHECK-NOT:   memref.alloc()
// CHECK:       hip.get_host_scratch
// CHECK:       memref.view {{.*}} : memref<?xi8> to memref<f32>
func.func @rank0_f32_host_scalar(%ctx: !hip.context, %x: f32) -> f32 {
  %a = memref.alloc() : memref<f32>
  memref.store %x, %a[] : memref<f32>
  %v = memref.load %a[] : memref<f32>
  memref.dealloc %a : memref<f32>
  return %v : f32
}

// --- Larger int alloc (>16 elements): NOT a candidate. ---
// CHECK-LABEL: func.func @large_int_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<32xi32>
func.func @large_int_left_alone(%ctx: !hip.context, %x: i32) -> i32 {
  %a = memref.alloc() : memref<32xi32>
  %c0 = arith.constant 0 : index
  memref.store %x, %a[%c0] : memref<32xi32>
  %v = memref.load %a[%c0] : memref<32xi32>
  memref.dealloc %a : memref<32xi32>
  return %v : i32
}

// --- Alloc with NO host I/O (pure GPU producer + GPU consumer): NOT a
//     candidate, must stay in GPU pool. ---
// CHECK-LABEL: func.func @pure_gpu_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<i32>
func.func @pure_gpu_left_alone(%ctx: !hip.context,
                               %src: memref<i64>) -> memref<i32> {
  %a = memref.alloc() : memref<i32>
  // Both producer and consumer are hip ops — no host load/store touches %a,
  // so it doesn't need host-mapped backing.
  hip.cast(%ctx) ins(%src : memref<i64>) outs(%a : memref<i32>) {to = 6 : i64}
  return %a : memref<i32>
}

// --- Alloc with BOTH host store AND hip-op consumer: this is the regression
//     pattern (rank-0 i64 stored from host, then read by hip.cast for GQA
//     seqlens_k). MUST be redirected to host scratch — hipHostMallocMapped
//     is GPU-accessible on UMA, so the hip consumer keeps working. ---
// CHECK-LABEL: func.func @host_store_then_hip_consumer
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK:         %[[SCRATCH:.*]] = hip.get_host_scratch(%[[CTX]],
// CHECK-NOT:     memref.alloc() : memref<i64>
// CHECK:         %[[V:.*]] = memref.view %[[SCRATCH]]{{.*}} : memref<?xi8> to memref<i64>
// CHECK:         memref.store {{.*}}, %[[V]][] : memref<i64>
// CHECK:         hip.cast{{.*}} ins(%[[V]] : memref<i64>)
func.func @host_store_then_hip_consumer(%ctx: !hip.context,
                                        %x: i64,
                                        %out: memref<i32>) {
  %a = memref.alloc() : memref<i64>
  memref.store %x, %a[] : memref<i64>
  hip.cast(%ctx) ins(%a : memref<i64>) outs(%out : memref<i32>) {to = 6 : i64}
  memref.dealloc %a : memref<i64>
  return
}

// --- Function whose arg 0 is NOT a !hip.context: the pass silently leaves
//     it alone (best-effort mitigation; utility funcs without runtime
//     access don't have a context to call hip.get_host_scratch on). The
//     regression this guards against is a refactor that drops the arg-0
//     check and synthesizes a context, which would emit a
//     hip.get_host_scratch with a dangling operand. ---
// CHECK-LABEL: func.func @no_context_arg_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<i64>
func.func @no_context_arg_left_alone(%x: i64) -> i64 {
  %a = memref.alloc() : memref<i64>
  memref.store %x, %a[] : memref<i64>
  %v = memref.load %a[] : memref<i64>
  memref.dealloc %a : memref<i64>
  return %v : i64
}

// --- GPU producer -> host load: pass must insert hip.host_sync ahead of
//     the load so the host sees fresh GPU writes through the host-mapped
//     view. This is the canonical Range(start, limit, delta) trip-count
//     pattern from vision encoders: hip.cast writes start as f32 to a
//     scratch view, the host then reads it for arith.divf trip-count
//     computation. Without the sync the load returns stale bytes (0.0),
//     trip count comes out 0, downstream hip.alloc(0) returns NULL, SEGV.
// CHECK-LABEL: func.func @gpu_write_then_host_load_gets_sync
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK:         %[[SCRATCH:.*]] = hip.get_host_scratch(%[[CTX]],
// CHECK:         %[[V:.*]] = memref.view %[[SCRATCH]]{{.*}} : memref<?xi8> to memref<f32>
// CHECK:         hip.cast{{.*}} outs(%[[V]] : memref<f32>)
// CHECK:         hip.host_sync(%[[CTX]])
// CHECK:         memref.load %[[V]][] : memref<f32>
func.func @gpu_write_then_host_load_gets_sync(%ctx: !hip.context,
                                              %src: memref<i64>) -> f32 {
  %a = memref.alloc() : memref<f32>
  hip.cast(%ctx) ins(%src : memref<i64>) outs(%a : memref<f32>) {to = 1 : i64}
  %v = memref.load %a[] : memref<f32>
  memref.dealloc %a : memref<f32>
  return %v : f32
}

// --- Alloc reachable to host I/O via memref.reinterpret_cast: must be
//     redirected. Without this the alloc lands in the GPU pool while
//     downstream host code reads through the recast. Canonical vision
//     encoder shape-arithmetic pattern (sub-graph of a `tensor.gather`
//     index chain).
// CHECK-LABEL: func.func @recast_then_host_load
// CHECK-NOT:   memref.alloc()
// CHECK:       hip.get_host_scratch
func.func @recast_then_host_load(%ctx: !hip.context,
                                 %src: memref<i64>) -> i64 {
  %a = memref.alloc() : memref<i64>
  hip.cast(%ctx) ins(%src : memref<i64>) outs(%a : memref<i64>) {to = 7 : i64}
  %rc = memref.reinterpret_cast %a to offset: [0], sizes: [1], strides: [1]
      : memref<i64> to memref<1xi64>
  %c0 = arith.constant 0 : index
  %v = memref.load %rc[%c0] : memref<1xi64>
  memref.dealloc %a : memref<i64>
  return %v : i64
}

// --- memref.copy users count as host I/O (host-side memcpy on bufferized
//     shape vectors).
// CHECK-LABEL: func.func @copy_user_counts_as_host_io
// CHECK-NOT:   memref.alloc()
// CHECK:       hip.get_host_scratch
func.func @copy_user_counts_as_host_io(%ctx: !hip.context,
                                       %dst: memref<1xi64, strided<[1]>>) {
  %a = memref.alloc() : memref<i64>
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  memref.store %c0_i64, %a[] : memref<i64>
  %rc = memref.reinterpret_cast %a to offset: [0], sizes: [1], strides: [1]
      : memref<i64> to memref<1xi64>
  memref.copy %rc, %dst : memref<1xi64> to memref<1xi64, strided<[1]>>
  memref.dealloc %a : memref<i64>
  return
}

// --- Pure host store -> host load (no hip op in between): no sync needed.
//     The block is never "dirty" because no hip op produced the data. ---
// CHECK-LABEL: func.func @host_only_load_no_sync
// CHECK-NOT:   hip.host_sync
func.func @host_only_load_no_sync(%ctx: !hip.context, %x: i64) -> i64 {
  %a = memref.alloc() : memref<i64>
  memref.store %x, %a[] : memref<i64>
  %v = memref.load %a[] : memref<i64>
  memref.dealloc %a : memref<i64>
  return %v : i64
}

// --- Multiple host loads after a single GPU producer: only the FIRST load
//     needs the sync. After the sync, GPU writes are visible to all
//     subsequent host reads in the block until the next hip op runs. ---
// CHECK-LABEL: func.func @one_sync_per_dirty_block
// CHECK:         hip.cast{{.*}} outs(
// CHECK:         hip.host_sync
// CHECK:         memref.load
// CHECK-NOT:     hip.host_sync
// CHECK:         memref.load
func.func @one_sync_per_dirty_block(%ctx: !hip.context,
                                    %src: memref<i64>) -> (f32, f32) {
  %a = memref.alloc() : memref<f32>
  %b = memref.alloc() : memref<f32>
  hip.cast(%ctx) ins(%src : memref<i64>) outs(%a : memref<f32>) {to = 1 : i64}
  %va = memref.load %a[] : memref<f32>
  %vb = memref.load %b[] : memref<f32>
  memref.dealloc %a : memref<f32>
  memref.dealloc %b : memref<f32>
  return %va, %vb : f32, f32
}
