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

// --- Float element type: NOT a candidate (likely a real GPU buffer). ---
// CHECK-LABEL: func.func @rank0_f32_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<f32>
func.func @rank0_f32_left_alone(%ctx: !hip.context, %x: f32) -> f32 {
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

// --- Host scalar reached through memref.reinterpret_cast: CSE fuses two
//     from_elements shape-arith buffers into one alloc and reinterpret_casts
//     it for the second (smaller) use, so the host store reaches its hip
//     consumer through a view. The pass MUST peek through the view and still
//     redirect the backing alloc to host scratch (otherwise the host store
//     lands in the GPU pool and SEGVs on real-device-memory targets). This is
//     the VLM-embedding regression pattern. ---
// CHECK-LABEL: func.func @host_scalar_via_reinterpret_cast
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK:         %[[SCRATCH:.*]] = hip.get_host_scratch(%[[CTX]],
// CHECK-NOT:     memref.alloc() : memref<3xi64>
// CHECK:         %[[V:.*]] = memref.view %[[SCRATCH]]{{.*}} : memref<?xi8> to memref<3xi64>
// CHECK:         memref.reinterpret_cast %[[V]]
func.func @host_scalar_via_reinterpret_cast(%ctx: !hip.context, %x: i64,
                                            %o0: memref<3xi32>,
                                            %o1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<3xi64>
  memref.store %x, %a[%c0] : memref<3xi64>
  hip.cast(%ctx) ins(%a : memref<3xi64>) outs(%o0 : memref<3xi32>) {to = 6 : i64}
  %rc = memref.reinterpret_cast %a to offset: [0], sizes: [1], strides: [1] : memref<3xi64> to memref<1xi64>
  memref.store %x, %rc[%c0] : memref<1xi64>
  hip.cast(%ctx) ins(%rc : memref<1xi64>) outs(%o1 : memref<1xi32>) {to = 6 : i64}
  memref.dealloc %a : memref<3xi64>
  return
}

// --- Host scalar whose view escapes to a consumer we can't vouch for (here:
//     returned from the function): the recursion must still REJECT it, since
//     we can't reason about the escaped view's memory expectations. ---
// CHECK-LABEL: func.func @host_scalar_view_escapes_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<3xi64>
func.func @host_scalar_view_escapes_left_alone(%ctx: !hip.context,
                                               %x: i64) -> memref<1xi64> {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<3xi64>
  memref.store %x, %a[%c0] : memref<3xi64>
  %rc = memref.reinterpret_cast %a to offset: [0], sizes: [1], strides: [1] : memref<3xi64> to memref<1xi64>
  return %rc : memref<1xi64>
}

// --- Host scalar consumed by memref.copy (the bufferized Concat /
//     insert_slice that stages grid-dim scalars into a shape vector, e.g.
//     vision rope `ReduceMax->Reshape->Gather->Range` arithmetic). memref.copy
//     is a terminal, host-mapping-safe use (the runtime memrefCopy issues
//     hipMemcpyAsync, which accepts host-mapped operands). The pass MUST accept
//     it and redirect the buffer to host scratch — otherwise the host store
//     into the buffer SEGVs on real-device-memory targets. ---
// CHECK-LABEL: func.func @host_scalar_via_memref_copy
// CHECK:         hip.get_host_scratch
// CHECK:         %[[V:.*]] = memref.view %{{.*}} : memref<?xi8> to memref<1xi64>
// CHECK:         memref.store {{.*}}, %[[V]]
// CHECK:         memref.copy %[[V]]
func.func @host_scalar_via_memref_copy(%ctx: !hip.context, %x: i64,
                                       %dst: memref<2xi64>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<1xi64>
  memref.store %x, %a[%c0] : memref<1xi64>
  %sv = memref.subview %dst[0] [1] [1] : memref<2xi64> to memref<1xi64, strided<[1]>>
  memref.copy %a, %sv : memref<1xi64> to memref<1xi64, strided<[1]>>
  memref.dealloc %a : memref<1xi64>
  return
}

// --- Host scalar used as the SHAPE operand of memref.reshape: the reshape
//     reads the (host-staged) shape vector; its result aliases the data
//     operand, not our buffer, so it is a terminal accept. The pass MUST
//     redirect the shape buffer to host scratch. (When the buffer is instead
//     the reshape's SOURCE, the pass recurses into the aliasing result.) ---
// CHECK-LABEL: func.func @host_scalar_as_reshape_shape
// CHECK:         hip.get_host_scratch
// CHECK:         memref.view %{{.*}} : memref<?xi8> to memref<2xi64>
func.func @host_scalar_as_reshape_shape(%ctx: !hip.context, %x: i64,
                                        %data: memref<?xf16>) -> memref<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %a = memref.alloc() : memref<2xi64>
  memref.store %x, %a[%c0] : memref<2xi64>
  memref.store %x, %a[%c1] : memref<2xi64>
  %r = memref.reshape %data(%a) : (memref<?xf16>, memref<2xi64>) -> memref<?x?xf16>
  return %r : memref<?x?xf16>
}

// --- Alloc carrying an explicit #hip.mem<host> space: a small integer alloc
//     with a memref.load user (e.g. a cross-space copy destination), otherwise
//     indistinguishable from a host-staged scalar — but MUST be left alone:
//     redirecting it here would build a memref.view whose #hip.mem<host> result
//     mismatches the space-less scratch base and fail the verifier. ---
// CHECK-LABEL: func.func @host_space_alloc_left_alone
// CHECK-NOT:   hip.get_host_scratch
// CHECK:       memref.alloc() : memref<i32, #hip.mem<host>>
func.func @host_space_alloc_left_alone(%ctx: !hip.context,
                                       %src: memref<i32, #hip.mem<device>>) -> i32 {
  %a = memref.alloc() : memref<i32, #hip.mem<host>>
  hip.memcpy_d2h_async(%ctx, %a, %src
      : memref<i32, #hip.mem<host>>, memref<i32, #hip.mem<device>>)
  hip.stream_sync(%ctx)
  %v = memref.load %a[] : memref<i32, #hip.mem<host>>
  memref.dealloc %a : memref<i32, #hip.mem<host>>
  return %v : i32
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
