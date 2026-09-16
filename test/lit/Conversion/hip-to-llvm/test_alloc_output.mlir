// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.alloc_output is lowered to a runtime call to
// @hipdnn_ep_alloc_output plus a memref descriptor over the returned pointer.
//
// This test validates:
// - !hip.context -> !llvm.ptr
// - A stack-allocated i64[rank] shape array (llvm.alloca) is populated with the
//   result dims, in type order: static dims become llvm.mlir.constant, dynamic
//   dims come from the op's operands.
// - The call signature is (state, out_idx, shape_ptr, rank, elem_size) -> ptr,
//   with out_idx / rank / elem_size emitted as i64 constants.
// - The returned generic (AS 0) pointer is addrspacecast to the memref's
//   address space (AS 1) only when they differ; for a default-AS (AS 0) result
//   no cast is emitted.
// - A standard memref descriptor (row-major) is built and is usable downstream
//   (consumed here by hip.sigmoid).
// - Works uniformly for static, fully dynamic, and mixed shapes, and for
//   dynamic sizes sourced from memref.dim, integer arithmetic, and data loads.
//
// Runtime API:
//   void* hipdnn_ep_alloc_output(void* state, int64_t out_idx,
//                                const int64_t* shape, int64_t rank,
//                                int64_t elem_size)
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// The runtime allocator is declared once (deduplicated across all call sites).
// CHECK: llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr

module {
  // Test 1: Fully static result (AS 1), no dynamic operands. Both dims are
  // compile-time constants stored into the shape array; f16 -> elem_size 2.
  func.func @alloc_static(%ctx: !hip.context) -> memref<3x5xf16, 1> {
    // CHECK-LABEL: llvm.func @alloc_static
    // CHECK-SAME:  (%[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr)
    // CHECK-SAME:  -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:       %[[D0:.*]] = llvm.mlir.constant(3 : index) : i64
    // CHECK:       %[[D1:.*]] = llvm.mlir.constant(5 : index) : i64
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64> {alignment = 8 : i64}
    // CHECK:       %[[G0:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[D0]], %[[G0]] : i64, !llvm.ptr
    // CHECK:       %[[G1:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[D1]], %[[G1]] : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(7 : i64) : i64
    // CHECK:       %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[CTX]], %[[OUTIDX]], %[[SHAPE]], %[[RANK]], %[[ELEM]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       %[[CAST:.*]] = llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    // CHECK:       llvm.insertvalue %[[CAST]], %{{.*}}[0]
    // CHECK:       llvm.insertvalue %[[CAST]], %{{.*}}[1]
    // CHECK:       llvm.insertvalue %[[D0]], %{{.*}}[3, 0]
    // CHECK:       llvm.insertvalue %[[D1]], %{{.*}}[3, 1]
    %0 = hip.alloc_output(%ctx) {out_idx = 7 : i64} : memref<3x5xf16, 1>
    return %0 : memref<3x5xf16, 1>
  }

  // Test 2: Fully dynamic result (AS 1). Both dims come from operands and are
  // stored verbatim into the shape array (in operand order).
  func.func @alloc_dynamic(%ctx: !hip.context, %m: index, %n: index) -> memref<?x?xf16, 1> {
    // CHECK-LABEL: llvm.func @alloc_dynamic
    // CHECK-SAME:  (%[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[M:[a-zA-Z0-9_]+]]: i64, %[[N:[a-zA-Z0-9_]+]]: i64)
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64> {alignment = 8 : i64}
    // CHECK:       %[[G0:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[M]], %[[G0]] : i64, !llvm.ptr
    // CHECK:       %[[G1:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[N]], %[[G1]] : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(11 : i64) : i64
    // CHECK:       %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       %[[ELEM:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[CTX]], %[[OUTIDX]], %[[SHAPE]], %[[RANK]], %[[ELEM]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %0 = hip.alloc_output(%ctx, %m, %n) {out_idx = 11 : i64} : memref<?x?xf16, 1>
    return %0 : memref<?x?xf16, 1>
  }

  // Test 3: Mixed shape (AS 1): dim 0 dynamic (operand), dim 1 static (128).
  // f32 -> elem_size 4.
  func.func @alloc_mixed(%ctx: !hip.context, %m: index) -> memref<?x128xf32, 1> {
    // CHECK-LABEL: llvm.func @alloc_mixed
    // CHECK-SAME:  (%[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[M:[a-zA-Z0-9_]+]]: i64)
    // CHECK:       %[[D1:.*]] = llvm.mlir.constant(128 : index) : i64
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64> {alignment = 8 : i64}
    // CHECK:       %[[G0:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[M]], %[[G0]] : i64, !llvm.ptr
    // CHECK:       %[[G1:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[D1]], %[[G1]] : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(13 : i64) : i64
    // CHECK:       %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK:       %[[ELEM:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[CTX]], %[[OUTIDX]], %[[SHAPE]], %[[RANK]], %[[ELEM]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %0 = hip.alloc_output(%ctx, %m) {out_idx = 13 : i64} : memref<?x128xf32, 1>
    return %0 : memref<?x128xf32, 1>
  }

  // Test 4: Rank-1 dynamic result (AS 1). Single-element shape array.
  func.func @alloc_rank1(%ctx: !hip.context, %n: index) -> memref<?xf32, 1> {
    // CHECK-LABEL: llvm.func @alloc_rank1
    // CHECK-SAME:  (%[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[N:[a-zA-Z0-9_]+]]: i64)
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<1 x i64> {alignment = 8 : i64}
    // CHECK:       %[[G0:.*]] = llvm.getelementptr %[[SHAPE]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, i64
    // CHECK:       llvm.store %[[N]], %[[G0]] : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(17 : i64) : i64
    // CHECK:       %[[RANK:.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK:       %[[ELEM:.*]] = llvm.mlir.constant(4 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[CTX]], %[[OUTIDX]], %[[SHAPE]], %[[RANK]], %[[ELEM]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %0 = hip.alloc_output(%ctx, %n) {out_idx = 17 : i64} : memref<?xf32, 1>
    return %0 : memref<?xf32, 1>
  }

  // Test 5: Default address space (AS 0): the runtime returns an AS 0 pointer,
  // so NO addrspacecast is emitted and the raw pointer goes straight into the
  // descriptor (whose pointer fields are AS 0).
  func.func @alloc_as0(%ctx: !hip.context) -> memref<4x8xf16> {
    // CHECK-LABEL: llvm.func @alloc_as0
    // CHECK-SAME:  -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64> {alignment = 8 : i64}
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(19 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %[[OUTIDX]], %[[SHAPE]], %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK-NOT:   llvm.addrspacecast
    // CHECK:       llvm.insertvalue %[[RAW]], %{{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %0 = hip.alloc_output(%ctx) {out_idx = 19 : i64} : memref<4x8xf16>
    return %0 : memref<4x8xf16>
  }

  // Test 6: Dynamic size sourced from memref.dim. The dim lowers to an
  // extractvalue of the source descriptor's size field and is stored into the
  // shape array; dim 1 is the static constant 64.
  func.func @alloc_dim(%ctx: !hip.context, %src: memref<?x64xf32, 1>) -> memref<?x64xf32, 1> {
    // CHECK-LABEL: llvm.func @alloc_dim
    // CHECK:       %[[DIM:.*]] = llvm.extractvalue %{{.*}}[3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:       %[[D1:.*]] = llvm.mlir.constant(64 : index) : i64
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i64> {alignment = 8 : i64}
    // CHECK:       llvm.store %[[DIM]], %{{.*}} : i64, !llvm.ptr
    // CHECK:       llvm.store %[[D1]], %{{.*}} : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(23 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %[[OUTIDX]], %[[SHAPE]], %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %c0 = arith.constant 0 : index
    %d = memref.dim %src, %c0 : memref<?x64xf32, 1>
    %0 = hip.alloc_output(%ctx, %d) {out_idx = 23 : i64} : memref<?x64xf32, 1>
    return %0 : memref<?x64xf32, 1>
  }

  // Test 7: Dynamic size computed by integer arithmetic (arith.muli). The
  // product feeds the shape array directly.
  func.func @alloc_arith(%ctx: !hip.context, %a: index, %b: index) -> memref<?xf32, 1> {
    // CHECK-LABEL: llvm.func @alloc_arith
    // CHECK-SAME:  (%[[CTX:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[A:[a-zA-Z0-9_]+]]: i64, %[[B:[a-zA-Z0-9_]+]]: i64)
    // CHECK:       %[[PROD:.*]] = llvm.mul %[[A]], %[[B]] : i64
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<1 x i64> {alignment = 8 : i64}
    // CHECK:       llvm.store %[[PROD]], %{{.*}} : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(31 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%[[CTX]], %[[OUTIDX]], %[[SHAPE]], %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %p = arith.muli %a, %b : index
    %0 = hip.alloc_output(%ctx, %p) {out_idx = 31 : i64} : memref<?xf32, 1>
    return %0 : memref<?xf32, 1>
  }

  // Test 8: Dynamic size loaded from data (memref.load). The loaded i64 feeds
  // the shape array (the arith.index_cast is identity once index == i64).
  func.func @alloc_data(%ctx: !hip.context, %shape: memref<1xi64, 1>) -> memref<?xf32, 1> {
    // CHECK-LABEL: llvm.func @alloc_data
    // CHECK:       %[[LOADED:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i64
    // CHECK:       %[[SHAPE:.*]] = llvm.alloca %{{.*}} x !llvm.array<1 x i64> {alignment = 8 : i64}
    // CHECK:       llvm.store %[[LOADED]], %{{.*}} : i64, !llvm.ptr
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(37 : i64) : i64
    // CHECK:       %[[RAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %[[OUTIDX]], %[[SHAPE]], %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       llvm.addrspacecast %[[RAW]] : !llvm.ptr to !llvm.ptr<1>
    %c0 = arith.constant 0 : index
    %v = memref.load %shape[%c0] : memref<1xi64, 1>
    %d = arith.index_cast %v : i64 to index
    %0 = hip.alloc_output(%ctx, %d) {out_idx = 37 : i64} : memref<?xf32, 1>
    return %0 : memref<?xf32, 1>
  }

  // Test 9: Downstream use. The allocated output buffer is consumed as the
  // `outs` of hip.sigmoid: the descriptor's aligned pointer must flow into the
  // runtime activation call, proving the lowered descriptor is usable.
  func.func @alloc_then_sigmoid(%ctx: !hip.context, %in: memref<3x5xf16, 1>) {
    // CHECK-LABEL: llvm.func @alloc_then_sigmoid
    // CHECK:       %[[OUTIDX:.*]] = llvm.mlir.constant(29 : i64) : i64
    // CHECK:       %[[OUTRAW:.*]] = llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %[[OUTIDX]], %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
    // CHECK:       %[[OUTCAST:.*]] = llvm.addrspacecast %[[OUTRAW]] : !llvm.ptr to !llvm.ptr<1>
    // CHECK:       llvm.insertvalue %[[OUTCAST]], %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK:       llvm.call @wrap_sigmoid(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
    %0 = hip.alloc_output(%ctx) {out_idx = 29 : i64} : memref<3x5xf16, 1>
    hip.sigmoid(%ctx) ins(%in : memref<3x5xf16, 1>) outs(%0 : memref<3x5xf16, 1>)
    return
  }
}
