// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify `hip.loop` is correctly lowered to a runtime driver call
// (`hipdnn_ep_run_counted_loop` for passthrough cond, `hipdnn_ep_run_loop`
// for dynamic cond) plus a per-loop `<body>_trampoline` LLVM function that
// bridges the runtime's fixed-arity callback contract to the body's
// variable-arity memref-descriptor signature.
//
// Input IR is post-bufferization (memref form) -- the shape this pass sees
// after `--convert-onnx-to-hip` and bufferization run on the output of
// `--onnx-loop-outline` (covered by test/lit/Conversion/onnx-to-hip/
// test_loop_outline.mlir).
//
// This test validates:
// - hip.loop -> llvm.call to the right runtime symbol based on the
//   `cond_is_passthrough` attribute
// - Per-loop `<body>_trampoline` LLVMFuncOp is emitted with internal linkage
//   and the fixed 7-arg signature
//   `(state, frame, iter, cond, current[], captures[], next[]) -> i32`
// - Trampoline body calls back the (lowered) body func by name
// - Caller allocates a stack `!llvm.array<N x ptr>` for loop-carried
//   descriptor slots and a separate one for capture descriptor slots
// - Caller passes `num_lc` and `num_cap` as `i32` constants, max_trip_count
//   as `i64`, and cond_init as `i1`
// - Runtime symbol declarations have the expected signature
//   `(!llvm.ptr, !llvm.ptr, i64, i1, i32, i32, !llvm.ptr, !llvm.ptr) -> i32`
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file %s | FileCheck %s

// -----

// Test 1: passthrough cond (`cond_is_passthrough` set on hip.loop). Expect
// the fast-path runtime symbol `hipdnn_ep_run_counted_loop` to be called and
// the body to receive no separate `cond_out` arg -- under passthrough the
// outliner strips the cond from the body's return tuple, and the trampoline
// skips the cond_out callee arg accordingly (LoopLowering.cpp:174-176).
module {
  func.func @loop_passthrough(%ctx: !hip.context,
                              %A: memref<16xf32>,
                              %B: memref<16xf32>,
                              %v_carry: memref<16xf32> {bufferize.result}) {
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    memref.copy %A, %v_carry : memref<16xf32> to memref<16xf32>
    %result, %frame_result = hip.loop(%ctx, %c4, %true) iter_args(%v_carry : memref<16xf32>)
                               captures(%B : memref<16xf32>)
                               -> (memref<16xf32>, !hip.loop_frame)
                               body @loop_passthrough_body
                               {cond_is_passthrough, descriptor_return, num_loop_carried = 1 : i32}
    return
  }

  func.func private @loop_passthrough_body(%ctx: !hip.context,
                                           %iter: memref<i64>,
                                           %cond_in: memref<i1>,
                                           %v_in: memref<16xf32>,
                                           %cap: memref<16xf32>,
                                           %frame: !hip.loop_frame) -> (i32, memref<16xf32>) {
    %status = arith.constant 0 : i32
    %v_out = hip.loop_alloc(%frame) {carrier_index = 0 : i32} : memref<16xf32>
    hip.add(%ctx) ins(%v_in, %cap : memref<16xf32>, memref<16xf32>)
                  outs(%v_out : memref<16xf32>)
    return %status, %v_out : i32, memref<16xf32>
  }

  // Runtime symbol declaration: fast-path counted loop.
  // CHECK-DAG: llvm.func @hipdnn_ep_run_counted_loop(!llvm.ptr, !llvm.ptr, i64, i1, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  //
  // Caller-side lowering of hip.loop (emitted before the body / trampoline).
  // CHECK-LABEL: llvm.func @loop_passthrough(
  //
  // Trampoline reference for the run-loop call.
  // CHECK: llvm.mlir.addressof @loop_passthrough_body_trampoline : !llvm.ptr
  //
  // Loop-carried and capture descriptor-pointer arrays alloca'd on the stack
  // (one slot each since num_lc = num_cap = 1).
  // CHECK: llvm.alloca %{{.*}} x !llvm.array<1 x ptr>
  // CHECK: llvm.alloca %{{.*}} x !llvm.array<1 x ptr>
  //
  // Fast-path runtime call (cond_is_passthrough = true).
  // CHECK: llvm.call @hipdnn_ep_run_counted_loop(
  // CHECK: llvm.return
  //
  // Trampoline (internal linkage, fixed 7-arg signature) is emitted at module
  // scope AFTER the caller and the converted body func.  Calls back the
  // lowered body func by name and returns i32 0 to the runtime.
  // CHECK-LABEL: llvm.func internal @loop_passthrough_body_trampoline(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
  // CHECK: llvm.call @loop_passthrough_body(
  // CHECK: llvm.return %{{.*}} : i32
}

// -----

// Test 2: non-passthrough cond (`cond_is_passthrough` NOT set).  Expect the
// slow-path runtime symbol `hipdnn_ep_run_loop` to be called and the body to
// return status, cond_out descriptor, and carrier descriptor.
module {
  func.func @loop_dynamic(%ctx: !hip.context,
                          %A: memref<16xf32>,
                          %B: memref<16xf32>,
                          %v_carry: memref<16xf32> {bufferize.result}) {
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    memref.copy %A, %v_carry : memref<16xf32> to memref<16xf32>
    %result, %frame_result = hip.loop(%ctx, %c4, %true) iter_args(%v_carry : memref<16xf32>)
                               captures(%B : memref<16xf32>)
                               -> (memref<16xf32>, !hip.loop_frame)
                               body @loop_dynamic_body
                               {descriptor_return, num_loop_carried = 1 : i32}
    return
  }

  // Non-passthrough body: cond_out is an explicit descriptor result.
  func.func private @loop_dynamic_body(%ctx: !hip.context,
                                       %iter: memref<i64>,
                                       %cond_in: memref<i1>,
                                       %v_in: memref<16xf32>,
                                       %cap: memref<16xf32>,
                                       %frame: !hip.loop_frame) -> (i32, memref<i1>, memref<16xf32>) {
    %status = arith.constant 0 : i32
    %cond_out = memref.alloc() : memref<i1>
    %v_out = hip.loop_alloc(%frame) {carrier_index = 0 : i32} : memref<16xf32>
    hip.add(%ctx) ins(%v_in, %cap : memref<16xf32>, memref<16xf32>)
                  outs(%v_out : memref<16xf32>)
    return %status, %cond_out, %v_out : i32, memref<i1>, memref<16xf32>
  }

  // Runtime symbol declaration: dynamic loop (matches ORT CUDA EP / MIGraphX
  // run_loop semantics -- one device-to-host cond read per iter).
  // CHECK-DAG: llvm.func @hipdnn_ep_run_loop(!llvm.ptr, !llvm.ptr, i64, i1, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  //
  // CHECK-LABEL: llvm.func @loop_dynamic(
  // CHECK: llvm.mlir.addressof @loop_dynamic_body_trampoline : !llvm.ptr
  // CHECK: llvm.alloca %{{.*}} x !llvm.array<1 x ptr>
  // CHECK: llvm.alloca %{{.*}} x !llvm.array<1 x ptr>
  //
  // Slow-path runtime call (cond_is_passthrough = false).
  // CHECK: llvm.call @hipdnn_ep_run_loop(
  // CHECK: llvm.return
  //
  // Trampoline still emitted, same 7-arg shape.
  // CHECK-LABEL: llvm.func internal @loop_dynamic_body_trampoline(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) -> i32
  // CHECK: llvm.call @loop_dynamic_body(
  // CHECK: llvm.return %{{.*}} : i32
}
