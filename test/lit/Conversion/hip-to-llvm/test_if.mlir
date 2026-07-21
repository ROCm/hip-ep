// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify `hip.if` lowers to `hipdnn_ep_run_if` plus per-branch trampolines.

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file %s | FileCheck %s

// -----

module {
  func.func @if_simple(%ctx: !hip.context, %out: memref<5xf32> {bufferize.result}) {
    %true = arith.constant true
    hip.if(%ctx, %true) outs(%out : memref<5xf32>)
        then @if_simple_then else @if_simple_else
        {num_outputs = 1 : i32}
    return
  }

  func.func private @if_simple_then(%ctx: !hip.context,
                                    %o: memref<5xf32> {bufferize.result}) {
    return
  }

  func.func private @if_simple_else(%ctx: !hip.context,
                                    %o: memref<5xf32> {bufferize.result}) {
    return
  }

  // CHECK-DAG: llvm.func @hipdnn_ep_run_if(!llvm.ptr, i1, !llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  // CHECK-LABEL: llvm.func @if_simple(
  // CHECK: llvm.mlir.addressof @if_simple_then_trampoline
  // CHECK: llvm.mlir.addressof @if_simple_else_trampoline
  // CHECK: llvm.call @hipdnn_ep_run_if(
  // CHECK-LABEL: llvm.func internal @if_simple_then_trampoline
  // CHECK: llvm.call @if_simple_then(
  // CHECK-LABEL: llvm.func internal @if_simple_else_trampoline
  // CHECK: llvm.call @if_simple_else(
}
