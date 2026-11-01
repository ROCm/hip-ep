// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// Do not forward through an unproved alias.
// CHECK-LABEL: llvm.func @alias(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @alias(%ctx: !hip.context, %a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  %alias = memref.cast %len : memref<1xi32> to memref<?xi32>
  memref.store %b, %alias[%c0] : memref<?xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// Different views name the same bytes. Forward the last exact store.
// CHECK-LABEL: llvm.func @overlap(
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg2 : i32
func.func @overlap(%ctx: !hip.context, %a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  %alias = memref.view %scratch[%c0][] : memref<?xi8> to memref<i32>
  memref.store %b, %alias[] : memref<i32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// An unknown call can change or resize the runtime scratch.
// CHECK-LABEL: llvm.func @unknown_call(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
module {
  func.func private @unknown(!hip.context)
  func.func @unknown_call(%ctx: !hip.context, %a: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    func.call @unknown(%ctx) : (!hip.context) -> ()
    %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
    return %r : i32
  }
}

// -----

// The runtime state owns all host-scratch requests.
// CHECK-LABEL: llvm.func @second_request(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @second_request(%ctx: !hip.context, %a: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  %other = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// A store after the call does not provide its input.
// CHECK-LABEL: llvm.func @late_store(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @late_store(%ctx: !hip.context, %a: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  memref.store %a, %len[%c0] : memref<1xi32>
  return %r : i32
}

// -----

// An i24 array has padding on this target. Bit widths cannot prove disjoint slots.
// CHECK-LABEL: llvm.func @padded_element(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @padded_element(%ctx: !hip.context, %a: i32, %b: i24) -> i32 {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c12 = arith.constant 12 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c12][] : memref<?xi8> to memref<1xi32>
  %other = memref.view %scratch[%c0][] : memref<?xi8> to memref<4xi24>
  memref.store %a, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  memref.store %b, %other[%c3] : memref<4xi24>
  return %r : i32
}

// -----

// Control flow must retain the readback path.
// CHECK-LABEL: llvm.func @control_flow(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @control_flow(%ctx: !hip.context, %a: i32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  cf.cond_br %cond, ^next, ^done
^next:
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  cf.br ^done
^done:
  return
}

// -----

// A called body can run again before its previous GPU readers finish.
// CHECK-LABEL: llvm.func @called_body(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
module {
  func.func private @called_body(%ctx: !hip.context, %a: i32) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
    return
  }
  func.func @caller(%ctx: !hip.context, %a: i32) {
    func.call @called_body(%ctx, %a) :
      (!hip.context, i32) -> ()
    func.call @called_body(%ctx, %a) :
      (!hip.context, i32) -> ()
    return
  }
}

// -----

// The length is known, but its readback also waits for the earlier cast.
// A host read of another buffer can depend on that wait.
// CHECK-LABEL: llvm.func @external_host_read(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @external_host_read(%ctx: !hip.context, %length: i32, %source: memref<1xi64>,
  %shared: memref<1xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%shared : memref<1xi32>) {to = 6 : i64}
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  %value = memref.load %shared[%c0] : memref<1xi32>
  return %value : i32
}

// -----

// A non-flat copy can lower to a host copy, not a stream operation.
// CHECK-LABEL: llvm.func @host_copy_fallback(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @host_copy_fallback(%ctx: !hip.context, %length: i32,
  %source: memref<2xi32, strided<[2]>>, %target: memref<2xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  memref.copy %source, %target : memref<2xi32, strided<[2]>> to memref<2xi32>
  return %r : i32
}

// -----

// A different context does not share the proven stream or completion wait.
// CHECK-LABEL: llvm.func @different_context(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @different_context(%ctx: !hip.context, %other: !hip.context, %length: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%other, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// A checked scratch load is safe. It reads the sole host store.
// CHECK-LABEL: llvm.func @checked_scratch_read(
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return
func.func @checked_scratch_read(%ctx: !hip.context, %length: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  %value = memref.load %len[%c0] : memref<1xi32>
  return %value : i32
}

// -----

// A later host write can race with an earlier GPU read outside scratch.
// CHECK-LABEL: llvm.func @external_host_write(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @external_host_write(%ctx: !hip.context, %length: i32, %shared: memref<1xi32>,
  %target: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%shared : memref<1xi32>)
    outs(%target : memref<1xi64>) {to = 7 : i64}
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  memref.store %length, %shared[%c0] : memref<1xi32>
  return %r : i32
}

// -----

// Other host effects also need a proof. Do not check only load and store.
// CHECK-LABEL: llvm.func @external_host_atomic(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @external_host_atomic(%ctx: !hip.context, %length: i32, %shared: memref<1xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  %old = memref.atomic_rmw addi %length, %shared[%c0] : (i32, memref<1xi32>) -> i32
  return %r : i32
}

// -----

// Stack storage expires before the caller's final stream wait.
// CHECK-LABEL: llvm.func @stack_lifetime(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @stack_lifetime(%ctx: !hip.context, %length: i32, %source: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %temporary = memref.alloca() : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%temporary : memref<1xi32>) {to = 6 : i64}
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}
