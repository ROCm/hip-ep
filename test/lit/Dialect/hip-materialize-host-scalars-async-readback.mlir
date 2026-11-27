// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --hip-materialize-host-scalars | FileCheck %s

// A host load of an element an async memref.copy wrote must wait for it.
// CHECK-LABEL: func.func @load_after_copy
// CHECK: %[[SHAPE:.*]] = memref.view {{.*}} to memref<4xi64>
// CHECK: memref.copy
// CHECK: %[[ELEM:.*]] = memref.subview %[[SHAPE]][%{{.*}}] [1] [1]
// CHECK: %[[SCALAR:.*]] = memref.collapse_shape %[[ELEM]] []
// CHECK: %[[V:.*]] = hip.readback_scalar(%{{.*}}, %[[SCALAR]]
// CHECK-NOT: memref.load
// CHECK: return %[[V]]
func.func @load_after_copy(%ctx: !hip.context, %src: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1 : i64
  %shape = memref.alloc() : memref<4xi64>
  memref.store %one, %shape[%c0] : memref<4xi64>
  %part = memref.subview %shape[1] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1], offset: 1>>
  memref.copy %src, %part
      : memref<1xi64> to memref<1xi64, strided<[1], offset: 1>>
  %v = memref.load %shape[%c1] : memref<4xi64>
  return %v : i64
}

// -----

// A host load of a HIP kernel's output must wait for the kernel.
// CHECK-LABEL: func.func @load_after_kernel
// CHECK: %[[A:.*]] = memref.view {{.*}} to memref<1xi64>
// CHECK: hip.add{{.*}}outs(%[[A]]
// CHECK: %[[V:.*]] = hip.readback_scalar(%{{.*}}, %{{.*}} : memref<i64
// CHECK-NOT: memref.load
// CHECK: return %[[V]]
func.func @load_after_kernel(%ctx: !hip.context, %x: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<1xi64>
  hip.add(%ctx) ins(%x, %x : memref<1xi64>, memref<1xi64>)
      outs(%a : memref<1xi64>)
  %v = memref.load %a[%c0] : memref<1xi64>
  return %v : i64
}

// -----

// A load before any async write reads what the host stored. Keep it.
// CHECK-LABEL: func.func @load_before_write
// CHECK: memref.store
// CHECK: memref.load
// CHECK-NOT: hip.readback_scalar
// CHECK: hip.add
func.func @load_before_write(%ctx: !hip.context, %x: memref<1xi64>,
    %out: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 7 : i64
  %a = memref.alloc() : memref<1xi64>
  memref.store %n, %a[%c0] : memref<1xi64>
  %v = memref.load %a[%c0] : memref<1xi64>
  hip.add(%ctx) ins(%a, %x : memref<1xi64>, memref<1xi64>)
      outs(%out : memref<1xi64>)
  return %v : i64
}

// -----

// An index load through a view of an async-copied byte buffer must wait too.
// CHECK-LABEL: func.func @index_load_through_view
// CHECK: memref.copy
// CHECK: %[[N:.*]] = memref.view %{{.*}} to memref<1xindex>
// CHECK: %[[ELEM:.*]] = memref.subview %[[N]][%{{.*}}] [1] [1]
// CHECK: %[[SCALAR:.*]] = memref.collapse_shape %[[ELEM]] []
// CHECK: %[[V:.*]] = hip.readback_scalar(%{{.*}}, %[[SCALAR]] {{.*}} -> index
// CHECK-NOT: memref.load
// CHECK: return %[[V]]
func.func @index_load_through_view(%ctx: !hip.context, %src: memref<8xi8>)
    -> index {
  %c0 = arith.constant 0 : index
  %buf = memref.alloc() : memref<8xi8>
  memref.copy %src, %buf : memref<8xi8> to memref<8xi8>
  %n = memref.view %buf[%c0][] : memref<8xi8> to memref<1xindex>
  %v = memref.load %n[%c0] : memref<1xindex>
  return %v : index
}

// -----

// Block order does not prove a load in a successor block runs before the
// copy lands. Synchronize it.
// CHECK-LABEL: func.func @load_in_successor
// CHECK: memref.copy
// CHECK: cf.br ^bb1
// CHECK: ^bb1:
// CHECK: %[[V:.*]] = hip.readback_scalar
// CHECK-NOT: memref.load
// CHECK: return %[[V]]
func.func @load_in_successor(%ctx: !hip.context, %src: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<1xi64>
  memref.copy %src, %a : memref<1xi64> to memref<1xi64>
  cf.br ^bb1
^bb1:
  %v = memref.load %a[%c0] : memref<1xi64>
  return %v : i64
}

// -----

// A loop-local slot is reused by the next iteration while this iteration's
// kernel may still be pending. Wait before the host initializes it again.
// The load precedes the kernel in its iteration, so it stays a plain load.
// CHECK-LABEL: func.func @loop_local_load_before_kernel
// CHECK: scf.for
// CHECK: hip.stream_wait(%{{.*}})
// CHECK: %[[A:.*]] = memref.view {{.*}} to memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[A]]
// CHECK: memref.load %[[A]]
// CHECK: hip.add{{.*}}outs(%[[A]]
func.func @loop_local_load_before_kernel(%ctx: !hip.context,
    %x: memref<1xi64>, %out: memref<4xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %n = arith.constant 7 : i64
  scf.for %i = %c0 to %c4 step %c1 {
    %a = memref.alloc() : memref<1xi64>
    memref.store %n, %a[%c0] : memref<1xi64>
    %v = memref.load %a[%c0] : memref<1xi64>
    memref.store %v, %out[%i] : memref<4xi64>
    hip.add(%ctx) ins(%x, %x : memref<1xi64>, memref<1xi64>)
        outs(%a : memref<1xi64>)
  }
  return
}

// -----

// A slot that is not loop-local needs no wait.
// CHECK-LABEL: func.func @no_wait_outside_loop
// CHECK-NOT: hip.stream_wait
// CHECK: return
func.func @no_wait_outside_loop(%ctx: !hip.context, %x: memref<1xi64>)
    -> i64 {
  %c0 = arith.constant 0 : index
  %n = arith.constant 7 : i64
  %a = memref.alloc() : memref<1xi64>
  memref.store %n, %a[%c0] : memref<1xi64>
  %v = memref.load %a[%c0] : memref<1xi64>
  hip.add(%ctx) ins(%x, %x : memref<1xi64>, memref<1xi64>)
      outs(%a : memref<1xi64>)
  return %v : i64
}

// -----

// A loop-local slot only read by an async copy is still in use by the queued
// copy when the next iteration stores into it. Wait before reusing it.
// CHECK-LABEL: func.func @loop_local_copy_source
// CHECK: scf.for
// CHECK: hip.stream_wait(%{{.*}})
// CHECK: %[[A:.*]] = memref.view {{.*}} to memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[A]]
// CHECK: memref.copy %[[A]]
func.func @loop_local_copy_source(%ctx: !hip.context, %dst: memref<1xi64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %n = arith.index_cast %i : index to i64
    %a = memref.alloc() : memref<1xi64>
    memref.store %n, %a[%c0] : memref<1xi64>
    memref.copy %a, %dst : memref<1xi64> to memref<1xi64>
  }
  return
}

// -----

// A readback waits on the function's context only. A buffer written by a
// HIP op on another context stays a device allocation.
// CHECK-LABEL: func.func @kernel_on_other_context
// CHECK-NOT: hip.get_host_scratch
// CHECK: memref.alloc
// CHECK: hip.add
// CHECK: memref.load
// CHECK-NOT: hip.readback_scalar
func.func @kernel_on_other_context(%ctx: !hip.context, %other: !hip.context,
    %x: memref<1xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<1xi64>
  hip.add(%other) ins(%x, %x : memref<1xi64>, memref<1xi64>)
      outs(%a : memref<1xi64>)
  %v = memref.load %a[%c0] : memref<1xi64>
  return %v : i64
}
