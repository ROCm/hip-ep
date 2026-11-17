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
