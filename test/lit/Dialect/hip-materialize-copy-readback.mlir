// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --split-input-file --hip-materialize-host-scalars --hip-optimize-memrefs --hip-pool-allocs | FileCheck %s

// Keep the copy-filled shape out of the device pool. Keep its copy and wait.
// CHECK-LABEL: func.func @copied_shape
// CHECK: hip.get_host_scratch
// CHECK-NOT: hip.get_pool
// CHECK: %[[SHAPE:.*]] = memref.view {{.*}} to memref<4xi64>
// CHECK: %[[PART:.*]] = memref.subview %[[SHAPE]][1]
// CHECK: memref.copy %{{.*}}, %[[PART]]
// CHECK: %[[SCALAR:.*]] = memref.collapse_shape %[[PART]]
// CHECK: hip.readback_scalar(%{{.*}}, %[[SCALAR]]
// CHECK: hip.add{{.*}}ins(%[[SHAPE]],
// CHECK-NOT: hip.get_pool
func.func @copied_shape(%ctx: !hip.context, %src: memref<1xi64>,
    %input: memref<4xi64>, %out: memref<4xi64>) -> i64 {
  %shape = memref.alloc() : memref<4xi64>
  %first = memref.subview %shape[0] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1]>>
  memref.copy %src, %first : memref<1xi64> to memref<1xi64, strided<[1]>>
  %part = memref.subview %shape[1] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1], offset: 1>>
  memref.copy %src, %part
      : memref<1xi64> to memref<1xi64, strided<[1], offset: 1>>
  %third = memref.subview %shape[2] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1], offset: 2>>
  memref.copy %src, %third : memref<1xi64> to memref<1xi64, strided<[1], offset: 2>>
  %fourth = memref.subview %shape[3] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1], offset: 3>>
  memref.copy %src, %fourth : memref<1xi64> to memref<1xi64, strided<[1], offset: 3>>
  %scalar = memref.collapse_shape %part []
      : memref<1xi64, strided<[1], offset: 1>> into memref<i64, strided<[], offset: 1>>
  %v = hip.readback_scalar(%ctx, %scalar : memref<i64, strided<[], offset: 1>>) -> i64
  hip.add(%ctx) ins(%shape, %input : memref<4xi64>, memref<4xi64>)
      outs(%out : memref<4xi64>)
  return %v : i64
}

// -----

// A copy alone does not make a device buffer a host-scratch candidate.
// CHECK-LABEL: func.func @no_readback
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
func.func @no_readback(%ctx: !hip.context, %src: memref<1xi64>,
    %out: memref<1xi64>) {
  %a = memref.alloc() : memref<1xi64>
  memref.copy %src, %a : memref<1xi64> to memref<1xi64>
  hip.add(%ctx) ins(%a, %src : memref<1xi64>, memref<1xi64>) outs(%out : memref<1xi64>)
  return
}

// -----

// A dynamic alias does not prove a range inside the scratch slot.
// CHECK-LABEL: func.func @unknown_offset
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
func.func @unknown_offset(%ctx: !hip.context, %src: memref<1xi64>, %offset: index) -> i64 {
  %a = memref.alloc() : memref<4xi64>
  %part = memref.subview %a[%offset] [1] [1]
      : memref<4xi64> to memref<1xi64, strided<[1], offset: ?>>
  memref.copy %src, %part : memref<1xi64> to memref<1xi64, strided<[1], offset: ?>>
  %v = hip.readback_scalar(%ctx, %part : memref<1xi64, strided<[1], offset: ?>>) -> i64
  return %v : i64
}

// -----

// Keep the size limit for copied buffers too.
// CHECK-LABEL: func.func @large_shape
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
func.func @large_shape(%ctx: !hip.context, %src: memref<1xi64>) -> i64 {
  %a = memref.alloc() : memref<17xi64>
  %part = memref.subview %a[0] [1] [1]
      : memref<17xi64> to memref<1xi64, strided<[1]>>
  memref.copy %src, %part : memref<1xi64> to memref<1xi64, strided<[1]>>
  %v = hip.readback_scalar(%ctx, %part : memref<1xi64, strided<[1]>>) -> i64
  return %v : i64
}

// -----

// Do not move kernel destinations under the copy/readback rule.
// CHECK-LABEL: func.func @kernel_writer
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
func.func @kernel_writer(%ctx: !hip.context, %src: memref<1xi64>) -> i64 {
  %a = memref.alloc() : memref<1xi64>
  memref.copy %src, %a : memref<1xi64> to memref<1xi64>
  hip.add(%ctx) ins(%src, %src : memref<1xi64>, memref<1xi64>) outs(%a : memref<1xi64>)
  %v = hip.readback_scalar(%ctx, %a : memref<1xi64>) -> i64
  return %v : i64
}

// -----

// Do not treat a non-flat copy as a context-stream copy.
// CHECK-LABEL: func.func @non_flat_copy
// CHECK-NOT: hip.get_host_scratch
// CHECK: hip.get_pool
func.func @non_flat_copy(%ctx: !hip.context, %src: memref<4xi64, strided<[2]>>) -> i64 {
  %a = memref.alloc() : memref<4xi64>
  memref.copy %src, %a : memref<4xi64, strided<[2]>> to memref<4xi64>
  %part = memref.reinterpret_cast %a to offset: [0], sizes: [], strides: []
      : memref<4xi64> to memref<i64>
  %v = hip.readback_scalar(%ctx, %part : memref<i64>) -> i64
  return %v : i64
}
