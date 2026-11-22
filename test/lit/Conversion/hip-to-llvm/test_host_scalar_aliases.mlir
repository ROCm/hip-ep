// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// Disjoint direct and alias accesses do not invalidate the known scalar.
// CHECK-LABEL: llvm.func @disjoint_alias_accesses(
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i32
func.func @disjoint_alias_accesses(%ctx: !hip.context, %length: i32, %dim: i64, %copy: memref<1xi64>,
    %copy3: memref<3xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %bytes = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %bytes) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<3xi64>
  memref.store %dim, %shape[%c0] : memref<3xi64>
  memref.store %dim, %shape[%c1] : memref<3xi64>
  memref.store %dim, %shape[%c2] : memref<3xi64>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %shape :
      memref<3xi64> -> memref<i64>, index, index, index
  %part = memref.reinterpret_cast %base to offset: [1], sizes: [1], strides: [1] :
      memref<i64> to memref<1xi64, strided<[1], offset: 1>>
  memref.store %dim, %part[%c0] : memref<1xi64, strided<[1], offset: 1>>
  memref.copy %shape, %copy3 : memref<3xi64> to memref<3xi64>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  memref.copy %part, %copy : memref<1xi64, strided<[1], offset: 1>> to memref<1xi64>
  return %r : i32
}

// -----

// The copy writes another region. It does not change the length.
// CHECK-LABEL: llvm.func @alias_copy_target(
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i32
func.func @alias_copy_target(%ctx: !hip.context, %length: i32, %copy: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %bytes = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %bytes) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<3xi64>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %shape :
      memref<3xi64> -> memref<i64>, index, index, index
  %part = memref.reinterpret_cast %base to offset: [1], sizes: [1], strides: [1] :
      memref<i64> to memref<1xi64, strided<[1], offset: 1>>
  memref.copy %copy, %part : memref<1xi64> to memref<1xi64, strided<[1], offset: 1>>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// The pending alias reader cannot overlap the later length store.
// CHECK-LABEL: llvm.func @alias_before_store(
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return %arg1 : i32
func.func @alias_before_store(%ctx: !hip.context, %length: i32, %dim: i64, %copy: memref<1xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %bytes = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %bytes) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<3xi64>
  memref.store %dim, %shape[%c0] : memref<3xi64>
  memref.store %dim, %shape[%c1] : memref<3xi64>
  memref.store %dim, %shape[%c2] : memref<3xi64>
  %base, %offset, %size, %stride = memref.extract_strided_metadata %shape :
      memref<3xi64> -> memref<i64>, index, index, index
  %part = memref.reinterpret_cast %base to offset: [1], sizes: [1], strides: [1] :
      memref<i64> to memref<1xi64, strided<[1], offset: 1>>
  memref.copy %part, %copy : memref<1xi64, strided<[1], offset: 1>> to memref<1xi64>
  memref.store %length, %len[%c0] : memref<1xi32>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}

// -----

// A later direct store can race with a shape-buffer reader.
// CHECK-LABEL: llvm.func @shape_late_store(
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.return
func.func @shape_late_store(%ctx: !hip.context, %length: i32, %dim: i64, %copy3: memref<3xi64>) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %bytes = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %bytes) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %length, %len[%c0] : memref<1xi32>
  %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<3xi64>
  memref.store %dim, %shape[%c0] : memref<3xi64>
  memref.store %dim, %shape[%c1] : memref<3xi64>
  memref.store %dim, %shape[%c2] : memref<3xi64>
  memref.copy %shape, %copy3 : memref<3xi64> to memref<3xi64>
  memref.store %dim, %shape[%c1] : memref<3xi64>
  %r = hip.readback_scalar(%ctx, %len : memref<1xi32>) -> i32
  return %r : i32
}
