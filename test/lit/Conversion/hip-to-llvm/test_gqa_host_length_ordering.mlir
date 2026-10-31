// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// The length is known, but its readback also waits for the earlier cast.
// A host read of another buffer can depend on that wait.
// CHECK-LABEL: llvm.func @external_host_read(
// CHECK: llvm.call @wrap_cast(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK: llvm.load
module {
  func.func @external_host_read(%ctx: !hip.context, %length: i32,
      %source: memref<1xi64>, %shared: memref<1xi32>,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) -> i32 {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.cast(%ctx) ins(%source : memref<1xi64>)
      outs(%shared : memref<1xi32>) {to = 6 : i64}
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    %value = memref.load %shared[%c0] : memref<1xi32>
    return %value : i32
  }
}

// -----

// A non-flat copy can lower to a host copy, not a stream operation.
// CHECK-LABEL: llvm.func @host_copy_fallback(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK: llvm.call @memrefCopy(
module {
  func.func @host_copy_fallback(%ctx: !hip.context, %length: i32,
      %source: memref<2xi32, strided<[2]>>, %target: memref<2xi32>,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    memref.copy %source, %target : memref<2xi32, strided<[2]>> to memref<2xi32>
    return
  }
}

// -----

// A different context does not share the proven stream or completion wait.
// CHECK-LABEL: llvm.func @different_context(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @different_context(%ctx: !hip.context, %other: !hip.context,
      %length: i32, %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%other) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}

// -----

// A checked scratch load is safe. It reads the sole host store.
// CHECK-LABEL: llvm.func @checked_scratch_read(
// CHECK: llvm.call @wrap_group_query_attention_host_length(
// CHECK: llvm.load
module {
  func.func @checked_scratch_read(%ctx: !hip.context, %length: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) -> i32 {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    %value = memref.load %len[%c0] : memref<1xi32>
    return %value : i32
  }
}

// -----

// A later host write can race with an earlier GPU read outside scratch.
// CHECK-LABEL: llvm.func @external_host_write(
// CHECK: llvm.call @wrap_cast(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK: llvm.store
module {
  func.func @external_host_write(%ctx: !hip.context, %length: i32,
      %shared: memref<1xi32>, %target: memref<1xi64>,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.cast(%ctx) ins(%shared : memref<1xi32>)
      outs(%target : memref<1xi64>) {to = 7 : i64}
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    memref.store %length, %shared[%c0] : memref<1xi32>
    return
  }
}

// -----

// Other host effects also need a proof. Do not check only load and store.
// CHECK-LABEL: llvm.func @external_host_atomic(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK: llvm.atomicrmw
module {
  func.func @external_host_atomic(%ctx: !hip.context, %length: i32,
      %shared: memref<1xi32>, %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    %old = memref.atomic_rmw addi %length, %shared[%c0] : (i32, memref<1xi32>) -> i32
    return
  }
}

// -----

// Stack storage expires before the caller's final stream wait.
// CHECK-LABEL: llvm.func @stack_lifetime(
// CHECK: llvm.alloca
// CHECK: llvm.call @wrap_cast(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @stack_lifetime(%ctx: !hip.context, %length: i32,
      %source: memref<1xi64>, %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    %temporary = memref.alloca() : memref<1xi32>
    hip.cast(%ctx) ins(%source : memref<1xi64>)
      outs(%temporary : memref<1xi32>) {to = 6 : i64}
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}
