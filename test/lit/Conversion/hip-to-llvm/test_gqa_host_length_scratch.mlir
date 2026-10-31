// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// A separate shape buffer can have several stores before its readers.
// CHECK-LABEL: llvm.func @shape_copy(
// CHECK: llvm.store %[[L:arg[0-9]+]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[L]])
module {
  func.func @shape_copy(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
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
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}

// -----

// A final shape copy can read an alias after all scratch stores.
// CHECK-LABEL: llvm.func @shape_alias_copy(
// CHECK: llvm.store %[[L:arg[0-9]+]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[L]])
module {
  func.func @shape_alias_copy(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %bytes = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %bytes) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %length, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    %shape = memref.view %scratch[%c64][] : memref<?xi8> to memref<3xi64>
    memref.store %dim, %shape[%c0] : memref<3xi64>
    memref.store %dim, %shape[%c1] : memref<3xi64>
    memref.store %dim, %shape[%c2] : memref<3xi64>
    %base, %offset, %size, %stride = memref.extract_strided_metadata %shape :
        memref<3xi64> -> memref<i64>, index, index, index
    %part = memref.reinterpret_cast %base to offset: [1], sizes: [1], strides: [1] :
        memref<i64> to memref<1xi64, strided<[1], offset: 1>>
    memref.copy %part, %copy : memref<1xi64, strided<[1], offset: 1>> to memref<1xi64>
    return
  }
}

// -----

// The alias store is disjoint from the length and precedes its readers.
// CHECK-LABEL: llvm.func @alias_store(
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %arg1)
module {
  func.func @alias_store(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
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
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}

// -----

// The copy writes another region. It does not change the length.
// CHECK-LABEL: llvm.func @alias_copy_target(
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %arg1)
module {
  func.func @alias_copy_target(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
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
    memref.copy %copy, %part : memref<1xi64> to memref<1xi64, strided<[1], offset: 1>>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}

// -----

// The pending alias reader cannot overlap the later length store.
// CHECK-LABEL: llvm.func @alias_before_store(
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %arg1)
module {
  func.func @alias_before_store(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
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
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}

// -----

// A later direct store can race with a shape-buffer reader.
// CHECK-LABEL: llvm.func @shape_late_store(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
module {
  func.func @shape_late_store(%ctx: !hip.context, %length: i32, %dim: i64,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
      %pv: memref<1x1x128x64xf16>, %copy: memref<1xi64>,
      %copy3: memref<3xi64>) {
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
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
}
