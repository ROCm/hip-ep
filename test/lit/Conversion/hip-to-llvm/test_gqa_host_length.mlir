// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// Forward each stored value, including truncation. Keep stores for GPU readers.
// CHECK-LABEL: llvm.func @distinct_lengths(
// CHECK: %[[A:.*]] = llvm.trunc {{.*}} : i64 to i32
// CHECK: llvm.store %[[A]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[A]])
// CHECK: llvm.store %[[B:arg[0-9]+]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[B]])
func.func @distinct_lengths(%ctx: !hip.context, %a: i64, %b: i32,
    %q: memref<1x1x64xf16>, %total: memref<i32>, %out: memref<1x1x64xf16>,
    %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  %n = arith.trunci %a : i64 to i32
  memref.store %n, %len[%c0] : memref<1xi32>
  hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
      memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
      memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
    outs(%out, %pk, %pv :
      memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
    {num_heads = 1 : i64, kv_num_heads = 1 : i64}
  %len2 = memref.view %scratch[%c64][] : memref<?xi8> to memref<1xi32>
  memref.store %b, %len2[%c0] : memref<1xi32>
  hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len2, %total :
      memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
      memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
    outs(%out, %pk, %pv :
      memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
    {num_heads = 1 : i64, kv_num_heads = 1 : i64}
  return
}

// -----

// A later store can race with an earlier GPU reader.
// CHECK-LABEL: llvm.func @rewritten_slot(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
func.func @rewritten_slot(%ctx: !hip.context, %a: i32, %b: i32, %q: memref<1x1x64xf16>,
    %total: memref<i32>, %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
    %pv: memref<1x1x128x64xf16>) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
      memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
      memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
    outs(%out, %pk, %pv :
      memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
    {num_heads = 1 : i64, kv_num_heads = 1 : i64}
  memref.store %b, %len[%c0] : memref<1xi32>
  return
}

// -----

// A host-mapped pointer does not prove a host-produced value.
// CHECK-LABEL: llvm.func @device_writer(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
func.func @device_writer(%ctx: !hip.context, %a: i32, %source: memref<1xi64>,
    %q: memref<1x1x64xf16>, %total: memref<i32>, %out: memref<1x1x64xf16>,
    %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %a, %len[%c0] : memref<1xi32>
  hip.cast(%ctx) ins(%source : memref<1xi64>)
    outs(%len : memref<1xi32>) {to = 6 : i64}
  hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
      memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
      memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
    outs(%out, %pk, %pv :
      memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
    {num_heads = 1 : i64, kv_num_heads = 1 : i64}
  return
}

// -----

// Keep multi-element length inputs on the existing path.
// CHECK-LABEL: llvm.func @multi_element(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
func.func @multi_element(%ctx: !hip.context, %a: i32, %b: i32, %q: memref<1x1x64xf16>,
    %total: memref<i32>, %out: memref<1x1x64xf16>, %pk: memref<1x1x128x64xf16>,
    %pv: memref<1x1x128x64xf16>) {
  %c0 = arith.constant 0 : index
  %size = arith.constant 128 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<2xi32>
  memref.store %a, %len[%c0] : memref<2xi32>
  %c1 = arith.constant 1 : index
  memref.store %b, %len[%c1] : memref<2xi32>
  hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
      memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
      memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<2xi32>, memref<i32>)
    outs(%out, %pk, %pv :
      memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
    {num_heads = 1 : i64, kv_num_heads = 1 : i64}
  return
}
