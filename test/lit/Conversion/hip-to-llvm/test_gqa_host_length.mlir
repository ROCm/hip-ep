// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --convert-hip-to-llvm %s | FileCheck %s

// Forward each slot's stored value. Keep the stores for GPU consumers.
// CHECK-LABEL: llvm.func @distinct_lengths(
// CHECK: llvm.store %[[A:arg[0-9]+]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[A]])
// CHECK: llvm.store %[[B:arg[0-9]+]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[B]])
module {
  func.func @distinct_lengths(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
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
}

// -----

// Forward the narrowed value, not its wider source.
// CHECK-LABEL: llvm.func @narrowed_length(
// CHECK: %[[N:.*]] = llvm.trunc {{.*}} : i64 to i32
// CHECK: llvm.store %[[N]],
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %[[N]])
module {
  func.func @narrowed_length(%ctx: !hip.context, %a: i64, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
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
    return
  }
}

// -----

// A later store can race with an earlier GPU reader.
// CHECK-LABEL: llvm.func @rewritten_slot(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @rewritten_slot(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
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

// A host-mapped pointer does not prove a host-produced value.
// CHECK-LABEL: llvm.func @device_writer(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @device_writer(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    hip.cast(%ctx) ins(%q : memref<1x1x64xf16>)
      outs(%len : memref<1xi32>) {to = 6 : i64}
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

// Do not forward through an unproved alias.
// CHECK-LABEL: llvm.func @alias(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @alias(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    %alias = memref.cast %len : memref<1xi32> to memref<?xi32>
    memref.store %b, %alias[%c0] : memref<?xi32>
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

// Different views name the same bytes. Forward the last exact store.
// CHECK-LABEL: llvm.func @overlap(
// CHECK: llvm.store %arg2,
// CHECK: llvm.call @wrap_group_query_attention_host_length({{.*}}, %arg2)
module {
  func.func @overlap(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    %alias = memref.view %scratch[%c0][] : memref<?xi8> to memref<i32>
    memref.store %b, %alias[] : memref<i32>
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

// An unknown call can change or resize the runtime scratch.
// CHECK-LABEL: llvm.func @unknown_call(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func private @unknown(!hip.context)
  func.func @unknown_call(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    func.call @unknown(%ctx) : (!hip.context) -> ()
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

// The runtime state owns all host-scratch requests.
// CHECK-LABEL: llvm.func @second_request(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @second_request(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    %other = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
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

// A store after the call does not provide its input.
// CHECK-LABEL: llvm.func @late_store(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @late_store(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %size = arith.constant 128 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    memref.store %a, %len[%c0] : memref<1xi32>
    return
  }
}

// -----

// Keep multi-element length inputs on the existing path.
// CHECK-LABEL: llvm.func @multi_element(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @multi_element(%ctx: !hip.context, %a: i32, %b: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
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
}

// -----

// An i24 array has padding on this target. Bit widths cannot prove disjoint slots.
// CHECK-LABEL: llvm.func @padded_element(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @padded_element(%ctx: !hip.context, %a: i32, %b: i24,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c12][] : memref<?xi8> to memref<1xi32>
    %other = memref.view %scratch[%c0][] : memref<?xi8> to memref<4xi24>
    memref.store %a, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    memref.store %b, %other[%c3] : memref<4xi24>
    return
  }
}

// -----

// Control flow must retain the readback path.
// CHECK-LABEL: llvm.func @control_flow(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func @control_flow(%ctx: !hip.context, %a: i32, %cond: i1,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    cf.cond_br %cond, ^next, ^done
  ^next:
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    cf.br ^done
  ^done:
    return
  }
}

// -----

// A called body can run again before its previous GPU readers finish.
// CHECK-LABEL: llvm.func @called_body(
// CHECK-NOT: llvm.call @wrap_group_query_attention_host_length
// CHECK: llvm.call @wrap_group_query_attention(
module {
  func.func private @called_body(%ctx: !hip.context, %a: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    %c0 = arith.constant 0 : index
    %size = arith.constant 64 : index
    %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
    %len = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
    memref.store %a, %len[%c0] : memref<1xi32>
    hip.gqa(%ctx) ins(%q, %q, %q, %pk, %pv, %len, %total :
        memref<1x1x64xf16>, memref<1x1x64xf16>, memref<1x1x64xf16>,
        memref<1x1x128x64xf16>, memref<1x1x128x64xf16>, memref<1xi32>, memref<i32>)
      outs(%out, %pk, %pv :
        memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>)
      {num_heads = 1 : i64, kv_num_heads = 1 : i64}
    return
  }
  func.func @caller(%ctx: !hip.context, %a: i32,
      %q: memref<1x1x64xf16>, %total: memref<i32>,
      %out: memref<1x1x64xf16>,
      %pk: memref<1x1x128x64xf16>, %pv: memref<1x1x128x64xf16>) {
    func.call @called_body(%ctx, %a, %q, %total, %out, %pk, %pv) :
      (!hip.context, i32, memref<1x1x64xf16>, memref<i32>,
       memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>) -> ()
    func.call @called_body(%ctx, %a, %q, %total, %out, %pk, %pv) :
      (!hip.context, i32, memref<1x1x64xf16>, memref<i32>,
       memref<1x1x64xf16>, memref<1x1x128x64xf16>, memref<1x1x128x64xf16>) -> ()
    return
  }
}
