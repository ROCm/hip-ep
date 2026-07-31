// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-split-duplicate-dps-inits %s | FileCheck %s

// When CSE has merged the distinct seeds of two init (DPS-out) operands of the
// SAME op onto one `tensor.empty`, the pass re-points every occurrence after
// the first to a fresh `bufferization.alloc_tensor`, so bufferization gives
// each tied result its own buffer (the GQA present_key/present_value clobber,
// where the kernel would otherwise read V as K from the shared buffer).

// CHECK-LABEL: func.func @gqa_duplicate_present_inits
func.func @gqa_duplicate_present_inits(
    %ctx: !hip.context,
    %query: tensor<1x1x4096xf16>,
    %key: tensor<1x1x1024xf16>,
    %value: tensor<1x1x1024xf16>,
    %past_key: tensor<1x8x127x128xf16>,
    %past_value: tensor<1x8x127x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>)
    -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {
  %eo  = tensor.empty() : tensor<1x1x4096xf16>
  // One shared empty feeds BOTH present_key and present_value.
  %ekv = tensor.empty() : tensor<1x8x128x128xf16>
  // CHECK: %[[EO:.*]] = tensor.empty() : tensor<1x1x4096xf16>
  // CHECK: %[[EKV:.*]] = tensor.empty() : tensor<1x8x128x128xf16>
  // The second occurrence is rewritten to a distinct alloc_tensor:
  // CHECK: %[[FRESH:.*]] = bufferization.alloc_tensor() : tensor<1x8x128x128xf16>
  // CHECK: hip.gqa(%{{.*}}) ins(
  // CHECK-SAME: outs(%[[EO]], %[[EKV]], %[[FRESH]] :
  %r:3 = hip.gqa(%ctx)
      ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
          tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
          tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
          tensor<1xi32>, tensor<i32>)
      outs(%eo, %ekv, %ekv :
           tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)
      {num_heads = 32 : i64, kv_num_heads = 8 : i64,
       scale = 0.0883883461 : f32, local_window_size = -1 : i64}
      : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
}

// -----

// A shared empty used by different DPS ops is also split. This keeps CSE from
// creating one large alias class that One-Shot must repeatedly analyze.

// CHECK-LABEL: func.func @cross_op_shared_empty
func.func @cross_op_shared_empty(
    %ctx: !hip.context,
    %input: tensor<?xf32>,
    %size: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %empty = tensor.empty(%size) : tensor<?xf32>
  // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[SIZE:.*]]) : tensor<?xf32>
  // CHECK: %[[SIGMOID:.*]] = hip.sigmoid
  // CHECK-SAME: outs(%[[EMPTY]] : tensor<?xf32>)
  %sigmoid = hip.sigmoid(%ctx)
      ins(%input : tensor<?xf32>)
      outs(%empty : tensor<?xf32>)
      : tensor<?xf32>
  // CHECK: %[[FRESH:.*]] = bufferization.alloc_tensor(%[[SIZE]]) : tensor<?xf32>
  // CHECK: %[[TANH:.*]] = hip.tanh
  // CHECK-SAME: outs(%[[FRESH]] : tensor<?xf32>)
  %tanh = hip.tanh(%ctx)
      ins(%input : tensor<?xf32>)
      outs(%empty : tensor<?xf32>)
      : tensor<?xf32>
  // CHECK: return %[[SIGMOID]], %[[TANH]]
  return %sigmoid, %tanh : tensor<?xf32>, tensor<?xf32>
}

// -----

// Single-use scratch empties (each feeding ONE init of ONE op) are left
// untouched, so benign CSE merges -- and pool packing -- are unaffected.

// CHECK-LABEL: func.func @no_duplicate_left_alone
func.func @no_duplicate_left_alone(
    %ctx: !hip.context,
    %query: tensor<1x1x4096xf16>,
    %key: tensor<1x1x1024xf16>,
    %value: tensor<1x1x1024xf16>,
    %past_key: tensor<1x8x127x128xf16>,
    %past_value: tensor<1x8x127x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>)
    -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {
  %eo = tensor.empty() : tensor<1x1x4096xf16>
  %ek = tensor.empty() : tensor<1x8x128x128xf16>
  %ev = tensor.empty() : tensor<1x8x128x128xf16>
  // CHECK-NOT: bufferization.alloc_tensor
  %r:3 = hip.gqa(%ctx)
      ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
          tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
          tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
          tensor<1xi32>, tensor<i32>)
      outs(%eo, %ek, %ev :
           tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)
      {num_heads = 32 : i64, kv_num_heads = 8 : i64,
       scale = 0.0883883461 : f32, local_window_size = -1 : i64}
      : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
}

// -----

// Safety restriction: a duplicate init whose seed is NOT a `tensor.empty`
// (here a block argument, whose contents are defined) is left untouched --
// substituting an uninitialized buffer could drop data the op reads. Only
// undefined-content `tensor.empty` seeds are rewritten.

// CHECK-LABEL: func.func @non_empty_duplicate_left_alone
func.func @non_empty_duplicate_left_alone(
    %ctx: !hip.context,
    %query: tensor<1x1x4096xf16>,
    %key: tensor<1x1x1024xf16>,
    %value: tensor<1x1x1024xf16>,
    %past_key: tensor<1x8x127x128xf16>,
    %past_value: tensor<1x8x127x128xf16>,
    %seqlens_k: tensor<1xi32>,
    %total_seq_len: tensor<i32>,
    %kv_buf: tensor<1x8x128x128xf16>)
    -> (tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>) {
  %eo = tensor.empty() : tensor<1x1x4096xf16>
  // CHECK-NOT: bufferization.alloc_tensor
  %r:3 = hip.gqa(%ctx)
      ins(%query, %key, %value, %past_key, %past_value, %seqlens_k, %total_seq_len :
          tensor<1x1x4096xf16>, tensor<1x1x1024xf16>, tensor<1x1x1024xf16>,
          tensor<1x8x127x128xf16>, tensor<1x8x127x128xf16>,
          tensor<1xi32>, tensor<i32>)
      outs(%eo, %kv_buf, %kv_buf :
           tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>)
      {num_heads = 32 : i64, kv_num_heads = 8 : i64,
       scale = 0.0883883461 : f32, local_window_size = -1 : i64}
      : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
  return %r#0, %r#1, %r#2 : tensor<1x1x4096xf16>, tensor<1x8x128x128xf16>, tensor<1x8x128x128xf16>
}
