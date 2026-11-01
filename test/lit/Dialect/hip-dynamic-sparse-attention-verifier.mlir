// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @dsa_bad_head_size(%ctx: !hip.context,
    %q: tensor<1x1x256xf16>, %k: tensor<1x1x64xf16>, %v: tensor<1x1x64xf16>,
    %pk: tensor<1x1x32x64xf16>, %pv: tensor<1x1x32x64xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x256xf16>, %ok: tensor<1x1x32x64xf16>,
    %ov: tensor<1x1x32x64xf16>)
    -> (tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>) {
  // expected-error @+1 {{head size must be 128 or 256, got 64}}
  %r:3 = hip.dynamic_sparse_attention(%ctx)
      ins(%q, %k, %v, %pk, %pv, %idx, %cnt, %seq :
          tensor<1x1x256xf16>, tensor<1x1x64xf16>, tensor<1x1x64xf16>,
          tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>, tensor<1x8xi32>,
          tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>,
           tensor<1x1x32x64xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x256xf16>, tensor<1x1x32x64xf16>, tensor<1x1x32x64xf16>
}

// -----

func.func @dsa_too_many_heads_per_group(%ctx: !hip.context,
    %q: tensor<1x1x4352xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x4352xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{num_heads / kv_num_heads must be at most 16, got 34}}
  %r:3 = hip.dynamic_sparse_attention(%ctx)
      ins(%q, %k, %v, %pk, %pv, %idx, %cnt, %seq :
          tensor<1x1x4352xf16>, tensor<1x1x128xf16>, tensor<1x1x128xf16>,
          tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>, tensor<1x8xi32>,
          tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 34 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x4352xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_present_capacity_mismatch(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x33x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{present_key dim 2 must be 32, got 33}}
  %r:3 = hip.dynamic_sparse_attention(%ctx)
      ins(%q, %k, %v, %pk, %pv, %idx, %cnt, %seq :
          tensor<1x1x512xf16>, tensor<1x1x128xf16>, tensor<1x1x128xf16>,
          tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>, tensor<1x8xi32>,
          tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x33x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_bad_indices_type(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi64>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{selected_indices must have element type 'i32', got 'i64'}}
  %r:3 = hip.dynamic_sparse_attention(%ctx)
      ins(%q, %k, %v, %pk, %pv, %idx, %cnt, %seq :
          tensor<1x1x512xf16>, tensor<1x1x128xf16>, tensor<1x1x128xf16>,
          tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>, tensor<1x8xi64>,
          tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = 1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}

// -----

func.func @dsa_negative_scale(%ctx: !hip.context,
    %q: tensor<1x1x512xf16>, %k: tensor<1x1x128xf16>, %v: tensor<1x1x128xf16>,
    %pk: tensor<1x1x32x128xf16>, %pv: tensor<1x1x32x128xf16>,
    %idx: tensor<1x8xi32>, %cnt: tensor<1xi32>, %seq: tensor<1xi32>,
    %out: tensor<1x1x512xf16>, %ok: tensor<1x1x32x128xf16>,
    %ov: tensor<1x1x32x128xf16>)
    -> (tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>) {
  // expected-error @+1 {{scale must be finite and non-negative}}
  %r:3 = hip.dynamic_sparse_attention(%ctx)
      ins(%q, %k, %v, %pk, %pv, %idx, %cnt, %seq :
          tensor<1x1x512xf16>, tensor<1x1x128xf16>, tensor<1x1x128xf16>,
          tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>, tensor<1x8xi32>,
          tensor<1xi32>, tensor<1xi32>)
      outs(%out, %ok, %ov : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>,
           tensor<1x1x32x128xf16>)
      {num_heads = 4 : i64, kv_num_heads = 1 : i64, scale = -1.250000e-01 : f32}
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
  return %r#0, %r#1, %r#2
      : tensor<1x1x512xf16>, tensor<1x1x32x128xf16>, tensor<1x1x32x128xf16>
}
