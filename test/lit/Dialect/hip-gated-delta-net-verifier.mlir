// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --verify-diagnostics --split-input-file %s

func.func @gdn_bad_head_size(%ctx: !hip.context,
    %qkv: tensor<1x1x352xf16>, %decay: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>, %state: tensor<1x4x16x40xf32>,
    %a_log: tensor<4xf32>, %dt_bias: tensor<4xf32>,
    %out: tensor<1x1x4x16xf16>, %final: tensor<1x4x16x40xf32>)
    -> (tensor<1x1x4x16xf16>, tensor<1x4x16x40xf32>) {
  // expected-error @+1 {{head size dk must be 32, 64, 128 or 256, got 40}}
  %r:2 = hip.gated_delta_net(%ctx)
      ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
          tensor<1x1x352xf16>, tensor<1x1x4xf32>, tensor<1x1x4xf32>,
          tensor<1x4x16x40xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%out, %final : tensor<1x1x4x16xf16>, tensor<1x4x16x40xf32>)
      : tensor<1x1x4x16xf16>, tensor<1x4x16x40xf32>
  return %r#0, %r#1 : tensor<1x1x4x16xf16>, tensor<1x4x16x40xf32>
}

// -----

func.func @gdn_bad_packed_width(%ctx: !hip.context,
    %qkv: tensor<1x1x200xf16>, %decay: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>, %state: tensor<1x4x16x32xf32>,
    %a_log: tensor<4xf32>, %dt_bias: tensor<4xf32>,
    %out: tensor<1x1x4x16xf16>, %final: tensor<1x4x16x32xf32>)
    -> (tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>) {
  // expected-error @+1 {{qkv dim 2 (200) minus Hv*dv (64) must be a positive multiple of 2*dk (64)}}
  %r:2 = hip.gated_delta_net(%ctx)
      ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
          tensor<1x1x200xf16>, tensor<1x1x4xf32>, tensor<1x1x4xf32>,
          tensor<1x4x16x32xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%out, %final : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>)
      : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
  return %r#0, %r#1 : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
}

// -----

func.func @gdn_bad_gqa(%ctx: !hip.context,
    %qkv: tensor<1x1x256xf16>, %decay: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>, %state: tensor<1x4x16x32xf32>,
    %a_log: tensor<4xf32>, %dt_bias: tensor<4xf32>,
    %out: tensor<1x1x4x16xf16>, %final: tensor<1x4x16x32xf32>)
    -> (tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>) {
  // expected-error @+1 {{Hv (4) must be a multiple of Hqk (3)}}
  %r:2 = hip.gated_delta_net(%ctx)
      ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
          tensor<1x1x256xf16>, tensor<1x1x4xf32>, tensor<1x1x4xf32>,
          tensor<1x4x16x32xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%out, %final : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>)
      : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
  return %r#0, %r#1 : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
}

// -----

func.func @gdn_bad_decay_heads(%ctx: !hip.context,
    %qkv: tensor<1x1x192xf16>, %decay: tensor<1x1x2xf32>,
    %beta: tensor<1x1x4xf32>, %state: tensor<1x4x16x32xf32>,
    %a_log: tensor<4xf32>, %dt_bias: tensor<4xf32>,
    %out: tensor<1x1x4x16xf16>, %final: tensor<1x4x16x32xf32>)
    -> (tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>) {
  // expected-error @+1 {{decay dim 2 must be 4, got 2}}
  %r:2 = hip.gated_delta_net(%ctx)
      ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
          tensor<1x1x192xf16>, tensor<1x1x2xf32>, tensor<1x1x4xf32>,
          tensor<1x4x16x32xf32>, tensor<4xf32>, tensor<4xf32>)
      outs(%out, %final : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>)
      : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
  return %r#0, %r#1 : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
}

// -----

func.func @gdn_bad_state_type(%ctx: !hip.context,
    %qkv: tensor<1x1x192xf16>, %decay: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>, %state: tensor<1x4x16x32xf16>,
    %a_log: tensor<4xf32>, %dt_bias: tensor<4xf32>,
    %out: tensor<1x1x4x16xf16>, %final: tensor<1x4x16x32xf32>)
    -> (tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>) {
  // expected-error @+1 {{initial_state must be f32, got 'f16'}}
  %r:2 = hip.gated_delta_net(%ctx)
      ins(%qkv, %decay, %beta, %state, %a_log, %dt_bias :
          tensor<1x1x192xf16>, tensor<1x1x4xf32>, tensor<1x1x4xf32>,
          tensor<1x4x16x32xf16>, tensor<4xf32>, tensor<4xf32>)
      outs(%out, %final : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>)
      : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
  return %r#0, %r#1 : tensor<1x1x4x16xf16>, tensor<1x4x16x32xf32>
}
