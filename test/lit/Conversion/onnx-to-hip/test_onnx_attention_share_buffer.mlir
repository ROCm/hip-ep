// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Cover the present-capacity rule under the convert-onnx-to-hip pass option
// `kv-share-buffer`, which says present IS the past allocation rather than a
// freshly grown past+current. In a session the option is set from the
// `kv_share_buffer` provider option, which has to agree with
// past_present_share_buffer in the model's genai_config.json.
//
// The default rule is capacity = max(past extent, valid total). That is right
// for a separately allocated present, and right for a shared one too as long
// as the shared buffer is allocated to max_length, where the past extent
// dominates. It breaks only when a buffer is deliberately allocated SHORTER
// than the context -- right-sizing a sliding-window layer's cache to its
// window -- where max(1024, 2240) asks ORT for a present the bound buffer
// cannot satisfy. Nothing in the operand shapes distinguishes the two
// deployments (both have past < total), so the option supplies the fact.
//
// Two shapes of present reach this code and they take different paths:
//
//   dynamic present seq dim -> capacity is computed in IR, and the flag picks
//     dim(past_key, 2) over the max(). This is the path the real decoder
//     takes; its present is tensor<?x8x?x256xf16>.
//
//   static present seq dim -> capacity is a constant read straight off the
//     result type and the computation is skipped entirely, so the option has
//     nothing to act on. That silent bypass is what section 2 pins: under the
//     flag a present pinned to an extent the shared past does not have is a
//     contradiction in the graph, so the converter declines instead of
//     emitting a capacity it knows to be wrong.
//
// The absence of any static-present case is what let the bypass go unnoticed:
// every capacity assertion in test_onnx_attention.mlir exercises the dynamic
// path, so the suite stayed green on a path the model never took.
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: hip-mlir-opt %s --hip-add-context-arg \
// RUN:   --convert-onnx-to-hip=kv-share-buffer=true --split-input-file \
// RUN:   | FileCheck %s --check-prefix=SHARE

// Section 1 -- dynamic present, the real decoder's shape.
//
// Default: capacity is max(past extent, mask kv extent).
// Shared:  capacity is the past extent alone, so no max() is emitted at all.
module {
  func.func @main_graph(
      %query: tensor<?x?x4096xf16>,
      %key: tensor<?x?x2048xf16>,
      %value: tensor<?x?x2048xf16>,
      %attn_mask: tensor<?x1x?x?xf16>,
      %past_key: tensor<?x8x?x256xf16>,
      %past_value: tensor<?x8x?x256xf16>)
      -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>) {

    // DEFAULT-LABEL: func.func @main_graph
    // DEFAULT: %[[PAST:.*]] = tensor.dim %{{.*}}, %c2
    // DEFAULT: %[[MASKKV:.*]] = tensor.dim %{{.*}}, %c3
    // DEFAULT: %[[CAP:.*]] = arith.maxsi %[[PAST]], %[[MASKKV]]
    // DEFAULT: tensor.empty(%{{.*}}, %[[CAP]]) : tensor<?x8x?x256xf16>
    // DEFAULT: hip.gqa

    // SHARE-LABEL: func.func @main_graph
    // SHARE-NOT: arith.maxsi
    // SHARE: %[[SPAST:.*]] = tensor.dim %{{.*}}, %c2
    // SHARE: tensor.empty(%{{.*}}, %[[SPAST]]) : tensor<?x8x?x256xf16>
    // SHARE: hip.gqa

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 1 : si64,
         scale = 1.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<?x?x4096xf16>, tensor<?x?x2048xf16>, tensor<?x?x2048xf16>,
           tensor<?x1x?x?xf16>, tensor<?x8x?x256xf16>,
           tensor<?x8x?x256xf16>)
        -> (tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
            tensor<?x8x?x256xf16>)

    return %out#0, %out#1, %out#2
        : tensor<?x?x4096xf16>, tensor<?x8x?x256xf16>,
          tensor<?x8x?x256xf16>
  }
}

// -----

// Section 2 -- static present (2240) that does NOT match the shared past
// buffer (1024). This is the bypass: the capacity is a constant lifted from
// the result type, so the flag never gets a say.
//
// Default: unchanged, capacity 2240, because a separately allocated present
//          really is past+current and the graph is self-consistent.
// Shared:  declined. present and past are one allocation and cannot have two
//          different extents; the type cannot be rewritten here either,
//          because present is a function result whose signature still says
//          2240. onnx.Attention survives unconverted.
module {
  func.func @main_graph(
      %query: tensor<1x2240x4096xf16>,
      %key: tensor<1x2240x2048xf16>,
      %value: tensor<1x2240x2048xf16>,
      %attn_mask: tensor<1x1x2240x2240xf16>,
      %past_key: tensor<1x8x1024x256xf16>,
      %past_value: tensor<1x8x1024x256xf16>)
      -> (tensor<1x2240x4096xf16>, tensor<1x8x2240x256xf16>,
          tensor<1x8x2240x256xf16>) {

    // DEFAULT-LABEL: func.func @main_graph
    // DEFAULT: arith.constant dense<2240> : tensor<i32>
    // DEFAULT: hip.gqa

    // SHARE-LABEL: func.func @main_graph
    // SHARE: onnx.Attention
    // SHARE-NOT: hip.gqa

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 1 : si64,
         scale = 1.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<1x2240x4096xf16>, tensor<1x2240x2048xf16>,
           tensor<1x2240x2048xf16>, tensor<1x1x2240x2240xf16>,
           tensor<1x8x1024x256xf16>, tensor<1x8x1024x256xf16>)
        -> (tensor<1x2240x4096xf16>, tensor<1x8x2240x256xf16>,
            tensor<1x8x2240x256xf16>)

    return %out#0, %out#1, %out#2
        : tensor<1x2240x4096xf16>, tensor<1x8x2240x256xf16>,
          tensor<1x8x2240x256xf16>
  }
}

// -----

// Section 3 -- static present that DOES match the shared past buffer, i.e. a
// right-sized window cache the exporter already spelled correctly. Nothing to
// disagree about, so both runs convert identically and the flag is inert.
module {
  func.func @main_graph(
      %query: tensor<1x2240x4096xf16>,
      %key: tensor<1x2240x2048xf16>,
      %value: tensor<1x2240x2048xf16>,
      %attn_mask: tensor<1x1x2240x2240xf16>,
      %past_key: tensor<1x8x1024x256xf16>,
      %past_value: tensor<1x8x1024x256xf16>)
      -> (tensor<1x2240x4096xf16>, tensor<1x8x1024x256xf16>,
          tensor<1x8x1024x256xf16>) {

    // DEFAULT-LABEL: func.func @main_graph
    // DEFAULT: arith.constant dense<1024> : tensor<i32>
    // DEFAULT: hip.gqa

    // SHARE-LABEL: func.func @main_graph
    // SHARE: arith.constant dense<1024> : tensor<i32>
    // SHARE: hip.gqa

    %out:3 = "onnx.Attention"(%query, %key, %value, %attn_mask, %past_key,
                              %past_value)
        {q_num_heads = 16 : si64,
         kv_num_heads = 8 : si64,
         is_causal = 1 : si64,
         scale = 1.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<1x2240x4096xf16>, tensor<1x2240x2048xf16>,
           tensor<1x2240x2048xf16>, tensor<1x1x2240x2240xf16>,
           tensor<1x8x1024x256xf16>, tensor<1x8x1024x256xf16>)
        -> (tensor<1x2240x4096xf16>, tensor<1x8x1024x256xf16>,
            tensor<1x8x1024x256xf16>)

    return %out#0, %out#1, %out#2
        : tensor<1x2240x4096xf16>, tensor<1x8x1024x256xf16>,
          tensor<1x8x1024x256xf16>
  }
}
