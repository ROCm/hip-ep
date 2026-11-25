// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --split-input-file --verify-diagnostics --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: not hip-mlir-opt %s --split-input-file --hipdnn-pipeline 2>&1 | FileCheck %s --check-prefix=CAPERROR
// CAPERROR: HIP PagedAttention requires softcap to be omitted or zero; soft-capping is unsupported

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.paged_attention
// CHECK-NOT: softcap
// CHECK: return
module {
  func.func @main_graph(
      %query: tensor<4x256xf16>,
      %key: tensor<4x128xf16>,
      %value: tensor<4x128xf16>,
      %key_cache: tensor<8x16x2x64xf16>,
      %value_cache: tensor<8x16x2x64xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<4x256xf16> {


    %0 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64,
         local_window_size = -1 : si64,
         scale = 0.000000e+00 : f32,
         softcap = 0.000000e+00 : f32}
        : (tensor<4x256xf16>, tensor<4x128xf16>, tensor<4x128xf16>,
           tensor<8x16x2x64xf16>, tensor<8x16x2x64xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x256xf16>

    // Attributes printed in alphabetical order by MLIR's attr-dict printer.

    return %0 : tensor<4x256xf16>
  }
}

// -----

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.paged_attention
// CHECK-NOT: softcap
// CHECK: return
module {
  func.func @main_graph(
      %query: tensor<4x256xf16>,
      %key: tensor<4x128xf16>,
      %value: tensor<4x128xf16>,
      %key_cache: tensor<8x16x2x64xf16>,
      %value_cache: tensor<8x16x2x64xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<4x256xf16> {


    %0 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64,
         local_window_size = -1 : si64,
         scale = 0.000000e+00 : f32}
        : (tensor<4x256xf16>, tensor<4x128xf16>, tensor<4x128xf16>,
           tensor<8x16x2x64xf16>, tensor<8x16x2x64xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x256xf16>

    // Attributes printed in alphabetical order by MLIR's attr-dict printer.

    return %0 : tensor<4x256xf16>
  }}

// -----

module {
  func.func @main_graph(
      %query: tensor<4x256xf16>,
      %key: tensor<4x128xf16>,
      %value: tensor<4x128xf16>,
      %key_cache: tensor<8x16x2x64xf16>,
      %value_cache: tensor<8x16x2x64xf16>,
      %cum_seq: tensor<2xi32>,
      %past_seqlens: tensor<1xi32>,
      %block_table: tensor<1x8xi32>) -> tensor<4x256xf16> {


    // expected-error @+1 {{HIP PagedAttention requires softcap to be omitted or zero; soft-capping is unsupported}}
    %0 = "onnx.Custom"(%query, %key, %value, %key_cache, %value_cache,
                        %cum_seq, %past_seqlens, %block_table)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft",
         num_heads = 4 : si64,
         kv_num_heads = 2 : si64,
         do_rotary = 0 : si64,
         rotary_interleaved = 0 : si64,
         local_window_size = -1 : si64,
         scale = 0.000000e+00 : f32,
         softcap = 1.000000e+00 : f32}
        : (tensor<4x256xf16>, tensor<4x128xf16>, tensor<4x128xf16>,
           tensor<8x16x2x64xf16>, tensor<8x16x2x64xf16>, tensor<2xi32>,
           tensor<1xi32>, tensor<1x8xi32>)
        -> tensor<4x256xf16>

    // Attributes printed in alphabetical order by MLIR's attr-dict printer.

    return %0 : tensor<4x256xf16>
  }
}
