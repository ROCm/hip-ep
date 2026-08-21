// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --split-input-file %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(
      %q: tensor<1x1x1280xf16>,
      %k: tensor<1x1x1280xf16>,
      %v: tensor<1x1x1280xf16>,
      %pastK: tensor<1x20x447x64xf16>,
      %pastV: tensor<1x20x447x64xf16>)
      -> (tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>,
          tensor<1x20x448x64xf16>) {
    %none0 = "onnx.NoValue"() {value} : () -> none
    %none1 = "onnx.NoValue"() {value} : () -> none
    %none2 = "onnx.NoValue"() {value} : () -> none
    // CHECK: error: default MultiHeadAttention runtime does not support bias, masks, past/cache inputs, or cache indirection
    %out:3 = "onnx.Custom"(%q, %k, %v, %none0, %none1, %none2,
                           %pastK, %pastV)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft", num_heads = 20 : si64}
        : (tensor<1x1x1280xf16>, tensor<1x1x1280xf16>,
           tensor<1x1x1280xf16>, none, none, none,
           tensor<1x20x447x64xf16>, tensor<1x20x447x64xf16>)
        -> (tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>,
            tensor<1x20x448x64xf16>)
    return %out#0, %out#1, %out#2
        : tensor<1x1x1280xf16>, tensor<1x20x448x64xf16>,
          tensor<1x20x448x64xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %q: tensor<1x8x130xf16>,
      %k: tensor<1x16x130xf16>,
      %v: tensor<1x16x130xf16>) -> tensor<1x8x130xf16> {
    // CHECK: error: multi_head_attention query hidden extent 130 must be divisible by num_heads 8
    %out = "onnx.Custom"(%q, %k, %v)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft", num_heads = 8 : si64}
        : (tensor<1x8x130xf16>, tensor<1x16x130xf16>,
           tensor<1x16x130xf16>) -> tensor<1x8x130xf16>
    return %out : tensor<1x8x130xf16>
  }
}

// -----

module {
  func.func @main_graph(
      %q: tensor<1x8x128xf32>,
      %k: tensor<1x16x128xf32>,
      %v: tensor<1x16x128xf32>) -> tensor<1x8x128xf32> {
    // CHECK: error: default MultiHeadAttention runtime requires fp16 Q/K/V and output
    %out = "onnx.Custom"(%q, %k, %v)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft", num_heads = 8 : si64}
        : (tensor<1x8x128xf32>, tensor<1x16x128xf32>,
           tensor<1x16x128xf32>) -> tensor<1x8x128xf32>
    return %out : tensor<1x8x128xf32>
  }
}

// -----

module {
  func.func @main_graph(
      %q: tensor<1x8x128xf16>,
      %k: tensor<1x16x128xf16>,
      %v: tensor<1x16x128xf16>)
      -> (tensor<1x8x128xf16>, tensor<1x8x16x16xf16>) {
    // CHECK: error: default MultiHeadAttention runtime supports only the primary output; present_key, present_value, and qk are unsupported
    %out:2 = "onnx.Custom"(%q, %k, %v)
        <{function_name = "MultiHeadAttention"}>
        {domain_name = "com.microsoft", num_heads = 8 : si64}
        : (tensor<1x8x128xf16>, tensor<1x16x128xf16>,
           tensor<1x16x128xf16>)
        -> (tensor<1x8x128xf16>, tensor<1x8x16x16xf16>)
    return %out#0, %out#1
        : tensor<1x8x128xf16>, tensor<1x8x16x16xf16>
  }
}

// CHECK-NOT: tensor.empty
// CHECK-NOT: hip.multi_head_attention
