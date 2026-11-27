// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -o %t
// RUN: FileCheck %s --check-prefix=CUSTOM --input-file=%t
// RUN: hip-mlir-opt %t -o %t.roundtrip
// RUN: diff %t %t.roundtrip
// RUN: hip-mlir-opt %s --mlir-print-op-generic -o %t.generic
// RUN: FileCheck %s --check-prefix=GENERIC --input-file=%t.generic
// RUN: hip-mlir-opt %t.generic -o %t.from-generic
// RUN: diff %t %t.from-generic
// RUN: hip-mlir-opt %t --mlir-print-op-generic -o %t.roundtrip-generic
// RUN: diff %t.generic %t.roundtrip-generic

#weight = #ggml.layout<ne = [4, 3, 1, 1], nb = [2, 8, 24, 24], offset = 0>
#dynamic = #ggml.layout<ne = [8, 1*token_count, 1, 1], nb = [4, 32, 32*token_count, 32*token_count], offset = 0>

// CUSTOM-LABEL: func.func @projection
// CUSTOM: ggml.mul_mat %{{[^, ]+}}, %{{[^ ]+}}
// CUSTOM-SAME: ggml.value_id = 7 : i64
// CUSTOM-SAME: precision = #ggml.compute_precision<f32>
// CUSTOM: ggml.rms_norm %{{.*}}
// CUSTOM: ggml.add %{{.*}}, %{{.*}}
// CUSTOM: ggml.reshape %{{.*}}
func.func @projection(%weight: tensor<1x1x3x4xf16, #weight>,
    %input: tensor<1x1x2x4xf32>, %bias: tensor<1x1x1x3xf32>)
    -> tensor<1x2x1x3xf32> {
  %product = ggml.mul_mat %weight, %input {
    precision = #ggml.compute_precision<f32>, hint = 0 : i64,
    ggml.source_name = "projection", ggml.value_id = 7 : i64
  } : (tensor<1x1x3x4xf16, #weight>, tensor<1x1x2x4xf32>) -> tensor<1x1x2x3xf32>
  %norm = ggml.rms_norm %product {epsilon = 1.0e-5 : f32}
    : (tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
  %sum = ggml.add %norm, %bias
    : (tensor<1x1x2x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x2x3xf32>
  %view = ggml.reshape %sum : (tensor<1x1x2x3xf32>) -> tensor<1x2x1x3xf32>
  return %view : tensor<1x2x1x3xf32>
}

// CUSTOM-LABEL: func.func @dynamic
// CUSTOM: ggml.rms_norm %{{[^ ]+}}
// CUSTOM-SAME: tensor<1x1x?x8xf32, #{{.*}}>
func.func @dynamic(%input: tensor<1x1x?x8xf32, #dynamic>)
    -> tensor<1x1x?x8xf32, #dynamic> {
  %result = ggml.rms_norm %input {epsilon = 1.0e-5 : f32}
    : (tensor<1x1x?x8xf32, #dynamic>) -> tensor<1x1x?x8xf32, #dynamic>
  return %result : tensor<1x1x?x8xf32, #dynamic>
}

// A single optional operand is recovered from the operand list. Default-valued
// properties and explicitly supplied properties survive both spellings.
// CUSTOM-LABEL: func.func @glu
// CUSTOM: ggml.glu %{{.*}} {
// CUSTOM-SAME: split = false
// CUSTOM: ggml.glu %{{.*}}, %{{.*}} {
// CUSTOM: ggml.glu %{{.*}}, %{{.*}} {
// CUSTOM-SAME: glu_kind = #ggml.glu_kind<swiglu_oai>
func.func @glu(%packed: tensor<1x1x2x8xf32>, %up: tensor<1x1x2x4xf32>) {
  %a = ggml.glu %packed {glu_kind = #ggml.glu_kind<swiglu>, split = false}
    : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x4xf32>
  %b = ggml.glu %a, %up {glu_kind = #ggml.glu_kind<swiglu>}
    : (tensor<1x1x2x4xf32>, tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xf32>
  %c = ggml.glu %a, %up {glu_kind = #ggml.glu_kind<swiglu_oai>,
    alpha = 1.702 : f32, limit = 7.0 : f32}
    : (tensor<1x1x2x4xf32>, tensor<1x1x2x4xf32>) -> tensor<1x1x2x4xf32>
  return
}

// CUSTOM-LABEL: func.func @attention
// CUSTOM: ggml.flash_attention_ext %{{[^,]+}}, %{{[^,]+}}, %{{[^ ]+}} {
// CUSTOM: ggml.flash_attention_ext %{{.*}} mask %{{[^ ]+}} {
// CUSTOM: ggml.flash_attention_ext %{{.*}} sinks %{{[^ ]+}} {
// CUSTOM: ggml.flash_attention_ext %{{.*}} mask %{{[^ ]+}} sinks %{{[^ ]+}} {
// GENERIC: "ggml.flash_attention_ext"
// GENERIC-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>
// GENERIC: "ggml.flash_attention_ext"
// GENERIC-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>
// GENERIC: "ggml.flash_attention_ext"
// GENERIC-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>
// GENERIC: "ggml.flash_attention_ext"
// GENERIC-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>
func.func @attention(%q: tensor<1x2x3x8xf32>, %k: tensor<1x1x4x8xf16>,
    %v: tensor<1x1x4x8xf16>, %mask: tensor<1x1x3x4xf16>, %sinks: tensor<2xf32>) {
  %a = ggml.flash_attention_ext %q, %k, %v {
    scale = 0.5 : f32, max_bias = 0.0 : f32, logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<default>
  } : (tensor<1x2x3x8xf32>, tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>) -> tensor<1x3x2x8xf32>
  %b = ggml.flash_attention_ext %q, %k, %v mask %mask {
    scale = 0.5 : f32, max_bias = 0.0 : f32, logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<default>
  } : (tensor<1x2x3x8xf32>, tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>, tensor<1x1x3x4xf16>) -> tensor<1x3x2x8xf32>
  %c = ggml.flash_attention_ext %q, %k, %v sinks %sinks {
    scale = 0.5 : f32, max_bias = 0.0 : f32, logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<default>
  } : (tensor<1x2x3x8xf32>, tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>, tensor<2xf32>) -> tensor<1x3x2x8xf32>
  %d = ggml.flash_attention_ext %q, %k, %v mask %mask sinks %sinks {
    scale = 0.5 : f32, max_bias = 0.0 : f32, logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<default>
  } : (tensor<1x2x3x8xf32>, tensor<1x1x4x8xf16>, tensor<1x1x4x8xf16>, tensor<1x1x3x4xf16>, tensor<2xf32>) -> tensor<1x3x2x8xf32>
  return
}
