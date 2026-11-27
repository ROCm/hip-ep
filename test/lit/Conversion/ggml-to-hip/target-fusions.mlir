// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/add-rms-q8.mlir --convert-ggml-to-hip | FileCheck %s --check-prefix=ADD-RMS
// RUN: hip-mlir-opt %t/packed-swiglu.mlir --convert-ggml-to-hip | FileCheck %s --check-prefix=PACKED-SWIGLU
// RUN: hip-mlir-opt %t/flash.mlir --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s --check-prefix=FLASH
// RUN: hip-mlir-opt %t/flash-cache-update.mlir --hip-add-context-arg --convert-ggml-to-hip | FileCheck %s --check-prefix=FLASH-UPDATE

// ADD-RMS-LABEL: func.func @add_rms_q8(
// ADD-RMS: %[[FUSED:.*]]:2 = hip.skip_rms_norm_q8_1
// ADD-RMS-NOT: hip.add
// ADD-RMS-NOT: hip.rms_norm
// ADD-RMS-NOT: hip.quantize_q8_1

//--- add-rms-q8.mlir


func.func @add_rms_q8(
    %ctx: !hip.context, %lhs: tensor<1x1x4x512xf32>,
    %rhs: tensor<1x1x4x512xf32>, %scale: tensor<512xf32>,
    %residual_init: tensor<1x1x4x512xf32>,
    %normalized_init: tensor<1x1x4x512xf32>,
    %q8_init: tensor<1x1x4x576xui8>)
    -> (tensor<1x1x4x512xf32>, tensor<1x1x4x576xui8>) {
  %residual = hip.add(%ctx)
    ins(%lhs, %rhs : tensor<1x1x4x512xf32>, tensor<1x1x4x512xf32>)
    outs(%residual_init : tensor<1x1x4x512xf32>)
      -> tensor<1x1x4x512xf32>
  %normalized = hip.rms_norm(%ctx)
    ins(%residual, %scale : tensor<1x1x4x512xf32>, tensor<512xf32>)
    outs(%normalized_init : tensor<1x1x4x512xf32>)
    {axis = -1 : i64, epsilon = 0.00001 : f32, stash_type = 1 : i64}
      : tensor<1x1x4x512xf32>
  %q8 = hip.quantize_q8_1(%ctx)
    ins(%normalized : tensor<1x1x4x512xf32>)
    outs(%q8_init : tensor<1x1x4x576xui8>)
    {K = 512 : i64} : tensor<1x1x4x576xui8>
  return %residual, %q8 : tensor<1x1x4x512xf32>, tensor<1x1x4x576xui8>
}

// PACKED-SWIGLU-LABEL: func.func @packed_swiglu(
// PACKED-SWIGLU: hip.kquant_swiglu_q8_1
// PACKED-SWIGLU-SAME: gate_format = "q4_k"
// PACKED-SWIGLU-NOT: hip.kquant_matmul_q8_1
// PACKED-SWIGLU-NOT: hip.swiglu

//--- packed-swiglu.mlir


func.func @packed_swiglu(
    %ctx: !hip.context, %q8: tensor<1x1x1x576xui8>,
    %gate_weight: tensor<864xui8>, %up_weight: tensor<864xui8>,
    %gate_init: tensor<1x1x1x3xf32>, %up_init: tensor<1x1x1x3xf32>,
    %output_init: tensor<1x1x1x3xf32>) -> tensor<1x1x1x3xf32> {
  %gate = hip.kquant_matmul_q8_1(%ctx)
    ins(%q8, %gate_weight : tensor<1x1x1x576xui8>, tensor<864xui8>)
    outs(%gate_init : tensor<1x1x1x3xf32>)
    {K = 512 : i64, N = 3 : i64, format = "q4_k"}
      : tensor<1x1x1x3xf32>
  %up = hip.kquant_matmul_q8_1(%ctx)
    ins(%q8, %up_weight : tensor<1x1x1x576xui8>, tensor<864xui8>)
    outs(%up_init : tensor<1x1x1x3xf32>)
    {K = 512 : i64, N = 3 : i64, format = "q4_k"}
      : tensor<1x1x1x3xf32>
  %result = hip.swiglu(%ctx)
    ins(%gate, %up : tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>)
    outs(%output_init : tensor<1x1x1x3xf32>) : tensor<1x1x1x3xf32>
  return %result : tensor<1x1x1x3xf32>
}


// FLASH-LABEL: func.func @flash(
// FLASH: hip.token_major_gqa
// FLASH-NOT: hip.transpose
// FLASH-NOT: = ggml.

// A cache produced by an indexed update does not hand attention its
// destination rows: llama.cpp reuses freed cells, so the rows a step writes
// do not bound the rows its queries see. The mask does.
// FLASH-UPDATE-LABEL: func.func @flash_cache_update(
// FLASH-UPDATE: hip.scatter_nd
// FLASH-UPDATE: hip.token_major_gqa
// FLASH-UPDATE-NOT: rows(
// FLASH-UPDATE-SAME: outs(
// FLASH-UPDATE-NOT: = ggml.

//--- flash.mlir
// The heads are permuted rather than copied, so the operands the fusion sees
// are strided views of dense buffers. Their extents and strides say so; nothing
// records the conclusion.
#query = #ggml.layout<ne = [64, 4, 3, 1], nb = [4, 256, 1024, 3072], offset = 0>
#key = #ggml.layout<ne = [64, 2, 16, 1], nb = [2, 128, 256, 4096], offset = 0>
#query_h = #ggml.layout<ne = [64, 3, 4, 1], nb = [4, 1024, 256, 3072], offset = 0>
#key_h = #ggml.layout<ne = [64, 16, 2, 1], nb = [2, 256, 128, 4096], offset = 0>

func.func @flash(
    %query: tensor<1x3x4x64xf32, #query> ,
    %key: tensor<1x16x2x64xf16, #key> ,
    %value: tensor<1x16x2x64xf16, #key> ,
    %mask: tensor<1x1x3x16xf16>) -> tensor<1x3x4x64xf32, #query> {
  %query_h = "ggml.permute"(%query) <{axes = [0, 2, 1, 3]}>  : (tensor<1x3x4x64xf32, #query>) -> tensor<1x4x3x64xf32, #query_h>
  %key_h = "ggml.permute"(%key) <{axes = [0, 2, 1, 3]}>  : (tensor<1x16x2x64xf16, #key>) -> tensor<1x2x16x64xf16, #key_h>
  %value_h = "ggml.permute"(%value) <{axes = [0, 2, 1, 3]}>  : (tensor<1x16x2x64xf16, #key>) -> tensor<1x2x16x64xf16, #key_h>
  %result = "ggml.flash_attention_ext"(%query_h, %key_h, %value_h, %mask) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
    scale = 0.125 : f32, max_bias = 0.0 : f32,
    logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<f32>
  }>  :
    (tensor<1x4x3x64xf32, #query_h>, tensor<1x2x16x64xf16, #key_h>, tensor<1x2x16x64xf16, #key_h>, tensor<1x1x3x16xf16>)
      -> tensor<1x3x4x64xf32, #query>
  return %result : tensor<1x3x4x64xf32, #query>
}

//--- flash-cache-update.mlir
func.func @flash_cache_update(
    %query: tensor<1x3x4x64xf32> {ggml.contiguous = true},
    %new_key: tensor<1x1x3x128xf32> {
      ggml.contiguous = true, ggml.storage_id = 0 : i64},
    %indices: tensor<1x1x1x3xi64> {
      ggml.contiguous = true, ggml.storage_id = 1 : i64},
    %key_cache: tensor<1x1x16x128xf16> {
      ggml.contiguous = true, ggml.storage_id = 2 : i64},
    %value: tensor<1x16x2x64xf16> {ggml.contiguous = true},
    %mask: tensor<1x1x3x16xf16>) -> tensor<1x3x4x64xf32> {
  %updated = "ggml.set_rows"(%new_key, %indices, %key_cache) {
    ggml.contiguous = true, ggml.output_storage_id = 2 : i64
  } : (tensor<1x1x3x128xf32>, tensor<1x1x1x3xi64>, tensor<1x1x16x128xf16>)
      -> tensor<1x1x16x128xf16>
  %key = "ggml.reshape"(%updated) {
    ggml.contiguous = true, ggml.nb = [2, 128, 256, 4096],
    ggml.ne = [64, 2, 16, 1]
  } : (tensor<1x1x16x128xf16>) -> tensor<1x16x2x64xf16>
  %query_h = "ggml.permute"(%query) <{axes = [0, 2, 1, 3]}> {
    ggml.contiguous = false
  } : (tensor<1x3x4x64xf32>) -> tensor<1x4x3x64xf32>
  %key_h = "ggml.permute"(%key) <{axes = [0, 2, 1, 3]}> {
    ggml.contiguous = false
  } : (tensor<1x16x2x64xf16>) -> tensor<1x2x16x64xf16>
  %value_h = "ggml.permute"(%value) <{axes = [0, 2, 1, 3]}> {
    ggml.contiguous = false
  } : (tensor<1x16x2x64xf16>) -> tensor<1x2x16x64xf16>
  %result = "ggml.flash_attention_ext"(%query_h, %key_h, %value_h, %mask) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
    scale = 0.125 : f32, max_bias = 0.0 : f32,
    logit_softcap = 0.0 : f32,
    precision = #ggml.compute_precision<f32>
  }> {ggml.contiguous = true} :
    (tensor<1x4x3x64xf32>, tensor<1x2x16x64xf16>,
     tensor<1x2x16x64xf16>, tensor<1x1x3x16xf16>)
      -> tensor<1x3x4x64xf32>
  return %result : tensor<1x3x4x64xf32>
}
