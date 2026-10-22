// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x1x512xf16>) -> tensor<1x1x512xf16> {
    return %arg0 : tensor<1x1x512xf16>
  }

  func.func @mrope_interleaved(
      %x: tensor<?x?x6144xf16>,
      %pos: tensor<3x?x?xi64>,
      %cos: tensor<262144x32xf16>,
      %sin: tensor<262144x32xf16>
  ) -> tensor<?x?x6144xf16> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 0 : si64,
      is_packed_batching = 0 : si64,
      mrope_layout = 1 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      num_heads = 24 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<?x?x6144xf16>, tensor<3x?x?xi64>, tensor<262144x32xf16>, tensor<262144x32xf16>) -> tensor<?x?x6144xf16>
    return %0 : tensor<?x?x6144xf16>
  }

  // CHECK-LABEL: func.func @mrope_interleaved
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x?x6144xf16>, %[[POS:.*]]: tensor<3x?x?xi64>, %[[COS:.*]]: tensor<262144x32xf16>, %[[SIN:.*]]: tensor<262144x32xf16>)
  // CHECK: %[[OUT:.*]] = tensor.empty(%{{.*}}, %{{.*}}) : tensor<?x?x6144xf16>
  // CHECK: %[[R:.*]] = hip.mrotary_embedding(%[[CTX]]) ins(%[[X]], %[[POS]], %[[COS]], %[[SIN]] : tensor<?x?x6144xf16>, tensor<3x?x?xi64>, tensor<262144x32xf16>, tensor<262144x32xf16>) outs(%[[OUT]] : tensor<?x?x6144xf16>)
  // CHECK-SAME: {mrope_layout = 1 : i64, mrope_section = [11, 11, 10], num_heads = 24 : i64, rotary_embedding_dim = 64 : i64} : tensor<?x?x6144xf16>
  // CHECK: return %[[R]]

  func.func @mrope_full_rotary_scaled(
      %x: tensor<2x7x64xf32>,
      %pos: tensor<3x2x7xi64>,
      %cos: tensor<128x16xf32>,
      %sin: tensor<128x16xf32>
  ) -> tensor<2x7x64xf32> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      mrope_section = [8 : si64, 4 : si64, 4 : si64],
      num_heads = 2 : si64,
      scale = 5.000000e-01 : f32
    } : (tensor<2x7x64xf32>, tensor<3x2x7xi64>, tensor<128x16xf32>, tensor<128x16xf32>) -> tensor<2x7x64xf32>
    return %0 : tensor<2x7x64xf32>
  }

  // CHECK-LABEL: func.func @mrope_full_rotary_scaled
  // CHECK: hip.mrotary_embedding
  // CHECK-SAME: {mrope_section = [8, 4, 4], num_heads = 2 : i64, rotary_embedding_dim = 32 : i64, scale = 5.000000e-01 : f32}

  func.func @mrope_interleaved_attr(
      %x: tensor<1x1x512xf16>,
      %pos: tensor<3x1x1xi64>,
      %cos: tensor<4096x32xf16>,
      %sin: tensor<4096x32xf16>
  ) -> tensor<1x1x512xf16> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 1 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      num_heads = 2 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<1x1x512xf16>, tensor<3x1x1xi64>, tensor<4096x32xf16>, tensor<4096x32xf16>) -> tensor<1x1x512xf16>
    return %0 : tensor<1x1x512xf16>
  }

  // CHECK-LABEL: func.func @mrope_interleaved_attr
  // CHECK-NOT: hip.mrotary_embedding
  // CHECK: "onnx.Custom"
  // CHECK-SAME: interleaved = 1

  func.func @mrope_bad_section(
      %x: tensor<1x1x512xf16>,
      %pos: tensor<3x1x1xi64>,
      %cos: tensor<4096x32xf16>,
      %sin: tensor<4096x32xf16>
  ) -> tensor<1x1x512xf16> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      mrope_section = [11 : si64, 11 : si64, 11 : si64],
      num_heads = 2 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<1x1x512xf16>, tensor<3x1x1xi64>, tensor<4096x32xf16>, tensor<4096x32xf16>) -> tensor<1x1x512xf16>
    return %0 : tensor<1x1x512xf16>
  }

  // CHECK-LABEL: func.func @mrope_bad_section
  // CHECK-NOT: hip.mrotary_embedding
  // CHECK: "onnx.Custom"
  // CHECK-SAME: mrope_section = [11 : si64, 11 : si64, 11 : si64]

  func.func @mrope_bnsh(
      %x: tensor<1x2x1x256xf16>,
      %pos: tensor<3x1x1xi64>,
      %cos: tensor<4096x32xf16>,
      %sin: tensor<4096x32xf16>
  ) -> tensor<1x2x1x256xf16> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      num_heads = 2 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<1x2x1x256xf16>, tensor<3x1x1xi64>, tensor<4096x32xf16>, tensor<4096x32xf16>) -> tensor<1x2x1x256xf16>
    return %0 : tensor<1x2x1x256xf16>
  }

  // CHECK-LABEL: func.func @mrope_bnsh
  // CHECK-NOT: hip.mrotary_embedding
  // CHECK: "onnx.Custom"

  func.func @mrope_dynamic_streams(
      %x: tensor<?x?x512xf16>,
      %pos: tensor<?x?x?xi64>,
      %cos: tensor<4096x32xf16>,
      %sin: tensor<4096x32xf16>
  ) -> tensor<?x?x512xf16> {
    %0 = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding",
      domain_name = "com.microsoft",
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      num_heads = 2 : si64,
      rotary_embedding_dim = 64 : si64
    } : (tensor<?x?x512xf16>, tensor<?x?x?xi64>, tensor<4096x32xf16>, tensor<4096x32xf16>) -> tensor<?x?x512xf16>
    return %0 : tensor<?x?x512xf16>
  }

  // CHECK-LABEL: func.func @mrope_dynamic_streams
  // CHECK-NOT: hip.mrotary_embedding
  // CHECK: "onnx.Custom"
}
