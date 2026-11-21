// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFER
// CHECK: "hip.qwen_qsa_gqa"
// CHECK-NOT: "onnx.Custom"
// BUFFER: "hip.qwen_qsa_gqa"
// BUFFER-SAME: memref<640x5120xbf16
// BUFFER-NOT: tensor<
module {
  func.func @main_graph(%q: tensor<?x?x6144xf16>, %k: tensor<?x?x1024xf16>,
      %hidden: tensor<?x?x5120xf16>, %kv: tensor<?x4x?x?xf16>,
      %len: tensor<?xi32>, %cap: tensor<i32>, %w: tensor<640x5120xbf16>,
      %n: tensor<128xbf16>, %rope: tensor<262144x32xf32>,
      %pos: tensor<3x?x?xi64>) ->
      (tensor<?x?x6144xf16>, tensor<?x4x?x?xf16>, tensor<?x4x?x?xf16>) {
    %r:3 = "onnx.Custom"(%q, %k, %k, %kv, %kv, %len, %cap, %hidden, %w, %n, %n, %n, %rope, %rope, %pos)
      {function_name = "QwenQsaGroupQueryAttention", domain_name = "com.amd"} :
      (tensor<?x?x6144xf16>, tensor<?x?x1024xf16>, tensor<?x?x1024xf16>,
       tensor<?x4x?x?xf16>, tensor<?x4x?x?xf16>, tensor<?xi32>, tensor<i32>,
       tensor<?x?x5120xf16>, tensor<640x5120xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<128xbf16>,
       tensor<262144x32xf32>, tensor<262144x32xf32>, tensor<3x?x?xi64>) ->
      (tensor<?x?x6144xf16>, tensor<?x4x?x?xf16>, tensor<?x4x?x?xf16>)
    return %r#0, %r#1, %r#2 : tensor<?x?x6144xf16>, tensor<?x4x?x?xf16>, tensor<?x4x?x?xf16>
  }
}
