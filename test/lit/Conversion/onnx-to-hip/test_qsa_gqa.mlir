// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFER
// CHECK: "hip.qsa_gqa"
// CHECK-NOT: "onnx.Custom"
// BUFFER: "hip.qsa_gqa"
// BUFFER-SAME: memref<640x4096xbf16
// BUFFER-NOT: tensor<
module {
  func.func @main_graph(%q: tensor<1x?x4096xf16>,
      %kv: tensor<1x32x?x128xf16>, %len: tensor<1xi32>, %cap: tensor<i32>,
      %w: tensor<640x4096xbf16>, %n: tensor<128xbf16>,
      %rope: tensor<65536x64xf32>) ->
      (tensor<1x?x4096xf16>, tensor<1x32x?x128xf16>, tensor<1x32x?x128xf16>) {
    %r:3 = "onnx.Custom"(%q, %q, %q, %kv, %kv, %len, %cap, %q, %w, %n, %n, %rope, %rope)
      {function_name = "QsaGroupQueryAttention", domain_name = "com.amd"} :
      (tensor<1x?x4096xf16>, tensor<1x?x4096xf16>, tensor<1x?x4096xf16>,
       tensor<1x32x?x128xf16>, tensor<1x32x?x128xf16>, tensor<1xi32>, tensor<i32>,
       tensor<1x?x4096xf16>, tensor<640x4096xbf16>, tensor<128xbf16>, tensor<128xbf16>,
       tensor<65536x64xf32>, tensor<65536x64xf32>) ->
      (tensor<1x?x4096xf16>, tensor<1x32x?x128xf16>, tensor<1x32x?x128xf16>)
    return %r#0, %r#1, %r#2 : tensor<1x?x4096xf16>, tensor<1x32x?x128xf16>, tensor<1x32x?x128xf16>
  }
}
