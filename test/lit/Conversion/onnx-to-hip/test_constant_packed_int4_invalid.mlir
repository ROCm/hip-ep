// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// An 8-bit external constant whose backing byte size is neither the full
// element count nor the packed ceil(numel/2) nibble count is malformed: it
// would otherwise lower silently as int8 over a wrong-length buffer. The
// constant lowering rejects it at the marking site.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip \
// RUN:   --split-input-file --verify-diagnostics %s

module {
  func.func @main_graph() -> tensor<8xf32> {
    // 8 logical i8 elements, but size = 5 (neither 8 full nor 4 packed).
    // expected-error @+1 {{8-bit external source byte size 5 matches neither the full element count 8 nor the packed nibble count 4}}
    %w = "onnx.Constant"() {location = "weights.bin", offset = 0 : i64,
                            size = 5 : i64} : () -> tensor<8xi8>
    %s = "onnx.Constant"() {value = dense<0.02> : tensor<f32>} : () -> tensor<f32>
    %z = "onnx.Constant"() {location = "weights.bin", offset = 5 : i64,
                            size = 1 : i64} : () -> tensor<1xi8>
    %dq = "onnx.Custom"(%w, %s, %z) {
      function_name = "DequantizeLinear", domain_name = "com.microsoft",
      onnx_node_name = "DQ_bad"
    } : (tensor<8xi8>, tensor<f32>, tensor<1xi8>) -> tensor<8xf32>
    return %dq : tensor<8xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
