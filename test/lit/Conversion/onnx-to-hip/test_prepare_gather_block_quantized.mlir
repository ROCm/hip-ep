// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify convert-onnx-to-hip legalizes logical INT4 GBQ weights to packed byte
// shapes and annotates unsigned_quant_storage / quantize_axis before lowering.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // Logical INT4 shape: scales[2048,12] * block_size 16 == data[2048,192].
  // External constant metadata only; raw bytes are already packed.

  func.func @main_graph(%indices: tensor<8xi64>) -> tensor<8x192xf16> {
    %data = "onnx.Constant"() {
      location = "weights.bin",
      offset = 0 : i64,
      size = 393216 : i64
    } : () -> tensor<2048x192xui8>
    %scales = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<2048x12xf16>} : () -> tensor<2048x12xf16>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 4 : si64,
      block_size = 16 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized_prepare"
    } : (tensor<2048x192xui8>, tensor<8xi64>, tensor<2048x12xf16>) -> tensor<8x192xf16>
    return %out : tensor<8x192xf16>
  }

  // CHECK-LABEL: func.func @main_graph
  // CHECK: "onnx.Constant"() {location = "weights.bin"
  // CHECK-SAME: } : () -> tensor<2048x96xui8>
  // CHECK: hip.gather_block_quantized
  // CHECK-SAME: unsigned_quant_storage
  // CHECK-SAME: quantize_axis = 1
}
