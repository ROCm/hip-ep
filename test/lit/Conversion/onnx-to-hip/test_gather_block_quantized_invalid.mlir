// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Explicit si8 and unsigned_quant_storage are contradictory. Reject before
// destination construction rather than silently changing runtime signedness.
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s 2>&1 | FileCheck %s

module {
  func.func @main_graph(
      %data: tensor<64x512xsi8>,
      %indices: tensor<4xi32>,
      %scales: tensor<2x512xf32>) -> tensor<64x4xf32> {
    // CHECK: error: GBQ `unsigned_quant_storage` conflicts with explicitly signed data
    %result = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 8 : si64,
      block_size = 32 : si64,
      gather_axis = 1 : si64,
      quantize_axis = 0 : si64,
      unsigned_quant_storage
    } : (tensor<64x512xsi8>, tensor<4xi32>, tensor<2x512xf32>) -> tensor<64x4xf32>
    return %result : tensor<64x4xf32>
  }
}

// CHECK-NOT: tensor.empty
// CHECK-NOT: hip.gather_block_quantized
