// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX SkipSimplifiedLayerNormalization (com.microsoft domain) is
// correctly lowered to hip.skip_rms_norm operation in tensor-first mode.
//
// This test validates:
// - Custom operation lowering (onnx.Custom -> hip.skip_rms_norm)
// - Domain check: only "com.microsoft" domain is converted
// - Skip RMS normalization attributes: epsilon, axis, stash_type
// - f16 element type support
// - 3D tensor (1x128x4096) normalization with skip connection
// - Proper !hip.context threading through operations
// - Tensor-first DPS: tensor.empty() used as output init
// - Fused operation: residual = x + skip, output = RMSNorm(residual) * scale
//
// Model: Llama-3.1-8B skip RMS layer normalization
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%x: tensor<1x128x4096xf16>,
                         %skip: tensor<1x128x4096xf16>,
                         %scale: tensor<4096xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {
    %output, %residual = "onnx.Custom"(%x, %skip, %scale) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32,
      axis = -1 : si64,
      stash_type = 1 : si64
    } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
    return %output, %residual : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<1x128x4096xf16>, %[[SKIP:.*]]: tensor<1x128x4096xf16>, %[[SCALE:.*]]: tensor<4096xf16>) -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: hip.skip_rms_norm(%[[CTX]]) ins(%[[X]], %[[SKIP]], %[[SCALE]] : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>, tensor<4096xf16>) outs({{.*}}, {{.*}} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {axis = -1 : i64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : i64} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
// CHECK-NOT: onnx.Custom
