// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// com.microsoft Gelu through the full pipeline, reusing hip.gelu (erf).

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x768xf16>) -> tensor<1x128x768xf16> {
    %gelu = "onnx.Custom"(%arg0) {
      function_name = "Gelu",
      domain_name = "com.microsoft"
    } : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    return %gelu : tensor<1x128x768xf16>
  }
}

// CHECK-NOT: onnx.Custom
// CHECK-NOT: hip.gelu
// CHECK: llvm.call @wrap_gelu
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
