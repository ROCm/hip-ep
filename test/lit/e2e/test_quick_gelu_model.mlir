// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// com.microsoft QuickGelu through the full pipeline, reusing hip.swish.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x77x768xf16>) -> tensor<1x77x768xf16> {
    %y = "onnx.Custom"(%arg0) {
      function_name = "QuickGelu",
      domain_name = "com.microsoft",
      alpha = 1.702 : f32
    } : (tensor<1x77x768xf16>) -> tensor<1x77x768xf16>
    return %y : tensor<1x77x768xf16>
  }
}

// CHECK-NOT: onnx.Custom
// CHECK-NOT: hip.swish
// CHECK: llvm.call @wrap_swish
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
