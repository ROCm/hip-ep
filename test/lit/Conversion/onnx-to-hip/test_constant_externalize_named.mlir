// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: onnx_node_name flows into externalized global symbol name.
//
// Verifies that finalizeExternalizedConstant() incorporates a sanitized
// onnx_node_name into the memref.global name.  The sanitizer replaces '/'
// with '_', collapses consecutive underscores, and strips leading/trailing
// underscores -- so "/model/MatMul_weights" becomes "model_MatMul_weights".
//===----------------------------------------------------------------------===//

// RUN: mkdir -p %t && hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip='externalize-min-num-elements=4 externalize-output-dir=%t' %s | FileCheck %s

// CHECK: memref.global "private" @hip_ext_constant_model_MatMul_weights_0 : memref<2x4xf32> {alignment = 64 : i64, hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 32 : i64}}
// CHECK: func.func @main_graph(%arg0: !hip.context) -> tensor<2x4xf32>
// CHECK-NEXT: %0 = memref.get_global @hip_ext_constant_model_MatMul_weights_0 : memref<2x4xf32>
// CHECK-NEXT: %1 = bufferization.to_tensor %0 restrict : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-NEXT: return %1 : tensor<2x4xf32>

module {
  func.func @main_graph() -> tensor<2x4xf32> {
    %0 = "onnx.Constant"() {onnx_node_name = "/model/MatMul_weights", value = dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>} : () -> tensor<2x4xf32>
    "onnx.Return"(%0) {onnx_node_name = "/Return"} : (tensor<2x4xf32>) -> ()
  }
}
