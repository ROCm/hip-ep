// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Ops whose rank the importer could not derive must stay unconverted instead
// of reaching a converter that reads the rank through an unchecked
// `mlir::cast<RankedTensorType>` (an access violation in release builds).
// The leftover onnx.* op is later rejected by bufferization.
//
// Canonical site: BertSquad (opset 8), where ONNX shape inference leaves the
// result of a Reshape with a runtime-computed shape unranked, and every op
// downstream of it inherits `tensor<*xT>`.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @unranked_operand_and_result
  // CHECK: %[[R:.*]] = "onnx.Reshape"
  // CHECK: %[[G:.*]] = "onnx.Gather"(%{{.*}}, %[[R]])
  // CHECK-SAME: -> tensor<*xf32>
  // CHECK: "onnx.Cast"(%[[G]])
  // CHECK-SAME: -> tensor<*xi64>
  func.func @unranked_operand_and_result(%x: tensor<?x256xi64>,
                                         %shape: tensor<3xi64>,
                                         %table: tensor<2x2xf32>)
      -> tensor<*xi64> {
    %r = "onnx.Reshape"(%x, %shape)
        : (tensor<?x256xi64>, tensor<3xi64>) -> tensor<*xi64>
    %g = "onnx.Gather"(%table, %r) {axis = 0 : si64}
        : (tensor<2x2xf32>, tensor<*xi64>) -> tensor<*xf32>
    %c = "onnx.Cast"(%g) {to = i64} : (tensor<*xf32>) -> tensor<*xi64>
    return %c : tensor<*xi64>
  }

  // CHECK-LABEL: func.func @cast_unranked_result
  // CHECK-NOT: hip.cast
  // CHECK: "onnx.Cast"
  // CHECK-SAME: -> tensor<*xf32>
  func.func @cast_unranked_result(%x: tensor<?x256xi64>) -> tensor<*xf32> {
    %c = "onnx.Cast"(%x) {to = f32} : (tensor<?x256xi64>) -> tensor<*xf32>
    return %c : tensor<*xf32>
  }
}
