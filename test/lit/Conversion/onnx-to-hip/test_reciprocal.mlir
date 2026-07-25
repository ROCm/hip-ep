// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  func.func @test_reciprocal(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Reciprocal"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_reciprocal
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<128xf32>) -> tensor<128xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[RESULT:.*]] = hip.reciprocal(%[[CTX]]) ins(%[[ARG]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>) : tensor<128xf32>
  // CHECK: return %[[RESULT]] : tensor<128xf32>

  func.func @test_reciprocal_dynamic(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x?xf16>) -> tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
  // CHECK-LABEL: func.func @test_reciprocal_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf16>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
  // CHECK: %[[RESULT:.*]] = hip.reciprocal(%[[CTX]]) ins(%[[ARG]] : tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
  // CHECK: return %[[RESULT]] : tensor<?x?xf16>

  func.func @test_reciprocal_multidim(%arg0: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16> {
    %0 = "onnx.Reciprocal"(%arg0) : (tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16>
    return %0 : tensor<2x3x4xbf16>
  }
  // CHECK-LABEL: func.func @test_reciprocal_multidim
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x3x4xbf16>
  // CHECK: %[[RESULT:.*]] = hip.reciprocal(%[[CTX]]) ins(%[[ARG]] : tensor<2x3x4xbf16>) outs(%[[INIT]] : tensor<2x3x4xbf16>) : tensor<2x3x4xbf16>
  // CHECK: return %[[RESULT]] : tensor<2x3x4xbf16>
}
