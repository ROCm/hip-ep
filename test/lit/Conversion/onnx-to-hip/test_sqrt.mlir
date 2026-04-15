// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  func.func @test_sqrt(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %0 = "onnx.Sqrt"(%arg0) : (tensor<128xf32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
  // CHECK-LABEL: func.func @test_sqrt
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<128xf32>) -> tensor<128xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
  // CHECK: %[[RESULT:.*]] = hip.sqrt(%[[CTX]]) ins(%[[ARG]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>) : tensor<128xf32>
  // CHECK: return %[[RESULT]] : tensor<128xf32>

  func.func @test_sqrt_f16(%arg0: tensor<1x128x512xf16>) -> tensor<1x128x512xf16> {
    %0 = "onnx.Sqrt"(%arg0) : (tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
    return %0 : tensor<1x128x512xf16>
  }
  // CHECK-LABEL: func.func @test_sqrt_f16
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<1x128x512xf16>) -> tensor<1x128x512xf16>
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<1x128x512xf16>
  // CHECK: %[[RESULT:.*]] = hip.sqrt(%[[CTX]]) ins(%[[ARG]] : tensor<1x128x512xf16>) outs(%[[INIT]] : tensor<1x128x512xf16>) : tensor<1x128x512xf16>
  // CHECK: return %[[RESULT]] : tensor<1x128x512xf16>

  func.func @test_sqrt_dynamic(%arg0: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %0 = "onnx.Sqrt"(%arg0) : (tensor<?x?xbf16>) -> tensor<?x?xbf16>
    return %0 : tensor<?x?xbf16>
  }
  // CHECK-LABEL: func.func @test_sqrt_dynamic
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG:.*]]: tensor<?x?xbf16>) -> tensor<?x?xbf16>
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xbf16>
  // CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xbf16>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xbf16>
  // CHECK: %[[RESULT:.*]] = hip.sqrt(%[[CTX]]) ins(%[[ARG]] : tensor<?x?xbf16>) outs(%[[INIT]] : tensor<?x?xbf16>) : tensor<?x?xbf16>
  // CHECK: return %[[RESULT]] : tensor<?x?xbf16>

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
