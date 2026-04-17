// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<128xf32>) -> tensor<128xf32> {
    %output = "onnx.Softplus"(%input) : (tensor<128xf32>) -> tensor<128xf32>
    return %output : tensor<128xf32>
  }

  // Dynamic shape test
  func.func @softplus_dynamic(%input: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %output = "onnx.Softplus"(%input) : (tensor<?x?xf16>) -> tensor<?x?xf16>
    return %output : tensor<?x?xf16>
  }

  // 3D tensor test
  func.func @softplus_3d(%input: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16> {
    %output = "onnx.Softplus"(%input) : (tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16>
    return %output : tensor<2x3x4xbf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG0:.*]]: tensor<128xf32>) -> tensor<128xf32>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<128xf32>
// CHECK: %[[RESULT:.*]] = hip.softplus(%[[CTX]]) ins(%[[ARG0]] : tensor<128xf32>) outs(%[[INIT]] : tensor<128xf32>) : tensor<128xf32>
// CHECK: return %[[RESULT]] : tensor<128xf32>

// CHECK-LABEL: func.func @softplus_dynamic
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG0:.*]]: tensor<?x?xf16>) -> tensor<?x?xf16>
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf16>
// CHECK: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf16>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf16>
// CHECK: %[[RESULT:.*]] = hip.softplus(%[[CTX]]) ins(%[[ARG0]] : tensor<?x?xf16>) outs(%[[INIT]] : tensor<?x?xf16>) : tensor<?x?xf16>
// CHECK: return %[[RESULT]] : tensor<?x?xf16>

// CHECK-LABEL: func.func @softplus_3d
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[ARG0:.*]]: tensor<2x3x4xbf16>) -> tensor<2x3x4xbf16>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<2x3x4xbf16>
// CHECK: %[[RESULT:.*]] = hip.softplus(%[[CTX]]) ins(%[[ARG0]] : tensor<2x3x4xbf16>) outs(%[[INIT]] : tensor<2x3x4xbf16>) : tensor<2x3x4xbf16>
// CHECK: return %[[RESULT]] : tensor<2x3x4xbf16>
