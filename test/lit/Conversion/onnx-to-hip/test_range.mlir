// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Test 1: int64 range
  // CHECK-LABEL: func.func @main_graph
  func.func @main_graph(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<?xi64> {
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[START:.*]]: tensor<i64>, %[[LIMIT:.*]]: tensor<i64>, %[[DELTA:.*]]: tensor<i64>) -> tensor<?xi64>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[C0]]) : tensor<?xi64>
    // CHECK: hip.range(%[[CTX]]) ins(%[[START]], %[[LIMIT]], %[[DELTA]] : tensor<i64>, tensor<i64>, tensor<i64>) outs(%[[EMPTY]] : tensor<?xi64>)
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
    return %0 : tensor<?xi64>
  }

  // Test 2: float32 range
  // CHECK-LABEL: func.func @test_range_f32
  func.func @test_range_f32(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<?xf32> {
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[START:.*]]: tensor<f32>, %[[LIMIT:.*]]: tensor<f32>, %[[DELTA:.*]]: tensor<f32>) -> tensor<?xf32>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[C0]]) : tensor<?xf32>
    // CHECK: hip.range(%[[CTX]]) ins(%[[START]], %[[LIMIT]], %[[DELTA]] : tensor<f32>, tensor<f32>, tensor<f32>) outs(%[[EMPTY]] : tensor<?xf32>)
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }

  // Test 3: int32 range
  // CHECK-LABEL: func.func @test_range_i32
  func.func @test_range_i32(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?xi32> {
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[START:.*]]: tensor<i32>, %[[LIMIT:.*]]: tensor<i32>, %[[DELTA:.*]]: tensor<i32>) -> tensor<?xi32>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[C0]]) : tensor<?xi32>
    // CHECK: hip.range(%[[CTX]]) ins(%[[START]], %[[LIMIT]], %[[DELTA]] : tensor<i32>, tensor<i32>, tensor<i32>) outs(%[[EMPTY]] : tensor<?xi32>)
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  // Test 4: float64 range
  // CHECK-LABEL: func.func @test_range_f64
  func.func @test_range_f64(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<?xf64> {
    // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[START:.*]]: tensor<f64>, %[[LIMIT:.*]]: tensor<f64>, %[[DELTA:.*]]: tensor<f64>) -> tensor<?xf64>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[EMPTY:.*]] = tensor.empty(%[[C0]]) : tensor<?xf64>
    // CHECK: hip.range(%[[CTX]]) ins(%[[START]], %[[LIMIT]], %[[DELTA]] : tensor<f64>, tensor<f64>, tensor<f64>) outs(%[[EMPTY]] : tensor<?xf64>)
    %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}
