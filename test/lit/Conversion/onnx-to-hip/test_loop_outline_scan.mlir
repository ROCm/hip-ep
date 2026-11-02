// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-loop-outline %s | FileCheck %s

// Scan outputs are not carried by hip.loop. A loop that yields one value per
// iteration lowers to scf.while: iteration 0 takes the unsqueezed chunk, and
// later iterations concatenate on axis 0 so the leading dim is the trip count
// that actually ran.
module {
  func.func @main_graph(%A: tensor<4xf32>) -> tensor<?x4xf32> {
    %M = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> tensor<?x4xf32>
    return %scan : tensor<?x4xf32>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.loop
// CHECK-DAG: %[[M:.*]] = arith.constant 2 : index
// CHECK: %[[SEED:.*]] = tensor.empty(%{{.*}}) : tensor<?x4xf32>
// CHECK: scf.while
// The condition is in the before-region, ahead of the scan concat.
// CHECK: scf.condition
// CHECK: onnx.Unsqueeze
// CHECK: onnx.Concat
// CHECK-NOT: hip.loop
