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

  // A static trip count in the result type is not the accumulator type. The
  // first chunk is tensor<1x4xf32>; casting that to tensor<2x4xf32> is illegal.
  // The loop accumulates tensor<?x4xf32> and casts once, after the loop.
  func.func @static_leading(%A: tensor<4xf32>) -> tensor<2x4xf32> {
    %M = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> tensor<2x4xf32>
    return %scan : tensor<2x4xf32>
  }

  // cond_init is omitted, so the body yielding false must not stop the loop.
  func.func @omitted_cond(%A: tensor<4xf32>) -> tensor<?x4xf32> {
    %M = "onnx.Constant"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.NoValue"() {value} : () -> none
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Constant"() {value = dense<0> : tensor<i1>} : () -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, none) -> tensor<?x4xf32>
    return %scan : tensor<?x4xf32>
  }

  // M = 0 and a dynamic result type. The per-iteration tensor is tensor<4xf32>,
  // so the empty seed is [0, 4], not [0, 0].
  func.func @zero_trip_inner(%A: tensor<4xf32>) -> tensor<?x?xf32> {
    %M = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<4xf32>) -> tensor<4xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> tensor<?x?xf32>
    return %scan : tensor<?x?xf32>
  }

  // The inner extent is dynamic on both the result and the per-iteration
  // tensor. It comes from the value the body yields, not from a zero.
  func.func @dynamic_inner(%A: tensor<?xf32>) -> tensor<?x?xf32> {
    %M = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %cond_init = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    %scan = "onnx.Loop"(%M, %cond_init) ({
    ^bb0(%iter: tensor<i64>, %cond_in: tensor<i1>):
      %cond_out = "onnx.Identity"(%cond_in) : (tensor<i1>) -> tensor<i1>
      %chunk = "onnx.Identity"(%A) : (tensor<?xf32>) -> tensor<?xf32>
      "onnx.Yield"(%cond_out, %chunk) : (tensor<i1>, tensor<?xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>) -> tensor<?x?xf32>
    return %scan : tensor<?x?xf32>
  }

  // cond_init is a function argument, so it may be the result of a GPU kernel.
  func.func @gpu_cond(%cond: tensor<i1>, %A: tensor<4xf32>) -> tensor<?x4xf32> {
    %M = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %scan = "onnx.Loop"(%M, %cond) ({
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

// CHECK-LABEL: func.func @static_leading
// CHECK: tensor.empty(%{{.*}}) : tensor<?x4xf32>
// CHECK: scf.while
// CHECK-NOT: tensor<1x4xf32> to tensor<2x4xf32>
// CHECK: tensor.cast %{{.*}} : tensor<?x4xf32> to tensor<2x4xf32>

// CHECK-LABEL: func.func @omitted_cond
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: scf.while ({{.*}}%[[TRUE]]
// CHECK: %[[KEEP:.*]] = arith.constant true
// CHECK: scf.yield %{{.*}}, %[[KEEP]],
// CHECK-NOT: hip.readback_scalar

// CHECK-LABEL: func.func @zero_trip_inner
// CHECK: tensor.empty(%{{.*}}) : tensor<?x4xf32>
// CHECK: tensor.cast %{{.*}} : tensor<?x4xf32> to tensor<?x?xf32>

// CHECK-LABEL: func.func @dynamic_inner
// CHECK: %[[DIM:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?xf32>
// CHECK: tensor.empty(%{{.*}}, %[[DIM]]) : tensor<?x?xf32>

// CHECK-LABEL: func.func @gpu_cond
// CHECK: hip.readback_scalar
// CHECK-NOT: tensor.extract
