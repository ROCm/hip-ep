// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-loop-outline --split-input-file --verify-diagnostics %s

module {
  func.func @rank_conflict(%seed: tensor<4xf32>) -> tensor<2x2xf32> {
    %m = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    // expected-error @below {{'onnx.Loop' op loop carrier #0 has contradictory ranks 1 and 2}}
    %r = "onnx.Loop"(%m, %c, %seed) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>, %current: tensor<4xf32>):
      "onnx.Yield"(%cond, %current) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4xf32>) -> tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }
}

// -----

module {
  func.func @element_conflict(%seed: tensor<4xf32>) -> tensor<4xf16> {
    %m = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    // expected-error @below {{'onnx.Loop' op loop carrier #0 has contradictory element types 'f32' and 'f16'}}
    %r = "onnx.Loop"(%m, %c, %seed) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>, %current: tensor<4xf32>):
      "onnx.Yield"(%cond, %current) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4xf32>) -> tensor<4xf16>
    return %r : tensor<4xf16>
  }
}

// -----

module {
  func.func @static_conflict(%seed: tensor<4xf32>) -> tensor<8xf32> {
    %m = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %c = "onnx.Constant"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
    // expected-error @below {{'onnx.Loop' op loop carrier #0 has contradictory static extents 4 and 8 at dimension 0}}
    %r = "onnx.Loop"(%m, %c, %seed) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>, %current: tensor<4xf32>):
      "onnx.Yield"(%cond, %current) : (tensor<i1>, tensor<4xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<4xf32>) -> tensor<8xf32>
    return %r : tensor<8xf32>
  }
}
