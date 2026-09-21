// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

module {
  func.func private @if_then(%ctx: !hip.context,
                             %x: tensor<4xf32>) -> tensor<4xf32> {
    return %x : tensor<4xf32>
  }

  func.func private @if_else(%ctx: !hip.context,
                             %x: tensor<4xf32>) -> tensor<4xf32> {
    %c = arith.constant dense<1.000000e+00> : tensor<4xf32>
    return %c : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @if_tensor
  // CHECK: %[[COND:.*]] = tensor.from_elements %arg1 : tensor<1xi1>
  // CHECK: tosa.cond_if %[[COND]]
  // CHECK: tosa.yield
  // CHECK: } else {
  // CHECK: arith.constant
  // CHECK: tosa.yield
  // CHECK-NOT: hip.if
  func.func @if_tensor(%ctx: !hip.context, %cond: i1, %x: tensor<4xf32>,
                       %init: tensor<4xf32>) -> tensor<4xf32>
      attributes {rock.kernel} {
    %r = hip.if(%ctx, %cond)
        outs(%init : tensor<4xf32>)
        captures(%x : tensor<4xf32>)
        -> (tensor<4xf32>)
        then @if_then else @if_else
        {num_outputs = 1 : i32}
    return %r : tensor<4xf32>
  }
}
