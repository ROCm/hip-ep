// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

module {
  func.func private @loop_body(
      %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
      %value: tensor<4xf32>, %step: tensor<4xf32>) -> tensor<4xf32> {
    return %value : tensor<4xf32>
  }

  // CHECK-LABEL: func.func @counted_loop
  // CHECK: arith.index_cast
  // CHECK: tensor.from_elements
  // CHECK: tosa.while_loop
  // CHECK: tosa.greater
  // CHECK: tosa.logical_and
  // CHECK: tosa.yield
  // CHECK: do {
  // CHECK: tosa.add
  // CHECK: tosa.yield
  // CHECK-NOT: hip.loop
  func.func @counted_loop(
      %ctx: !hip.context, %max_trip_count: index, %cond: i1,
      %value: tensor<4xf32>, %step: tensor<4xf32>) -> tensor<4xf32>
      attributes {rock.kernel} {
    %r = hip.loop(%ctx, %max_trip_count, %cond)
        iter_args(%value : tensor<4xf32>)
        captures(%step : tensor<4xf32>)
        -> (tensor<4xf32>)
        body @loop_body
        {num_loop_carried = 1 : i32, cond_is_passthrough}
    return %r : tensor<4xf32>
  }
}
