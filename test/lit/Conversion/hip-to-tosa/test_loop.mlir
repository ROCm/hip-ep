// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

module {
  func.func private @loop_body(
      %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
      %value: tensor<4xf32>, %step: tensor<4xf32>) -> tensor<4xf32> {
    %init = tensor.empty() : tensor<4xf32>
    %add = hip.add(%ctx)
        ins(%value, %step : tensor<4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    return %add : tensor<4xf32>
  }

  // The outlined body declares iter as a rank-0 tensor<i64>, so the TOSA
  // loop must carry it at that rank for the clone to stay well typed.
  func.func private @iter_body(
      %ctx: !hip.context, %iter: tensor<i64>, %cond: tensor<i1>,
      %value: tensor<4xf32>, %step: tensor<4xf32>) -> tensor<4xf32> {
    %fi = tensor.empty() : tensor<f32>
    %iterf = hip.cast(%ctx) ins(%iter : tensor<i64>)
        outs(%fi : tensor<f32>) {to = 1 : i64} : tensor<f32>
    %init = tensor.empty() : tensor<4xf32>
    %add = hip.add(%ctx)
        ins(%value, %iterf : tensor<4xf32>, tensor<f32>)
        outs(%init : tensor<4xf32>) -> tensor<4xf32>
    return %add : tensor<4xf32>
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
  // CHECK: tosa.add
  // CHECK: tosa.yield
  // CHECK-NOT: hip.loop
  // CHECK-NOT: hip.add
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

  // CHECK-LABEL: func.func @loop_using_iter
  // CHECK: tosa.while_loop
  // CHECK: ^bb0(%[[ITER:[a-zA-Z0-9_]+]]: tensor<i64>,
  // CHECK: tosa.cast %[[ITER]] : (tensor<i64>) -> tensor<f32>
  // CHECK: tosa.add
  // CHECK-NOT: hip.loop
  // CHECK-NOT: hip.cast
  func.func @loop_using_iter(
      %ctx: !hip.context, %max_trip_count: index, %cond: i1,
      %value: tensor<4xf32>, %step: tensor<4xf32>) -> tensor<4xf32>
      attributes {rock.kernel} {
    %r = hip.loop(%ctx, %max_trip_count, %cond)
        iter_args(%value : tensor<4xf32>)
        captures(%step : tensor<4xf32>)
        -> (tensor<4xf32>)
        body @iter_body
        {num_loop_carried = 1 : i32, cond_is_passthrough}
    return %r : tensor<4xf32>
  }
}
