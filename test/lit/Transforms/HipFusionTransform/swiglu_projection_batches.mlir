// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// One activation does not prove equal projection batches. For example, A's
// batch can be 1 while the two weight batches are 1 and 4. The final Mul then
// broadcasts; SwiGLU cannot. Destinations are arguments to isolate this proof
// from allocation. These tests check compiler shapes, not runtime support.
// CHECK-LABEL: func.func @unknown_projection_batches
// CHECK-NOT: hip.swiglu
// CHECK: hip.sigmoid
// CHECK: hip.mul
// CHECK: hip.mul
// CHECK-NOT: hip.swiglu
// CHECK: return
func.func @unknown_projection_batches(
    %ctx: !hip.context, %a: tensor<?x2x4xf32>,
    %wg: tensor<?x4x8xf32>, %wu: tensor<?x4x8xf32>,
    %eg: tensor<?x2x8xf32>, %eu: tensor<?x2x8xf32>,
    %es: tensor<?x2x8xf32>, %ea: tensor<?x2x8xf32>,
    %ey: tensor<?x2x8xf32>) -> tensor<?x2x8xf32> {
  %gate = hip.matmul(%ctx) ins(%a, %wg : tensor<?x2x4xf32>, tensor<?x4x8xf32>)
      outs(%eg : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %up = hip.matmul(%ctx) ins(%a, %wu : tensor<?x2x4xf32>, tensor<?x4x8xf32>)
      outs(%eu : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<?x2x8xf32>)
      outs(%es : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %silu = hip.mul(%ctx) ins(%gate, %s : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ea : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  %y = hip.mul(%ctx) ins(%silu, %up : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ey : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  return %y : tensor<?x2x8xf32>
}

// -----

// Both weight batches are 1. Each projection has A's batch extent.
// CHECK-LABEL: func.func @singleton_weight_batches
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: {{hip\.mul\(}}
// CHECK: hip.swiglu
func.func @singleton_weight_batches(
    %ctx: !hip.context, %a: tensor<?x2x4xf32>,
    %wg: tensor<1x4x8xf32>, %wu: tensor<1x4x8xf32>,
    %eg: tensor<?x2x8xf32>, %eu: tensor<?x2x8xf32>,
    %es: tensor<?x2x8xf32>, %ea: tensor<?x2x8xf32>,
    %ey: tensor<?x2x8xf32>) -> tensor<?x2x8xf32> {
  %gate = hip.matmul(%ctx) ins(%a, %wg : tensor<?x2x4xf32>, tensor<1x4x8xf32>)
      outs(%eg : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %up = hip.matmul(%ctx) ins(%a, %wu : tensor<?x2x4xf32>, tensor<1x4x8xf32>)
      outs(%eu : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<?x2x8xf32>)
      outs(%es : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %silu = hip.mul(%ctx) ins(%gate, %s : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ea : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  %y = hip.mul(%ctx) ins(%silu, %up : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ey : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  return %y : tensor<?x2x8xf32>
}

// -----

// Both activation batches are 1. Each projection has the shared weight batch.
// CHECK-LABEL: func.func @singleton_activation_batches
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: {{hip\.mul\(}}
// CHECK: hip.swiglu
func.func @singleton_activation_batches(
    %ctx: !hip.context, %ag: tensor<1x2x4xf32>, %au: tensor<1x2x4xf32>,
    %w: tensor<?x4x8xf32>,
    %eg: tensor<?x2x8xf32>, %eu: tensor<?x2x8xf32>,
    %es: tensor<?x2x8xf32>, %ea: tensor<?x2x8xf32>,
    %ey: tensor<?x2x8xf32>) -> tensor<?x2x8xf32> {
  %gate = hip.matmul(%ctx) ins(%ag, %w : tensor<1x2x4xf32>, tensor<?x4x8xf32>)
      outs(%eg : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %up = hip.matmul(%ctx) ins(%au, %w : tensor<1x2x4xf32>, tensor<?x4x8xf32>)
      outs(%eu : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<?x2x8xf32>)
      outs(%es : tensor<?x2x8xf32>) : tensor<?x2x8xf32>
  %silu = hip.mul(%ctx) ins(%gate, %s : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ea : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  %y = hip.mul(%ctx) ins(%silu, %up : tensor<?x2x8xf32>, tensor<?x2x8xf32>)
      outs(%ey : tensor<?x2x8xf32>) -> tensor<?x2x8xf32>
  return %y : tensor<?x2x8xf32>
}
