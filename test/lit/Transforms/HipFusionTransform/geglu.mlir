// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Gated-MLP decoders with a GELU gate (Gemma) lower GeGLU to two primitive
// HIP ops:
//
//   g = hip.gelu(gate); y = hip.mul(g, up)
//
// hip-fusion-transform collapses that into one hip.geglu that keeps the gelu
// approximation. The multiply is commutative, and the gelu result must be
// single-use or the fusion would change what the graph computes.

// CHECK-LABEL: func.func @geglu_tanh
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[GATE:.*]]: tensor<8x16xf16>, %[[UP:.*]]: tensor<8x16xf16>)
// CHECK-NOT: hip.gelu
// CHECK-NOT: hip.mul
// CHECK: hip.geglu(%[[CTX]]) ins(%[[GATE]], %[[UP]] : tensor<8x16xf16>, tensor<8x16xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<8x16xf16>) {approximate = "tanh"} : tensor<8x16xf16>
func.func @geglu_tanh(%ctx: !hip.context, %gate: tensor<8x16xf16>,
                      %up: tensor<8x16xf16>) -> tensor<8x16xf16> {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %g = hip.gelu(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) {approximate = "tanh"} : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%g, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y : tensor<8x16xf16>
}

// -----

// Exact (erf) gelu, with the multiply commuted relative to the export.
// CHECK-LABEL: func.func @geglu_erf_commuted
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[GATE:.*]]: tensor<4x32xf32>, %[[UP:.*]]: tensor<4x32xf32>)
// CHECK-NOT: hip.gelu
// CHECK: hip.geglu(%[[CTX]]) ins(%[[GATE]], %[[UP]] : tensor<4x32xf32>, tensor<4x32xf32>)
// CHECK-NOT: tanh
func.func @geglu_erf_commuted(%ctx: !hip.context, %gate: tensor<4x32xf32>,
                              %up: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %e0 = tensor.empty() : tensor<4x32xf32>
  %g = hip.gelu(%ctx) ins(%gate : tensor<4x32xf32>)
       outs(%e0 : tensor<4x32xf32>) : tensor<4x32xf32>
  %e1 = tensor.empty() : tensor<4x32xf32>
  %y = hip.mul(%ctx) ins(%up, %g : tensor<4x32xf32>, tensor<4x32xf32>)
       outs(%e1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %y : tensor<4x32xf32>
}

// -----

// Gate and up project one activation, which makes their dynamic axes equal.
// A tensor.dim of the gelu result is retargeted to the gate.
// CHECK-LABEL: func.func @geglu_dynamic_projections
// CHECK-NOT: hip.gelu
// CHECK-NOT: {{hip\.mul\(}}
// CHECK: %[[GATE:.*]] = hip.matmul
// CHECK: tensor.dim %[[GATE]]
// CHECK: hip.geglu(%{{.*}}) ins(%[[GATE]], %{{.*}} : tensor<?x?x10240xf16>, tensor<?x?x10240xf16>)
func.func @geglu_dynamic_projections(%ctx: !hip.context,
                                     %hidden: tensor<?x?x2560xf16>,
                                     %wGate: tensor<2560x10240xf16>,
                                     %wUp: tensor<2560x10240xf16>)
    -> tensor<?x?x10240xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %hidden, %c0 : tensor<?x?x2560xf16>
  %d1 = tensor.dim %hidden, %c1 : tensor<?x?x2560xf16>
  %eg = tensor.empty(%d0, %d1) : tensor<?x?x10240xf16>
  %gate = hip.matmul(%ctx)
      ins(%hidden, %wGate : tensor<?x?x2560xf16>, tensor<2560x10240xf16>)
      outs(%eg : tensor<?x?x10240xf16>) : tensor<?x?x10240xf16>
  %eu = tensor.empty(%d0, %d1) : tensor<?x?x10240xf16>
  %up = hip.matmul(%ctx)
      ins(%hidden, %wUp : tensor<?x?x2560xf16>, tensor<2560x10240xf16>)
      outs(%eu : tensor<?x?x10240xf16>) : tensor<?x?x10240xf16>
  %dg0 = tensor.dim %gate, %c0 : tensor<?x?x10240xf16>
  %dg1 = tensor.dim %gate, %c1 : tensor<?x?x10240xf16>
  %e0 = tensor.empty(%dg0, %dg1) : tensor<?x?x10240xf16>
  %g = hip.gelu(%ctx) ins(%gate : tensor<?x?x10240xf16>)
       outs(%e0 : tensor<?x?x10240xf16>) {approximate = "tanh"}
       : tensor<?x?x10240xf16>
  %d2 = tensor.dim %g, %c0 : tensor<?x?x10240xf16>
  %d3 = tensor.dim %g, %c1 : tensor<?x?x10240xf16>
  %e1 = tensor.empty(%d2, %d3) : tensor<?x?x10240xf16>
  %y = hip.mul(%ctx) ins(%g, %up : tensor<?x?x10240xf16>, tensor<?x?x10240xf16>)
       outs(%e1 : tensor<?x?x10240xf16>) -> tensor<?x?x10240xf16>
  return %y : tensor<?x?x10240xf16>
}

// -----

// The gelu result has another value use, so it must stay materialized.
// CHECK-LABEL: func.func @geglu_blocked_gelu_multi_use
// CHECK-NOT: hip.geglu
// CHECK: hip.gelu
// CHECK: hip.mul
func.func @geglu_blocked_gelu_multi_use(%ctx: !hip.context,
                                        %gate: tensor<8x16xf16>,
                                        %up: tensor<8x16xf16>)
    -> (tensor<8x16xf16>, tensor<8x16xf16>) {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %g = hip.gelu(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) {approximate = "tanh"} : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%g, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y, %g : tensor<8x16xf16>, tensor<8x16xf16>
}

// -----

// Broadcasting gate/up shapes have no flat-indexed fused form.
// CHECK-LABEL: func.func @geglu_blocked_broadcast
// CHECK-NOT: hip.geglu
// CHECK: hip.gelu
// CHECK: hip.mul
func.func @geglu_blocked_broadcast(%ctx: !hip.context, %gate: tensor<8x16xf16>,
                                   %up: tensor<1x16xf16>) -> tensor<8x16xf16> {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %g = hip.gelu(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) {approximate = "tanh"} : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%g, %up : tensor<8x16xf16>, tensor<1x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y : tensor<8x16xf16>
}

// -----

// Two unbound dynamic tensors have the same type and may still differ at
// runtime, including a 1 that hip.mul broadcasts. hip.geglu does not.
// CHECK-LABEL: func.func @geglu_blocked_dynamic_broadcast
// CHECK-NOT: hip.geglu
// CHECK: hip.gelu
// CHECK: hip.mul
func.func @geglu_blocked_dynamic_broadcast(%ctx: !hip.context,
                                           %gate: tensor<?x256xf16>,
                                           %up: tensor<?x256xf16>)
    -> tensor<?x256xf16> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %gate, %c0 : tensor<?x256xf16>
  %e0 = tensor.empty(%d0) : tensor<?x256xf16>
  %g = hip.gelu(%ctx) ins(%gate : tensor<?x256xf16>)
       outs(%e0 : tensor<?x256xf16>) {approximate = "tanh"} : tensor<?x256xf16>
  %e1 = tensor.empty(%d0) : tensor<?x256xf16>
  %y = hip.mul(%ctx) ins(%g, %up : tensor<?x256xf16>, tensor<?x256xf16>)
       outs(%e1 : tensor<?x256xf16>) -> tensor<?x256xf16>
  return %y : tensor<?x256xf16>
}
