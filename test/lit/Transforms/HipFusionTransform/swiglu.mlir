// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Gated-MLP decoders lower SwiGLU to three primitive HIP ops:
//
//   s = hip.sigmoid(gate); a = hip.mul(gate, s); y = hip.mul(a, up)
//
// hip-fusion-transform collapses that into one hip.swiglu. Both multiplies
// are commutative, so every operand ordering has to match, and the
// temporaries must be single-use or the fusion would change what the graph
// computes.

// CHECK-LABEL: func.func @swiglu_canonical
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[GATE:.*]]: tensor<8x16xf16>, %[[UP:.*]]: tensor<8x16xf16>)
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK: hip.swiglu(%[[CTX]]) ins(%[[GATE]], %[[UP]] : tensor<8x16xf16>, tensor<8x16xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<8x16xf16>) : tensor<8x16xf16>
func.func @swiglu_canonical(%ctx: !hip.context, %gate: tensor<8x16xf16>,
                            %up: tensor<8x16xf16>) -> tensor<8x16xf16> {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e2 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e2 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y : tensor<8x16xf16>
}

// -----

// Both multiplies commuted relative to the export.
// CHECK-LABEL: func.func @swiglu_commuted
// CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[GATE:.*]]: tensor<4x32xf32>, %[[UP:.*]]: tensor<4x32xf32>)
// CHECK-NOT: hip.mul
// CHECK: hip.swiglu(%[[CTX]]) ins(%[[GATE]], %[[UP]] : tensor<4x32xf32>, tensor<4x32xf32>)
func.func @swiglu_commuted(%ctx: !hip.context, %gate: tensor<4x32xf32>,
                           %up: tensor<4x32xf32>) -> tensor<4x32xf32> {
  %e0 = tensor.empty() : tensor<4x32xf32>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<4x32xf32>)
       outs(%e0 : tensor<4x32xf32>) : tensor<4x32xf32>
  %e1 = tensor.empty() : tensor<4x32xf32>
  %a = hip.mul(%ctx) ins(%s, %gate : tensor<4x32xf32>, tensor<4x32xf32>)
       outs(%e1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  %e2 = tensor.empty() : tensor<4x32xf32>
  %y = hip.mul(%ctx) ins(%up, %a : tensor<4x32xf32>, tensor<4x32xf32>)
       outs(%e2 : tensor<4x32xf32>) -> tensor<4x32xf32>
  return %y : tensor<4x32xf32>
}

// -----

// Dynamic leading dims: the fused op reuses the outer mul's destination.
// CHECK-LABEL: func.func @swiglu_dynamic_bf16
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK: hip.swiglu
func.func @swiglu_dynamic_bf16(%ctx: !hip.context,
                               %gate: tensor<?x?x14336xbf16>,
                               %up: tensor<?x?x14336xbf16>)
    -> tensor<?x?x14336xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %gate, %c0 : tensor<?x?x14336xbf16>
  %d1 = tensor.dim %gate, %c1 : tensor<?x?x14336xbf16>
  %e0 = tensor.empty(%d0, %d1) : tensor<?x?x14336xbf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<?x?x14336xbf16>)
       outs(%e0 : tensor<?x?x14336xbf16>) : tensor<?x?x14336xbf16>
  %e1 = tensor.empty(%d0, %d1) : tensor<?x?x14336xbf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<?x?x14336xbf16>, tensor<?x?x14336xbf16>)
       outs(%e1 : tensor<?x?x14336xbf16>) -> tensor<?x?x14336xbf16>
  %e2 = tensor.empty(%d0, %d1) : tensor<?x?x14336xbf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<?x?x14336xbf16>, tensor<?x?x14336xbf16>)
       outs(%e2 : tensor<?x?x14336xbf16>) -> tensor<?x?x14336xbf16>
  return %y : tensor<?x?x14336xbf16>
}

// -----

// `gate` feeding another consumer does not block the fusion: the fused op
// re-reads it exactly like the primitive chain did.
// CHECK-LABEL: func.func @swiglu_gate_reused
// CHECK: hip.swiglu
// CHECK: hip.add
func.func @swiglu_gate_reused(%ctx: !hip.context, %gate: tensor<8x16xf16>,
                              %up: tensor<8x16xf16>)
    -> (tensor<8x16xf16>, tensor<8x16xf16>) {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e2 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e2 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e3 = tensor.empty() : tensor<8x16xf16>
  %extra = hip.add(%ctx) ins(%gate, %up : tensor<8x16xf16>, tensor<8x16xf16>)
           outs(%e3 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y, %extra : tensor<8x16xf16>, tensor<8x16xf16>
}

// -----

// No gating multiply at all: this is not a SwiGLU chain.
// CHECK-LABEL: func.func @silu_ungated
// CHECK-NOT: hip.swiglu
// CHECK: hip.sigmoid
// CHECK: hip.mul
func.func @silu_ungated(%ctx: !hip.context, %x: tensor<8x16xf16>) -> tensor<8x16xf16> {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%x : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%x, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %a : tensor<8x16xf16>
}

// -----

// The SiLU result has a second reader, so materializing it is still required.
// CHECK-LABEL: func.func @swiglu_blocked_silu_multi_use
// CHECK-NOT: hip.swiglu
// CHECK: hip.sigmoid
// CHECK: hip.mul
func.func @swiglu_blocked_silu_multi_use(%ctx: !hip.context,
                                         %gate: tensor<8x16xf16>,
                                         %up: tensor<8x16xf16>)
    -> (tensor<8x16xf16>, tensor<8x16xf16>) {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e2 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e2 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y, %a : tensor<8x16xf16>, tensor<8x16xf16>
}

// -----

// The sigmoid result has a second reader, so the chain must stay.
// CHECK-LABEL: func.func @swiglu_blocked_sigmoid_multi_use
// CHECK-NOT: hip.swiglu
// CHECK: hip.sigmoid
func.func @swiglu_blocked_sigmoid_multi_use(%ctx: !hip.context,
                                            %gate: tensor<8x16xf16>,
                                            %up: tensor<8x16xf16>)
    -> (tensor<8x16xf16>, tensor<8x16xf16>) {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e2 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e2 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y, %s : tensor<8x16xf16>, tensor<8x16xf16>
}

// -----

// Broadcasting gate/up shapes have no flat-indexed fused form.
// CHECK-LABEL: func.func @swiglu_blocked_broadcast
// CHECK-NOT: hip.swiglu
// CHECK: hip.sigmoid
// CHECK: hip.mul
func.func @swiglu_blocked_broadcast(%ctx: !hip.context, %gate: tensor<8x16xf16>,
                                    %up: tensor<1x16xf16>) -> tensor<8x16xf16> {
  %e0 = tensor.empty() : tensor<8x16xf16>
  %s = hip.sigmoid(%ctx) ins(%gate : tensor<8x16xf16>)
       outs(%e0 : tensor<8x16xf16>) : tensor<8x16xf16>
  %e1 = tensor.empty() : tensor<8x16xf16>
  %a = hip.mul(%ctx) ins(%gate, %s : tensor<8x16xf16>, tensor<8x16xf16>)
       outs(%e1 : tensor<8x16xf16>) -> tensor<8x16xf16>
  %e2 = tensor.empty() : tensor<8x16xf16>
  %y = hip.mul(%ctx) ins(%a, %up : tensor<8x16xf16>, tensor<1x16xf16>)
       outs(%e2 : tensor<8x16xf16>) -> tensor<8x16xf16>
  return %y : tensor<8x16xf16>
}
