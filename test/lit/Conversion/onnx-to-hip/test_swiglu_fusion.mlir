// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

// Gated-MLP decoders export SwiGLU as three primitive elementwise nodes:
//
//   s = Sigmoid(gate); a = Mul(gate, s); y = Mul(a, up)
//
// SwiGluConversion collapses that into one hip.swiglu. Both multiplies are
// commutative, so every operand ordering has to match, and the temporaries
// must be single-use or the fusion would change what the graph computes.

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Canonical operand order, as exported.
  func.func @swiglu_canonical(%gate: tensor<8x16xf16>, %up: tensor<8x16xf16>)
      -> tensor<8x16xf16> {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %y : tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_canonical
  // CHECK-SAME: (%[[CTX:.*]]: !hip.context, %[[GATE:.*]]: tensor<8x16xf16>, %[[UP:.*]]: tensor<8x16xf16>)
  // CHECK-NOT: onnx.
  // CHECK-NOT: hip.sigmoid
  // CHECK-NOT: hip.mul
  // CHECK: hip.swiglu(%[[CTX]]) ins(%[[GATE]], %[[UP]] : tensor<8x16xf16>, tensor<8x16xf16>)
  // CHECK-SAME: : tensor<8x16xf16>

  // Both multiplies commuted relative to the export.
  func.func @swiglu_commuted(%gate: tensor<4x32xf32>, %up: tensor<4x32xf32>)
      -> tensor<4x32xf32> {
    %s = "onnx.Sigmoid"(%gate) : (tensor<4x32xf32>) -> tensor<4x32xf32>
    %a = "onnx.Mul"(%s, %gate)
        : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    %y = "onnx.Mul"(%up, %a)
        : (tensor<4x32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %y : tensor<4x32xf32>
  }

  // CHECK-LABEL: func.func @swiglu_commuted
  // CHECK-SAME: (%[[CTX2:.*]]: !hip.context, %[[GATE2:.*]]: tensor<4x32xf32>, %[[UP2:.*]]: tensor<4x32xf32>)
  // CHECK-NOT: hip.mul
  // CHECK: hip.swiglu(%[[CTX2]]) ins(%[[GATE2]], %[[UP2]] : tensor<4x32xf32>, tensor<4x32xf32>)

  // Dynamic leading dim: the DPS init takes its extent from gate.
  func.func @swiglu_dynamic_bf16(%gate: tensor<?x?x14336xbf16>,
                                 %up: tensor<?x?x14336xbf16>)
      -> tensor<?x?x14336xbf16> {
    %s = "onnx.Sigmoid"(%gate) : (tensor<?x?x14336xbf16>) -> tensor<?x?x14336xbf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<?x?x14336xbf16>, tensor<?x?x14336xbf16>) -> tensor<?x?x14336xbf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<?x?x14336xbf16>, tensor<?x?x14336xbf16>) -> tensor<?x?x14336xbf16>
    return %y : tensor<?x?x14336xbf16>
  }

  // CHECK-LABEL: func.func @swiglu_dynamic_bf16
  // CHECK: tensor.dim
  // CHECK: hip.swiglu

  // `gate` feeding another consumer does not block the fusion: the fused op
  // re-reads it exactly like the primitive chain did.
  func.func @swiglu_gate_reused(%gate: tensor<8x16xf16>, %up: tensor<8x16xf16>)
      -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %extra = "onnx.Add"(%gate, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %y, %extra : tensor<8x16xf16>, tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_gate_reused
  // CHECK: hip.swiglu
  // CHECK: hip.add

  // No gating multiply at all: this is not a SwiGLU chain.
  func.func @silu_ungated(%x: tensor<8x16xf16>) -> tensor<8x16xf16> {
    %s = "onnx.Sigmoid"(%x) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%x, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %a : tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @silu_ungated
  // CHECK-NOT: hip.swiglu
  // CHECK: hip.sigmoid
  // CHECK: hip.mul

  // The SiLU result has a second reader, so materializing it is still
  // required and the gated fusion must not fire.
  func.func @swiglu_blocked_silu_multi_use(%gate: tensor<8x16xf16>,
                                           %up: tensor<8x16xf16>)
      -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %y, %a : tensor<8x16xf16>, tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_blocked_silu_multi_use
  // CHECK-NOT: hip.swiglu
  // CHECK: hip.sigmoid
  // CHECK: hip.mul

  // The sigmoid result has a second reader, so neither fusion is valid.
  func.func @swiglu_blocked_sigmoid_multi_use(%gate: tensor<8x16xf16>,
                                              %up: tensor<8x16xf16>)
      -> (tensor<8x16xf16>, tensor<8x16xf16>) {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    return %y, %s : tensor<8x16xf16>, tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_blocked_sigmoid_multi_use
  // CHECK-NOT: hip.swiglu
  // CHECK-NOT: hip.swish
  // CHECK: hip.sigmoid

  // Broadcasting gate/up shapes have no flat-indexed fused form, so the chain
  // stays on the generic elementwise path.
  func.func @swiglu_blocked_broadcast(%gate: tensor<8x16xf16>,
                                      %up: tensor<1x16xf16>)
      -> tensor<8x16xf16> {
    %s = "onnx.Sigmoid"(%gate) : (tensor<8x16xf16>) -> tensor<8x16xf16>
    %a = "onnx.Mul"(%gate, %s)
        : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %y = "onnx.Mul"(%a, %up)
        : (tensor<8x16xf16>, tensor<1x16xf16>) -> tensor<8x16xf16>
    return %y : tensor<8x16xf16>
  }

  // CHECK-LABEL: func.func @swiglu_blocked_broadcast
  // CHECK-NOT: hip.swiglu
  // CHECK: hip.sigmoid
  // CHECK: hip.mul
}
