// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.add is lowered 1-1 to tosa.add inside a rock.kernel function, so
// rocMLIR can absorb it into a fused kernel.
//
// This test validates:
// - Same-rank operands map directly to tosa.add
// - Size-1 dimensions rely on TOSA's implicit broadcast
// - The hip context and DPS outs operand are both dropped
// - The pass is a no-op on functions without rock.kernel
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK: tosa.add %arg1, %arg2 : (tensor<2x8xf16>, tensor<2x8xf16>) -> tensor<2x8xf16>
// CHECK-NOT: hip.add
func.func @add(%ctx: !hip.context, %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
               %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// Size-1 dims are TOSA-broadcastable, so this still maps 1-1.
// CHECK-LABEL: func.func @add_broadcast
// CHECK: tosa.add
// CHECK-NOT: hip.add
func.func @add_broadcast(%ctx: !hip.context, %x: tensor<1x128x32xf16>,
                         %y: tensor<1x1x32xf16>, %init: tensor<1x128x32xf16>)
    -> tensor<1x128x32xf16> attributes {rock.kernel} {
  %r = hip.add(%ctx) ins(%x, %y : tensor<1x128x32xf16>, tensor<1x1x32xf16>)
                     outs(%init : tensor<1x128x32xf16>) -> tensor<1x128x32xf16>
  return %r : tensor<1x128x32xf16>
}

// The pass early-returns unless the function is an outlined kernel.
// CHECK-LABEL: func.func @add_not_a_kernel
// CHECK: hip.add
// CHECK-NOT: tosa.add
func.func @add_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                            %y: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> {
  %r = hip.add(%ctx) ins(%x, %y : tensor<2x8xf16>, tensor<2x8xf16>)
                     outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
