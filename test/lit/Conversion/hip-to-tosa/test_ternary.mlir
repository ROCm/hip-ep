// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.where lowers to tosa.select inside a rock.kernel function, so
// rocMLIR can absorb it into a fused kernel, and verify the forms the
// conversion rejects.
//
// hip.where is ternary (cond, x, y) and cannot use BinaryConverter: the
// predicate is i1 while the result is not. Rank equalization is the same
// pairwise EqualizeRanks used for the binary ops, applied to all three
// operands.
//
// FILE LAYOUT:
// Converting cases in the first chunk; each rejection in its own chunk.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @where
// CHECK: tosa.select %arg1, %arg2, %arg3
// CHECK-NOT: hip.where
func.func @where(%ctx: !hip.context, %cond: tensor<2x8xi1>,
                 %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
                 %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x8xi1>, tensor<2x8xf16>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// Size-1 condition relies on TOSA's implicit broadcast after ranks match.
// CHECK-LABEL: func.func @where_size1_cond
// CHECK: tosa.select
// CHECK-NOT: hip.where
func.func @where_size1_cond(%ctx: !hip.context, %cond: tensor<2x1xi1>,
                             %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
                             %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x1xi1>, tensor<2x8xf16>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// A lower-rank true-value is reshaped with leading 1s first.
// CHECK-LABEL: func.func @where_rank_extend
// CHECK: tosa.reshape
// CHECK: tosa.select
// CHECK-NOT: hip.where
func.func @where_rank_extend(%ctx: !hip.context, %cond: tensor<2x8xi1>,
                             %x: tensor<8xf16>, %y: tensor<2x8xf16>,
                             %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x8xi1>, tensor<8xf16>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// The shape hip-fuse-rocmlir actually produces.
// CHECK-LABEL: func.func @where_outlined_kernel
// CHECK: tosa.select
// CHECK-NOT: hip.where
func.func @where_outlined_kernel(%cond: tensor<2x8xi1>, %x: tensor<2x8xf16>,
                                 %y: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x8xi1>, tensor<2x8xf16>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

func.func @dynamic_shape(%ctx: !hip.context, %cond: tensor<?x8xi1>,
                          %x: tensor<?x8xf16>, %y: tensor<?x8xf16>,
                          %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.where'}}
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<?x8xi1>, tensor<?x8xf16>, tensor<?x8xf16>)
                       outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// hip.where requires an i1 condition; an i32 mask is not tosa.select.
func.func @cond_not_i1(%ctx: !hip.context, %cond: tensor<2x8xi32>,
                        %x: tensor<2x8xf16>, %y: tensor<2x8xf16>,
                        %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.where'}}
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x8xi32>, tensor<2x8xf16>, tensor<2x8xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

func.func @incompatible_broadcast(%ctx: !hip.context, %cond: tensor<2x8xi1>,
                                  %x: tensor<2x8xf16>, %y: tensor<4xf16>,
                                  %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.where'}}
  %r = hip.where(%ctx) ins(%cond, %x, %y :
                           tensor<2x8xi1>, tensor<2x8xf16>, tensor<4xf16>)
                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}
