// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for hipsr.matmul that exercise its own code:
//   - round-trip with the shape region omitted (the form onnx->hipsr emits)
//   - verifier / parser diagnostics: rank-0 operand, DPS init/result type
//     mismatch, non-shaped operand, empty shape-region block
//
// The shape region is populated by a later pass (not this stage), so
// populateShapeRegion is covered with that pass, not here. Generic
// shape-region structural tests live in shape_region_verify.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Shape region omitted: parses, verifies, and prints back with no
// `shape_region` keyword (the optional region group prints nothing when the
// region has zero blocks). This is the exact form convert-onnx-to-hipsr emits.
// CHECK-LABEL: func.func @matmul_no_shape_region
// CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// CHECK-SAME: outs(%{{.+}} : tensor<?x1024xf16>) -> tensor<?x1024xf16>
// CHECK-NOT: shape_region
func.func @matmul_no_shape_region(%a: tensor<?x4096xf16>,
                                  %b: tensor<4096x1024xf16>,
                                  %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  %0 = hipsr.matmul ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<?x1024xf16>) -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

//===----------------------------------------------------------------------===//
// FAIL cases (compile-time diagnostics a LIT run observes).
//===----------------------------------------------------------------------===//

// Rank-0 (scalar) A: a scalar tensor is a valid ranked tensor, so the operand
// type constraint accepts it, but matmul needs a contraction dim so the
// verifier rejects it.
func.func @matmul_rank0_a(%a: tensor<f16>, %b: tensor<4096x1024xf16>,
                          %init: tensor<1024xf16>) -> tensor<1024xf16> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %0 = hipsr.matmul ins(%a, %b : tensor<f16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<1024xf16>) -> tensor<1024xf16>
  return %0 : tensor<1024xf16>
}

// -----

// Rank-0 (scalar) B: same contract as A, on the second operand.
func.func @matmul_rank0_b(%a: tensor<64x4096xf16>, %b: tensor<f16>,
                          %init: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error@+1 {{operand B must be at least 1-D}}
  %0 = hipsr.matmul ins(%a, %b : tensor<64x4096xf16>, tensor<f16>)
                    outs(%init : tensor<64xf16>) -> tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// DPS init/result type mismatch: the DPS verifier requires init and result
// types to be equal. Here the result batch-0 dim (4) differs from the init (2).
func.func @matmul_init_result_mismatch(%a: tensor<2x3x64x4096xf16>,
                                       %b: tensor<2x3x4096x1024xf16>,
                                       %init: tensor<2x3x64x1024xf16>)
    -> tensor<4x3x64x1024xf16> {
  // expected-error@+1 {{to match type of corresponding result}}
  %0 = hipsr.matmul ins(%a, %b : tensor<2x3x64x4096xf16>, tensor<2x3x4096x1024xf16>)
                    outs(%init : tensor<2x3x64x1024xf16>) -> tensor<4x3x64x1024xf16>
  return %0 : tensor<4x3x64x1024xf16>
}

// -----

// DPS init/result element-type mismatch: init is f16, result is f32. Same DPS
// rule (the full type must match, not just the shape).
func.func @matmul_elem_type_mismatch(%a: tensor<2x64x4096xf16>,
                                     %b: tensor<2x4096x1024xf16>,
                                     %init: tensor<2x64x1024xf16>)
    -> tensor<2x64x1024xf32> {
  // expected-error@+1 {{to match type of corresponding result}}
  %0 = hipsr.matmul ins(%a, %b : tensor<2x64x4096xf16>, tensor<2x4096x1024xf16>)
                    outs(%init : tensor<2x64x1024xf16>) -> tensor<2x64x1024xf32>
  return %0 : tensor<2x64x1024xf32>
}

// -----

// Non-shaped operand: A is a scalar f16, which the operand type constraint
// rejects. The generic form needs an empty region ({}) for the (optional) shape
// region so the region-count check passes and the operand-type check fires.
func.func @matmul_operand_not_shaped(%a: f16, %b: tensor<4096x1024xf16>,
                                     %init: tensor<64x1024xf16>)
    -> tensor<64x1024xf16> {
  // expected-error@+1 {{operand #0 must be ranked tensor or device memref}}
  %0 = "hipsr.matmul"(%a, %b, %init) ({}) : (f16, tensor<4096x1024xf16>, tensor<64x1024xf16>)
      -> tensor<64x1024xf16>
  return %0 : tensor<64x1024xf16>
}

// -----

// A present shape region must not be a lone empty block (an absent region is
// expressed by omitting the region entirely).
func.func @matmul_empty_shape_region(%a: tensor<2x3x64x4096xf16>,
                                     %b: tensor<2x3x4096x1024xf16>,
                                     %init: tensor<2x3x64x1024xf16>)
    -> tensor<2x3x64x1024xf16> {
  // expected-error@+1 {{expects a non-empty block}}
  %0 = hipsr.matmul ins(%a, %b : tensor<2x3x64x4096xf16>, tensor<2x3x4096x1024xf16>)
                    outs(%init : tensor<2x3x64x1024xf16>) -> tensor<2x3x64x1024xf16>
                    shape_region {
  }
  return %0 : tensor<2x3x64x1024xf16>
}
