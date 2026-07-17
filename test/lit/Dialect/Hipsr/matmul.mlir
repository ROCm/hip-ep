// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for hipsr.matmul (ONNX/NumPy matmul semantics):
//   - round-trip with the shape region omitted (the form onnx->hipsr emits)
//   - round-trip with populated shape regions covering the shapes
//     populateShapeRegion handles: 2-D, batched N-D, NumPy batch broadcast, and
//     1-D operand promotion. Regions are hand-written (no pass calls
//     populateShapeRegion yet), so they mirror what it emits.
//   - fail cases the parser/verifier catches: DPS init/result type mismatch,
//     non-shaped operand, empty shape region
//
// The K / broadcast checks are cf.asserts that fire at run time, so their
// failure is covered by the numeric/runtime tests, not here. Generic
// shape-region structural tests live in shape_region_verify.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// Shape region omitted: parses, verifies, and prints back with no
// `shape_region` keyword (the optional region group prints nothing when the
// region has zero blocks). This is the exact form convert-onnx-to-hipsr emits.
// CHECK-LABEL: func.func @matmul_no_shape_region
func.func @matmul_no_shape_region(%a: tensor<?x4096xf16>,
                                  %b: tensor<4096x1024xf16>,
                                  %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<?x4096xf16>, tensor<4096x1024xf16>) outs(%{{.+}} : tensor<?x1024xf16>) -> tensor<?x1024xf16>
  // CHECK-NOT: shape_region
  %0 = hipsr.matmul ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<?x1024xf16>) -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// Populated shape region (2-D): the contraction dim K is checked with
// arith.cmpi eq + cf.assert before the shape is yielded. The assert message
// text is checked so a regression that drops the K check fails.
// CHECK-LABEL: func.func @matmul_shape_region_k_check
func.func @matmul_shape_region_k_check(%a: tensor<?x?xf16>, %b: tensor<?x?xf16>,
                                       %init: tensor<?x?xf16>) -> tensor<?x?xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<?x?xf16>, tensor<?x?xf16>) outs(%{{.+}} : tensor<?x?xf16>) -> tensor<?x?xf16> shape_region {
  // CHECK:   %[[SHA:.+]] = shape.shape_of %{{.+}} : tensor<?x?xf16> -> tensor<2xindex>
  // CHECK:   %[[SHB:.+]] = shape.shape_of %{{.+}} : tensor<?x?xf16> -> tensor<2xindex>
  // CHECK:   %[[EQ:.+]] = arith.cmpi eq, %{{.+}}, %{{.+}} : index
  // CHECK:   cf.assert %[[EQ]], "hipsr.matmul: A and B contraction dim (K) must be equal"
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
                    outs(%init : tensor<?x?xf16>) -> tensor<?x?xf16> shape_region {
    %shA = shape.shape_of %a : tensor<?x?xf16> -> tensor<2xindex>
    %shB = shape.shape_of %b : tensor<?x?xf16> -> tensor<2xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ka = shape.get_extent %shA, %c1 : tensor<2xindex>, index -> index
    %kb = shape.get_extent %shB, %c0 : tensor<2xindex>, index -> index
    %eq = arith.cmpi eq, %ka, %kb : index
    cf.assert %eq, "hipsr.matmul: A and B contraction dim (K) must be equal"
    %m = shape.get_extent %shA, %c0 : tensor<2xindex>, index -> index
    %n = shape.get_extent %shB, %c1 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%m, %n) : [f16]
  }
  return %0 : tensor<?x?xf16>
}

// -----

// Populated shape region (batched 3-D x 3-D), equal-batch case: K check plus an
// equality batch-dim check. (When both batch dims are known non-1, broadcast
// degenerates to equality; this hand-written region uses the plain equality
// form.)
// CHECK-LABEL: func.func @matmul_shape_region_batch_check
func.func @matmul_shape_region_batch_check(%a: tensor<?x?x?xf16>,
                                           %b: tensor<?x?x?xf16>,
                                           %init: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  // CHECK: cf.assert %{{.+}}, "hipsr.matmul: A and B contraction dim (K) must be equal"
  // CHECK: cf.assert %{{.+}}, "hipsr.matmul: batch dims of A and B must be equal"
  // CHECK: hipsr.shape_yield (%{{.+}}, %{{.+}}, %{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<?x?x?xf16>, tensor<?x?x?xf16>)
                    outs(%init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16> shape_region {
    %shA = shape.shape_of %a : tensor<?x?x?xf16> -> tensor<3xindex>
    %shB = shape.shape_of %b : tensor<?x?x?xf16> -> tensor<3xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ka = shape.get_extent %shA, %c2 : tensor<3xindex>, index -> index
    %kb = shape.get_extent %shB, %c1 : tensor<3xindex>, index -> index
    %eqk = arith.cmpi eq, %ka, %kb : index
    cf.assert %eqk, "hipsr.matmul: A and B contraction dim (K) must be equal"
    %ba = shape.get_extent %shA, %c0 : tensor<3xindex>, index -> index
    %bb = shape.get_extent %shB, %c0 : tensor<3xindex>, index -> index
    %eqb = arith.cmpi eq, %ba, %bb : index
    cf.assert %eqb, "hipsr.matmul: batch dims of A and B must be equal"
    %m = shape.get_extent %shA, %c1 : tensor<3xindex>, index -> index
    %n = shape.get_extent %shB, %c2 : tensor<3xindex>, index -> index
    hipsr.shape_yield (%ba, %m, %n) : [f16]
  }
  return %0 : tensor<?x?x?xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Multi-dimensional PASS cases (parse + verify + round-trip).
//===----------------------------------------------------------------------===//

// 3-D x 2-D broadcast weight: batch + M from A, N from B's last dim. Region
// omitted, as the conversion emits it.
// CHECK-LABEL: func.func @matmul_3d_by_2d_broadcast
func.func @matmul_3d_by_2d_broadcast(%a: tensor<2x?x4096xf16>,
                                     %b: tensor<4096x1024xf16>,
                                     %init: tensor<2x?x1024xf16>)
    -> tensor<2x?x1024xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<2x?x4096xf16>, tensor<4096x1024xf16>) outs(%{{.+}} : tensor<2x?x1024xf16>) -> tensor<2x?x1024xf16>
  // CHECK-NOT: shape_region
  %0 = hipsr.matmul ins(%a, %b : tensor<2x?x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<2x?x1024xf16>) -> tensor<2x?x1024xf16>
  return %0 : tensor<2x?x1024xf16>
}

// -----

// 4-D x 4-D, equal-batch case: two batch dims, so the region checks K plus both
// batch dims and yields (batch0, batch1, M, N). Uses the equality form (see
// matmul_broadcast_batch for the general NumPy-broadcast form).
// CHECK-LABEL: func.func @matmul_4d_batched
func.func @matmul_4d_batched(%a: tensor<?x?x?x?xf16>, %b: tensor<?x?x?x?xf16>,
                             %init: tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) outs(%{{.+}} : tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16> shape_region {
  // CHECK:   cf.assert %{{.+}}, "hipsr.matmul: A and B contraction dim (K) must be equal"
  // Two batch dims -> two batch-equality asserts.
  // CHECK:   cf.assert %{{.+}}, "hipsr.matmul: batch dims of A and B must be equal"
  // CHECK:   cf.assert %{{.+}}, "hipsr.matmul: batch dims of A and B must be equal"
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>)
                    outs(%init : tensor<?x?x?x?xf16>) -> tensor<?x?x?x?xf16>
                    shape_region {
    %shA = shape.shape_of %a : tensor<?x?x?x?xf16> -> tensor<4xindex>
    %shB = shape.shape_of %b : tensor<?x?x?x?xf16> -> tensor<4xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    // K: A dim 3 == B dim 2.
    %ka = shape.get_extent %shA, %c3 : tensor<4xindex>, index -> index
    %kb = shape.get_extent %shB, %c2 : tensor<4xindex>, index -> index
    %eqk = arith.cmpi eq, %ka, %kb : index
    cf.assert %eqk, "hipsr.matmul: A and B contraction dim (K) must be equal"
    // Batch dim 0.
    %b0a = shape.get_extent %shA, %c0 : tensor<4xindex>, index -> index
    %b0b = shape.get_extent %shB, %c0 : tensor<4xindex>, index -> index
    %eqb0 = arith.cmpi eq, %b0a, %b0b : index
    cf.assert %eqb0, "hipsr.matmul: batch dims of A and B must be equal"
    // Batch dim 1.
    %b1a = shape.get_extent %shA, %c1 : tensor<4xindex>, index -> index
    %b1b = shape.get_extent %shB, %c1 : tensor<4xindex>, index -> index
    %eqb1 = arith.cmpi eq, %b1a, %b1b : index
    cf.assert %eqb1, "hipsr.matmul: batch dims of A and B must be equal"
    // Output shape: (batch0, batch1, M, N).
    %m = shape.get_extent %shA, %c2 : tensor<4xindex>, index -> index
    %n = shape.get_extent %shB, %c3 : tensor<4xindex>, index -> index
    hipsr.shape_yield (%b0a, %b1a, %m, %n) : [f16]
  }
  return %0 : tensor<?x?x?x?xf16>
}

// -----

// NumPy-broadcast batch dim (ONNX/NumPy matmul): A batch dim is 1, B's is
// dynamic, so the result takes B's. The region folds the batch dim with
// arith.cmpi/arith.select and asserts it is broadcastable, matching what
// populateShapeRegion emits.
// CHECK-LABEL: func.func @matmul_broadcast_batch
func.func @matmul_broadcast_batch(%a: tensor<1x?x?xf16>, %b: tensor<?x?x?xf16>,
                                  %init: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
  // CHECK: cf.assert %{{.+}}, "hipsr.matmul: A and B contraction dim (K) must be equal"
  // CHECK: cf.assert %{{.+}}, "hipsr.matmul: batch dims of A and B must be broadcastable"
  // CHECK: %[[BC:.+]] = arith.select %{{.+}}, %{{.+}}, %{{.+}} : index
  // CHECK: hipsr.shape_yield (%[[BC]], %{{.+}}, %{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<1x?x?xf16>, tensor<?x?x?xf16>)
                    outs(%init : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                    shape_region {
    %shA = shape.shape_of %a : tensor<1x?x?xf16> -> tensor<3xindex>
    %shB = shape.shape_of %b : tensor<?x?x?xf16> -> tensor<3xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ka = shape.get_extent %shA, %c2 : tensor<3xindex>, index -> index
    %kb = shape.get_extent %shB, %c1 : tensor<3xindex>, index -> index
    %eqk = arith.cmpi eq, %ka, %kb : index
    cf.assert %eqk, "hipsr.matmul: A and B contraction dim (K) must be equal"
    // Broadcast batch dim 0: da==1 ? db : (db==1 ? da : (assert da==db; da)).
    %da = shape.get_extent %shA, %c0 : tensor<3xindex>, index -> index
    %db = shape.get_extent %shB, %c0 : tensor<3xindex>, index -> index
    %one = arith.constant 1 : index
    %aIs1 = arith.cmpi eq, %da, %one : index
    %bIs1 = arith.cmpi eq, %db, %one : index
    %eqd = arith.cmpi eq, %da, %db : index
    %c01 = arith.ori %eqd, %aIs1 : i1
    %compat = arith.ori %c01, %bIs1 : i1
    cf.assert %compat, "hipsr.matmul: batch dims of A and B must be broadcastable"
    %bc = arith.select %aIs1, %db, %da : index
    %m = shape.get_extent %shA, %c1 : tensor<3xindex>, index -> index
    %n = shape.get_extent %shB, %c2 : tensor<3xindex>, index -> index
    hipsr.shape_yield (%bc, %m, %n) : [f16]
  }
  return %0 : tensor<?x?x?xf16>
}

// -----

// 1-D B (ONNX/NumPy matmul): B (K) is promoted to (K,1) for the multiply and the
// trailing 1 is stripped, so (M,K) @ (K) -> (M). K is A's last dim == B's sole
// dim; the yielded shape is just (M).
// CHECK-LABEL: func.func @matmul_1d_rhs
func.func @matmul_1d_rhs(%a: tensor<?x4096xf16>, %b: tensor<4096xf16>,
                         %init: tensor<?xf16>) -> tensor<?xf16> {
  // CHECK: cf.assert %{{.+}}, "hipsr.matmul: A and B contraction dim (K) must be equal"
  // CHECK: hipsr.shape_yield (%{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<?x4096xf16>, tensor<4096xf16>)
                    outs(%init : tensor<?xf16>) -> tensor<?xf16> shape_region {
    %shA = shape.shape_of %a : tensor<?x4096xf16> -> tensor<2xindex>
    %shB = shape.shape_of %b : tensor<4096xf16> -> tensor<1xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ka = shape.get_extent %shA, %c1 : tensor<2xindex>, index -> index
    %kb = shape.get_extent %shB, %c0 : tensor<1xindex>, index -> index
    %eqk = arith.cmpi eq, %ka, %kb : index
    cf.assert %eqk, "hipsr.matmul: A and B contraction dim (K) must be equal"
    // Only M survives (N came from B's promoted, then stripped, unit dim).
    %m = shape.get_extent %shA, %c0 : tensor<2xindex>, index -> index
    hipsr.shape_yield (%m) : [f16]
  }
  return %0 : tensor<?xf16>
}

// -----

// Fully static 3-D batched matmul with a populated region: parses, verifies,
// and round-trips with the region intact.
// CHECK-LABEL: func.func @matmul_static_batched
func.func @matmul_static_batched(%a: tensor<8x64x4096xf16>,
                                 %b: tensor<8x4096x1024xf16>,
                                 %init: tensor<8x64x1024xf16>)
    -> tensor<8x64x1024xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<8x64x4096xf16>, tensor<8x4096x1024xf16>) outs(%{{.+}} : tensor<8x64x1024xf16>) -> tensor<8x64x1024xf16> shape_region {
  // CHECK:   cf.assert %{{.+}}, "hipsr.matmul: A and B contraction dim (K) must be equal"
  // CHECK:   cf.assert %{{.+}}, "hipsr.matmul: batch dims of A and B must be equal"
  // CHECK:   hipsr.shape_yield (%{{.+}}, %{{.+}}, %{{.+}}) : [f16]
  %0 = hipsr.matmul ins(%a, %b : tensor<8x64x4096xf16>, tensor<8x4096x1024xf16>)
                    outs(%init : tensor<8x64x1024xf16>) -> tensor<8x64x1024xf16>
                    shape_region {
    %shA = shape.shape_of %a : tensor<8x64x4096xf16> -> tensor<3xindex>
    %shB = shape.shape_of %b : tensor<8x4096x1024xf16> -> tensor<3xindex>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ka = shape.get_extent %shA, %c2 : tensor<3xindex>, index -> index
    %kb = shape.get_extent %shB, %c1 : tensor<3xindex>, index -> index
    %eqk = arith.cmpi eq, %ka, %kb : index
    cf.assert %eqk, "hipsr.matmul: A and B contraction dim (K) must be equal"
    %ba = shape.get_extent %shA, %c0 : tensor<3xindex>, index -> index
    %bb = shape.get_extent %shB, %c0 : tensor<3xindex>, index -> index
    %eqb = arith.cmpi eq, %ba, %bb : index
    cf.assert %eqb, "hipsr.matmul: batch dims of A and B must be equal"
    %m = shape.get_extent %shA, %c1 : tensor<3xindex>, index -> index
    %n = shape.get_extent %shB, %c2 : tensor<3xindex>, index -> index
    hipsr.shape_yield (%ba, %m, %n) : [f16]
  }
  return %0 : tensor<8x64x1024xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Multi-dimensional FAIL cases (compile-time diagnostics a LIT run observes).
//===----------------------------------------------------------------------===//

// DPS init/result type mismatch (4-D): the DPS verifier requires init and
// result types to be equal. Here the result batch-0 dim (4) differs from the
// init (2).
func.func @matmul_4d_init_result_mismatch(%a: tensor<2x3x64x4096xf16>,
                                          %b: tensor<2x3x4096x1024xf16>,
                                          %init: tensor<2x3x64x1024xf16>)
    -> tensor<4x3x64x1024xf16> {
  // expected-error@+1 {{to match type of corresponding result}}
  %0 = hipsr.matmul ins(%a, %b : tensor<2x3x64x4096xf16>, tensor<2x3x4096x1024xf16>)
                    outs(%init : tensor<2x3x64x1024xf16>) -> tensor<4x3x64x1024xf16>
  return %0 : tensor<4x3x64x1024xf16>
}

// -----

// DPS init/result element-type mismatch (3-D): init is f16, result is f32.
// Same DPS rule (the full type must match, not just the shape).
func.func @matmul_3d_elem_type_mismatch(%a: tensor<2x64x4096xf16>,
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
func.func @matmul_4d_empty_shape_region(%a: tensor<2x3x64x4096xf16>,
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

// -----

// Rank-0 (scalar) operand: a scalar tensor is a valid ranked tensor, so the
// operand type constraint accepts it, but matmul needs a contraction dim so the
// verifier rejects it.
func.func @matmul_rank0_operand(%a: tensor<f16>, %b: tensor<4096x1024xf16>,
                                %init: tensor<1024xf16>) -> tensor<1024xf16> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %0 = hipsr.matmul ins(%a, %b : tensor<f16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<1024xf16>) -> tensor<1024xf16>
  return %0 : tensor<1024xf16>
}
