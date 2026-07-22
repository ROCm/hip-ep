// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Tests for hipsr.matmul. Default prefix: empty-region round-trip and verifier
// diagnostics. POPULATE prefix: -hipsr-populate-shape-region. Generic
// shape-region structural rules live in shape_region_verify.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics -hipsr-populate-shape-region %s | FileCheck %s --check-prefix=POPULATE

// Omitted shape region round-trips with no `shape_region` keyword (CHECK-NEXT
// return proves nothing was printed for it).
// CHECK-LABEL: func.func @matmul_no_shape_region
// CHECK:      hipsr.matmul(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// CHECK-SAME:   outs(%{{.+}} : tensor<?x1024xf16>) : tensor<?x1024xf16>
// CHECK-NEXT: return
func.func @matmul_no_shape_region(%ctx: !hipsr.context,
                                  %a: tensor<?x4096xf16>,
                                  %b: tensor<4096x1024xf16>,
                                  %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

//===----------------------------------------------------------------------===//
// FAIL cases (compile-time diagnostics a LIT run observes).
//===----------------------------------------------------------------------===//

// Rank-0 A: a valid ranked tensor (passes the type constraint) but has no
// contraction dim, so the verifier rejects it.
func.func @matmul_rank0_a(%ctx: !hipsr.context, %a: tensor<f16>,
                          %b: tensor<4096x1024xf16>,
                          %init: tensor<1024xf16>) -> tensor<1024xf16> {
  // expected-error@+1 {{operand A must be at least 1-D}}
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<f16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<1024xf16>) : tensor<1024xf16>
  return %0 : tensor<1024xf16>
}

// -----

// Rank-0 (scalar) B: same contract as A, on the second operand.
func.func @matmul_rank0_b(%ctx: !hipsr.context, %a: tensor<64x4096xf16>,
                          %b: tensor<f16>,
                          %init: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error@+1 {{operand B must be at least 1-D}}
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<64x4096xf16>, tensor<f16>)
                    outs(%init : tensor<64xf16>) : tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// DPS verifier requires equal init/result types; here batch-0 differs (4 vs 2).
func.func @matmul_init_result_mismatch(%ctx: !hipsr.context,
                                       %a: tensor<2x3x64x4096xf16>,
                                       %b: tensor<2x3x4096x1024xf16>,
                                       %init: tensor<2x3x64x1024xf16>)
    -> tensor<4x3x64x1024xf16> {
  // expected-error@+1 {{to match type of corresponding result}}
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<2x3x64x4096xf16>, tensor<2x3x4096x1024xf16>)
                    outs(%init : tensor<2x3x64x1024xf16>) : tensor<4x3x64x1024xf16>
  return %0 : tensor<4x3x64x1024xf16>
}

// -----

// Non-shaped operand: A is a plain f16, rejected by the operand type
// constraint. Generic form with an empty region ({}) so the operand-type check
// is what fires.
func.func @matmul_operand_not_shaped(%ctx: !hipsr.context, %a: f16,
                                     %b: tensor<4096x1024xf16>,
                                     %init: tensor<64x1024xf16>)
    -> tensor<64x1024xf16> {
  // expected-error@+1 {{operand #1 must be ranked tensor or device memref}}
  %0 = "hipsr.matmul"(%ctx, %a, %b, %init) ({}) : (!hipsr.context, f16, tensor<4096x1024xf16>, tensor<64x1024xf16>)
      -> tensor<64x1024xf16>
  return %0 : tensor<64x1024xf16>
}

// -----

//===----------------------------------------------------------------------===//
// Shape-region population (POPULATE RUN line). The 2-D case pins the full
// dataflow; the rest assert only the structure that differs from it.
//===----------------------------------------------------------------------===//

// Canonical 2-D case, checked end-to-end: entry-block args (read via %[[A]]/
// %[[B]], since the region is IsolatedFromAbove), the K-equality guard, then
// the output dims (M from A, N from B) yielded under that assumption.
// POPULATE-LABEL: func.func @matmul_2d
// POPULATE: hipsr.matmul(%{{.+}}) ins(%{{.+}}, %{{.+}} : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// POPULATE-SAME: shape_region {
// POPULATE:   ^bb0(%[[A:.+]]: tensor<?x4096xf16>, %[[B:.+]]: tensor<4096x1024xf16>):
// POPULATE:   %[[SHA:.+]] = shape.shape_of %[[A]]
// POPULATE:   %[[SHB:.+]] = shape.shape_of %[[B]]
// POPULATE:   %[[KA:.+]] = shape.get_extent %[[SHA]]
// POPULATE:   %[[SA:.+]] = shape.from_extents %[[KA]]
// POPULATE:   %[[KB:.+]] = shape.get_extent %[[SHB]]
// POPULATE:   %[[SB:.+]] = shape.from_extents %[[KB]]
// POPULATE:   %[[W:.+]] = shape.cstr_eq %[[SA]], %[[SB]]
// POPULATE:   %[[D:.+]]:2 = shape.assuming %[[W]] -> (index, index) {
// POPULATE:     %[[M:.+]] = shape.get_extent %[[SHA]]
// POPULATE:     %[[N:.+]] = shape.get_extent %[[SHB]]
// POPULATE:     shape.assuming_yield %[[M]], %[[N]] : index, index
// POPULATE:   }
// POPULATE:   hipsr.shape_yield (%[[D]]#0, %[[D]]#1) : [f16]
func.func @matmul_2d(%ctx: !hipsr.context, %a: tensor<?x4096xf16>,
                     %b: tensor<4096x1024xf16>,
                     %init: tensor<?x1024xf16>) -> tensor<?x1024xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<?x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<?x1024xf16>) : tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// 1-D A acts as (1,K); the promoted leading 1 is stripped, so only N survives.
// POPULATE-LABEL: func.func @matmul_1d_a
// POPULATE: shape_region {
// POPULATE:   shape.cstr_eq
// POPULATE:   %[[D:.+]] = shape.assuming %{{.+}} -> (index) {
// POPULATE:     %[[N:.+]] = shape.get_extent
// POPULATE:     shape.assuming_yield %[[N]] : index
// POPULATE:   }
// POPULATE:   hipsr.shape_yield (%[[D]]) : [f16]
func.func @matmul_1d_a(%ctx: !hipsr.context, %a: tensor<4096xf16>,
                       %b: tensor<4096x1024xf16>,
                       %init: tensor<1024xf16>) -> tensor<1024xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<1024xf16>) : tensor<1024xf16>
  return %0 : tensor<1024xf16>
}

// -----

// 1-D B acts as (K,1); the promoted trailing 1 is stripped, so only M survives.
// POPULATE-LABEL: func.func @matmul_1d_b
// POPULATE: shape_region {
// POPULATE:   shape.cstr_eq
// POPULATE:   %[[D:.+]] = shape.assuming %{{.+}} -> (index) {
// POPULATE:     %[[M:.+]] = shape.get_extent
// POPULATE:     shape.assuming_yield %[[M]] : index
// POPULATE:   }
// POPULATE:   hipsr.shape_yield (%[[D]]) : [f16]
func.func @matmul_1d_b(%ctx: !hipsr.context, %a: tensor<64x4096xf16>,
                       %b: tensor<4096xf16>,
                       %init: tensor<64xf16>) -> tensor<64xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<64x4096xf16>, tensor<4096xf16>)
                    outs(%init : tensor<64xf16>) : tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Both 1-D -> scalar: no M or N, so the K guard wraps an empty region and
// shape_yield has no dims.
// POPULATE-LABEL: func.func @matmul_both_1d
// POPULATE: shape_region {
// POPULATE:   shape.cstr_eq
// POPULATE:   shape.assuming %{{.+}} {
// POPULATE:   hipsr.shape_yield () : [f16]
func.func @matmul_both_1d(%ctx: !hipsr.context, %a: tensor<4096xf16>,
                          %b: tensor<4096xf16>,
                          %init: tensor<f16>) -> tensor<f16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<4096xf16>, tensor<4096xf16>)
                    outs(%init : tensor<f16>) : tensor<f16>
  return %0 : tensor<f16>
}

// -----

// Batched: leading dims broadcast (a dim of 1 takes the other side), so A's
// dynamic batch-0 against B's static 1 emits an arith.select.
// POPULATE-LABEL: func.func @matmul_batched
// POPULATE: shape_region {
// POPULATE:   shape.cstr_eq
// POPULATE:   %[[D:.+]]:4 = shape.assuming %{{.+}} -> (index, index, index, index) {
// POPULATE:     arith.select
// POPULATE:     shape.assuming_yield
// POPULATE:   }
// POPULATE:   hipsr.shape_yield (%[[D]]#0, %[[D]]#1, %[[D]]#2, %[[D]]#3) : [f16]
func.func @matmul_batched(%ctx: !hipsr.context, %a: tensor<?x8x64x4096xf16>,
                          %b: tensor<1x8x4096x1024xf16>,
                          %init: tensor<?x8x64x1024xf16>) -> tensor<?x8x64x1024xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<?x8x64x4096xf16>, tensor<1x8x4096x1024xf16>)
                    outs(%init : tensor<?x8x64x1024xf16>) : tensor<?x8x64x1024xf16>
  return %0 : tensor<?x8x64x1024xf16>
}

// -----

// Idempotent: the pass only fills empty regions, so a hand-written region
// survives untouched (no generated shape.shape_of over it).
// POPULATE-LABEL: func.func @matmul_already_populated
// POPULATE: shape_region {
// POPULATE:   %[[C64:.+]] = arith.constant 64 : index
// POPULATE:   %[[C1024:.+]] = arith.constant 1024 : index
// POPULATE:   hipsr.shape_yield (%[[C64]], %[[C1024]]) : [f16]
// POPULATE-NOT: shape.shape_of
func.func @matmul_already_populated(%ctx: !hipsr.context, %a: tensor<64x4096xf16>,
                                    %b: tensor<4096x1024xf16>,
                                    %init: tensor<64x1024xf16>) -> tensor<64x1024xf16> {
  %0 = hipsr.matmul(%ctx) ins(%a, %b : tensor<64x4096xf16>, tensor<4096x1024xf16>)
                    outs(%init : tensor<64x1024xf16>) : tensor<64x1024xf16> shape_region {
  ^bb0(%aarg: tensor<64x4096xf16>, %barg: tensor<4096x1024xf16>):
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    hipsr.shape_yield (%c64, %c1024) : [f16]
  }
  return %0 : tensor<64x1024xf16>
}
