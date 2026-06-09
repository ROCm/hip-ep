// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Exercise the PrecisionFreeCast simplifier inside simplify-onnx in isolation.
// Mirrors ORT CPU's InsertCastTransformer / PrecisionFreeCast: promotes a
// fp16 producer (Add / Sum / Gemm / MatMul) feeding `Cast(fp16->fp32)` ->
// fp32-pure consumer to a fp32 producer, eliminating the fp16 saturation
// that would otherwise feed LayerNormalization an Inf -- which then poisons
// var = E[x^2] - E[x]^2 = Inf - Inf = NaN.
//
// Cases covered:
//   1. Canonical block residual: Add(fp16) -> Cast(fp32) -> LN
//      Promoted to Add(fp32) -> LN with per-input fp16->fp32 casts.
//   2. Cascading residual chain: two Adds -> Cast -> LN.
//      The second anchor's Add input scan peeks through the bridge produced
//      by the first anchor and uses the upstream fp32 Add directly --
//      single top-down walk fully promotes the chain to fp32.
//   3. Original Add has another fp16 user (i.e. not just the Cast->LN).
//      A Cast(fp32->fp16) bridge is inserted so the other user still compiles.
//   4. Negative: Add not consumed by a Cast->fp32_consumer (Mul fp16 only).
//      No rewrite -- the Add stays in fp16.
//   5. Positive (PFC v2): Cast(fp16->fp32) feeds an fp32-pure non-LN consumer
//      (Mul fp32). PFC v2 still fires because the consumer is fp32-pure --
//      this models the canonical vision pattern where Gemm(fp16) feeds an
//      already-promoted fp32 Add (not LN).
//   6. Negative: Already-fp32 Add (the simplifier is a no-op).
//   7. PFC v2: Gemm(fp16) -> Cast(fp32) -> fp32 consumer. Promotes Gemm to
//      fp32 result with per-input casts AND preserves alpha/beta/transA/
//      transB attributes.
//   8. PFC v2: MatMul(fp16) -> Cast(fp32) -> fp32 consumer. Promotes MatMul
//      to fp32 result with per-input casts.
//   9. PFC v2: chained anchor — Gemm(fp16) -> Cast -> fp32 Add (was already
//      fp32 from a PFC v1 promotion in the residual chain). The pass loop
//      catches the Gemm anchor on a subsequent iteration. Models block 21
//      vision encoder: linear_85 Gemm output cast feeds fp32 add_3461.
//
// The pass is pure ONNX-dialect, so the RUN line invokes it standalone (no
// --hip-add-context-arg). End-to-end coverage through convert-onnx-to-hip is
// covered by the existing test_layer_normalization.mlir on the post-promote IR.
// ============================================================================

// RUN: hip-mlir-opt --simplify-onnx %s | FileCheck %s

module {
  // -------------------------------------------------------------------------
  // Case 1: canonical Add(fp16,fp16) -> Cast(fp32) -> LN
  // The Add gets promoted; per-input fp16->fp32 casts are inserted, the
  // anchor Cast disappears (LN reads the fp32 Add directly), and the
  // original fp16 Add is erased because it has no remaining users.
  // -------------------------------------------------------------------------
  func.func @basic_residual_promoted(%a: tensor<2x4xf16>, %b: tensor<2x4xf16>,
                                     %scale: tensor<4xf32>, %bias: tensor<4xf32>)
      -> tensor<2x4xf32> {
    %add = "onnx.Add"(%a, %b) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %up = "onnx.Cast"(%add) : (tensor<2x4xf16>) -> tensor<2x4xf32>
    %ln = "onnx.LayerNormalization"(%up, %scale, %bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %ln : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @basic_residual_promoted
  // Each fp16 input gets cast to fp32 individually.
  // CHECK: %[[CA:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>
  // CHECK: %[[CB:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>
  // CHECK: %[[ADD32:.+]] = "onnx.Add"(%[[CA]], %[[CB]]){{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: "onnx.LayerNormalization"(%[[ADD32]],
  // The anchor fp16->fp32 Cast on the Add result must be gone.
  // CHECK-NOT: "onnx.Cast"({{.*}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>

  // -------------------------------------------------------------------------
  // Case 2: residual chain (block 22's Add then block 22's MLP residual Add).
  // Top-down walk + peek-through eliminates the intermediate bridge so the
  // whole chain ends up in fp32.
  // -------------------------------------------------------------------------
  func.func @cascaded_residual_chain_promoted(
      %a: tensor<2x4xf16>, %b: tensor<2x4xf16>, %mlp_out: tensor<2x4xf16>,
      %scale1: tensor<4xf32>, %bias1: tensor<4xf32>,
      %scale2: tensor<4xf32>, %bias2: tensor<4xf32>)
      -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    // First residual: feeds LN1 AND the second residual Add.
    %add1 = "onnx.Add"(%a, %b) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %up1 = "onnx.Cast"(%add1) : (tensor<2x4xf16>) -> tensor<2x4xf32>
    %ln1 = "onnx.LayerNormalization"(%up1, %scale1, %bias1)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    // Second residual: %add1 + %mlp_out, then Cast->LN.
    %add2 = "onnx.Add"(%add1, %mlp_out) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %up2 = "onnx.Cast"(%add2) : (tensor<2x4xf16>) -> tensor<2x4xf32>
    %ln2 = "onnx.LayerNormalization"(%up2, %scale2, %bias2)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %ln1, %ln2 : tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @cascaded_residual_chain_promoted
  // Two fp32 Adds remain (one per residual), with a 3rd Cast feeding mlp_out
  // into fp32 for %add2. No fp16 onnx.Add must remain.
  // CHECK-NOT: "onnx.Add"({{.*}}) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
  // Both LNs receive fp32 Add results directly.
  // CHECK-COUNT-2: "onnx.LayerNormalization"({{.*}}) {{.*}} : (tensor<2x4xf32>,
  // The anchor Casts on the original fp16 Add results must be gone.
  // CHECK-NOT: "onnx.Cast"({{.*}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>

  // -------------------------------------------------------------------------
  // Case 3: Add(fp16) feeds Cast->LN AND a separate fp16 consumer (e.g. an
  // unrelated downstream op that genuinely wants fp16). The promotion still
  // fires for the LN path; a fp32->fp16 bridge keeps the fp16 user alive.
  // -------------------------------------------------------------------------
  func.func @add_with_extra_fp16_user_keeps_bridge(
      %a: tensor<2x4xf16>, %b: tensor<2x4xf16>, %c: tensor<2x4xf16>,
      %scale: tensor<4xf32>, %bias: tensor<4xf32>)
      -> (tensor<2x4xf32>, tensor<2x4xf16>) {
    %add = "onnx.Add"(%a, %b) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %up = "onnx.Cast"(%add) : (tensor<2x4xf16>) -> tensor<2x4xf32>
    %ln = "onnx.LayerNormalization"(%up, %scale, %bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    // Extra fp16 user of the original Add. After the rewrite this consumer
    // must read from a Cast(fp32->fp16) bridge built on the new fp32 Add.
    %extra = "onnx.Mul"(%add, %c) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    return %ln, %extra : tensor<2x4xf32>, tensor<2x4xf16>
  }

  // CHECK-LABEL: func.func @add_with_extra_fp16_user_keeps_bridge
  // CHECK: %[[ADD32:.+]] = "onnx.Add"({{.*}}) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[BRIDGE:.+]] = "onnx.Cast"(%[[ADD32]]) : (tensor<2x4xf32>) -> tensor<2x4xf16>
  // CHECK: "onnx.LayerNormalization"(%[[ADD32]],
  // CHECK: "onnx.Mul"(%[[BRIDGE]],

  // -------------------------------------------------------------------------
  // Case 4: Add result is NOT consumed by Cast->LN; it's consumed by an Mul.
  // The pass must leave it untouched.
  // -------------------------------------------------------------------------
  func.func @add_not_feeding_ln_unchanged(%a: tensor<2x4xf16>, %b: tensor<2x4xf16>)
      -> tensor<2x4xf16> {
    %add = "onnx.Add"(%a, %b) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %y = "onnx.Mul"(%add, %add) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    return %y : tensor<2x4xf16>
  }

  // CHECK-LABEL: func.func @add_not_feeding_ln_unchanged
  // CHECK: "onnx.Add"({{.*}}) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
  // CHECK: "onnx.Mul"({{.*}}) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>

  // -------------------------------------------------------------------------
  // Case 5: Cast(fp16->fp32) feeds an fp32-pure non-LN consumer (Mul fp32).
  // PFC v2 fires because the consumer is fp32-pure (Mul fp32 produces fp32).
  // This models the chained-anchor pattern: Gemm/Add fp16 feeding an already-
  // promoted fp32 op upstream of LN -- the same shape that PFC v1's residual-
  // promotion produces in a multi-block chain. Eliminating the fp16 storage
  // here is exactly what kills the saturation source.
  // -------------------------------------------------------------------------
  func.func @cast_feeds_fp32_pure_consumer_promoted(
      %a: tensor<2x4xf16>, %b: tensor<2x4xf16>, %c: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %add = "onnx.Add"(%a, %b) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>
    %up = "onnx.Cast"(%add) : (tensor<2x4xf16>) -> tensor<2x4xf32>
    %y = "onnx.Mul"(%up, %c) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @cast_feeds_fp32_pure_consumer_promoted
  // Add gets promoted; the Mul reads from the new fp32 Add directly.
  // CHECK: %[[CA:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>
  // CHECK: %[[CB:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<2x4xf16>) -> tensor<2x4xf32>
  // CHECK: %[[ADD32:.+]] = "onnx.Add"(%[[CA]], %[[CB]]){{.*}} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: "onnx.Mul"(%[[ADD32]],
  // CHECK-NOT: "onnx.Add"({{.*}}) : (tensor<2x4xf16>, tensor<2x4xf16>) -> tensor<2x4xf16>

  // -------------------------------------------------------------------------
  // Case 6: Add is already fp32 (the standard ORT path on a model that has
  // already been promoted upstream, or an fp32 model). The simplifier must
  // be a no-op.
  // -------------------------------------------------------------------------
  func.func @already_fp32_unchanged(
      %a: tensor<2x4xf32>, %b: tensor<2x4xf32>,
      %scale: tensor<4xf32>, %bias: tensor<4xf32>)
      -> tensor<2x4xf32> {
    %add = "onnx.Add"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %ln = "onnx.LayerNormalization"(%add, %scale, %bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %ln : tensor<2x4xf32>
  }

  // CHECK-LABEL: func.func @already_fp32_unchanged
  // CHECK: "onnx.Add"({{.*}}) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK-NOT: "onnx.Cast"
  // CHECK: "onnx.LayerNormalization"

  // -------------------------------------------------------------------------
  // Case 7 (PFC v2): Gemm(fp16) -> Cast(fp32) -> LN.
  // Models the QKV projection pattern that produces fp16-saturating dot
  // products (block 21 vision: linear_85 -> Cast -> add_3461 chain). The
  // Gemm result is promoted to fp32; per-input fp16->fp32 casts are
  // inserted; alpha/beta/transA/transB attributes survive the rewrite.
  // -------------------------------------------------------------------------
  func.func @gemm_anchor_promoted(
      %x: tensor<8x4xf16>, %w: tensor<4x16xf16>, %bias: tensor<16xf16>,
      %scale: tensor<16xf32>, %ln_bias: tensor<16xf32>)
      -> tensor<8x16xf32> {
    %g = "onnx.Gemm"(%x, %w, %bias) {
        alpha = 1.0 : f32, beta = 1.0 : f32,
        transA = 0 : si64, transB = 0 : si64
    } : (tensor<8x4xf16>, tensor<4x16xf16>, tensor<16xf16>) -> tensor<8x16xf16>
    %up = "onnx.Cast"(%g) : (tensor<8x16xf16>) -> tensor<8x16xf32>
    %ln = "onnx.LayerNormalization"(%up, %scale, %ln_bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<8x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
    return %ln : tensor<8x16xf32>
  }

  // CHECK-LABEL: func.func @gemm_anchor_promoted
  // Each fp16 Gemm input is cast to fp32 individually.
  // CHECK: %[[CX:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<8x4xf16>) -> tensor<8x4xf32>
  // CHECK: %[[CW:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<4x16xf16>) -> tensor<4x16xf32>
  // CHECK: %[[CB:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<16xf16>) -> tensor<16xf32>
  // Promoted Gemm with attributes preserved; result is fp32.
  // CHECK: %[[G32:.+]] = "onnx.Gemm"(%[[CX]], %[[CW]], %[[CB]])
  // CHECK-SAME: alpha = 1.000000e+00 : f32
  // CHECK-SAME: beta = 1.000000e+00 : f32
  // CHECK-SAME: transA = 0 : si64
  // CHECK-SAME: transB = 0 : si64
  // CHECK-SAME: -> tensor<8x16xf32>
  // CHECK: "onnx.LayerNormalization"(%[[G32]],
  // No fp16 Gemm or fp16->fp32 anchor Cast must remain.
  // CHECK-NOT: "onnx.Gemm"({{.*}}) {{.*}} -> tensor<8x16xf16>
  // CHECK-NOT: "onnx.Cast"({{.*}}) : (tensor<8x16xf16>) -> tensor<8x16xf32>

  // -------------------------------------------------------------------------
  // Case 8 (PFC v2): MatMul(fp16) -> Cast(fp32) -> LN.
  // -------------------------------------------------------------------------
  func.func @matmul_anchor_promoted(
      %x: tensor<8x4xf16>, %w: tensor<4x16xf16>,
      %scale: tensor<16xf32>, %ln_bias: tensor<16xf32>)
      -> tensor<8x16xf32> {
    %m = "onnx.MatMul"(%x, %w)
        : (tensor<8x4xf16>, tensor<4x16xf16>) -> tensor<8x16xf16>
    %up = "onnx.Cast"(%m) : (tensor<8x16xf16>) -> tensor<8x16xf32>
    %ln = "onnx.LayerNormalization"(%up, %scale, %ln_bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<8x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
    return %ln : tensor<8x16xf32>
  }

  // CHECK-LABEL: func.func @matmul_anchor_promoted
  // CHECK: %[[CX:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<8x4xf16>) -> tensor<8x4xf32>
  // CHECK: %[[CW:.+]] = "onnx.Cast"(%{{.+}}) : (tensor<4x16xf16>) -> tensor<4x16xf32>
  // CHECK: %[[M32:.+]] = "onnx.MatMul"(%[[CX]], %[[CW]]){{.*}} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  // CHECK: "onnx.LayerNormalization"(%[[M32]],
  // CHECK-NOT: "onnx.MatMul"({{.*}}) {{.*}} -> tensor<8x16xf16>

  // -------------------------------------------------------------------------
  // Case 9 (PFC v2): Gemm(fp16) -> Cast -> Add(fp16) -> Cast -> LN, where the
  // Add is the residual that PFC v1 already handled.
  // Iteration: pass round 1 promotes the Add to fp32, which converts the
  // Gemm's downstream Cast(fp16->fp32) into a "feeds-fp32-pure-consumer"
  // pattern; pass round 2 then promotes the Gemm. End state has fp32 Gemm
  // feeding fp32 Add feeding fp32 LN with no fp16 storage between them --
  // exactly what ORT CPU's MLAS does internally for an fp16 GEMM->Add->LN
  // residual chain.
  // -------------------------------------------------------------------------
  func.func @chained_gemm_then_residual_promoted(
      %x: tensor<8x4xf16>, %w: tensor<4x16xf16>, %bias: tensor<16xf16>,
      %skip: tensor<8x16xf16>,
      %scale: tensor<16xf32>, %ln_bias: tensor<16xf32>)
      -> tensor<8x16xf32> {
    %g = "onnx.Gemm"(%x, %w, %bias) {
        alpha = 1.0 : f32, beta = 1.0 : f32,
        transA = 0 : si64, transB = 0 : si64
    } : (tensor<8x4xf16>, tensor<4x16xf16>, tensor<16xf16>) -> tensor<8x16xf16>
    %add = "onnx.Add"(%g, %skip) : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
    %up = "onnx.Cast"(%add) : (tensor<8x16xf16>) -> tensor<8x16xf32>
    %ln = "onnx.LayerNormalization"(%up, %scale, %ln_bias)
        {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64}
        : (tensor<8x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<8x16xf32>
    return %ln : tensor<8x16xf32>
  }

  // CHECK-LABEL: func.func @chained_gemm_then_residual_promoted
  // After two iterations the Gemm AND the Add must both be fp32; no fp16
  // op may remain on the path Gemm -> Add -> LN.
  // CHECK-NOT: "onnx.Gemm"({{.*}}) {{.*}} -> tensor<8x16xf16>
  // CHECK-NOT: "onnx.Add"({{.*}}) : (tensor<8x16xf16>, tensor<8x16xf16>) -> tensor<8x16xf16>
  // CHECK: %[[G32:.+]] = "onnx.Gemm"({{.*}}) {{.*}} -> tensor<8x16xf32>
  // CHECK: "onnx.Add"(%[[G32]],
  // CHECK-SAME: -> tensor<8x16xf32>
  // CHECK: "onnx.LayerNormalization"
}
