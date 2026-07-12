/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FoldQuantDequant.cpp - Remove activation fake-quant round-trips ----===//
//
// Folds hip.ms_dequantize_linear(hip.ms_quantize_linear(x)) -> x when the DQ
// and Q share the same scale, zero_point, and axis (a true round-trip), and the
// Q's pre-quantization input type equals the DQ's result type.
//
// ORCA's QDQ-format decode graph quantizes activations to ui16 and immediately
// dequantizes them (per-matmul-input fake-quant). At ui16 the round-trip error
// is ~1e-5 -- well below the fp16 the model already computes in -- so the pair
// is ~identity, yet each is a separate GPU kernel launch. A decode step fires
// ~960 such DQ + ~720 Q; folding them removes ~1700 tiny kernels/token.
//
// Safety: weight/param dequantizes (input is a constant, not a Q) never match.
// Real requantizes (DQ/Q with DIFFERENT scale or zp) never match. A Q left with
// no remaining uses is erased; a Q still feeding a non-DQ consumer is preserved.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "fold-quant-dequant"

STATISTIC(NumDQFolded, "Number of dequantize(quantize(x)) round-trips folded");
STATISTIC(NumQErased, "Number of now-dead quantize ops erased");
STATISTIC(NumRmsFused, "Number of decomposed RMSNorm-L2 chains fused");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FOLDQUANTDEQUANTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct FoldQuantDequantPass
    : public impl::FoldQuantDequantPassBase<FoldQuantDequantPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect first, mutate after: erasing during the walk would invalidate
    // the walk iterator.
    SmallVector<MsDequantizeLinearOp> foldable;
    module.walk([&](MsDequantizeLinearOp dq) {
      auto q = dq.getInput().getDefiningOp<MsQuantizeLinearOp>();
      if (!q)
        return;                                    // DQ of a constant/weight
      if (dq.getScale() != q.getScale())
        return;                                    // different scale
      if (dq.getZeroPoint() != q.getZeroPoint())
        return;                                    // different / missing zp
      if (dq.getAxis() != q.getAxis())
        return;
      // Must return to the original (pre-quantize) type for the fold to be
      // type-correct.
      if (q.getInput().getType() != dq->getResult(0).getType())
        return;
      foldable.push_back(dq);
    });

    for (MsDequantizeLinearOp dq : foldable) {
      auto q = dq.getInput().getDefiningOp<MsQuantizeLinearOp>();
      if (!q)
        continue;                                  // already cleaned up
      Value orig = q.getInput();
      dq->getResult(0).replaceAllUsesWith(orig);
      dq.erase();
      ++NumDQFolded;
      if (q->use_empty()) {
        q.erase();
        ++NumQErased;
      }
    }
    // Dead tensor.empty inits left by the erased ops are removed by the
    // canonicalize pass that follows in the pipeline.

    fuseRmsNormL2(module);
  }

  // Phase 2: fuse the decomposed RMSNorm-L2 chain
  //   sq = mul(a, a) -> reduce_sum -> sqrt -> reciprocal
  //   norm = mul(a, reciprocal) -> out = mul(weight, norm)
  // into a single hip.orca_rmsnorm_l2(a, weight). The 1/N mean factor and eps
  // were folded into `weight` at export, so the fused op uses sum + no eps and
  // is numerically exact (same fp32 ops, just fused into one kernel).
  static void fuseRmsNormL2(ModuleOp module) {
    SmallVector<ReduceSumOp> anchors;
    module.walk([&](ReduceSumOp rs) { anchors.push_back(rs); });

    for (ReduceSumOp rs : anchors) {
      // data must be mul(a, a) with identical operands.
      auto sq = rs.getData().getDefiningOp<MulOp>();
      if (!sq)
        continue;
      Value a = sq.getLhs();
      if (a != sq.getRhs())
        continue;

      Value rsRes = rs->getResult(0);
      if (!rsRes.hasOneUse())
        continue;
      auto sqrtOp = dyn_cast<SqrtOp>(*rsRes.getUsers().begin());
      if (!sqrtOp)
        continue;
      Value sqrtRes = sqrtOp->getResult(0);
      if (!sqrtRes.hasOneUse())
        continue;
      auto recip = dyn_cast<ReciprocalOp>(*sqrtRes.getUsers().begin());
      if (!recip)
        continue;
      Value recipRes = recip->getResult(0);
      if (!recipRes.hasOneUse())
        continue;
      auto normMul = dyn_cast<MulOp>(*recipRes.getUsers().begin());
      if (!normMul)
        continue;
      // normMul must be mul(a, recip) in either operand order.
      Value other = (normMul.getLhs() == recipRes) ? normMul.getRhs()
                    : (normMul.getRhs() == recipRes) ? normMul.getLhs()
                                                     : Value();
      if (other != a)
        continue;
      Value normRes = normMul->getResult(0);
      if (!normRes.hasOneUse())
        continue;
      auto wMul = dyn_cast<MulOp>(*normRes.getUsers().begin());
      if (!wMul)
        continue;
      // wMul = mul(weight, norm) in either order.
      Value weight = (wMul.getLhs() == normRes) ? wMul.getRhs()
                     : (wMul.getRhs() == normRes) ? wMul.getLhs()
                                                  : Value();
      if (!weight)
        continue;

      auto aTy = dyn_cast<RankedTensorType>(a.getType());
      if (!aTy || aTy.getRank() < 1)
        continue;
      Value finalRes = wMul->getResult(0);
      if (finalRes.getType() != a.getType())
        continue; // fused output must have the input's shape/type

      OpBuilder b(wMul);
      Value ctx = sq.getCtx();
      Value out = tensor::EmptyOp::create(b, wMul.getLoc(), aTy.getShape(),
                                          aTy.getElementType());
      auto fused = OrcaRmsNormL2Op::create(
          b, wMul.getLoc(), TypeRange{finalRes.getType()}, ctx, a, weight, out,
          b.getI64IntegerAttr(aTy.getRank() - 1));

      finalRes.replaceAllUsesWith(fused->getResult(0));
      // Erase the now-dead chain leaf-to-root (weight and `a` are preserved).
      wMul.erase();
      normMul.erase();
      recip.erase();
      sqrtOp.erase();
      rs.erase();
      if (sq->use_empty())
        sq.erase();
      ++NumRmsFused;
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
