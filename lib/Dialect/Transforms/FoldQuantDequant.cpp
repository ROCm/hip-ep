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

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "fold-quant-dequant"

STATISTIC(NumDQFolded, "Number of dequantize(quantize(x)) round-trips folded");
STATISTIC(NumQErased, "Number of now-dead quantize ops erased");

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
  }
};

} // namespace

} // namespace hip
} // namespace mlir
