/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SiluFusion.cpp - Fold Mul(x, Sigmoid(x)) into hip.silu ------------===//
//
// The Qwen3.5 SwiGLU MLP emits SiLU as a separate `onnx.Sigmoid` + `onnx.Mul`
// (x * sigmoid(x)) rather than a fused activation. There is a `hip.silu` op
// (output = input * sigmoid(input)) with a kernel + lowering, but no pass
// creates it from the primitive pair, so each SiLU runs as two launch-bound
// kernels (56 sites in text.onnx). This pass folds the pair into one
// `hip.silu`, halving those launches. Output is bit-identical iff the silu
// kernel computes x*sigmoid(x) the same way (validated by text match).
//
// Rooted on `onnx.Mul`; matches when one operand is `onnx.Sigmoid(s)` and the
// OTHER operand is exactly `s` (true SiLU, not a gate `a*sigmoid(b)`).
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "silu-fusion"

STATISTIC(NumSiluFused, "Mul(x, Sigmoid(x)) pairs folded into hip.silu");

namespace mlir {
namespace hip {

namespace {

struct SiluFuse : public mlir::RewritePattern {
  SiluFuse(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *mulOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (mulOp->getNumOperands() != 2 || mulOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(mulOp, "mul.arity");

    // One operand must be Sigmoid(s); the OTHER must be exactly s.
    mlir::Value x;
    mlir::Operation *sigOp = nullptr;
    for (int i = 0; i < 2; ++i) {
      mlir::Operation *def = mulOp->getOperand(i).getDefiningOp();
      if (def && def->getName().getStringRef() == "onnx.Sigmoid" &&
          def->getNumOperands() == 1 &&
          def->getOperand(0) == mulOp->getOperand(1 - i)) {
        sigOp = def;
        x = mulOp->getOperand(1 - i);
        break;
      }
    }
    if (!sigOp)
      return rewriter.notifyMatchFailure(mulOp, "not_silu");

    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(mulOp->getResult(0).getType());
    if (!xType || !outType || outType.getShape() != xType.getShape() ||
        outType.getElementType() != xType.getElementType())
      return rewriter.notifyMatchFailure(mulOp, "type_mismatch");

    auto ctxOrFailure = getContextArg(mulOp, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(mulOp, "no_context_arg");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = mulOp->getLoc();
    rewriter.setInsertionPoint(mulOp);
    mlir::Value init = createEmptyTensor(rewriter, loc, outType, x);
    auto hipOp =
        mlir::hip::SiluOp::create(rewriter, loc, outType, context, x, init);
    rewriter.replaceOp(mulOp, hipOp->getResult(0));
    if (sigOp->use_empty())
      rewriter.eraseOp(sigOp);

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused SiLU at " << loc << "\n");
    ++NumSiluFused;
    return mlir::success();
  }
};

} // namespace

void populateSiluFusionPatterns(mlir::RewritePatternSet &patterns,
                                mlir::MLIRContext *ctx) {
  if (const char *env = std::getenv("HIPDNN_EP_SILU_FUSE"))
    if (std::string(env) == "0")
      return;
  patterns.add<SiluFuse>(ctx);
}

} // namespace hip
} // namespace mlir
