/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FuseMulAddPass.cpp - Plugin fusion pass: hip.fused_mul_add ---------===//
//
// Out-of-tree fusion pass contributed by the hip-fusion plugin.
//
// Pattern:
//   %t = hip.mul(%ctx) ins(%x, %b) outs(%init0) : ...
//   %y = hip.add(%ctx) ins(%t, %a) outs(%init1) : ...
//
// becomes:
//   %y = hip.fused_mul_add(%ctx) ins(%x, %b, %a) outs(%init1) : ...
//        == (b * x) + a
//
// Guard: the `hip.mul` result must be single-use (the consuming `hip.add`).
// Both `add(mul, a)` and `add(a, mul)` operand orderings are recognized.
//
// Mirrors the in-tree FuseAddMul.cpp exemplar but lives entirely in a
// dynamically-loaded plugin: it matches the in-tree hip.mul / hip.add ops
// (resolved from the host at load time) and produces the plugin-contributed
// hip.fused_mul_add op.
//
//===----------------------------------------------------------------------===//

#include "plugins/fusion/Passes.h"
#include "plugins/fusion/PluginOps.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-fuse-mul-add"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HIPFUSEMULADDPASS
#include "plugins/fusion/Passes.h.inc"

namespace {

/// Rewrites `hip.add(hip.mul(x, b), a)` into `hip.fused_mul_add(x, b, a)`
/// when the intermediate `hip.mul` result is single-use.
struct FuseMulAddPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    // Try lhs-as-mul first, then rhs-as-mul.
    MulOp mulOp = lhs.getDefiningOp<MulOp>();
    Value addend = rhs;
    if (!mulOp) {
      mulOp = rhs.getDefiningOp<MulOp>();
      addend = lhs;
    }
    if (!mulOp)
      return rewriter.notifyMatchFailure(addOp, "no producing hip.mul");

    // Single-use guard: do not duplicate the multiplication.
    if (!mulOp->hasOneUse())
      return rewriter.notifyMatchFailure(mulOp, "hip.mul result has >1 uses");

    // Both ops must share the same hip.context value.
    if (mulOp.getCtx() != addOp.getCtx())
      return rewriter.notifyMatchFailure(addOp, "mul/add ctx values differ");

    Value x = mulOp.getLhs();
    Value b = mulOp.getRhs();
    Value a = addend;
    Value init = addOp.getOutput();

    auto fused = FusedMulAddOp::create(
        rewriter, addOp.getLoc(), addOp.getResult(0).getType(), addOp.getCtx(),
        x, b, a, init);

    rewriter.replaceOp(addOp, fused.getResult(0));
    rewriter.eraseOp(mulOp);

    LLVM_DEBUG(llvm::dbgs() << "  fused mul+add -> hip.fused_mul_add\n");
    return success();
  }
};

struct HipFuseMulAddPass
    : public impl::HipFuseMulAddPassBase<HipFuseMulAddPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.empty())
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMulAddPattern>(&getContext());

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
