/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRUSEOUTPUTALLOCATORPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct ReplaceOutputAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Find if this alloc is returned (graph output) and at which index.
    func::ReturnOp returnOp = nullptr;
    int64_t outIdx = -1;
    for (OpOperand &use : allocOp->getUses()) {
      if (auto ret = dyn_cast<func::ReturnOp>(use.getOwner())) {
        returnOp = ret;
        outIdx = use.getOperandNumber();
        break; // first occurrence (aliased outputs share one rewrite)
      }
    }
    if (!returnOp)
      return failure(); // not a graph output

    auto func = allocOp->getParentOfType<func::FuncOp>();
    if (!func || !func.isPublic())
      return failure();

    if (func.getNumArguments() == 0 ||
        !isa<ContextType>(func.getArgument(0).getType()))
      return failure();
    Value ctx = func.getArgument(0);

    // EP owns the buffer now; drop any dealloc (it references the alloc
    // result).
    for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
        rewriter.eraseOp(dealloc);

    auto allocOutput = AllocOutputOp::create(
        rewriter, allocOp.getLoc(), allocOp.getType(), ctx,
        allocOp.getDynamicSizes(), rewriter.getI64IntegerAttr(outIdx));

    rewriter.replaceOp(allocOp, allocOutput.getResult());
    return success();
  }
};

struct HipsrUseOutputAllocatorPass
    : impl::HipsrUseOutputAllocatorPassBase<HipsrUseOutputAllocatorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceOutputAllocPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir