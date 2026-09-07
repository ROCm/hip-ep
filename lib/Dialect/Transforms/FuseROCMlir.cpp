#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir"

namespace {

class FuseGemmPointwise : public OpRewritePattern<GemmOp> {
public:
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp op,
                                PatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(op, "no-op");
  }
};

class FuseConvPointwise : public OpRewritePattern<ConvOp> {
public:
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(op, "multi-use");

    if (auto maxOp = dyn_cast_if_present<MaxOp>(*op->user_begin())) {
      // Create a new function
      auto parentModule = op->getParentOfType<ModuleOp>();
      func::FuncOp newFunc;
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        auto funcType = rewriter.getFunctionType(
            llvm::map_to_vector(op.getDpsInputs(),
                                [](Value v) { return v.getType(); }),
            maxOp->getResultTypes());
        newFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                       "convRelu" + std::to_string(counter++),
                                       funcType);
        auto *funcBlock = newFunc.addEntryBlock();
        IRMapping mapping;
        for (auto [idx, operand] : llvm::enumerate(op.getDpsInputs())) {
          mapping.map(operand, funcBlock->getArgument(idx));
        }

        // Clone Conv op
        rewriter.setInsertionPointToStart(&newFunc.getRegion().front());
        for (auto init : op.getDpsInits()) {
          rewriter.clone(*init.getDefiningOp(), mapping);
        }
        rewriter.clone(*op, mapping);

        // Clone Max op
        for (auto init : maxOp.getDpsInits()) {
          rewriter.clone(*init.getDefiningOp(), mapping);
        }
        auto maxSecondOperand =
            rewriter.clone(*maxOp->getOperand(2).getDefiningOp(), mapping);
        if (auto maxConstOp =
                dyn_cast_if_present<ConstantOp>(maxSecondOperand)) {
          maxConstOp->removeAttr("serialization_order");
        }
        rewriter.clone(*maxOp, mapping);
        func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(),
                               mapping.lookup(maxOp->getResult(0)));
      }

      auto callOp = func::CallOp::create(rewriter, rewriter.getUnknownLoc(),
                                         newFunc, op.getDpsInputs());
      rewriter.replaceOp(maxOp, callOp);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "failure");
  }

private:
  static inline int counter = 0;
};

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseGemmPointwise, FuseConvPointwise>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

}; // namespace mlir::hip
