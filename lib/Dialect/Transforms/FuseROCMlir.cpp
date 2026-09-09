/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir"

namespace {

template <typename AnchorOp>
class FuseAnchorPointwise : public OpRewritePattern<AnchorOp> {
public:
  FuseAnchorPointwise(MLIRContext *context, int *counter,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<AnchorOp>(context, benefit), counter(counter) {}

  LogicalResult matchAndRewrite(AnchorOp anchorOp,
                                PatternRewriter &rewriter) const override {
    DestinationStyleOpInterface endOp = anchorOp;
    Operation *prevOp = nullptr;
    SetVector<Value> operands;
    SetVector<Operation *> ops;

    // Match all pointwise-ops
    do {
      if (prevOp) {
        endOp = dyn_cast<DestinationStyleOpInterface>(*prevOp->user_begin());
      }
      auto newOperands = endOp.getDpsInputs();
      if (prevOp) {
        newOperands.erase(newOperands.begin() +
                          prevOp->use_begin()->getOperandNumber());
      }
      operands.insert_range(newOperands);
      for (auto init : endOp.getDpsInits()) {
        ops.insert(init.getDefiningOp());
      }
      ops.insert(endOp);
      prevOp = endOp;
    } while (endOp->hasOneUse() && isaPointwiseOp(*endOp->user_begin()));

    // Remove the hip.context operand
    auto context = operands.front();
    if (!isa<mlir::hip::ContextType>(context.getType())) {
      return rewriter.notifyMatchFailure(anchorOp,
                                         "first operand not hip.context");
    }
    operands.erase(operands.begin());

    auto parentModule = anchorOp->template getParentOfType<ModuleOp>();
    func::FuncOp newFunc;
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      auto funcType = rewriter.getFunctionType(
          llvm::map_to_vector(operands, [](Value v) { return v.getType(); }),
          endOp->getResultTypes());

      newFunc = func::FuncOp::create(rewriter, rewriter.getUnknownLoc(),
                                     "rocMlir" + std::to_string((*counter)++),
                                     funcType);
      newFunc->setAttr("rock.kernel", rewriter.getUnitAttr());
      newFunc->setAttr("rock.arch", rewriter.getStringAttr("gfx1151"));
      auto *funcBlock = newFunc.addEntryBlock();
      rewriter.setInsertionPointToStart(funcBlock);
      IRMapping mapping;

      // Map the hip.context to ub.poison
      mapping.map(context,
                  ub::PoisonOp::create(rewriter, rewriter.getUnknownLoc(),
                                       context.getType()));

      // Map other operands
      for (auto [idx, operand] : llvm::enumerate(operands)) {
        mapping.map(operand, funcBlock->getArgument(idx));
      }

      for (auto *op : ops) {
        rewriter.clone(*op, mapping);
      }
      auto returns =
          llvm::map_to_vector(endOp->getResults(), [&mapping](Value v) {
            return mapping.lookup(v);
          });
      func::ReturnOp::create(rewriter, rewriter.getUnknownLoc(), returns);
    }

    rewriter.setInsertionPointAfter(endOp);
    auto rocMlirOp = RocMlirOp::create(
        rewriter, rewriter.getUnknownLoc(), endOp->getResultTypes(),
        SymbolRefAttr::get(newFunc), context,
        SmallVector<Value>(operands.begin(), operands.end()),
        endOp.getDpsInits().front());

    rewriter.replaceOp(endOp, rocMlirOp);
    return success();
  }

  bool isaPointwiseOp(Operation *op) const {
    return isa_and_present<MulOp, AddOp, MinOp, MaxOp, SiluOp, SigmoidOp,
                           TanhOp, SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp,
                           LeakyReluOp, ReciprocalOp, SqrtOp, DivOp, EqualOp,
                           AndOp, OrOp, NotOp, CosOp, ErfOp, SinOp, CeilOp,
                           RoundOp, AtanOp, FloorOp, ExpOp, LogOp, AbsOp, NegOp,
                           SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp,
                           MiopenSoftmaxOp, QuantizeLinearOp,
                           DequantizeLinearOp>(op);
  }

private:
  int *counter = nullptr;
};

class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    int counter = 0;
    patterns.add<FuseAnchorPointwise<MatmulOp>, FuseAnchorPointwise<ConvOp>,
                 FuseAnchorPointwise<GemmOp>>(ctx, &counter);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }

private:
  int counter = 0;
};

} // namespace

}; // namespace mlir::hip
