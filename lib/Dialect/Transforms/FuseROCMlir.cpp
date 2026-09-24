/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
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

    // Fail on dynamic shapes
    for (Value operand : anchorOp->getOperands()) {
      if (auto tensorType = dyn_cast<TensorType>(operand.getType());
          tensorType && !tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(anchorOp,
                                           "only static shapes are supported");
    }

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
        // An init with no producer is a block argument, which cannot be cloned.
        // It is picked up as a kernel argument below instead.
        if (Operation *initOp = init.getDefiningOp())
          ops.insert(initOp);
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

    // `ops` is cloned into an IsolatedFromAbove func, so every value it reads
    // has to resolve inside that func. The DPS inputs above cover the data,
    // but they do not cover what the inits are built from: an init is only a
    // bare tensor.empty when the anchor already has the rank rocMLIR wants.
    // A 1-D convolution does not -- convert-onnx-to-hip widens it to 2-D by
    // wrapping the operands *and the init* in tensor.expand_shape -- so the
    // init's producer is a reshape whose own operand is the empty. Cloning
    // just the reshape leaves it pointing at an empty back in main_graph.
    //
    // Absorb producers that carry no operands and no side effects, so a
    // materialised destination travels with the ops that use it instead of
    // becoming a kernel argument. That keeps the signature rocMLIR sees
    // unchanged for the shapes that already worked. Values already headed for
    // the argument list stay there: a weight is passed in even though
    // hip.constant would qualify to be cloned.
    SmallVector<Operation *> worklist(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (Value operand : op->getOperands()) {
        if (operands.contains(operand))
          continue;
        Operation *producer = operand.getDefiningOp();
        if (!producer || ops.contains(producer))
          continue;
        if (producer->getNumOperands() != 0 || !isMemoryEffectFree(producer))
          continue;
        ops.insert(producer);
        worklist.push_back(producer);
      }
    }

    // Whatever is still read from outside has to be passed in, which costs one
    // more kernel argument and is what any init producer that cannot be
    // rematerialized -- a caller-supplied buffer, say -- falls back on.
    for (Operation *op : ops)
      for (Value operand : op->getOperands())
        if (operand != context && !ops.contains(operand.getDefiningOp()))
          operands.insert(operand);

    // Cloning follows this order, so a producer absorbed above has to come
    // before the op that reads it.
    ops = topologicalSort(ops);

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
                           SubOp, CastOp, LessOp, SignOp, ModOp, WhereOp>(op);
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
