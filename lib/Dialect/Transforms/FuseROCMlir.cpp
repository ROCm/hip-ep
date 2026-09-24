/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
          llvm::map_to_vector(operands, [](Value v) -> Type {
            auto type = dyn_cast<RankedTensorType>(v.getType());
            if (type && type.getRank() == 0)
              return RankedTensorType::get({1}, type.getElementType(),
                                           type.getEncoding());
            return v.getType();
          }),
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
    SmallVector<Value> dispatchOperands;
    dispatchOperands.reserve(operands.size());
    Value rankOneShape;
    for (Value operand : operands) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getRank() != 0) {
        dispatchOperands.push_back(operand);
        continue;
      }

      // rocMLIR's MIGraphX path represents scalar elementwise operands as
      // one-element tensors. Match that ABI at the outlining boundary instead
      // of passing a rank-0 buffer into rock.transforms_to_ptr: a scalar has no
      // coordinate to linearize, while tensor<1xT> has the expected offset 0.
      if (!rankOneShape) {
        auto shapeType =
            RankedTensorType::get({1}, rewriter.getIndexType());
        auto shapeAttr =
            DenseIntElementsAttr::get(shapeType, ArrayRef<int64_t>{1});
        rankOneShape = arith::ConstantOp::create(
            rewriter, rewriter.getUnknownLoc(), shapeType, shapeAttr);
      }
      auto rankOneType =
          RankedTensorType::get({1}, type.getElementType(), type.getEncoding());
      dispatchOperands.push_back(tensor::ReshapeOp::create(
          rewriter, rewriter.getUnknownLoc(), rankOneType, operand,
          rankOneShape));
    }
    auto rocMlirOp = RocMlirOp::create(
        rewriter, rewriter.getUnknownLoc(), endOp->getResultTypes(),
        SymbolRefAttr::get(newFunc), context,
        dispatchOperands, endOp.getDpsInits().front());

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
