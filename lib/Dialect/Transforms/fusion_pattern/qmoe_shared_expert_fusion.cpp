/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- qmoe_shared_expert_fusion.cpp - shared expert into hip.qmoe --------===//
//
// A shared expert has the shape of one more routed expert, so the fused QMoE
// decode kernels can run it as an extra slot and the separate SwiGLU, down
// projection, sigmoid, multiply and add launches disappear.
//
//   Before:
//     %m  = hip.qmoe(%ctx) ins(%x, %router, ...) outs(%e0 : t) : t
//     %g  = hip.matmul_nbits(%ctx) ins(%x, %Wg, %Sg : ...) {N = I}
//     %u  = hip.matmul_nbits(%ctx) ins(%x, %Wu, %Su : ...) {N = I}
//     %a  = hip.swiglu(%ctx) ins(%g, %u : ...)
//     %d  = hip.matmul_nbits(%ctx) ins(%a, %Wd, %Sd : ...) {K = I, N = H}
//     %l  = hip.matmul_nbits(%ctx) ins(%x, %Wl, %Sl : ...) {N = 1}
//     %s  = hip.sigmoid(%ctx) ins(%l : ...)
//     %sd = hip.mul(%ctx) ins(%d, %s : t, tensor<..x1xf16>) outs(%e1 : t)
//     %y  = hip.add(%ctx) ins(%m, %sd : t, t) outs(%e2 : t)
//   After:
//     %l  = hip.matmul_nbits(%ctx) ins(%x, %Wl, %Sl : ...) {N = 1}
//     %y  = hip.qmoe(%ctx) ins(%x, %router, ...)
//             shared_expert(%Wg, %Sg, %Wu, %Su, %Wd, %Sd, %l : ...)
//             outs(%e2 : t) : t
//
// The fused op sits at the add, whose destination it takes over, so every
// operand already dominates it. Shape queries of the erased values read only
// leading axes, which all come from %x, and are retargeted there.
//
// The runtime computes the shared expert itself on every path (decode slot,
// prefill, fp fallback), so the match only has to establish the semantics:
// symmetric int4 projections at the QMoE's block size, all reading %x.
//
//===----------------------------------------------------------------------===//

#include "qmoe_shared_expert_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace hip {
namespace fusion_transform {
namespace {

using mlir::hip::AddOp;
using mlir::hip::MatMulNBitsOp;
using mlir::hip::MulOp;
using mlir::hip::QMoEOp;
using mlir::hip::SigmoidOp;
using mlir::hip::SwigluOp;

// True when every use is either `consumer` or a tensor.dim.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

// A projection the shared-expert kernels can run: symmetric int4 with fp16
// scales at the routed experts' block size.
bool isPlainInt4Projection(MatMulNBitsOp mm, int64_t blockSize) {
  return mm && !mm.getZeroPoints() && !mm.getGIdx() && !mm.getBias() &&
         !mm.getAQ() && mm.getBits() == 4 && mm.getBlockSize() == blockSize &&
         mm.getScaleElemSize() == 2 && mm->getNumResults() == 1;
}

template <typename OpT>
bool matchEither(mlir::Value lhs, mlir::Value rhs, OpT &op,
                 mlir::Value &other) {
  if ((op = lhs.getDefiningOp<OpT>())) {
    other = rhs;
    return true;
  }
  if ((op = rhs.getDefiningOp<OpT>())) {
    other = lhs;
    return true;
  }
  return false;
}

struct QMoESharedExpertFusion : public mlir::OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp add, mlir::PatternRewriter &rewriter) const override {
    QMoEOp moe;
    mlir::Value gatedValue;
    if (!matchEither(add.getLhs(), add.getRhs(), moe, gatedValue))
      return rewriter.notifyMatchFailure(add, "no hip.qmoe operand");
    if (moe.hasSharedExpert() || moe.getExpertWeightBits() != 4 ||
        moe->getNumResults() != 1)
      return rewriter.notifyMatchFailure(add, "qmoe cannot take a shared "
                                              "expert");

    auto gated = gatedValue.getDefiningOp<MulOp>();
    if (!gated)
      return rewriter.notifyMatchFailure(add, "shared branch is not hip.mul");
    MatMulNBitsOp down;
    mlir::Value gateProb;
    if (!matchEither(gated.getLhs(), gated.getRhs(), down, gateProb))
      return rewriter.notifyMatchFailure(add, "no down projection");
    auto sigmoid = gateProb.getDefiningOp<SigmoidOp>();
    if (!sigmoid)
      return rewriter.notifyMatchFailure(add, "shared gate is not sigmoid");
    auto logit = sigmoid.getX().getDefiningOp<MatMulNBitsOp>();
    auto swiglu = down.getA().getDefiningOp<SwigluOp>();
    if (!logit || !swiglu)
      return rewriter.notifyMatchFailure(add, "shared expert shape mismatch");
    auto gate = swiglu.getGate().getDefiningOp<MatMulNBitsOp>();
    auto up = swiglu.getUp().getDefiningOp<MatMulNBitsOp>();

    mlir::Value x = moe.getInput();
    const int64_t blockSize = moe.getBlockSize();
    for (MatMulNBitsOp mm : {gate, up, down, logit})
      if (!isPlainInt4Projection(mm, blockSize))
        return rewriter.notifyMatchFailure(add,
                                           "projection is not plain int4");
    if (gate.getA() != x || up.getA() != x || logit.getA() != x)
      return rewriter.notifyMatchFailure(add,
                                         "projections read another input");

    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType || !xType.getElementType().isF16() || xType.getRank() < 2 ||
        xType.isDynamicDim(xType.getRank() - 1))
      return rewriter.notifyMatchFailure(add, "input is not [..., H] f16");
    const int64_t hidden = xType.getDimSize(xType.getRank() - 1);
    const int64_t inter = gate.getN();
    if (up.getN() != inter || gate.getK() != hidden || up.getK() != hidden ||
        down.getK() != inter || down.getN() != hidden ||
        logit.getK() != hidden || logit.getN() != 1)
      return rewriter.notifyMatchFailure(add, "shared expert sizes mismatch");

    auto resultType = add->getResult(0).getType();
    if (moe->getResult(0).getType() != resultType ||
        gated->getResult(0).getType() != resultType ||
        down->getResult(0).getType() != resultType || x.getType() != resultType)
      return rewriter.notifyMatchFailure(add, "shared expert broadcasts");
    // The runtime reads one fp16 gate logit per token of %x.
    llvm::SmallVector<int64_t> logitShape(xType.getShape());
    logitShape.back() = 1;
    auto logitType =
        mlir::RankedTensorType::get(logitShape, xType.getElementType());
    if (logit->getResult(0).getType() != logitType ||
        sigmoid->getResult(0).getType() != logitType)
      return rewriter.notifyMatchFailure(add, "gate logit is not [..., 1] "
                                              "f16");

    mlir::Value ctx = add.getCtx();
    for (mlir::Value c : {moe.getHandle(), gated.getCtx(), sigmoid.getCtx(),
                          down.getHandle(), swiglu.getCtx(), gate.getHandle(),
                          up.getHandle(), logit.getHandle()})
      if (c != ctx)
        return rewriter.notifyMatchFailure(add, "ops must share a context");

    if (!hasSingleValueUse(moe->getResult(0), add) ||
        !hasSingleValueUse(gated->getResult(0), add) ||
        !hasSingleValueUse(down->getResult(0), gated) ||
        !hasSingleValueUse(sigmoid->getResult(0), gated) ||
        !hasSingleValueUse(swiglu->getResult(0), down) ||
        !hasSingleValueUse(gate->getResult(0), swiglu) ||
        !hasSingleValueUse(up->getResult(0), swiglu) || gate == up)
      return rewriter.notifyMatchFailure(
          add, "shared expert intermediates must have one value use");

    // Leading axes of every erased value are the input's; the trailing one
    // is static. Uses other than shape queries are the erased ops and the add.
    mlir::Operation *erased[] = {gated,  sigmoid, down,
                                 swiglu, gate,    up,   moe};
    if (!llvm::all_of(erased, [](mlir::Operation *op) {
          return mlir::isMemoryEffectFree(op);
        }))
      return rewriter.notifyMatchFailure(add, "chain has memory effects");
    for (mlir::Operation *op : erased) {
      mlir::Value v = op->getResult(0);
      auto vType = mlir::cast<mlir::RankedTensorType>(v.getType());
      for (mlir::Operation *user : v.getUsers()) {
        auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user);
        if (!dim)
          continue;
        std::optional<int64_t> index = dim.getConstantIndex();
        if (!index || *index >= vType.getRank() ||
            (*index == vType.getRank() - 1 && vType.isDynamicDim(*index)))
          return rewriter.notifyMatchFailure(add, "unexpected shape query");
      }
    }

    rewriter.setInsertionPoint(add);
    auto fused = QMoEOp::create(
        rewriter, rewriter.getFusedLoc({moe.getLoc(), add.getLoc()}),
        resultType, ctx, x, moe.getRouterProbs(), moe.getFc1ExpertsWeights(),
        moe.getFc1Scales(), moe.getFc2ExpertsWeights(), moe.getFc2Scales(),
        moe.getFc1ExpertsBias(), moe.getFc2ExpertsBias(),
        moe.getFc3ExpertsWeights(), moe.getFc3Scales(),
        moe.getFc3ExpertsBias(), moe.getFc1ZeroPoints(),
        moe.getFc2ZeroPoints(), moe.getFc3ZeroPoints(),
        moe.getRouterWeights(), gate.getB(), gate.getScales(), up.getB(),
        up.getScales(), down.getB(), down.getScales(), logit->getResult(0),
        moe.getInputQ(), moe.getInputScale(), add.getOutput(), moe.getExpertWeightBitsAttr(), moe.getKAttr(),
        moe.getBlockSizeAttr(), moe.getNormalizeRoutingWeightsAttr(),
        moe.getSwigluFusionAttr(), moe.getUseSparseMixerAttr(),
        moe.getActivationAlphaAttr(), moe.getActivationBetaAttr(),
        moe.getSwigluLimitAttr(), moe.getActivationTypeAttr());

    for (mlir::Operation *op : erased) {
      mlir::Value v = op->getResult(0);
      for (mlir::OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
        auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
        if (!dim)
          continue;
        int64_t index = *dim.getConstantIndex();
        auto vType = mlir::cast<mlir::RankedTensorType>(v.getType());
        rewriter.setInsertionPoint(dim);
        if (vType.isDynamicDim(index))
          rewriter.modifyOpInPlace(dim, [&] { use.set(x); });
        else
          rewriter.replaceOp(dim, mlir::arith::ConstantIndexOp::create(
                                      rewriter, dim.getLoc(),
                                      vType.getDimSize(index)));
      }
    }

    rewriter.replaceOp(add, fused->getResult(0));
    llvm::SmallVector<mlir::Operation *> inits;
    for (mlir::Operation *op : erased) {
      for (mlir::Value init :
           mlir::cast<mlir::DestinationStyleOpInterface>(op).getDpsInits())
        if (mlir::Operation *def = init.getDefiningOp())
          inits.push_back(def);
      rewriter.eraseOp(op);
    }
    // Drop destinations that only fed erased ops. Two members may share one;
    // the set keeps it from being erased twice. A producer with side effects
    // stays, and the greedy driver sweeps the shape queries left dead.
    llvm::SmallPtrSet<mlir::Operation *, 8> dropped;
    for (mlir::Operation *def : inits)
      if (!dropped.contains(def) && mlir::isOpTriviallyDead(def)) {
        dropped.insert(def);
        rewriter.eraseOp(def);
      }
    return mlir::success();
  }
};

} // namespace

void populateQMoESharedExpertFusionPattern(mlir::RewritePatternSet &patterns,
                                           mlir::PatternBenefit benefit) {
  patterns.add<QMoESharedExpertFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
