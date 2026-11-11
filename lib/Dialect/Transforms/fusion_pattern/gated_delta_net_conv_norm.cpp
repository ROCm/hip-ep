/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gated_delta_net_conv_norm.cpp - linear-attention core to one op ----===//
//
// Decode spends most of the linear-attention core on launches and on the
// strided copies bufferization inserts for the column slices, so the chain
// from the projections to the gated norm becomes one op the runtime can run
// as a single kernel for few tokens.
//
//   Before:
//     %x   = tensor.extract_slice %proj[0, 0, 0] [%d0, %d1, C] [1, 1, 1]
//     %c:2 = hip.causal_conv_with_state(%ctx) ins(%x, %w, %bias, %past)
//              outs(%e0, %e1) {activation = "silu", channels_last = true}
//     %ds  = tensor.extract_slice %gates[0, 0, dc] [%d0, %d1, Hv] [1, 1, 1]
//     %d   = hip.cast(%ctx) ins(%ds) outs(%e2) {to = 1}
//     %bs  = tensor.extract_slice %gates[0, 0, bc] [%d0, %d1, Hv] [1, 1, 1]
//     %b   = hip.cast(%ctx) ins(%bs) outs(%e3) {to = 1}
//     %o:2 = hip.gated_delta_net(%ctx) ins(%c#0, %d, %b, %S, %A, %dt)
//              outs(%e4, %e5)
//     %f   = tensor.collapse_shape %o#0 [[0], [1], [2, 3]]
//     %z   = tensor.extract_slice %proj[0, 0, zc] [%d0, %d1, Hv*dv] [1, 1, 1]
//     %y   = hip.gated_rms_norm(%ctx) ins(%f, %nw, %z) outs(%e6)
//              {activation = "sigmoid"}
//   After:
//     %e7  = tensor.empty(%d0, %d1) : [.., C]
//     %e8  = tensor.empty(%d0, %d1) : [.., Hv*dv]
//     %r:9 = hip.gated_delta_net_conv_norm(%ctx)
//              ins(%proj, %w, %bias, %past, %gates, %S, %A, %dt, %nw)
//              outs(%e6, %e1, %e5, %e7, %e0, %e2, %e3, %e4, %e8)
//     uses of %y -> %r#0, %c#1 -> %r#1, %o#1 -> %r#2
//
// A channels-first convolution bracketed by [0, 2, 1] transposes, the form
// before the canonicalizer folds the pair, is accepted the same way; its
// output init is then a fresh [.., C] tensor. The intermediates must have no
// other value use; a tensor.dim of one of them on a batch or sequence axis is
// answered from the slice source instead.
//
//===----------------------------------------------------------------------===//

#include "gated_delta_net_conv_norm.hpp"
#include "column_slice.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"

namespace hip {
namespace fusion_transform {
namespace {

constexpr int64_t kHeadSize = 128;

// An intermediate the rewrite erases, and the value that answers its shape
// queries: axisMap[i] is the axis of `source` equal to axis i, or -1 when
// axis i is not available from `source`.
struct ShapeAlias {
  mlir::Value value;
  mlir::Value source;
  llvm::SmallVector<int64_t, 4> axisMap;
};

// True when `alias.value` feeds exactly `consumer` and otherwise only
// tensor.dim queries the rewrite can answer from `alias.source`.
bool feedsOnly(const ShapeAlias &alias, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : alias.value.getUses()) {
    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner())) {
      std::optional<int64_t> index = dim.getConstantIndex();
      if (index && *index >= 0 &&
          *index < static_cast<int64_t>(alias.axisMap.size()) &&
          alias.axisMap[*index] >= 0)
        continue;
      return false;
    }
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

void retargetShapeQueries(mlir::PatternRewriter &rewriter,
                          const ShapeAlias &alias) {
  llvm::SmallVector<mlir::tensor::DimOp> dims;
  for (mlir::Operation *user : alias.value.getUsers())
    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user))
      dims.push_back(dim);
  for (mlir::tensor::DimOp dim : dims) {
    rewriter.setInsertionPoint(dim);
    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(
        dim, alias.source, alias.axisMap[*dim.getConstantIndex()]);
  }
}

bool isLastTwoAxisSwap(mlir::hip::TransposeOp transpose) {
  mlir::ArrayAttr perm = transpose.getPerm();
  if (perm.size() != 3)
    return false;
  const int64_t expected[3] = {0, 2, 1};
  for (int64_t i : llvm::seq<int64_t>(0, 3)) {
    auto value = mlir::dyn_cast<mlir::IntegerAttr>(perm[i]);
    if (!value || value.getInt() != expected[i])
      return false;
  }
  return true;
}

// The static last-axis offset of a column slice of `source`.
std::optional<int64_t> columnOffset(mlir::tensor::ExtractSliceOp slice,
                                    mlir::Value source, int64_t width) {
  if (!slice || slice.getSource() != source)
    return std::nullopt;
  int64_t offset = slice.getStaticOffsets().back();
  if (mlir::ShapedType::isDynamic(offset) ||
      !isColumnSlice(slice, source, offset, width))
    return std::nullopt;
  return offset;
}

// Every user of `value` sits after `anchor` in its block.
bool usedOnlyAfter(mlir::Value value, mlir::Operation *anchor) {
  return llvm::all_of(value.getUsers(), [&](mlir::Operation *user) {
    mlir::Operation *top = anchor->getBlock()->findAncestorOpInBlock(*user);
    return top && anchor->isBeforeInBlock(top);
  });
}

struct GatedDeltaNetConvNormFusion
    : public mlir::OpRewritePattern<mlir::hip::GatedRmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::GatedRmsNormOp norm,
                  mlir::PatternRewriter &rewriter) const override {
    if (norm->getNumResults() != 1 || norm.getActivation() != "sigmoid")
      return rewriter.notifyMatchFailure(norm, "not a sigmoid-gated norm");
    auto normWeightType =
        mlir::dyn_cast<mlir::RankedTensorType>(norm.getScale().getType());
    if (!normWeightType || normWeightType.getRank() != 1 ||
        normWeightType.getDimSize(0) != kHeadSize ||
        !normWeightType.getElementType().isF16())
      return rewriter.notifyMatchFailure(norm, "norm weight is not f16[128]");

    auto collapse =
        norm.getInput().getDefiningOp<mlir::tensor::CollapseShapeOp>();
    if (!collapse ||
        collapse.getReassociationIndices() !=
            llvm::SmallVector<mlir::ReassociationIndices>{{0}, {1}, {2, 3}})
      return rewriter.notifyMatchFailure(norm, "input is not a head collapse");
    auto gdn = collapse.getSrc().getDefiningOp<mlir::hip::GatedDeltaNetOp>();
    if (!gdn || gdn->getNumResults() != 2 ||
        collapse.getSrc() != gdn->getResult(0))
      return rewriter.notifyMatchFailure(norm, "input is not gated_delta_net");
    auto stateType =
        mlir::dyn_cast<mlir::RankedTensorType>(gdn.getInitialState().getType());
    if (!stateType || stateType.getRank() != 4 ||
        stateType.getDimSize(2) != kHeadSize ||
        stateType.getDimSize(3) != kHeadSize)
      return rewriter.notifyMatchFailure(norm, "head sizes must be 128");
    const int64_t numHeadsV = stateType.getDimSize(1);
    if (mlir::ShapedType::isDynamic(numHeadsV))
      return rewriter.notifyMatchFailure(norm, "dynamic value heads");
    const int64_t valueWidth = numHeadsV * kHeadSize;

    // The convolution, channels-last or bracketed by a transpose pair.
    mlir::hip::TransposeOp leading;
    mlir::hip::TransposeOp trailing;
    auto conv = gdn.getQkv().getDefiningOp<mlir::hip::CausalConvWithStateOp>();
    if (!conv) {
      trailing = gdn.getQkv().getDefiningOp<mlir::hip::TransposeOp>();
      if (!trailing || !isLastTwoAxisSwap(trailing))
        return rewriter.notifyMatchFailure(norm, "qkv is not a convolution");
      conv =
          trailing.getInput().getDefiningOp<mlir::hip::CausalConvWithStateOp>();
      if (!conv || conv.getChannelsLast())
        return rewriter.notifyMatchFailure(norm, "qkv is not a convolution");
      leading = conv.getInput().getDefiningOp<mlir::hip::TransposeOp>();
      if (!leading || !isLastTwoAxisSwap(leading))
        return rewriter.notifyMatchFailure(norm, "conv input not transposed");
    } else if (!conv.getChannelsLast()) {
      return rewriter.notifyMatchFailure(norm, "channels-first convolution");
    }
    if (conv->getNumResults() != 2 || conv.getNdim() != 1 ||
        (conv.getActivation() != "silu" && conv.getActivation() != "swish") ||
        !conv.getBias() || !conv.getPastState() || conv.getDilation() != 1 ||
        conv.getStateChannelsLast())
      return rewriter.notifyMatchFailure(norm, "unsupported convolution");
    auto weightType =
        mlir::dyn_cast<mlir::RankedTensorType>(conv.getWeight().getType());
    if (!weightType || !weightType.hasStaticShape() ||
        weightType.getRank() != 3 || weightType.getDimSize(1) != 1 ||
        !weightType.getElementType().isF16())
      return rewriter.notifyMatchFailure(norm, "conv weight is not [C, 1, K]");
    const int64_t channels = weightType.getDimSize(0);
    const int64_t kernelSize = weightType.getDimSize(2);
    // The channels-last convolution kernel's width envelope, as in the
    // transpose-pair fold.
    if (kernelSize < 2 || kernelSize > 128)
      return rewriter.notifyMatchFailure(norm, "unsupported kernel size");

    mlir::Value convInput = leading ? leading.getInput() : conv.getInput();
    auto convSlice = convInput.getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!convSlice)
      return rewriter.notifyMatchFailure(norm, "conv input is not a slice");
    mlir::Value proj = convSlice.getSource();
    auto projType = mlir::dyn_cast<mlir::RankedTensorType>(proj.getType());
    if (!projType || projType.getRank() != 3 || projType.isDynamicDim(2) ||
        !projType.getElementType().isF16() ||
        !isColumnSlice(convSlice, proj, 0, channels))
      return rewriter.notifyMatchFailure(norm, "conv input is not proj[:C]");

    auto zSlice = norm.getGate().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    std::optional<int64_t> zColumn = columnOffset(zSlice, proj, valueWidth);
    if (!zColumn)
      return rewriter.notifyMatchFailure(norm, "gate is not a column of proj");

    auto decayCast = gdn.getDecay().getDefiningOp<mlir::hip::CastOp>();
    auto betaCast = gdn.getBeta().getDefiningOp<mlir::hip::CastOp>();
    if (!decayCast || !betaCast || decayCast.getTo() != 1 ||
        betaCast.getTo() != 1)
      return rewriter.notifyMatchFailure(norm, "decay/beta are not f32 casts");
    auto decaySlice =
        decayCast.getInput().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    auto betaSlice =
        betaCast.getInput().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!decaySlice || !betaSlice)
      return rewriter.notifyMatchFailure(norm, "decay/beta are not slices");
    mlir::Value gates = decaySlice.getSource();
    auto gatesType = mlir::dyn_cast<mlir::RankedTensorType>(gates.getType());
    if (!gatesType || gatesType.getRank() != 3 || gatesType.isDynamicDim(2) ||
        !gatesType.getElementType().isF16())
      return rewriter.notifyMatchFailure(norm, "gates must be f16 [.., G]");
    std::optional<int64_t> decayColumn =
        columnOffset(decaySlice, gates, numHeadsV);
    std::optional<int64_t> betaColumn =
        columnOffset(betaSlice, gates, numHeadsV);
    if (!decayColumn || !betaColumn)
      return rewriter.notifyMatchFailure(norm, "decay/beta are not columns");

    mlir::Value ctx = norm.getCtx();
    if (conv.getCtx() != ctx || gdn.getCtx() != ctx ||
        decayCast.getCtx() != ctx || betaCast.getCtx() != ctx ||
        (leading && leading.getCtx() != ctx) ||
        (trailing && trailing.getCtx() != ctx))
      return rewriter.notifyMatchFailure(norm, "ops must share a context");

    // Single uses, answering batch and sequence shape queries from the
    // slice sources.
    const llvm::SmallVector<int64_t, 4> rows = {0, 1, -1};
    llvm::SmallVector<std::pair<ShapeAlias, mlir::Operation *>> chain;
    if (leading) {
      chain.push_back({{convSlice.getResult(), proj, rows}, leading});
      chain.push_back({{leading->getResult(0), proj, {0, -1, 1}}, conv});
      chain.push_back({{conv->getResult(0), proj, {0, -1, 1}}, trailing});
      chain.push_back({{trailing->getResult(0), proj, rows}, gdn});
    } else {
      chain.push_back({{convSlice.getResult(), proj, rows}, conv});
      chain.push_back({{conv->getResult(0), proj, rows}, gdn});
    }
    chain.push_back({{decaySlice.getResult(), gates, rows}, decayCast});
    chain.push_back({{betaSlice.getResult(), gates, rows}, betaCast});
    chain.push_back({{decayCast->getResult(0), gates, rows}, gdn});
    chain.push_back({{betaCast->getResult(0), gates, rows}, gdn});
    chain.push_back({{gdn->getResult(0), proj, {0, 1, -1, -1}}, collapse});
    chain.push_back({{collapse.getResult(), proj, rows}, norm});
    chain.push_back({{zSlice.getResult(), proj, rows}, norm});
    for (const auto &[alias, consumer] : chain)
      if (!feedsOnly(alias, consumer))
        return rewriter.notifyMatchFailure(norm,
                                           "intermediates have other uses");
    if (!usedOnlyAfter(conv->getResult(1), norm) ||
        !usedOnlyAfter(gdn->getResult(1), norm))
      return rewriter.notifyMatchFailure(norm, "state read before the norm");

    for (const auto &[alias, consumer] : chain)
      retargetShapeQueries(rewriter, alias);

    rewriter.setInsertionPoint(norm);
    mlir::Location loc = norm.getLoc();
    llvm::SmallVector<mlir::Value> rowDims;
    for (int64_t i : llvm::seq<int64_t>(0, 2))
      if (projType.isDynamicDim(i))
        rowDims.push_back(mlir::tensor::DimOp::create(rewriter, loc, proj, i));
    auto rowsTensor = [&](int64_t width) -> mlir::Value {
      const int64_t shape[3] = {projType.getDimSize(0), projType.getDimSize(1),
                                width};
      return mlir::tensor::EmptyOp::create(rewriter, loc, shape,
                                           projType.getElementType(), rowDims);
    };
    mlir::Value convIn = rowsTensor(channels);
    mlir::Value convOut = leading ? rowsTensor(channels) : conv.getOutput();
    mlir::Value zInit = rowsTensor(valueWidth);

    llvm::SmallVector<mlir::Value> outs = {norm.getOutput(),
                                           conv.getPresentState(),
                                           gdn.getFinalState(),
                                           convIn,
                                           convOut,
                                           decayCast.getOutput(),
                                           betaCast.getOutput(),
                                           gdn.getOutput(),
                                           zInit};
    llvm::SmallVector<mlir::Type> resultTypes;
    for (mlir::Value out : outs)
      resultTypes.push_back(out.getType());
    auto fused = mlir::hip::GatedDeltaNetConvNormOp::create(
        rewriter, loc, resultTypes, ctx, proj, conv.getWeight(), conv.getBias(),
        conv.getPastState(), gates, gdn.getInitialState(), gdn.getALog(),
        gdn.getDtBias(), norm.getScale(), outs,
        rewriter.getI64IntegerAttr(channels),
        rewriter.getI64IntegerAttr(*decayColumn),
        rewriter.getI64IntegerAttr(*betaColumn),
        rewriter.getI64IntegerAttr(*zColumn), gdn.getScaleAttr(),
        norm.getEpsilonAttr());

    llvm::SmallSetVector<mlir::Operation *, 3> deadInits;
    if (leading)
      for (mlir::Value init :
           {leading.getOutput(), conv.getOutput(), trailing.getOutput()})
        if (mlir::Operation *def = init.getDefiningOp())
          deadInits.insert(def);
    rewriter.replaceAllUsesWith(conv->getResult(1), fused->getResult(1));
    rewriter.replaceAllUsesWith(gdn->getResult(1), fused->getResult(2));
    rewriter.replaceOp(norm, fused->getResult(0));
    rewriter.eraseOp(collapse);
    rewriter.eraseOp(gdn);
    if (trailing)
      rewriter.eraseOp(trailing);
    rewriter.eraseOp(conv);
    if (leading)
      rewriter.eraseOp(leading);
    for (mlir::Operation *op :
         {convSlice.getOperation(), zSlice.getOperation(),
          decayCast.getOperation(), betaCast.getOperation(),
          decaySlice.getOperation(), betaSlice.getOperation()})
      rewriter.eraseOp(op);
    for (mlir::Operation *def : deadInits)
      if (mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

} // namespace

void populateGatedDeltaNetConvNormPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit) {
  patterns.add<GatedDeltaNetConvNormFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
