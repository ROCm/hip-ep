/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- hyper_connection_mixer.cpp - pre-mix gate path to one op -----------===//
//
// A hyper-connection boundary projects the normalized streams once and splits
// the result: the leading columns go through a scaled Swish and a second
// 4-bit projection into the pre-mix gate, the trailing `branches` columns are
// the next injection gate. Decode spends most of that path on launches and
// strided copies, so the whole path becomes one op the runtime can run as a
// single kernel for few rows.
//
//   Before:
//     %p = hip.matmul_nbits(%ctx) ins(%x, ...) outs(%e0) : [.., K + C]
//     %a = tensor.extract_slice %p[0, 0, 0] [%d0, %d1, K] [1, 1, 1]
//     %g = tensor.extract_slice %p[0, 0, K] [%d0, %d1, C] [1, 1, 1]
//     %c = hip.constant {value = dense<s> : tensor<f16>}
//     %s = hip.mul(%ctx) ins(%a, %c) outs(%e1) -> [.., K]
//     %w = hip.swish(%ctx) ins(%s) outs(%e2) {alpha = al}
//     %m = hip.matmul_nbits(%ctx) ins(%w, %B, %S) outs(%e3) : [.., C * H]
//     %y = hip.hyper_connection_pre_mix(%ctx) ins(%streams, %m) outs(%e4)
//              {flattened = 1, gate_layout = "feature", gate_sigmoid = 1}
//   After:
//     %e5 = tensor.empty(%d0, %d1) : [.., C]
//     %r:4 = hip.hyper_connection_mixer(%ctx) ins(%streams, %p, %B, %S)
//              outs(%e4, %e5, %e2, %e3) {act_scale = s, swish_alpha = al}
//     uses of %y -> %r#0, uses of %g -> %r#1
//
// The intermediates must have no other value use. A tensor.dim of one of
// them on a leading axis is retargeted to %p, whose leading axes they share.
//
//===----------------------------------------------------------------------===//

#include "hyper_connection_mixer.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/Sequence.h"

namespace hip {
namespace fusion_transform {
namespace {

// True when every use is `consumer` or a tensor.dim of a leading axis, which
// the rewrite can answer from the projection instead.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer,
                       int64_t rank) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner())) {
      std::optional<int64_t> index = dim.getConstantIndex();
      if (index && *index >= 0 && *index < rank - 1)
        continue;
      return false;
    }
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

// The splat value of a one-element f16 hip.constant.
std::optional<float> splatF16Scalar(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::hip::ConstantOp>();
  if (!constant || !constant.getValue())
    return std::nullopt;
  auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(*constant.getValue());
  if (!dense || !dense.isSplat() || dense.getNumElements() != 1 ||
      !dense.getElementType().isF16())
    return std::nullopt;
  return dense.getSplatValue<llvm::APFloat>().convertToFloat();
}

// A rank-preserving, unit-stride slice of `source` that keeps every leading
// axis whole and takes `width` columns of the last axis from `offset`.
bool isColumnSlice(mlir::tensor::ExtractSliceOp slice, mlir::Value source,
                   int64_t offset, int64_t width) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType = slice.getResultType();
  if (slice.getSource() != source || !sourceType ||
      resultType.getRank() != sourceType.getRank())
    return false;
  const int64_t rank = sourceType.getRank();
  auto offsets = slice.getMixedOffsets();
  auto sizes = slice.getMixedSizes();
  auto strides = slice.getMixedStrides();
  for (int64_t i : llvm::seq<int64_t>(0, rank)) {
    if (mlir::getConstantIntValue(strides[i]) != std::optional<int64_t>(1))
      return false;
    const bool last = i + 1 == rank;
    if (mlir::getConstantIntValue(offsets[i]) !=
        std::optional<int64_t>(last ? offset : 0))
      return false;
    if (last) {
      if (mlir::getConstantIntValue(sizes[i]) != std::optional<int64_t>(width))
        return false;
      continue;
    }
    if (std::optional<int64_t> size = mlir::getConstantIntValue(sizes[i])) {
      if (sourceType.isDynamicDim(i) || *size != sourceType.getDimSize(i))
        return false;
      continue;
    }
    auto dim = mlir::dyn_cast<mlir::Value>(sizes[i])
                   .getDefiningOp<mlir::tensor::DimOp>();
    if (!dim || dim.getSource() != source ||
        dim.getConstantIndex() != std::optional<int64_t>(i))
      return false;
  }
  return true;
}

void retargetShapeQueries(mlir::PatternRewriter &rewriter, mlir::Value from,
                          mlir::Value to) {
  rewriter.replaceUsesWithIf(from, to, [](mlir::OpOperand &operand) {
    return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
  });
}

struct HyperConnectionMixerFusion
    : public mlir::OpRewritePattern<mlir::hip::HyperConnectionPreMixOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::HyperConnectionPreMixOp pre,
                  mlir::PatternRewriter &rewriter) const override {
    if (pre->getNumResults() != 1 || pre.getFlattened() != 1 ||
        pre.getGateLayout() != "feature" || pre.getGateSigmoid() != 1)
      return rewriter.notifyMatchFailure(pre, "not a sigmoid feature pre-mix");
    const int64_t branches = pre.getBranches();
    const int64_t hidden = pre.getHidden();
    auto streamsType =
        mlir::dyn_cast<mlir::RankedTensorType>(pre.getStreams().getType());
    if (!streamsType || !streamsType.getElementType().isF16())
      return rewriter.notifyMatchFailure(pre, "streams must be f16 tensors");
    const int64_t rank = streamsType.getRank();

    auto mix = pre.getPreMix().getDefiningOp<mlir::hip::MatMulNBitsOp>();
    if (!mix || mix->getNumResults() != 1 || mix.getBits() != 4 ||
        mix.getZeroPoints() || mix.getGIdx() || mix.getBias() ||
        mix.getScaleElemSize() != 2 ||
        static_cast<int64_t>(mix.getN()) != branches * hidden)
      return rewriter.notifyMatchFailure(pre, "gate is not a 4-bit projection");
    const int64_t actWidth = mix.getK();
    const int64_t blockSize = mix.getBlockSize();
    if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0 ||
        actWidth % blockSize != 0)
      return rewriter.notifyMatchFailure(pre, "unsupported block size");

    auto swish = mix.getA().getDefiningOp<mlir::hip::SwishOp>();
    if (!swish || swish->getNumResults() != 1)
      return rewriter.notifyMatchFailure(pre, "projection input is not swish");
    auto mul = swish.getInput().getDefiningOp<mlir::hip::MulOp>();
    if (!mul || mul->getNumResults() != 1)
      return rewriter.notifyMatchFailure(pre, "swish input is not a mul");

    std::optional<float> actScale = splatF16Scalar(mul.getRhs());
    mlir::Value sliced = mul.getLhs();
    if (!actScale) {
      actScale = splatF16Scalar(mul.getLhs());
      sliced = mul.getRhs();
    }
    auto slice = sliced.getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!actScale || !slice)
      return rewriter.notifyMatchFailure(pre, "mul is not slice * scalar");
    mlir::Value proj = slice.getSource();
    auto projType = mlir::dyn_cast<mlir::RankedTensorType>(proj.getType());
    if (!projType || projType.getRank() != rank ||
        projType.getElementType() != streamsType.getElementType() ||
        projType.isDynamicDim(rank - 1) ||
        projType.getDimSize(rank - 1) != actWidth + branches ||
        !isColumnSlice(slice, proj, 0, actWidth))
      return rewriter.notifyMatchFailure(pre, "not a column split of proj");
    if (mul->getResult(0).getType() != slice.getResultType() ||
        swish->getResult(0).getType() != slice.getResultType())
      return rewriter.notifyMatchFailure(pre, "the act path broadcasts");

    if (mul.getCtx() != pre.getCtx() || swish.getCtx() != pre.getCtx() ||
        mix.getHandle() != pre.getCtx())
      return rewriter.notifyMatchFailure(pre, "ops must share a context");
    if (!hasSingleValueUse(slice.getResult(), mul, rank) ||
        !hasSingleValueUse(mul->getResult(0), swish, rank) ||
        !hasSingleValueUse(swish->getResult(0), mix, rank) ||
        !hasSingleValueUse(mix->getResult(0), pre, rank))
      return rewriter.notifyMatchFailure(pre, "intermediates have other uses");

    llvm::SmallVector<int64_t> gateShape(projType.getShape());
    gateShape.back() = branches;
    auto gateType =
        mlir::RankedTensorType::get(gateShape, projType.getElementType());

    // The injection-gate columns, if something reads them after the pre-mix.
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> gateSlices;
    for (mlir::Operation *user : proj.getUsers()) {
      auto candidate = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user);
      if (!candidate || candidate.getResultType() != gateType ||
          !isColumnSlice(candidate, proj, actWidth, branches))
        continue;
      bool later = llvm::all_of(candidate->getUsers(), [&](mlir::Operation *u) {
        mlir::Operation *top = pre->getBlock()->findAncestorOpInBlock(*u);
        return top && pre->isBeforeInBlock(top);
      });
      if (later)
        gateSlices.push_back(candidate);
    }

    rewriter.setInsertionPoint(pre);
    mlir::Location loc = pre.getLoc();
    llvm::SmallVector<mlir::Value> gateDims;
    for (int64_t i : llvm::seq<int64_t>(0, rank - 1))
      if (projType.isDynamicDim(i))
        gateDims.push_back(mlir::tensor::DimOp::create(rewriter, loc, proj, i));
    mlir::Value gateInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, gateShape, projType.getElementType(), gateDims);

    const mlir::Value intermediates[] = {slice.getResult(), mul->getResult(0),
                                         swish->getResult(0),
                                         mix->getResult(0)};
    for (mlir::Value value : intermediates)
      retargetShapeQueries(rewriter, value, proj);

    mlir::Operation *mulInit = mul.getOutput().getDefiningOp();
    mlir::Operation *scaleDef = mul.getLhs() == sliced
                                    ? mul.getRhs().getDefiningOp()
                                    : mul.getLhs().getDefiningOp();
    llvm::SmallVector<mlir::Type> resultTypes{
        pre->getResult(0).getType(), gateType, swish->getResult(0).getType(),
        mix->getResult(0).getType()};
    auto mixer = mlir::hip::HyperConnectionMixerOp::create(
        rewriter, loc, resultTypes, pre.getCtx(), pre.getStreams(), proj,
        mix.getB(), mix.getScales(),
        mlir::ValueRange{pre.getOutput(), gateInit, swish.getOutput(),
                         mix.getOutput()},
        rewriter.getI64IntegerAttr(branches),
        rewriter.getI64IntegerAttr(hidden),
        rewriter.getI64IntegerAttr(actWidth),
        rewriter.getI64IntegerAttr(blockSize),
        rewriter.getF32FloatAttr(*actScale),
        rewriter.getF32FloatAttr(
            static_cast<float>(swish.getAlpha().convertToDouble())),
        pre.getReductionScaleAttr());

    rewriter.replaceOp(pre, mixer->getResult(0));
    for (mlir::tensor::ExtractSliceOp gateSlice : gateSlices)
      rewriter.replaceOp(gateSlice, mixer->getResult(1));
    rewriter.eraseOp(mix);
    rewriter.eraseOp(swish);
    rewriter.eraseOp(mul);
    rewriter.eraseOp(slice);
    for (mlir::Operation *def : {mulInit, scaleDef})
      if (def && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

} // namespace

void populateHyperConnectionMixerPattern(mlir::RewritePatternSet &patterns,
                                         mlir::PatternBenefit benefit) {
  patterns.add<HyperConnectionMixerFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
