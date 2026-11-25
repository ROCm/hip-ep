/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipConstantData.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>

#include <cmath>
#include <optional>

namespace mlir::hip {
#define GEN_PASS_DEF_FOLDBATCHNORMPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Inference BatchNorm with constant parameters is the per-channel affine map
/// y = s * x + t, with s = scale / sqrt(var + eps) and t = B - mean * s. A
/// convolution is linear in its weights per output channel, so s moves into
/// the filter and t into the bias. Because the map distributes over a sum, the
/// same holds through an add of two convolutions: both filters are scaled and
/// t is added once, to one of the two biases.
///
/// Before:
///   %a = hip.conv(%ctx) ins(%x, %wa : ..., tensor<128x128x3x3xf32>) ...
///   %b = hip.conv(%ctx) ins(%y, %wb : ..., tensor<128x384x3x3xf32>) ...
///   %s = hip.add(%ctx) ins(%a, %b : ...) ...
///   %n = hip.batch_norm(%ctx) ins(%s, %scale, %B, %mean, %var : ...) ...
///   %r = hip.max(%ctx) ins(%n, %zero : ...) ...
/// After:
///   %wa2 = hip.constant {value = dense<s * wa>} : tensor<128x128x3x3xf32>
///   %t   = hip.constant {value = dense<t>} : tensor<128xf32>
///   %a = hip.conv(%ctx) ins(%x, %wa2, %t : ...) ...
///   %wb2 = hip.constant {value = dense<s * wb>} : tensor<128x384x3x3xf32>
///   %b = hip.conv(%ctx) ins(%y, %wb2 : ...) ...
///   %s = hip.add(%ctx) ins(%a, %b : ...) ...
///   %r = hip.max(%ctx) ins(%s, %zero : ...) ...
///
/// Every intermediate result must have the batch norm as its only consumer,
/// since the rescaled convolutions no longer produce the original values.
struct FoldBatchNormIntoConv : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern<BatchNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override;
};

bool isFoldableFloat(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

/// Per-channel values of a 1-D constant of length `channels`.
std::optional<SmallVector<double>> readVector(Value value, int64_t channels) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!constant || !type || !type.hasStaticShape() || type.getRank() != 1 ||
      type.getDimSize(0) != channels || !isFoldableFloat(type.getElementType()))
    return std::nullopt;
  FailureOr<DenseElementsAttr> data = readConstantData(constant, type);
  if (failed(data))
    return std::nullopt;
  SmallVector<double> values;
  values.reserve(channels);
  for (const APFloat &v : data->getValues<APFloat>())
    values.push_back(v.convertToDouble());
  return values;
}

/// `type` filled with `values`, rounded to its element type.
DenseElementsAttr buildFloatAttr(RankedTensorType type,
                                 ArrayRef<double> values) {
  Type elementType = type.getElementType();
  if (elementType.isF32()) {
    SmallVector<float> narrowed(values.begin(), values.end());
    return DenseElementsAttr::get(type, ArrayRef<float>(narrowed));
  }
  if (elementType.isF64())
    return DenseElementsAttr::get(type, values);
  const llvm::fltSemantics &semantics =
      cast<FloatType>(elementType).getFloatSemantics();
  SmallVector<APFloat> rounded;
  rounded.reserve(values.size());
  for (double v : values) {
    APFloat f(v);
    bool losesInfo = false;
    f.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
    rounded.push_back(f);
  }
  return DenseElementsAttr::get(type, rounded);
}

/// A convolution that the batch norm can be folded into: its only consumer is
/// on the path to the batch norm, and its filter and bias (if any) are
/// readable here.
struct FoldableConv {
  ConvOp op;
  DenseElementsAttr weights;
  std::optional<SmallVector<double>> bias;
};

LogicalResult
FoldBatchNormIntoConv::matchAndRewrite(BatchNormOp op,
                                       PatternRewriter &rewriter) const {
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() < 3 ||
      op.getInput().getType() != resultType ||
      !isFoldableFloat(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op, "expected static float NC... input");
  const int64_t channels = resultType.getDimSize(1);

  // Collect the convolutions feeding the batch norm, directly or through one
  // add whose operands are both convolutions of the batch norm's shape.
  SmallVector<ConvOp, 2> convs;
  AddOp add;
  Value input = op.getInput();
  if (!input.hasOneUse())
    return rewriter.notifyMatchFailure(op, "input has other consumers");
  if (auto conv = input.getDefiningOp<ConvOp>()) {
    convs.push_back(conv);
  } else if ((add = input.getDefiningOp<AddOp>())) {
    if (add->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor-mode add");
    for (Value operand : {add.getLhs(), add.getRhs()}) {
      auto conv = operand.getDefiningOp<ConvOp>();
      if (!conv || operand.getType() != resultType || !operand.hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "add operand is not a single-use conv of the same shape");
      convs.push_back(conv);
    }
  } else {
    return rewriter.notifyMatchFailure(op, "input is not a conv or an add");
  }

  SmallVector<FoldableConv, 2> foldable;
  for (ConvOp conv : convs) {
    if (conv->getNumResults() != 1 ||
        conv->getResult(0).getType() != resultType)
      return rewriter.notifyMatchFailure(op, "conv result does not match");
    auto weightType = dyn_cast<RankedTensorType>(conv.getWeights().getType());
    auto weightConst = conv.getWeights().getDefiningOp<ConstantOp>();
    if (!weightType || !weightConst || !weightType.hasStaticShape() ||
        weightType.getDimSize(0) != channels ||
        weightType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(op, "conv filter is not a constant");
    FailureOr<DenseElementsAttr> weights =
        readConstantData(weightConst, weightType);
    if (failed(weights))
      return rewriter.notifyMatchFailure(op, "conv filter is not readable");
    FoldableConv entry{conv, *weights, std::nullopt};
    if (Value bias = conv.getBias()) {
      entry.bias = readVector(bias, channels);
      if (!entry.bias)
        return rewriter.notifyMatchFailure(op, "conv bias is not readable");
    }
    foldable.push_back(std::move(entry));
  }

  std::optional<SmallVector<double>> gamma =
      readVector(op.getScale(), channels);
  std::optional<SmallVector<double>> beta = readVector(op.getBias(), channels);
  std::optional<SmallVector<double>> mean = readVector(op.getMean(), channels);
  std::optional<SmallVector<double>> var =
      readVector(op.getVariance(), channels);
  if (!gamma || !beta || !mean || !var)
    return rewriter.notifyMatchFailure(op,
                                       "batch norm parameters not readable");

  const double epsilon = op.getEpsilon().convertToDouble();
  SmallVector<double> scale(channels), shift(channels);
  for (int64_t c : llvm::seq<int64_t>(channels)) {
    scale[c] = (*gamma)[c] / std::sqrt((*var)[c] + epsilon);
    shift[c] = (*beta)[c] - (*mean)[c] * scale[c];
  }

  auto biasType =
      RankedTensorType::get({channels}, resultType.getElementType());
  SmallVector<Value, 2> replacements;
  for (auto [index, entry] : llvm::enumerate(foldable)) {
    ConvOp conv = entry.op;
    auto weightType = cast<RankedTensorType>(conv.getWeights().getType());
    const int64_t perChannel = weightType.getNumElements() / channels;

    SmallVector<double> weights;
    weights.reserve(weightType.getNumElements());
    int64_t flat = 0;
    for (const APFloat &w : entry.weights.getValues<APFloat>())
      weights.push_back(w.convertToDouble() * scale[flat++ / perChannel]);

    // The shift is added once: through an add, the second conv's output is
    // summed with the first's, so a shift on both would count it twice.
    std::optional<SmallVector<double>> bias;
    if (entry.bias || index == 0) {
      bias.emplace(channels, 0.0);
      for (int64_t c : llvm::seq<int64_t>(channels))
        (*bias)[c] = (entry.bias ? (*entry.bias)[c] * scale[c] : 0.0) +
                     (index == 0 ? shift[c] : 0.0);
    }

    rewriter.setInsertionPoint(conv);
    Location loc = conv.getLoc();
    Value newWeights = ConstantOp::create(rewriter, loc, weightType,
                                          buildFloatAttr(weightType, weights))
                           .getResult();
    SmallVector<Value> operands{conv.getCtx(), conv.getInput(), newWeights};
    if (bias)
      operands.push_back(ConstantOp::create(rewriter, loc, biasType,
                                            buildFloatAttr(biasType, *bias))
                             .getResult());
    operands.push_back(conv.getOutput());
    auto folded = ConvOp::create(rewriter, loc, TypeRange{resultType}, operands,
                                 conv->getAttrs());
    replacements.push_back(folded->getResult(0));
  }

  Value result;
  if (add) {
    rewriter.modifyOpInPlace(add, [&] {
      add.getLhsMutable().assign(replacements[0]);
      add.getRhsMutable().assign(replacements[1]);
    });
    result = add->getResult(0);
  } else {
    result = replacements[0];
  }
  rewriter.replaceOp(op, result);
  for (ConvOp conv : convs)
    rewriter.eraseOp(conv);
  return success();
}

struct FoldBatchNormPass
    : public impl::FoldBatchNormPassBase<FoldBatchNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBatchNormIntoConv>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
