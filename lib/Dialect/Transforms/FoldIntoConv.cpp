/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipConstantData.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <cmath>
#include <optional>

namespace mlir::hip {
#define GEN_PASS_DEF_FOLDINTOCONVPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// The per-channel affine map y[n, c, ...] = scale[c] * x[n, c, ...] +
/// shift[c] that a consumer applies to a convolution's output.
struct ChannelAffine {
  SmallVector<double> scale;
  SmallVector<double> shift;
};

bool isFoldableFloat(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

std::optional<SmallVector<double>> readFloats(ConstantOp constant,
                                              RankedTensorType type) {
  FailureOr<DenseElementsAttr> data = readConstantData(constant, type);
  if (failed(data))
    return std::nullopt;
  SmallVector<double> values;
  values.reserve(type.getNumElements());
  for (const APFloat &v : data->getValues<APFloat>())
    values.push_back(v.convertToDouble());
  return values;
}

/// Per-channel values of a 1-D constant of length `channels`.
std::optional<SmallVector<double>> readVector(Value value, int64_t channels) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!constant || !type || !type.hasStaticShape() || type.getRank() != 1 ||
      type.getDimSize(0) != channels || !isFoldableFloat(type.getElementType()))
    return std::nullopt;
  return readFloats(constant, type);
}

/// Per-channel values of a constant that broadcasts against `resultType`
/// ([N, C, ...]) along the channel axis only: its rank is at most the result's
/// and, right-aligned, every dim is 1 except the one on the channel axis,
/// which is 1 or C. So [C, 1, 1], [1, C, 1, 1], [1] and a scalar qualify, and
/// broadcasting it never changes the result shape.
std::optional<SmallVector<double>>
readChannelBroadcast(Value value, RankedTensorType resultType) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!constant || !type || !type.hasStaticShape() ||
      type.getRank() > resultType.getRank() ||
      type.getElementType() != resultType.getElementType())
    return std::nullopt;
  const int64_t channels = resultType.getDimSize(1);
  // Axis of `type` that lines up with the result's channel axis; negative
  // when the constant's rank does not reach it.
  const int64_t channelAxis = 1 - (resultType.getRank() - type.getRank());
  for (int64_t axis : llvm::seq<int64_t>(type.getRank())) {
    const int64_t size = type.getDimSize(axis);
    if (size != 1 && !(axis == channelAxis && size == channels))
      return std::nullopt;
  }
  std::optional<SmallVector<double>> values = readFloats(constant, type);
  if (values && values->size() == 1)
    values->assign(channels, values->front());
  return values;
}

/// `type` filled with `values` rounded to its element type, or std::nullopt
/// if a value does not round to a finite number: a fold must not turn a
/// finite filter into infinities.
std::optional<DenseElementsAttr> buildFiniteFloatAttr(RankedTensorType type,
                                                      ArrayRef<double> values) {
  Type elementType = type.getElementType();
  if (elementType.isF64()) {
    if (!llvm::all_of(values, [](double v) { return std::isfinite(v); }))
      return std::nullopt;
    return DenseElementsAttr::get(type, values);
  }
  const llvm::fltSemantics &semantics =
      cast<FloatType>(elementType).getFloatSemantics();
  SmallVector<APFloat> rounded;
  rounded.reserve(values.size());
  for (double v : values) {
    APFloat f(v);
    bool losesInfo = false;
    f.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
    if (!f.isFinite())
      return std::nullopt;
    rounded.push_back(f);
  }
  return DenseElementsAttr::get(type, rounded);
}

/// True if `value` is read only as a DPS input of `user`, apart from
/// tensor.dim ops, which read only the shape that the fold preserves. A use as
/// `user`'s init would keep the original value alive after the rewrite.
bool feedsOnly(Value value, Operation *user) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(user);
  return dps && llvm::all_of(value.getUses(), [&](OpOperand &use) {
           if (isa<tensor::DimOp>(use.getOwner()))
             return true;
           return use.getOwner() == user && !dps.isDpsInit(&use);
         });
}

/// The new filter and bias payloads of one convolution; a null attribute
/// keeps the conv's current operand.
struct FoldedConv {
  DenseElementsAttr weights;
  DenseElementsAttr bias;
};

/// Output-side fold. A convolution is linear in its weights per output
/// channel, so a per-channel affine map applied to its output moves into the
/// conv: scale[c] multiplies output channel c of the filter and the bias,
/// and shift[c] is added to the bias. Because the map distributes over a sum,
/// the same holds through an add of two convolutions: both filters and biases
/// are scaled and the shift is added once, to one of the two biases.
///
/// `consumer` applies the map to `input`; `readAffine` reads its constant
/// parameters for the consumer's result type, or returns std::nullopt if they
/// are not readable. The patterns below are the sources of such a map. Like
/// any weight fold, a zero scale makes the channel exactly the shift even
/// where the unfolded conv output would have been inf or NaN.
///
/// Before:
///   %a = hip.conv(%ctx) ins(%x, %wa : ..., tensor<128x128x3x3xf32>) ...
///   %b = hip.conv(%ctx) ins(%y, %wb : ..., tensor<128x384x3x3xf32>) ...
///   %s = hip.add(%ctx) ins(%a, %b : ...) ...
///   %n = hip.batch_norm(%ctx) ins(%s, %gamma, %beta, %mean, %var : ...) ...
///   %r = hip.max(%ctx) ins(%n, %zero : ...) ...
/// After:
///   %wa2 = hip.constant {value = dense<scale * wa>} : tensor<128x128x3x3xf32>
///   %t   = hip.constant {value = dense<shift>} : tensor<128xf32>
///   %a = hip.conv(%ctx) ins(%x, %wa2, %t : ...) ...
///   %wb2 = hip.constant {value = dense<scale * wb>} : tensor<128x384x3x3xf32>
///   %b = hip.conv(%ctx) ins(%y, %wb2 : ...) ...
///   %s = hip.add(%ctx) ins(%a, %b : ...) ...
///   %r = hip.max(%ctx) ins(%s, %zero : ...) ...
///
/// Only the channel dim must be static: batch and spatial dims may be dynamic,
/// and each rewritten conv keeps its init and result type. Every intermediate
/// result must feed nothing but the next op on the path to the consumer, since
/// the rescaled convolutions no longer produce the original values. Shape
/// queries (tensor.dim) are the exception: the fold preserves every shape, so
/// they are moved to the rewritten conv.
LogicalResult foldChannelAffineIntoConv(
    PatternRewriter &rewriter, Operation *consumer, Value input,
    function_ref<std::optional<ChannelAffine>(RankedTensorType)> readAffine) {
  if (consumer->getNumResults() != 1)
    return rewriter.notifyMatchFailure(consumer, "expected tensor mode");
  auto resultType =
      dyn_cast<RankedTensorType>(consumer->getResult(0).getType());
  if (!resultType || resultType.getRank() < 3 || resultType.isDynamicDim(1) ||
      input.getType() != resultType ||
      !isFoldableFloat(resultType.getElementType()))
    return rewriter.notifyMatchFailure(
        consumer, "expected float NC... input with a static channel dim, "
                  "of the result type");
  const int64_t channels = resultType.getDimSize(1);

  // Collect the convolutions producing the input, directly or through one add
  // whose operands are both convolutions of the result type.
  SmallVector<ConvOp, 2> convs;
  AddOp add;
  if (!feedsOnly(input, consumer))
    return rewriter.notifyMatchFailure(consumer, "input has other consumers");
  if (auto conv = input.getDefiningOp<ConvOp>()) {
    convs.push_back(conv);
  } else if ((add = input.getDefiningOp<AddOp>())) {
    if (add->getNumResults() != 1)
      return rewriter.notifyMatchFailure(consumer, "expected tensor-mode add");
    for (Value operand : {add.getLhs(), add.getRhs()}) {
      auto conv = operand.getDefiningOp<ConvOp>();
      if (!conv || operand.getType() != resultType || !feedsOnly(operand, add))
        return rewriter.notifyMatchFailure(
            consumer, "add operand is not a single-use conv of the same type");
      convs.push_back(conv);
    }
    if (convs[0] == convs[1])
      return rewriter.notifyMatchFailure(consumer, "add of a conv to itself");
  } else {
    return rewriter.notifyMatchFailure(consumer,
                                       "input is not a conv or an add");
  }

  auto biasType =
      RankedTensorType::get({channels}, resultType.getElementType());
  for (ConvOp conv : convs) {
    if (conv->getNumResults() != 1 ||
        conv->getResult(0).getType() != resultType)
      return rewriter.notifyMatchFailure(consumer,
                                         "conv result does not match");
    auto weightType = dyn_cast<RankedTensorType>(conv.getWeights().getType());
    if (!weightType || !weightType.hasStaticShape() ||
        weightType.getDimSize(0) != channels ||
        weightType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(consumer, "unexpected conv filter");
    if (Value bias = conv.getBias(); bias && bias.getType() != biasType)
      return rewriter.notifyMatchFailure(consumer, "unexpected conv bias");
  }

  std::optional<ChannelAffine> affine = readAffine(resultType);
  if (!affine)
    return rewriter.notifyMatchFailure(consumer,
                                       "per-channel parameters not readable");
  const bool scales =
      llvm::any_of(affine->scale, [](double s) { return s != 1.0; });
  const bool shifts =
      llvm::any_of(affine->shift, [](double t) { return t != 0.0; });

  // Compute every new payload before touching the IR, so a filter or bias that
  // is not readable or would round to infinity leaves the IR unchanged. A
  // filter or bias is read only when it changes.
  SmallVector<FoldedConv, 2> folded;
  for (auto [index, conv] : llvm::enumerate(convs)) {
    FoldedConv &out = folded.emplace_back();
    if (scales) {
      auto weightType = cast<RankedTensorType>(conv.getWeights().getType());
      auto weightConst = conv.getWeights().getDefiningOp<ConstantOp>();
      std::optional<SmallVector<double>> weights =
          weightConst ? readFloats(weightConst, weightType) : std::nullopt;
      if (!weights)
        return rewriter.notifyMatchFailure(consumer,
                                           "conv filter is not readable");
      const int64_t perChannel = weightType.getNumElements() / channels;
      for (auto [flat, w] : llvm::enumerate(*weights))
        w *= affine->scale[flat / perChannel];
      std::optional<DenseElementsAttr> attr =
          buildFiniteFloatAttr(weightType, *weights);
      if (!attr)
        return rewriter.notifyMatchFailure(consumer, "folded filter overflows");
      out.weights = *attr;
    }

    // The shift is added once: through an add, the second conv's output is
    // summed with the first's, so a shift on both would count it twice.
    const bool addsShift = shifts && index == 0;
    if (!addsShift && !(scales && conv.getBias()))
      continue;
    std::optional<SmallVector<double>> oldBias;
    if (Value bias = conv.getBias()) {
      oldBias = readVector(bias, channels);
      if (!oldBias)
        return rewriter.notifyMatchFailure(consumer,
                                           "conv bias is not readable");
    }
    SmallVector<double> bias(channels, 0.0);
    for (int64_t c : llvm::seq<int64_t>(channels))
      bias[c] = (oldBias ? (*oldBias)[c] * affine->scale[c] : 0.0) +
                (addsShift ? affine->shift[c] : 0.0);
    std::optional<DenseElementsAttr> attr =
        buildFiniteFloatAttr(biasType, bias);
    if (!attr)
      return rewriter.notifyMatchFailure(consumer, "folded bias overflows");
    out.bias = *attr;
  }

  SmallVector<Value, 2> replacements;
  for (auto [conv, payload] : llvm::zip_equal(convs, folded)) {
    if (!payload.weights && !payload.bias) {
      replacements.push_back(conv->getResult(0));
      continue;
    }
    rewriter.setInsertionPoint(conv);
    Location loc = conv.getLoc();
    Value weights = conv.getWeights();
    if (payload.weights)
      weights = ConstantOp::create(rewriter, loc,
                                   cast<RankedTensorType>(weights.getType()),
                                   payload.weights)
                    .getResult();
    SmallVector<Value> operands{conv.getCtx(), conv.getInput(), weights};
    if (payload.bias)
      operands.push_back(
          ConstantOp::create(rewriter, loc, biasType, payload.bias)
              .getResult());
    else if (Value bias = conv.getBias())
      operands.push_back(bias);
    operands.push_back(conv.getOutput());
    auto rewritten = ConvOp::create(rewriter, loc, TypeRange{resultType},
                                    operands, conv->getAttrs());
    // The remaining tensor.dim users read a shape the rewrite preserves.
    rewriter.replaceAllUsesExcept(conv->getResult(0), rewritten->getResult(0),
                                  add ? add.getOperation() : consumer);
    replacements.push_back(rewritten->getResult(0));
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
  rewriter.replaceOp(consumer, result);
  for (auto [conv, payload] : llvm::zip_equal(convs, folded))
    if (payload.weights || payload.bias)
      rewriter.eraseOp(conv);
  return success();
}

/// Source: an inference hip.batch_norm with constant parameters,
/// scale = gamma / sqrt(var + eps), shift = beta - mean * scale.
struct BatchNormIntoConv : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern<BatchNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp op,
                                PatternRewriter &rewriter) const override {
    return foldChannelAffineIntoConv(
        rewriter, op, op.getInput(),
        [&](RankedTensorType type) -> std::optional<ChannelAffine> {
          const int64_t channels = type.getDimSize(1);
          std::optional<SmallVector<double>> gamma =
              readVector(op.getScale(), channels);
          std::optional<SmallVector<double>> beta =
              readVector(op.getBias(), channels);
          std::optional<SmallVector<double>> mean =
              readVector(op.getMean(), channels);
          std::optional<SmallVector<double>> var =
              readVector(op.getVariance(), channels);
          if (!gamma || !beta || !mean || !var)
            return std::nullopt;
          const double epsilon = op.getEpsilon().convertToDouble();
          ChannelAffine affine{SmallVector<double>(channels),
                               SmallVector<double>(channels)};
          for (int64_t c : llvm::seq<int64_t>(channels)) {
            affine.scale[c] = (*gamma)[c] / std::sqrt((*var)[c] + epsilon);
            affine.shift[c] = (*beta)[c] - (*mean)[c] * affine.scale[c];
          }
          return affine;
        });
  }
};

/// Source: a hip.mul (`IsScale`: scale = the constant, shift = 0) or a hip.add
/// (scale = 1, shift = the constant) of a conv output and a constant that
/// broadcasts along the channel axis only ([C, 1, 1], [1, C, 1, 1], or a
/// scalar), e.g. a layer scale. A mul followed by an add folds in two steps,
/// since the first fold leaves a conv that feeds the add directly.
template <typename BinaryOp, bool IsScale>
struct ChannelConstantIntoConv : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.getLhs();
    Value constant = op.getRhs();
    if (input.getDefiningOp<ConstantOp>())
      std::swap(input, constant);
    if (!constant.getDefiningOp<ConstantOp>())
      return rewriter.notifyMatchFailure(op, "no constant operand");
    return foldChannelAffineIntoConv(
        rewriter, op, input,
        [&](RankedTensorType type) -> std::optional<ChannelAffine> {
          std::optional<SmallVector<double>> values =
              readChannelBroadcast(constant, type);
          if (!values)
            return std::nullopt;
          const int64_t channels = type.getDimSize(1);
          if constexpr (IsScale)
            return ChannelAffine{std::move(*values),
                                 SmallVector<double>(channels, 0.0)};
          return ChannelAffine{SmallVector<double>(channels, 1.0),
                               std::move(*values)};
        });
  }
};

struct FoldIntoConvPass : public impl::FoldIntoConvPassBase<FoldIntoConvPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<BatchNormIntoConv,
                 ChannelConstantIntoConv<MulOp, /*IsScale=*/true>,
                 ChannelConstantIntoConv<AddOp, /*IsScale=*/false>>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
