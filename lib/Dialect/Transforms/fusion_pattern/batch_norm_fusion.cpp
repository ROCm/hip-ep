/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- batch_norm_fusion.cpp - elementwise batch norm to hip.batch_norm ---===//
//
// Some frontends, such as torch-mlir, decompose an inference batch
// normalization into primitive HIP ops. Fuse them back together, so that the
// normalization can be described as a singular hip.batch_norm.
//
//   Before:
//     %ve = hip.add(%ctx) ins(%variance, %eps)         // variance + epsilon
//     %sd = hip.sqrt(%ctx) ins(%ve)
//     %is = hip.reciprocal(%ctx) ins(%sd)
//     %mv = tensor.expand_shape %mean [[0, 1, 2]]      // [C] -> [C, 1, 1]
//     %iv = tensor.expand_shape %is [[0, 1, 2]]
//     %sv = tensor.expand_shape %scale [[0, 1, 2]]
//     %bv = tensor.expand_shape %bias [[0, 1, 2]]
//     %c  = hip.sub(%ctx) ins(%x, %mv)                 // x - mean
//     %n  = hip.mul(%ctx) ins(%c, %iv)
//     %s  = hip.mul(%ctx) ins(%n, %sv)
//     %y  = hip.add(%ctx) ins(%s, %bv)
//   After:
//     %y = hip.batch_norm(%ctx) ins(%x, %scale, %bias, %mean, %variance)
//              outs(%init) {epsilon = ...}
//
// Every multiply and add commutes, so each operand order is matched, and the
// two steps may apply the scale and the normalization in either order. The
// normalization may also divide by sqrt(variance + epsilon), and the inverse
// standard deviation may be written pow(variance + epsilon, -0.5) or
// pow(sqrt(variance + epsilon), -1) instead of reciprocal(sqrt(...)).
//
// A per-channel vector reaches the full-size ops through a chain of
// tensor.expand_shape ops, which is walked back to the [C] vector.
//
// The scale and the bias are optional, as in a batch normalization without
// learned parameters. hip.batch_norm always takes both, so a missing one is
// replaced by a constant vector of ones or zeros. The pattern is rooted at the
// last op of the chain, which is the add of the bias, else the multiply of the
// scale, else the normalization; a root that feeds a further per-channel
// multiply or add leaves the match to that op.
//
// Safety: the full-size intermediates must have a single value use (a
// tensor.dim query is not a value use; it is retargeted to x) and the type of
// x, so nothing broadcasts the input itself. The per-channel operations are
// left in place, as other users may share them, and are erased once dead.
//
//===----------------------------------------------------------------------===//

#include "batch_norm_fusion.hpp"

#include "fusion_utils.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <type_traits>
#include <utility>

namespace hip {
namespace fusion_transform {
namespace {

/// The value that a chain of tensor.expand_shape ops started from, or `value`
/// itself when it is not an expand_shape result.
mlir::Value skipExpandShapes(mlir::Value value) {
  while (auto expand = value.getDefiningOp<mlir::tensor::ExpandShapeOp>())
    value = expand.getSrc();
  return value;
}

/// `type` is the 1-D per-channel vector type of `data`, laid out as
/// (N, C, ...).
bool isChannelVector(mlir::Type type, mlir::RankedTensorType data) {
  auto vector = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return vector && vector.getRank() == 1 &&
         vector.getDimSize(0) == data.getDimSize(1) &&
         vector.getElementType() == data.getElementType();
}

/// `viewType` broadcasts a per-channel vector along dimension 1 of `data`,
/// which is laid out as (N, C, ...).
bool isChannelView(mlir::Type viewType, mlir::RankedTensorType data) {
  auto view = mlir::dyn_cast<mlir::RankedTensorType>(viewType);
  if (!view || view.getRank() > data.getRank() || !view.hasStaticShape() ||
      view.getElementType() != data.getElementType())
    return false;
  // Broadcasting aligns trailing dimensions, so view's dimension
  // `channelDim` is the one that lines up with data's dimension 1.
  const int64_t channelDim = 1 - (data.getRank() - view.getRank());
  if (channelDim < 0)
    return false;
  return llvm::all_of(llvm::seq<int64_t>(view.getRank()), [&](int64_t dim) {
    return view.getDimSize(dim) == (dim == channelDim ? data.getDimSize(1) : 1);
  });
}

/// For a commutative binary op with one full-size operand of type `data` and
/// one per-channel view of it, the pair (full-size operand, view); empty when
/// the op does not have that shape.
template <typename BinaryOp>
std::optional<std::pair<mlir::Value, mlir::Value>>
splitChannelOperand(BinaryOp op, mlir::RankedTensorType data) {
  for (auto [full, view] : {std::pair(op.getLhs(), op.getRhs()),
                            std::pair(op.getRhs(), op.getLhs())})
    if (full.getType() == data && isChannelView(view.getType(), data))
      return std::pair(full, view);
  return std::nullopt;
}

/// `op`'s result feeds a multiply or add that applies one more per-channel
/// view to it, so `op` is not the last op of a batch normalization chain.
bool feedsChannelOp(mlir::Operation *op, mlir::RankedTensorType data) {
  mlir::Value result = op->getResult(0);
  for (mlir::Operation *user : result.getUsers()) {
    std::optional<std::pair<mlir::Value, mlir::Value>> split;
    if (auto add = mlir::dyn_cast<mlir::hip::AddOp>(user))
      split = splitChannelOperand(add, data);
    else if (auto mul = mlir::dyn_cast<mlir::hip::MulOp>(user))
      split = splitChannelOperand(mul, data);
    if (split && split->first == result)
      return true;
  }
  return false;
}

struct Chain {
  mlir::Value input;
  // Each parameter as the [C] vector its view started from. The scale and the
  // bias are null when the chain does not apply them.
  mlir::Value mean, variance, scale, bias;
  std::optional<llvm::APFloat> epsilon;
  // Consumers first, so each op is erased after everything that reads it.
  // The root is not in it: it is replaced.
  llvm::SmallVector<mlir::Operation *> ops;
};

/// `op` computes on a [C] vector.
bool isChannelVectorOp(mlir::Operation *op, mlir::RankedTensorType data) {
  return isChannelVector(op->getResult(0).getType(), data);
}

/// hip.batch_norm carries epsilon as an f32 attribute. Rounding to f32 is
/// fine, but an epsilon that underflows or overflows f32 changes the result,
/// e.g. a zero variance then divides by zero.
bool fitsF32(llvm::APFloat value) {
  bool losesInfo = false;
  llvm::APFloat::opStatus status =
      value.convert(llvm::APFloat::IEEEsingle(),
                    llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return !(status & (llvm::APFloat::opUnderflow | llvm::APFloat::opOverflow));
}

/// `sum` is variance + epsilon. Sets the chain's variance and epsilon.
bool matchVarianceSum(mlir::Value sum, mlir::RankedTensorType data,
                      Chain &chain) {
  auto add = sum.getDefiningOp<mlir::hip::AddOp>();
  if (!add || !isChannelVectorOp(add, data))
    return false;

  for (auto [variance, epsilonValue] :
       {std::pair(add.getLhs(), add.getRhs()),
        std::pair(add.getRhs(), add.getLhs())}) {
    std::optional<llvm::APFloat> epsilon =
        getScalarFloatConstantValue(epsilonValue);
    if (!epsilon || !fitsF32(*epsilon) ||
        !isChannelVector(variance.getType(), data))
      continue;
    chain.variance = variance;
    chain.epsilon = epsilon;
    return true;
  }
  return false;
}

/// `value` is sqrt(variance + epsilon).
bool matchStandardDeviation(mlir::Value value, mlir::RankedTensorType data,
                            Chain &chain) {
  auto sqrt = value.getDefiningOp<mlir::hip::SqrtOp>();
  return sqrt && isChannelVectorOp(sqrt, data) &&
         matchVarianceSum(sqrt.getX(), data, chain);
}

/// `inverse` is 1 / sqrt(variance + epsilon), as reciprocal(sqrt(...)), as
/// pow(sqrt(...), -1), or as pow(variance + epsilon, -0.5). The ONNX converter
/// rewrites these exponents before a hip.pow exists, but other frontends, such
/// as Torch-MLIR, may still emit them.
bool matchInverseStandardDeviation(mlir::Value inverse,
                                   mlir::RankedTensorType data, Chain &chain) {
  if (auto reciprocal = inverse.getDefiningOp<mlir::hip::ReciprocalOp>())
    return isChannelVectorOp(reciprocal, data) &&
           matchStandardDeviation(reciprocal.getX(), data, chain);
  auto pow = inverse.getDefiningOp<mlir::hip::PowOp>();
  if (!pow || !isChannelVectorOp(pow, data))
    return false;
  const double exponent = pow.getExponent().convertToDouble();
  if (exponent == -1.0)
    return matchStandardDeviation(pow.getInput(), data, chain);
  return exponent == -0.5 && matchVarianceSum(pow.getInput(), data, chain);
}

/// One multiply or divide of the full-size tensor by a per-channel view.
struct Step {
  mlir::Operation *op;
  mlir::Value full;
  mlir::Value view;
  bool divides;
};

std::optional<Step> asStep(mlir::Operation *op, mlir::RankedTensorType data) {
  if (auto mul = mlir::dyn_cast<mlir::hip::MulOp>(op)) {
    if (auto split = splitChannelOperand(mul, data))
      return Step{op, split->first, split->second, /*divides=*/false};
  } else if (auto div = mlir::dyn_cast<mlir::hip::DivOp>(op)) {
    // A division does not commute: the full-size tensor is the numerator.
    if (div.getLhs().getType() == data &&
        isChannelView(div.getRhs().getType(), data))
      return Step{op, div.getLhs(), div.getRhs(), /*divides=*/true};
  }
  return std::nullopt;
}

/// `step` normalizes: it multiplies by 1 / sqrt(variance + epsilon) or divides
/// by sqrt(variance + epsilon).
bool matchNormalization(const Step &step, mlir::RankedTensorType data,
                        Chain &chain) {
  mlir::Value vector = skipExpandShapes(step.view);
  return step.divides ? matchStandardDeviation(vector, data, chain)
                      : matchInverseStandardDeviation(vector, data, chain);
}

/// Peels the bias off the end of the chain, so that the steps before it look
/// the same with or without a bias.
std::optional<mlir::Value>
matchBias(mlir::hip::AddOp add, mlir::RankedTensorType data, Chain &chain) {
  auto shifted = splitChannelOperand(add, data);
  if (!shifted)
    return std::nullopt;
  chain.bias = skipExpandShapes(shifted->second);
  return shifted->first;
}

/// The scale and the normalization may come in either order, so both are
/// collected before deciding which is which. `root` is the first step when
/// `rootIsStep`, else the bias add.
llvm::SmallVector<Step, 2>
collectScaleAndNormalizationSteps(mlir::Operation *root, bool rootIsStep,
                                  mlir::Value current,
                                  mlir::RankedTensorType data) {
  llvm::SmallVector<Step, 2> steps;
  mlir::Operation *consumer = root;
  while (steps.size() < 2) {
    mlir::Operation *op = nullptr;
    if (rootIsStep && steps.empty())
      op = root;
    else if (mlir::Operation *def = current.getDefiningOp();
             def && hasSingleValueUse(current, consumer))
      op = def;
    std::optional<Step> step = op ? asStep(op, data) : std::nullopt;
    if (!step)
      break;
    steps.push_back(*step);
    current = step->full;
    consumer = op;
  }
  return steps;
}

/// x enters the chain only through this subtraction, so it also yields the
/// input and the mean.
mlir::hip::SubOp matchCentering(mlir::Value current, mlir::Operation *consumer,
                                mlir::RankedTensorType data, Chain &chain) {
  auto centered = singleUseDef<mlir::hip::SubOp>(current, consumer);
  if (!centered || centered.getLhs().getType() != data ||
      !isChannelView(centered.getRhs().getType(), data))
    return nullptr;
  chain.input = centered.getLhs();
  chain.mean = skipExpandShapes(centered.getRhs());
  return centered;
}

/// The math does not fix the order of scale and normalization, so each
/// assignment is tried. Only a multiply can be the scale.
bool assignScaleAndNormalization(llvm::ArrayRef<Step> steps,
                                 mlir::RankedTensorType data, Chain &chain) {
  // (scale step, normalizing step); a null scale step means the chain does not
  // scale.
  llvm::SmallVector<std::pair<const Step *, const Step *>, 2> roles;
  if (steps.size() == 1) {
    roles.emplace_back(nullptr, &steps[0]);
  } else {
    if (!steps[0].divides)
      roles.emplace_back(&steps[0], &steps[1]);
    if (!steps[1].divides)
      roles.emplace_back(&steps[1], &steps[0]);
  }
  for (auto [scaleStep, normalizationStep] : roles) {
    if (!matchNormalization(*normalizationStep, data, chain))
      continue;
    if (scaleStep)
      chain.scale = skipExpandShapes(scaleStep->view);
    return true;
  }
  return false;
}

/// The fused op runs on the root's context, so every op it replaces must too.
bool usesContext(llvm::ArrayRef<mlir::Operation *> ops, mlir::Value ctx) {
  return llvm::all_of(ops, [&](mlir::Operation *op) {
    // The HIP context is operand 0 of every HIP op.
    return op->getOperand(0) == ctx;
  });
}

/// The init tensors of the erased ops become dead with them. They are found
/// before the rewrite, as an op that is not DPS must fail the match.
std::optional<llvm::SmallVector<mlir::Operation *>>
collectInitDefs(llvm::ArrayRef<mlir::Operation *> ops) {
  llvm::SmallVector<mlir::Operation *> initDefs;
  for (mlir::Operation *op : ops) {
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
    if (!dps)
      return std::nullopt;
    for (mlir::Value init : dps.getDpsInits())
      if (mlir::Operation *def = init.getDefiningOp())
        initDefs.push_back(def);
  }
  return initDefs;
}

/// Kept apart from the matching so that every check runs before the first IR
/// change: a failed match must leave the IR unchanged.
template <typename RootOp>
void buildBatchNorm(mlir::PatternRewriter &rewriter, RootOp root,
                    mlir::RankedTensorType data, const Chain &chain,
                    llvm::ArrayRef<mlir::Operation *> initDefs) {
  auto vectorType =
      mlir::RankedTensorType::get({data.getDimSize(1)}, data.getElementType());
  auto constantVector = [&](double value) -> mlir::Value {
    mlir::Attribute element =
        rewriter.getFloatAttr(data.getElementType(), value);
    return mlir::hip::ConstantOp::create(
        rewriter, root.getLoc(), vectorType,
        mlir::DenseElementsAttr::get(vectorType,
                                     llvm::ArrayRef<mlir::Attribute>{element}));
  };

  // hip.batch_norm carries epsilon as an f32 attribute, rounded from the type
  // the graph added it in.
  llvm::APFloat epsilon = *chain.epsilon;
  bool losesInfo = false;
  epsilon.convert(llvm::APFloat::IEEEsingle(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);

  for (mlir::Operation *op : chain.ops)
    retargetShapeQueries(rewriter, op->getResult(0), chain.input);
  mlir::Value scale = chain.scale ? chain.scale : constantVector(1.0);
  mlir::Value bias = chain.bias ? chain.bias : constantVector(0.0);
  auto batchNorm = mlir::hip::BatchNormOp::create(
      rewriter, root.getLoc(), data, root.getCtx(), chain.input, scale, bias,
      chain.mean, chain.variance, root.getOutput(),
      rewriter.getFloatAttr(rewriter.getF32Type(), epsilon));
  rewriter.replaceOp(root, batchNorm.getResult(0));
  for (mlir::Operation *op : chain.ops)
    rewriter.eraseOp(op);
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  for (mlir::Operation *def : initDefs)
    if (seen.insert(def).second && mlir::isOpTriviallyDead(def))
      rewriter.eraseOp(def);
}

/// Matches the chain ending at a `hip.add` of the bias, or at the last
/// `hip.mul` or `hip.div` when the chain has no bias.
template <typename RootOp>
struct BatchNormFusion : public mlir::OpRewritePattern<RootOp> {
  using mlir::OpRewritePattern<RootOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RootOp root, mlir::PatternRewriter &rewriter) const override {
    if (root->getNumResults() != 1)
      return rewriter.notifyMatchFailure(root, "not a tensor op");
    auto data =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!data || data.getRank() < 2 ||
        mlir::ShapedType::isDynamic(data.getDimSize(1)) ||
        !mlir::isa<mlir::FloatType>(data.getElementType()))
      return rewriter.notifyMatchFailure(
          root, "batch norm needs a static channel dimension 1");
    if (feedsChannelOp(root, data))
      return rewriter.notifyMatchFailure(root, "not the end of the chain");

    constexpr bool endsInBias = std::is_same_v<RootOp, mlir::hip::AddOp>;
    Chain chain;
    mlir::Value current = root->getResult(0);
    if constexpr (endsInBias) {
      std::optional<mlir::Value> shifted = matchBias(root, data, chain);
      if (!shifted)
        return rewriter.notifyMatchFailure(root, "no per-channel bias");
      current = *shifted;
    }

    llvm::SmallVector<Step, 2> steps =
        collectScaleAndNormalizationSteps(root, !endsInBias, current, data);
    if (steps.empty())
      return rewriter.notifyMatchFailure(root, "no multiply or divide");
    mlir::hip::SubOp centered =
        matchCentering(steps.back().full, steps.back().op, data, chain);
    if (!centered)
      return rewriter.notifyMatchFailure(root, "no x - mean");
    if (!assignScaleAndNormalization(steps, data, chain))
      return rewriter.notifyMatchFailure(root,
                                         "no normalization by sqrt(var + eps)");
    for (mlir::Value parameter : {chain.mean, chain.scale, chain.bias})
      if (parameter && !isChannelVector(parameter.getType(), data))
        return rewriter.notifyMatchFailure(root, "parameter is not a vector");

    for (const Step &step : steps)
      if (step.op != root.getOperation())
        chain.ops.push_back(step.op);
    chain.ops.push_back(centered);
    if (!usesContext(chain.ops, root.getCtx()))
      return rewriter.notifyMatchFailure(root,
                                         "batch norm ops must share a context");
    auto initDefs = collectInitDefs(chain.ops);
    if (!initDefs)
      return rewriter.notifyMatchFailure(root, "batch norm op is not DPS");

    buildBatchNorm(rewriter, root, data, chain, *initDefs);
    return mlir::success();
  }
};

} // namespace

void populateBatchNormFusionPattern(mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit) {
  patterns
      .add<BatchNormFusion<mlir::hip::AddOp>, BatchNormFusion<mlir::hip::MulOp>,
           BatchNormFusion<mlir::hip::DivOp>>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
