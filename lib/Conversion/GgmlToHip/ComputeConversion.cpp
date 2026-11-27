/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ComputeConversion.cpp - Lower GGML compute operations ------------===//

#include "GgmlToHipUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace mlir {
namespace hip {

Value castTensor(PatternRewriter &rewriter, Location loc, Value context,
                 Value input, Type targetElementType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    emitError(loc) << "cannot cast non-ranked tensor " << input.getType();
    return {};
  }
  if (inputType.getElementType() == targetElementType) {
    return input;
  }
  const std::optional<int64_t> code =
      getHipCastElementTypeCode(targetElementType);
  if (!code) {
    emitError(loc) << "cannot cast tensor to unsupported element type "
                   << targetElementType;
    return {};
  }
  auto resultType =
      RankedTensorType::get(inputType.getShape(), targetElementType);
  Value init = createGgmlEmptyTensor(rewriter, loc, resultType, input);
  if (!init) {
    return {};
  }
  auto cast =
      hip::CastOp::create(rewriter, loc, TypeRange{resultType}, context, input,
                          init, rewriter.getI64IntegerAttr(*code));
  if (cast.getResultTensors().size() != 1) {
    emitError(loc) << "hip.cast builder produced "
                   << cast.getResultTensors().size()
                   << " results for a tensor destination; expected one";
    return {};
  }
  return cast.getResultTensors().front();
}

namespace {

static Value denseF32Splat(PatternRewriter &rewriter, Location loc,
                           ArrayRef<int64_t> shape, float value) {
  auto type = RankedTensorType::get(shape, rewriter.getF32Type());
  auto data = DenseElementsAttr::get(type, rewriter.getF32FloatAttr(value));
  return hip::ConstantOp::create(rewriter, loc, type, data).getResult();
}

/// Verify the dimension identities that a shape-preserving lowering relies on.
/// Static dimensions need no metadata; dynamic dimensions must carry the same
/// GGML expression on the source and result values.
static LogicalResult verifyShapePreservingExtents(Operation *op, Value input,
                                                  Value result) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!inputType || !resultType ||
      inputType.getRank() != resultType.getRank()) {
    return op->emitOpError(
        "shape-preserving lowering requires equal ranked tensor operands");
  }
  for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
    if (!haveEquivalentGgmlDimensionExtent(input, dimension, result,
                                           dimension)) {
      return op->emitOpError()
             << "shape-preserving dimension " << dimension
             << " lacks matching static or symbolic GGML shape provenance";
    }
  }
  return success();
}

struct GluToHip final : OpRewritePattern<ggml::GluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::GluOp op,
                                PatternRewriter &rewriter) const override {
    if (StringRef refusal = gluRefusal(
            ggml::stringifyGluKind(op.getGluKind()),
            op.getSplit() && op.getUp(), op.getSwapped(),
            op.getAlpha().convertToDouble(), op.getLimit().convertToDouble());
        !refusal.empty()) {
      return rewriter.notifyMatchFailure(op, refusal);
    }
    if (failed(
            verifyShapePreservingExtents(op, op.getGate(), op.getResult())) ||
        failed(verifyShapePreservingExtents(op, op.getUp(), op.getResult()))) {
      return failure();
    }
    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }
    Value outputInit = createGgmlEmptyTensor(
        rewriter, op.getLoc(), op.getResult().getType(), op.getGate());
    if (!outputInit) {
      return failure();
    }
    auto output =
        SwigluOp::create(rewriter, op.getLoc(), TypeRange{outputInit.getType()},
                         *context, op.getGate(), op.getUp(), outputInit);
    rewriter.replaceOp(op, output.getResultTensors());
    return success();
  }
};

struct SetRowsToHip final : OpRewritePattern<ggml::SetRowsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::SetRowsOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }
    auto updatesType = dyn_cast<RankedTensorType>(op.getUpdates().getType());
    auto indicesType = dyn_cast<RankedTensorType>(op.getIndices().getType());
    auto destinationType =
        dyn_cast<RankedTensorType>(op.getDestination().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!updatesType || !indicesType || !destinationType || !resultType ||
        updatesType.getRank() != 4 || indicesType.getRank() != 4 ||
        destinationType.getRank() != 4 || resultType != destinationType) {
      return rewriter.notifyMatchFailure(
          op, "requires the supported rank-4 row-update form");
    }
    if (!indicesType.getElementType().isInteger(64)) {
      return op.emitOpError(
          "HIP lowering requires i64 SET_ROWS indices for hip.scatter_nd");
    }
    if (!updatesType.getElementType().isF32() ||
        !destinationType.getElementType().isF16()) {
      return rewriter.notifyMatchFailure(
          op, "requires f32 rows written into an f16 cache");
    }
    // The current decomposition collapses all dimensions outside [rows,
    // width]. Accepting a batch or another non-unit outer dimension here
    // would silently merge independent cache updates.
    if (updatesType.getDimSize(0) != 1 || updatesType.getDimSize(1) != 1 ||
        indicesType.getDimSize(0) != 1 || indicesType.getDimSize(1) != 1 ||
        indicesType.getDimSize(2) != 1 || destinationType.getDimSize(0) != 1 ||
        destinationType.getDimSize(1) != 1) {
      return op.emitOpError(
          "HIP lowering requires updates dimensions 0 and 1, indices "
          "dimensions 0 through 2, and destination dimensions 0 and 1 to "
          "be statically equal to one");
    }
    // The capacity may be the session's: the cache's slots then come from
    // the cache itself.
    if (destinationType.isDynamicDim(3) || updatesType.isDynamicDim(3)) {
      return rewriter.notifyMatchFailure(op, "requires a static row width");
    }
    if (!haveEquivalentGgmlDimensionExtent(op.getUpdates(), 2, op.getIndices(),
                                           3) ||
        !haveEquivalentGgmlDimensionExtent(op.getUpdates(), 3,
                                           op.getDestination(), 3)) {
      return op.emitOpError(
          "HIP lowering cannot prove compatible SET_ROWS row count and width");
    }

    const int64_t width = destinationType.getDimSize(3);
    const int64_t capacity = destinationType.getDimSize(2);
    const OpFoldResult slots =
        ShapedType::isDynamic(capacity)
            ? OpFoldResult(tensor::DimOp::create(rewriter, op.getLoc(),
                                                 op.getDestination(), 2)
                               .getResult())
            : OpFoldResult(rewriter.getIndexAttr(capacity));
    const int64_t tokenDim = updatesType.getDimSize(2);
    Value tokens =
        tensor::DimOp::create(rewriter, op.getLoc(), op.getUpdates(), 2);
    auto updates2Type =
        RankedTensorType::get({tokenDim, width}, updatesType.getElementType());
    Value updates2 =
        reshapeTensor(rewriter, op.getLoc(), op.getUpdates(), updates2Type,
                      {OpFoldResult(tokens), rewriter.getIndexAttr(width)});
    auto destination2Type = RankedTensorType::get(
        {capacity, width}, destinationType.getElementType());
    Value destination2 =
        reshapeTensor(rewriter, op.getLoc(), op.getDestination(),
                      destination2Type, {slots, rewriter.getIndexAttr(width)});

    auto indices1Type = RankedTensorType::get({indicesType.getDimSize(3)},
                                              indicesType.getElementType());
    Value indices1 = reshapeTensor(rewriter, op.getLoc(), op.getIndices(),
                                   indices1Type, {OpFoldResult(tokens)});
    auto indices2Type = RankedTensorType::get({indicesType.getDimSize(3), 1},
                                              indicesType.getElementType());
    Value indices2 =
        reshapeTensor(rewriter, op.getLoc(), indices1, indices2Type,
                      {OpFoldResult(tokens), rewriter.getIndexAttr(1)});

    // Keep the allocation root at the public rank-4 state type. The output
    // allocator recognizes that root directly; allocating rank 2 here and
    // reshaping only after ScatterND would expose the wrong ABI rank.
    SmallVector<Value> dynamicSizes;
    if (ShapedType::isDynamic(capacity))
      dynamicSizes.push_back(cast<Value>(slots));
    Value resultInit = bufferization::AllocTensorOp::create(
        rewriter, op.getLoc(), resultType, dynamicSizes);
    Value init =
        reshapeTensor(rewriter, op.getLoc(), resultInit, destination2Type,
                      {slots, rewriter.getIndexAttr(width)});

    // The rows rounded to the cache's type, then written: a state updated in
    // place (its output donated the input's buffer) is not copied, as
    // hip.scatter_nd does not copy data onto itself.
    Value rows = castTensor(rewriter, op.getLoc(), *context, updates2,
                            destinationType.getElementType());
    if (!rows)
      return failure();
    Value written =
        hip::ScatterNDOp::create(
            rewriter, op.getLoc(), TypeRange{destination2Type}, *context,
            destination2, indices2, rows, init, rewriter.getStringAttr("none"))
            .getResultTensors()
            .front();
    rewriter.replaceOp(
        op,
        reshapeTensor(rewriter, op.getLoc(), written, resultType,
                      {rewriter.getIndexAttr(resultType.getDimSize(0)),
                       rewriter.getIndexAttr(resultType.getDimSize(1)), slots,
                       rewriter.getIndexAttr(resultType.getDimSize(3))}));
    return success();
  }
};

/// What a rotation's angles follow from, as llama.cpp's GPU RoPE
/// (ggml-cuda/rope.cu) derives it, in float: `theta_scale` is
/// `powf(base, -2.0f / dimensions)` and a pair's power
/// `powf(theta_scale, pair)`; and the frequency factors a model may supply,
/// one f32 a pair, which a split is handed as a tensor.
struct RotationConstants {
  SmallVector<float> powers;
  /// Null when the model supplies none.
  Value frequencyFactors;
  float amplitude = 1.0f;
};

static FailureOr<RotationConstants> rotationConstants(ggml::RopeOp rotation) {
  const int64_t dimensions = rotation.getDimensions();
  const int64_t pairs = dimensions / 2;
  const float base = rotation.getFrequencyBase().convertToFloat();
  RotationConstants constants;
  constants.amplitude = rotation.getAttentionFactor().convertToFloat();

  const float thetaScale =
      std::pow(base, -2.0f / static_cast<float>(dimensions));
  for (int64_t pair = 0; pair < pairs; ++pair)
    constants.powers.push_back(std::pow(thetaScale, static_cast<float>(pair)));

  Value given = rotation.getFrequencyFactors();
  if (!given)
    return constants;
  auto type = dyn_cast<RankedTensorType>(given.getType());
  if (!type || !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getNumElements() != pairs)
    return rotation.emitOpError(
        "frequency factors are not one f32 a rotary pair");
  constants.frequencyFactors = given;
  return constants;
}

/// The cosines and sines a rotation turns this step's rows by, [1, rows,
/// pairs] f32, computed in the step as llama.cpp's GPU RoPE computes them, in
/// the same float operations (rotationConstants): each pair's angle is its
/// position times its power of the base, divided by its frequency factor,
/// times the frequency scale, and the cosine and sine are scaled by the
/// amplitude. `positions` is [1, rows, 1], one position a row. Nothing bounds
/// the positions.
static FailureOr<std::pair<Value, Value>>
rotationAngles(PatternRewriter &rewriter, ggml::RopeOp op, Value context,
               Value positions) {
  const Location loc = op.getLoc();
  FailureOr<RotationConstants> constants = rotationConstants(op);
  if (failed(constants))
    return failure();
  const int64_t pairs = static_cast<int64_t>(constants->powers.size());
  auto positionsType = cast<RankedTensorType>(positions.getType());
  if (positionsType.getRank() != 3 || positionsType.getDimSize(2) != 1)
    return op.emitOpError("its positions do not fit its rotary dimension");

  // The constants, named so every rotation that has the same ones shares
  // them (shareConstants), and with them, once CSE runs, the computation.
  const auto perPair = [&](ArrayRef<float> values, StringRef name) -> Value {
    auto type = RankedTensorType::get({1, 1, pairs}, rewriter.getF32Type());
    auto constant = hip::ConstantOp::create(
        rewriter, loc, type, DenseElementsAttr::get(type, values));
    constant.setSourceNameAttr(
        rewriter.getStringAttr(("ggml.rope." + name).str()));
    return constant.getResult();
  };
  const auto scalar = [&](float value, StringRef name) -> Value {
    Value splat = denseF32Splat(rewriter, loc, {1, 1, 1}, value);
    splat.getDefiningOp<hip::ConstantOp>().setSourceNameAttr(
        rewriter.getStringAttr(("ggml.rope." + name).str()));
    return splat;
  };
  auto anglesType = RankedTensorType::get(
      {1, positionsType.getDimSize(1), pairs}, rewriter.getF32Type());
  SmallVector<Value> rows;
  if (anglesType.isDynamicDim(1))
    rows.push_back(tensor::DimOp::create(rewriter, loc, positions, 1));
  const auto empty = [&]() -> Value {
    return tensor::EmptyOp::create(rewriter, loc, anglesType.getShape(),
                                   rewriter.getF32Type(), rows);
  };
  const auto mul = [&](Value lhs, Value rhs) -> Value {
    return hip::MulOp::create(rewriter, loc, context, lhs, rhs, empty())
        .getResultTensors()
        .front();
  };

  Value rowPositions =
      castTensor(rewriter, loc, context, positions, rewriter.getF32Type());
  if (!rowPositions)
    return failure();
  // theta_base = pos * powf(theta_scale, i); theta_extrap = theta_base / ff.
  Value unscaled = mul(rowPositions, perPair(constants->powers, "powers"));
  if (constants->frequencyFactors) {
    auto type = RankedTensorType::get({1, 1, pairs}, rewriter.getF32Type());
    Value factors =
        reshapeTensor(rewriter, loc, constants->frequencyFactors, type,
                      {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                       rewriter.getIndexAttr(pairs)});
    if (!factors)
      return op.emitOpError("cannot read its frequency factors as one a pair");
    unscaled =
        hip::DivOp::create(rewriter, loc, context, unscaled, factors, empty())
            .getResultTensors()
            .front();
  }
  // theta_interp = freq_scale * theta_extrap.
  const float frequencyScale = op.getFrequencyScale().convertToFloat();
  Value angles = frequencyScale == 1.0f
                     ? unscaled
                     : mul(unscaled, scalar(frequencyScale, "frequency_scale"));
  Value cosine = hip::CosOp::create(rewriter, loc, context, angles, empty())
                     .getResultTensors()
                     .front();
  Value sine = hip::SinOp::create(rewriter, loc, context, angles, empty())
                   .getResultTensors()
                   .front();
  if (constants->amplitude != 1.0f) {
    Value scale = scalar(constants->amplitude, "amplitude");
    cosine = mul(cosine, scale);
    sine = mul(sine, scale);
  }
  return std::make_pair(cosine, sine);
}

/// A rotation of [B, S, H, D] rows, B = 1, by per-row cosines and sines
/// ([1, S, pairs], rotationAngles): hip.rope on the same storage as [B, S,
/// H*D], its tables indexed by the row.
static FailureOr<Value> rotateRows(PatternRewriter &rewriter, ggml::RopeOp op,
                                   Value context, Value cosine, Value sine,
                                   bool interleaved) {
  const Location loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(op.getInput().getType());
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  int64_t hiddenSize = 0;
  if (llvm::MulOverflow(inputType.getDimSize(2), inputType.getDimSize(3),
                        hiddenSize))
    return rewriter.notifyMatchFailure(op, "RoPE hidden size overflow");
  // GGML presents RoPE input in contiguous [B, S, H, D] order. hip.rope's
  // rank-3 form interprets the identical storage as [B, S, H*D], so flatten
  // the two static inner dimensions instead of materializing BNSH before
  // RoPE and transposing it back afterwards.
  Value tokens = tensor::DimOp::create(rewriter, loc, op.getInput(), 1);
  auto bshType = RankedTensorType::get(
      {inputType.getDimSize(0), inputType.getDimSize(1), hiddenSize},
      inputType.getElementType());
  Value bsh =
      reshapeTensor(rewriter, loc, op.getInput(), bshType,
                    {rewriter.getIndexAttr(inputType.getDimSize(0)),
                     OpFoldResult(tokens), rewriter.getIndexAttr(hiddenSize)});
  if (!bsh)
    return rewriter.notifyMatchFailure(op, "cannot flatten RoPE input");
  Value init = createGgmlEmptyTensor(rewriter, loc, bshType, bsh);
  if (!init)
    return failure();
  auto rope = hip::RopeOp::create(
      rewriter, loc, context, bsh, /*position_ids=*/Value(), cosine, sine, init,
      rewriter.getI64IntegerAttr(interleaved ? 1 : 0),
      rewriter.getI64IntegerAttr(inputType.getDimSize(2)),
      rewriter.getI64IntegerAttr(op.getDimensions()));
  Value restored = reshapeTensor(
      rewriter, loc, rope.getResultTensors().front(), resultType,
      {rewriter.getIndexAttr(resultType.getDimSize(0)), OpFoldResult(tokens),
       rewriter.getIndexAttr(resultType.getDimSize(2)),
       rewriter.getIndexAttr(resultType.getDimSize(3))});
  if (!restored || restored.getType() != resultType)
    return rewriter.notifyMatchFailure(op, "cannot restore GGML RoPE layout");
  return restored;
}

struct RopeToHip final : OpRewritePattern<ggml::RopeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::RopeOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t, 4> sections;
    if (ArrayAttr declared = op.getSectionsAttr())
      for (Attribute value : declared)
        sections.push_back(cast<IntegerAttr>(value).getInt());
    if (StringRef refusal =
            ropeRefusal(ggml::stringifyRopeMode(op.getMode()),
                        static_cast<int64_t>(op.getNPast()),
                        static_cast<int64_t>(op.getLegacyContext()),
                        static_cast<int64_t>(op.getNOffs()), sections,
                        op.getExtensionFactor().convertToDouble());
        !refusal.empty()) {
      return rewriter.notifyMatchFailure(op, refusal);
    }
    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }

    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto positionsType =
        dyn_cast<RankedTensorType>(op.getPositions().getType());
    Value frequencyFactors = op.getFrequencyFactors();
    auto frequencyFactorsType =
        frequencyFactors
            ? dyn_cast<RankedTensorType>(frequencyFactors.getType())
            : RankedTensorType();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inputType || !positionsType || !resultType ||
        inputType.getRank() != 4 || positionsType.getRank() != 4 ||
        (frequencyFactorsType && frequencyFactorsType.getRank() != 4) ||
        resultType != inputType) {
      return rewriter.notifyMatchFailure(
          op, "requires rank-4 [B,S,H,D] input and matching result");
    }
    if (!inputType.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(op, "the cosines and sines are f32");
    }
    // hip.rope receives positions reshaped to [1, tokens], and the generated
    // cache is shared across that single batch/outer coordinate. Do not
    // flatten batched positions or frequency factors into that representation.
    if (inputType.getDimSize(0) != 1 || positionsType.getDimSize(0) != 1 ||
        positionsType.getDimSize(1) != 1 || positionsType.getDimSize(2) != 1 ||
        (frequencyFactorsType && (frequencyFactorsType.getDimSize(0) != 1 ||
                                  frequencyFactorsType.getDimSize(1) != 1 ||
                                  frequencyFactorsType.getDimSize(2) != 1))) {
      return op.emitOpError(
          "HIP lowering requires the input batch dimension and all outer "
          "position and frequency-factor dimensions to be statically equal "
          "to one");
    }
    if (inputType.isDynamicDim(2) || inputType.isDynamicDim(3)) {
      return rewriter.notifyMatchFailure(
          op, "requires static head count and head dimension");
    }
    if (!positionsType.getElementType().isInteger(32)) {
      return op.emitOpError(
          "HIP lowering requires i32 GGML RoPE position indices");
    }
    if (!op->hasAttr("ggml.positions_cover_rows") &&
        !haveEquivalentGgmlDimensionExtent(op.getInput(), 1, op.getPositions(),
                                           3)) {
      return op.emitOpError(
          "HIP lowering cannot prove that RoPE positions cover every token");
    }
    if (!op->hasAttr("ggml.positions_cover_rows") &&
        failed(
            verifyShapePreservingExtents(op, op.getInput(), op.getResult()))) {
      return failure();
    }

    // One position a row. Every extent below comes from the positions alone,
    // so every rotation by the same positions makes the same cosines and
    // sines, which CSE then computes once a step.
    Value tokens =
        tensor::DimOp::create(rewriter, op.getLoc(), op.getPositions(), 3);
    Value positions =
        reshapeTensor(rewriter, op.getLoc(), op.getPositions(),
                      RankedTensorType::get({1, positionsType.getDimSize(3), 1},
                                            positionsType.getElementType()),
                      {rewriter.getIndexAttr(1), OpFoldResult(tokens),
                       rewriter.getIndexAttr(1)});
    FailureOr<std::pair<Value, Value>> angles =
        rotationAngles(rewriter, op, *context, positions);
    if (failed(angles))
      return failure();
    FailureOr<Value> rotated =
        rotateRows(rewriter, op, *context, angles->first, angles->second,
                   /*interleaved=*/op.getMode() == ggml::RopeMode::Normal);
    if (failed(rotated))
      return failure();
    rewriter.replaceOp(op, *rotated);
    return success();
  }
};

} // namespace

void populateGgmlComputeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<GluToHip, SetRowsToHip, RopeToHip>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
