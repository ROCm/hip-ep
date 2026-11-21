/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "QLinearPlan.h"
#include "RebiasUnsignedQuantize.h"

#include "hip/Conversion/HipToTosa/HipToTosaLegality.h"

#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <cmath>
#include <cstdlib>
#include <limits>

namespace mlir::hip {

static DenseElementsAttr getInlineDense(Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return {};
  return dyn_cast_if_present<DenseElementsAttr>(constant.getValueAttr());
}

static SmallVector<int64_t> getIntValues(DenseElementsAttr attr) {
  bool isUnsigned = attr.getElementType().isUnsignedInteger();
  return llvm::map_to_vector(attr.getValues<APInt>(), [&](const APInt &v) {
    return isUnsigned ? static_cast<int64_t>(v.getZExtValue())
                      : v.getSExtValue();
  });
}

static bool isQuantizedByte(Type type) {
  return type.isUnsignedInteger(8) || type.isSignlessInteger(8);
}

// A correction that varies over the output plane is one i32 per output
// element of one image; past this size the conv keeps its HIP kernel.
static constexpr int64_t kMaxQLinearConvCorrection = int64_t{1} << 22;

// Below this requantize multiplier, every i32 accumulator scales to at most
// 2^31 * 2^-33 = 0.25 in magnitude, which rounds to 0 also after the f32
// rounding of the product. Such a channel always writes the output zero point.
static constexpr float kDeadChannelMultiplier = 0x1p-33f;

// The per-tensor input scale times each weight scale, over the output scale,
// in f32 and in ORT CPU's order, one entry per channel. With `dead`, a
// channel below kDeadChannelMultiplier gets 0 and is marked, so that its
// output is the zero point for any accumulator and its zero-point correction
// can be dropped. Nullopt when a multiplier is not finite, and with `dead`
// also when every channel is dead: the op's output is then a constant
// (getDeadQLinearOutput), and an integer op scaled by 0 would fold away and
// leave a rocMLIR kernel with no conv or matmul.
static std::optional<SmallVector<float>>
getRequantMultipliers(DenseElementsAttr inputScale,
                      ArrayRef<float> weightScales,
                      DenseElementsAttr outputScale, int64_t channels,
                      SmallVectorImpl<bool> *dead) {
  float xScale = *inputScale.getValues<float>().begin();
  float yScale = *outputScale.getValues<float>().begin();
  SmallVector<float> multipliers;
  multipliers.reserve(channels);
  for (int64_t k : llvm::seq<int64_t>(channels)) {
    float wScale = weightScales[weightScales.size() == 1 ? 0 : k];
    multipliers.push_back(xScale * wScale / yScale);
  }
  if (!llvm::all_of(multipliers, [](float m) { return std::isfinite(m); }))
    return std::nullopt;
  if (dead) {
    dead->assign(channels, false);
    for (auto [m, isDead] : llvm::zip(multipliers, *dead))
      if (std::fabs(m) < kDeadChannelMultiplier) {
        m = 0.0f;
        isDead = true;
      }
    if (!llvm::is_contained(*dead, false))
      return std::nullopt;
  }
  return multipliers;
}

// The multipliers as a rank-0 constant when they are all equal, otherwise in
// `channelShape`, the broadcast shape of a per-channel multiplier.
static DenseElementsAttr getMultiplierAttr(MLIRContext *ctx,
                                           ArrayRef<float> multipliers,
                                           ArrayRef<int64_t> channelShape) {
  Type f32 = Float32Type::get(ctx);
  if (llvm::all_equal(multipliers))
    return DenseElementsAttr::get(RankedTensorType::get({}, f32),
                                  multipliers.front());
  return DenseElementsAttr::get(RankedTensorType::get(channelShape, f32),
                                multipliers);
}

static DenseElementsAttr getRequantMultiplier(DenseElementsAttr inputScale,
                                              ArrayRef<float> weightScales,
                                              DenseElementsAttr outputScale,
                                              ArrayRef<int64_t> channelShape) {
  int64_t channels = channelShape[1];
  std::optional<SmallVector<float>> multipliers = getRequantMultipliers(
      inputScale, weightScales, outputScale, channels, /*dead=*/nullptr);
  if (!multipliers)
    return {};
  return getMultiplierAttr(inputScale.getContext(), *multipliers, channelShape);
}

static DenseElementsAttr getScalarZeroPoint(DenseElementsAttr zp,
                                            Type elementType) {
  return DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                *zp.getValues<APInt>().begin());
}

// The i32 bias of an integer conv on a rebiased input: B - z' * sum(w) over
// the taps that each output reads, with z' = x_zp - offset. Null when every
// entry is 0. The weights must be inline only when z' is not 0.
//
// `weightZeroPoints` (one per output channel, or empty for 0) holds zw', the
// weight zero points of i8 weights. The full sum is then
//
//   sum (x' - z')(w - zw') = conv(x', w) - zw' * sum(x') - z' * sum(w)
//                            + z' * zw' * taps
//
// over the taps that read the input, and this bias takes the last two terms.
// The caller computes zw' * sum(x') as a second conv. A channel in `dead`
// gets 0, since its output does not depend on the accumulator.
static FailureOr<DenseElementsAttr>
getConvCorrectedBias(RankedTensorType inputType, RankedTensorType weightType,
                     RankedTensorType resultType, int64_t inputZeroPoint,
                     DenseElementsAttr weights, ArrayRef<int64_t> biasValues,
                     ArrayAttr strides, ArrayAttr dilations, ArrayAttr pads,
                     ArrayRef<int64_t> weightZeroPoints = {},
                     ArrayRef<bool> dead = {}) {
  ArrayRef<int64_t> inShape = inputType.getShape();
  ArrayRef<int64_t> wShape = weightType.getShape();
  ArrayRef<int64_t> outShape = resultType.getShape();
  int64_t channels = wShape[0];
  int64_t outH = outShape[2], outW = outShape[3];

  auto attrValue = [](ArrayAttr attr, unsigned index) {
    return cast<IntegerAttr>(attr[index]).getInt();
  };
  int64_t strideH = attrValue(strides, 0);
  int64_t strideW = attrValue(strides, 1);
  int64_t dilationH = attrValue(dilations, 0);
  int64_t dilationW = attrValue(dilations, 1);
  int64_t padTop = attrValue(pads, 0);
  int64_t padLeft = attrValue(pads, 1);

  int64_t offset = getSignedRebiasOffset(inputType.getElementType());
  int64_t zpShift = inputZeroPoint - offset;
  if (zpShift != 0 && !weights)
    return failure();

  // The taps along one axis that read the input at output position `o`; a
  // contiguous range because the taps are equally spaced.
  auto validTaps = [](int64_t o, int64_t stride, int64_t pad, int64_t dilation,
                      int64_t kernel, int64_t size) {
    int64_t lo = kernel, hi = -1;
    for (int64_t k : llvm::seq<int64_t>(kernel)) {
      int64_t at = o * stride - pad + k * dilation;
      if (at >= 0 && at < size) {
        lo = std::min(lo, k);
        hi = std::max(hi, k);
      }
    }
    return std::make_pair(lo, hi);
  };
  SmallVector<std::pair<int64_t, int64_t>> rowTaps, colTaps;
  for (int64_t oh : llvm::seq<int64_t>(outH))
    rowTaps.push_back(
        validTaps(oh, strideH, padTop, dilationH, wShape[2], inShape[2]));
  for (int64_t ow : llvm::seq<int64_t>(outW))
    colTaps.push_back(
        validTaps(ow, strideW, padLeft, dilationW, wShape[3], inShape[3]));

  // tapSums[k][r][s] = sum over the input channels of w[k, c, r, s].
  int64_t taps = wShape[2] * wShape[3];
  SmallVector<int64_t> tapSums(channels * taps, 0);
  if (zpShift != 0) {
    int64_t index = 0;
    for (const APInt &v : weights.getValues<APInt>()) {
      int64_t k = index / (wShape[1] * taps);
      tapSums[k * taps + index % taps] += v.getSExtValue();
      ++index;
    }
  }
  auto correctedBias = [&](int64_t k, int64_t oh, int64_t ow) -> int64_t {
    if (!dead.empty() && dead[k])
      return 0;
    auto [r0, r1] = rowTaps[oh];
    auto [s0, s1] = colTaps[ow];
    int64_t sum = 0;
    for (int64_t r = r0; r <= r1; ++r)
      for (int64_t s = s0; s <= s1; ++s)
        sum += tapSums[k * taps + r * wShape[3] + s];
    int64_t value = biasValues[k] - zpShift * sum;
    if (!weightZeroPoints.empty()) {
      int64_t validTaps = std::max<int64_t>(r1 - r0 + 1, 0) *
                          std::max<int64_t>(s1 - s0 + 1, 0) * wShape[1];
      value += zpShift * weightZeroPoints[k] * validTaps;
    }
    return value;
  };

  bool uniform =
      zpShift == 0 || (llvm::all_equal(rowTaps) && llvm::all_equal(colTaps));
  int64_t planeSize = uniform ? 1 : outH * outW;
  if (channels * planeSize > kMaxQLinearConvCorrection)
    return failure();
  SmallVector<int32_t> biasData;
  biasData.reserve(channels * planeSize);
  bool allZero = true;
  for (int64_t k : llvm::seq<int64_t>(channels))
    for (int64_t position : llvm::seq<int64_t>(planeSize)) {
      int64_t value = correctedBias(k, position / outW, position % outW);
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        return failure();
      allZero &= value == 0;
      biasData.push_back(static_cast<int32_t>(value));
    }
  if (allZero)
    return DenseElementsAttr();
  SmallVector<int64_t> shape = {1, channels, uniform ? 1 : outH,
                                uniform ? 1 : outW};
  return DenseElementsAttr::get(
      RankedTensorType::get(shape,
                            IntegerType::get(inputType.getContext(), 32)),
      ArrayRef<int32_t>(biasData));
}

// The rocMLIR route needs every quantization parameter, the weights, and the
// bias as inline constants, a weight zero point of 0 (rocMLIR has no
// zero-point operand), and static shapes. Anything else keeps the HIP kernel.
std::optional<QLinearPlan> planQLinearConv(QLinearConvOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || !resultType.hasStaticShape())
    return std::nullopt;
  Type inputElement = inputType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(inputElement) || !isQuantizedByte(outputElement) ||
      !weightType.getElementType().isSignlessInteger(8))
    return std::nullopt;

  DenseElementsAttr inputScale = getInlineDense(op.getInputScale());
  DenseElementsAttr inputZp = getInlineDense(op.getInputZeroPoint());
  DenseElementsAttr weightScale = getInlineDense(op.getWeightScale());
  DenseElementsAttr weightZp = getInlineDense(op.getWeightZeroPoint());
  DenseElementsAttr outputScale = getInlineDense(op.getOutputScale());
  DenseElementsAttr outputZp = getInlineDense(op.getOutputZeroPoint());
  DenseElementsAttr weights = getInlineDense(op.getWeights());
  DenseElementsAttr bias;
  if (Value biasValue = op.getBias()) {
    bias = getInlineDense(biasValue);
    if (!bias)
      return std::nullopt;
  }
  if (!inputScale || !inputZp || !weightScale || !weightZp || !outputScale ||
      !outputZp || !weights || inputScale.getNumElements() != 1 ||
      inputZp.getNumElements() != 1 || outputScale.getNumElements() != 1 ||
      outputZp.getNumElements() != 1)
    return std::nullopt;
  if (llvm::any_of(getIntValues(weightZp), [](int64_t v) { return v != 0; }))
    return std::nullopt;

  int64_t channels = weightType.getDimSize(0);
  SmallVector<float> weightScales =
      llvm::to_vector(weightScale.getValues<float>());
  if (weightScales.size() != 1 &&
      static_cast<int64_t>(weightScales.size()) != channels)
    return std::nullopt;
  SmallVector<int64_t> biasValues(channels, 0);
  if (bias) {
    if (bias.getNumElements() != channels)
      return std::nullopt;
    biasValues = getIntValues(bias);
  }

  SmallVector<bool> dead;
  std::optional<SmallVector<float>> multipliers = getRequantMultipliers(
      inputScale, weightScales, outputScale, channels, &dead);
  if (!multipliers)
    return std::nullopt;
  FailureOr<DenseElementsAttr> correctedBias = getConvCorrectedBias(
      inputType, weightType, resultType, getIntValues(inputZp).front(), weights,
      biasValues, op.getStrides(), op.getDilations(), op.getPads(),
      /*weightZeroPoints=*/{}, dead);
  if (failed(correctedBias))
    return std::nullopt;

  QLinearPlan plan;
  plan.bias = *correctedBias;
  plan.multiplier =
      getMultiplierAttr(op.getContext(), *multipliers, {1, channels, 1, 1});
  plan.outputZeroPoint = getScalarZeroPoint(outputZp, outputElement);
  return plan;
}

// The largest |acc - zw' * sum(x') + bias| of one channel, bounded by the
// largest i8 input magnitude: it must fit i32 for the kernel to be exact.
static bool fitsInt32(int64_t absWeightSum, int64_t absZeroPointTerm,
                      int64_t absBias) {
  constexpr int64_t kMaxInput = 128;
  return kMaxInput * (absWeightSum + absZeroPointTerm) + absBias <=
         std::numeric_limits<int32_t>::max();
}

std::optional<AsymmetricQLinearPlan>
planAsymmetricQLinearConv(QLinearConvOp op) {
  if (op->getNumResults() != 1)
    return std::nullopt;
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(op.getWeights().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || !resultType.hasStaticShape())
    return std::nullopt;
  Type inputElement = inputType.getElementType();
  Type weightElement = weightType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(inputElement) || !isQuantizedByte(outputElement) ||
      !isQuantizedByte(weightElement))
    return std::nullopt;

  DenseElementsAttr inputScale = getInlineDense(op.getInputScale());
  DenseElementsAttr inputZp = getInlineDense(op.getInputZeroPoint());
  DenseElementsAttr weightScale = getInlineDense(op.getWeightScale());
  DenseElementsAttr weightZp = getInlineDense(op.getWeightZeroPoint());
  DenseElementsAttr outputScale = getInlineDense(op.getOutputScale());
  DenseElementsAttr outputZp = getInlineDense(op.getOutputZeroPoint());
  DenseElementsAttr weights = getInlineDense(op.getWeights());
  DenseElementsAttr bias;
  if (Value biasValue = op.getBias()) {
    bias = getInlineDense(biasValue);
    if (!bias)
      return std::nullopt;
  }
  if (!inputScale || !inputZp || !weightScale || !weightZp || !outputScale ||
      !outputZp || !weights || inputScale.getNumElements() != 1 ||
      inputZp.getNumElements() != 1 || outputScale.getNumElements() != 1 ||
      outputZp.getNumElements() != 1)
    return std::nullopt;

  int64_t channels = weightType.getDimSize(0);
  SmallVector<float> weightScales =
      llvm::to_vector(weightScale.getValues<float>());
  SmallVector<int64_t> weightZps = getIntValues(weightZp);
  auto isPerTensorOrChannel = [&](size_t size) {
    return size == 1 || static_cast<int64_t>(size) == channels;
  };
  if (!isPerTensorOrChannel(weightScales.size()) ||
      !isPerTensorOrChannel(weightZps.size()))
    return std::nullopt;

  // zw' and the weights as i8: (w - offset) - (zw - offset) = w - zw.
  int64_t weightOffset = getSignedRebiasOffset(weightElement);
  SmallVector<int64_t> shiftedZps;
  shiftedZps.reserve(channels);
  for (int64_t k : llvm::seq<int64_t>(channels))
    shiftedZps.push_back(weightZps[weightZps.size() == 1 ? 0 : k] -
                         weightOffset);
  bool hasWeightZeroPoint =
      llvm::any_of(shiftedZps, [](int64_t zp) { return zp != 0; });
  if (weightOffset == 0 && !hasWeightZeroPoint)
    return std::nullopt;
  Type i8 = IntegerType::get(op.getContext(), 8);
  SmallVector<int8_t> shiftedWeights =
      llvm::map_to_vector(getIntValues(weights), [&](int64_t w) {
        return static_cast<int8_t>(w - weightOffset);
      });
  auto i8WeightType = weightType.clone(i8);
  DenseElementsAttr i8Weights =
      DenseElementsAttr::get(i8WeightType, ArrayRef<int8_t>(shiftedWeights));

  SmallVector<int64_t> biasValues(channels, 0);
  if (bias) {
    if (bias.getNumElements() != channels)
      return std::nullopt;
    biasValues = getIntValues(bias);
  }
  SmallVector<bool> dead;
  std::optional<SmallVector<float>> multipliers = getRequantMultipliers(
      inputScale, weightScales, outputScale, channels, &dead);
  if (!multipliers)
    return std::nullopt;
  FailureOr<DenseElementsAttr> correctedBias = getConvCorrectedBias(
      inputType, i8WeightType, resultType, getIntValues(inputZp).front(),
      i8Weights, biasValues, op.getStrides(), op.getDilations(), op.getPads(),
      hasWeightZeroPoint ? ArrayRef<int64_t>(shiftedZps) : ArrayRef<int64_t>(),
      dead);
  if (failed(correctedBias))
    return std::nullopt;

  // Every live channel's i32 sum must stay exact.
  int64_t perChannel = i8WeightType.getNumElements() / channels;
  SmallVector<int64_t> absBias(channels, 0);
  if (DenseElementsAttr corrected = *correctedBias) {
    int64_t plane = corrected.getNumElements() / channels;
    int64_t index = 0;
    for (const APInt &v : corrected.getValues<APInt>()) {
      int64_t k = index++ / plane;
      absBias[k] = std::max<int64_t>(absBias[k], std::abs(v.getSExtValue()));
    }
  }
  for (int64_t k : llvm::seq<int64_t>(channels)) {
    if (dead[k])
      continue;
    int64_t absWeightSum = 0;
    for (int8_t w :
         ArrayRef<int8_t>(shiftedWeights).slice(k * perChannel, perChannel))
      absWeightSum += std::abs(static_cast<int64_t>(w));
    if (!fitsInt32(absWeightSum, std::abs(shiftedZps[k]) * perChannel,
                   absBias[k]))
      return std::nullopt;
  }

  AsymmetricQLinearPlan plan;
  plan.weights = i8Weights;
  if (hasWeightZeroPoint) {
    SmallVector<int8_t> filter;
    filter.reserve(i8WeightType.getNumElements());
    for (int64_t k : llvm::seq<int64_t>(channels))
      filter.append(perChannel, static_cast<int8_t>(shiftedZps[k]));
    plan.zeroPointFilter =
        DenseElementsAttr::get(i8WeightType, ArrayRef<int8_t>(filter));
  }
  plan.requantize.bias = *correctedBias;
  plan.requantize.multiplier =
      getMultiplierAttr(op.getContext(), *multipliers, {1, channels, 1, 1});
  plan.requantize.outputZeroPoint = getScalarZeroPoint(outputZp, outputElement);
  return plan;
}

// The f32 values of a constant conv bias: an inline f32 constant or a
// dequantize of one with inline parameters, as (b - zp) * scale.
static DenseElementsAttr getConstantConvBias(Value bias, int64_t channels) {
  auto biasType = RankedTensorType::get({1, channels, 1, 1},
                                        Float32Type::get(bias.getContext()));
  if (DenseElementsAttr values = getInlineDense(bias)) {
    if (!values.getElementType().isF32() || values.getNumElements() != channels)
      return {};
    return values.reshape(biasType);
  }
  auto dequantize = bias.getDefiningOp<DequantizeLinearOp>();
  if (!dequantize || dequantize.getPackedInt4() ||
      dequantize.getBlockSize() != 0)
    return {};
  DenseElementsAttr data = getInlineDense(dequantize.getInput());
  DenseElementsAttr scale = getInlineDense(dequantize.getScale());
  DenseElementsAttr zeroPoint;
  if (Value zp = dequantize.getZeroPoint()) {
    zeroPoint = getInlineDense(zp);
    if (!zeroPoint)
      return {};
  }
  if (!data || !scale || !isa<IntegerType>(data.getElementType()) ||
      !scale.getElementType().isF32() || data.getNumElements() != channels)
    return {};
  auto perChannel = [&](DenseElementsAttr attr) {
    return attr.getNumElements() == 1 || attr.getNumElements() == channels;
  };
  if (!perChannel(scale) || (zeroPoint && !perChannel(zeroPoint)))
    return {};
  SmallVector<int64_t> ints = getIntValues(data);
  SmallVector<float> scales = llvm::to_vector(scale.getValues<float>());
  SmallVector<int64_t> zps =
      zeroPoint ? getIntValues(zeroPoint) : SmallVector<int64_t>{0};
  SmallVector<float> values;
  for (int64_t k : llvm::seq<int64_t>(channels)) {
    int64_t zp = zps[zps.size() == 1 ? 0 : k];
    float s = scales[scales.size() == 1 ? 0 : k];
    values.push_back(static_cast<float>(ints[k] - zp) * s);
  }
  return DenseElementsAttr::get(biasType, ArrayRef<float>(values));
}

// conv(DQ(x), DQ(w), bias) with static 2-D shapes and an f32 result, where
// DQ(x) is per-tensor with inline parameters on u8 or i8, and DQ(w) is
// per-tensor or per-output-channel on i8 with inline scales and a zero weight
// zero point. The bias, if any, is constant. The weights must be inline only
// when the input zero point needs a correction. Anything else keeps the float
// conv.
std::optional<DequantizedConvPlan> planDequantizedConv(ConvOp op) {
  if (op->getNumResults() != 1)
    return std::nullopt;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto input = op.getInput().getDefiningOp<DequantizeLinearOp>();
  auto weights = op.getWeights().getDefiningOp<DequantizeLinearOp>();
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getRank() != 4 || !resultType.getElementType().isF32() ||
      !input || !weights)
    return std::nullopt;
  for (DequantizeLinearOp dequantize : {input, weights})
    if (dequantize.getPackedInt4() || dequantize.getBlockSize() != 0 ||
        dequantize->getNumResults() != 1)
      return std::nullopt;
  auto inputType = dyn_cast<RankedTensorType>(input.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(weights.getInput().getType());
  if (!inputType || !weightType || !inputType.hasStaticShape() ||
      !weightType.hasStaticShape() || inputType.getRank() != 4 ||
      weightType.getRank() != 4 ||
      !isQuantizedByte(inputType.getElementType()) ||
      !weightType.getElementType().isSignlessInteger(8))
    return std::nullopt;

  DenseElementsAttr inputScale = getInlineDense(input.getScale());
  DenseElementsAttr weightScale = getInlineDense(weights.getScale());
  if (!inputScale || !weightScale || inputScale.getNumElements() != 1 ||
      !inputScale.getElementType().isF32() ||
      !weightScale.getElementType().isF32())
    return std::nullopt;
  int64_t inputZp = 0;
  if (Value zp = input.getZeroPoint()) {
    DenseElementsAttr attr = getInlineDense(zp);
    if (!attr || attr.getNumElements() != 1)
      return std::nullopt;
    inputZp = getIntValues(attr).front();
  }
  if (Value zp = weights.getZeroPoint()) {
    DenseElementsAttr attr = getInlineDense(zp);
    if (!attr ||
        llvm::any_of(getIntValues(attr), [](int64_t v) { return v != 0; }))
      return std::nullopt;
  }
  int64_t channels = weightType.getDimSize(0);
  SmallVector<float> weightScales =
      llvm::to_vector(weightScale.getValues<float>());
  int64_t axis =
      weights.getAxis() < 0 ? weights.getAxis() + 4 : weights.getAxis();
  if (weightScales.size() != 1 &&
      (static_cast<int64_t>(weightScales.size()) != channels || axis != 0))
    return std::nullopt;

  DequantizedConvPlan plan;
  plan.input = input.getInput();
  plan.weights = weights.getInput();
  if (Value bias = op.getBias()) {
    plan.bias = getConstantConvBias(bias, channels);
    if (!plan.bias)
      return std::nullopt;
  }
  FailureOr<DenseElementsAttr> correction = getConvCorrectedBias(
      inputType, weightType, resultType, inputZp,
      getInlineDense(weights.getInput()), SmallVector<int64_t>(channels, 0),
      op.getStrides(), op.getDilations(), op.getPads());
  if (failed(correction))
    return std::nullopt;
  plan.correction = *correction;
  auto unitScale = DenseElementsAttr::get(
      RankedTensorType::get({}, Float32Type::get(op.getContext())), 1.0f);
  plan.multiplier = getRequantMultiplier(inputScale, weightScales, unitScale,
                                         {1, channels, 1, 1});
  if (!plan.multiplier)
    return std::nullopt;
  return plan;
}

// The matmul form of the plan, with z' = a_zp - offset and zb' the column
// zero points of i8 B (empty for 0):
//
//   sum_k (a' - z')(b - zb') = matmul(a', b) - zb' * rowsum(a')
//                              - z' * colsum(b) + z' * zb' * K
//
// The bias takes the last two terms, one value per output column; the caller
// computes zb' * rowsum(a') as a second matmul. B must be an inline constant
// only when z' is not 0, or with `zeroPointFilter`, which holds zb' in B's
// shape. `weights` replaces B when it is set. A dead column gets 0.
static std::optional<AsymmetricQLinearPlan> planMatMul(QLinearMatMulOp op,
                                                       bool asymmetric) {
  auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(op.getB().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!aType || !bType || !resultType || bType.isDynamicDim(1) ||
      (asymmetric && !bType.hasStaticShape()))
    return std::nullopt;
  Type aElement = aType.getElementType();
  Type bElement = bType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(aElement) || !isQuantizedByte(outputElement) ||
      !isQuantizedByte(bElement))
    return std::nullopt;

  DenseElementsAttr aScale = getInlineDense(op.getAScale());
  DenseElementsAttr aZp = getInlineDense(op.getAZeroPoint());
  DenseElementsAttr bScale = getInlineDense(op.getBScale());
  DenseElementsAttr bZp = getInlineDense(op.getBZeroPoint());
  DenseElementsAttr yScale = getInlineDense(op.getYScale());
  DenseElementsAttr yZp = getInlineDense(op.getYZeroPoint());
  if (!aScale || !aZp || !bScale || !bZp || !yScale || !yZp)
    return std::nullopt;

  int64_t depth = bType.getDimSize(0);
  int64_t n = bType.getDimSize(1);
  SmallVector<float> bScales = llvm::to_vector(bScale.getValues<float>());
  SmallVector<int64_t> bZps = getIntValues(bZp);
  auto isPerTensorOrColumn = [&](size_t size) {
    return size == 1 || static_cast<int64_t>(size) == n;
  };
  if (!isPerTensorOrColumn(bScales.size()) || !isPerTensorOrColumn(bZps.size()))
    return std::nullopt;
  int64_t bOffset = getSignedRebiasOffset(bElement);
  SmallVector<int64_t> shiftedZps;
  shiftedZps.reserve(n);
  for (int64_t j : llvm::seq<int64_t>(n))
    shiftedZps.push_back(bZps[bZps.size() == 1 ? 0 : j] - bOffset);
  bool hasWeightZeroPoint =
      llvm::any_of(shiftedZps, [](int64_t zp) { return zp != 0; });
  // The plain plan takes i8 B with zero point 0; the asymmetric plan the rest.
  if (asymmetric != (bOffset != 0 || hasWeightZeroPoint))
    return std::nullopt;

  int64_t zpShift = getIntValues(aZp).front() - getSignedRebiasOffset(aElement);
  AsymmetricQLinearPlan plan;
  SmallVector<int64_t> bValues;
  if (zpShift != 0 || asymmetric) {
    DenseElementsAttr b = getInlineDense(op.getB());
    if (!b)
      return std::nullopt;
    bValues = llvm::map_to_vector(getIntValues(b),
                                  [&](int64_t v) { return v - bOffset; });
  }

  SmallVector<bool> dead;
  std::optional<SmallVector<float>> multipliers =
      getRequantMultipliers(aScale, bScales, yScale, n, &dead);
  if (!multipliers)
    return std::nullopt;

  SmallVector<int64_t> columnSums(n, 0), absColumnSums(n, 0);
  for (auto [index, v] : llvm::enumerate(bValues)) {
    columnSums[index % n] += v;
    absColumnSums[index % n] += std::abs(v);
  }
  SmallVector<int32_t> biasData;
  biasData.reserve(n);
  for (int64_t j : llvm::seq<int64_t>(n)) {
    int64_t value =
        dead[j] ? 0
                : -zpShift * columnSums[j] + zpShift * shiftedZps[j] * depth;
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max())
      return std::nullopt;
    if (asymmetric && !dead[j] &&
        !fitsInt32(absColumnSums[j], std::abs(shiftedZps[j]) * depth,
                   std::abs(value)))
      return std::nullopt;
    biasData.push_back(static_cast<int32_t>(value));
  }
  if (llvm::any_of(biasData, [](int32_t v) { return v != 0; }))
    plan.requantize.bias = DenseElementsAttr::get(
        RankedTensorType::get({1, n}, IntegerType::get(op.getContext(), 32)),
        ArrayRef<int32_t>(biasData));
  plan.requantize.multiplier =
      getMultiplierAttr(op.getContext(), *multipliers, {1, n});
  plan.requantize.outputZeroPoint = getScalarZeroPoint(yZp, outputElement);

  if (asymmetric) {
    auto i8BType = bType.clone(IntegerType::get(op.getContext(), 8));
    SmallVector<int8_t> shifted = llvm::map_to_vector(
        bValues, [](int64_t v) { return static_cast<int8_t>(v); });
    plan.weights = DenseElementsAttr::get(i8BType, ArrayRef<int8_t>(shifted));
    if (hasWeightZeroPoint) {
      SmallVector<int8_t> row = llvm::map_to_vector(
          shiftedZps, [](int64_t zp) { return static_cast<int8_t>(zp); });
      SmallVector<int8_t> filter;
      filter.reserve(depth * n);
      for ([[maybe_unused]] int64_t r : llvm::seq<int64_t>(depth))
        llvm::append_range(filter, row);
      plan.zeroPointFilter =
          DenseElementsAttr::get(i8BType, ArrayRef<int8_t>(filter));
    }
  }
  return plan;
}

std::optional<QLinearPlan> planQLinearMatMul(QLinearMatMulOp op) {
  std::optional<AsymmetricQLinearPlan> plan = planMatMul(op, false);
  if (!plan)
    return std::nullopt;
  return plan->requantize;
}

std::optional<AsymmetricQLinearPlan>
planAsymmetricQLinearMatMul(QLinearMatMulOp op) {
  if (op->getNumResults() != 1)
    return std::nullopt;
  auto aType = dyn_cast<RankedTensorType>(op.getA().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!aType || !resultType || !aType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return std::nullopt;
  return planMatMul(op, true);
}

bool isPlannedQLinear(Operation *op) {
  if (auto conv = dyn_cast<QLinearConvOp>(op))
    return conv->getNumResults() == 1 && planQLinearConv(conv).has_value();
  if (auto matmul = dyn_cast<QLinearMatMulOp>(op))
    return matmul->getNumResults() == 1 &&
           planQLinearMatMul(matmul).has_value();
  if (auto conv = dyn_cast<ConvOp>(op))
    return planDequantizedConv(conv).has_value();
  return false;
}

DenseElementsAttr getDeadQLinearOutput(Operation *op) {
  Value inputScale, weightScale, outputScale, outputZp, output;
  int64_t channels;
  if (auto conv = dyn_cast<QLinearConvOp>(op)) {
    auto weightType = dyn_cast<RankedTensorType>(conv.getWeights().getType());
    if (!weightType || weightType.isDynamicDim(0))
      return {};
    channels = weightType.getDimSize(0);
    inputScale = conv.getInputScale();
    weightScale = conv.getWeightScale();
    outputScale = conv.getOutputScale();
    outputZp = conv.getOutputZeroPoint();
    output = conv.getOutput();
  } else if (auto matmul = dyn_cast<QLinearMatMulOp>(op)) {
    auto bType = dyn_cast<RankedTensorType>(matmul.getB().getType());
    if (!bType || bType.getRank() < 2 ||
        bType.isDynamicDim(bType.getRank() - 1))
      return {};
    channels = bType.getDimSize(bType.getRank() - 1);
    inputScale = matmul.getAScale();
    weightScale = matmul.getBScale();
    outputScale = matmul.getYScale();
    outputZp = matmul.getYZeroPoint();
    output = matmul.getOutput();
  } else {
    return {};
  }
  auto resultType = dyn_cast<RankedTensorType>(output.getType());
  if (op->getNumResults() != 1 || !resultType || !resultType.hasStaticShape())
    return {};
  DenseElementsAttr xScale = getInlineDense(inputScale);
  DenseElementsAttr wScale = getInlineDense(weightScale);
  DenseElementsAttr yScale = getInlineDense(outputScale);
  DenseElementsAttr yZp = getInlineDense(outputZp);
  if (!xScale || !wScale || !yScale || !yZp ||
      !xScale.getElementType().isF32() || !wScale.getElementType().isF32() ||
      !yScale.getElementType().isF32() || xScale.getNumElements() != 1 ||
      yScale.getNumElements() != 1 || yZp.getNumElements() != 1 ||
      yZp.getElementType() != resultType.getElementType())
    return {};
  SmallVector<float> weightScales = llvm::to_vector(wScale.getValues<float>());
  if (weightScales.size() != 1 &&
      static_cast<int64_t>(weightScales.size()) != channels)
    return {};
  std::optional<SmallVector<float>> multipliers = getRequantMultipliers(
      xScale, weightScales, yScale, channels, /*dead=*/nullptr);
  if (!multipliers || !llvm::all_of(*multipliers, [](float m) {
        return std::fabs(m) < kDeadChannelMultiplier;
      }))
    return {};
  return DenseElementsAttr::get(resultType, *yZp.getValues<APInt>().begin());
}

// The epilogue that both QLinear rewrites share: add the corrected bias, cast
// to f32, scale, and quantize with unit scale, which rounds half to even,
// adds the zero point, and saturates, as ORT CPU does.
Value emitTensorEmpty(RewriterBase &rewriter, Location loc,
                      RankedTensorType type) {
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                 type.getElementType());
}

Value emitConstant(RewriterBase &rewriter, Location loc,
                   DenseElementsAttr value) {
  return ConstantOp::create(rewriter, loc,
                            cast<RankedTensorType>(value.getType()), value);
}

// (acc + bias) in i32, then cast to f32 and multiplied by `multiplier`.
Value emitScaledAccumulator(RewriterBase &rewriter, Location loc, Value ctx,
                            Value acc, DenseElementsAttr bias,
                            DenseElementsAttr multiplier) {
  auto accType = cast<RankedTensorType>(acc.getType());
  auto floatType = accType.clone(rewriter.getF32Type());
  if (bias)
    acc = AddOp::create(rewriter, loc, TypeRange{accType}, ctx, acc,
                        emitConstant(rewriter, loc, bias),
                        emitTensorEmpty(rewriter, loc, accType))
              .getResult(0);
  Value scaled = CastOp::create(rewriter, loc, TypeRange{floatType}, ctx, acc,
                                emitTensorEmpty(rewriter, loc, floatType),
                                rewriter.getI64IntegerAttr(/*ONNX FLOAT=*/1))
                     .getResult(0);
  return MulOp::create(rewriter, loc, TypeRange{floatType}, ctx, scaled,
                       emitConstant(rewriter, loc, multiplier),
                       emitTensorEmpty(rewriter, loc, floatType))
      .getResult(0);
}

Value emitRequantizeEpilogue(RewriterBase &rewriter, Location loc, Value ctx,
                             Value acc, const QLinearPlan &plan, Value output) {
  auto resultType = cast<RankedTensorType>(output.getType());
  auto constant = [&](DenseElementsAttr value) {
    return emitConstant(rewriter, loc, value);
  };
  Value scaled = emitScaledAccumulator(rewriter, loc, ctx, acc, plan.bias,
                                       plan.multiplier);
  Value one = constant(DenseElementsAttr::get(
      RankedTensorType::get({}, rewriter.getF32Type()), 1.0f));
  return QuantizeLinearOp::create(
             rewriter, loc, TypeRange{resultType}, ctx, scaled, one,
             constant(plan.outputZeroPoint), output,
             rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(0),
             rewriter.getI64IntegerAttr(0), rewriter.getI64IntegerAttr(1))
      .getResult(0);
}

} // namespace mlir::hip
