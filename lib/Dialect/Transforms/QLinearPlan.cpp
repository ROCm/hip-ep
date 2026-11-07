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

// The per-tensor input scale times each weight scale, over the output scale,
// in f32 and in ORT CPU's order. `channelShape` is the broadcast shape of a
// per-channel multiplier.
static DenseElementsAttr getRequantMultiplier(DenseElementsAttr inputScale,
                                              ArrayRef<float> weightScales,
                                              DenseElementsAttr outputScale,
                                              ArrayRef<int64_t> channelShape) {
  float xScale = *inputScale.getValues<float>().begin();
  float yScale = *outputScale.getValues<float>().begin();
  SmallVector<float> multipliers = llvm::map_to_vector(
      weightScales, [&](float wScale) { return xScale * wScale / yScale; });
  if (!llvm::all_of(multipliers, [](float m) { return std::isfinite(m); }))
    return {};
  Type f32 = Float32Type::get(inputScale.getContext());
  if (llvm::all_equal(multipliers))
    return DenseElementsAttr::get(RankedTensorType::get({}, f32),
                                  multipliers.front());
  return DenseElementsAttr::get(RankedTensorType::get(channelShape, f32),
                                ArrayRef<float>(multipliers));
}

static DenseElementsAttr getScalarZeroPoint(DenseElementsAttr zp,
                                            Type elementType) {
  return DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                *zp.getValues<APInt>().begin());
}

// The i32 bias of an integer conv on a rebiased input: B - z' * sum(w) over
// the taps that each output reads, with z' = x_zp - offset. Null when every
// entry is 0. The weights must be inline only when z' is not 0.
static FailureOr<DenseElementsAttr>
getConvCorrectedBias(RankedTensorType inputType, RankedTensorType weightType,
                     RankedTensorType resultType, int64_t inputZeroPoint,
                     DenseElementsAttr weights, ArrayRef<int64_t> biasValues,
                     ArrayAttr strides, ArrayAttr dilations, ArrayAttr pads) {
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
  auto correctedBias = [&](int64_t k, int64_t oh, int64_t ow) {
    auto [r0, r1] = rowTaps[oh];
    auto [s0, s1] = colTaps[ow];
    int64_t sum = 0;
    for (int64_t r = r0; r <= r1; ++r)
      for (int64_t s = s0; s <= s1; ++s)
        sum += tapSums[k * taps + r * wShape[3] + s];
    return biasValues[k] - zpShift * sum;
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

  FailureOr<DenseElementsAttr> correctedBias = getConvCorrectedBias(
      inputType, weightType, resultType, getIntValues(inputZp).front(), weights,
      biasValues, op.getStrides(), op.getDilations(), op.getPads());
  if (failed(correctedBias))
    return std::nullopt;

  QLinearPlan plan;
  plan.bias = *correctedBias;
  plan.multiplier = getRequantMultiplier(inputScale, weightScales, outputScale,
                                         {1, channels, 1, 1});
  if (!plan.multiplier)
    return std::nullopt;
  plan.outputZeroPoint = getScalarZeroPoint(outputZp, outputElement);
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

// The matmul form of the plan: matmul(a - a_zp, b) = matmul(a - offset, b)
// - z' * colsum(b), with one bias value per output column. B must be an inline
// constant only when z' is not 0.
std::optional<QLinearPlan> planQLinearMatMul(QLinearMatMulOp op) {
  auto aType = cast<RankedTensorType>(op.getA().getType());
  auto bType = cast<RankedTensorType>(op.getB().getType());
  auto resultType = cast<RankedTensorType>(op.getOutput().getType());
  Type aElement = aType.getElementType();
  Type outputElement = resultType.getElementType();
  if (!isQuantizedByte(aElement) || !isQuantizedByte(outputElement) ||
      !bType.getElementType().isSignlessInteger(8))
    return std::nullopt;

  DenseElementsAttr aScale = getInlineDense(op.getAScale());
  DenseElementsAttr aZp = getInlineDense(op.getAZeroPoint());
  DenseElementsAttr bScale = getInlineDense(op.getBScale());
  DenseElementsAttr bZp = getInlineDense(op.getBZeroPoint());
  DenseElementsAttr yScale = getInlineDense(op.getYScale());
  DenseElementsAttr yZp = getInlineDense(op.getYZeroPoint());
  if (!aScale || !aZp || !bScale || !bZp || !yScale || !yZp)
    return std::nullopt;
  if (llvm::any_of(getIntValues(bZp), [](int64_t v) { return v != 0; }))
    return std::nullopt;

  int64_t n = bType.getDimSize(1);
  int64_t offset = getSignedRebiasOffset(aElement);
  int64_t zpShift = getIntValues(aZp).front() - offset;

  QLinearPlan plan;
  if (zpShift != 0) {
    DenseElementsAttr b = getInlineDense(op.getB());
    if (!b)
      return std::nullopt;
    SmallVector<int64_t> columnSums(n, 0);
    int64_t index = 0;
    for (const APInt &v : b.getValues<APInt>())
      columnSums[index++ % n] += v.getSExtValue();
    SmallVector<int32_t> biasData;
    biasData.reserve(n);
    for (int64_t sum : columnSums) {
      int64_t value = -zpShift * sum;
      if (value < std::numeric_limits<int32_t>::min() ||
          value > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      biasData.push_back(static_cast<int32_t>(value));
    }
    if (llvm::any_of(biasData, [](int32_t v) { return v != 0; }))
      plan.bias = DenseElementsAttr::get(
          RankedTensorType::get({1, n}, IntegerType::get(op.getContext(), 32)),
          ArrayRef<int32_t>(biasData));
  }

  SmallVector<float> bScales = llvm::to_vector(bScale.getValues<float>());
  plan.multiplier = getRequantMultiplier(aScale, bScales, yScale, {1, n});
  if (!plan.multiplier)
    return std::nullopt;
  plan.outputZeroPoint = getScalarZeroPoint(yZp, outputElement);
  return plan;
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

// The epilogue that both QLinear rewrites share: add the corrected bias, cast
// to f32, scale, and quantize with unit scale, which rounds half to even,
// adds the zero point, and saturates, as ORT CPU does.
Value emitTensorEmpty(PatternRewriter &rewriter, Location loc,
                      RankedTensorType type) {
  return tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                 type.getElementType());
}

Value emitConstant(PatternRewriter &rewriter, Location loc,
                   DenseElementsAttr value) {
  return ConstantOp::create(rewriter, loc,
                            cast<RankedTensorType>(value.getType()), value);
}

// (acc + bias) in i32, then cast to f32 and multiplied by `multiplier`.
Value emitScaledAccumulator(PatternRewriter &rewriter, Location loc, Value ctx,
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

Value emitRequantizeEpilogue(PatternRewriter &rewriter, Location loc, Value ctx,
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
