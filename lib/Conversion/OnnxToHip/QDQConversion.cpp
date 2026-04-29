/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX QuantizeLinear / DequantizeLinear support.
//
// Runtime Q/DQ currently decomposes into existing HIP cast/elementwise ops.
// Constant weight DQ folds to normal float constants so Conv/Gemm/MatMul can
// keep using the validated float kernels.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static Value resolveOptional(Value v) {
  if (!v || isa<NoneType>(v.getType()))
    return nullptr;
  if (Operation *def = v.getDefiningOp())
    if (def->getName().getStringRef() == "onnx.NoValue")
      return nullptr;
  return v;
}

static ElementsAttr getConstElements(Value v) {
  if (!v)
    return {};
  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  if (def->getName().getStringRef() == "onnx.Constant")
    return def->getAttrOfType<ElementsAttr>("value");
  if (auto c = dyn_cast<arith::ConstantOp>(def))
    return dyn_cast<ElementsAttr>(c.getValue());
  return {};
}

static FailureOr<SmallVector<int64_t>> readIntegerConst(Value v) {
  auto elems = dyn_cast_or_null<DenseElementsAttr>(getConstElements(v));
  if (!elems)
    return failure();
  SmallVector<int64_t> out;
  out.reserve(elems.getNumElements());
  auto elemType = elems.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    for (APInt val : elems.getValues<APInt>())
      out.push_back(intType.isUnsigned() ? static_cast<int64_t>(val.getZExtValue())
                                         : val.getSExtValue());
    return out;
  }
  return failure();
}

static FailureOr<SmallVector<double>> readFloatConst(Value v) {
  auto elems = dyn_cast_or_null<DenseElementsAttr>(getConstElements(v));
  if (!elems)
    return failure();
  SmallVector<double> out;
  out.reserve(elems.getNumElements());
  if (!isa<FloatType>(elems.getElementType()))
    return failure();
  for (APFloat val : elems.getValues<APFloat>())
    out.push_back(val.convertToDouble());
  return out;
}

static Value makeScalarZeroPoint(PatternRewriter &rewriter, Location loc,
                                 Type elemType) {
  auto type = RankedTensorType::get({}, elemType);
  auto intType = cast<IntegerType>(elemType);
  Attribute zero = IntegerAttr::get(intType, 0);
  auto attr = DenseElementsAttr::get(type, zero);
  return arith::ConstantOp::create(rewriter, loc, type, attr);
}

static Value buildCastTo(PatternRewriter &rewriter, Location loc, Value context,
                         Value input, RankedTensorType outType,
                         int64_t onnxDataType) {
  Value init = createEmptyTensor(rewriter, loc, outType, input);
  return CastOp::create(rewriter, loc, outType, context, input, init,
                        rewriter.getI64IntegerAttr(onnxDataType))
      .getResult(0);
}

static RankedTensorType toF32TensorType(RankedTensorType type,
                                        PatternRewriter &rewriter) {
  return RankedTensorType::get(type.getShape(), rewriter.getF32Type());
}

static Value buildDequantize(PatternRewriter &rewriter, Location loc,
                             Value context, Value input, Value scale,
                             Value zeroPoint, RankedTensorType outType) {
  if (!zeroPoint)
    zeroPoint = makeScalarZeroPoint(
        rewriter, loc, cast<RankedTensorType>(input.getType()).getElementType());
  auto inputType = cast<RankedTensorType>(input.getType());
  auto zpType = cast<RankedTensorType>(zeroPoint.getType());
  auto f32InputType = toF32TensorType(inputType, rewriter);
  auto f32ZpType = toF32TensorType(zpType, rewriter);
  auto f32OutType = toF32TensorType(outType, rewriter);

  Value inputF32 = buildCastTo(rewriter, loc, context, input, f32InputType, 1);
  Value zpF32 = buildCastTo(rewriter, loc, context, zeroPoint, f32ZpType, 1);
  Value subInit =
      createBroadcastEmptyTensor(rewriter, loc, f32InputType, {inputF32, zpF32});
  Value centered =
      SubOp::create(rewriter, loc, f32InputType, context, inputF32, zpF32,
                    subInit)
          .getResult(0);
  Value mulInit =
      createBroadcastEmptyTensor(rewriter, loc, f32OutType, {centered, scale});
  Value deq =
      MulOp::create(rewriter, loc, f32OutType, context, centered, scale,
                    mulInit)
          .getResult(0);
  if (outType.getElementType().isF32())
    return deq;
  return buildCastTo(rewriter, loc, context, deq, outType,
                     outType.getElementType().isF16() ? 10 : 16);
}

static Value buildQuantize(PatternRewriter &rewriter, Location loc,
                           Value context, Value input, Value scale,
                           Value zeroPoint, RankedTensorType outType) {
  if (!zeroPoint)
    zeroPoint = makeScalarZeroPoint(rewriter, loc, outType.getElementType());
  auto inputType = cast<RankedTensorType>(input.getType());
  auto zpType = cast<RankedTensorType>(zeroPoint.getType());
  auto f32InputType = toF32TensorType(inputType, rewriter);
  auto f32ZpType = toF32TensorType(zpType, rewriter);
  Value inputF32 = inputType.getElementType().isF32()
                       ? input
                       : buildCastTo(rewriter, loc, context, input, f32InputType,
                                     1);
  Value divInit =
      createBroadcastEmptyTensor(rewriter, loc, f32InputType, {inputF32, scale});
  Value scaled =
      BinaryElementwiseOp::create(rewriter, loc, f32InputType, context,
                                  inputF32, scale, divInit,
                                  rewriter.getI64IntegerAttr(0))
          .getResult(0);
  Value zpF32 = buildCastTo(rewriter, loc, context, zeroPoint, f32ZpType, 1);
  Value addInit =
      createBroadcastEmptyTensor(rewriter, loc, f32InputType, {scaled, zpF32});
  Value shifted =
      AddOp::create(rewriter, loc, f32InputType, context, scaled, zpF32,
                    addInit)
          .getResult(0);
  Value roundInit = createEmptyTensor(rewriter, loc, f32InputType, shifted);
  Value rounded =
      UnaryElementwiseOp::create(rewriter, loc, f32InputType, context, shifted,
                                 roundInit, rewriter.getI64IntegerAttr(5),
                                 rewriter.getF32FloatAttr(0.0f),
                                 rewriter.getF32FloatAttr(0.0f))
          .getResult(0);

  float clipMin = -128.0f;
  float clipMax = 127.0f;
  if (auto intType = dyn_cast<IntegerType>(outType.getElementType())) {
    if (intType.getSignedness() == IntegerType::Unsigned) {
      clipMin = 0.0f;
      clipMax = 255.0f;
    }
  }
  Value clipInit = createEmptyTensor(rewriter, loc, f32InputType, rounded);
  Value clipped =
      UnaryElementwiseOp::create(rewriter, loc, f32InputType, context, rounded,
                                 clipInit, rewriter.getI64IntegerAttr(8),
                                 rewriter.getF32FloatAttr(clipMin),
                                 rewriter.getF32FloatAttr(clipMax))
          .getResult(0);
  int64_t outOnnxType = 3;
  if (auto intType = dyn_cast<IntegerType>(outType.getElementType()))
    if (intType.getSignedness() == IntegerType::Unsigned)
      outOnnxType = 2;
  return buildCastTo(rewriter, loc, context, clipped, outType, outOnnxType);
}

static Value buildCenteredQuantizedToF32(PatternRewriter &rewriter,
                                         Location loc, Value context,
                                         Value input, Value zeroPoint,
                                         RankedTensorType outType) {
  if (!zeroPoint)
    zeroPoint = makeScalarZeroPoint(
        rewriter, loc, cast<RankedTensorType>(input.getType()).getElementType());
  auto inputType = cast<RankedTensorType>(input.getType());
  auto zpType = cast<RankedTensorType>(zeroPoint.getType());
  auto f32InputType = toF32TensorType(inputType, rewriter);
  auto f32ZpType = toF32TensorType(zpType, rewriter);
  Value inputF32 = buildCastTo(rewriter, loc, context, input, f32InputType, 1);
  Value zpF32 = buildCastTo(rewriter, loc, context, zeroPoint, f32ZpType, 1);
  Value subInit =
      createBroadcastEmptyTensor(rewriter, loc, outType, {inputF32, zpF32});
  return SubOp::create(rewriter, loc, outType, context, inputF32, zpF32,
                       subInit)
      .getResult(0);
}

static LogicalResult lowerQDQ(Operation *op, PatternRewriter &rewriter) {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();

  if (op->getNumOperands() < 2 || op->getNumOperands() > 3)
    return rewriter.notifyMatchFailure(
        op, "Q/DQ expects 2 or 3 operands: x, scale [, zero_point]");

  Value input = op->getOperand(0);
  Value scale = op->getOperand(1);
  Value zeroPoint =
      op->getNumOperands() == 3 ? resolveOptional(op->getOperand(2)) : nullptr;

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType)
    return rewriter.notifyMatchFailure(op, "Q/DQ requires ranked tensors");
  Value repl;
  if (op->getName().getStringRef() == "onnx.DequantizeLinear")
    repl = buildDequantize(rewriter, op->getLoc(), *ctxOrFailure, input, scale,
                           zeroPoint, resultType);
  else
    repl = buildQuantize(rewriter, op->getLoc(), *ctxOrFailure, input, scale,
                         zeroPoint, resultType);
  rewriter.replaceOp(op, repl);
  return success();
}

struct DequantizeLinearToHip : public RewritePattern {
  DequantizeLinearToHip(MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return lowerQDQ(op, rewriter);
  }
};

struct QuantizeLinearToHip : public RewritePattern {
  QuantizeLinearToHip(MLIRContext *ctx)
      : RewritePattern("onnx.QuantizeLinear", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return lowerQDQ(op, rewriter);
  }
};

struct QLinearMatMulToHip : public RewritePattern {
  QLinearMatMulToHip(MLIRContext *ctx)
      : RewritePattern("onnx.QLinearMatMul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 8)
      return failure();
    auto ctxOr = getContextArg(op, rewriter);
    if (failed(ctxOr))
      return failure();
    Location loc = op->getLoc();
    Value a = op->getOperand(0), aScale = op->getOperand(1);
    Value aZp = resolveOptional(op->getOperand(2));
    Value b = op->getOperand(3), bScale = op->getOperand(4);
    Value bZp = resolveOptional(op->getOperand(5));
    Value yScale = op->getOperand(6);
    Value yZp = resolveOptional(op->getOperand(7));

    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    auto yType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!aType || !bType || !yType)
      return failure();
    auto f32 = rewriter.getF32Type();
    auto aF32Type = RankedTensorType::get(aType.getShape(), f32);
    auto bF32Type = RankedTensorType::get(bType.getShape(), f32);
    auto matmulF32Type = RankedTensorType::get(yType.getShape(), f32);

    Value aF32 = buildDequantize(rewriter, loc, *ctxOr, a, aScale, aZp,
                                 aF32Type);
    Value bF32 = buildDequantize(rewriter, loc, *ctxOr, b, bScale, bZp,
                                 bF32Type);
    Value mmInit = createEmptyTensor(rewriter, loc, matmulF32Type, aF32);
    Value mm = MatmulOp::create(rewriter, loc, matmulF32Type, *ctxOr, aF32,
                                bF32, mmInit)
                   .getResult(0);
    Value q = buildQuantize(rewriter, loc, *ctxOr, mm, yScale, yZp, yType);
    rewriter.replaceOp(op, q);
    return success();
  }
};

struct MatMulIntegerToHip : public RewritePattern {
  MatMulIntegerToHip(MLIRContext *ctx)
      : RewritePattern("onnx.MatMulInteger", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2)
      return failure();
    auto ctxOr = getContextArg(op, rewriter);
    if (failed(ctxOr))
      return failure();
    Location loc = op->getLoc();
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value aZp = op->getNumOperands() > 2 ? resolveOptional(op->getOperand(2))
                                         : nullptr;
    Value bZp = op->getNumOperands() > 3 ? resolveOptional(op->getOperand(3))
                                         : nullptr;
    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    auto yType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!aType || !bType || !yType)
      return failure();
    auto f32 = rewriter.getF32Type();
    auto aF32Type = RankedTensorType::get(aType.getShape(), f32);
    auto bF32Type = RankedTensorType::get(bType.getShape(), f32);
    auto mmF32Type = RankedTensorType::get(yType.getShape(), f32);
    Value aF32 =
        buildCenteredQuantizedToF32(rewriter, loc, *ctxOr, a, aZp, aF32Type);
    Value bF32 =
        buildCenteredQuantizedToF32(rewriter, loc, *ctxOr, b, bZp, bF32Type);
    Value mmInit = createEmptyTensor(rewriter, loc, mmF32Type, aF32);
    Value mm = MatmulOp::create(rewriter, loc, mmF32Type, *ctxOr, aF32, bF32,
                                mmInit)
                   .getResult(0);
    Value castInit = createEmptyTensor(rewriter, loc, yType, mm);
    Value cast = CastOp::create(rewriter, loc, yType, *ctxOr, mm, castInit,
                                rewriter.getI64IntegerAttr(6))
                     .getResult(0);
    rewriter.replaceOp(op, cast);
    return success();
  }
};

struct QLinearConvToHip : public RewritePattern {
  QLinearConvToHip(MLIRContext *ctx)
      : RewritePattern("onnx.QLinearConv", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 8)
      return failure();
    if (op->getNumOperands() > 8 && resolveOptional(op->getOperand(8)))
      return rewriter.notifyMatchFailure(
          op, "QLinearConv bias is not implemented yet");
    auto ctxOr = getContextArg(op, rewriter);
    if (failed(ctxOr))
      return failure();
    Location loc = op->getLoc();
    Value x = op->getOperand(0), xScale = op->getOperand(1);
    Value xZp = resolveOptional(op->getOperand(2));
    Value w = op->getOperand(3), wScale = op->getOperand(4);
    Value wZp = resolveOptional(op->getOperand(5));
    Value yScale = op->getOperand(6);
    Value yZp = resolveOptional(op->getOperand(7));
    auto xType = dyn_cast<RankedTensorType>(x.getType());
    auto wType = dyn_cast<RankedTensorType>(w.getType());
    auto yType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!xType || !wType || !yType)
      return failure();
    auto f32 = rewriter.getF32Type();
    auto xF32Type = RankedTensorType::get(xType.getShape(), f32);
    auto wF32Type = RankedTensorType::get(wType.getShape(), f32);
    auto convF32Type = RankedTensorType::get(yType.getShape(), f32);
    Value xF32 =
        buildDequantize(rewriter, loc, *ctxOr, x, xScale, xZp, xF32Type);
    Value wF32 =
        buildDequantize(rewriter, loc, *ctxOr, w, wScale, wZp, wF32Type);
    Value convInit = createEmptyTensor(rewriter, loc, convF32Type, xF32);
    SmallVector<int64_t> kernelShape;
    for (int64_t i = 2; i < wType.getRank(); ++i)
      kernelShape.push_back(wType.getDimSize(i));
    SmallVector<int64_t> strides(kernelShape.size(), 1);
    SmallVector<int64_t> pads(kernelShape.size() * 2, 0);
    SmallVector<int64_t> dilations(kernelShape.size(), 1);
    int64_t group = 1;
    SmallVector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("kernel_shape",
                                          rewriter.getI64ArrayAttr(kernelShape)));
    attrs.push_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides)));
    attrs.push_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads)));
    attrs.push_back(rewriter.getNamedAttr(
        "dilations", rewriter.getI64ArrayAttr(dilations)));
    attrs.push_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));
    SmallVector<Value> operands = {*ctxOr, xF32, wF32, convInit};
    Value conv =
        ConvOp::create(rewriter, loc, TypeRange{convF32Type}, operands, attrs)
            .getResult(0);
    Value q = buildQuantize(rewriter, loc, *ctxOr, conv, yScale, yZp, yType);
    rewriter.replaceOp(op, q);
    return success();
  }
};

struct FoldConstantDequantizeLinear : public RewritePattern {
  FoldConstantDequantizeLinear(MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 3)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !isa<FloatType>(resultType.getElementType()))
      return failure();

    auto xOr = readIntegerConst(op->getOperand(0));
    auto scaleOr = readFloatConst(op->getOperand(1));
    if (failed(xOr) || failed(scaleOr) || scaleOr->empty())
      return failure();

    SmallVector<int64_t> zp;
    if (op->getNumOperands() == 3 && resolveOptional(op->getOperand(2))) {
      auto zpOr = readIntegerConst(op->getOperand(2));
      if (failed(zpOr))
        return failure();
      zp = *zpOr;
    } else {
      zp.assign(scaleOr->size(), 0);
    }
    if (zp.empty())
      zp.push_back(0);

    auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return failure();
    int64_t inner = 1;
    int64_t axis = inputType.getRank() == 0 ? 0 : 1;
    for (int64_t i = axis + 1; i < inputType.getRank(); ++i) {
      if (inputType.isDynamicDim(i))
        return failure();
      inner *= inputType.getDimSize(i);
    }
    int64_t scaleN = scaleOr->size();

    SmallVector<Attribute> attrs;
    attrs.reserve(xOr->size());
    Type elemType = resultType.getElementType();
    for (int64_t i = 0, e = xOr->size(); i < e; ++i) {
      int64_t qidx = scaleN <= 1 ? 0 : ((i / inner) % scaleN);
      int64_t zero = zp.size() == 1 ? zp[0] : zp[qidx];
      double v = static_cast<double>((*xOr)[i] - zero) * (*scaleOr)[qidx];
      attrs.push_back(FloatAttr::get(elemType, v));
    }
    auto folded = DenseElementsAttr::get(resultType, attrs);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, folded);
    return success();
  }
};

} // namespace

void populateQDQConstantFoldPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<FoldConstantDequantizeLinear>(ctx);
}

void populateQDQConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<DequantizeLinearToHip, QuantizeLinearToHip, QLinearMatMulToHip,
               MatMulIntegerToHip, QLinearConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
