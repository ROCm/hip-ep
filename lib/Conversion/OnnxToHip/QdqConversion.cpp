/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// onnx.QuantizeLinear -> hip.quantize_linear
// y = saturate((x / scale) + zero_point)
// onnx.DequantizeLinear -> hip.dequantize_linear
// y = (x - zero_point) * scale
//
// com.microsoft QLinearAdd / QLinearMul / QLinearConcat /
// QLinearGlobalAveragePool are decomposed here into native
// DequantizeLinear + compute + QuantizeLinear, before PDLL fusion.
// Per-tensor Add and Mul then fuse to hip.qadd / hip.qmul.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// MorphiZen imports com.microsoft QuantizeLinear / DequantizeLinear function
/// ops as generic onnx.Custom operations. Canonicalize them to the native ONNX
/// operation names so that the QDQ lowering patterns below, which only match
/// the native spelling, work for both importer representations.
struct CustomQdqToNativeOnnx : public mlir::RewritePattern {
  CustomQdqToNativeOnnx(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto functionName = op->getAttrOfType<mlir::StringAttr>("function_name");
    if (!functionName)
      return mlir::failure();

    llvm::StringRef nativeOpName;
    if (functionName.getValue() == "QuantizeLinear")
      nativeOpName = "onnx.QuantizeLinear";
    else if (functionName.getValue() == "DequantizeLinear")
      nativeOpName = "onnx.DequantizeLinear";
    else
      return mlir::failure();

    mlir::OperationState state(op->getLoc(), nativeOpName);
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      llvm::StringRef name = attr.getName().getValue();
      if (name != "function_name" && name != "domain_name")
        state.addAttribute(attr.getName(), attr.getValue());
    }

    mlir::Operation *nativeOp = rewriter.create(state);
    rewriter.replaceOp(op, nativeOp->getResults());
    return mlir::success();
  }
};

mlir::Value getOptionalOperand(mlir::Operation *op, size_t idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value val = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(val.getType()))
    return nullptr;
  return val;
}

static int64_t getOnnxIntAttr(mlir::Operation *op, llvm::StringRef name,
                              int64_t defaultValue) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getValue().getSExtValue();
  return defaultValue;
}

struct QdqOperands {
  mlir::Value context;
  mlir::Value input;
  mlir::Value scale;
  mlir::Value zeroPoint;
  mlir::Value init;
  mlir::RankedTensorType resultType;
};

mlir::FailureOr<QdqOperands> matchQdqCommon(mlir::Operation *op,
                                            mlir::PatternRewriter &rewriter) {
  size_t numOperands = op->getNumOperands();
  if (numOperands < 2 || numOperands > 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 2-3 inputs and 1 output");

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked output");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "failed to get context argument");

  QdqOperands operands;
  operands.context = *ctxOrFailure;
  operands.input = op->getOperand(0);
  operands.scale = op->getOperand(1);
  operands.zeroPoint = getOptionalOperand(op, 2);
  operands.resultType = resultType;
  operands.init =
      createEmptyTensor(rewriter, op->getLoc(), resultType, operands.input);
  return operands;
}

struct QuantizeLinearToHip : public mlir::RewritePattern {
  QuantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.QuantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operandsOr = matchQdqCommon(op, rewriter);
    if (mlir::failed(operandsOr))
      return mlir::failure();
    const QdqOperands &in = *operandsOr;
    // read attributes with default values
    auto axisAttr = rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "axis", 1));
    auto blockSizeAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "block_size", 0));
    auto precisionAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "precision", 0));
    auto saturateAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "saturate", 1));

    auto hipOp = mlir::hip::QuantizeLinearOp::create(
        rewriter, op->getLoc(), mlir::TypeRange{in.resultType}, in.context,
        in.input, in.scale, in.zeroPoint, in.init, axisAttr, blockSizeAttr,
        precisionAttr, saturateAttr);
    // Nothing stamps `packed_int4` onto a QuantizeLinear yet -- unlike a
    // packed constant, a 4-bit target leaves no trace in the IR to detect.
    // Forwarded so the lowering and runtime below are reachable once one does.
    if (op->hasAttr(kPackedInt4Attr))
      hipOp->setAttr(kPackedInt4Attr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

struct DequantizeLinearToHip : public mlir::RewritePattern {
  DequantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operandsOr = matchQdqCommon(op, rewriter);
    if (mlir::failed(operandsOr))
      return mlir::failure();
    const QdqOperands &in = *operandsOr;

    auto axisAttr = rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "axis", 1));
    auto blockSizeAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "block_size", 0));

    auto hipOp = mlir::hip::DequantizeLinearOp::create(
        rewriter, op->getLoc(), mlir::TypeRange{in.resultType}, in.context,
        in.input, in.scale, in.zeroPoint, in.init, axisAttr, blockSizeAttr);
    // Carried over verbatim: constant lowering already decided this, and the
    // types here cannot show it.
    if (op->hasAttr(kPackedInt4Attr))
      hipOp->setAttr(kPackedInt4Attr, rewriter.getUnitAttr());
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// com.microsoft QLinear* -> DequantizeLinear + compute + QuantizeLinear
//===----------------------------------------------------------------------===//
//
// Operand layouts follow onnxruntime's contrib schemas. Optional zero points
// arrive as onnx.NoValue and are omitted, which the QDQ fusion treats as 0.
// This runs in the pre-PDLL canonicalization set, so Add and Mul fuse.

bool isMicrosoftFunction(mlir::Operation *op, llvm::StringRef functionName) {
  auto functionAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  return functionAttr && functionAttr.getValue() == functionName &&
         domainAttr && domainAttr.getValue() == "com.microsoft";
}

/// Real operand at \p idx, or null when the operand is absent or onnx.NoValue.
mlir::Value realOperand(mlir::Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands())
    return {};
  mlir::Value value = op->getOperand(idx);
  if (!value || mlir::isa<mlir::NoneType>(value.getType()))
    return {};
  return value;
}

bool isInt8Element(mlir::Type elementType) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType);
  return intType && intType.getWidth() == 8;
}

mlir::RankedTensorType f32TypeOf(mlir::RankedTensorType type) {
  return mlir::RankedTensorType::get(type.getShape(),
                                     mlir::Float32Type::get(type.getContext()));
}

mlir::Value emitOnnxOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                       llvm::StringRef name, mlir::ValueRange operands,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs = {}) {
  mlir::OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(attrs);
  return rewriter.create(state)->getResult(0);
}

mlir::Value emitDequantize(mlir::PatternRewriter &rewriter, mlir::Location loc,
                           mlir::Value input, mlir::Value scale,
                           mlir::Value zeroPoint,
                           mlir::RankedTensorType floatType) {
  llvm::SmallVector<mlir::Value, 3> operands{input, scale};
  if (zeroPoint)
    operands.push_back(zeroPoint);
  return emitOnnxOp(rewriter, loc, "onnx.DequantizeLinear", operands,
                    floatType);
}

mlir::Value emitQuantize(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value input, mlir::Value scale,
                         mlir::Value zeroPoint, mlir::Type resultType) {
  llvm::SmallVector<mlir::Value, 3> operands{input, scale};
  if (zeroPoint)
    operands.push_back(zeroPoint);
  return emitOnnxOp(rewriter, loc, "onnx.QuantizeLinear", operands, resultType);
}

mlir::LogicalResult lowerQLinearMath(mlir::Operation *op,
                                     mlir::PatternRewriter &rewriter,
                                     llvm::StringRef functionName,
                                     llvm::StringRef computeOp) {
  if (!isMicrosoftFunction(op, functionName))
    return rewriter.notifyMatchFailure(op, "not the requested QLinear op");
  if (op->getNumResults() != 1 || op->getNumOperands() < 7 ||
      op->getNumOperands() > 8)
    return rewriter.notifyMatchFailure(
        op, "QLinear math expects 7 or 8 inputs and 1 output");

  // A, A_scale, A_zp?, B, B_scale, B_zp?, C_scale, C_zp?
  mlir::Value a = realOperand(op, 0);
  mlir::Value aScale = realOperand(op, 1);
  mlir::Value aZp = realOperand(op, 2);
  mlir::Value b = realOperand(op, 3);
  mlir::Value bScale = realOperand(op, 4);
  mlir::Value bZp = realOperand(op, 5);
  mlir::Value cScale = realOperand(op, 6);
  mlir::Value cZp = realOperand(op, 7);
  if (!a || !aScale || !b || !bScale || !cScale)
    return rewriter.notifyMatchFailure(op, "missing data or scale operand");

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!aType || !bType || !resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensors");
  if (!isInt8Element(aType.getElementType()) ||
      aType.getElementType() != bType.getElementType() ||
      aType.getElementType() != resultType.getElementType())
    return rewriter.notifyMatchFailure(op, "QLinear math is int8 or uint8");

  mlir::Location loc = op->getLoc();
  mlir::Value aFloat =
      emitDequantize(rewriter, loc, a, aScale, aZp, f32TypeOf(aType));
  mlir::Value bFloat =
      emitDequantize(rewriter, loc, b, bScale, bZp, f32TypeOf(bType));
  // The custom result shape is the broadcast of A and B.
  mlir::Value sum = emitOnnxOp(rewriter, loc, computeOp, {aFloat, bFloat},
                               f32TypeOf(resultType));
  mlir::Value quantized =
      emitQuantize(rewriter, loc, sum, cScale, cZp, resultType);
  rewriter.replaceOp(op, quantized);
  return mlir::success();
}

struct QLinearAddToQdq : public mlir::RewritePattern {
  QLinearAddToQdq(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return lowerQLinearMath(op, rewriter, "QLinearAdd", "onnx.Add");
  }
};

struct QLinearMulToQdq : public mlir::RewritePattern {
  QLinearMulToQdq(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return lowerQLinearMath(op, rewriter, "QLinearMul", "onnx.Mul");
  }
};

struct QLinearConcatToQdq : public mlir::RewritePattern {
  QLinearConcatToQdq(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isMicrosoftFunction(op, "QLinearConcat"))
      return rewriter.notifyMatchFailure(op, "not QLinearConcat");
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected one result");

    // Y_scale, Y_zero_point, then (tensor, scale, zero_point) triples.
    unsigned numOperands = op->getNumOperands();
    if (numOperands < 5 || (numOperands - 2) % 3 != 0)
      return rewriter.notifyMatchFailure(
          op, "QLinearConcat inputs must be scale, zero point, then triples");

    auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
    if (!axisAttr)
      return rewriter.notifyMatchFailure(op, "missing axis");
    int64_t axis = axisAttr.getValue().getSExtValue();

    mlir::Value yScale = realOperand(op, 0);
    mlir::Value yZp = realOperand(op, 1);
    if (!yScale)
      return rewriter.notifyMatchFailure(op, "missing Y_scale");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !isInt8Element(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "result must be ranked int8");

    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Value> scales;
    llvm::SmallVector<mlir::Value> zeroPoints;
    llvm::SmallVector<mlir::RankedTensorType> inputTypes;
    for (unsigned i = 2; i < numOperands; i += 3) {
      mlir::Value input = realOperand(op, i);
      mlir::Value scale = realOperand(op, i + 1);
      if (!input || !scale)
        return rewriter.notifyMatchFailure(op, "incomplete concat triple");
      auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      // Contrib QLinearConcat allows each input to be int8 or uint8
      // independently of the output. Inputs are dequantized before the
      // concat, so only the 8-bit width matters here.
      if (!inputType || !isInt8Element(inputType.getElementType()))
        return rewriter.notifyMatchFailure(
            op, "concat inputs must be ranked int8 or uint8");
      inputs.push_back(input);
      scales.push_back(scale);
      zeroPoints.push_back(realOperand(op, i + 2));
      inputTypes.push_back(inputType);
    }

    auto si64 = mlir::IntegerType::get(rewriter.getContext(), 64,
                                       mlir::IntegerType::Signed);
    mlir::Location loc = op->getLoc();
    llvm::SmallVector<mlir::Value> dequantized;
    dequantized.reserve(inputs.size());
    for (auto [input, scale, zeroPoint, inputType] :
         llvm::zip(inputs, scales, zeroPoints, inputTypes)) {
      dequantized.push_back(emitDequantize(rewriter, loc, input, scale,
                                           zeroPoint, f32TypeOf(inputType)));
    }
    mlir::Value concatenated = emitOnnxOp(
        rewriter, loc, "onnx.Concat", dequantized, f32TypeOf(resultType),
        {rewriter.getNamedAttr("axis", mlir::IntegerAttr::get(si64, axis))});
    mlir::Value quantized =
        emitQuantize(rewriter, loc, concatenated, yScale, yZp, resultType);
    rewriter.replaceOp(op, quantized);
    return mlir::success();
  }
};

struct QLinearGlobalAveragePoolToQdq : public mlir::RewritePattern {
  QLinearGlobalAveragePoolToQdq(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isMicrosoftFunction(op, "QLinearGlobalAveragePool"))
      return rewriter.notifyMatchFailure(op, "not QLinearGlobalAveragePool");
    if (op->getNumResults() != 1 || op->getNumOperands() != 5)
      return rewriter.notifyMatchFailure(
          op, "expected X, x_scale, x_zero_point, y_scale, y_zero_point");

    mlir::Value x = realOperand(op, 0);
    mlir::Value xScale = realOperand(op, 1);
    mlir::Value xZp = realOperand(op, 2);
    mlir::Value yScale = realOperand(op, 3);
    mlir::Value yZp = realOperand(op, 4);
    if (!x || !xScale || !yScale)
      return rewriter.notifyMatchFailure(op, "missing data or scale operand");

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");
    int64_t rank = inputType.getRank();
    if (rank < 3 || resultType.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "expected rank >= 3");
    if (!isInt8Element(inputType.getElementType()) ||
        inputType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(op, "pool is int8 or uint8");

    int64_t channelsLast = getOnnxIntAttr(op, "channels_last", 0);
    if (channelsLast != 0 && channelsLast != 1)
      return rewriter.notifyMatchFailure(op, "channels_last must be 0 or 1");

    auto inputShape = inputType.getShape();
    auto resultShape = resultType.getShape();
    auto compatible = [](int64_t lhs, int64_t rhs) {
      return lhs == mlir::ShapedType::kDynamic ||
             rhs == mlir::ShapedType::kDynamic || lhs == rhs;
    };
    if (!compatible(inputShape[0], resultShape[0]))
      return rewriter.notifyMatchFailure(op, "batch dim mismatch");
    int64_t channelDim = channelsLast ? inputShape[rank - 1] : inputShape[1];
    int64_t resultChannel =
        channelsLast ? resultShape[rank - 1] : resultShape[1];
    if (!compatible(channelDim, resultChannel))
      return rewriter.notifyMatchFailure(op, "channel dim mismatch");
    for (int64_t i = 0; i < rank; ++i) {
      bool spatial = channelsLast ? (i > 0 && i + 1 < rank) : i >= 2;
      if (spatial && resultShape[i] != mlir::ShapedType::kDynamic &&
          resultShape[i] != 1)
        return rewriter.notifyMatchFailure(op, "pooled spatial dims must be 1");
    }

    mlir::Location loc = op->getLoc();
    mlir::Value dequantized =
        emitDequantize(rewriter, loc, x, xScale, xZp, f32TypeOf(inputType));

    // GlobalAveragePoolToHip copies a dynamic result dim from the input at
    // the same index, and the pool lowering assumes spatial sizes are the
    // static value 1. Force N/C through and every spatial dim to 1, even
    // when the quantized result still says '?' for those axes.
    auto pooledType = [&](int64_t n, int64_t c) {
      llvm::SmallVector<int64_t> shape(rank, 1);
      shape[0] = n;
      shape[1] = c;
      return mlir::RankedTensorType::get(shape, rewriter.getF32Type());
    };

    mlir::Value poolInput = dequantized;
    mlir::RankedTensorType poolType = pooledType(inputShape[0], inputShape[1]);
    if (channelsLast) {
      // NHWC -> NCHW so GlobalAveragePool reduces the spatial axes.
      llvm::SmallVector<int64_t> toChannelsFirst;
      toChannelsFirst.push_back(0);
      toChannelsFirst.push_back(rank - 1);
      for (int64_t i = 1; i < rank - 1; ++i)
        toChannelsFirst.push_back(i);
      llvm::SmallVector<int64_t> nchwShape(rank);
      for (auto [i, src] : llvm::enumerate(toChannelsFirst))
        nchwShape[i] = inputShape[src];
      poolInput = emitOnnxOp(
          rewriter, loc, "onnx.Transpose", poolInput,
          mlir::RankedTensorType::get(nchwShape, rewriter.getF32Type()),
          {rewriter.getNamedAttr("perm",
                                 rewriter.getI64ArrayAttr(toChannelsFirst))});
      poolType = pooledType(nchwShape[0], nchwShape[1]);
    }

    mlir::Value pooled = emitOnnxOp(rewriter, loc, "onnx.GlobalAveragePool",
                                    poolInput, poolType);

    mlir::Value toQuantize = pooled;
    if (channelsLast) {
      // NCHW pooled [N, C, 1, ...] -> channels-last [N, 1, ..., C].
      llvm::SmallVector<int64_t> toChannelsLast;
      toChannelsLast.push_back(0);
      for (int64_t i = 2; i < rank; ++i)
        toChannelsLast.push_back(i);
      toChannelsLast.push_back(1);
      auto pooledShape = poolType.getShape();
      llvm::SmallVector<int64_t> channelsLastShape(rank, 1);
      channelsLastShape[0] = pooledShape[0];
      channelsLastShape[rank - 1] = pooledShape[1];
      toQuantize = emitOnnxOp(
          rewriter, loc, "onnx.Transpose", pooled,
          mlir::RankedTensorType::get(channelsLastShape, rewriter.getF32Type()),
          {rewriter.getNamedAttr("perm",
                                 rewriter.getI64ArrayAttr(toChannelsLast))});
    }

    mlir::Value quantized =
        emitQuantize(rewriter, loc, toQuantize, yScale, yZp, resultType);
    rewriter.replaceOp(op, quantized);
    return mlir::success();
  }
};

} // namespace

void populateCustomQdqCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<CustomQdqToNativeOnnx, QLinearAddToQdq, QLinearMulToQdq,
               QLinearConcatToQdq, QLinearGlobalAveragePoolToQdq>(ctx);
}

void populateQdqConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<QuantizeLinearToHip, DequantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
