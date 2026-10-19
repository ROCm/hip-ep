/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ConvIntegerConversion.cpp - onnx.ConvInteger -> hip.conv -----------===//
//
// ONNX ConvInteger: y = conv(x - x_zero_point, w - w_zero_point), result i32.
// x and w are int8 or uint8. x_zero_point is a scalar (default 0).
// w_zero_point is a scalar or a length-M vector, one value per output
// channel (default 0).
//
// hip.conv of f32 lowers through the convolution kernel and through
// tosa.conv2d. That accumulator is an exact int32 dot product only while
// every partial sum fits in the 24-bit mantissa. An 8-bit value minus an
// 8-bit zero point is at most 255 in magnitude, so each product is at most
// 65025 and the reduction length (C/group)*kH*kW must be at most 258. A
// longer reduction casts to i32 and accumulates there. ONNX allows the int32
// accumulator to wrap.
//
// A 1-D weight zero point is reshaped to [M, 1, 1, 1] before the subtract.
// hip.sub right-aligns unequal ranks into the 4D broadcast ABI, so a bare
// tensor<MxT> would land on W rather than the output-channel axis.
//
//   Before:
//     %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp)
//         : (tensor<NxCxHxWxui8>, tensor<MxCxKxKxui8>, ...) ->
//         tensor<NxMxH'xW'xi32>
//
//   After:
//     %xf = hip.cast %x to f32
//     %wf = hip.cast %w to f32
//     %xd = hip.sub %xf, %xzp
//     %wd = hip.sub %wf, reshape(%wzp, [M, 1, 1, 1])
//     %acc = hip.conv %xd, %wd
//     %y = hip.cast %acc to i32
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {
namespace {

static mlir::LogicalResult readI64Array(mlir::ArrayAttr attr,
                                        llvm::SmallVectorImpl<int64_t> &out) {
  for (mlir::Attribute value : attr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(value);
    if (!intAttr)
      return mlir::failure();
    out.push_back(intAttr.getValue().getSExtValue());
  }
  return mlir::success();
}

static bool isEightBit(mlir::Type type) { return type.isInteger(8); }

/// True when `255 * 255 * reductionLength` is an exact f32 integer.
static bool eightBitDotIsExactInF32(int64_t reductionLength) {
  constexpr int64_t kMaxAbsProduct = 255 * 255;
  constexpr int64_t kMantissaLimit = int64_t{1} << 24;
  return reductionLength >= 0 &&
         reductionLength <= kMantissaLimit / kMaxAbsProduct;
}

static mlir::Value realOperand(mlir::Operation *op, unsigned index) {
  if (index >= op->getNumOperands())
    return {};
  mlir::Value value = op->getOperand(index);
  if (!value || mlir::isa<mlir::NoneType>(value.getType()))
    return {};
  return value;
}

/// Scalar (rank 0 or a single-element rank-1 tensor) or a vector whose
/// length is the output-channel count. A rank-4 zero point is accepted when
/// it already broadcasts onto \p data.
static bool zeroPointShapeOk(mlir::RankedTensorType zpType,
                             mlir::RankedTensorType dataType, bool perChannel) {
  if (!zpType.hasStaticShape())
    return false;
  if (zpType.getRank() == 0)
    return true;
  if (zpType.getRank() == 1 && zpType.getDimSize(0) == 1)
    return true;
  if (!perChannel)
    return false;
  if (zpType.getRank() == 1 && zpType.getDimSize(0) == dataType.getDimSize(0))
    return true;
  if (zpType.getRank() != dataType.getRank())
    return false;
  for (int64_t i : llvm::seq<int64_t>(zpType.getRank())) {
    int64_t zpDim = zpType.getDimSize(i);
    int64_t dataDim = dataType.getDimSize(i);
    if (zpDim != 1 && zpDim != dataDim)
      return false;
  }
  return true;
}

static mlir::Value reshapeTo(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value src,
                             mlir::RankedTensorType dstType) {
  if (src.getType() == dstType)
    return src;
  llvm::SmallVector<mlir::Value> dims;
  for (int64_t dim : dstType.getShape())
    dims.push_back(mlir::arith::ConstantIndexOp::create(rewriter, loc, dim));
  auto shapeType =
      mlir::RankedTensorType::get({dstType.getRank()}, rewriter.getIndexType());
  mlir::Value shape =
      mlir::tensor::FromElementsOp::create(rewriter, loc, shapeType, dims);
  return mlir::tensor::ReshapeOp::create(rewriter, loc, dstType, src, shape)
      .getResult();
}

/// Broadcast \p zp onto \p data. A length-M vector becomes [M, 1, 1, 1].
/// A single element becomes a rank-0 scalar, which the 4D elementwise ABI
/// left-pads to 1x1x1x1.
static mlir::Value alignZeroPoint(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value zp,
                                  mlir::RankedTensorType dataType,
                                  bool perChannel) {
  auto zpType = mlir::cast<mlir::RankedTensorType>(zp.getType());
  if (zpType.getRank() == dataType.getRank())
    return zp;
  if (perChannel && zpType.getRank() == 1 && zpType.getDimSize(0) > 1) {
    llvm::SmallVector<int64_t> shape(dataType.getRank(), 1);
    shape[0] = zpType.getDimSize(0);
    auto expanded = mlir::RankedTensorType::get(shape, zpType.getElementType());
    return reshapeTo(rewriter, loc, zp, expanded);
  }
  auto scalar = mlir::RankedTensorType::get({}, zpType.getElementType());
  return reshapeTo(rewriter, loc, zp, scalar);
}

static mlir::Value castTo(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value context, mlir::Value input,
                          mlir::Type elemType, int64_t onnxTo) {
  auto srcType = mlir::cast<mlir::RankedTensorType>(input.getType());
  if (srcType.getElementType() == elemType)
    return input;
  auto dstType = mlir::RankedTensorType::get(srcType.getShape(), elemType);
  mlir::Value init = createEmptyTensor(rewriter, loc, dstType, input);
  auto casted = mlir::hip::CastOp::create(rewriter, loc, context, input, init,
                                          rewriter.getI64IntegerAttr(onnxTo));
  return casted->getResult(0);
}

static mlir::Value subtractZeroPoint(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value context,
                                     mlir::Value data, mlir::Value zp) {
  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, dataType, data);
  return mlir::hip::SubOp::create(rewriter, loc, context, data, zp, init)
      ->getResult(0);
}

/// onnx.ConvInteger -> hip.cast / hip.sub / hip.conv / hip.cast.
struct ConvIntegerToHip : public mlir::RewritePattern {
  ConvIntegerToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConvInteger", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ConvIntegerToHip::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  if (op->getNumResults() != 1 || op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger expects x, w, and one i32 result");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto weightType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightType || !resultType)
    return rewriter.notifyMatchFailure(op,
                                       "ConvInteger requires ranked tensors");
  // hip.sub's lowering passes a 4D broadcast descriptor and rejects rank > 4.
  // Rank 4 is the NCHW form used by quantized conv graphs.
  if (inputType.getRank() != 4 || weightType.getRank() != 4 ||
      resultType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger lowering supports rank-4 NCHW only");
  if (!isEightBit(inputType.getElementType()) ||
      !isEightBit(weightType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "ConvInteger x and w must be int8 or uint8");
  if (!resultType.getElementType().isInteger(32) ||
      resultType.getElementType().isUnsignedInteger(32))
    return rewriter.notifyMatchFailure(op, "ConvInteger result must be i32");
  if (!inputType.hasStaticShape() || !weightType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "ConvInteger lowering requires static shapes");

  mlir::Value xZp = realOperand(op, 2);
  mlir::Value wZp = realOperand(op, 3);
  if (xZp) {
    auto xZpType = mlir::dyn_cast<mlir::RankedTensorType>(xZp.getType());
    if (!xZpType || xZpType.getElementType() != inputType.getElementType() ||
        !zeroPointShapeOk(xZpType, inputType, /*perChannel=*/false))
      return rewriter.notifyMatchFailure(
          op, "ConvInteger x_zero_point must be a scalar of x's element type");
  }
  if (wZp) {
    auto wZpType = mlir::dyn_cast<mlir::RankedTensorType>(wZp.getType());
    if (!wZpType || wZpType.getElementType() != weightType.getElementType() ||
        !zeroPointShapeOk(wZpType, weightType, /*perChannel=*/true))
      return rewriter.notifyMatchFailure(
          op, "ConvInteger w_zero_point must be a scalar or a length-M vector "
              "of w's element type");
  }

  const int64_t spatialDims = 2;
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    if (mlir::failed(readI64Array(attr, kernelShape)))
      return rewriter.notifyMatchFailure(op, "ConvInteger kernel_shape");
  } else {
    kernelShape.push_back(weightType.getDimSize(2));
    kernelShape.push_back(weightType.getDimSize(3));
  }
  if (static_cast<int64_t>(kernelShape.size()) != spatialDims)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger kernel_shape must have one entry per spatial axis");

  llvm::SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    if (mlir::failed(readI64Array(attr, strides)))
      return rewriter.notifyMatchFailure(op, "ConvInteger strides");
  } else {
    strides.assign(spatialDims, 1);
  }
  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    if (mlir::failed(readI64Array(attr, dilations)))
      return rewriter.notifyMatchFailure(op, "ConvInteger dilations");
  } else {
    dilations.assign(spatialDims, 1);
  }
  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    if (mlir::failed(readI64Array(attr, pads)))
      return rewriter.notifyMatchFailure(op, "ConvInteger pads");
  } else {
    pads.assign(spatialDims * 2, 0);
  }
  if (static_cast<int64_t>(strides.size()) != spatialDims ||
      static_cast<int64_t>(dilations.size()) != spatialDims)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger strides/dilations arity does not match the spatial "
            "rank");
  for (int64_t stride : strides) {
    if (stride <= 0)
      return rewriter.notifyMatchFailure(op, "ConvInteger stride must be > 0");
  }
  for (int64_t dilation : dilations) {
    if (dilation <= 0)
      return rewriter.notifyMatchFailure(op,
                                         "ConvInteger dilation must be > 0");
  }
  for (int64_t kernel : kernelShape) {
    if (kernel <= 0)
      return rewriter.notifyMatchFailure(op, "ConvInteger kernel must be > 0");
  }

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();
  if (group < 1)
    return rewriter.notifyMatchFailure(op, "ConvInteger group must be >= 1");

  llvm::StringRef autoPad = "NOTSET";
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("auto_pad"))
    autoPad = attr.getValue();
  if (autoPad == "VALID") {
    pads.assign(spatialDims * 2, 0);
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    pads.resize(spatialDims * 2);
    for (int64_t i : llvm::seq<int64_t>(spatialDims)) {
      const int64_t effK = (kernelShape[i] - 1) * dilations[i] + 1;
      int64_t padTotal = (resultType.getDimSize(2 + i) - 1) * strides[i] +
                         effK - inputType.getDimSize(2 + i);
      if (padTotal < 0)
        padTotal = 0;
      const int64_t half = padTotal / 2;
      pads[i] = (autoPad == "SAME_UPPER") ? half : padTotal - half;
      pads[spatialDims + i] = padTotal - pads[i];
    }
  } else if (autoPad != "NOTSET") {
    return rewriter.notifyMatchFailure(op, "ConvInteger unknown auto_pad");
  }
  if (static_cast<int64_t>(pads.size()) != spatialDims * 2)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger pads arity does not match the spatial rank");

  const int64_t inC = inputType.getDimSize(1);
  const int64_t outC = weightType.getDimSize(0);
  const int64_t wC = weightType.getDimSize(1);
  if (inC % group != 0 || wC != inC / group)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger group does not divide the input channels");
  if (resultType.getDimSize(0) != inputType.getDimSize(0) ||
      resultType.getDimSize(1) != outC)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger result batch or channel count does not match");
  if (kernelShape[0] != weightType.getDimSize(2) ||
      kernelShape[1] != weightType.getDimSize(3))
    return rewriter.notifyMatchFailure(
        op, "ConvInteger kernel_shape does not match the weight spatial dims");

  for (int64_t i : llvm::seq<int64_t>(spatialDims)) {
    const int64_t numerator = inputType.getDimSize(2 + i) + pads[i] +
                              pads[spatialDims + i] -
                              dilations[i] * (kernelShape[i] - 1) - 1;
    if (numerator < 0)
      return rewriter.notifyMatchFailure(
          op, "ConvInteger spatial extent is smaller than the kernel");
    const int64_t expected = numerator / strides[i] + 1;
    if (resultType.getDimSize(2 + i) != expected)
      return rewriter.notifyMatchFailure(
          op, "ConvInteger result spatial shape does not match the conv "
              "formula");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  // ONNX FLOAT is datatype 1 and INT32 is 6. The cast kernel is selected
  // from the buffer types, not from this attribute.
  const int64_t reduction = (inC / group) * kernelShape[0] * kernelShape[1];
  const bool exactF32 = eightBitDotIsExactInF32(reduction);
  mlir::Type accType = rewriter.getI32Type();
  if (exactF32)
    accType = rewriter.getF32Type();
  const int64_t onnxAcc = exactF32 ? 1 : 6;

  mlir::Value inputAcc =
      castTo(rewriter, loc, context, op->getOperand(0), accType, onnxAcc);
  mlir::Value weightAcc =
      castTo(rewriter, loc, context, op->getOperand(1), accType, onnxAcc);
  if (xZp) {
    mlir::Value xZpAcc = castTo(rewriter, loc, context, xZp, accType, onnxAcc);
    xZpAcc =
        alignZeroPoint(rewriter, loc, xZpAcc, inputType, /*perChannel=*/false);
    inputAcc = subtractZeroPoint(rewriter, loc, context, inputAcc, xZpAcc);
  }
  if (wZp) {
    mlir::Value wZpAcc = castTo(rewriter, loc, context, wZp, accType, onnxAcc);
    wZpAcc =
        alignZeroPoint(rewriter, loc, wZpAcc, weightType, /*perChannel=*/true);
    weightAcc = subtractZeroPoint(rewriter, loc, context, weightAcc, wZpAcc);
  }

  auto convType = mlir::RankedTensorType::get(resultType.getShape(), accType);
  mlir::Value convInit = createEmptyTensor(rewriter, loc, convType, inputAcc);

  llvm::SmallVector<mlir::Value> operands = {context, inputAcc, weightAcc,
                                             convInit};
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape",
                                        rewriter.getI64ArrayAttr(kernelShape)));
  attrs.push_back(
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides)));
  attrs.push_back(
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads)));
  attrs.push_back(
      rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr(dilations)));
  attrs.push_back(
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));
  auto conv = mlir::hip::ConvOp::create(rewriter, loc, operands, attrs);
  if (!exactF32) {
    rewriter.replaceOp(op, conv.getResult(0));
    return mlir::success();
  }

  mlir::Value outInit = createEmptyTensor(rewriter, loc, resultType, convInit);
  auto casted = mlir::hip::CastOp::create(
      rewriter, loc, context, conv.getResult(0), outInit,
      rewriter.getI64IntegerAttr(/*onnxTo=*/6));
  rewriter.replaceOp(op, casted->getResult(0));
  return mlir::success();
}

} // namespace

void populateConvIntegerConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<ConvIntegerToHip>(ctx);
}

} // namespace hip
} // namespace mlir
