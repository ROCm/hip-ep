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
// 65025 and the reduction length (C/group)*product(kernel_shape) must be at
// most 258. A
// longer reduction casts to i32 and accumulates there. ONNX allows the int32
// accumulator to wrap.
//
// A 1-D weight zero point is reshaped to [M, 1, ...] before the subtract.
// hip.sub right-aligns unequal ranks into the broadcast ABI, so a bare
// tensor<MxT> would land on the trailing spatial axis rather than the
// output-channel axis.
//
//   Before:
//     %y = "onnx.ConvInteger"(%x, %w, %xzp, %wzp)
//         : (tensor<NxCxHxWxui8>, tensor<MxCxKxKxui8>, ...) ->
//         tensor<NxMxH'xW'xi32>
//
//   After (default, and every dynamic ConvInteger):
//     %xf = hip.cast %x to f32
//     %wf = hip.cast %w to f32
//     %xd = hip.sub %xf, %xzp
//     %wd = hip.sub %wf, reshape(%wzp, [M, 1, 1, 1])
//     %acc = hip.conv %xd, %wd
//     %y = hip.cast %acc to i32
//
//   After, only when convert-onnx-to-hip is given conv-integer-quant-conv
//   (the rocMLIR pipeline), every dimension is static, and the op is in
//   the entry block of @main_graph:
//     hip.quant_conv of the int8 operands, plus one hip.quant_conv per
//     zero point that is not the rebase offset. uint8 is rebased to int8
//     by subtracting 128. An omitted uint8 zero point is numeric 0, which
//     rebases to -128, so it is materialized and still corrected.
//     Anywhere else, including an outlined Loop/Scan/If body, and a
//     dynamic batch stay on hip.conv: FuseRocMlirAnchorPass only outlines
//     anchors in @main_graph's entry block, and an unoutlined
//     hip.quant_conv has no lowering.
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

/// ConvInteger needs compile-time channel and spatial extents for validation
/// and for selecting the accumulator path. The runtime convolution ABI,
/// however, obtains N from the input memref descriptor, so batch may remain
/// dynamic.
static bool hasStaticNonBatchShape(mlir::RankedTensorType type) {
  for (int64_t dim : llvm::seq<int64_t>(1, type.getRank()))
    if (type.isDynamicDim(dim))
      return false;
  return true;
}

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

/// Broadcast \p zp onto \p data. A length-M vector becomes [M, 1, ...].
/// A single element becomes a rank-0 scalar.
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

/// True when every element of a constant zero point is the rebase offset, so
/// the correction convolution is the zero tensor. A non-constant zero point
/// is not known to be zero.
static bool rebasedZeroPointIsZero(mlir::Value zp) {
  auto cst = zp.getDefiningOp<mlir::hip::ConstantOp>();
  if (!cst || !cst.getValue())
    return false;
  auto attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(*cst.getValue());
  if (!attr)
    return false;
  auto elem = mlir::cast<mlir::RankedTensorType>(zp.getType()).getElementType();
  const bool ui8 = elem.isUnsignedInteger(8);
  if (!ui8 && !elem.isSignlessInteger(8))
    return false;
  for (const llvm::APInt &value : attr) {
    const int64_t n =
        ui8 ? static_cast<int64_t>(value.getZExtValue()) : value.getSExtValue();
    if (n != (ui8 ? 128 : 0))
      return false;
  }
  return true;
}

/// hip.quant_conv takes signless i8. uint8 is shifted by 128 into that range.
static mlir::Value toSigned8(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value context,
                             mlir::Value value) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (type.getElementType().isSignlessInteger(8))
    return value;
  // hip.quant_conv weights must be signless i8. A signed int8 tensor is the
  // same bits; only the type changes.
  if (type.getElementType().isSignedInteger(8)) {
    auto dstType =
        mlir::RankedTensorType::get(type.getShape(), rewriter.getI8Type());
    return mlir::tensor::BitcastOp::create(rewriter, loc, dstType, value)
        .getResult();
  }
  if (!type.getElementType().isUnsignedInteger(8))
    return {};
  mlir::Value wide = castTo(rewriter, loc, context, value,
                            rewriter.getI32Type(), /*onnxTo=*/6);
  auto offsetType = mlir::RankedTensorType::get({}, rewriter.getI32Type());
  mlir::Value offset = mlir::hip::ConstantOp::create(
      rewriter, loc, offsetType,
      mlir::DenseElementsAttr::get(offsetType,
                                   rewriter.getI32IntegerAttr(128)));
  mlir::Value shifted = subtractZeroPoint(rewriter, loc, context, wide, offset);
  return castTo(rewriter, loc, context, shifted, rewriter.getI8Type(),
                /*onnxTo=*/3);
}

static mlir::Value broadcastTo(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value context,
                               mlir::Value value,
                               mlir::RankedTensorType dstType) {
  if (value.getType() == dstType)
    return value;
  auto shapeType =
      mlir::RankedTensorType::get({dstType.getRank()}, rewriter.getI64Type());
  mlir::Value shape = mlir::arith::ConstantOp::create(
      rewriter, loc, shapeType,
      mlir::DenseElementsAttr::get(shapeType, dstType.getShape()));
  mlir::Value init = createEmptyTensor(rewriter, loc, dstType, value);
  return mlir::hip::ExpandOp::create(rewriter, loc, context, value, shape, init)
      ->getResult(0);
}

/// An omitted uint8 zero point is 0. After the operand is rebased by -128
/// that 0 is -128, so the correction convolution is required. A signed
/// operand's omitted zero point is already 0 in the type hip.quant_conv uses.
static mlir::Value
materializeOmittedUnsignedZeroPoint(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value data,
                                    mlir::Value zp) {
  if (zp)
    return zp;
  auto elem =
      mlir::cast<mlir::RankedTensorType>(data.getType()).getElementType();
  if (!elem.isUnsignedInteger(8))
    return {};
  auto zpType = mlir::RankedTensorType::get({}, elem);
  return mlir::hip::ConstantOp::create(
      rewriter, loc, zpType,
      mlir::DenseElementsAttr::get(zpType, rewriter.getIntegerAttr(elem, 0)));
}

/// FuseRocMlirAnchorPass outlines anchors only from @main_graph's entry
/// block. A hip.quant_conv left in any other function, including an outlined
/// Loop, Scan, or If body, has no lowering.
static bool inMainGraphEntry(mlir::Operation *op) {
  auto func = op->getParentOfType<mlir::func::FuncOp>();
  return func && func.getSymName() == "main_graph" && !func.getBody().empty() &&
         op->getBlock() == &func.getBody().front();
}

/// Static ConvInteger as int8 hip.quant_conv. Returns failure when the conv
/// cannot be formed, and the caller keeps the cast/sub/hip.conv lowering.
static mlir::LogicalResult tryLowerQuantConv(
    mlir::PatternRewriter &rewriter, mlir::Operation *op, mlir::Value context,
    mlir::Location loc, mlir::RankedTensorType inputType,
    mlir::RankedTensorType weightType, mlir::RankedTensorType resultType,
    bool is1D, llvm::ArrayRef<int64_t> kernelShapeIn,
    llvm::ArrayRef<int64_t> stridesIn, llvm::ArrayRef<int64_t> dilationsIn,
    llvm::ArrayRef<int64_t> padsIn, int64_t group) {
  if (!inMainGraphEntry(op) || inputType.isDynamicDim(0))
    return mlir::failure();
  for (int64_t pad : padsIn) {
    if (pad < 0)
      return mlir::failure();
  }

  llvm::SmallVector<int64_t> kernelShape(kernelShapeIn);
  llvm::SmallVector<int64_t> strides(stridesIn);
  llvm::SmallVector<int64_t> dilations(dilationsIn);
  llvm::SmallVector<int64_t> pads(padsIn);
  mlir::Value input = op->getOperand(0);
  mlir::Value weight = op->getOperand(1);
  mlir::Value xZp = realOperand(op, 2);
  mlir::Value wZp = realOperand(op, 3);

  if (is1D) {
    kernelShape.insert(kernelShape.begin(), 1);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
    pads = {0, pads[0], 0, pads[1]};

    auto expand = [&](mlir::Value value) -> mlir::Value {
      auto srcType = mlir::cast<mlir::RankedTensorType>(value.getType());
      llvm::SmallVector<int64_t> shape(srcType.getShape());
      shape.insert(shape.end() - 1, 1);
      auto rank4 = mlir::RankedTensorType::get(shape, srcType.getElementType());
      llvm::SmallVector<mlir::ReassociationIndices> reassoc = {
          {0}, {1}, {2, 3}};
      return mlir::tensor::ExpandShapeOp::create(rewriter, loc, rank4, value,
                                                 reassoc)
          .getResult();
    };
    input = expand(input);
    weight = expand(weight);
    // A full-rank weight zero point, such as [M, 1, 1] over [M, C, K], has
    // to gain the same inserted 1. Left at rank 3, alignZeroPoint sees a
    // rank mismatch with the rank-4 weight and reshapes it to a scalar.
    if (wZp) {
      auto zpType = mlir::dyn_cast<mlir::RankedTensorType>(wZp.getType());
      if (zpType && zpType.getRank() == 3)
        wZp = expand(wZp);
    }
    inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    weightType = mlir::cast<mlir::RankedTensorType>(weight.getType());
    llvm::SmallVector<int64_t> outShape(resultType.getShape());
    outShape.insert(outShape.end() - 1, 1);
    resultType =
        mlir::RankedTensorType::get(outShape, resultType.getElementType());
  }

  xZp = materializeOmittedUnsignedZeroPoint(rewriter, loc, input, xZp);
  wZp = materializeOmittedUnsignedZeroPoint(rewriter, loc, weight, wZp);
  input = toSigned8(rewriter, loc, context, input);
  weight = toSigned8(rewriter, loc, context, weight);
  if (!input || !weight)
    return mlir::failure();
  inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  weightType = mlir::cast<mlir::RankedTensorType>(weight.getType());
  if (xZp && rebasedZeroPointIsZero(xZp))
    xZp = {};
  if (wZp && rebasedZeroPointIsZero(wZp))
    wZp = {};
  if (xZp) {
    xZp = toSigned8(rewriter, loc, context, xZp);
    if (!xZp)
      return mlir::failure();
    xZp = alignZeroPoint(rewriter, loc, xZp, inputType, /*perChannel=*/false);
    xZp = broadcastTo(rewriter, loc, context, xZp, inputType);
  }
  if (wZp) {
    wZp = toSigned8(rewriter, loc, context, wZp);
    if (!wZp)
      return mlir::failure();
    wZp = alignZeroPoint(rewriter, loc, wZp, weightType, /*perChannel=*/true);
    wZp = broadcastTo(rewriter, loc, context, wZp, weightType);
  }

  auto quant = [&](mlir::Value lhs, mlir::Value rhs) {
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, lhs);
    return mlir::hip::QuantConvOp::create(
               rewriter, loc, mlir::TypeRange{resultType}, context, lhs, rhs,
               init, rewriter.getI64ArrayAttr(kernelShape),
               rewriter.getI64ArrayAttr(strides),
               rewriter.getI64ArrayAttr(pads),
               rewriter.getI64ArrayAttr(dilations),
               rewriter.getI64IntegerAttr(group))
        .getResult(0);
  };
  auto combine = [&](bool add, mlir::Value lhs, mlir::Value rhs) {
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, lhs);
    if (add)
      return mlir::hip::AddOp::create(rewriter, loc, context, lhs, rhs, init)
          ->getResult(0);
    return mlir::hip::SubOp::create(rewriter, loc, context, lhs, rhs, init)
        ->getResult(0);
  };

  mlir::Value acc = quant(input, weight);
  if (xZp)
    acc = combine(/*add=*/false, acc, quant(xZp, weight));
  if (wZp)
    acc = combine(/*add=*/false, acc, quant(input, wZp));
  if (xZp && wZp)
    acc = combine(/*add=*/true, acc, quant(xZp, wZp));

  if (is1D) {
    llvm::SmallVector<int64_t> collapsed(
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
            .getShape());
    auto collapsedType =
        mlir::RankedTensorType::get(collapsed, resultType.getElementType());
    llvm::SmallVector<mlir::ReassociationIndices> reassoc = {{0}, {1}, {2, 3}};
    acc = mlir::tensor::CollapseShapeOp::create(rewriter, loc, collapsedType,
                                                acc, reassoc)
              .getResult();
  }
  rewriter.replaceOp(op, acc);
  return mlir::success();
}

/// onnx.ConvInteger -> hip.cast / hip.sub / hip.conv / hip.cast.
struct ConvIntegerToHip : public mlir::RewritePattern {
  ConvIntegerToHip(mlir::MLIRContext *ctx, bool quantConv)
      : RewritePattern("onnx.ConvInteger", /*benefit=*/1, ctx),
        quantConv(quantConv) {}

  bool quantConv = false;

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
  if ((inputType.getRank() != 3 && inputType.getRank() != 4) ||
      weightType.getRank() != inputType.getRank() ||
      resultType.getRank() != inputType.getRank())
    return rewriter.notifyMatchFailure(
        op, "ConvInteger lowering supports rank-3 NCL and rank-4 NCHW only");
  if (!isEightBit(inputType.getElementType()) ||
      !isEightBit(weightType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "ConvInteger x and w must be int8 or uint8");
  if (!resultType.getElementType().isInteger(32) ||
      resultType.getElementType().isUnsignedInteger(32))
    return rewriter.notifyMatchFailure(op, "ConvInteger result must be i32");
  if (!hasStaticNonBatchShape(inputType) || !weightType.hasStaticShape() ||
      !hasStaticNonBatchShape(resultType))
    return rewriter.notifyMatchFailure(
        op, "ConvInteger lowering requires static weights, channels, and "
            "spatial dimensions");
  if (inputType.isDynamicDim(0) != resultType.isDynamicDim(0))
    return rewriter.notifyMatchFailure(
        op, "ConvInteger input and result batch dimensions must both be "
            "static or both be dynamic");

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

  const bool is1D = inputType.getRank() == 3;
  const int64_t spatialDims = inputType.getRank() - 2;
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    if (mlir::failed(readI64Array(attr, kernelShape)))
      return rewriter.notifyMatchFailure(op, "ConvInteger kernel_shape");
  } else {
    for (int64_t dim : llvm::seq<int64_t>(spatialDims))
      kernelShape.push_back(weightType.getDimSize(dim + 2));
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
  if ((!inputType.isDynamicDim(0) &&
       resultType.getDimSize(0) != inputType.getDimSize(0)) ||
      resultType.getDimSize(1) != outC)
    return rewriter.notifyMatchFailure(
        op, "ConvInteger result batch or channel count does not match");
  for (int64_t dim : llvm::seq<int64_t>(spatialDims)) {
    if (kernelShape[dim] != weightType.getDimSize(dim + 2))
      return rewriter.notifyMatchFailure(
          op,
          "ConvInteger kernel_shape does not match the weight spatial dims");
  }

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

  if (quantConv &&
      mlir::succeeded(tryLowerQuantConv(
          rewriter, op, context, loc, inputType, weightType, resultType, is1D,
          kernelShape, strides, dilations, pads, group)))
    return mlir::success();

  // ONNX FLOAT is datatype 1 and INT32 is 6. The cast kernel is selected
  // from the buffer types, not from this attribute.
  int64_t reduction = inC / group;
  for (int64_t kernel : kernelShape)
    reduction *= kernel;
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

  llvm::SmallVector<mlir::ReassociationIndices> reassoc1d = {{0}, {1}, {2, 3}};
  auto expandTo2D = [&](mlir::Value value,
                        mlir::RankedTensorType type) -> mlir::Value {
    llvm::SmallVector<int64_t> shape(type.getShape());
    shape.insert(shape.end() - 1, 1);
    auto expandedType =
        mlir::RankedTensorType::get(shape, type.getElementType());
    llvm::SmallVector<mlir::OpFoldResult> outputShape;
    for (int64_t expandedDim : llvm::seq<int64_t>(4)) {
      if (expandedDim == 2) {
        outputShape.push_back(rewriter.getIndexAttr(1));
        continue;
      }
      const int64_t sourceDim = expandedDim < 2 ? expandedDim : 2;
      if (type.isDynamicDim(sourceDim))
        outputShape.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, value, sourceDim)
                .getResult());
      else
        outputShape.push_back(
            rewriter.getIndexAttr(type.getDimSize(sourceDim)));
    }
    return mlir::tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                               value, reassoc1d, outputShape);
  };

  if (is1D) {
    inputAcc = expandTo2D(
        inputAcc, mlir::cast<mlir::RankedTensorType>(inputAcc.getType()));
    weightAcc = expandTo2D(
        weightAcc, mlir::cast<mlir::RankedTensorType>(weightAcc.getType()));
    convInit = expandTo2D(convInit, convType);
    kernelShape.insert(kernelShape.begin(), 1);
    strides.insert(strides.begin(), 1);
    pads = {0, pads[0], 0, pads[1]};
    dilations.insert(dilations.begin(), 1);
  }

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
  mlir::Value convResult = conv.getResult(0);
  if (is1D)
    convResult = mlir::tensor::CollapseShapeOp::create(rewriter, loc, convType,
                                                       convResult, reassoc1d);
  if (!exactF32) {
    rewriter.replaceOp(op, convResult);
    return mlir::success();
  }

  mlir::Value outInit = createEmptyTensor(rewriter, loc, resultType, convInit);
  auto casted =
      mlir::hip::CastOp::create(rewriter, loc, context, convResult, outInit,
                                rewriter.getI64IntegerAttr(/*onnxTo=*/6));
  rewriter.replaceOp(op, casted->getResult(0));
  return mlir::success();
}

} // namespace

void populateConvIntegerConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx, bool quantConv) {
  patterns.add<ConvIntegerToHip>(ctx, quantConv);
}

} // namespace hip
} // namespace mlir
