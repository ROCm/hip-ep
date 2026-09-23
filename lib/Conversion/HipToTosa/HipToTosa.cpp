/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Tosa/Utils/ConversionUtils.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <type_traits>

namespace mlir::hip {

#define GEN_PASS_DEF_CONVERTHIPTOTOSAPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// TOSA broadcasts size-1 dimensions only and requires every operand to carry
// the result's rank, so hip's ONNX/NumPy rank-extending broadcast does not
// always survive a 1-1 mapping.
bool isTosaBroadcastableShape(RankedTensorType operandType,
                              RankedTensorType resultType) {
  if (!operandType.hasStaticShape())
    return false;
  if (operandType.getRank() != resultType.getRank())
    return false;

  ArrayRef<int64_t> shape = operandType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (int64_t i = 0, e = resultType.getRank(); i < e; ++i)
    if (shape[i] != resultShape[i] && shape[i] != 1)
      return false;
  return true;
}

bool isTosaCompatibleOperand(Value operand, RankedTensorType resultType) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType)
    return false;
  if (operandType.getElementType() != resultType.getElementType())
    return false;
  return isTosaBroadcastableShape(operandType, resultType);
}

// TOSA carries reshape/slice shapes as !tosa.shape SSA operands.
static Value createConstShape(ConversionPatternRewriter &rewriter, Location loc,
                              ArrayRef<int64_t> extents) {
  return tosa::ConstShapeOp::create(
      rewriter, loc,
      tosa::shapeType::get(rewriter.getContext(), extents.size()),
      rewriter.getIndexTensorAttr(extents));
}

// Reshape `input` to `shape` via tosa.reshape + tosa.const_shape.
static Value reshapeTo(Value input, ArrayRef<int64_t> shape,
                       ConversionPatternRewriter &rewriter) {
  auto type = cast<RankedTensorType>(input.getType());
  auto shapeConst = createConstShape(rewriter, rewriter.getUnknownLoc(), shape);
  return tosa::ReshapeOp::create(rewriter, rewriter.getUnknownLoc(),
                                 type.clone(shape), input, shapeConst);
}

static Value transposeTo(Value input, ArrayRef<int64_t> shape,
                         ArrayRef<int32_t> permutation,
                         ConversionPatternRewriter &rewriter, Location loc) {
  auto type = cast<RankedTensorType>(input.getType());
  return tosa::TransposeOp::create(rewriter, loc, type.clone(shape), input,
                                   rewriter.getDenseI32ArrayAttr(permutation));
}

// Crop `input` to `shape`, anchored at the origin, via tosa.slice.
static Value sliceTo(Value input, ArrayRef<int64_t> shape,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto type = cast<RankedTensorType>(input.getType());
  auto shapeType = tosa::shapeType::get(rewriter.getContext(), shape.size());
  auto start = tosa::ConstShapeOp::create(
      rewriter, loc, shapeType,
      rewriter.getIndexTensorAttr(SmallVector<int64_t>(shape.size(), 0)));
  auto size = tosa::ConstShapeOp::create(rewriter, loc, shapeType,
                                         rewriter.getIndexTensorAttr(shape));
  return tosa::SliceOp::create(rewriter, loc, type.clone(shape), input, start,
                               size);
}

// Same as sliceTo, but the crop is allowed to start off the origin.
static Value sliceAt(Value input, ArrayRef<int64_t> start,
                     ArrayRef<int64_t> size,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto type = cast<RankedTensorType>(input.getType());
  auto shapeType = tosa::shapeType::get(rewriter.getContext(), size.size());
  auto startConst = tosa::ConstShapeOp::create(
      rewriter, loc, shapeType, rewriter.getIndexTensorAttr(start));
  auto sizeConst = tosa::ConstShapeOp::create(
      rewriter, loc, shapeType, rewriter.getIndexTensorAttr(size));
  return tosa::SliceOp::create(rewriter, loc, type.clone(size), input,
                               startConst, sizeConst);
}

static SmallVector<int64_t> getI64Values(ArrayAttr attrs) {
  SmallVector<int64_t> values;
  values.reserve(attrs.size());
  for (Attribute attr : attrs)
    values.push_back(cast<IntegerAttr>(attr).getInt());
  return values;
}

// hip.conv uses ONNX's NCHW input/output and OIHW weight layouts, while
// tosa.conv2d is defined on NHWC and OHWI. Keep the TOSA op semantically valid
// by transposing to its canonical layouts and transpose the result back:
//
//   input  [N,C,H,W] -> [N,H,W,C]
//   weight [K,C,Y,X] -> [K,Y,X,C]
//   output [N,H,W,K] -> [N,K,H,W]
//
// rocMLIR folds these transposes into the Rock convolution's layout metadata,
// so they describe the original storage rather than becoming data movement.
struct ConvConverter final : public OpConversionPattern<hip::ConvOp> {
  using OpConversionPattern<hip::ConvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto weightType =
        dyn_cast<RankedTensorType>(adaptor.getWeights().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inputType || !weightType || !resultType ||
        !inputType.hasStaticShape() || !weightType.hasStaticShape() ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected static ranked input, weight, and result tensors");
    if (inputType.getRank() != 4 || weightType.getRank() != 4 ||
        resultType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "expected 2D convolution");

    Type elementType = resultType.getElementType();
    if (inputType.getElementType() != elementType ||
        weightType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "input, weight, and result element types must match");
    Type accType;
    if (elementType.isF16() || elementType.isBF16() || elementType.isF32())
      accType = rewriter.getF32Type();
    else
      return rewriter.notifyMatchFailure(
          op, "only f16, bf16, and f32 convolution are supported");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> weightShape = weightType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    // tosa.conv2d has no grouped form (weight IC must equal input C). Depthwise
    // is a separate op and is not handled here.
    if (op.getGroup() != 1)
      return rewriter.notifyMatchFailure(
          op, "grouped convolution has no TOSA conv2d spelling");
    if (inputShape[0] != resultShape[0] || weightShape[1] != inputShape[1] ||
        weightShape[0] != resultShape[1])
      return rewriter.notifyMatchFailure(op, "incompatible batch or channels");

    SmallVector<int64_t> kernelShape = getI64Values(op.getKernelShape());
    if (kernelShape.size() != 2 || kernelShape[0] != weightShape[2] ||
        kernelShape[1] != weightShape[3])
      return rewriter.notifyMatchFailure(
          op, "kernel_shape disagrees with the weight tensor");

    // TOSA CONV2D bias is T<out_t>, not acc_t. The MLIR verifier requires a
    // float bias to match the result element type, even when acc_type is f32
    // for an f16/bf16 convolution. Length 1 is a legal TOSA broadcast (BC==1).
    Value bias = adaptor.getBias();
    if (bias) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 1 ||
          biasType.getElementType() != elementType ||
          (biasType.getDimSize(0) != resultShape[1] &&
           biasType.getDimSize(0) != 1))
        return rewriter.notifyMatchFailure(op, "incompatible bias tensor");
    } else {
      auto biasType = RankedTensorType::get({resultShape[1]}, elementType);
      bias = tosa::ConstOp::create(
          rewriter, op.getLoc(), biasType,
          DenseElementsAttr::get(biasType, rewriter.getZeroAttr(elementType)));
    }

    SmallVector<int64_t> strides = getI64Values(op.getStrides());
    SmallVector<int64_t> dilations = getI64Values(op.getDilations());
    SmallVector<int64_t> pads = getI64Values(op.getPads());
    if (strides.size() != 2 || dilations.size() != 2 || pads.size() != 4)
      return rewriter.notifyMatchFailure(
          op, "expected 2D stride, dilation, and padding attributes");

    // ONNX floors the output size, so a strided window that overruns the
    // padded input just drops the trailing partial window. TOSA instead
    // requires the window arithmetic to divide exactly:
    //
    //   O == (I - 1 + pad_before + pad_after - (K - 1) * dilation) / stride + 1
    //
    // Absorb that remainder by shrinking the trailing pad, and crop the input
    // for whatever the pad cannot cover -- a stride-2 1x1 kernel has no
    // padding to give back. Either way only elements the ONNX convolution
    // never reads are removed, so the result is unchanged.
    SmallVector<int64_t> padBefore(2), padAfter(2), crop(2, 0);
    for (int64_t dim : llvm::seq<int64_t>(2)) {
      if (strides[dim] < 1 || dilations[dim] < 1)
        return rewriter.notifyMatchFailure(op, "expected positive stride and "
                                               "dilation");
      padBefore[dim] = pads[dim];
      padAfter[dim] = pads[dim + 2];
      if (padBefore[dim] < 0 || padAfter[dim] < 0)
        return rewriter.notifyMatchFailure(op, "expected non-negative padding");

      int64_t inputSize = inputShape[dim + 2];
      int64_t span = inputSize - 1 + padBefore[dim] + padAfter[dim] -
                     (weightShape[dim + 2] - 1) * dilations[dim];
      if (span < 0)
        return rewriter.notifyMatchFailure(op, "kernel larger than the padded "
                                               "input");

      int64_t remainder = span % strides[dim];
      int64_t fromPad = std::min(padAfter[dim], remainder);
      padAfter[dim] -= fromPad;
      crop[dim] = remainder - fromPad;
      if (crop[dim] >= inputSize)
        return rewriter.notifyMatchFailure(op, "convolution reads no input");
      if ((span - remainder) / strides[dim] + 1 != resultShape[dim + 2])
        return rewriter.notifyMatchFailure(
            op, "result shape disagrees with the convolution window");
    }

    Value input = transposeTo(
        adaptor.getInput(),
        {inputShape[0], inputShape[2], inputShape[3], inputShape[1]},
        {0, 2, 3, 1}, rewriter, op.getLoc());
    if (crop[0] != 0 || crop[1] != 0)
      input = sliceTo(input,
                      {inputShape[0], inputShape[2] - crop[0],
                       inputShape[3] - crop[1], inputShape[1]},
                      rewriter, op.getLoc());
    Value weight = transposeTo(
        adaptor.getWeights(),
        {weightShape[0], weightShape[2], weightShape[3], weightShape[1]},
        {0, 2, 3, 1}, rewriter, op.getLoc());
    auto nhwkType = resultType.clone(
        {resultShape[0], resultShape[2], resultShape[3], resultShape[1]});

    // TOSA orders padding [top, bottom, left, right].
    auto tosaPads = rewriter.getDenseI64ArrayAttr(
        {padBefore[0], padAfter[0], padBefore[1], padAfter[1]});
    auto conv = tosa::Conv2DOp::create(
        rewriter, op.getLoc(), nhwkType, input, weight, bias, tosaPads,
        rewriter.getDenseI64ArrayAttr(strides),
        rewriter.getDenseI64ArrayAttr(dilations), TypeAttr::get(accType));

    rewriter.replaceOp(op, transposeTo(conv.getResult(), resultShape,
                                       {0, 3, 1, 2}, rewriter, op.getLoc()));
    return success();
  }
};

// The hip context and the DPS `outs` buffer are both dropped: the result type
// already encodes the destination.
struct MatMulConverter final : public OpConversionPattern<hip::MatmulOp> {
  using OpConversionPattern<hip::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    // tosa.matmul is a plain A @ B; transposes must have been folded away.
    if (op.getTransA() != 0 || op.getTransB() != 0)
      return rewriter.notifyMatchFailure(op, "transA/transB unsupported");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");

    auto aType = dyn_cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(adaptor.getB().getType());
    if (!aType || !aType.hasStaticShape() || !bType || !bType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "operands not static ranked");
    if (aType.getRank() < 2 || bType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "only [..,M,K] x [K,N] (rank-2 B) is supported");

    // tosa.matmul requires rank-3 operands with *equal* batch sizes -- it does
    // not broadcast a size-1 batch against a larger one. hip.matmul here has an
    // unbatched (rank-2) B, so instead of broadcasting B's batch up to A's,
    // collapse all of A's leading dims and M into a single dimension:
    //
    //   A[.., M, K] -> [1, prod(..)*M, K]
    //   B[K, N]     -> [1, K, N]
    //   matmul      -> [1, prod(..)*M, N]
    //   result      -> [.., M, N]   (original result shape)
    ArrayRef<int64_t> aShape = aType.getShape();
    int64_t k = aShape.back();
    int64_t collapsedM = 1;
    for (int64_t d : aShape.drop_back())
      collapsedM *= d;
    int64_t n = bType.getShape().back();

    Value a = reshapeTo(adaptor.getA(), {1, collapsedM, k}, rewriter);
    Value b = reshapeTo(adaptor.getB(), {1, k, n}, rewriter);

    auto matmulType = resultType.clone({1, collapsedM, n});
    // The quant-info builder appends the (zero) zero-point operands that
    // tosa.matmul requires for float inputs.
    Value matmul =
        tosa::MatMulOp::create(rewriter, op.getLoc(), matmulType, a, b)
            .getResult();

    rewriter.replaceOp(op, reshapeTo(matmul, resultType.getShape(), rewriter));
    return success();
  }
};

// The hip context and the DPS `outs` buffer are both dropped: the result type
// already encodes the destination.
//
// tosa.mul's shift operand right-shifts the product of i32 inputs. hip.mul is
// a plain multiply with no rescale, so the shift is zero; for float operands
// the op's verifier requires zero as well.
Value createZeroMulShift(ConversionPatternRewriter &rewriter, Location loc) {
  auto shiftType = RankedTensorType::get({1}, rewriter.getI8Type());
  return tosa::ConstOp::create(
      rewriter, loc, shiftType,
      DenseElementsAttr::get(shiftType,
                             rewriter.getIntegerAttr(rewriter.getI8Type(), 0)));
}

// Splat a float scalar across `type`, matching rock::tosa::getZeroTensor's
// tosa.const shape (the result tensor, not a rank-0 scalar).
Value createSplatFloat(ConversionPatternRewriter &rewriter, Location loc,
                       RankedTensorType type, double value) {
  auto elemType = cast<FloatType>(type.getElementType());
  APFloat ap(value);
  bool losesInfo = false;
  ap.convert(elemType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
             &losesInfo);
  return tosa::ConstOp::create(
      rewriter, loc, type,
      DenseElementsAttr::get(type, rewriter.getFloatAttr(elemType, ap)));
}

// ONNX CausalConvWithState / hip.causal_conv_with_state is Concat(past, x) +
// depthwise Conv + Slice(present). TOSA has no causal or 1-D depthwise op, so
// the same expansion is spelled with tosa.concat (or a zero past),
// tosa.depthwise_conv2d (length as W, H=1), and tosa.slice.
//
//   Before (channels-first, k=4):
//     %y, %s = hip.causal_conv_with_state(%x, %w, %b, %past)
//         : tensor<1x64x128xf16>, tensor<1x64x3xf16>
//   After:
//     %p = tosa.concat %past, %x {axis = 2}
//         : tensor<1x64x131xf16>
//     %s = tosa.slice %p  // last k-1 = 3 along the length
//     %y = NCHW<->NHWC around tosa.depthwise_conv2d, then optional SiLU
//
// present_state is the last (k-1) values of the concatenated sequence, which
// is the ONNX contract including the short-input / zero-pad case. k=1 makes
// that tensor empty, which TOSA forbids, so it is rejected.
struct CausalConvWithStateConverter final
    : public OpConversionPattern<CausalConvWithStateOp> {
  using OpConversionPattern<CausalConvWithStateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CausalConvWithStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 2)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    if (op.getNdim() != 1)
      return rewriter.notifyMatchFailure(op, "only 1-D causal conv is TOSA");

    Location loc = op.getLoc();
    auto inType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto wType = dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
    auto yType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto sType = dyn_cast<RankedTensorType>(op.getResult(1).getType());
    if (!inType || !wType || !yType || !sType || !inType.hasStaticShape() ||
        !wType.hasStaticShape() || !yType.hasStaticShape() ||
        !sType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (inType.getRank() != 3 || wType.getRank() != 3 || yType.getRank() != 3 ||
        sType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected rank-3 1-D tensors");
    if (wType.getDimSize(1) != 1)
      return rewriter.notifyMatchFailure(op,
                                         "weight must be depthwise [C,1,k]");

    Type elemType = yType.getElementType();
    Type accType;
    if (elemType.isF16() || elemType.isBF16() || elemType.isF32())
      accType = rewriter.getF32Type();
    else
      return rewriter.notifyMatchFailure(op, "only f16, bf16, and f32");
    if (inType.getElementType() != elemType ||
        wType.getElementType() != elemType)
      return rewriter.notifyMatchFailure(op, "element types must match");

    int64_t channels = wType.getDimSize(0);
    int64_t k = wType.getDimSize(2);
    int64_t stateLen = k - 1;
    if (stateLen <= 0)
      return rewriter.notifyMatchFailure(
          op, "k=1 present_state is an empty TOSA tensor");

    bool channelsLast = op.getChannelsLast();
    int64_t batch = inType.getDimSize(0);
    int64_t length = channelsLast ? inType.getDimSize(1) : inType.getDimSize(2);
    int64_t inChannels =
        channelsLast ? inType.getDimSize(2) : inType.getDimSize(1);
    if (inChannels != channels || yType.getShape() != inType.getShape() ||
        sType.getShape() != ArrayRef<int64_t>({batch, channels, stateLen}))
      return rewriter.notifyMatchFailure(op, "incompatible conv shapes");

    StringRef activation = op.getActivation();
    if (activation != "none" && activation != "silu" && activation != "swish")
      return rewriter.notifyMatchFailure(op, "unsupported fused activation");

    Value nchw = adaptor.getInput();
    if (channelsLast)
      nchw = transposeTo(nchw, {batch, channels, length}, {0, 2, 1}, rewriter,
                         loc);

    Value past = adaptor.getPastState();
    if (past) {
      auto pastType = dyn_cast<RankedTensorType>(past.getType());
      if (!pastType || pastType.getShape() != sType.getShape() ||
          pastType.getElementType() != elemType)
        return rewriter.notifyMatchFailure(op, "incompatible past_state");
    } else {
      past = tosa::ConstOp::create(
          rewriter, loc, sType,
          DenseElementsAttr::get(sType, rewriter.getZeroAttr(elemType)));
    }

    int64_t paddedLen = stateLen + length;
    auto paddedType =
        RankedTensorType::get({batch, channels, paddedLen}, elemType);
    Value padded = tosa::ConcatOp::create(rewriter, loc, paddedType,
                                          ValueRange{past, nchw},
                                          rewriter.getI32IntegerAttr(2));
    Value present = sliceAt(padded, {0, 0, length}, {batch, channels, stateLen},
                            rewriter, loc);

    Value bias = adaptor.getBias();
    if (bias) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 1 ||
          biasType.getElementType() != elemType ||
          (biasType.getDimSize(0) != channels && biasType.getDimSize(0) != 1))
        return rewriter.notifyMatchFailure(op, "incompatible bias tensor");
    } else {
      auto biasType = RankedTensorType::get({channels}, elemType);
      bias = tosa::ConstOp::create(
          rewriter, loc, biasType,
          DenseElementsAttr::get(biasType, rewriter.getZeroAttr(elemType)));
    }

    Value inputNhwc =
        reshapeTo(padded, {batch, channels, 1, paddedLen}, rewriter);
    inputNhwc = transposeTo(inputNhwc, {batch, 1, paddedLen, channels},
                            {0, 2, 3, 1}, rewriter, loc);
    Value weightTosa = transposeTo(adaptor.getWeight(), {1, k, channels},
                                   {1, 2, 0}, rewriter, loc);
    weightTosa = reshapeTo(weightTosa, {1, k, channels, 1}, rewriter);

    auto nhwcOutType =
        RankedTensorType::get({batch, 1, length, channels}, elemType);
    Value conv =
        tosa::DepthwiseConv2DOp::create(
            rewriter, loc, nhwcOutType, inputNhwc, weightTosa, bias,
            rewriter.getDenseI64ArrayAttr({0, 0, 0, 0}),
            rewriter.getDenseI64ArrayAttr({1, 1}),
            rewriter.getDenseI64ArrayAttr({1, 1}), TypeAttr::get(accType))
            .getResult();
    Value nchwOut = transposeTo(conv, {batch, channels, 1, length},
                                {0, 3, 1, 2}, rewriter, loc);
    nchwOut = reshapeTo(nchwOut, {batch, channels, length}, rewriter);

    if (activation == "silu" || activation == "swish") {
      Value sigmoid = tosa::SigmoidOp::create(
          rewriter, loc,
          RankedTensorType::get({batch, channels, length}, elemType), nchwOut);
      nchwOut = tosa::MulOp::create(rewriter, loc, nchwOut.getType(), nchwOut,
                                    sigmoid, createZeroMulShift(rewriter, loc));
    }

    Value y = nchwOut;
    if (channelsLast)
      y = transposeTo(y, yType.getShape(), {0, 2, 1}, rewriter, loc);
    rewriter.replaceOp(op, {y, present});
    return success();
  }
};

// Multiply by a scalar that ONNX carries as an f32 attribute. hipBLASLt scales
// in f32 even when the data is f16 or bf16 (scaleType = HIP_R_32F while
// dataType is HIP_R_16F), so for those types widen to f32 around the multiply
// rather than rounding the scalar down to the data type and losing it there.
static Value scaleByF32(Value value, float scale, RankedTensorType type,
                        ConversionPatternRewriter &rewriter, Location loc) {
  bool widen = !type.getElementType().isF32();
  RankedTensorType mulType = widen ? type.clone(rewriter.getF32Type()) : type;
  if (widen)
    value = tosa::CastOp::create(rewriter, loc, mulType, value);
  value = tosa::MulOp::create(rewriter, loc, mulType, value,
                              createSplatFloat(rewriter, loc, mulType, scale),
                              createZeroMulShift(rewriter, loc));
  if (widen)
    value = tosa::CastOp::create(rewriter, loc, type, value);
  return value;
}

// hip.gemm is ONNX Gemm: Y = alpha * A' * B' + beta * C, where A and B are
// optionally transposed and C broadcasts to [M, N]. TOSA has no fused
// equivalent, so this spells the whole thing out. MIGraphX's ONNX parser
// desugars Gemm the same way -- into a dot plus a broadcast add -- and rocMLIR
// fuses the resulting chain back into one kernel, so the expansion costs
// nothing downstream.
struct GemmConverter final : public OpConversionPattern<hip::GemmOp> {
  using OpConversionPattern<hip::GemmOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto aType = dyn_cast<RankedTensorType>(adaptor.getInputA().getType());
    auto bType = dyn_cast<RankedTensorType>(adaptor.getInputB().getType());
    if (!resultType || !resultType.hasStaticShape() || !aType ||
        !aType.hasStaticShape() || !bType || !bType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (aType.getRank() != 2 || bType.getRank() != 2 ||
        resultType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "ONNX Gemm is 2-D");

    Type elementType = resultType.getElementType();
    if (aType.getElementType() != elementType ||
        bType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "A, B, and result element types must match");
    // alpha and beta are f32 attributes folded in as elementwise multiplies in
    // the result type, so an integer gemm would silently round them away. f64
    // is excluded separately: the hipBLASLt path accepts it, but TOSA has no
    // f64 tensor type, and failing legalization beats emitting invalid TOSA.
    if (!elementType.isF32() && !elementType.isF16() && !elementType.isBF16())
      return rewriter.notifyMatchFailure(
          op, "only f32, f16, and bf16 gemm are supported");

    Location loc = op.getLoc();
    bool transA = op.getTransA() != 0;
    bool transB = op.getTransB() != 0;

    // Once the optional transposes are applied the operands are [M,K] x [K,N].
    int64_t m = aType.getDimSize(transA ? 1 : 0);
    int64_t ka = aType.getDimSize(transA ? 0 : 1);
    int64_t kb = bType.getDimSize(transB ? 1 : 0);
    int64_t n = bType.getDimSize(transB ? 0 : 1);
    if (ka != kb)
      return rewriter.notifyMatchFailure(op, "A and B disagree on K");
    if (resultType.getDimSize(0) != m || resultType.getDimSize(1) != n)
      return rewriter.notifyMatchFailure(op,
                                         "result shape disagrees with A and B");

    Value a = adaptor.getInputA();
    if (transA)
      a = transposeTo(a, {m, ka}, {1, 0}, rewriter, loc);
    Value b = adaptor.getInputB();
    if (transB)
      b = transposeTo(b, {kb, n}, {1, 0}, rewriter, loc);

    // tosa.matmul is batched, so carry a batch of one through and drop it.
    a = reshapeTo(a, {1, m, ka}, rewriter);
    b = reshapeTo(b, {1, kb, n}, rewriter);
    Value matmul =
        tosa::MatMulOp::create(rewriter, loc, resultType.clone({1, m, n}), a, b)
            .getResult();
    Value result = reshapeTo(matmul, {m, n}, rewriter);

    if (op.getAlpha().convertToFloat() != 1.0f)
      result = scaleByF32(result, op.getAlpha().convertToFloat(), resultType,
                          rewriter, loc);

    if (Value c = adaptor.getInputC()) {
      auto cType = dyn_cast<RankedTensorType>(c.getType());
      if (!cType || !cType.hasStaticShape() ||
          cType.getElementType() != elementType)
        return rewriter.notifyMatchFailure(op, "unsupported C tensor");
      ArrayRef<int64_t> cShape = cType.getShape();
      if (cType.getRank() > 2)
        return rewriter.notifyMatchFailure(op, "C rank exceeds the result");
      // ONNX broadcasts C to [M, N] unidirectionally. TOSA expands only size-1
      // dimensions and needs matching rank, so left-pad C's shape with ones.
      SmallVector<int64_t> padded(2, 1);
      for (auto [idx, dim] : llvm::enumerate(cShape))
        padded[2 - cShape.size() + idx] = dim;
      if ((padded[0] != 1 && padded[0] != m) ||
          (padded[1] != 1 && padded[1] != n))
        return rewriter.notifyMatchFailure(op, "C is not broadcastable to Y");
      if (cShape != ArrayRef<int64_t>(padded))
        c = reshapeTo(c, padded, rewriter);

      if (op.getBeta().convertToFloat() != 1.0f)
        c = scaleByF32(c, op.getBeta().convertToFloat(),
                       cast<RankedTensorType>(cType.clone(padded)), rewriter,
                       loc);
      result = tosa::AddOp::create(rewriter, loc, resultType, result, c);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Covers the hip ops whose operands are (ctx, lhs, rhs, output) and whose TOSA
// counterpart preserves the element type. Comparisons (hip.equal, hip.less) do
// not belong here: they produce i1, which isTosaCompatibleOperand's
// element-type check rejects, so ComparisonConverter below takes them.
// hip.div does not either, since TOSA has no single divide; DivConverter below
// spells one out per element type.
//
// tosa.maximum and tosa.minimum additionally carry a nan_mode attribute, but
// ODS defaults it to PROPAGATE, which is what ONNX Max/Min do, so the
// two-operand builder below is correct for them unchanged.
//
// BoolOnly marks the logical ops (hip.and, hip.or). They are spelled
// tosa.bitwise_and / tosa.bitwise_or rather than the tosa.logical_* that would
// be the obvious mapping, because rocMLIR's RockTosaToElementwise has no
// pattern for the logical forms and its conversion target marks every
// surviving tosa op illegal -- the kernel would convert cleanly here and then
// fail to lower. The bitwise forms do have patterns (arith.andi / arith.ori),
// and on i1 they compute exactly the logical operation.
//
// The gate is what keeps that true: bitwise and logical only coincide on i1,
// so a wider integer has to be rejected rather than silently given bitwise
// semantics. hip.and's operands are declared Hip_TensorOrMemRef and nothing
// upstream narrows them, so the check cannot be left to the type system.
template <typename HipOpTy, typename TosaOpTy, bool BoolOnly = false>
struct BinaryConverter final : public OpConversionPattern<HipOpTy> {
  using OpConversionPattern<HipOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<HipOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(HipOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (BoolOnly && !resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "tosa op requires an i1 tensor");

    // hip broadcasting rank-extends the way ONNX/NumPy do, so a ReLU lowered
    // from ONNX arrives as hip.max(tensor<1x64x112x112xf16>, tensor<f16>),
    // while TOSA requires both operands to already carry the result's rank and
    // broadcasts size-1 dimensions only. EqualizeRanks reshapes the shorter
    // operand by prepending 1s; the TOSA op then broadcasts those dimensions.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (failed(tosa::EqualizeRanks(rewriter, op.getLoc(), lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");
    // Re-check after equalization so a dimension that still cannot broadcast is
    // rejected rather than emitting invalid TOSA.
    if (!isTosaCompatibleOperand(lhs, resultType) ||
        !isTosaCompatibleOperand(rhs, resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    // tosa.mul is the one op in this set that is not two-operand; its shift
    // is excluded from its own same-rank verification, so equalizing the two
    // data operands above is still all that is required.
    if constexpr (std::is_same_v<TosaOpTy, tosa::MulOp>)
      rewriter.replaceOpWithNewOp<TosaOpTy>(
          op, resultType, lhs, rhs, createZeroMulShift(rewriter, op.getLoc()));
    else
      rewriter.replaceOpWithNewOp<TosaOpTy>(op, resultType, lhs, rhs);
    return success();
  }
};

// The element types this pass can actually put through TOSA.
//
// Both predicates are deliberately allow-lists rather than "everything except
// the one type we know is broken". Tosa_FloatTensor is AnyFloat and Tosa_Int
// is any signless or unsigned integer, so f64, f80, f128, the float8 variants,
// i4 and i128 all satisfy the op verifiers while nothing downstream can lower
// them -- rocMLIR has no path for any of them and the TOSA profiles do not
// carry them. Excluding only f64 would still hand a float8 or i4 model to TOSA
// that then fails later. The float set is the one ConvConverter and
// GemmConverter above already use; the integer set is the widths ONNX actually
// produces.
static bool isTosaExpressibleFloat(Type elementType) {
  return elementType.isF16() || elementType.isBF16() || elementType.isF32();
}

static bool isTosaExpressibleInt(Type elementType) {
  return elementType.isSignlessInteger(1) || elementType.isSignlessInteger(8) ||
         elementType.isSignlessInteger(16) ||
         elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
}

// ONNX has no signless integers, and ORT imports ONNX bool as ui8 rather than
// i1 -- see the ui8 cases in test/lit/Conversion/hip-to-llvm/test_and.mlir and
// test/lit/Conversion/onnx-to-hip/test_greater.mlir, which the runtime path
// serves today. TOSA integers are signless and carry no unsigned ordering, so
// those element types have no faithful spelling in this pass.
//
// The predicates below therefore drive legality rather than being left to fail
// inside a converter. The distinction matters: an op this pass marks illegal
// but cannot rewrite aborts the whole function's conversion, taking the
// fusible ops around it down with it, whereas an op left legal stays a hip op
// and reaches the runtime lowering that already handles it.
//
// That only holds while each predicate claims exactly what its converter
// accepts. A predicate that is looser anywhere -- a type, a rank, a static
// shape, an operand it never looks at -- reintroduces the same abort it exists
// to prevent, so each one below mirrors its pattern's preconditions rather
// than approximating them.
static bool isTosaExpressibleCompareOperand(Type elementType) {
  return isTosaExpressibleFloat(elementType) ||
         isTosaExpressibleInt(elementType);
}

// Shared operand precondition for the elementwise patterns: a static ranked
// tensor of the given element type that broadcasts up to the result shape.
//
// This mirrors what the patterns themselves do. tosa::EqualizeRanks prepends
// 1s until the operand carries the result's rank, and isTosaBroadcastableShape
// then requires every dimension to equal the result's or be 1. Approximating
// it with a rank comparison is not enough: tensor<4xi1> into a 2x8 result has
// the smaller rank but still cannot broadcast, because the prepended 1 leaves
// 4 against 8.
static bool isBroadcastableOperandOf(Value operand, Type elementType,
                                     RankedTensorType resultType) {
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || !type.hasStaticShape() || type.getElementType() != elementType)
    return false;
  int64_t rank = type.getRank();
  int64_t resultRank = resultType.getRank();
  if (rank > resultRank)
    return false;
  ArrayRef<int64_t> shape = type.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim = shape[i];
    int64_t resultDim = resultShape[resultRank - rank + i];
    if (dim != resultDim && dim != 1)
      return false;
  }
  return true;
}

// hip.and, hip.or and hip.not are spelled with the tosa.bitwise_* ops, which
// coincide with the logical operation only on i1, so i1 is the whole of what
// this pass can claim for them. Every tensor operand has to be i1 and static
// as well: BinaryConverter rejects a dynamic or un-broadcastable operand even
// when the result looks fine, and claiming those would abort the conversion.
static bool isTosaExpressibleLogical(Operation *op, ValueRange operands) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      !resultType.getElementType().isInteger(1))
    return false;
  Type i1 = resultType.getElementType();
  for (Value operand : operands)
    if (!isBroadcastableOperandOf(operand, i1, resultType))
      return false;
  return true;
}

// A comparison is expressible when its result is i1 and the compared type is
// one TOSA can carry. Both halves matter: a ui8-result hip.less from OnnxToHip
// and a ui8-operand comparison are each valid hip that this pass must leave
// alone rather than reject.
template <typename CompareOpTy>
static bool isTosaExpressibleCompare(CompareOpTy op) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape() ||
      !resultType.getElementType().isInteger(1))
    return false;

  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  if (!lhsType)
    return false;
  Type operandElemType = lhsType.getElementType();
  if (!isTosaExpressibleCompareOperand(operandElemType))
    return false;
  for (Value operand : {op.getLhs(), op.getRhs()})
    if (!isBroadcastableOperandOf(operand, operandElemType, resultType))
      return false;
  return true;
}

// ONNX Sign covers floats and signed integers. i1 is excluded because -1 is
// not representable in it, and unsigned is excluded for want of an ordering.
// SignConverter additionally requires the input type to equal the result type
// exactly -- it builds every constant and comparison at the result type -- so
// the predicate checks that rather than the result alone.
static bool isTosaExpressibleSign(SignOp op) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  if (op.getX().getType() != resultType)
    return false;
  Type elementType = resultType.getElementType();
  if (elementType.isInteger(1))
    return false;
  return isTosaExpressibleFloat(elementType) ||
         isTosaExpressibleInt(elementType);
}

// Covers hip.equal and hip.less, whose result is i1 while their operands carry
// the type being compared. That mismatch is why they cannot use
// BinaryConverter: isTosaCompatibleOperand checks an operand's element type
// against the result's, which would reject every well-formed comparison. The
// broadcast check below is made against the operand element type instead, and
// the result is required to be i1 so a malformed hip op is named here rather
// than by the TOSA verifier.
//
// TOSA spells the ordered comparisons as tosa.greater and tosa.greater_equal
// only, so hip.less converts to tosa.greater with its operands swapped.
// OnnxToHip already leans on the same identity in the other direction --
// onnx.Greater is decomposed to hip.less(B, A) -- so an ONNX Greater arrives
// here swapped and leaves as the tosa.greater it started out as.
template <typename HipOpTy, typename TosaOpTy, bool SwapOperands = false>
struct ComparisonConverter final : public OpConversionPattern<HipOpTy> {
  using OpConversionPattern<HipOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<HipOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(HipOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "comparison must produce i1");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (failed(tosa::EqualizeRanks(rewriter, op.getLoc(), lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType ||
        lhsType.getElementType() != rhsType.getElementType())
      return rewriter.notifyMatchFailure(op, "operand element types differ");
    // Matching operand types alone do not make a TOSA comparison lowerable;
    // the compared type still has to be one TOSA can carry. Unsigned integers
    // are excluded rather than passed through: ONNX Equal/Less accept
    // ui8/ui16/ui32/ui64 and OnnxToHip preserves them, but TOSA integers are
    // signless, so a ui8 255 would compare as -1.
    Type operandElemType = lhsType.getElementType();
    if (!isTosaExpressibleCompareOperand(operandElemType))
      return rewriter.notifyMatchFailure(
          op, "comparison operand type has no tosa spelling");

    // Both TOSA comparisons carry SameOperandsElementType, and each operand
    // still has to broadcast up to the result's shape. Checking against a
    // result-shaped tensor of the operand element type applies the shape rule
    // without the element-type rule that i1 would fail.
    auto operandShaped = resultType.clone(lhsType.getElementType());
    if (!isTosaCompatibleOperand(lhs, operandShaped) ||
        !isTosaCompatibleOperand(rhs, operandShaped))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    if constexpr (SwapOperands)
      std::swap(lhs, rhs);
    rewriter.replaceOpWithNewOp<TosaOpTy>(op, resultType, lhs, rhs);
    return success();
  }
};

// TOSA has no single divide, so hip.div splits on element type. Integers get
// tosa.intdiv; floats get the reciprocal-then-multiply that tosa.intdiv's own
// description prescribes ("Floating point divide should use RECIPROCAL and
// MUL"). MIGraphXToTosa lowers migraphx.div the same two ways.
//
// The split is also why hip.div is the one hip op here whose legality turns on
// the element type. tosa.intdiv takes Tosa_Int32Or64Tensor, which is signless
// i32 and i64 only, whereas hip.div carries anything the runtime can name --
// ui8, i8, ui16 and i16 among them -- because nothing upstream narrows it: the
// operand constraint is AnyRankedTensor and OnnxToHip copies the ONNX element
// type through verbatim.
//
// Those widths are rejected here rather than left alone. Passing one through
// looks like the conservative choice but is not available: this pass only runs
// inside a rock.kernel, and rocMLIR compiles that kernel to an ELF, so a
// surviving hip op fails there instead -- later, and reported as an op from a
// dialect it has never heard of rather than as the unsupported element type it
// actually is. Failing here names the type.
//
// MIGraphXToTosa reaches unsigned, which this pass cannot: its type converter
// rewrites unsigned to signless and a tosa.custom "unsigned_div" carries the
// signedness in its name instead, which works because arith.divui takes the
// signless type that arrives. Reproducing that needs a type converter over the
// whole pass, not a change to this pattern. On sub-32-bit signed integers
// MIGraphXToTosa does no better than rejecting them: it emits a tosa.intdiv
// that fails the verifier.
static bool isTosaExpressibleDivType(Type elementType) {
  if (isa<FloatType>(elementType))
    return true;
  return elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
}

struct DivConverter final : public OpConversionPattern<hip::DivOp> {
  using OpConversionPattern<hip::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");

    Type elementType = resultType.getElementType();
    if (!isTosaExpressibleDivType(elementType))
      return op->emitError("hip.div has no TOSA spelling for element type ")
             << elementType << ": tosa.intdiv takes signless i32 and i64 only";

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (failed(tosa::EqualizeRanks(rewriter, op.getLoc(), lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");
    if (!isTosaCompatibleOperand(lhs, resultType) ||
        !isTosaCompatibleOperand(rhs, resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    if (isa<IntegerType>(elementType)) {
      rewriter.replaceOpWithNewOp<tosa::IntDivOp>(op, resultType, lhs, rhs);
      return success();
    }

    // The reciprocal is taken at the divisor's own rank-equalized shape rather
    // than the result's, so a divisor that broadcasts is reciprocated once per
    // distinct element instead of once per result element; the multiply
    // broadcasts it back up.
    Value recip =
        tosa::ReciprocalOp::create(rewriter, op.getLoc(), rhs.getType(), rhs)
            .getResult();
    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op, resultType, lhs, recip, createZeroMulShift(rewriter, op.getLoc()));
    return success();
  }
};

// Elementwise unary hip ops all share the (ctx, x, y) operand shape, and their
// TOSA counterparts are Tosa_ElementwiseUnaryOp, which carries
// SameOperandsAndResultShape and SameOperandsAndResultElementType. So unlike
// the binary ops there is no broadcasting to reason about: the operand has to
// match the result exactly, and the broadcast-tolerant check above would
// wrongly admit a size-1 operand and emit invalid TOSA.
//
// FloatOnly marks the ops that must not see an integer operand. tosa.sin and
// tosa.cos take Tosa_FloatTensor, so an integer would fail the TOSA verifier
// outright; the rest take Tosa_Tensor but are only available in TOSA's FP
// profile, so an integer would verify and then have no lowering. Both are
// unreachable from a valid ONNX model, where all ten are float-only, so the
// gate documents the constraint rather than rejecting real inputs.
template <typename HipOpTy, typename TosaOpTy, bool FloatOnly = false>
struct UnaryConverter final : public OpConversionPattern<HipOpTy> {
  using OpConversionPattern<HipOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<HipOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(HipOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getX().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (FloatOnly && !isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");

    // tosa.negate takes zero-point operands, but its quant-info builder
    // materializes them from this same (result type, input) signature.
    rewriter.replaceOpWithNewOp<TosaOpTy>(op, resultType, adaptor.getX());
    return success();
  }
};

// hip.transpose and tosa.transpose share ONNX's permutation convention.
struct TransposeConverter final : public OpConversionPattern<hip::TransposeOp> {
  using OpConversionPattern<hip::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!resultType || !resultType.hasStaticShape() || !inputType ||
        !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (inputType.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "expected rank 1 or higher");

    SmallVector<int32_t> perms;
    perms.reserve(op.getPerm().size());
    for (Attribute perm : op.getPerm())
      perms.push_back(static_cast<int32_t>(cast<IntegerAttr>(perm).getInt()));

    rewriter.replaceOpWithNewOp<tosa::TransposeOp>(
        op, resultType, adaptor.getInput(),
        rewriter.getDenseI32ArrayAttr(perms));
    return success();
  }
};

// Expand to a multiply-by-ones form that TosaToRock folds to layout transforms.
static bool isTosaExpressibleExpand(hip::ExpandOp op) {
  if (op.getNumResults() != 1)
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!resultType || !resultType.hasStaticShape() || !inputType ||
      !inputType.hasStaticShape())
    return false;
  if (resultType.getElementType() != inputType.getElementType())
    return false;

  int64_t offset = resultType.getRank() - inputType.getRank();
  if (offset < 0)
    return false;

  ArrayRef<int64_t> shape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (int64_t i = 0, e = inputType.getRank(); i < e; ++i)
    if (shape[i] != resultShape[i + offset] && shape[i] != 1)
      return false;
  return true;
}

struct ExpandConverter final : public OpConversionPattern<hip::ExpandOp> {
  using OpConversionPattern<hip::ExpandOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleExpand(op))
      return rewriter.notifyMatchFailure(op, "expected a static broadcast");

    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    Value ones = tosa::ConstOp::create(
        rewriter, op.getLoc(), resultType,
        cast<ElementsAttr>(rewriter.getOneAttr(resultType)));

    Value input = adaptor.getInput();
    if (failed(tosa::EqualizeRanks(rewriter, op.getLoc(), input, ones)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");

    rewriter.replaceOpWithNewOp<tosa::MulOp>(
        op, resultType, input, ones, createZeroMulShift(rewriter, op.getLoc()));
    return success();
  }
};

template <typename TensorOpTy> static bool isStaticReshape(TensorOpTy op) {
  auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  return srcType && srcType.hasStaticShape() && resultType &&
         resultType.hasStaticShape();
}

template <typename TensorOpTy>
struct ReshapeConverter final : public OpConversionPattern<TensorOpTy> {
  using OpConversionPattern<TensorOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TensorOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TensorOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStaticReshape(op))
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, resultType, adaptor.getSrc(),
        createConstShape(rewriter, op.getLoc(), resultType.getShape()));
    return success();
  }
};

// TOSA has no sqrt. Emit tosa.reciprocal(tosa.rsqrt(x)), which
// RockTosaToElementwise folds back into a single math.sqrt.
struct SqrtConverter final : public OpConversionPattern<SqrtOp> {
  using OpConversionPattern<SqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getX().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");

    auto rsqrt = tosa::RsqrtOp::create(rewriter, op.getLoc(), resultType,
                                       adaptor.getX());
    rewriter.replaceOpWithNewOp<tosa::ReciprocalOp>(op, resultType, rsqrt);
    return success();
  }
};

// TOSA has no silu/swish. Both expand to the ONNX definition:
//   silu(x)        = x * sigmoid(x)
//   swish(x,alpha) = x * sigmoid(alpha * x)
// with alpha default 1, which is silu. hip.silu is already a FuseROCMlir
// pointwise consumer; hip.swish is not, so the swish pattern still needs
// rock.kernel on the test function.
//
// Before:
//   %y = hip.silu(%ctx) ins(%x : tensor<2x8xf16>)
//                       outs(%init : tensor<2x8xf16>) -> tensor<2x8xf16>
// After:
//   %s = tosa.sigmoid %x
//   %y = tosa.mul %x, %s
LogicalResult matchStaticFloatSameType(Operation *op, Value input,
                                       RankedTensorType &resultType,
                                       ConversionPatternRewriter &rewriter) {
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");
  resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
  if (input.getType() != resultType)
    return rewriter.notifyMatchFailure(
        op, "operand and result types must match exactly");
  if (!isa<FloatType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");
  return success();
}

Value emitMulSigmoid(Value x, Value preSigmoid, RankedTensorType ty,
                     ConversionPatternRewriter &rewriter, Location loc) {
  Value sig = tosa::SigmoidOp::create(rewriter, loc, ty, preSigmoid);
  return tosa::MulOp::create(rewriter, loc, ty, x, sig,
                             createZeroMulShift(rewriter, loc));
}

struct SiluConverter final : public OpConversionPattern<SiluOp> {
  using OpConversionPattern<SiluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    if (failed(matchStaticFloatSameType(op, adaptor.getInput(), resultType,
                                        rewriter)))
      return failure();
    rewriter.replaceOp(op,
                       emitMulSigmoid(adaptor.getInput(), adaptor.getInput(),
                                      resultType, rewriter, op.getLoc()));
    return success();
  }
};

struct SwishConverter final : public OpConversionPattern<SwishOp> {
  using OpConversionPattern<SwishOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SwishOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    if (failed(matchStaticFloatSameType(op, adaptor.getInput(), resultType,
                                        rewriter)))
      return failure();
    Location loc = op.getLoc();
    Value x = adaptor.getInput();
    double alphaVal = op.getAlpha().convertToDouble();
    Value preSigmoid = x;
    if (alphaVal != 1.0)
      preSigmoid = tosa::MulOp::create(
          rewriter, loc, resultType, x,
          createSplatFloat(rewriter, loc, resultType, alphaVal),
          createZeroMulShift(rewriter, loc));
    rewriter.replaceOp(
        op, emitMulSigmoid(x, preSigmoid, resultType, rewriter, loc));
    return success();
  }
};

// TOSA has no gelu. Expand to the formula hip.gelu's own description
// spells out, matching wrap_gelu / hip_elementwise_gelu:
//
//   erf (approximate = "none"):
//     y = 0.5 * x * (1 + erf(x * 1/sqrt(2)))
//   tanh (approximate = "tanh", also hip.fast_gelu):
//     y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Cubing uses tosa.mul rather than tosa.pow so it holds for every float
// type TOSA accepts. Dividing by sqrt(2) is a multiply by the reciprocal
// constant, because TOSA has no float divide.
//
// hip.bias_gelu is Gelu(data + last-dim-broadcast(bias)) with the erf
// form. hip.fast_gelu is the tanh form, with optional bias added first.
enum class GeluKind { Erf, Tanh };

static LogicalResult matchGeluTensor(Operation *op, Value input,
                                     ConversionPatternRewriter &rewriter,
                                     RankedTensorType &resultType) {
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");
  resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
  if (input.getType() != resultType)
    return rewriter.notifyMatchFailure(
        op, "operand and result types must match exactly");
  if (!isa<FloatType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");
  return success();
}

static FailureOr<Value> addBroadcastBias(ConversionPatternRewriter &rewriter,
                                         Location loc,
                                         RankedTensorType resultType,
                                         Value data, Value bias) {
  if (failed(tosa::EqualizeRanks(rewriter, loc, data, bias)))
    return failure();
  if (!isTosaCompatibleOperand(data, resultType) ||
      !isTosaCompatibleOperand(bias, resultType))
    return failure();
  return tosa::AddOp::create(rewriter, loc, resultType, data, bias).getResult();
}

static Value emitGelu(ConversionPatternRewriter &rewriter, Location loc,
                      RankedTensorType type, Value x, GeluKind kind) {
  Value shift = createZeroMulShift(rewriter, loc);
  auto mul = [&](Value lhs, Value rhs) {
    return tosa::MulOp::create(rewriter, loc, type, lhs, rhs, shift);
  };
  auto add = [&](Value lhs, Value rhs) {
    return tosa::AddOp::create(rewriter, loc, type, lhs, rhs);
  };

  Value half = createSplatFloat(rewriter, loc, type, 0.5);
  Value one = createSplatFloat(rewriter, loc, type, 1.0);
  Value inner;
  if (kind == GeluKind::Tanh) {
    Value x2 = mul(x, x);
    Value x3 = mul(x2, x);
    Value coeff = createSplatFloat(rewriter, loc, type, 0.044715);
    Value k = createSplatFloat(rewriter, loc, type, 0.7978845608028654);
    Value tanhArg = mul(k, add(x, mul(coeff, x3)));
    inner = add(one, tosa::TanhOp::create(rewriter, loc, type, tanhArg));
  } else {
    Value invSqrt2 = createSplatFloat(rewriter, loc, type, 0.7071067811865476);
    inner =
        add(one, tosa::ErfOp::create(rewriter, loc, type, mul(x, invSqrt2)));
  }
  return mul(mul(x, half), inner);
}

// Before:
//   %r = hip.gelu(%ctx) ins(%x : tensor<2x8xf16>)
//                       outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
// After (exact):
//   %c = tosa.const dense<0.7071...>
//   %s = tosa.mul %x, %c
//   %e = tosa.erf %s
//   %t = tosa.add %one, %e
//   %r = tosa.mul (tosa.mul %x, %half), %t
struct GeluConverter final : public OpConversionPattern<GeluOp> {
  using OpConversionPattern<GeluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    if (failed(matchGeluTensor(op, adaptor.getInput(), rewriter, resultType)))
      return failure();

    StringRef approximate = op.getApproximate();
    if (approximate != "none" && approximate != "tanh")
      return rewriter.notifyMatchFailure(
          op, "approximate must be \"none\" or \"tanh\"");

    GeluKind kind = approximate == "tanh" ? GeluKind::Tanh : GeluKind::Erf;
    rewriter.replaceOp(op, emitGelu(rewriter, op.getLoc(), resultType,
                                    adaptor.getInput(), kind));
    return success();
  }
};

struct BiasGeluConverter final : public OpConversionPattern<BiasGeluOp> {
  using OpConversionPattern<BiasGeluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BiasGeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    if (failed(matchGeluTensor(op, adaptor.getData(), rewriter, resultType)))
      return failure();

    FailureOr<Value> biased =
        addBroadcastBias(rewriter, op.getLoc(), resultType, adaptor.getData(),
                         adaptor.getBias());
    if (failed(biased))
      return rewriter.notifyMatchFailure(op, "bias is not tosa-broadcastable");

    rewriter.replaceOp(op, emitGelu(rewriter, op.getLoc(), resultType, *biased,
                                    GeluKind::Erf));
    return success();
  }
};

struct FastGeluConverter final : public OpConversionPattern<FastGeluOp> {
  using OpConversionPattern<FastGeluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FastGeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    if (failed(matchGeluTensor(op, adaptor.getInput(), rewriter, resultType)))
      return failure();

    Value x = adaptor.getInput();
    if (Value bias = adaptor.getBias()) {
      FailureOr<Value> biased =
          addBroadcastBias(rewriter, op.getLoc(), resultType, x, bias);
      if (failed(biased))
        return rewriter.notifyMatchFailure(op,
                                           "bias is not tosa-broadcastable");
      x = *biased;
    }

    rewriter.replaceOp(
        op, emitGelu(rewriter, op.getLoc(), resultType, x, GeluKind::Tanh));
    return success();
  }
};

// TOSA has no softplus. Expand to the numerically stable form the
// hip_softplus kernel uses:
//   softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
//
// The naive log(1 + exp(x)) overflows for values ONNX still considers
// valid (f16 20 is ordinary; exp(20) is already Inf), so a fused kernel
// that used it would disagree with wrap_softplus rather than fail.
//
// Before:
//   %r = hip.softplus(%ctx) ins(%x : tensor<2x8xf16>)
//                           outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
// After:
//   %abs = tosa.abs %x
//   %neg = tosa.negate %abs
//   %e = tosa.exp %neg
//   %one = tosa.const dense<1.0> : tensor<2x8xf16>
//   %s = tosa.add %e, %one
//   %l = tosa.log %s
//   %zero = tosa.const dense<0.0> : tensor<2x8xf16>
//   %m = tosa.maximum %x, %zero
//   %r = tosa.add %m, %l
struct SoftplusConverter final : public OpConversionPattern<SoftplusOp> {
  using OpConversionPattern<SoftplusOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SoftplusOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getX().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");

    Location loc = op.getLoc();
    Value x = adaptor.getX();
    Value abs = tosa::AbsOp::create(rewriter, loc, resultType, x);
    Value neg = tosa::NegateOp::create(rewriter, loc, resultType, abs);
    Value exp = tosa::ExpOp::create(rewriter, loc, resultType, neg);
    Value one = createSplatFloat(rewriter, loc, resultType, 1.0);
    Value sum = tosa::AddOp::create(rewriter, loc, resultType, exp, one);
    Value log = tosa::LogOp::create(rewriter, loc, resultType, sum);
    Value zero = createSplatFloat(rewriter, loc, resultType, 0.0);
    Value relu = tosa::MaximumOp::create(rewriter, loc, resultType, x, zero);
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType, relu, log);
    return success();
  }
};

// hip.not is logical negation on i1. It becomes tosa.bitwise_xor against an
// all-ones constant rather than the tosa.logical_not that would map 1-1,
// because RockTosaToElementwise has a pattern for neither tosa.logical_not nor
// tosa.bitwise_not, while tosa.bitwise_xor lowers to arith.xori. On i1,
// `x ^ true` is exactly `!x`.
//
// That is also why this cannot be a UnaryConverter: the xor needs a second
// operand. The i1 requirement is the same one hip.and and hip.or carry, and
// for the same reason -- bitwise and logical coincide only on i1.
struct LogicalNotConverter final : public OpConversionPattern<NotOp> {
  using OpConversionPattern<NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getX().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (!resultType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(op, "hip.not requires an i1 tensor");

    Location loc = op.getLoc();
    Value ones = tosa::ConstOp::create(
        rewriter, loc, resultType,
        DenseElementsAttr::get(resultType, rewriter.getBoolAttr(true)));
    rewriter.replaceOpWithNewOp<tosa::BitwiseXorOp>(op, resultType,
                                                    adaptor.getX(), ones);
    return success();
  }
};

// hip.sign is ONNX Sign: +1 where x > 0, -1 where x < 0, and 0 elsewhere.
// TOSA has no sign op, so this expands to the nested select that the
// definition spells out directly:
//   select(x > 0, 1, select(0 > x, -1, 0))
// tosa.greater is TOSA's only strict ordered comparison, which is why the
// negative test reads `0 > x` rather than a `less`. Signed zero lands on 0 as
// ONNX requires, both comparisons being false for it.
//
// Both comparisons are also false for NaN, which would put it on the same 0.
// ONNX defines sign(NaN) = NaN, and lib/Runtime/real/sign.cpp propagates it as
// a deliberate delta from ORT (whose _Signum returns 0), so the float case is
// wrapped in an ordered self-compare: `x == x` is false for exactly NaN, and
// forwarding x there keeps a fused model agreeing with the unfused one.
// Integers have no NaN, so they skip the guard.
//
// Before:
//   %r = hip.sign(%ctx) ins(%x : tensor<4xf32>)
//                       outs(%init : tensor<4xf32>) : tensor<4xf32>
//
// After:
//   %zero   = "tosa.const"() <{values = dense<0.0> : tensor<4xf32>}>
//   %pos    = tosa.greater %x, %zero  : (tensor<4xf32>, tensor<4xf32>)
//                                        -> tensor<4xi1>
//   %isneg  = tosa.greater %zero, %x  : ... -> tensor<4xi1>
//   %neg1   = "tosa.const"() <{values = dense<-1.0> : tensor<4xf32>}>
//   %inner  = tosa.select %isneg, %neg1, %zero  : ... -> tensor<4xf32>
//   %one    = "tosa.const"() <{values = dense<1.0> : tensor<4xf32>}>
//   %signum = tosa.select %pos, %one, %inner    : ... -> tensor<4xf32>
//   %ord    = tosa.equal %x, %x       : ... -> tensor<4xi1>
//   %r      = tosa.select %ord, %signum, %x     : ... -> tensor<4xf32>
//
// The integer case is the same without the %ord guard and its select.
struct SignConverter final : public OpConversionPattern<SignOp> {
  using OpConversionPattern<SignOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getX().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");

    // ONNX Sign is defined over both floats and signed integers, and the
    // tosa.greater / tosa.select pair below takes either. i1 is excluded
    // because -1 is not representable in it.
    Type elementType = resultType.getElementType();
    bool isFloat = isTosaExpressibleFloat(elementType);
    if (!isFloat && !elementType.isSignlessInteger())
      return rewriter.notifyMatchFailure(
          op, "expected a tosa-expressible float or signless integer tensor");
    if (elementType.isInteger(1))
      return rewriter.notifyMatchFailure(op, "i1 cannot represent -1");

    Location loc = op.getLoc();
    Value x = adaptor.getX();
    auto splat = [&](double value) -> Value {
      if (isFloat)
        return createSplatFloat(rewriter, loc, resultType, value);
      return tosa::ConstOp::create(
          rewriter, loc, resultType,
          DenseElementsAttr::get(
              resultType, rewriter.getIntegerAttr(
                              elementType, static_cast<int64_t>(value))));
    };

    Value zero = splat(0.0);
    auto predType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    Value isPositive =
        tosa::GreaterOp::create(rewriter, loc, predType, x, zero);
    Value isNegative =
        tosa::GreaterOp::create(rewriter, loc, predType, zero, x);
    // The inner select reuses `zero` as its else value, so the expansion needs
    // three constants rather than four.
    Value negativeOrZero = tosa::SelectOp::create(
        rewriter, loc, resultType, isNegative, splat(-1.0), zero);
    Value signum = tosa::SelectOp::create(rewriter, loc, resultType, isPositive,
                                          splat(1.0), negativeOrZero);
    if (!isFloat) {
      rewriter.replaceOp(op, signum);
      return success();
    }

    // ONNX defines sign(NaN) = NaN, and lib/Runtime/real/sign.cpp propagates
    // it deliberately -- ORT's own _Signum returns 0 there, and the HIP kernel
    // documents diverging from ORT to follow the spec. Both comparisons above
    // are false for NaN, so the selects alone would return zero and a fused
    // model would disagree with the unfused one. tosa.equal is an ordered
    // compare, making x == x false for exactly NaN, so this guard forwards the
    // input unchanged in that case and costs one compare and one select
    // otherwise.
    Value isOrdered = tosa::EqualOp::create(rewriter, loc, predType, x, x);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, isOrdered,
                                                signum, x);
    return success();
  }
};

// hip.where is ternary (cond, x, y). tosa.select is the 1-1 mapping; it cannot
// use BinaryConverter because the predicate is i1 while the result is not.
// EqualizeRanks is pairwise, so the three operands are equalized the same way
// CreateOpAndInferShape does for Select.
struct WhereConverter final : public OpConversionPattern<WhereOp> {
  using OpConversionPattern<WhereOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");

    Value cond = adaptor.getCondition();
    Value x = adaptor.getX();
    Value y = adaptor.getY();
    Location loc = op.getLoc();
    if (failed(tosa::EqualizeRanks(rewriter, loc, cond, x)) ||
        failed(tosa::EqualizeRanks(rewriter, loc, cond, y)) ||
        failed(tosa::EqualizeRanks(rewriter, loc, x, y)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");

    auto condType = dyn_cast<RankedTensorType>(cond.getType());
    if (!condType || !condType.getElementType().isInteger(1) ||
        !isTosaBroadcastableShape(condType, resultType))
      return rewriter.notifyMatchFailure(
          op, "condition is not a tosa-broadcastable i1 tensor");
    if (!isTosaCompatibleOperand(x, resultType) ||
        !isTosaCompatibleOperand(y, resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, cond, x, y);
    return success();
  }
};

// hip.leaky_relu is unary plus an `alpha` attribute; TOSA has no matching op.
// y = x >= 0 ? x : alpha * x. For alpha in [0, 1] that is
//   tosa.maximum(x, tosa.mul(x, splat(alpha))).
// Otherwise emit tosa.select(tosa.greater(x, 0), x, scaled).
struct LeakyReluConverter final : public OpConversionPattern<LeakyReluOp> {
  using OpConversionPattern<LeakyReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LeakyReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    Value x = adaptor.getInput();
    if (x.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");

    Location loc = op.getLoc();
    double alphaVal = op.getAlpha().convertToDouble();
    Value alpha = createSplatFloat(rewriter, loc, resultType, alphaVal);
    Value scaled = tosa::MulOp::create(rewriter, loc, resultType, x, alpha,
                                       createZeroMulShift(rewriter, loc));
    if (alphaVal >= 0.0 && alphaVal <= 1.0) {
      rewriter.replaceOpWithNewOp<tosa::MaximumOp>(
          op, resultType, x, scaled, tosa::NanPropagationMode::IGNORE);
      return success();
    }

    Value zero = createSplatFloat(rewriter, loc, resultType, 0.0);
    auto predType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    Value pred = tosa::GreaterOp::create(rewriter, loc, predType, x, zero);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, pred, x,
                                                scaled);
    return success();
  }
};

bool extractConstantInts(Value v, SmallVectorImpl<int64_t> &out) {
  out.clear();
  IntegerAttr intAttr;
  DenseIntElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&intAttr))) {
    out.push_back(intAttr.getInt());
    return true;
  }
  if (matchPattern(v, m_Constant(&denseAttr))) {
    for (APInt e : denseAttr.getValues<APInt>())
      out.push_back(e.getSExtValue());
    return true;
  }
  return false;
}

// TOSA reduce ops take one i32 axis and always leave that dim as size 1.
// hip.reduce_* carry ONNX axes as a tensor plus keepdims /
// noop_with_empty_axes.
// Rewriter-free single-axis match. Returns an empty StringRef on success and
// the reason otherwise, so a caller that wants a diagnostic can report it and
// a caller that only wants a yes/no answer can ignore it. That split is what
// lets the legality predicates below ask exactly the question the patterns
// will answer, rather than a similar-looking one.
StringRef findSingleReduceAxis(Value axes, int64_t rank,
                               int64_t noopWithEmptyAxes, int32_t &axis) {
  SmallVector<int64_t, 4> axesVals;
  if (!extractConstantInts(axes, axesVals))
    return "axes must be a constant";
  if (axesVals.empty())
    return noopWithEmptyAxes
               ? "empty axes identity"
               : "empty axes reduce-all is not a single tosa reduce";
  if (axesVals.size() != 1)
    return "tosa reduce supports a single axis";

  int64_t resolved = axesVals[0];
  if (resolved < 0)
    resolved += rank;
  if (resolved < 0 || resolved >= rank)
    return "axis out of range";
  axis = static_cast<int32_t>(resolved);
  return StringRef();
}

RankedTensorType keepdimsReduceType(RankedTensorType dataType, int32_t axis) {
  SmallVector<int64_t> shape(dataType.getShape().begin(),
                             dataType.getShape().end());
  shape[axis] = 1;
  return RankedTensorType::get(shape, dataType.getElementType());
}

// TOSA has no reduce_mean. Scale by 1/N then reduce_sum on one axis
// (keepdims=1). Used by hip.reduce_mean and by RMS/LN over several axes.
FailureOr<Value> emitTosaKeepdimsReduceMean(Value data, int32_t axis,
                                            ConversionPatternRewriter &rewriter,
                                            Location loc, Operation *op) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || !dataType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked data");
  if (!isa<FloatType>(dataType.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "tosa reduce_mean lowering requires a float tensor");
  if (axis < 0 || axis >= dataType.getRank())
    return rewriter.notifyMatchFailure(op, "reduce axis out of range");
  int64_t n = dataType.getDimSize(axis);
  if (n <= 0)
    return rewriter.notifyMatchFailure(op, "reduce axis extent must be > 0");

  Type elemType = dataType.getElementType();
  auto oneType = RankedTensorType::get({1}, elemType);
  Value nConst =
      createSplatFloat(rewriter, loc, oneType, static_cast<double>(n));
  auto inv = tosa::ReciprocalOp::create(rewriter, loc, oneType, nConst);
  SmallVector<int64_t> ones(dataType.getRank(), 1);
  Value onesShape = tosa::getTosaConstShape(rewriter, loc, ones);
  auto invTy = RankedTensorType::get(ones, elemType);
  Value invShaped =
      tosa::ReshapeOp::create(rewriter, loc, invTy, inv, onesShape);
  Value scaled = tosa::MulOp::create(rewriter, loc, dataType, data, invShaped,
                                     createZeroMulShift(rewriter, loc));
  auto reducedTy = keepdimsReduceType(dataType, axis);
  return tosa::ReduceSumOp::create(rewriter, loc, reducedTy, scaled,
                                   rewriter.getI32IntegerAttr(axis))
      .getResult();
}

// hip.miopen.softmax is last-dim softmax. TOSA has no softmax op; expand to
// reduce_max/sub/exp/reduce_sum/reciprocal/mul, which TosaToRock attention
// matching expects.
struct SoftmaxConverter final : public OpConversionPattern<MiopenSoftmaxOp> {
  using OpConversionPattern<MiopenSoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MiopenSoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    Value input = adaptor.getInput();
    if (input.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    if (resultType.getRank() < 1)
      return rewriter.notifyMatchFailure(op, "tosa reduce requires rank >= 1");
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "tosa softmax requires a float tensor");

    int32_t axis = static_cast<int32_t>(resultType.getRank() - 1);
    auto reducedTy = keepdimsReduceType(resultType, axis);
    Location loc = op.getLoc();
    IntegerAttr axisAttr = rewriter.getI32IntegerAttr(axis);
    auto rmax =
        tosa::ReduceMaxOp::create(rewriter, loc, reducedTy, input, axisAttr);
    auto sub = tosa::SubOp::create(rewriter, loc, resultType, input, rmax);
    auto exp = tosa::ExpOp::create(rewriter, loc, resultType, sub);
    auto rsum =
        tosa::ReduceSumOp::create(rewriter, loc, reducedTy, exp, axisAttr);
    auto rec = tosa::ReciprocalOp::create(rewriter, loc, reducedTy, rsum);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, exp, rec,
                                             createZeroMulShift(rewriter, loc));
    return success();
  }
};

void replaceWithTosaReduce(Operation *op, Value reduced,
                           RankedTensorType resultType, bool keepdims,
                           ConversionPatternRewriter &rewriter) {
  if (keepdims) {
    rewriter.replaceOp(op, reduced);
    return;
  }
  Value shape =
      tosa::getTosaConstShape(rewriter, op->getLoc(), resultType.getShape());
  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(op, resultType, reduced, shape);
}

struct HipReduceMatch {
  RankedTensorType resultType;
  int32_t axis = 0;
  bool keepdims = false;
  bool identity = false;
};

// Everything the hip reductions share, with no rewriter: the single constant
// axis, keepdims, and the noop_with_empty_axes identity. Empty StringRef on
// success, reason otherwise.
//
// This exists as one implementation on purpose. The reductions are claimed by
// dynamic legality, and that only avoids aborting a conversion while the
// predicate claims exactly what the pattern accepts. A predicate that checks
// the element type and leaves the structure to the pattern will claim a
// dynamic, multi-axis or non-constant-axis reduction that the pattern then
// refuses, and an op marked illegal that no pattern rewrites takes the whole
// function's conversion down with it -- including the fusible ops around it.
// Both the patterns and the predicates therefore go through here.
StringRef matchHipReduceCore(Operation *op, Value data, Value axes,
                             int64_t keepdims, int64_t noopWithEmptyAxes,
                             HipReduceMatch &match) {
  match = HipReduceMatch{};
  if (op->getNumResults() != 1)
    return "expected tensor mode";
  match.resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!match.resultType || !match.resultType.hasStaticShape())
    return "expected a static ranked tensor";
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || !dataType.hasStaticShape())
    return "expected static ranked data";
  if (dataType.getRank() < 1)
    return "tosa reduce requires rank >= 1";

  match.keepdims = keepdims != 0;
  SmallVector<int64_t, 4> axesVals;
  if (!extractConstantInts(axes, axesVals))
    return "axes must be a constant";
  if (axesVals.empty() && noopWithEmptyAxes) {
    if (data.getType() != match.resultType)
      return "identity type mismatch";
    match.identity = true;
    return StringRef();
  }

  StringRef reason = findSingleReduceAxis(axes, dataType.getRank(),
                                          noopWithEmptyAxes, match.axis);
  if (!reason.empty())
    return reason;
  auto reducedTy = keepdimsReduceType(dataType, match.axis);
  if (match.keepdims && match.resultType != reducedTy)
    return "keepdims result type mismatch";
  if (!match.keepdims && match.resultType.getRank() != dataType.getRank() - 1)
    return "keepdims=0 rank mismatch";
  return StringRef();
}

LogicalResult matchHipReduce(Operation *op, Value data, Value axes,
                             int64_t keepdims, int64_t noopWithEmptyAxes,
                             ConversionPatternRewriter &rewriter,
                             RankedTensorType &resultType, Value &dataOut,
                             int32_t &axis, bool &keepdimsOut, bool &identity) {
  HipReduceMatch match;
  StringRef reason =
      matchHipReduceCore(op, data, axes, keepdims, noopWithEmptyAxes, match);
  if (!reason.empty())
    return rewriter.notifyMatchFailure(op, reason);
  resultType = match.resultType;
  dataOut = data;
  axis = match.axis;
  keepdimsOut = match.keepdims;
  identity = match.identity;
  return success();
}

// The element types this pass can actually put through TOSA.
//
// Both are allow-lists rather than "everything except the type we know is
// broken". Tosa_FloatTensor is AnyFloat and Tosa_Int is any signless or
// unsigned integer, so f64, f80, f128, the float8 variants, i4 and i128 all
// satisfy the op verifiers while nothing downstream can lower them -- and
// onnx.ReduceL2 explicitly permits f64 input. The float set is the one
// GemmConverter above already uses; the integer set names the widths ONNX
// produces.
static bool isTosaExpressibleFloat(Type elementType) {
  return elementType.isF16() || elementType.isBF16() || elementType.isF32();
}

static bool isTosaExpressibleInt(Type elementType) {
  return elementType.isSignlessInteger(1) || elementType.isSignlessInteger(8) ||
         elementType.isSignlessInteger(16) ||
         elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
}

// ONNX reductions accept unsigned element types and OnnxToHip preserves them,
// but no TOSA lowering takes an unsigned tensor: `tosa.reduce_product` on
// tensor<2x8xui32> dies in tosa-to-linalg with "'arith.constant' op integer
// return type must be signless", where the signless i32 form lowers to a
// linalg.reduce. So this one predicate covers every reduction here, ordered or
// not.
//
// Worth recording why reinterpreting the bits is not a shortcut: for sum and
// product it would actually work, since two's-complement add and multiply give
// the same bits whichever way the sign bit is read, but tosa.reduce_max and
// tosa.reduce_min are ordered and would read a ui8 255 as -1, turning a
// maximum into a minimum. Any future unsigned support has to handle those two
// differently rather than uniformly.
static bool isTosaReduceType(Type elementType) {
  return isTosaExpressibleFloat(elementType) ||
         isTosaExpressibleInt(elementType);
}

// A reduction is expressible when its element type has a TOSA spelling *and*
// matchHipReduceCore accepts its structure. Checking only the element type
// would leave a dynamic, multi-axis or non-constant-axis reduction claimed for
// a pattern that refuses it, which aborts the function's whole conversion
// instead of leaving the op for the runtime kernel that handles it.
template <typename ReduceOpTy>
static bool isTosaExpressibleReduce(ReduceOpTy op,
                                    bool (*elementTypeOk)(Type)) {
  auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
  if (!dataType || !elementTypeOk(dataType.getElementType()))
    return false;
  HipReduceMatch match;
  return matchHipReduceCore(op, op.getData(), op.getAxes(), op.getKeepdims(),
                            op.getNoopWithEmptyAxes(), match)
      .empty();
}

// Covers the hip reductions that map 1-1 onto a TOSA reduce: reduce_sum,
// reduce_max, reduce_min and reduce_prod. matchHipReduce already does
// everything the hip reductions share -- the single constant axis, keepdims,
// and the noop_with_empty_axes identity -- so only the op mapping varies. TOSA
// always reduces with keepdims=1, hence the optional reshape at the end.
//
// hip.reduce_mean and hip.reduce_l2 are not in this set: TOSA has neither, so
// each spells out its own expansion below.
//
// tosa.reduce_max and tosa.reduce_min additionally carry a nan_mode attribute,
// but ODS defaults it to PROPAGATE, which is what ONNX ReduceMax/ReduceMin do,
// so the builder below is correct for them unchanged.
//
// The element type is not re-checked here: these ops are claimed by
// isTosaExpressibleReduce, which asks isTosaReduceType before marking one
// illegal, so a type this pass cannot express never reaches the pattern.
template <typename HipOpTy, typename TosaOpTy>
struct ReduceConverter final : public OpConversionPattern<HipOpTy> {
  using OpConversionPattern<HipOpTy>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<HipOpTy>::OpAdaptor;

  LogicalResult
  matchAndRewrite(HipOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    Value data;
    int32_t axis = 0;
    bool keepdims = true;
    bool identity = false;
    if (failed(matchHipReduce(op, adaptor.getData(), adaptor.getAxes(),
                              op.getKeepdims(), op.getNoopWithEmptyAxes(),
                              rewriter, resultType, data, axis, keepdims,
                              identity)))
      return failure();
    if (identity) {
      rewriter.replaceOp(op, data);
      return success();
    }
    auto dataType = cast<RankedTensorType>(data.getType());
    auto reducedTy = keepdimsReduceType(dataType, axis);
    auto reduced = TosaOpTy::create(rewriter, op.getLoc(), reducedTy, data,
                                    rewriter.getI32IntegerAttr(axis));
    replaceWithTosaReduce(op, reduced, resultType, keepdims, rewriter);
    return success();
  }
};

// TOSA has no reduce_mean. Scale by 1/N then reduce_sum.
struct ReduceMeanConverter final : public OpConversionPattern<ReduceMeanOp> {
  using OpConversionPattern<ReduceMeanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    Value data;
    int32_t axis = 0;
    bool keepdims = true;
    bool identity = false;
    if (failed(matchHipReduce(op, adaptor.getData(), adaptor.getAxes(),
                              op.getKeepdims(), op.getNoopWithEmptyAxes(),
                              rewriter, resultType, data, axis, keepdims,
                              identity)))
      return failure();
    if (identity) {
      rewriter.replaceOp(op, data);
      return success();
    }
    FailureOr<Value> reduced =
        emitTosaKeepdimsReduceMean(data, axis, rewriter, op.getLoc(), op);
    if (failed(reduced))
      return failure();
    replaceWithTosaReduce(op, *reduced, resultType, keepdims, rewriter);
    return success();
  }
};

// TOSA has no reduce_l2. ONNX ReduceL2 is sqrt(sum(x^2)), which hip.reduce_l2's
// own description spells out, so it expands to mul + reduce_sum + square root.
//
// The square root is reciprocal(rsqrt(x)), because TOSA has no sqrt either;
// this is the expansion SqrtConverter already uses for hip.sqrt. It stays
// correct on an all-zero reduction, where rsqrt(0) is +inf and
// reciprocal(+inf) is 0, which is the norm ONNX asks for.
//
// The squaring is a plain tosa.mul of the data with itself rather than a
// tosa.pow, so it holds for every float type TOSA accepts instead of only
// those with a pow lowering.
//
// The square and the sum are computed in f32 even for an f16 input, and the
// result narrowed at the end. Squaring in f16 overflows well inside the range
// ONNX considers valid: 300 is an ordinary f16, but 300^2 is 90000 against an
// f16 maximum of 65504, so the norm of a single 300 would come back +inf
// instead of 300. reduce_l2_f16_kernel in lib/Runtime/Kernels/hip is the
// contract to match -- it widens with __half2float, squares and accumulates
// in float, and narrows once with __float2half at the end -- and a fused
// kernel that disagreed with it would be a silent numerical difference rather
// than a failure.
//
// Before:
//   %r = hip.reduce_l2(%ctx) ins(%data, %axes : tensor<2x8xf16>, tensor<1xi64>)
//                            outs(%init : tensor<2x1xf16>) {keepdims = 1}
//
// After:
//   %wide = tosa.cast %data                     : tensor<2x8xf16> -> 2x8xf32
//   %sq   = tosa.mul %wide, %wide, %shift       : tensor<2x8xf32>
//   %sum  = tosa.reduce_sum %sq {axis = 1}      : tensor<2x1xf32>
//   %rs   = tosa.rsqrt %sum                     : tensor<2x1xf32>
//   %norm = tosa.reciprocal %rs                 : tensor<2x1xf32>
//   %r    = tosa.cast %norm                     : tensor<2x1xf32> -> 2x1xf16
struct ReduceL2Converter final : public OpConversionPattern<ReduceL2Op> {
  using OpConversionPattern<ReduceL2Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceL2Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    Value data;
    int32_t axis = 0;
    bool keepdims = true;
    bool identity = false;
    if (failed(matchHipReduce(op, adaptor.getData(), adaptor.getAxes(),
                              op.getKeepdims(), op.getNoopWithEmptyAxes(),
                              rewriter, resultType, data, axis, keepdims,
                              identity)))
      return failure();
    // noop_with_empty_axes reduces nothing, so ONNX defines the result as the
    // input itself rather than as an elementwise norm.
    if (identity) {
      rewriter.replaceOp(op, data);
      return success();
    }
    auto dataType = cast<RankedTensorType>(data.getType());
    if (!isTosaExpressibleFloat(dataType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "tosa reduce_l2 lowering requires a tosa-expressible float "
              "tensor");

    Location loc = op.getLoc();
    // Widen a narrower float so the square and the accumulation happen in f32,
    // matching the runtime kernel. f32 input needs no cast.
    Type f32 = rewriter.getF32Type();
    Type elementType = dataType.getElementType();
    bool widen = elementType != f32;
    Value wide = data;
    RankedTensorType wideTy = dataType;
    if (widen) {
      wideTy = dataType.clone(f32);
      wide = tosa::CastOp::create(rewriter, loc, wideTy, data);
    }

    Value squared = tosa::MulOp::create(rewriter, loc, wideTy, wide, wide,
                                        createZeroMulShift(rewriter, loc));
    auto reducedTy = keepdimsReduceType(wideTy, axis);
    Value sum = tosa::ReduceSumOp::create(rewriter, loc, reducedTy, squared,
                                          rewriter.getI32IntegerAttr(axis));
    Value rsqrt = tosa::RsqrtOp::create(rewriter, loc, reducedTy, sum);
    Value norm = tosa::ReciprocalOp::create(rewriter, loc, reducedTy, rsqrt);
    if (widen)
      norm = tosa::CastOp::create(rewriter, loc,
                                  keepdimsReduceType(dataType, axis), norm);
    replaceWithTosaReduce(op, norm, resultType, keepdims, rewriter);
    return success();
  }
};

// rocMLIR's rock-tosa-to-elementwise lowers these tosa.custom names; a plain
// tosa.cast float->int is illegal there.
constexpr StringLiteral kRockCustomOpDomain = "rocmlir";
constexpr StringLiteral kRockUnsignedCast = "unsigned_cast";
constexpr StringLiteral kRockFpToIntCast = "fp_to_int_cast";

Value emitTosaCast(ConversionPatternRewriter &rewriter, Location loc,
                   Value input, Type resElemType) {
  auto inType = cast<RankedTensorType>(input.getType());
  Type inElem = inType.getElementType();
  auto outType = RankedTensorType::get(inType.getShape(), resElemType);
  if (inElem == resElemType)
    return input;
  if (inElem.isUnsignedInteger() || resElemType.isUnsignedInteger()) {
    return tosa::CustomOp::create(rewriter, loc, outType, kRockUnsignedCast,
                                  kRockCustomOpDomain, "", input)
        .getResult(0);
  }
  if (isa<FloatType>(inElem) && isa<IntegerType>(resElemType)) {
    return tosa::CustomOp::create(rewriter, loc, outType, kRockFpToIntCast,
                                  kRockCustomOpDomain, "", input)
        .getResult(0);
  }
  return tosa::CastOp::create(rewriter, loc, outType, input);
}

Value emitTosaMul(ConversionPatternRewriter &rewriter, Location loc, Value lhs,
                  Value rhs, RankedTensorType resultType) {
  return tosa::MulOp::create(rewriter, loc, resultType, lhs, rhs,
                             createZeroMulShift(rewriter, loc));
}

// ONNX QDQ scale/ZP is a scalar, a 1-D per-axis vector, or already ranked
// like the data. TOSA only broadcasts size-1 dims at matching rank, so a
// 1-D vector on `axis` has to be reshaped to [1, ..., C, ..., 1] first.
LogicalResult reshapeQdqParam(ConversionPatternRewriter &rewriter, Location loc,
                              Value &param, RankedTensorType dataType,
                              int64_t axis, Operation *op) {
  auto paramType = dyn_cast<RankedTensorType>(param.getType());
  if (!paramType || !paramType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "qdq param must be a static tensor");

  int64_t rank = dataType.getRank();
  if (rank == 0) {
    // TOSA elementwise ops need matching rank. ONNX per-tensor scale/ZP is
    // sometimes tensor<1xT>; fold that to a rank-0 scalar.
    if (paramType.getRank() == 1 && paramType.getDimSize(0) == 1) {
      auto scalarType = RankedTensorType::get({}, paramType.getElementType());
      Value shape = tosa::getTosaConstShape(rewriter, loc, ArrayRef<int64_t>{});
      param = tosa::ReshapeOp::create(rewriter, loc, scalarType, param, shape);
      return success();
    }
    if (paramType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "rank-0 qdq expects a scalar");
    return success();
  }

  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return rewriter.notifyMatchFailure(op, "qdq axis out of range");

  if (paramType.getRank() == rank) {
    auto asDataRank =
        RankedTensorType::get(dataType.getShape(), paramType.getElementType());
    if (!isTosaBroadcastableShape(paramType, asDataRank))
      return rewriter.notifyMatchFailure(op, "qdq param not broadcastable");
    return success();
  }

  SmallVector<int64_t> targetShape(rank, 1);
  if (paramType.getRank() == 0) {
    // per-tensor scalar: all-ones shape of the data rank
  } else if (paramType.getRank() == 1) {
    int64_t n = paramType.getDimSize(0);
    int64_t axisExtent = dataType.getDimSize(axis);
    if (n != 1 && n != axisExtent)
      return rewriter.notifyMatchFailure(
          op, "1-d qdq param length must match the quantized axis");
    targetShape[axis] = n;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported qdq param rank");
  }

  auto newType = RankedTensorType::get(targetShape, paramType.getElementType());
  if (paramType == newType)
    return success();
  Value shape = tosa::getTosaConstShape(rewriter, loc, targetShape);
  param = tosa::ReshapeOp::create(rewriter, loc, newType, param, shape);
  return success();
}

LogicalResult matchQdqCommon(Operation *op, Value input, Value scale,
                             Value zeroPoint, int64_t axis, int64_t blockSize,
                             ConversionPatternRewriter &rewriter,
                             RankedTensorType &resultType, Value &inputOut,
                             Value &scaleOut, Value &zpOut) {
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");
  resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked input");
  if (inputType.getShape() != resultType.getShape())
    return rewriter.notifyMatchFailure(op, "qdq cannot change the shape");
  if (blockSize != 0)
    return rewriter.notifyMatchFailure(op, "blocked qdq is not a tosa mul");
  if (op->hasAttr("packed_int4"))
    return rewriter.notifyMatchFailure(op, "packed_int4 is not tosa");

  Location loc = op->getLoc();
  scaleOut = scale;
  if (failed(reshapeQdqParam(rewriter, loc, scaleOut, inputType, axis, op)))
    return failure();
  zpOut = zeroPoint;
  if (zpOut &&
      failed(reshapeQdqParam(rewriter, loc, zpOut, inputType, axis, op)))
    return failure();
  inputOut = input;
  return success();
}

// The i'th element of an ONNX Range, start + i * delta, evaluated in the
// result element type.
Value emitRangeConst(ConversionPatternRewriter &rewriter, Location loc,
                     RankedTensorType type, DenseElementsAttr start,
                     DenseElementsAttr delta) {
  int64_t length = type.getDimSize(0);
  DenseElementsAttr values;
  if (auto floatType = dyn_cast<FloatType>(type.getElementType())) {
    APFloat first = start.getSplatValue<APFloat>();
    APFloat step = delta.getSplatValue<APFloat>();
    SmallVector<APFloat> elements;
    for (int64_t i : llvm::seq<int64_t>(length)) {
      APFloat index(floatType.getFloatSemantics());
      index.convertFromAPInt(APInt(64, i), /*IsSigned=*/true,
                             APFloat::rmNearestTiesToEven);
      elements.push_back(first + step * index);
    }
    values = DenseElementsAttr::get(type, elements);
  } else {
    auto intType = cast<IntegerType>(type.getElementType());
    unsigned width = intType.getWidth();
    unsigned mathWidth = std::max(64u, width + 1);
    APInt first = start.getSplatValue<APInt>().sextOrTrunc(mathWidth);
    APInt step = delta.getSplatValue<APInt>().sextOrTrunc(mathWidth);
    SmallVector<APInt> elements;
    for (int64_t i : llvm::seq<int64_t>(length))
      elements.push_back(
          (first + step * APInt(mathWidth, i, /*isSigned=*/true)).trunc(width));
    values = DenseElementsAttr::get(type, elements);
  }
  return tosa::ConstOp::create(rewriter, loc, type, values);
}

// [0, 1, ..., length-1] in the range's element type.
Value emitIotaConst(ConversionPatternRewriter &rewriter, Location loc,
                    RankedTensorType type) {
  int64_t length = type.getDimSize(0);
  DenseElementsAttr values;
  if (auto floatType = dyn_cast<FloatType>(type.getElementType())) {
    SmallVector<APFloat> elements;
    for (int64_t i : llvm::seq<int64_t>(length)) {
      APFloat index(floatType.getFloatSemantics());
      index.convertFromAPInt(APInt(64, i), /*IsSigned=*/true,
                             APFloat::rmNearestTiesToEven);
      elements.push_back(index);
    }
    values = DenseElementsAttr::get(type, elements);
  } else {
    unsigned width = type.getElementType().getIntOrFloatBitWidth();
    SmallVector<APInt> elements;
    for (int64_t i : llvm::seq<int64_t>(length))
      elements.push_back(APInt(width, i));
    values = DenseElementsAttr::get(type, elements);
  }
  return tosa::ConstOp::create(rewriter, loc, type, values);
}

// ONNX Range has no TOSA counterpart -- TOSA has no iota or arange. The
// sequence is affine in the index (output[i] = start + i * delta), so a range
// whose length is statically known is start + delta * iota, and constant
// bounds collapse that to a single tosa.const. The limit operand is dropped:
// the trip count it determines is already the result extent.
//
// A dynamic length has no TOSA spelling, so it is rejected rather than lowered
// to invalid TOSA.
struct RangeConverter final : public OpConversionPattern<RangeOp> {
  using OpConversionPattern<RangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || resultType.getRank() != 1 ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static 1-D result");
    // ONNX admits an empty Range, but a TOSA number tensor cannot be empty.
    if (resultType.getDimSize(0) <= 0)
      return rewriter.notifyMatchFailure(op, "empty range is not a TOSA value");

    Type elemType = resultType.getElementType();
    if (!isa<FloatType, IntegerType>(elemType) || elemType.isUnsignedInteger())
      return rewriter.notifyMatchFailure(op, "unsupported range element type");

    Location loc = op.getLoc();
    DenseElementsAttr startAttr, deltaAttr;
    if (matchPattern(adaptor.getStart(), m_Constant(&startAttr)) &&
        matchPattern(adaptor.getDelta(), m_Constant(&deltaAttr)) &&
        startAttr.isSplat() && deltaAttr.isSplat()) {
      rewriter.replaceOp(
          op, emitRangeConst(rewriter, loc, resultType, startAttr, deltaAttr));
      return success();
    }

    // Runtime bounds: scale the iota instead of folding it. ONNX Range bounds
    // are scalars, so reshaping them to rank 1 is all TOSA broadcasting needs.
    for (Value bound : {adaptor.getStart(), adaptor.getDelta()}) {
      auto boundType = dyn_cast<RankedTensorType>(bound.getType());
      if (!boundType || boundType.getNumElements() != 1 ||
          boundType.getElementType() != elemType)
        return rewriter.notifyMatchFailure(op, "expected scalar range bounds");
    }
    Value start = reshapeTo(adaptor.getStart(), {1}, rewriter);
    Value delta = reshapeTo(adaptor.getDelta(), {1}, rewriter);
    Value scaled =
        emitTosaMul(rewriter, loc, emitIotaConst(rewriter, loc, resultType),
                    delta, resultType);
    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType, scaled, start);
    return success();
  }
};

// hip.cast is 1-1 with tosa.cast when both endpoints are signed or float.
// Float->int and any unsigned endpoint go through rocMLIR tosa.custom, because
// rock-tosa-to-elementwise rejects tosa.cast float->int.
//
// ONNX CastLike shares this path: simplify-onnx rewrites it to a plain
// onnx.Cast (the target dtype is carried by the result type, and the type
// donor is never read), which convert-onnx-to-hip lowers to hip.cast.
struct CastConverter final : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked input");
    if (inputType.getShape() != resultType.getShape())
      return rewriter.notifyMatchFailure(op, "cast cannot change the shape");

    Type inElem = inputType.getElementType();
    Type outElem = resultType.getElementType();
    if (!isa<FloatType, IntegerType>(inElem) ||
        !isa<FloatType, IntegerType>(outElem))
      return rewriter.notifyMatchFailure(op, "unsupported cast element type");

    rewriter.replaceOp(
        op, emitTosaCast(rewriter, op.getLoc(), adaptor.getInput(), outElem));
    return success();
  }
};

// y = (x - zp) * scale.
struct DequantizeLinearConverter final
    : public OpConversionPattern<DequantizeLinearOp> {
  using OpConversionPattern<DequantizeLinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DequantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType;
    Value input, scale, zp;
    if (failed(matchQdqCommon(op, adaptor.getInput(), adaptor.getScale(),
                              adaptor.getZeroPoint(), op.getAxis(),
                              op.getBlockSize(), rewriter, resultType, input,
                              scale, zp)))
      return failure();
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "dequant result must be float");
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(op, "scale type must match result");

    Location loc = op.getLoc();
    Value shifted =
        emitTosaCast(rewriter, loc, input, resultType.getElementType());
    if (zp) {
      Value zpCast =
          emitTosaCast(rewriter, loc, zp, resultType.getElementType());
      shifted = tosa::SubOp::create(rewriter, loc, resultType, shifted, zpCast);
    }
    rewriter.replaceOp(op,
                       emitTosaMul(rewriter, loc, shifted, scale, resultType));
    return success();
  }
};

// y = saturate(x / scale + zp).
struct QuantizeLinearConverter final
    : public OpConversionPattern<QuantizeLinearOp> {
  using OpConversionPattern<QuantizeLinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getSaturate() == 0)
      return rewriter.notifyMatchFailure(op,
                                         "saturate=0 wrap is not a tosa.clamp");

    RankedTensorType resultType;
    Value input, scale, zp;
    if (failed(matchQdqCommon(op, adaptor.getInput(), adaptor.getScale(),
                              adaptor.getZeroPoint(), op.getAxis(),
                              op.getBlockSize(), rewriter, resultType, input,
                              scale, zp)))
      return failure();
    auto inputType = cast<RankedTensorType>(input.getType());
    Type inElem = inputType.getElementType();
    Type outElem = resultType.getElementType();
    if (!isa<FloatType>(inElem) || !isa<IntegerType>(outElem))
      return rewriter.notifyMatchFailure(
          op, "quantize expects float input and integer output");
    auto scaleType = cast<RankedTensorType>(scale.getType());
    if (scaleType.getElementType() != inElem)
      return rewriter.notifyMatchFailure(op, "scale type must match input");

    Location loc = op.getLoc();
    auto inv = tosa::ReciprocalOp::create(rewriter, loc, scaleType, scale);
    Value scaled = emitTosaMul(rewriter, loc, input, inv, inputType);

    if (!zp) {
      rewriter.replaceOp(op, emitTosaCast(rewriter, loc, scaled, outElem));
      return success();
    }

    Type biasElem = rewriter.getI32Type();
    auto i32Type = RankedTensorType::get(resultType.getShape(), biasElem);
    Value asI32 = emitTosaCast(rewriter, loc, scaled, biasElem);
    Value zpI32 = emitTosaCast(rewriter, loc, zp, biasElem);
    Value biased = tosa::AddOp::create(rewriter, loc, i32Type, asI32, zpI32);

    unsigned width = outElem.getIntOrFloatBitWidth();
    APInt minI = outElem.isUnsignedInteger() ? APInt::getMinValue(width)
                                             : APInt::getSignedMinValue(width);
    APInt maxI = outElem.isUnsignedInteger() ? APInt::getMaxValue(width)
                                             : APInt::getSignedMaxValue(width);
    int64_t minValI =
        outElem.isUnsignedInteger() ? minI.getZExtValue() : minI.getSExtValue();
    int64_t maxValI =
        outElem.isUnsignedInteger() ? maxI.getZExtValue() : maxI.getSExtValue();
    Attribute minVal = rewriter.getIntegerAttr(biasElem, minValI);
    Attribute maxVal = rewriter.getIntegerAttr(biasElem, maxValI);
    Value clamped =
        tosa::ClampOp::create(rewriter, loc, i32Type, biased, minVal, maxVal,
                              tosa::NanPropagationMode::PROPAGATE);
    rewriter.replaceOp(op, emitTosaCast(rewriter, loc, clamped, outElem));
    return success();
  }
};

int64_t normalizeNormAxis(int64_t axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  return axis;
}

// Keepdims mean over [firstAxis, rank): one TOSA mean per axis.
FailureOr<Value> emitReduceMeanFromAxis(Value input, int64_t firstAxis,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc, Operation *op) {
  auto ty = dyn_cast<RankedTensorType>(input.getType());
  if (!ty || !ty.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "norm input must be static");
  int64_t rank = ty.getRank();
  if (firstAxis < 0 || firstAxis >= rank)
    return rewriter.notifyMatchFailure(op, "norm axis out of range");
  Value reduced = input;
  for (int64_t a : llvm::seq(firstAxis, rank)) {
    FailureOr<Value> next = emitTosaKeepdimsReduceMean(
        reduced, static_cast<int32_t>(a), rewriter, loc, op);
    if (failed(next))
      return failure();
    reduced = *next;
  }
  return reduced;
}

LogicalResult matchSuffixNormParam(Value &param, RankedTensorType dataTy,
                                   int64_t firstAxis,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, Operation *op) {
  auto paramType = dyn_cast<RankedTensorType>(param.getType());
  if (!paramType || !paramType.hasStaticShape())
    return rewriter.notifyMatchFailure(op,
                                       "norm param must be a static tensor");
  int64_t rank = dataTy.getRank();
  if (firstAxis < 0 || firstAxis >= rank)
    return rewriter.notifyMatchFailure(op, "norm axis out of range");

  if (paramType.getRank() == rank) {
    auto asDataRank =
        RankedTensorType::get(dataTy.getShape(), paramType.getElementType());
    if (!isTosaBroadcastableShape(paramType, asDataRank))
      return rewriter.notifyMatchFailure(op, "norm param not broadcastable");
    return success();
  }

  ArrayRef<int64_t> suffix = dataTy.getShape().drop_front(firstAxis);
  int64_t suffixNumel = 1;
  for (int64_t d : suffix)
    suffixNumel *= d;

  SmallVector<int64_t> targetShape(rank, 1);
  if (paramType.getRank() == 0 ||
      (paramType.getRank() == 1 && paramType.getDimSize(0) == 1)) {
    // scalar / size-1 vector: all-ones of the data rank
  } else if (paramType.getRank() == static_cast<int64_t>(suffix.size()) &&
             paramType.getShape() == suffix) {
    for (int64_t i : llvm::seq<int64_t>(0, suffix.size()))
      targetShape[firstAxis + i] = suffix[i];
  } else if (paramType.getRank() == 1 &&
             paramType.getDimSize(0) == suffixNumel) {
    for (int64_t i : llvm::seq<int64_t>(0, suffix.size()))
      targetShape[firstAxis + i] = suffix[i];
  } else {
    return rewriter.notifyMatchFailure(
        op, "norm param must match the normalized suffix (or flatten it)");
  }

  auto newType = RankedTensorType::get(targetShape, paramType.getElementType());
  if (paramType == newType)
    return success();
  Value shape = tosa::getTosaConstShape(rewriter, loc, targetShape);
  param = tosa::ReshapeOp::create(rewriter, loc, newType, param, shape);
  return success();
}

FailureOr<Value> emitAffineScaleBias(Value normalized, Value scale, Value bias,
                                     RankedTensorType outTy, int64_t axis,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, Operation *op,
                                     bool perAxis = false) {
  auto matchParam = [&](Value &p) {
    return perAxis ? reshapeQdqParam(rewriter, loc, p, outTy, axis, op)
                   : matchSuffixNormParam(p, outTy, axis, rewriter, loc, op);
  };
  Value s = scale;
  if (failed(matchParam(s)))
    return failure();
  if (failed(tosa::EqualizeRanks(rewriter, loc, normalized, s)))
    return rewriter.notifyMatchFailure(op, "norm scale not broadcastable");
  Value y = emitTosaMul(rewriter, loc, normalized, s, outTy);
  if (!bias)
    return y;
  Value b = bias;
  if (failed(matchParam(b)))
    return failure();
  if (failed(tosa::EqualizeRanks(rewriter, loc, y, b)))
    return rewriter.notifyMatchFailure(op, "norm bias not broadcastable");
  return tosa::AddOp::create(rewriter, loc, outTy, y, b).getResult();
}

// RMS: y = x * rsqrt(mean(x^2) + eps) * scale. Optional FP32 stash for the
// stats, matching hip.rms_norm / SimplifiedLayerNormalization /
// RMSNormalization.
FailureOr<Value> emitRmsNorm(Value input, Value scale, int64_t axis,
                             float epsilon, bool stash,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Operation *op) {
  auto inTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inTy || !inTy.hasStaticShape() || !isa<FloatType>(inTy.getElementType()))
    return rewriter.notifyMatchFailure(op, "RMS input must be a static float");
  int64_t rank = inTy.getRank();
  axis = normalizeNormAxis(axis, rank);
  if (axis < 0 || axis >= rank)
    return rewriter.notifyMatchFailure(op, "RMS axis out of range");

  Type origElem = inTy.getElementType();
  Type workElem =
      (stash && !origElem.isF32()) ? rewriter.getF32Type() : origElem;
  Value x = emitTosaCast(rewriter, loc, input, workElem);
  auto xTy = cast<RankedTensorType>(x.getType());
  Value xSq = emitTosaMul(rewriter, loc, x, x, xTy);
  FailureOr<Value> meanOr =
      emitReduceMeanFromAxis(xSq, axis, rewriter, loc, op);
  if (failed(meanOr))
    return failure();
  Value mean = *meanOr;
  auto meanTy = cast<RankedTensorType>(mean.getType());
  Value eps = createSplatFloat(rewriter, loc, meanTy, epsilon);
  Value varEps = tosa::AddOp::create(rewriter, loc, meanTy, mean, eps);
  Value rrms = tosa::RsqrtOp::create(rewriter, loc, meanTy, varEps);
  Value n = emitTosaMul(rewriter, loc, x, rrms, xTy);
  n = emitTosaCast(rewriter, loc, n, origElem);
  return emitAffineScaleBias(n, scale, Value(), inTy, axis, rewriter, loc, op);
}

// (x - mean) * rsqrt(var + eps) over [firstAxis, rank). Optional stash.
FailureOr<Value> emitLayerNormCore(Value input, int64_t firstAxis,
                                   float epsilon, bool stash,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc, Operation *op, Value &meanOut,
                                   Value &invStdOut) {
  auto inTy = dyn_cast<RankedTensorType>(input.getType());
  if (!inTy || !inTy.hasStaticShape() || !isa<FloatType>(inTy.getElementType()))
    return rewriter.notifyMatchFailure(op, "LN input must be a static float");
  Type origElem = inTy.getElementType();
  Type workElem =
      (stash && !origElem.isF32()) ? rewriter.getF32Type() : origElem;
  Value x = emitTosaCast(rewriter, loc, input, workElem);
  auto xTy = cast<RankedTensorType>(x.getType());
  FailureOr<Value> meanOr =
      emitReduceMeanFromAxis(x, firstAxis, rewriter, loc, op);
  if (failed(meanOr))
    return failure();
  meanOut = *meanOr;
  Value delta = tosa::SubOp::create(rewriter, loc, xTy, x, meanOut);
  Value sq = emitTosaMul(rewriter, loc, delta, delta, xTy);
  FailureOr<Value> varOr =
      emitReduceMeanFromAxis(sq, firstAxis, rewriter, loc, op);
  if (failed(varOr))
    return failure();
  auto varTy = cast<RankedTensorType>((*varOr).getType());
  Value eps = createSplatFloat(rewriter, loc, varTy, epsilon);
  Value varEps = tosa::AddOp::create(rewriter, loc, varTy, *varOr, eps);
  invStdOut = tosa::RsqrtOp::create(rewriter, loc, varTy, varEps);
  Value n = emitTosaMul(rewriter, loc, delta, invStdOut, xTy);
  return emitTosaCast(rewriter, loc, n, origElem);
}

struct RmsNormConverter final : public OpConversionPattern<RmsNormOp> {
  using OpConversionPattern<RmsNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultTy || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getInput().getType() != resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "RMS input/result types must match");
    FailureOr<Value> y =
        emitRmsNorm(adaptor.getInput(), adaptor.getScale(), op.getAxis(),
                    op.getEpsilon().convertToFloat(), op.getStashType() != 0,
                    rewriter, op.getLoc(), op);
    if (failed(y))
      return failure();
    rewriter.replaceOp(op, *y);
    return success();
  }
};

struct LayerNormConverter final : public OpConversionPattern<LayerNormOp> {
  using OpConversionPattern<LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() < 1 || op.getNumResults() > 3)
      return rewriter.notifyMatchFailure(op, "expected 1-3 tensor results");
    auto yTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!yTy || !yTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static Y tensor");
    if (adaptor.getInput().getType() != yTy)
      return rewriter.notifyMatchFailure(op, "LN input/Y types must match");
    auto inTy = cast<RankedTensorType>(adaptor.getInput().getType());
    int64_t axis = normalizeNormAxis(op.getAxis(), inTy.getRank());
    Value mean, invStd;
    FailureOr<Value> n = emitLayerNormCore(
        adaptor.getInput(), axis, op.getEpsilon().convertToFloat(),
        op.getStashType() != 0, rewriter, op.getLoc(), op, mean, invStd);
    if (failed(n))
      return failure();
    FailureOr<Value> y =
        emitAffineScaleBias(*n, adaptor.getScale(), adaptor.getBias(), yTy,
                            axis, rewriter, op.getLoc(), op);
    if (failed(y))
      return failure();
    SmallVector<Value> results = {*y};
    if (op.getNumResults() >= 2) {
      auto meanTy = dyn_cast<RankedTensorType>(op.getResult(1).getType());
      if (!meanTy || !meanTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "mean must be a static tensor");
      Value m =
          emitTosaCast(rewriter, op.getLoc(), mean, meanTy.getElementType());
      if (cast<RankedTensorType>(m.getType()).getShape() != meanTy.getShape())
        m = reshapeTo(m, meanTy.getShape(), rewriter);
      results.push_back(m);
    }
    if (op.getNumResults() == 3) {
      auto isTy = dyn_cast<RankedTensorType>(op.getResult(2).getType());
      if (!isTy || !isTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op,
                                           "inv_std must be a static tensor");
      Value isv =
          emitTosaCast(rewriter, op.getLoc(), invStd, isTy.getElementType());
      if (cast<RankedTensorType>(isv.getType()).getShape() != isTy.getShape())
        isv = reshapeTo(isv, isTy.getShape(), rewriter);
      results.push_back(isv);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct InstanceNormConverter final
    : public OpConversionPattern<InstanceNormOp> {
  using OpConversionPattern<InstanceNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    auto yTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!yTy || !yTy.hasStaticShape() || yTy.getRank() < 3)
      return rewriter.notifyMatchFailure(
          op, "instance_norm expects a static rank >= 3 tensor");
    if (adaptor.getInput().getType() != yTy)
      return rewriter.notifyMatchFailure(op, "IN input/Y types must match");
    Value mean, invStd;
    FailureOr<Value> n = emitLayerNormCore(
        adaptor.getInput(), /*firstAxis=*/2, op.getEpsilon().convertToFloat(),
        /*stash=*/true, rewriter, op.getLoc(), op, mean, invStd);
    if (failed(n))
      return failure();
    FailureOr<Value> y =
        emitAffineScaleBias(*n, adaptor.getScale(), adaptor.getBias(), yTy,
                            /*axis=*/1, rewriter, op.getLoc(), op,
                            /*perAxis=*/true);
    if (failed(y))
      return failure();
    rewriter.replaceOp(op, *y);
    return success();
  }
};

struct SkipRmsNormConverter final : public OpConversionPattern<SkipRmsNormOp> {
  using OpConversionPattern<SkipRmsNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SkipRmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() < 1 || op.getNumResults() > 2)
      return rewriter.notifyMatchFailure(op, "expected 1-2 tensor results");
    auto yTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!yTy || !yTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static Y tensor");
    Value x = adaptor.getInput();
    Value skip = adaptor.getSkip();
    Location loc = op.getLoc();
    if (failed(tosa::EqualizeRanks(rewriter, loc, x, skip)))
      return rewriter.notifyMatchFailure(op, "skip not broadcastable");
    auto sumTy = dyn_cast<RankedTensorType>(x.getType());
    if (!sumTy || sumTy != yTy)
      return rewriter.notifyMatchFailure(op, "skip-sum type must match Y");
    Value sum = tosa::AddOp::create(rewriter, loc, yTy, x, skip);
    if (Value bias = adaptor.getBias()) {
      Value b = bias;
      if (failed(matchSuffixNormParam(b, yTy, yTy.getRank() - 1, rewriter, loc,
                                      op)))
        return failure();
      if (failed(tosa::EqualizeRanks(rewriter, loc, sum, b)))
        return rewriter.notifyMatchFailure(op, "skip bias not broadcastable");
      sum = tosa::AddOp::create(rewriter, loc, yTy, sum, b);
    }
    FailureOr<Value> y = emitRmsNorm(sum, adaptor.getGamma(), /*axis=*/-1,
                                     op.getEpsilon().convertToFloat(),
                                     /*stash=*/true, rewriter, loc, op);
    if (failed(y))
      return failure();
    SmallVector<Value> results = {*y};
    if (op.getNumResults() == 2) {
      auto skipTy = dyn_cast<RankedTensorType>(op.getResult(1).getType());
      if (!skipTy || skipTy != yTy)
        return rewriter.notifyMatchFailure(op,
                                           "input_skip_bias_sum must match Y");
      results.push_back(sum);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

static bool isTosaExpressibleSlice(tensor::ExtractSliceOp op) {
  auto sourceType = dyn_cast<RankedTensorType>(op.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!sourceType || !sourceType.hasStaticShape() || !resultType ||
      !resultType.hasStaticShape())
    return false;
  if (resultType.getRank() != sourceType.getRank())
    return false;
  for (OpFoldResult stride : op.getMixedStrides())
    if (getConstantIntValue(stride) != std::optional<int64_t>(1))
      return false;
  for (OpFoldResult offset : op.getMixedOffsets())
    if (!getConstantIntValue(offset))
      return false;
  return true;
}

struct ExtractSliceConverter final
    : public OpConversionPattern<tensor::ExtractSliceOp> {
  using OpConversionPattern<tensor::ExtractSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleSlice(op))
      return rewriter.notifyMatchFailure(
          op, "expected a static, unstrided, same-rank extract");

    SmallVector<int64_t> starts;
    for (OpFoldResult offset : op.getMixedOffsets())
      starts.push_back(*getConstantIntValue(offset));

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tosa::SliceOp>(
        op, resultType, adaptor.getSource(),
        createConstShape(rewriter, op.getLoc(), starts),
        createConstShape(rewriter, op.getLoc(), resultType.getShape()));
    return success();
  }
};

struct StaticConcatMatch {
  RankedTensorType resultType;
  int64_t axis;
  SmallVector<Value> inputs;
  SmallVector<tensor::InsertSliceOp> inserts;
  tensor::EmptyOp init;
};

static std::optional<StaticConcatMatch>
matchStaticConcat(tensor::InsertSliceOp root) {
  // Only claim the last insertion. Earlier insertions remain legal until the
  // root rewrite replaces the complete chain.
  if (llvm::any_of(root->getUsers(), [&](Operation *user) {
        auto next = dyn_cast<tensor::InsertSliceOp>(user);
        return next && next.getDest() == root.getResult();
      }))
    return std::nullopt;

  auto resultType = dyn_cast<RankedTensorType>(root.getResult().getType());
  if (!resultType || !resultType.hasStaticShape() ||
      llvm::any_of(resultType.getShape(), [](int64_t dim) { return dim <= 0; }))
    return std::nullopt;

  SmallVector<tensor::InsertSliceOp> inserts;
  Value dest = root.getResult();
  while (auto insert = dest.getDefiningOp<tensor::InsertSliceOp>()) {
    if (insert.getResult().getType() != resultType)
      return std::nullopt;
    inserts.push_back(insert);
    dest = insert.getDest();
  }
  auto init = dest.getDefiningOp<tensor::EmptyOp>();
  if (!init || init.getType() != resultType || inserts.size() < 2)
    return std::nullopt;
  std::reverse(inserts.begin(), inserts.end());

  int64_t rank = resultType.getRank();
  for (int64_t axis = 0; axis < rank; ++axis) {
    int64_t expectedOffset = 0;
    SmallVector<Value> inputs;
    bool valid = true;
    for (tensor::InsertSliceOp insert : inserts) {
      auto sourceType =
          dyn_cast<RankedTensorType>(insert.getSource().getType());
      if (!sourceType || !sourceType.hasStaticShape() ||
          sourceType.getRank() != rank ||
          sourceType.getElementType() != resultType.getElementType()) {
        valid = false;
        break;
      }

      for (int64_t dim = 0; dim < rank; ++dim) {
        auto offset = getConstantIntValue(insert.getMixedOffsets()[dim]);
        auto size = getConstantIntValue(insert.getMixedSizes()[dim]);
        auto stride = getConstantIntValue(insert.getMixedStrides()[dim]);
        int64_t sourceDim = sourceType.getDimSize(dim);
        int64_t wantedOffset = dim == axis ? expectedOffset : 0;
        if (sourceDim <= 0 || offset != wantedOffset || size != sourceDim ||
            stride != 1 ||
            (dim != axis && sourceDim != resultType.getDimSize(dim))) {
          valid = false;
          break;
        }
      }
      if (!valid)
        break;
      expectedOffset += sourceType.getDimSize(axis);
      inputs.push_back(insert.getSource());
    }

    if (valid && expectedOffset == resultType.getDimSize(axis))
      return StaticConcatMatch{resultType, axis, std::move(inputs),
                               std::move(inserts), init};
  }
  return std::nullopt;
}

struct InsertSliceConcatConverter final
    : public OpConversionPattern<tensor::InsertSliceOp> {
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<StaticConcatMatch> match = matchStaticConcat(op);
    if (!match)
      return rewriter.notifyMatchFailure(
          op, "expected a static contiguous concat insertion chain");

    Value concat = tosa::ConcatOp::create(
        rewriter, op.getLoc(), match->resultType, match->inputs,
        rewriter.getI32IntegerAttr(match->axis));
    rewriter.replaceOp(op, concat);

    // Remove the now-dead decomposition from the root back to tensor.empty.
    for (size_t i = match->inserts.size() - 1; i-- > 0;)
      if (match->inserts[i]->use_empty())
        rewriter.eraseOp(match->inserts[i]);
    if (match->init->use_empty())
      rewriter.eraseOp(match->init);
    return success();
  }
};

// onnx.Shape / Size / Identity / ConstantOfShape are not 1-1 TOSA ops
// (except Identity → tosa.identity). After convert-onnx-to-hip they arrive
// as:
//   Identity          SSA forward, or a same-type tensor.cast
//   Shape (static)    arith.constant dims + tensor.from_elements
//   Size (static)     arith.constant rank-0 i64; hip.size is the dynamic path
//   ConstantOfShape   arith.constant splat, or tensor.splat
// Fold those residues to tosa.const / tosa.identity so a rock.kernel can
// absorb them. Runtime shape queries (tensor.dim, dynamic hip.size) stay
// unconverted: TOSA has no shape-of.
//
// TOSA number tensors cannot have a zero extent (`tensor<0xi64>`). Leave
// those as arith.constant / tensor.from_elements / tensor.splat; Reduce
// empty-axes identity uses exactly that type.
bool hasPositiveStaticExtents(RankedTensorType type) {
  return type && type.hasStaticShape() &&
         llvm::all_of(type.getShape(), [](int64_t d) { return d > 0; });
}

bool isTosaExpressibleTensorConst(arith::ConstantOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  return hasPositiveStaticExtents(type) &&
         isa<DenseElementsAttr>(op.getValue());
}

bool isTosaExpressibleFromElements(tensor::FromElementsOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!hasPositiveStaticExtents(type))
    return false;
  return llvm::all_of(op.getElements(),
                      [](Value v) { return matchPattern(v, m_Constant()); });
}

bool isTosaExpressibleSplat(tensor::SplatOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!hasPositiveStaticExtents(type) || !op.getDynamicSizes().empty())
    return false;
  return matchPattern(op.getInput(), m_Constant());
}

Value emitTosaConst(ConversionPatternRewriter &rewriter, Location loc,
                    RankedTensorType type, ElementsAttr values) {
  return tosa::ConstOp::create(rewriter, loc, type, values);
}

struct TensorConstConverter final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleTensorConst(op))
      return rewriter.notifyMatchFailure(op, "expected a static dense tensor");
    auto type = cast<RankedTensorType>(op.getType());
    rewriter.replaceOp(op, emitTosaConst(rewriter, op.getLoc(), type,
                                         cast<ElementsAttr>(op.getValue())));
    return success();
  }
};

struct FromElementsConverter final
    : public OpConversionPattern<tensor::FromElementsOp> {
  using OpConversionPattern<tensor::FromElementsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::FromElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleFromElements(op))
      return rewriter.notifyMatchFailure(
          op, "expected a static from_elements of constants");
    auto type = cast<RankedTensorType>(op.getType());
    SmallVector<Attribute> elems;
    elems.reserve(adaptor.getElements().size());
    for (Value v : adaptor.getElements()) {
      Attribute attr;
      if (!matchPattern(v, m_Constant(&attr)))
        return rewriter.notifyMatchFailure(op, "element is not a constant");
      elems.push_back(attr);
    }
    rewriter.replaceOp(op, emitTosaConst(rewriter, op.getLoc(), type,
                                         DenseElementsAttr::get(type, elems)));
    return success();
  }
};

struct SplatConverter final : public OpConversionPattern<tensor::SplatOp> {
  using OpConversionPattern<tensor::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleSplat(op))
      return rewriter.notifyMatchFailure(op,
                                         "expected a static constant splat");
    auto type = cast<RankedTensorType>(op.getType());
    Attribute attr;
    if (!matchPattern(adaptor.getInput(), m_Constant(&attr)))
      return rewriter.notifyMatchFailure(op, "splat input is not a constant");
    TypedAttr typed = dyn_cast<TypedAttr>(attr);
    if (!typed)
      return rewriter.notifyMatchFailure(op, "splat input is not a typed attr");
    rewriter.replaceOp(op, emitTosaConst(rewriter, op.getLoc(), type,
                                         DenseElementsAttr::get(type, typed)));
    return success();
  }
};

// hip.size is the dynamic OnnxToHip path. A static input still folds to
// tosa.const(prod(shape)), matching SizeToConstant. Dynamic dims cannot
// be read in TOSA.
struct SizeConverter final : public OpConversionPattern<SizeOp> {
  using OpConversionPattern<SizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getX().getType());
    if (!resultType || !resultType.hasStaticShape() || !inputType ||
        !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (resultType.getRank() != 0 || !resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must be tensor<i64>");

    int64_t n = 1;
    for (int64_t d : inputType.getShape()) {
      if (d < 0)
        return rewriter.notifyMatchFailure(op, "expected a static shape");
      n *= d;
    }
    auto attr = DenseElementsAttr::get(
        resultType, rewriter.getIntegerAttr(rewriter.getI64Type(), n));
    rewriter.replaceOp(op,
                       emitTosaConst(rewriter, op.getLoc(), resultType, attr));
    return success();
  }
};

Value transposePerm(Value input, ArrayRef<int32_t> perm,
                    ConversionPatternRewriter &rewriter, Location loc) {
  auto inTy = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> outShape;
  outShape.reserve(perm.size());
  for (int32_t p : perm)
    outShape.push_back(inTy.getDimSize(p));
  auto outTy = RankedTensorType::get(outShape, inTy.getElementType());
  return tosa::TransposeOp::create(rewriter, loc, outTy, input,
                                   rewriter.getDenseI32ArrayAttr(perm))
      .getResult();
}

Value sliceOffsetSize(Value input, ArrayRef<int64_t> starts,
                      ArrayRef<int64_t> sizes,
                      ConversionPatternRewriter &rewriter, Location loc) {
  auto outTy = RankedTensorType::get(
      sizes, cast<RankedTensorType>(input.getType()).getElementType());
  return tosa::SliceOp::create(rewriter, loc, outTy, input,
                               createConstShape(rewriter, loc, starts),
                               createConstShape(rewriter, loc, sizes))
      .getResult();
}

Value tileMultiples(Value input, ArrayRef<int64_t> multiples,
                    ArrayRef<int64_t> outShape,
                    ConversionPatternRewriter &rewriter, Location loc) {
  auto outTy = RankedTensorType::get(
      outShape, cast<RankedTensorType>(input.getType()).getElementType());
  return tosa::TileOp::create(rewriter, loc, outTy, input,
                              createConstShape(rewriter, loc, multiples))
      .getResult();
}

Value createSplatInt(ConversionPatternRewriter &rewriter, Location loc,
                     RankedTensorType type, int64_t value) {
  auto elemType = cast<IntegerType>(type.getElementType());
  return tosa::ConstOp::create(
      rewriter, loc, type,
      DenseElementsAttr::get(type, rewriter.getIntegerAttr(elemType, value)));
}

// TOSA has no round, and the obvious floor(x + 0.5) is the wrong rounding:
// hip.round is ONNX Round, which breaks ties to even, so it owes 2 on 2.5 and
// -4 on -4.5 where floor(x + 0.5) gives 3 and -4.
//
// tosa.cast from float to integer does round half to even, so a cast out and
// back would spell it in two ops, but it saturates at the integer range and
// would need a guard for the magnitudes that no longer fit -- which are exactly
// the values already integral, and so exactly the ones that need no rounding.
// The guard's threshold is per float type, and the tie rule would be inherited
// from whatever the backend's cast does rather than stated here. Building it
// out of tosa.floor avoids both: one element type throughout, and the rule
// written down.
//
//   f = floor(x)        the integer below x
//   d = x - f           the fraction, always in [0, 1)
// Step up when d > 0.5, stay when d < 0.5, and on the tie step up only when f
// is odd, which is the half-to-even rule. f is odd exactly when halving and
// doubling it fails to round-trip; halving is exact, so that test costs
// nothing.
//
// The infinities and NaN need no case of their own. floor leaves them be and
// d becomes inf - inf = NaN, so both comparisons are false and x falls through
// unrounded, which is what it should do.
//
// Before:
//   %y = hip.round(%ctx) ins(%x : tensor<4xf32>) outs(%i : tensor<4xf32>)
// After:
//   %f = tosa.floor %x
//   %d = tosa.sub %x, %f
//   %up = tosa.bitwise_or (tosa.greater %d, 0.5),
//                         (tosa.bitwise_and (tosa.equal %d, 0.5), f is odd)
//   %y = tosa.select %up, (tosa.ceil %x), %f
//
// The predicates are combined with the bitwise ops rather than the logical
// ones, which coincide on i1. rocMLIR's RockTosaToElementwise has patterns for
// tosa.bitwise_and/_or/_xor and none for any tosa.logical_*, and it marks every
// surviving tosa op illegal, so the logical spelling converts cleanly here and
// then kills the kernel it was meant to enable. hip.and/or/not are spelled the
// same way for the same reason.
struct RoundConverter final : public OpConversionPattern<RoundOp> {
  using OpConversionPattern<RoundOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RoundOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    Value x = adaptor.getX();
    if (x.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    // ONNX Round is float-only, so an integer here is unreachable from a valid
    // model. f64 is reachable -- ONNX has a double tensor type -- but TOSA has
    // no f64 tensor type, so the expansion would build tosa.floor on an element
    // type TOSA cannot represent. Both are named rather than left to fail
    // later, for the reason GemmConverter names its own f64 rejection.
    Type elementType = resultType.getElementType();
    if (!elementType.isF32() && !elementType.isF16() && !elementType.isBF16())
      return op->emitError("hip.round has no TOSA spelling for element type ")
             << elementType << ": the expansion needs f32, f16, or bf16";

    Location loc = op.getLoc();
    auto predType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    Value half = createSplatFloat(rewriter, loc, resultType, 0.5);
    Value two = createSplatFloat(rewriter, loc, resultType, 2.0);
    Value shift = createZeroMulShift(rewriter, loc);

    Value floored = tosa::FloorOp::create(rewriter, loc, resultType, x);
    Value fraction = tosa::SubOp::create(rewriter, loc, resultType, x, floored);
    Value above =
        tosa::GreaterOp::create(rewriter, loc, predType, fraction, half);
    Value tie = tosa::EqualOp::create(rewriter, loc, predType, fraction, half);

    Value halved =
        tosa::MulOp::create(rewriter, loc, resultType, floored, half, shift);
    Value rounded = tosa::MulOp::create(
        rewriter, loc, resultType,
        tosa::FloorOp::create(rewriter, loc, resultType, halved), two, shift);
    // Halving and doubling drops exactly the low bit, so an odd floor comes
    // back one smaller and an even one comes back unchanged. Asking which is
    // the greater reads that off without a negation.
    Value isOdd =
        tosa::GreaterOp::create(rewriter, loc, predType, floored, rounded);

    Value stepUp = tosa::BitwiseOrOp::create(
        rewriter, loc, predType, above,
        tosa::BitwiseAndOp::create(rewriter, loc, predType, tie, isOdd));
    // The step-up value is tosa.ceil rather than floor + 1 so that a negative
    // input rounding to zero keeps its sign. nearbyintf, which the round
    // runtime uses, returns -0 on [-0.5, 0), and floor + 1 would return +0
    // there. The two agree everywhere the branch is taken: it is taken only
    // when the fraction is non-zero, and ceil is floor + 1 on every
    // non-integral value.
    Value next = tosa::CeilOp::create(rewriter, loc, resultType, x);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, stepUp, next,
                                                floored);
    return success();
  }
};

// TOSA has no modulo either, but for integers the remainder identity spells one
// out. tosa.intdiv truncates towards zero, so lhs - (lhs / rhs) * rhs is C's %,
// whose sign follows the dividend -- which is ONNX Mod's fmod = 1 exactly.
//
// The default fmod = 0 wants the sign to follow the divisor instead. The two
// agree except where the remainder and the divisor have opposite signs, and
// there they differ by one divisor, so adding it back converts one to the
// other. A zero remainder is already right under both rules and must be left
// alone, or a divide that came out exact would gain a spurious divisor.
//
// Floats are rejected rather than expanded. fmod needs the truncated quotient
// exactly, and reciprocal-then-multiply cannot supply it: an error of one ulp
// in lhs/rhs moves the truncation across an integer boundary and the result is
// then wrong by a whole divisor, not by an ulp. The quotient need not even be
// representable -- fmod(1e30, 3) asks for a truncation no f32 can hold -- which
// is why libm computes it by iterated reduction instead of a division, and why
// no fixed sequence of TOSA ops stands in for it.
static bool isTosaExpressibleModType(Type elementType) {
  return elementType.isSignlessInteger(32) || elementType.isSignlessInteger(64);
}

struct ModConverter final : public OpConversionPattern<hip::ModOp> {
  using OpConversionPattern<hip::ModOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::ModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");

    // Named here for the same reason hip.div names its own: the pass runs only
    // inside a rock.kernel, so a hip.mod left behind fails in rocMLIR instead,
    // later and as an op from a dialect it has never heard of.
    Type elementType = resultType.getElementType();
    if (isa<FloatType>(elementType))
      return op->emitError("hip.mod has no TOSA spelling for element type ")
             << elementType
             << ": fmod needs the exact truncated quotient, which a "
                "reciprocal and a multiply cannot give";
    if (!isTosaExpressibleModType(elementType))
      return op->emitError("hip.mod has no TOSA spelling for element type ")
             << elementType << ": tosa.intdiv takes signless i32 and i64 only";
    // Past here the element type is signless i32 or i64, where fmod = 1 is
    // outside the contract rather than merely unimplemented. Hip_ModOp
    // documents it as the floating-point rule and wrap_mod refuses it on an
    // integer data type, which is ONNX Mod-13's constraint -- opset 13 ties
    // fmod = 0 to the integer types and fmod = 1 to the float ones. Emitting
    // the truncated remainder here is the Mod-28 reading instead, and it would
    // answer inside a fused kernel what the runtime declines outside one, so
    // the same graph would give two different results depending on whether it
    // happened to be fused. Widening this belongs in the runtime first.
    if (op.getFmod() != 0)
      return op->emitError("hip.mod with fmod = 1 needs a floating-point "
                           "element type, got ")
             << elementType;

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (failed(tosa::EqualizeRanks(rewriter, loc, lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");
    if (!isTosaCompatibleOperand(lhs, resultType) ||
        !isTosaCompatibleOperand(rhs, resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    auto predType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    // Dividing the most negative value by -1 overflows, and tosa.intdiv becomes
    // arith.divsi and then an LLVM sdiv, where that pair is undefined behaviour
    // rather than a merely wrong value -- a later select cannot take it back.
    // Every remainder by -1 is zero under both fmod rules, and dividing by 1
    // instead produces exactly that (lhs - (lhs / 1) * 1), so substituting the
    // divisor keeps the result and leaves no overflowing division behind.
    Value minusOne = createSplatInt(rewriter, loc, resultType, -1);
    Value divisor = tosa::SelectOp::create(
        rewriter, loc, resultType,
        tosa::EqualOp::create(rewriter, loc, predType, rhs, minusOne),
        createSplatInt(rewriter, loc, resultType, 1), rhs);

    Value quotient =
        tosa::IntDivOp::create(rewriter, loc, resultType, lhs, divisor);
    Value product =
        tosa::MulOp::create(rewriter, loc, resultType, quotient, divisor,
                            createZeroMulShift(rewriter, loc));
    Value remainder =
        tosa::SubOp::create(rewriter, loc, resultType, lhs, product);

    Value zero = createSplatInt(rewriter, loc, resultType, 0);
    Value signsDiffer = tosa::BitwiseXorOp::create(
        rewriter, loc, predType,
        tosa::GreaterOp::create(rewriter, loc, predType, zero, remainder),
        tosa::GreaterOp::create(rewriter, loc, predType, zero, rhs));
    // x ^ true is !x, which is how the and/or/not lowerings spell negation for
    // the same reason: tosa.logical_not has no downstream pattern either.
    Value nonZero = tosa::BitwiseXorOp::create(
        rewriter, loc, predType,
        tosa::EqualOp::create(rewriter, loc, predType, remainder, zero),
        createSplatInt(rewriter, loc, predType, 1));
    Value adjust = tosa::BitwiseAndOp::create(rewriter, loc, predType,
                                              signsDiffer, nonZero);
    Value shifted =
        tosa::AddOp::create(rewriter, loc, resultType, remainder, rhs);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultType, adjust, shifted,
                                                remainder);
    return success();
  }
};

// hip.tile repeats each dimension, which is exactly tosa.tile. The one
// difference is where the repeat counts live: hip carries them as an operand
// (ONNX Tile takes `repeats` as an input), while TOSA wants a !tosa.shape, so
// the operand has to be constant to convert at all.
static bool isTosaExpressibleTile(hip::TileOp op) {
  if (op->getNumResults() != 1)
    return false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  if (!resultType || !resultType.hasStaticShape() || !inputType ||
      !inputType.hasStaticShape())
    return false;
  if (resultType.getElementType() != inputType.getElementType())
    return false;

  SmallVector<int64_t, 4> repeats;
  if (!extractConstantInts(op.getRepeats(), repeats))
    return false;
  if (static_cast<int64_t>(repeats.size()) != inputType.getRank() ||
      inputType.getRank() != resultType.getRank())
    return false;
  // A zero repeat empties the dimension. TOSA's multiples must be positive, so
  // that case has no spelling here rather than a wrong one.
  for (auto [repeat, in, out] :
       llvm::zip_equal(repeats, inputType.getShape(), resultType.getShape())) {
    if (repeat < 1 || in * repeat != out)
      return false;
  }
  return true;
}

struct TileConverter final : public OpConversionPattern<hip::TileOp> {
  using OpConversionPattern<hip::TileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::TileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleTile(op))
      return rewriter.notifyMatchFailure(op, "not a tosa-expressible tile");

    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    // Read the operand the legality predicate read, not the adaptor's: the
    // repeats only become a multiples attribute, never an operand of the op
    // being built, so there is nothing to gain from the remapped value and a
    // materialization standing in its place would make the extraction fail
    // here after the predicate had already claimed the op.
    SmallVector<int64_t, 4> repeats;
    if (!extractConstantInts(op.getRepeats(), repeats))
      return rewriter.notifyMatchFailure(op, "repeats must be constant");
    rewriter.replaceOp(op, tileMultiples(adaptor.getInput(), repeats,
                                         resultType.getShape(), rewriter,
                                         op.getLoc()));
    return success();
  }
};

// TOSA has no arctangent, and no identity reaches one from the transcendentals
// it does carry: the usual rewrites land on asin or on complex arithmetic, and
// tosa.table is an integer lookup that will not take a float tensor. So unlike
// round and mod, this expansion approximates. It is the Cephes atanf
// algorithm; the runtime calls the device atanf, so the two agree to within
// their error bounds rather than bit-for-bit.
//
// atan is odd, so the work happens on |x| and the sign goes back on at the end.
// No polynomial holds accuracy across [0, inf), so the domain is folded onto
// [0, tan(pi/8)] by the angle-difference identity, splitting at tan(pi/8) and
// tan(3pi/8):
//
//   ax > 2.4142   y0 = pi/2   r = -1/ax          atan(ax) = pi/2 - atan(1/ax)
//   ax > 0.4142   y0 = pi/4   r = (ax-1)/(ax+1)  atan(ax) = pi/4 + atan(r)
//   otherwise     y0 = 0      r = ax
//
// An odd minimax polynomial finishes the reduced argument: atan(r) is r plus
// r*z times a degree-3 polynomial in z = r*r. The three-way split is what keeps
// that degree down -- folding only at ax > 1 would need roughly twice the terms
// for the same error. Cephes bounds this at a few ulp in f32.
//
// TOSA has no control flow, so both reduced arguments are computed on every
// lane and tosa.select picks between them. An unselected lane can hold an
// infinity -- 1/ax is inf at ax = 0 -- but select discards it without doing
// arithmetic on it, so it cannot spread. The same dead lane is load-bearing at
// the other end: at ax = inf the reciprocal is 0, r is -0, and the result comes
// out exactly pi/2.
//
// f16 and bf16 are widened to f32 for the expansion and narrowed back. The
// runtime does the same -- its f16 kernel is __half2float, atanf, __float2half
// -- and evaluating a degree-9 polynomial in 11 mantissa bits would throw away
// most of what the minimax fit buys. f64 is named rather than expanded, the way
// RoundConverter names its own: ONNX has a double tensor type and TOSA has no
// f64 tensor type.
//
// The signed zero is put back by hand. atan(-0) is -0, but abs erases the sign
// and `0 > x` is false for -0, so the sign-restoring select would hand back +0.
// Selecting x itself wherever x == 0 returns it: the comparison holds for both
// zeros, and atan(+-0) is +-0.
//
// Before:
//   %y = hip.atan(%ctx) ins(%x : tensor<4xf32>) outs(%i : tensor<4xf32>)
// After:
//   %ax = tosa.abs %x
//   %r  = tosa.select (tosa.greater %ax, 2.4142), (-1/%ax),
//           (tosa.select (tosa.greater %ax, 0.4142), (%ax-1)/(%ax+1), %ax)
//   %y0 = tosa.select (tosa.greater %ax, 2.4142), pi/2,
//           (tosa.select (tosa.greater %ax, 0.4142), pi/4, 0)
//   %y  = %y0 + (poly(%r*%r) * %r*%r * %r + %r)
//   %y  = tosa.select (tosa.greater 0, %x), -%y, %y
//   %y  = tosa.select (tosa.equal %x, 0), %x, %y
struct AtanConverter final : public OpConversionPattern<AtanOp> {
  using OpConversionPattern<AtanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    Value x = adaptor.getX();
    if (x.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operand and result types must match exactly");
    Type elementType = resultType.getElementType();
    if (!elementType.isF32() && !elementType.isF16() && !elementType.isBF16())
      return op->emitError("hip.atan has no TOSA spelling for element type ")
             << elementType << ": the expansion needs f32, f16, or bf16";

    Location loc = op.getLoc();
    Type computeElem = rewriter.getF32Type();
    auto computeType =
        RankedTensorType::get(resultType.getShape(), computeElem);
    auto predType =
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    Value xc = emitTosaCast(rewriter, loc, x, computeElem);

    auto splat = [&](double v) -> Value {
      return createSplatFloat(rewriter, loc, computeType, v);
    };
    auto mul = [&](Value a, Value b) -> Value {
      return emitTosaMul(rewriter, loc, a, b, computeType);
    };
    auto add = [&](Value a, Value b) -> Value {
      return tosa::AddOp::create(rewriter, loc, computeType, a, b);
    };
    auto sub = [&](Value a, Value b) -> Value {
      return tosa::SubOp::create(rewriter, loc, computeType, a, b);
    };
    auto recip = [&](Value a) -> Value {
      return tosa::ReciprocalOp::create(rewriter, loc, computeType, a);
    };
    auto greater = [&](Value a, Value b) -> Value {
      return tosa::GreaterOp::create(rewriter, loc, predType, a, b);
    };
    auto select = [&](Value p, Value a, Value b) -> Value {
      return tosa::SelectOp::create(rewriter, loc, computeType, p, a, b);
    };

    // Each step is bound to a name rather than nested, so the order the ops
    // come out in is the order written here. C++ leaves the evaluation order of
    // call arguments unspecified, and nesting these would let it vary.
    Value zero = splat(0.0);
    Value one = splat(1.0);
    // tosa.negate carries zero-point operands, so the two negations here are
    // spelled as a multiply instead.
    Value minusOne = splat(-1.0);
    Value ax = tosa::AbsOp::create(rewriter, loc, computeType, xc);

    // tan(3*pi/8) and tan(pi/8), the two fold points.
    Value isHigh = greater(ax, splat(2.414213562373095));
    Value isMid = greater(ax, splat(0.4142135623730950));

    Value invAx = recip(ax);
    Value highArg = mul(invAx, minusOne);
    Value midNum = sub(ax, one);
    Value midDen = add(ax, one);
    Value invMidDen = recip(midDen);
    Value midArg = mul(midNum, invMidDen);

    Value midOrLow = select(isMid, midArg, ax);
    Value reduced = select(isHigh, highArg, midOrLow);
    Value midOffset = select(isMid, splat(0.7853981633974483), zero);
    Value offset = select(isHigh, splat(1.5707963267948966), midOffset);

    // Cephes minimax coefficients for atan on [0, tan(pi/8)], in Horner order.
    Value z = mul(reduced, reduced);
    Value poly = splat(8.05374449538e-2);
    for (double coeff :
         {-1.38776856032e-1, 1.99777106478e-1, -3.33329491539e-1}) {
      Value scaled = mul(poly, z);
      poly = add(scaled, splat(coeff));
    }
    Value polyZ = mul(poly, z);
    Value tail = mul(polyZ, reduced);
    Value series = add(tail, reduced);
    Value y = add(offset, series);

    Value isNegative = greater(zero, xc);
    Value negated = mul(y, minusOne);
    y = select(isNegative, negated, y);

    Value isZero = tosa::EqualOp::create(rewriter, loc, predType, xc, zero);
    y = select(isZero, xc, y);

    rewriter.replaceOp(op, emitTosaCast(rewriter, loc, y, elementType));
    return success();
  }
};

// hip.pad carries ONNX Pad: `pads` as an operand, an optional scalar fill, an
// optional `axes` subset, and a mode. tosa.pad is the constant-mode case of
// that and nothing else -- there is no TOSA reflect, edge or wrap -- so only
// mode="constant" converts.
//
// The two also disagree on layout. ONNX groups the pads as all the begins
// followed by all the ends, [x0_begin, x1_begin, .., x0_end, x1_end, ..],
// while TOSA interleaves them per dimension, [d0_lo, d0_hi, d1_lo, d1_hi, ..],
// so the operand is transposed into place rather than copied.
//
// Negative pads are excluded: ONNX added them as a crop in opset 18 and TOSA
// accepts them, but the result dimension then stops being inferable from the
// operands alone, which is what the shape check below relies on.
static bool matchTosaPad(hip::PadOp op, SmallVectorImpl<int64_t> &interleaved) {
  interleaved.clear();
  if (op->getNumResults() != 1)
    return false;
  if (op.getMode() != "constant")
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto dataType = dyn_cast<RankedTensorType>(op.getData().getType());
  if (!resultType || !resultType.hasStaticShape() || !dataType ||
      !dataType.hasStaticShape())
    return false;
  int64_t rank = dataType.getRank();
  if (rank < 1 || resultType.getRank() != rank)
    return false;

  SmallVector<int64_t, 8> pads;
  if (!extractConstantInts(op.getPads(), pads))
    return false;

  // Without `axes` the pads cover every dimension in order; with it they cover
  // only the listed ones and the rest stay unpadded. An `axes` that is present
  // but empty means every dimension as well, not none: wrap_pad selects the
  // per-axis layout on axes_host.empty(), and it fills axes_host only when the
  // operand is both present and non-empty, so an absent operand and a
  // zero-length one arrive there alike.
  SmallVector<int64_t, 4> axes;
  if (op.getAxes() && !extractConstantInts(op.getAxes(), axes))
    return false;
  if (axes.empty())
    for (int64_t i = 0; i < rank; ++i)
      axes.push_back(i);
  if (static_cast<int64_t>(pads.size()) !=
      2 * static_cast<int64_t>(axes.size()))
    return false;

  int64_t numAxes = axes.size();
  SmallVector<int64_t, 8> lo(rank, 0), hi(rank, 0);
  SmallVector<bool, 8> padded(rank, false);
  for (int64_t i = 0; i < numAxes; ++i) {
    int64_t axis = axes[i];
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return false;
    // A repeated axis would silently drop one of the two paddings. Reading
    // lo/hi back would miss the repeat when the earlier pad was (0, 0), so the
    // axes seen are tracked separately.
    if (padded[axis])
      return false;
    padded[axis] = true;
    if (pads[i] < 0 || pads[i + numAxes] < 0)
      return false;
    lo[axis] = pads[i];
    hi[axis] = pads[i + numAxes];
  }

  for (int64_t d = 0; d < rank; ++d) {
    if (dataType.getDimSize(d) + lo[d] + hi[d] != resultType.getDimSize(d))
      return false;
    interleaved.push_back(lo[d]);
    interleaved.push_back(hi[d]);
  }
  return true;
}

// tosa.pad takes the fill as a one-element tensor operand. hip carries it as a
// rank-0 tensor, so it is rematerialized as a TOSA constant instead of being
// reshaped, which means it has to be readable here. Only the output matching
// `elementType` is written; an absent operand is ONNX's default fill of zero.
// Both the legality predicate and the converter go through this, so an op is
// never claimed on terms the rewrite cannot then meet.
static bool matchPadConstant(hip::PadOp op, Type elementType, double &fpFill,
                             int64_t &intFill) {
  fpFill = 0.0;
  intFill = 0;
  bool isFloat = isa<FloatType>(elementType);
  if (!isFloat && !isa<IntegerType>(elementType))
    return false;

  Value cval = op.getConstantValue();
  if (!cval)
    return true;

  if (isFloat) {
    DenseFPElementsAttr dense;
    if (matchPattern(cval, m_Constant(&dense)) && dense.isSplat()) {
      fpFill = dense.getSplatValue<APFloat>().convertToDouble();
      return true;
    }
    FloatAttr scalar;
    if (matchPattern(cval, m_Constant(&scalar))) {
      fpFill = scalar.getValueAsDouble();
      return true;
    }
    return false;
  }

  // A splat collapses to its single value whatever its extent, which is what
  // the float path above already accepts. Requiring exactly one element here
  // instead would decline an integer fill spelled dense<5> : tensor<4xi32>
  // while converting the f32 spelling of the same thing.
  SmallVector<int64_t, 1> ints;
  if (!extractConstantInts(cval, ints) || ints.empty())
    return false;
  if (!llvm::all_equal(ints))
    return false;
  intFill = ints.front();
  return true;
}

static bool isTosaExpressiblePad(hip::PadOp op) {
  SmallVector<int64_t, 8> interleaved;
  if (!matchTosaPad(op, interleaved))
    return false;
  // matchTosaPad has already established the result is a ranked tensor.
  Type elementType =
      cast<RankedTensorType>(op.getResult(0).getType()).getElementType();
  double fpFill;
  int64_t intFill;
  return matchPadConstant(op, elementType, fpFill, intFill);
}

struct PadConverter final : public OpConversionPattern<hip::PadOp> {
  using OpConversionPattern<hip::PadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 8> interleaved;
    if (!matchTosaPad(op, interleaved))
      return rewriter.notifyMatchFailure(op, "not a tosa-expressible pad");

    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
    Type elementType = resultType.getElementType();
    auto padConstType = RankedTensorType::get({1}, elementType);

    double fpFill;
    int64_t intFill;
    if (!matchPadConstant(op, elementType, fpFill, intFill))
      return rewriter.notifyMatchFailure(
          op, "pad constant_value must be a scalar constant");
    Value padConst = isa<FloatType>(elementType)
                         ? createSplatFloat(rewriter, loc, padConstType, fpFill)
                         : createSplatInt(rewriter, loc, padConstType, intFill);

    rewriter.replaceOpWithNewOp<tosa::PadOp>(
        op, resultType, adaptor.getData(),
        createConstShape(rewriter, loc, interleaved), padConst);
    return success();
  }
};

// hip.constant carries a statically-shaped constant from the importer to
// hip-externalize-constants, which is the only pass allowed to choose between
// inline and external storage. Only the inline form has data to hand to TOSA;
// the file-backed and memory-address forms name a byte range that nothing has
// read yet, so there is nothing to put in a tosa.const and they decline.
//
// This is a coverage op rather than a live one today. The ONNX-to-HIP pipeline
// externalizes every carrier and then runs VerifyNoConstantCarriersPass, which
// fails the compile if one survives, so a carrier does not normally reach this
// pass at all. It matters the moment a constant does land inside a rock.kernel,
// because rocMLIR is handed that function on its own and knows no hip op.
static bool isTosaExpressibleConstant(hip::ConstantOp op) {
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType || !resultType.hasStaticShape())
    return false;
  // tosa.const takes the attribute as-is, so it has to be an elements
  // attribute whose type already matches the result.
  auto value = dyn_cast_or_null<DenseElementsAttr>(op.getValueAttr());
  if (!value || value.getType() != resultType)
    return false;
  Type elementType = resultType.getElementType();
  return isa<FloatType>(elementType) || isa<IntegerType>(elementType);
}

struct ConstantConverter final : public OpConversionPattern<hip::ConstantOp> {
  using OpConversionPattern<hip::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleConstant(op))
      return rewriter.notifyMatchFailure(op, "not a tosa-expressible constant");

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, resultType, cast<DenseElementsAttr>(op.getValueAttr()));
    return success();
  }
};

// TOSA resizes with an integer rational scale rather than a float one. The
// input coordinate it samples for output index `o` on an axis is
//
//   in = (o * scale_d + offset) / scale_n
//
// and it derives the output extent back out of that, requiring
//
//   out - 1 == ((in_extent - 1) * scale_n - offset + border) / scale_d
//
// to divide exactly. Each ONNX coordinate_transformation_mode is an affine map
// from `o` to the input coordinate, so each one is just a choice of the triple.
// Writing OUT and IN for the extents of one axis:
//
//   asymmetric      in = o * IN/OUT
//                   n = OUT, d = IN, offset = 0
//   half_pixel      in = (o + 0.5) * IN/OUT - 0.5
//                   n = 2*OUT, d = 2*IN, offset = IN - OUT
//   align_corners   in = o * (IN-1)/(OUT-1)
//                   n = OUT-1, d = IN-1, offset = 0
//
// half_pixel doubles the ratio because its offset is a half-integer otherwise;
// TOSA takes integers only, so the whole triple is scaled by two instead of
// rounding, which keeps the map exact. align_corners needs both extents above
// one, or its ratio has a zero in it.
//
// The border is not a fourth choice -- it is whatever makes TOSA's derived
// extent come back as OUT, so solving the relation above for it makes the
// divisibility hold by construction rather than by luck.
struct ResizeAxisParams {
  int64_t scaleN, scaleD, offset, border;
};

static std::optional<ResizeAxisParams>
planResizeAxis(int64_t inExtent, int64_t outExtent, int64_t coordTransform) {
  if (inExtent <= 0 || outExtent <= 0)
    return std::nullopt;

  ResizeAxisParams p;
  switch (coordTransform) {
  case 0: // half_pixel
    p.scaleN = 2 * outExtent;
    p.scaleD = 2 * inExtent;
    p.offset = inExtent - outExtent;
    break;
  case 1: // asymmetric
    p.scaleN = outExtent;
    p.scaleD = inExtent;
    p.offset = 0;
    break;
  case 2: // align_corners
    if (inExtent < 2 || outExtent < 2)
      return std::nullopt;
    p.scaleN = outExtent - 1;
    p.scaleD = inExtent - 1;
    p.offset = 0;
    break;
  default:
    return std::nullopt;
  }
  p.border = (outExtent - 1) * p.scaleD + p.offset - (inExtent - 1) * p.scaleN;
  // tosa.resize requires every scale value to be positive.
  if (p.scaleN <= 0 || p.scaleD <= 0)
    return std::nullopt;
  return p;
}

// hip.resize is (N, C, D_1..D_k) with the spatial axes trailing; tosa.resize is
// 4-D NHWC with exactly two spatial axes, so only the k == 2 case maps and it
// needs a transpose on each side.
//
// Nearest is declined rather than lowered. TOSA's NEAREST_NEIGHBOR breaks a tie
// upward, hip.resize carries ONNX's round_prefer_floor, which breaks it
// downward, and the tie is reachable: an asymmetric 2x upsample lands exactly
// halfway on every odd output index, so the two disagree on half the output.
//
// Integers are declined for a different reason: TOSA's integer BILINEAR leaves
// the result scaled by scale_y_n * scale_x_n for a following rescale to undo,
// and this pass emits no such rescale. ONNX Resize on a real model is float.
static bool isTosaExpressibleResize(hip::ResizeOp op) {
  if (op->getNumResults() != 1)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !inputType.hasStaticShape() || !resultType ||
      !resultType.hasStaticShape())
    return false;
  if (inputType.getElementType() != resultType.getElementType())
    return false;

  Type elementType = resultType.getElementType();
  if (!elementType.isF32() && !elementType.isF16() && !elementType.isBF16())
    return false;

  // N, C, H, W: the two leading axes are carried through untouched.
  if (inputType.getRank() != 4 || resultType.getRank() != 4)
    return false;
  if (inputType.getDimSize(0) != resultType.getDimSize(0) ||
      inputType.getDimSize(1) != resultType.getDimSize(1))
    return false;

  if (op.getMode() != 1)
    return false;
  int64_t coordTransform = op.getCoordTransform();
  return planResizeAxis(inputType.getDimSize(2), resultType.getDimSize(2),
                        coordTransform)
             .has_value() &&
         planResizeAxis(inputType.getDimSize(3), resultType.getDimSize(3),
                        coordTransform)
             .has_value();
}

struct ResizeConverter final : public OpConversionPattern<hip::ResizeOp> {
  using OpConversionPattern<hip::ResizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::ResizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleResize(op))
      return rewriter.notifyMatchFailure(op, "not a tosa-expressible resize");

    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    Type elementType = resultType.getElementType();

    int64_t coordTransform = op.getCoordTransform();
    ResizeAxisParams y = *planResizeAxis(
        inputType.getDimSize(2), resultType.getDimSize(2), coordTransform);
    ResizeAxisParams x = *planResizeAxis(
        inputType.getDimSize(3), resultType.getDimSize(3), coordTransform);

    Value nhwc = transposePerm(adaptor.getInput(), {0, 2, 3, 1}, rewriter, loc);
    auto resizedType = RankedTensorType::get(
        {resultType.getDimSize(0), resultType.getDimSize(2),
         resultType.getDimSize(3), resultType.getDimSize(1)},
        elementType);
    // Bound rather than passed inline: C++ leaves the evaluation order of call
    // arguments unspecified, so building these in the argument list would let
    // the order they are emitted in vary by host compiler.
    Value scale = createConstShape(rewriter, loc,
                                   {y.scaleN, y.scaleD, x.scaleN, x.scaleD});
    Value offset = createConstShape(rewriter, loc, {y.offset, x.offset});
    Value border = createConstShape(rewriter, loc, {y.border, x.border});
    Value resized = tosa::ResizeOp::create(
                        rewriter, loc, resizedType, nhwc, scale, offset, border,
                        tosa::ResizeModeAttr::get(rewriter.getContext(),
                                                  tosa::ResizeMode::BILINEAR))
                        .getResult();
    rewriter.replaceOp(op, transposePerm(resized, {0, 3, 1, 2}, rewriter, loc));
    return success();
  }
};

// ONNX permits indices in [-extent, extent-1]. TOSA requires non-negative
// in-range indices, so normalize the negative half before gathering.
Value normalizeNegativeIndices(Value indices, int64_t extent,
                               ConversionPatternRewriter &rewriter,
                               Location loc) {
  auto type = cast<RankedTensorType>(indices.getType());
  Value zero = createSplatInt(rewriter, loc, type, 0);
  Value extentSplat = createSplatInt(rewriter, loc, type, extent);
  Value isNegative = tosa::GreaterOp::create(
      rewriter, loc,
      RankedTensorType::get(type.getShape(), rewriter.getI1Type()), zero,
      indices);
  Value wrapped =
      tosa::AddOp::create(rewriter, loc, type, indices, extentSplat);
  return tosa::SelectOp::create(rewriter, loc, type, isNegative, wrapped,
                                indices);
}

// Take component `component` out of the trailing dimension of `indices`,
// dropping that dimension.
Value extractIndexComponent(Value indices, int64_t component,
                            ConversionPatternRewriter &rewriter, Location loc) {
  auto type = cast<RankedTensorType>(indices.getType());
  SmallVector<int64_t> starts(type.getRank(), 0);
  starts.back() = component;
  SmallVector<int64_t> sizes(type.getShape());
  sizes.back() = 1;
  auto shapeType = tosa::shapeType::get(rewriter.getContext(), type.getRank());
  auto start = tosa::ConstShapeOp::create(rewriter, loc, shapeType,
                                          rewriter.getIndexTensorAttr(starts));
  auto size = tosa::ConstShapeOp::create(rewriter, loc, shapeType,
                                         rewriter.getIndexTensorAttr(sizes));
  Value sliced = tosa::SliceOp::create(rewriter, loc, type.clone(sizes),
                                       indices, start, size);
  return reshapeTo(sliced, type.getShape().drop_back(), rewriter);
}

SmallVector<int64_t> permuteShape(ArrayRef<int64_t> shape,
                                  ArrayRef<int32_t> permutation) {
  SmallVector<int64_t> permuted;
  for (int32_t dim : permutation)
    permuted.push_back(shape[dim]);
  return permuted;
}

// TOSA gathers and scatters index the middle of [N,K,C], so an ONNX op that
// indexes `axis` elementwise has to move it to the back, where the remaining
// dimensions are contiguous and collapse into N. `fromBack` undoes the move:
// the axis sits last and belongs at `axis`, which pushed each dimension after
// it one place forward.
void axisToBackPermutations(int64_t rank, int64_t axis,
                            SmallVectorImpl<int32_t> &toBack,
                            SmallVectorImpl<int32_t> &fromBack) {
  for (int64_t i = 0; i < rank; ++i)
    if (i != axis)
      toBack.push_back(static_cast<int32_t>(i));
  toBack.push_back(static_cast<int32_t>(axis));
  for (int64_t i = 0; i < rank; ++i)
    fromBack.push_back(
        static_cast<int32_t>(i < axis ? i : (i == axis ? rank - 1 : i - 1)));
}

// Fold the index tuples held in the trailing dimension of `indices`, shaped
// [N, W, extents.size()], into one index into `extents` flattened together,
// weighting each component by its row-major stride. The trailing component
// has stride 1, so a one-wide tuple costs no arithmetic at all.
Value linearizeIndexTuple(Value indices, ArrayRef<int64_t> extents,
                          ConversionPatternRewriter &rewriter, Location loc) {
  auto linearTy = RankedTensorType::get(
      cast<RankedTensorType>(indices.getType()).getShape().drop_back(),
      rewriter.getI32Type());
  int64_t stride = 1;
  for (int64_t extent : extents)
    stride *= extent;

  Value linear;
  for (auto [i, extent] : llvm::enumerate(extents)) {
    stride /= extent;
    Value component = extractIndexComponent(indices, i, rewriter, loc);
    component = normalizeNegativeIndices(component, extent, rewriter, loc);
    if (stride != 1)
      component =
          tosa::MulOp::create(rewriter, loc, linearTy, component,
                              createSplatInt(rewriter, loc, linearTy, stride),
                              createZeroMulShift(rewriter, loc));
    linear =
        linear ? tosa::AddOp::create(rewriter, loc, linearTy, linear, component)
               : component;
  }
  return linear;
}

// ONNX Gather indexes one axis with an indices tensor of arbitrary rank. TOSA
// gather has the canonical batched form [N,K,C] x [N,W] -> [N,W,C]. Flatten
// the dimensions around the gathered axis into N/C, replicate the common ONNX
// indices across N, gather, then restore the ONNX result shape.
struct GatherConverter final : public OpConversionPattern<GatherOp> {
  using OpConversionPattern<GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto dataTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!dataTy || !indicesTy || !resultTy || !dataTy.hasStaticShape() ||
        !indicesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected static ranked data, indices, and result tensors");

    int64_t rank = dataTy.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");

    int64_t n = 1;
    int64_t c = 1;
    for (int64_t i = 0; i < axis; ++i)
      n *= dataTy.getDimSize(i);
    for (int64_t i = axis + 1; i < rank; ++i)
      c *= dataTy.getDimSize(i);
    int64_t k = dataTy.getDimSize(axis);
    int64_t w = indicesTy.getNumElements();
    if (k <= 0)
      return rewriter.notifyMatchFailure(op, "gathered axis must be non-empty");

    SmallVector<int64_t> expectedShape;
    expectedShape.append(dataTy.getShape().begin(),
                         dataTy.getShape().begin() + axis);
    expectedShape.append(indicesTy.getShape().begin(),
                         indicesTy.getShape().end());
    expectedShape.append(dataTy.getShape().begin() + axis + 1,
                         dataTy.getShape().end());
    if (resultTy.getShape() != ArrayRef<int64_t>(expectedShape))
      return rewriter.notifyMatchFailure(op, "result shape is not ONNX Gather");

    Location loc = op.getLoc();
    Value values = reshapeTo(adaptor.getData(), {n, k, c}, rewriter);
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(),
                                 rewriter.getI32Type());
    indices = reshapeTo(indices, {1, w}, rewriter);
    if (n != 1)
      indices = tileMultiples(indices, {n, 1}, {n, w}, rewriter, loc);

    indices = normalizeNegativeIndices(indices, k, rewriter, loc);

    auto gatheredTy = RankedTensorType::get({n, w, c}, dataTy.getElementType());
    Value gathered =
        tosa::GatherOp::create(rewriter, loc, gatheredTy, values, indices);
    rewriter.replaceOp(op, reshapeTo(gathered, resultTy.getShape(), rewriter));
    return success();
  }
};

// TOSA has no one_hot. Expand static OneHot to broadcasted elementwise ops:
//
//   canonical = indices < 0 ? indices + depth : indices
//   output = select(canonical == [0, ..., depth - 1], on, off)
//
// The index and class tensors are reshaped and tiled to the output shape.
// Values outside [-depth, depth - 1] compare unequal to every class and
// therefore produce the off value, matching hip_one_hot.
//
// Before:
//   %y = hip.one_hot(%ctx)
//          ins(%indices, %depth, %values : tensor<2xi64>, tensor<i64>,
//                                             tensor<2xf32>)
//          outs(%init : tensor<2x4xf32>) : tensor<2x4xf32>
// After:
//   %canonical = tosa.select (tosa.greater 0, %indices),
//                            (tosa.add %indices, 4), %indices
//   %classes = tosa.const [0, 1, 2, 3]
//   %pred = tosa.equal (tiled %canonical), (tiled %classes)
//   %y = tosa.select %pred, %on, %off
struct OneHotConverter final : public OpConversionPattern<OneHotOp> {
  using OpConversionPattern<OneHotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OneHotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto depthTy = dyn_cast<RankedTensorType>(adaptor.getDepth().getType());
    auto valuesTy = dyn_cast<RankedTensorType>(adaptor.getValues().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!indicesTy || !depthTy || !valuesTy || !resultTy ||
        !indicesTy.hasStaticShape() || !depthTy.hasStaticShape() ||
        !valuesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (!isa<IntegerType>(indicesTy.getElementType()) ||
        !isa<IntegerType>(depthTy.getElementType()))
      return rewriter.notifyMatchFailure(op,
                                         "indices and depth must be integer");
    if (valuesTy.getRank() != 1 || valuesTy.getDimSize(0) != 2)
      return rewriter.notifyMatchFailure(op, "values must have shape [2]");
    if (valuesTy.getElementType() != resultTy.getElementType())
      return rewriter.notifyMatchFailure(op, "values and result types differ");
    if (resultTy.getRank() != indicesTy.getRank() + 1)
      return rewriter.notifyMatchFailure(
          op, "result rank must be indices rank plus one");

    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += resultTy.getRank();
    if (axis < 0 || axis >= resultTy.getRank())
      return rewriter.notifyMatchFailure(op, "axis out of range");

    SmallVector<int64_t> expectedShape;
    expectedShape.reserve(resultTy.getRank());
    expectedShape.append(indicesTy.getShape().begin(),
                         indicesTy.getShape().begin() + axis);
    int64_t depth = resultTy.getDimSize(axis);
    expectedShape.push_back(depth);
    expectedShape.append(indicesTy.getShape().begin() + axis,
                         indicesTy.getShape().end());
    if (depth <= 0 || expectedShape != resultTy.getShape())
      return rewriter.notifyMatchFailure(op, "invalid static OneHot shape");
    if (llvm::any_of(resultTy.getShape(), [](int64_t dim) { return dim <= 0; }))
      return rewriter.notifyMatchFailure(op, "expected positive extents");

    Location loc = op.getLoc();
    auto indexElemTy = cast<IntegerType>(indicesTy.getElementType());
    SmallVector<int64_t> indexShape(resultTy.getShape().begin(),
                                    resultTy.getShape().end());
    indexShape[axis] = 1;
    Value indices = reshapeTo(adaptor.getIndices(), indexShape, rewriter);

    auto expandedIndicesTy = RankedTensorType::get(indexShape, indexElemTy);
    Value zero = createSplatInt(rewriter, loc, expandedIndicesTy, 0);
    Value depthSplat = createSplatInt(rewriter, loc, expandedIndicesTy, depth);
    auto indexPredTy = RankedTensorType::get(indexShape, rewriter.getI1Type());
    Value isNegative =
        tosa::GreaterOp::create(rewriter, loc, indexPredTy, zero, indices);
    Value wrapped = tosa::AddOp::create(rewriter, loc, expandedIndicesTy,
                                        indices, depthSplat);
    indices = tosa::SelectOp::create(rewriter, loc, expandedIndicesTy,
                                     isNegative, wrapped, indices);

    SmallVector<int64_t> indexMultiples(resultTy.getRank(), 1);
    indexMultiples[axis] = depth;
    indices = tileMultiples(indices, indexMultiples, resultTy.getShape(),
                            rewriter, loc);

    SmallVector<int64_t> classShape(resultTy.getRank(), 1);
    classShape[axis] = depth;
    auto classTy = RankedTensorType::get(classShape, indexElemTy);
    SmallVector<APInt> classValues;
    classValues.reserve(depth);
    for (int64_t i = 0; i < depth; ++i)
      classValues.emplace_back(indexElemTy.getWidth(), i);
    Value classes = tosa::ConstOp::create(
        rewriter, loc, classTy, DenseElementsAttr::get(classTy, classValues));

    SmallVector<int64_t> classMultiples(resultTy.getShape().begin(),
                                        resultTy.getShape().end());
    classMultiples[axis] = 1;
    classes = tileMultiples(classes, classMultiples, resultTy.getShape(),
                            rewriter, loc);

    auto predTy =
        RankedTensorType::get(resultTy.getShape(), rewriter.getI1Type());
    Value pred = tosa::EqualOp::create(rewriter, loc, predTy, indices, classes);

    SmallVector<int64_t> scalarShape(resultTy.getRank(), 1);
    Value off =
        reshapeTo(sliceOffsetSize(adaptor.getValues(), {0}, {1}, rewriter, loc),
                  scalarShape, rewriter);
    Value on =
        reshapeTo(sliceOffsetSize(adaptor.getValues(), {1}, {1}, rewriter, loc),
                  scalarShape, rewriter);
    rewriter.replaceOpWithNewOp<tosa::SelectOp>(op, resultTy, pred, on, off);
    return success();
  }
};

// TOSA has no grid_sample. tosa.resize only scales a regular lattice, so
// hip.grid_sample expands to the unnormalize / gather / interpolate sequence
// hip_grid_sample uses:
//
//   align_corners=1: ((g + 1) * (size - 1)) / 2
//   align_corners=0: ((g + 1) * size - 1) / 2
//   nearest:  floor(coord + 0.5)
//   bilinear: four-neighbour lerp
//
// Input is NCHW; TOSA gather is [N,K,C] x [N,W] -> [N,W,C], so the spatial
// plane is flattened after an NHWC transpose. Reflection padding is left
// unconverted: it needs a periodic fold TOSA cannot express with clamp.
//
// Before:
//   %y = hip.grid_sample(%ctx)
//            ins(%x, %g : tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>)
//            outs(%init : tensor<1x3x4x4xf32>)
//            {mode = 1, padding_mode = 0, align_corners = 0}
// After (bilinear):
//   %nhwc = tosa.transpose %x
//   %flat = tosa.reshape %nhwc
//   %v00/%v01/%v10/%v11 = tosa.gather ...
//   %y = tosa.transpose (weighted sum)
Value unnormalizeCoord(Value g, int64_t size, int64_t alignCorners,
                       RankedTensorType ty, ConversionPatternRewriter &rewriter,
                       Location loc) {
  if (size <= 1)
    return createSplatFloat(rewriter, loc, ty, 0.0);
  Value one = createSplatFloat(rewriter, loc, ty, 1.0);
  Value gp1 = tosa::AddOp::create(rewriter, loc, ty, g, one);
  if (alignCorners)
    return emitTosaMul(rewriter, loc, gp1,
                       createSplatFloat(rewriter, loc, ty,
                                        0.5 * static_cast<double>(size - 1)),
                       ty);
  Value scaled = emitTosaMul(
      rewriter, loc, gp1,
      createSplatFloat(rewriter, loc, ty, static_cast<double>(size)), ty);
  Value inner = tosa::AddOp::create(rewriter, loc, ty, scaled,
                                    createSplatFloat(rewriter, loc, ty, -1.0));
  return emitTosaMul(rewriter, loc, inner,
                     createSplatFloat(rewriter, loc, ty, 0.5), ty);
}

Value clampIndex(Value idx, int64_t lo, int64_t hi, RankedTensorType ty,
                 ConversionPatternRewriter &rewriter, Location loc) {
  Value loV = createSplatInt(rewriter, loc, ty, lo);
  Value hiV = createSplatInt(rewriter, loc, ty, hi);
  Value low = tosa::MaximumOp::create(rewriter, loc, ty, idx, loV);
  return tosa::MinimumOp::create(rewriter, loc, ty, low, hiV);
}

Value inRangeMask(Value idx, int64_t extent, RankedTensorType idxTy,
                  ConversionPatternRewriter &rewriter, Location loc) {
  auto predTy = RankedTensorType::get(idxTy.getShape(), rewriter.getI1Type());
  Value zero = createSplatInt(rewriter, loc, idxTy, 0);
  Value hi = createSplatInt(rewriter, loc, idxTy, extent);
  Value isNeg = tosa::GreaterOp::create(rewriter, loc, predTy, zero, idx);
  Value ge0 = tosa::BitwiseNotOp::create(rewriter, loc, predTy, isNeg);
  Value lt = tosa::GreaterOp::create(rewriter, loc, predTy, hi, idx);
  return tosa::BitwiseAndOp::create(rewriter, loc, predTy, ge0, lt);
}

Value gatherSpatial(Value values, Value y, Value x, int64_t n, int64_t ho,
                    int64_t wo, int64_t c, int64_t width,
                    ConversionPatternRewriter &rewriter, Location loc) {
  auto idxTy = cast<RankedTensorType>(y.getType());
  Value wSplat = createSplatInt(rewriter, loc, idxTy, width);
  Value linear = tosa::AddOp::create(
      rewriter, loc, idxTy,
      tosa::MulOp::create(rewriter, loc, idxTy, y, wSplat,
                          createZeroMulShift(rewriter, loc)),
      x);
  linear = reshapeTo(linear, {n, ho * wo}, rewriter);
  auto gatheredTy = RankedTensorType::get(
      {n, ho * wo, c},
      cast<RankedTensorType>(values.getType()).getElementType());
  Value gathered =
      tosa::GatherOp::create(rewriter, loc, gatheredTy, values, linear);
  return reshapeTo(gathered, {n, ho, wo, c}, rewriter);
}

Value applyZerosMask(Value sample, Value validY, Value validX,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto predTy = cast<RankedTensorType>(validY.getType());
  Value valid =
      tosa::BitwiseAndOp::create(rewriter, loc, predTy, validY, validX);
  auto sampleTy = cast<RankedTensorType>(sample.getType());
  SmallVector<int64_t> unsqueeze(predTy.getShape().begin(),
                                 predTy.getShape().end());
  unsqueeze.push_back(1);
  Value pred = reshapeTo(valid, unsqueeze, rewriter);
  if (failed(tosa::EqualizeRanks(rewriter, loc, pred, sample)))
    return sample;
  Value zero = createSplatFloat(rewriter, loc, sampleTy, 0.0);
  return tosa::SelectOp::create(rewriter, loc, sampleTy, pred, sample, zero);
}

Value bcastMul(Value sample, Value weight, ConversionPatternRewriter &rewriter,
               Location loc) {
  auto sampleTy = cast<RankedTensorType>(sample.getType());
  ArrayRef<int64_t> shape = sampleTy.getShape();
  Value w = reshapeTo(weight, {shape[0], shape[1], shape[2], 1}, rewriter);
  if (failed(tosa::EqualizeRanks(rewriter, loc, sample, w)))
    return sample;
  return emitTosaMul(rewriter, loc, sample, w, sampleTy);
}

struct GridSampleConverter final : public OpConversionPattern<GridSampleOp> {
  using OpConversionPattern<GridSampleOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GridSampleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto inputTy = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto gridTy = dyn_cast<RankedTensorType>(adaptor.getGrid().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inputTy || !gridTy || !resultTy || !inputTy.hasStaticShape() ||
        !gridTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (inputTy.getRank() != 4 || gridTy.getRank() != 4 ||
        resultTy.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4-D NCHW grid_sample");
    if (gridTy.getDimSize(3) != 2)
      return rewriter.notifyMatchFailure(op, "grid last dim must be 2");
    Type elemTy = resultTy.getElementType();
    if (!isa<FloatType>(elemTy) || inputTy.getElementType() != elemTy ||
        gridTy.getElementType() != elemTy)
      return rewriter.notifyMatchFailure(op, "tosa op requires a float tensor");

    int64_t n = inputTy.getDimSize(0);
    int64_t c = inputTy.getDimSize(1);
    int64_t h = inputTy.getDimSize(2);
    int64_t w = inputTy.getDimSize(3);
    int64_t ho = gridTy.getDimSize(1);
    int64_t wo = gridTy.getDimSize(2);
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0 || ho <= 0 || wo <= 0)
      return rewriter.notifyMatchFailure(op, "expected positive extents");
    if (resultTy.getShape() != ArrayRef<int64_t>({n, c, ho, wo}))
      return rewriter.notifyMatchFailure(op, "result shape is not NCHW output");
    if (gridTy.getDimSize(0) != n)
      return rewriter.notifyMatchFailure(op, "grid batch must match input");

    int64_t mode = op.getMode();
    int64_t padding = op.getPaddingMode();
    int64_t align = op.getAlignCorners();
    if (mode != 0 && mode != 1)
      return rewriter.notifyMatchFailure(op,
                                         "mode must be nearest or bilinear");
    if (padding == 2)
      return rewriter.notifyMatchFailure(
          op, "reflection padding is not expressed in TOSA");
    if (padding != 0 && padding != 1)
      return rewriter.notifyMatchFailure(op, "unsupported padding_mode");

    Location loc = op.getLoc();
    auto coordTy = RankedTensorType::get({n, ho, wo}, elemTy);
    Value gx = reshapeTo(sliceOffsetSize(adaptor.getGrid(), {0, 0, 0, 0},
                                         {n, ho, wo, 1}, rewriter, loc),
                         {n, ho, wo}, rewriter);
    Value gy = reshapeTo(sliceOffsetSize(adaptor.getGrid(), {0, 0, 0, 1},
                                         {n, ho, wo, 1}, rewriter, loc),
                         {n, ho, wo}, rewriter);
    Value x = unnormalizeCoord(gx, w, align, coordTy, rewriter, loc);
    Value y = unnormalizeCoord(gy, h, align, coordTy, rewriter, loc);

    Value nhwc = transposePerm(adaptor.getInput(), {0, 2, 3, 1}, rewriter, loc);
    Value values = reshapeTo(nhwc, {n, h * w, c}, rewriter);
    auto i32CoordTy = RankedTensorType::get({n, ho, wo}, rewriter.getI32Type());
    auto nhwcTy = RankedTensorType::get({n, ho, wo, c}, elemTy);

    auto sampleAt = [&](Value ys, Value xs, Value maskY, Value maskX) {
      // TOSA gather requires in-range indices. Clamp first, then zero OOB
      // samples when padding_mode is zeros.
      Value yy = clampIndex(ys, 0, h - 1, i32CoordTy, rewriter, loc);
      Value xx = clampIndex(xs, 0, w - 1, i32CoordTy, rewriter, loc);
      Value sample =
          gatherSpatial(values, yy, xx, n, ho, wo, c, w, rewriter, loc);
      if (padding == 0)
        sample = applyZerosMask(sample, maskY, maskX, rewriter, loc);
      return sample;
    };

    Value resultNHWC;
    if (mode == 0) {
      Value half = createSplatFloat(rewriter, loc, coordTy, 0.5);
      Value yN = tosa::FloorOp::create(
          rewriter, loc, coordTy,
          tosa::AddOp::create(rewriter, loc, coordTy, y, half));
      Value xN = tosa::FloorOp::create(
          rewriter, loc, coordTy,
          tosa::AddOp::create(rewriter, loc, coordTy, x, half));
      Value yi = emitTosaCast(rewriter, loc, yN, rewriter.getI32Type());
      Value xi = emitTosaCast(rewriter, loc, xN, rewriter.getI32Type());
      Value maskY = inRangeMask(yi, h, i32CoordTy, rewriter, loc);
      Value maskX = inRangeMask(xi, w, i32CoordTy, rewriter, loc);
      resultNHWC = sampleAt(yi, xi, maskY, maskX);
    } else {
      Value y0f = tosa::FloorOp::create(rewriter, loc, coordTy, y);
      Value x0f = tosa::FloorOp::create(rewriter, loc, coordTy, x);
      Value y0 = emitTosaCast(rewriter, loc, y0f, rewriter.getI32Type());
      Value x0 = emitTosaCast(rewriter, loc, x0f, rewriter.getI32Type());
      Value oneI = createSplatInt(rewriter, loc, i32CoordTy, 1);
      Value y1 = tosa::AddOp::create(rewriter, loc, i32CoordTy, y0, oneI);
      Value x1 = tosa::AddOp::create(rewriter, loc, i32CoordTy, x0, oneI);
      Value dy = tosa::SubOp::create(rewriter, loc, coordTy, y, y0f);
      Value dx = tosa::SubOp::create(rewriter, loc, coordTy, x, x0f);
      Value oneF = createSplatFloat(rewriter, loc, coordTy, 1.0);
      Value omx = tosa::SubOp::create(rewriter, loc, coordTy, oneF, dx);
      Value omy = tosa::SubOp::create(rewriter, loc, coordTy, oneF, dy);

      Value mY0 = inRangeMask(y0, h, i32CoordTy, rewriter, loc);
      Value mY1 = inRangeMask(y1, h, i32CoordTy, rewriter, loc);
      Value mX0 = inRangeMask(x0, w, i32CoordTy, rewriter, loc);
      Value mX1 = inRangeMask(x1, w, i32CoordTy, rewriter, loc);
      Value v00 = sampleAt(y0, x0, mY0, mX0);
      Value v01 = sampleAt(y0, x1, mY0, mX1);
      Value v10 = sampleAt(y1, x0, mY1, mX0);
      Value v11 = sampleAt(y1, x1, mY1, mX1);

      Value t00 =
          bcastMul(bcastMul(v00, omx, rewriter, loc), omy, rewriter, loc);
      Value t01 =
          bcastMul(bcastMul(v01, dx, rewriter, loc), omy, rewriter, loc);
      Value t10 =
          bcastMul(bcastMul(v10, omx, rewriter, loc), dy, rewriter, loc);
      Value t11 = bcastMul(bcastMul(v11, dx, rewriter, loc), dy, rewriter, loc);
      Value s0 = tosa::AddOp::create(rewriter, loc, nhwcTy, t00, t01);
      Value s1 = tosa::AddOp::create(rewriter, loc, nhwcTy, t10, t11);
      resultNHWC = tosa::AddOp::create(rewriter, loc, nhwcTy, s0, s1);
    }

    rewriter.replaceOp(op,
                       transposePerm(resultNHWC, {0, 3, 1, 2}, rewriter, loc));
    return success();
  }
};

// ONNX GatherElements reads one element per output position: `indices` has
// data's rank and holds, at every position, the coordinate to read along
// `axis` while the remaining coordinates are the position's own. TOSA gather
// fetches a whole contiguous C-wide slice per index, so drive it with C = 1 --
// move `axis` last, collapse every other dimension into the batch N, and the
// single-element "slice" it then fetches is exactly what ONNX asks for.
struct GatherElementsConverter final
    : public OpConversionPattern<GatherElementsOp> {
  using OpConversionPattern<GatherElementsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto dataTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!dataTy || !indicesTy || !resultTy || !dataTy.hasStaticShape() ||
        !indicesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected static ranked data, indices, and result tensors");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");

    int64_t rank = dataTy.getRank();
    if (indicesTy.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "indices must have data's rank");

    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Off the gathered axis an output position reads data at its own
    // coordinate, so N is a shared batch only when the two agree there.
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis && dataTy.getDimSize(i) != indicesTy.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "data and indices disagree off the gathered axis");
    if (resultTy.getShape() != indicesTy.getShape() ||
        resultTy.getElementType() != dataTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "result type is not ONNX GatherElements");

    int64_t k = dataTy.getDimSize(axis);
    int64_t w = indicesTy.getDimSize(axis);
    int64_t n = 1;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        n *= indicesTy.getDimSize(i);
    if (k <= 0 || w <= 0 || n <= 0)
      return rewriter.notifyMatchFailure(
          op, "gathered dimensions must be non-empty");

    SmallVector<int32_t> axisToBack, axisFromBack;
    axisToBackPermutations(rank, axis, axisToBack, axisFromBack);
    SmallVector<int64_t> dataBackShape =
        permuteShape(dataTy.getShape(), axisToBack);
    SmallVector<int64_t> indicesBackShape =
        permuteShape(indicesTy.getShape(), axisToBack);

    Location loc = op.getLoc();
    Value values = adaptor.getData();
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(),
                                 rewriter.getI32Type());
    if (axis != rank - 1) {
      values = transposeTo(values, dataBackShape, axisToBack, rewriter, loc);
      indices =
          transposeTo(indices, indicesBackShape, axisToBack, rewriter, loc);
    }
    values = reshapeTo(values, {n, k, 1}, rewriter);
    indices = reshapeTo(indices, {n, w}, rewriter);
    indices = normalizeNegativeIndices(indices, k, rewriter, loc);

    auto gatheredTy = RankedTensorType::get({n, w, 1}, dataTy.getElementType());
    Value gathered =
        tosa::GatherOp::create(rewriter, loc, gatheredTy, values, indices);
    Value result = reshapeTo(gathered, indicesBackShape, rewriter);
    if (axis != rank - 1)
      result =
          transposeTo(result, resultTy.getShape(), axisFromBack, rewriter, loc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ONNX GatherND indexes the `tuple` dimensions following `batch_dims` with an
// index tuple held in the trailing dimension of `indices`. TOSA gather indexes
// one dimension, so flatten those dimensions into a single K and fold each
// tuple into the matching row-major offset. The batch dimensions need no such
// work: they map straight onto TOSA's N, and row-major layout already leaves
// data in the [N, K, C] order the op wants, so no transpose is needed either.
struct GatherNDConverter final : public OpConversionPattern<GatherNDOp> {
  using OpConversionPattern<GatherNDOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto dataTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!dataTy || !indicesTy || !resultTy || !dataTy.hasStaticShape() ||
        !indicesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected static ranked data, indices, and result tensors");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");

    int64_t rank = dataTy.getRank();
    int64_t indicesRank = indicesTy.getRank();
    int64_t batchDims = op.getBatchDims();
    if (batchDims < 0 || batchDims >= indicesRank || batchDims > rank)
      return rewriter.notifyMatchFailure(op, "batch_dims out of range");

    // The trailing dimension carries the tuple rather than a gathered
    // position, and names how many of data's dimensions each tuple indexes.
    int64_t tuple = indicesTy.getDimSize(indicesRank - 1);
    if (tuple < 1 || batchDims + tuple > rank)
      return rewriter.notifyMatchFailure(
          op, "index tuple does not name a valid slice of data");
    for (int64_t i = 0; i < batchDims; ++i)
      if (dataTy.getDimSize(i) != indicesTy.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "data and indices disagree on the batch dims");

    SmallVector<int64_t> expectedShape(indicesTy.getShape().drop_back());
    expectedShape.append(dataTy.getShape().begin() + batchDims + tuple,
                         dataTy.getShape().end());
    if (resultTy.getShape() != ArrayRef<int64_t>(expectedShape))
      return rewriter.notifyMatchFailure(op,
                                         "result shape is not ONNX GatherND");

    int64_t n = 1;
    int64_t k = 1;
    int64_t c = 1;
    int64_t w = 1;
    for (int64_t i = 0; i < batchDims; ++i)
      n *= dataTy.getDimSize(i);
    for (int64_t i = batchDims; i < batchDims + tuple; ++i)
      k *= dataTy.getDimSize(i);
    for (int64_t i = batchDims + tuple; i < rank; ++i)
      c *= dataTy.getDimSize(i);
    for (int64_t i = batchDims; i < indicesRank - 1; ++i)
      w *= indicesTy.getDimSize(i);
    // Also rules out a zero extent below, where each one divides the stride.
    if (k <= 0)
      return rewriter.notifyMatchFailure(op, "gathered dims must be non-empty");

    Location loc = op.getLoc();
    Value values = reshapeTo(adaptor.getData(), {n, k, c}, rewriter);
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(),
                                 rewriter.getI32Type());
    indices = reshapeTo(indices, {n, w, tuple}, rewriter);

    Value linear = linearizeIndexTuple(
        indices, dataTy.getShape().slice(batchDims, tuple), rewriter, loc);

    auto gatheredTy = RankedTensorType::get({n, w, c}, dataTy.getElementType());
    Value gathered =
        tosa::GatherOp::create(rewriter, loc, gatheredTy, values, linear);
    rewriter.replaceOp(op, reshapeTo(gathered, resultTy.getShape(), rewriter));
    return success();
  }
};

// Only ONNX reduction "none" reaches tosa.scatter. The accumulating modes
// exist precisely to define what happens when two updates land on one
// position, which tosa.scatter forbids outright ("It is not permitted to
// repeat the same output index"), and they must also read the existing value
// back, which a single scatter cannot do. Under "none" the two agree: ONNX
// requires distinct indices there as well, so nothing is given up.
//
// That ban is also why W may not exceed K: more updates than the scattered
// range holds must repeat an index, so such an op is not valid ONNX either.
LogicalResult checkScatterIsOverwrite(Operation *op, StringRef reduction,
                                      int64_t k, int64_t w,
                                      ConversionPatternRewriter &rewriter) {
  if (reduction != "none")
    return rewriter.notifyMatchFailure(
        op, "only reduction 'none' maps to tosa.scatter");
  if (k <= 0)
    return rewriter.notifyMatchFailure(op, "scattered dims must be non-empty");
  if (w > k)
    return rewriter.notifyMatchFailure(
        op, "more updates than the scattered dims hold, so an index repeats");
  return success();
}

// ONNX ScatterElements writes one element per update position, the inverse of
// GatherElements, and reaches TOSA the same way: move `axis` last so the other
// dimensions collapse into N, then scatter with C = 1 so each written "slice"
// is the single element ONNX means.
struct ScatterElementsConverter final
    : public OpConversionPattern<ScatterElementsOp> {
  using OpConversionPattern<ScatterElementsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScatterElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto dataTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto updatesTy = dyn_cast<RankedTensorType>(adaptor.getUpdates().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!dataTy || !indicesTy || !updatesTy || !resultTy ||
        !dataTy.hasStaticShape() || !indicesTy.hasStaticShape() ||
        !updatesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked data, "
                                             "indices, updates, and result "
                                             "tensors");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");
    if (updatesTy.getShape() != indicesTy.getShape() ||
        updatesTy.getElementType() != dataTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "updates must have the indices shape and data element type");
    if (resultTy.getShape() != dataTy.getShape() ||
        resultTy.getElementType() != dataTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "result must have the data shape and element type");

    int64_t rank = dataTy.getRank();
    if (indicesTy.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "indices must have data's rank");

    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Off the scattered axis an update lands at its own coordinate, so N is a
    // shared batch only when the two agree there.
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis && dataTy.getDimSize(i) != indicesTy.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "data and indices disagree off the scattered axis");

    int64_t k = dataTy.getDimSize(axis);
    int64_t w = indicesTy.getDimSize(axis);
    if (failed(checkScatterIsOverwrite(op, op.getReduction(), k, w, rewriter)))
      return failure();
    int64_t n = 1;
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis)
        n *= dataTy.getDimSize(i);

    SmallVector<int32_t> axisToBack, axisFromBack;
    axisToBackPermutations(rank, axis, axisToBack, axisFromBack);
    SmallVector<int64_t> dataBackShape =
        permuteShape(dataTy.getShape(), axisToBack);

    Location loc = op.getLoc();
    Value values = adaptor.getData();
    Value updates = adaptor.getUpdates();
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(),
                                 rewriter.getI32Type());
    if (axis != rank - 1) {
      // updates carries the indices shape, so it takes the same permutation.
      SmallVector<int64_t> indicesBackShape =
          permuteShape(indicesTy.getShape(), axisToBack);
      values = transposeTo(values, dataBackShape, axisToBack, rewriter, loc);
      indices =
          transposeTo(indices, indicesBackShape, axisToBack, rewriter, loc);
      updates =
          transposeTo(updates, indicesBackShape, axisToBack, rewriter, loc);
    }
    values = reshapeTo(values, {n, k, 1}, rewriter);
    updates = reshapeTo(updates, {n, w, 1}, rewriter);
    indices = reshapeTo(indices, {n, w}, rewriter);
    indices = normalizeNegativeIndices(indices, k, rewriter, loc);

    auto scatteredTy =
        RankedTensorType::get({n, k, 1}, dataTy.getElementType());
    Value scattered = tosa::ScatterOp::create(rewriter, loc, scatteredTy,
                                              values, indices, updates);
    Value result = reshapeTo(scattered, dataBackShape, rewriter);
    if (axis != rank - 1)
      result =
          transposeTo(result, resultTy.getShape(), axisFromBack, rewriter, loc);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ONNX ScatterND is the inverse of GatherND, but carries no batch_dims: every
// tuple indexes data's leading dimensions directly. So N is 1, the leading
// dimensions the tuple names flatten into K, the trailing ones into C, and the
// tuples fold into linear indices exactly as they do for GatherND.
struct ScatterNDConverter final : public OpConversionPattern<ScatterNDOp> {
  using OpConversionPattern<ScatterNDOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ScatterNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto dataTy = dyn_cast<RankedTensorType>(adaptor.getData().getType());
    auto indicesTy = dyn_cast<RankedTensorType>(adaptor.getIndices().getType());
    auto updatesTy = dyn_cast<RankedTensorType>(adaptor.getUpdates().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!dataTy || !indicesTy || !updatesTy || !resultTy ||
        !dataTy.hasStaticShape() || !indicesTy.hasStaticShape() ||
        !updatesTy.hasStaticShape() || !resultTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked data, "
                                             "indices, updates, and result "
                                             "tensors");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");
    if (resultTy.getShape() != dataTy.getShape())
      return rewriter.notifyMatchFailure(op, "result must have the data shape");

    int64_t rank = dataTy.getRank();
    int64_t indicesRank = indicesTy.getRank();
    if (indicesRank < 1)
      return rewriter.notifyMatchFailure(op,
                                         "indices must have rank at least 1");
    int64_t tuple = indicesTy.getDimSize(indicesRank - 1);
    if (tuple < 1 || tuple > rank)
      return rewriter.notifyMatchFailure(
          op, "index tuple does not name a valid slice of data");

    SmallVector<int64_t> expectedUpdates(indicesTy.getShape().drop_back());
    expectedUpdates.append(dataTy.getShape().begin() + tuple,
                           dataTy.getShape().end());
    if (updatesTy.getShape() != ArrayRef<int64_t>(expectedUpdates))
      return rewriter.notifyMatchFailure(op,
                                         "updates shape is not ONNX ScatterND");

    int64_t k = 1;
    int64_t c = 1;
    int64_t w = 1;
    for (int64_t i = 0; i < tuple; ++i)
      k *= dataTy.getDimSize(i);
    for (int64_t i = tuple; i < rank; ++i)
      c *= dataTy.getDimSize(i);
    for (int64_t i = 0; i < indicesRank - 1; ++i)
      w *= indicesTy.getDimSize(i);
    // Also rules out a zero extent below, where each one divides the stride.
    if (failed(checkScatterIsOverwrite(op, op.getReduction(), k, w, rewriter)))
      return failure();

    Location loc = op.getLoc();
    Value values = reshapeTo(adaptor.getData(), {1, k, c}, rewriter);
    Value updates = reshapeTo(adaptor.getUpdates(), {1, w, c}, rewriter);
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(),
                                 rewriter.getI32Type());
    indices = reshapeTo(indices, {1, w, tuple}, rewriter);
    Value linear = linearizeIndexTuple(
        indices, dataTy.getShape().take_front(tuple), rewriter, loc);

    auto scatteredTy =
        RankedTensorType::get({1, k, c}, dataTy.getElementType());
    Value scattered = tosa::ScatterOp::create(rewriter, loc, scatteredTy,
                                              values, linear, updates);
    rewriter.replaceOp(op, reshapeTo(scattered, resultTy.getShape(), rewriter));
    return success();
  }
};

// Integers mask with a value below their own range, so the rounds run one
// width up to have somewhere to put it. i64 has no wider type to move to.
constexpr unsigned kMaxTopKIntWidth = 32;

// The element type the rounds run in: integers widen, floats stay put.
Type topKRoundElementType(Builder &builder, Type elemType) {
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return builder.getIntegerType(intTy.getWidth() * 2);
  return elemType;
}

// A value the input cannot hold, so masking a position with it takes that
// position out of every later round.
//
// The obvious choices do not work. Negative infinity and the signed minimum
// are both ordinary input values, and masking with one of those leaves the
// position tied with its own mask: an input of [-inf, -inf] would report
// index 0 twice instead of 0 and 1.
//
// Floats get that separation from NaN, which the rounds already drop on both
// sides because they reduce with NanPropagationMode::IGNORE -- no finite or
// infinite input can imitate a lane that is not there. Integers have no such
// lane, so they run in a wider type (topKRoundElementType) and mask one below
// the range the input arrived in, which is unreachable by construction.
Value createLosingSentinel(ConversionPatternRewriter &rewriter, Location loc,
                           RankedTensorType roundType, Type inputElemType) {
  Type elemType = roundType.getElementType();
  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    APFloat nan = APFloat::getNaN(floatTy.getFloatSemantics());
    return tosa::ConstOp::create(
        rewriter, loc, roundType,
        DenseElementsAttr::get(roundType, rewriter.getFloatAttr(floatTy, nan)));
  }
  unsigned inputWidth = cast<IntegerType>(inputElemType).getWidth();
  unsigned roundWidth = cast<IntegerType>(elemType).getWidth();
  APInt below = APInt::getSignedMinValue(inputWidth).sext(roundWidth) - 1;
  return createSplatInt(rewriter, loc, roundType, below.getSExtValue());
}

// [0, 1, ..., extent-1] along `axis`, size 1 elsewhere so it broadcasts over
// the whole tensor.
Value createAxisIota(ConversionPatternRewriter &rewriter, Location loc,
                     ArrayRef<int64_t> shape, int64_t axis, Type elemType) {
  SmallVector<int64_t> iotaShape(shape.size(), 1);
  iotaShape[axis] = shape[axis];
  auto type = RankedTensorType::get(iotaShape, elemType);
  unsigned width = cast<IntegerType>(elemType).getIntOrFloatBitWidth();
  SmallVector<APInt> steps;
  for (int64_t i = 0; i < shape[axis]; ++i)
    steps.push_back(APInt(width, i));
  return tosa::ConstOp::create(rewriter, loc, type,
                               DenseElementsAttr::get(type, steps));
}

// Each round of the TopK expansion costs a reduce_max, an argmax, a compare
// and a select, so an unbounded K would unroll into an unusable kernel.
constexpr int64_t kMaxTopKUnroll = 16;

// The capability gate for TopK: a form this pass cannot express stays a hip op
// rather than failing the conversion. Shape disagreements are deliberately not
// listed, so those remain hard failures inside the pattern.
bool isTosaExpressibleTopK(TopKOp op) {
  if (op.getNumResults() != 2)
    return false;
  auto xTy = dyn_cast<RankedTensorType>(op.getX().getType());
  auto valuesTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!xTy || !valuesTy || !xTy.hasStaticShape() || !valuesTy.hasStaticShape())
    return false;

  Type elemType = xTy.getElementType();
  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    if (!floatTy.isF32() && !floatTy.isF16() && !floatTy.isBF16())
      return false;
  } else if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    // Smallest-first negates the input, which an integer range cannot take.
    if (!intTy.isSignless() || !op.getLargest())
      return false;
    // The mask needs a value below the input range, which only exists if
    // there is a wider type to run the rounds in.
    if (intTy.getWidth() > kMaxTopKIntWidth)
      return false;
  } else {
    return false;
  }

  int64_t rank = xTy.getRank();
  int64_t axis = op.getAxis();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return false;
  // A result of the wrong rank is a shape disagreement rather than a
  // capability limit, so it belongs to the pattern's hard failures. Claim it
  // here without reading K, which would index past the end of that shape.
  if (valuesTy.getRank() != rank)
    return true;
  int64_t k = valuesTy.getDimSize(axis);
  return k >= 1 && k <= kMaxTopKUnroll;
}

// ONNX TopK, expanded as K rounds of "take the maximum, then mask it out so
// the next round finds the runner-up". TOSA has no sort and no top-k op;
// tosa.argmax and tosa.reduce_max are its only order-aware operations, so
// there is no shorter shape for this.
//
// The mask compares positions, not values. Masking everything equal to the
// round's maximum would erase both halves of a tie, so an input holding two
// equal maxima would report one of them and then skip to the third element
// where ONNX wants both. Comparing an iota against the round's argmax removes
// exactly one element, because argmax names exactly one position.
//
// K is read from the result shape, not from the `k` operand: that operand is a
// runtime tensor, while the values result is K-wide along `axis` by
// construction.
struct TopKConverter final : public OpConversionPattern<TopKOp> {
  using OpConversionPattern<TopKOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TopKOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 2)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto xTy = dyn_cast<RankedTensorType>(adaptor.getX().getType());
    auto valuesTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto indicesTy = dyn_cast<RankedTensorType>(op.getResult(1).getType());
    if (!xTy || !valuesTy || !indicesTy || !xTy.hasStaticShape() ||
        !valuesTy.hasStaticShape() || !indicesTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "expected static ranked input and results");
    if (valuesTy.getElementType() != xTy.getElementType())
      return rewriter.notifyMatchFailure(
          op, "values must carry the input element type");
    if (!isa<IntegerType>(indicesTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "indices must be integers");
    if (valuesTy.getShape() != indicesTy.getShape())
      return rewriter.notifyMatchFailure(
          op, "values and indices must have one shape");

    Type elemType = xTy.getElementType();
    // TOSA has no f64 tensor type, and an integer has to be signless to reduce.
    if (auto floatTy = dyn_cast<FloatType>(elemType)) {
      if (!floatTy.isF32() && !floatTy.isF16() && !floatTy.isBF16())
        return rewriter.notifyMatchFailure(op, "unsupported float width");
    } else if (auto intTy = dyn_cast<IntegerType>(elemType)) {
      if (!intTy.isSignless())
        return rewriter.notifyMatchFailure(op, "expected a signless integer");
      if (intTy.getWidth() > kMaxTopKIntWidth)
        return rewriter.notifyMatchFailure(
            op, "integer is too wide to leave room for the round mask");
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    int64_t rank = xTy.getRank();
    // Every shape check below indexes the results by an axis taken from the
    // input, so the ranks have to agree before any of them runs.
    if (valuesTy.getRank() != rank)
      return rewriter.notifyMatchFailure(op,
                                         "results must have the input's rank");
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");
    for (int64_t i = 0; i < rank; ++i)
      if (i != axis && xTy.getDimSize(i) != valuesTy.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "results disagree with the input off the selected axis");

    int64_t extent = xTy.getDimSize(axis);
    int64_t k = valuesTy.getDimSize(axis);
    if (k < 1 || k > extent)
      return rewriter.notifyMatchFailure(
          op, "K does not fit within the selected axis");
    if (k > kMaxTopKUnroll)
      return rewriter.notifyMatchFailure(
          op, "K is too wide to unroll into repeated tosa.argmax rounds");

    // Smallest-first would need an argmin, which TOSA does not have, so the
    // input is negated and the same largest-first rounds run. Negating the
    // minimum of an integer range overflows, so integers keep to largest.
    bool largest = op.getLargest();
    if (!largest && !isa<FloatType>(elemType))
      return rewriter.notifyMatchFailure(
          op, "smallest-first needs a negate the integer range cannot take");

    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(axis));

    // The rounds run in roundElem, which is elemType for floats and one width
    // up for integers so the mask has somewhere below the input range to sit.
    Type roundElem = topKRoundElementType(rewriter, elemType);
    auto roundTy = RankedTensorType::get(xTy.getShape(), roundElem);

    // A round reduces the axis to one element; argmax drops it entirely.
    SmallVector<int64_t> sliceShape(xTy.getShape());
    sliceShape[axis] = 1;
    SmallVector<int64_t> argMaxShape(xTy.getShape());
    argMaxShape.erase(argMaxShape.begin() + axis);
    auto valueSliceTy = RankedTensorType::get(sliceShape, roundElem);
    auto argMaxTy = RankedTensorType::get(argMaxShape, i32);
    auto maskTy = RankedTensorType::get(xTy.getShape(), rewriter.getI1Type());

    Value cur = emitTosaCast(rewriter, loc, adaptor.getX(), roundElem);
    if (!largest)
      cur = tosa::NegateOp::create(rewriter, loc, roundTy, cur);
    Value iota = createAxisIota(rewriter, loc, xTy.getShape(), axis, i32);
    Value sentinel = createLosingSentinel(rewriter, loc, roundTy, elemType);

    SmallVector<Value> valueSlices, indexSlices;
    for (int64_t round = 0; round < k; ++round) {
      // A NaN is ignored rather than propagated, so it never takes a top slot
      // from a real value. ONNX leaves this unspecified.
      Value roundValue = tosa::ReduceMaxOp::create(
          rewriter, loc, valueSliceTy, cur, static_cast<uint32_t>(axis),
          tosa::NanPropagationMode::IGNORE);
      Value roundIndex = tosa::ArgMaxOp::create(
          rewriter, loc, argMaxTy, cur, static_cast<uint32_t>(axis),
          tosa::NanPropagationMode::IGNORE);
      roundIndex = reshapeTo(roundIndex, sliceShape, rewriter);
      valueSlices.push_back(roundValue);
      indexSlices.push_back(roundIndex);

      // The final round leaves nothing to mask for.
      if (round + 1 == k)
        break;
      Value taken =
          tosa::EqualOp::create(rewriter, loc, maskTy, iota, roundIndex);
      cur =
          tosa::SelectOp::create(rewriter, loc, roundTy, taken, sentinel, cur);
    }

    // The rounds already run largest first, which is what sorted=true asks
    // for; sorted=false accepts any order, so neither needs extra work.
    auto roundValuesTy = RankedTensorType::get(valuesTy.getShape(), roundElem);
    Value values = valueSlices.front();
    Value indices = indexSlices.front();
    if (k > 1) {
      values = tosa::ConcatOp::create(rewriter, loc, roundValuesTy, valueSlices,
                                      axisAttr);
      indices = tosa::ConcatOp::create(
          rewriter, loc, RankedTensorType::get(valuesTy.getShape(), i32),
          indexSlices, axisAttr);
    }
    if (!largest)
      values = tosa::NegateOp::create(rewriter, loc, roundValuesTy, values);
    // A selected value is one the input held, so narrowing back to the input
    // width is exact; only the mask ever needed the extra room.
    values = emitTosaCast(rewriter, loc, values, elemType);
    indices = emitTosaCast(rewriter, loc, indices, indicesTy.getElementType());
    rewriter.replaceOp(op, {values, indices});
    return success();
  }
};

// Packed uint8 int4: low nibble is the first value, high nibble the second.
// Cast to i32 so nibble extract is an unsigned bit pattern, then interleave.
Value unpackInt4LastDim(Value packed, ConversionPatternRewriter &rewriter,
                        Location loc) {
  Type i32 = rewriter.getI32Type();
  Value asI32 = emitTosaCast(rewriter, loc, packed, i32);
  auto i32Ty = cast<RankedTensorType>(asI32.getType());
  Value mask = createSplatInt(rewriter, loc, i32Ty, 0x0F);
  Value shift = createSplatInt(rewriter, loc, i32Ty, 4);
  Value lo = tosa::BitwiseAndOp::create(rewriter, loc, i32Ty, asI32, mask);
  // tosa.cast sign-extends signless storage, so for a byte >= 0x80 the shift
  // pulls copies of the sign bit down into the high nibble. Mask after
  // shifting; this is a no-op for ui8 operands, which zero-extend instead.
  Value hi =
      tosa::LogicalRightShiftOp::create(rewriter, loc, i32Ty, asI32, shift);
  hi = tosa::BitwiseAndOp::create(rewriter, loc, i32Ty, hi, mask);

  SmallVector<int64_t> unsqueeze(i32Ty.getShape().begin(),
                                 i32Ty.getShape().end());
  unsqueeze.push_back(1);
  Value loU = reshapeTo(lo, unsqueeze, rewriter);
  Value hiU = reshapeTo(hi, unsqueeze, rewriter);

  SmallVector<int64_t> catShape = unsqueeze;
  catShape.back() = 2;
  auto catTy = RankedTensorType::get(catShape, i32);
  int32_t axis = static_cast<int32_t>(catShape.size() - 1);
  Value cat = tosa::ConcatOp::create(rewriter, loc, catTy, ValueRange{loU, hiU},
                                     rewriter.getI32IntegerAttr(axis));

  SmallVector<int64_t> flat(i32Ty.getShape().begin(), i32Ty.getShape().end());
  flat.back() *= 2;
  return reshapeTo(cat, flat, rewriter);
}

Value sliceLastDimTo(Value input, int64_t extent,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto ty = cast<RankedTensorType>(input.getType());
  if (ty.getShape().back() == extent)
    return input;
  SmallVector<int64_t> starts(ty.getRank(), 0);
  SmallVector<int64_t> sizes(ty.getShape().begin(), ty.getShape().end());
  sizes.back() = extent;
  return sliceOffsetSize(input, starts, sizes, rewriter, loc);
}

// Per-block [N, k_blocks] -> [N, K] by repeating each block along K.
Value broadcastBlocksAlongK(Value perBlock, int64_t n, int64_t k,
                            int64_t kBlocks, int64_t blockSize,
                            ConversionPatternRewriter &rewriter, Location loc) {
  Value x = reshapeTo(perBlock, {n, kBlocks, 1}, rewriter);
  x = tileMultiples(x, {1, 1, blockSize}, {n, kBlocks, blockSize}, rewriter,
                    loc);
  x = reshapeTo(x, {n, kBlocks * blockSize}, rewriter);
  return sliceLastDimTo(x, k, rewriter, loc);
}

Value emitUnbatchedMatmul(Value a, Value b, RankedTensorType resultType,
                          ConversionPatternRewriter &rewriter, Location loc) {
  auto aType = cast<RankedTensorType>(a.getType());
  auto bType = cast<RankedTensorType>(b.getType());
  int64_t k = aType.getShape().back();
  int64_t collapsedM = 1;
  for (int64_t d : aType.getShape().drop_back())
    collapsedM *= d;
  int64_t n = bType.getShape().back();
  Value a3 = reshapeTo(a, {1, collapsedM, k}, rewriter);
  Value b3 = reshapeTo(b, {1, k, n}, rewriter);
  auto matmulType = resultType.clone({1, collapsedM, n});
  Value matmul =
      tosa::MatMulOp::create(rewriter, loc, matmulType, a3, b3).getResult();
  return reshapeTo(matmul, resultType.getShape(), rewriter);
}

// hip.matmul_nbits -> unpack int4 B, block-dequant, transpose, tosa.matmul.

struct MatMulNBitsConverter final
    : public OpConversionPattern<hip::MatMulNBitsOp> {
  using OpConversionPattern<hip::MatMulNBitsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::MatMulNBitsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    if (adaptor.getGIdx())
      return rewriter.notifyMatchFailure(op, "g_idx is not supported");
    if (op.getBits() != 4)
      return rewriter.notifyMatchFailure(op, "only bits=4 is supported");

    int64_t k = op.getK();
    int64_t n = op.getN();
    int64_t blockSize = op.getBlockSize();
    if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0)
      return rewriter.notifyMatchFailure(
          op, "block_size must be a power of 2 and >= 16");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (!isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(op, "result must be float");

    auto aType = dyn_cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = dyn_cast<RankedTensorType>(adaptor.getB().getType());
    auto scaleType = dyn_cast<RankedTensorType>(adaptor.getScales().getType());
    if (!aType || !aType.hasStaticShape() || !bType ||
        !bType.hasStaticShape() || !scaleType || !scaleType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "operands not static ranked");
    if (aType.getRank() < 1 || aType.getShape().back() != k)
      return rewriter.notifyMatchFailure(op, "A trailing dim must be K");
    if (resultType.getShape().back() != n)
      return rewriter.notifyMatchFailure(op, "result trailing dim must be N");
    if (!isa<FloatType>(scaleType.getElementType()))
      return rewriter.notifyMatchFailure(op, "scales must be float");

    int64_t kBlocks = (k + blockSize - 1) / blockSize;
    int64_t blobSize = blockSize * op.getBits() / 8;
    if (bType.getRank() != 3 || bType.getDimSize(0) != n ||
        bType.getDimSize(1) != kBlocks || bType.getDimSize(2) != blobSize)
      return rewriter.notifyMatchFailure(
          op, "B must be packed [N, k_blocks, blob_size]");
    if (scaleType.getNumElements() != n * kBlocks)
      return rewriter.notifyMatchFailure(op, "scales must have N * k_blocks");

    Location loc = op.getLoc();
    Value packed = reshapeTo(adaptor.getB(), {n, kBlocks * blobSize}, rewriter);
    Value unpacked = unpackInt4LastDim(packed, rewriter, loc);
    unpacked = sliceLastDimTo(unpacked, k, rewriter, loc);

    Type computeElem = resultType.getElementType();
    Value scales =
        emitTosaCast(rewriter, loc, adaptor.getScales(), computeElem);
    scales =
        broadcastBlocksAlongK(scales, n, k, kBlocks, blockSize, rewriter, loc);

    Value shifted = emitTosaCast(rewriter, loc, unpacked, computeElem);
    auto nkFloat = RankedTensorType::get({n, k}, computeElem);

    Value zp = adaptor.getZeroPoints();
    if (!zp) {
      auto zpTy = RankedTensorType::get({1, 1}, computeElem);
      zp = createSplatFloat(rewriter, loc, zpTy, 8.0);
    } else {
      auto zpTy = dyn_cast<RankedTensorType>(zp.getType());
      if (!zpTy || !zpTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "zero_points not static ranked");
      if (isa<FloatType>(zpTy.getElementType())) {
        if (zpTy.getNumElements() != n * kBlocks)
          return rewriter.notifyMatchFailure(op,
                                             "fp zp must have N * k_blocks");
        zp = emitTosaCast(rewriter, loc, zp, computeElem);
        zp = broadcastBlocksAlongK(zp, n, k, kBlocks, blockSize, rewriter, loc);
      } else if (isa<IntegerType>(zpTy.getElementType())) {
        // zp_elem_size=1 is ONNX's packed nibble stream
        // [N, ceil(k_blocks/2)], not "one raw zp per block". Element count
        // cannot tell those apart when k_blocks==1 (both are [N,1]).
        if (op.getZpElemSize() == 1) {
          if (n == 0 || zpTy.getNumElements() % n != 0)
            return rewriter.notifyMatchFailure(op,
                                               "packed zp shape is invalid");
          zp = reshapeTo(zp, {n, zpTy.getNumElements() / n}, rewriter);
          zp = unpackInt4LastDim(zp, rewriter, loc);
          zp = sliceLastDimTo(zp, kBlocks, rewriter, loc);
        } else {
          if (zpTy.getNumElements() != n * kBlocks)
            return rewriter.notifyMatchFailure(op, "zp must have N * k_blocks");
          zp = reshapeTo(zp, {n, kBlocks}, rewriter);
        }
        zp = broadcastBlocksAlongK(zp, n, k, kBlocks, blockSize, rewriter, loc);
        zp = emitTosaCast(rewriter, loc, zp, computeElem);
      } else {
        return rewriter.notifyMatchFailure(op, "unsupported zero_points type");
      }
    }

    shifted = tosa::SubOp::create(rewriter, loc, nkFloat, shifted, zp);
    Value weight = emitTosaMul(rewriter, loc, shifted, scales, nkFloat);
    Value weightT = transposePerm(weight, {1, 0}, rewriter, loc);

    Value y =
        emitUnbatchedMatmul(adaptor.getA(), weightT, resultType, rewriter, loc);
    if (Value bias = adaptor.getBias()) {
      auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasTy || !biasTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "bias not static ranked");
      if (!isa<FloatType>(biasTy.getElementType()))
        return rewriter.notifyMatchFailure(op, "bias must be float");
      bias = emitTosaCast(rewriter, loc, bias, computeElem);
      SmallVector<int64_t> biasShape(resultType.getRank(), 1);
      biasShape.back() = n;
      if (biasTy.getNumElements() != n)
        return rewriter.notifyMatchFailure(op, "bias must have N elements");
      bias = reshapeTo(bias, biasShape, rewriter);
      y = tosa::AddOp::create(rewriter, loc, resultType, y, bias);
    }

    rewriter.replaceOp(op, y);
    return success();
  }
};

// Shape and storage plan for hip.gather_block_quantized. The legality gate and
// the rewrite both derive it from the same function so the set of shapes the
// pass claims cannot drift from the set it can actually emit.
struct GbqPlan {
  int64_t vocab;       // gathered extent of `data` (axis 0)
  int64_t mid;         // product of the dims between gather and quantize axes
  int64_t dataLast;    // storage extent of the quantized axis (bytes)
  int64_t scalesLast;  // per-block extent of the quantized axis
  int64_t logicalLast; // scalesLast * block_size
  int64_t rows;        // indices count * mid
  int64_t w;           // indices count
  int64_t bits;
  int64_t blockSize;
  bool isUnsigned;
};

std::optional<GbqPlan>
planGatherBlockQuantized(hip::GatherBlockQuantizedOp op) {
  if (op.getNumResults() != 1)
    return std::nullopt;

  auto dataTy = dyn_cast<RankedTensorType>(op.getData().getType());
  auto indicesTy = dyn_cast<RankedTensorType>(op.getIndices().getType());
  auto scalesTy = dyn_cast<RankedTensorType>(op.getScales().getType());
  auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!dataTy || !indicesTy || !scalesTy || !resultTy ||
      !dataTy.hasStaticShape() || !indicesTy.hasStaticShape() ||
      !scalesTy.hasStaticShape() || !resultTy.hasStaticShape())
    return std::nullopt;

  GbqPlan plan;
  plan.bits = op.getBits();
  if (plan.bits != 4 && plan.bits != 8)
    return std::nullopt;

  plan.blockSize = op.getBlockSize();
  if (plan.blockSize < 16 || (plan.blockSize & (plan.blockSize - 1)) != 0)
    return std::nullopt;

  int64_t rank = dataTy.getRank();
  if (rank < 2 || scalesTy.getRank() != rank)
    return std::nullopt;

  // gather_axis == 0 is what ONNX mandates for uint8 data and is the quantized
  // embedding lookup the op exists for. quantize_axis must be the trailing
  // axis, which is where sub-byte values are packed.
  int64_t gatherAxis = op.getGatherAxis();
  if (gatherAxis < 0)
    gatherAxis += rank;
  int64_t quantizeAxis = op.getQuantizeAxis();
  if (quantizeAxis < 0)
    quantizeAxis += rank;
  if (gatherAxis != 0 || quantizeAxis != rank - 1)
    return std::nullopt;

  auto dataElem = dyn_cast<IntegerType>(dataTy.getElementType());
  auto idxElem = dyn_cast<IntegerType>(indicesTy.getElementType());
  if (!dataElem || dataElem.getWidth() != 8 || !idxElem ||
      !(idxElem.isSignlessInteger(32) || idxElem.isSignedInteger(32) ||
        idxElem.isSignlessInteger(64) || idxElem.isSignedInteger(64)) ||
      !isa<FloatType>(scalesTy.getElementType()))
    return std::nullopt;

  // TOSA has no f64 tensor type; decline rather than emit unverifiable ops.
  Type outElem = resultTy.getElementType();
  if (!outElem.isF32() && !outElem.isF16() && !outElem.isBF16())
    return std::nullopt;

  // The op contract is that output takes its element type from scales, but
  // nothing verifies it, so the mismatch has to be declined here. Letting it
  // through would not be harmless: the converter casts scales to the output
  // type, which silently rewrites the arithmetic when the two are merely
  // different, and for f64 scales the gather is emitted on a tensor TOSA
  // cannot represent before that cast is ever reached.
  if (scalesTy.getElementType() != outElem)
    return std::nullopt;

  plan.vocab = dataTy.getDimSize(0);
  if (scalesTy.getDimSize(0) != plan.vocab)
    return std::nullopt;
  plan.mid = 1;
  for (int64_t i = 1; i < rank - 1; ++i) {
    if (scalesTy.getDimSize(i) != dataTy.getDimSize(i))
      return std::nullopt;
    plan.mid *= dataTy.getDimSize(i);
  }

  plan.dataLast = dataTy.getDimSize(rank - 1);
  plan.scalesLast = scalesTy.getDimSize(rank - 1);
  plan.logicalLast = plan.scalesLast * plan.blockSize;
  if (plan.scalesLast <= 0 || plan.vocab <= 0)
    return std::nullopt;

  // The runtime always unpacks two nibbles per byte for bits == 4, so the byte
  // extent has to be exactly half the logical extent. bits == 8 is unpacked.
  if (plan.bits == 4) {
    if (plan.logicalLast != plan.dataLast * 2)
      return std::nullopt;
  } else if (plan.logicalLast != plan.dataLast) {
    return std::nullopt;
  }

  // zero_points carries one value per block in its own byte. The packed-nibble
  // form is declined: with scalesLast == 1 it is indistinguishable from the
  // per-byte form, the same ambiguity MatMulNBits resolves via zp_elem_size.
  if (Value zp = op.getZeroPoints()) {
    auto zpTy = dyn_cast<RankedTensorType>(zp.getType());
    if (!zpTy || !zpTy.hasStaticShape() ||
        !isa<IntegerType>(zpTy.getElementType()) ||
        zpTy.getShape() != scalesTy.getShape())
      return std::nullopt;
  }

  plan.w = indicesTy.getNumElements();
  plan.rows = plan.w * plan.mid;

  // Output rank is q + (r - 1) over the logical, not the packed, extent.
  SmallVector<int64_t> expected(indicesTy.getShape().begin(),
                                indicesTy.getShape().end());
  for (int64_t i = 1; i < rank - 1; ++i)
    expected.push_back(dataTy.getDimSize(i));
  expected.push_back(plan.logicalLast);
  if (resultTy.getShape() != ArrayRef<int64_t>(expected))
    return std::nullopt;

  plan.isUnsigned = op.getUnsignedQuantStorage() ||
                    dataTy.getElementType().isUnsignedInteger();
  return plan;
}

bool isTosaExpressibleGatherBlockQuantized(hip::GatherBlockQuantizedOp op) {
  return planGatherBlockQuantized(op).has_value();
}

// Gather `perBlock`-shaped side data (scales / zero points) with the same
// index vector that drives the data gather, then collapse to [rows, trailing].
Value gatherRows(Value table, Value indices, const GbqPlan &plan,
                 int64_t trailing, ConversionPatternRewriter &rewriter,
                 Location loc) {
  auto elemTy = cast<RankedTensorType>(table.getType()).getElementType();
  int64_t rowWidth = plan.mid * trailing;
  Value values = reshapeTo(table, {1, plan.vocab, rowWidth}, rewriter);
  auto gatheredTy = RankedTensorType::get({1, plan.w, rowWidth}, elemTy);
  Value gathered =
      tosa::GatherOp::create(rewriter, loc, gatheredTy, values, indices);
  return reshapeTo(gathered, {plan.rows, trailing}, rewriter);
}

// hip.gather_block_quantized -> tosa.gather on the packed rows + block dequant.
//
// Dequantization is elementwise, so gathering first and dequantizing second
// matches dequantizing the whole table and then gathering -- and it only ever
// touches the rows the indices name, which is the point of the fused op.
// `scales` and `zero_points` share `data`'s layout on every axis except
// quantize_axis, so one index vector drives all three gathers.
struct GatherBlockQuantizedConverter final
    : public OpConversionPattern<hip::GatherBlockQuantizedOp> {
  using OpConversionPattern<hip::GatherBlockQuantizedOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::GatherBlockQuantizedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<GbqPlan> maybePlan = planGatherBlockQuantized(op);
    if (!maybePlan)
      return rewriter.notifyMatchFailure(
          op, "unsupported gather_block_quantized configuration");
    const GbqPlan &plan = *maybePlan;

    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Type computeElem = resultTy.getElementType();
    Type i32 = rewriter.getI32Type();

    // ONNX permits indices in [-vocab, vocab-1]; TOSA requires them in range.
    auto idxTy = RankedTensorType::get({1, plan.w}, i32);
    Value indices = emitTosaCast(rewriter, loc, adaptor.getIndices(), i32);
    indices = reshapeTo(indices, {1, plan.w}, rewriter);
    Value zero = createSplatInt(rewriter, loc, idxTy, 0);
    Value extent = createSplatInt(rewriter, loc, idxTy, plan.vocab);
    Value isNegative = tosa::GreaterOp::create(
        rewriter, loc, RankedTensorType::get({1, plan.w}, rewriter.getI1Type()),
        zero, indices);
    Value wrapped = tosa::AddOp::create(rewriter, loc, idxTy, indices, extent);
    indices = tosa::SelectOp::create(rewriter, loc, idxTy, isNegative, wrapped,
                                     indices);

    // Gather the packed rows, then widen them to logical values.
    Value q = gatherRows(adaptor.getData(), indices, plan, plan.dataLast,
                         rewriter, loc);
    if (plan.bits == 4)
      q = unpackInt4LastDim(q, rewriter, loc);
    else
      q = emitTosaCast(rewriter, loc, q, i32);

    auto qIntTy = RankedTensorType::get({plan.rows, plan.logicalLast}, i32);
    if (plan.isUnsigned) {
      // unpackInt4LastDim already masks both nibbles; only whole bytes need
      // clamping back out of tosa.cast's sign extension.
      if (plan.bits == 8) {
        Value mask = createSplatInt(rewriter, loc, qIntTy, 0xFF);
        q = tosa::BitwiseAndOp::create(rewriter, loc, qIntTy, q, mask);
      }
    } else if (plan.bits == 4) {
      // Sign-extend the nibble. (v ^ 8) - 8 maps [0,15] onto [-8,7] and is the
      // branch-free equivalent of the runtime's shl-then-arithmetic-shr.
      Value eight = createSplatInt(rewriter, loc, qIntTy, 8);
      q = tosa::BitwiseXorOp::create(rewriter, loc, qIntTy, q, eight);
      q = tosa::SubOp::create(rewriter, loc, qIntTy, q, eight);
    }

    // Per-block scales follow the same rows, then repeat across their block.
    Value scales = gatherRows(adaptor.getScales(), indices, plan,
                              plan.scalesLast, rewriter, loc);
    scales = emitTosaCast(rewriter, loc, scales, computeElem);
    scales =
        broadcastBlocksAlongK(scales, plan.rows, plan.logicalLast,
                              plan.scalesLast, plan.blockSize, rewriter, loc);

    auto computeTy =
        RankedTensorType::get({plan.rows, plan.logicalLast}, computeElem);
    Value zeroPoint;
    if (Value zp = adaptor.getZeroPoints()) {
      zp = gatherRows(zp, indices, plan, plan.scalesLast, rewriter, loc);
      zp = emitTosaCast(rewriter, loc, zp, i32);
      auto zpIntTy = RankedTensorType::get({plan.rows, plan.scalesLast}, i32);
      if (plan.isUnsigned) {
        Value mask = createSplatInt(rewriter, loc, zpIntTy,
                                    plan.bits == 4 ? 0x0F : 0xFF);
        zp = tosa::BitwiseAndOp::create(rewriter, loc, zpIntTy, zp, mask);
      }
      zp =
          broadcastBlocksAlongK(zp, plan.rows, plan.logicalLast,
                                plan.scalesLast, plan.blockSize, rewriter, loc);
      zeroPoint = emitTosaCast(rewriter, loc, zp, computeElem);
    } else {
      // ONNX defaults: 0 for signed storage, 2^(bits-1) for unsigned. Leaving
      // this at 0 for uint4 shifts every value by +8*scale.
      double defaultZp =
          plan.isUnsigned ? static_cast<double>(1LL << (plan.bits - 1)) : 0.0;
      zeroPoint = createSplatFloat(
          rewriter, loc, RankedTensorType::get({1, 1}, computeElem), defaultZp);
    }

    Value shifted = tosa::SubOp::create(
        rewriter, loc, computeTy, emitTosaCast(rewriter, loc, q, computeElem),
        zeroPoint);
    Value dequantized = emitTosaMul(rewriter, loc, shifted, scales, computeTy);
    rewriter.replaceOp(op,
                       reshapeTo(dequantized, resultTy.getShape(), rewriter));
    return success();
  }
};

Value createI32Dense(ConversionPatternRewriter &rewriter, Location loc,
                     ArrayRef<int64_t> shape, ArrayRef<int32_t> values) {
  auto ty = RankedTensorType::get(shape, rewriter.getI32Type());
  return tosa::ConstOp::create(rewriter, loc, ty,
                               DenseElementsAttr::get(ty, values));
}

Value createArangeI32(ConversionPatternRewriter &rewriter, Location loc,
                      int64_t n) {
  SmallVector<int32_t> vals;
  vals.reserve(n);
  for (int64_t i : llvm::seq<int64_t>(0, n))
    vals.push_back(static_cast<int32_t>(i));
  return createI32Dense(rewriter, loc, {n}, vals);
}

Value createSplatI32(ConversionPatternRewriter &rewriter, Location loc,
                     ArrayRef<int64_t> shape, int32_t value) {
  auto ty = RankedTensorType::get(shape, rewriter.getI32Type());
  return tosa::ConstOp::create(
      rewriter, loc, ty,
      DenseElementsAttr::get(ty, rewriter.getI32IntegerAttr(value)));
}

LogicalResult unpackToBnsh(Value input, int64_t numHeads, int64_t headDim,
                           ConversionPatternRewriter &rewriter, Location loc,
                           Operation *op, Value &out) {
  auto ty = dyn_cast<RankedTensorType>(input.getType());
  if (!ty || !ty.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
  if (ty.getRank() == 4) {
    if (ty.getDimSize(1) != numHeads || ty.getDimSize(3) != headDim)
      return rewriter.notifyMatchFailure(op, "rank-4 layout is not BNSH");
    out = input;
    return success();
  }
  if (ty.getRank() != 3)
    return rewriter.notifyMatchFailure(op, "Q/K/V must be rank 3 or 4");
  int64_t b = ty.getDimSize(0);
  int64_t s = ty.getDimSize(1);
  if (ty.getDimSize(2) != numHeads * headDim)
    return rewriter.notifyMatchFailure(op, "hidden size is not heads * dim");
  Value r = reshapeTo(input, {b, s, numHeads, headDim}, rewriter);
  out = transposePerm(r, {0, 2, 1, 3}, rewriter, loc);
  return success();
}

LogicalResult broadcastKvHeads(Value kv, int64_t numHeads, int64_t kvHeads,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Operation *op, Value &out) {
  if (numHeads == kvHeads) {
    out = kv;
    return success();
  }
  if (numHeads % kvHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "num_heads must be divisible by kv_num_heads");
  auto ty = cast<RankedTensorType>(kv.getType());
  int64_t b = ty.getDimSize(0);
  int64_t s = ty.getDimSize(2);
  int64_t d = ty.getDimSize(3);
  int64_t repeat = numHeads / kvHeads;
  Value r = reshapeTo(kv, {b, kvHeads, 1, s, d}, rewriter);
  Value t = tileMultiples(r, {1, 1, repeat, 1, 1}, {b, kvHeads, repeat, s, d},
                          rewriter, loc);
  out = reshapeTo(t, {b, numHeads, s, d}, rewriter);
  return success();
}

Value applySelectNegInf(Value scores, Value pred,
                        ConversionPatternRewriter &rewriter, Location loc) {
  auto scoresTy = cast<RankedTensorType>(scores.getType());
  Value negInf = createSplatFloat(rewriter, loc, scoresTy,
                                  -std::numeric_limits<double>::infinity());
  Value p = pred;
  Value s = scores;
  if (failed(tosa::EqualizeRanks(rewriter, loc, p, s)))
    return scores;
  return tosa::SelectOp::create(rewriter, loc, scoresTy, p, negInf, s)
      .getResult();
}

Value softmaxLastDim(Value input, Value extraDenom,
                     ConversionPatternRewriter &rewriter, Location loc) {
  auto resultType = cast<RankedTensorType>(input.getType());
  int32_t axis = static_cast<int32_t>(resultType.getRank() - 1);
  auto reducedTy = keepdimsReduceType(resultType, axis);
  IntegerAttr axisAttr = rewriter.getI32IntegerAttr(axis);
  auto rmax =
      tosa::ReduceMaxOp::create(rewriter, loc, reducedTy, input, axisAttr);
  auto sub = tosa::SubOp::create(rewriter, loc, resultType, input, rmax);
  auto exp = tosa::ExpOp::create(rewriter, loc, resultType, sub);
  Value rsum =
      tosa::ReduceSumOp::create(rewriter, loc, reducedTy, exp, axisAttr)
          .getResult();
  if (extraDenom) {
    Value denom = extraDenom;
    if (succeeded(tosa::EqualizeRanks(rewriter, loc, rsum, denom)))
      rsum = tosa::AddOp::create(rewriter, loc, reducedTy, rsum, denom)
                 .getResult();
  }
  auto rec = tosa::ReciprocalOp::create(rewriter, loc, reducedTy, rsum);
  return emitTosaMul(rewriter, loc, exp, rec, resultType);
}

Value packBnshToOutput(Value y4, RankedTensorType yType,
                       ConversionPatternRewriter &rewriter, Location loc) {
  if (yType.getRank() == 4)
    return y4;
  Value yT = transposePerm(y4, {0, 2, 1, 3}, rewriter, loc);
  return reshapeTo(yT, yType.getShape(), rewriter);
}

LogicalResult addAttentionBias(Value &scores, Value bias, RankedTensorType qkTy,
                               int64_t batch, int64_t numHeads,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Operation *op) {
  auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasTy || !biasTy.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "attention_bias must be static");
  if (biasTy.getRank() == 4) {
    int64_t b0 = biasTy.getDimSize(0);
    int64_t h0 = biasTy.getDimSize(1);
    int64_t sq = biasTy.getDimSize(2);
    int64_t skv = biasTy.getDimSize(3);
    // Keep singleton batch/head axes until they have been tiled. Flattening
    // [1, H, ...] or [B, 1, ...] first would drop the broadcast dim and
    // fail EqualizeRanks against scores [B*H, ...].
    if ((b0 != 1 && b0 != batch) || (h0 != 1 && h0 != numHeads))
      return rewriter.notifyMatchFailure(
          op, "attention_bias batch/head dims do not broadcast");
    int64_t tileB = batch / b0;
    int64_t tileH = numHeads / h0;
    if (tileB != 1 || tileH != 1)
      bias = tileMultiples(bias, {tileB, tileH, 1, 1},
                           {batch, numHeads, sq, skv}, rewriter, loc);
    bias = reshapeTo(bias, {batch * numHeads, sq, skv}, rewriter);
  }
  if (failed(tosa::EqualizeRanks(rewriter, loc, scores, bias)))
    return rewriter.notifyMatchFailure(op, "attention_bias not broadcastable");
  scores = tosa::AddOp::create(rewriter, loc, qkTy, scores, bias);
  return success();
}

Value applyCausalPrefill(Value scores, Value kIdx, bool enable, int64_t seqQ,
                         int64_t seqKv, ConversionPatternRewriter &rewriter,
                         Location loc) {
  if (!enable || seqQ <= 1)
    return scores;
  int64_t pastLen = seqKv - seqQ;
  Value qIdx = createArangeI32(rewriter, loc, seqQ);
  if (pastLen != 0) {
    Value off =
        createSplatI32(rewriter, loc, {seqQ}, static_cast<int32_t>(pastLen));
    qIdx = tosa::AddOp::create(
        rewriter, loc, RankedTensorType::get({seqQ}, rewriter.getI32Type()),
        qIdx, off);
  }
  qIdx = reshapeTo(qIdx, {1, seqQ, 1}, rewriter);
  auto predTy = RankedTensorType::get({1, seqQ, seqKv}, rewriter.getI1Type());
  Value pred = tosa::GreaterOp::create(rewriter, loc, predTy, kIdx, qIdx);
  return applySelectNegInf(scores, pred, rewriter, loc);
}

LogicalResult concatGrowingPast(Value kCur, Value vCur, Value pastK,
                                Value pastV, int64_t batch, int64_t kvHeads,
                                int64_t kDim, int64_t vDim,
                                std::optional<int64_t> presentSeqDim,
                                ConversionPatternRewriter &rewriter,
                                Location loc, Operation *op, Value &presentK,
                                Value &presentV, int64_t &seqKv) {
  presentK = kCur;
  presentV = vCur;
  seqKv = cast<RankedTensorType>(kCur.getType()).getDimSize(2);
  if (!pastK && !pastV)
    return success();
  if (!pastK || !pastV)
    return rewriter.notifyMatchFailure(op, "past K/V must be paired");
  auto pastKTy = dyn_cast<RankedTensorType>(pastK.getType());
  if (!pastKTy || !pastKTy.hasStaticShape() || pastKTy.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "past_key must be static BNSH");
  int64_t pastLen = pastKTy.getDimSize(2);
  int64_t curLen = seqKv;
  if (presentSeqDim && pastLen == *presentSeqDim && curLen > 0)
    return rewriter.notifyMatchFailure(
        op, "share-buffer present==past is unsupported");
  if (pastLen == 0)
    return success();
  seqKv = pastLen + curLen;
  auto catKTy = RankedTensorType::get(
      {batch, kvHeads, seqKv, kDim},
      cast<RankedTensorType>(kCur.getType()).getElementType());
  auto catVTy = RankedTensorType::get(
      {batch, kvHeads, seqKv, vDim},
      cast<RankedTensorType>(vCur.getType()).getElementType());
  presentK =
      tosa::ConcatOp::create(rewriter, loc, catKTy, ValueRange{pastK, kCur},
                             rewriter.getI32IntegerAttr(2));
  presentV =
      tosa::ConcatOp::create(rewriter, loc, catVTy, ValueRange{pastV, vCur},
                             rewriter.getI32IntegerAttr(2));
  return success();
}

// Look up [max_pos, half] cos/sin rows with integer position_ids [B, S].
// TOSA gather is [N,K,C] x [N,W] -> [N,W,C]; out-of-range ids are clamped
// the way the HIP RoPE kernel does.
LogicalResult gatherRopeCacheRows(Value cache, Value positionIds, int64_t batch,
                                  int64_t seqLen, int64_t half,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, Operation *op,
                                  Value &gathered) {
  auto cacheTy = dyn_cast<RankedTensorType>(cache.getType());
  auto posTy = dyn_cast<RankedTensorType>(positionIds.getType());
  if (!cacheTy || !cacheTy.hasStaticShape() || cacheTy.getRank() != 2 ||
      cacheTy.getDimSize(1) != half)
    return rewriter.notifyMatchFailure(
        op, "indexed RoPE caches must match [max_position, rotary_dim/2]");
  if (!posTy || !posTy.hasStaticShape() || posTy.getRank() != 2 ||
      posTy.getDimSize(0) != batch || posTy.getDimSize(1) != seqLen ||
      !isa<IntegerType>(posTy.getElementType()))
    return rewriter.notifyMatchFailure(
        op, "position_ids must have integer shape [batch, seq]");

  int64_t maxPos = cacheTy.getDimSize(0);
  if (maxPos <= 0)
    return rewriter.notifyMatchFailure(op, "RoPE cache must have a row");

  Value values = reshapeTo(cache, {1, maxPos, half}, rewriter);
  if (batch != 1)
    values = tileMultiples(values, {batch, 1, 1}, {batch, maxPos, half},
                           rewriter, loc);
  Value indices =
      emitTosaCast(rewriter, loc, positionIds, rewriter.getI32Type());
  indices = reshapeTo(indices, {batch, seqLen}, rewriter);

  auto indicesTy =
      RankedTensorType::get({batch, seqLen}, rewriter.getI32Type());
  Value zero = createSplatInt(rewriter, loc, indicesTy, 0);
  Value last = createSplatInt(rewriter, loc, indicesTy, maxPos - 1);
  indices = tosa::MaximumOp::create(rewriter, loc, indicesTy, indices, zero);
  indices = tosa::MinimumOp::create(rewriter, loc, indicesTy, indices, last);

  auto gatheredTy =
      RankedTensorType::get({batch, seqLen, half}, cacheTy.getElementType());
  gathered = tosa::GatherOp::create(rewriter, loc, gatheredTy, values, indices);
  return success();
}

LogicalResult applyRopeExpanded(Value &tensor, Value cos, Value sin,
                                int64_t rotaryDim, bool interleaved,
                                ConversionPatternRewriter &rewriter,
                                Location loc, Operation *op) {
  auto ty = cast<RankedTensorType>(tensor.getType());
  int64_t b = ty.getDimSize(0);
  int64_t h = ty.getDimSize(1);
  int64_t seqLen = ty.getDimSize(2);
  int64_t d = ty.getDimSize(3);
  if (rotaryDim <= 0 || rotaryDim > d || rotaryDim % 2 != 0)
    return rewriter.notifyMatchFailure(
        op, "RoPE rotary dim must be positive, even, and <= head dim");
  int64_t half = rotaryDim / 2;

  Value rotary = tensor;
  Value tail;
  if (rotaryDim != d) {
    rotary = sliceOffsetSize(tensor, {0, 0, 0, 0}, {b, h, seqLen, rotaryDim},
                             rewriter, loc);
    tail = sliceOffsetSize(tensor, {0, 0, 0, rotaryDim},
                           {b, h, seqLen, d - rotaryDim}, rewriter, loc);
  }

  Value x1, x2;
  if (!interleaved) {
    x1 = sliceOffsetSize(rotary, {0, 0, 0, 0}, {b, h, seqLen, half}, rewriter,
                         loc);
    x2 = sliceOffsetSize(rotary, {0, 0, 0, half}, {b, h, seqLen, half},
                         rewriter, loc);
  } else {
    Value r = reshapeTo(rotary, {b, h, seqLen, half, 2}, rewriter);
    x1 = sliceOffsetSize(r, {0, 0, 0, 0, 0}, {b, h, seqLen, half, 1}, rewriter,
                         loc);
    x2 = sliceOffsetSize(r, {0, 0, 0, 0, 1}, {b, h, seqLen, half, 1}, rewriter,
                         loc);
    x1 = reshapeTo(x1, {b, h, seqLen, half}, rewriter);
    x2 = reshapeTo(x2, {b, h, seqLen, half}, rewriter);
  }

  auto halfTy = cast<RankedTensorType>(x1.getType());
  Value nx2 = tosa::NegateOp::create(rewriter, loc, halfTy, x2);
  Value x1c = emitTosaMul(rewriter, loc, x1, cos, halfTy);
  Value x2s = emitTosaMul(rewriter, loc, nx2, sin, halfTy);
  Value left = tosa::AddOp::create(rewriter, loc, halfTy, x1c, x2s);
  Value x2c = emitTosaMul(rewriter, loc, x2, cos, halfTy);
  Value x1s = emitTosaMul(rewriter, loc, x1, sin, halfTy);
  Value right = tosa::AddOp::create(rewriter, loc, halfTy, x2c, x1s);

  Value rotated;
  if (!interleaved) {
    auto catTy =
        RankedTensorType::get({b, h, seqLen, rotaryDim}, ty.getElementType());
    rotated =
        tosa::ConcatOp::create(rewriter, loc, catTy, ValueRange{left, right},
                               rewriter.getI32IntegerAttr(3));
  } else {
    Value le = reshapeTo(left, {b, h, seqLen, half, 1}, rewriter);
    Value ri = reshapeTo(right, {b, h, seqLen, half, 1}, rewriter);
    auto catTy =
        RankedTensorType::get({b, h, seqLen, half, 2}, ty.getElementType());
    Value cat = tosa::ConcatOp::create(rewriter, loc, catTy, ValueRange{le, ri},
                                       rewriter.getI32IntegerAttr(4));
    rotated = reshapeTo(cat, {b, h, seqLen, rotaryDim}, rewriter);
  }
  if (tail) {
    tensor =
        tosa::ConcatOp::create(rewriter, loc, ty, ValueRange{rotated, tail},
                               rewriter.getI32IntegerAttr(3));
  } else {
    tensor = rotated;
  }
  return success();
}

LogicalResult applyRope(Value &tensor, Value cos, Value sin, Value positionIds,
                        int64_t seqLen, bool interleaved,
                        ConversionPatternRewriter &rewriter, Location loc,
                        Operation *op) {
  auto ty = cast<RankedTensorType>(tensor.getType());
  int64_t b = ty.getDimSize(0);
  int64_t h = ty.getDimSize(1);
  int64_t d = ty.getDimSize(3);
  if (d % 2 != 0)
    return rewriter.notifyMatchFailure(op, "RoPE head dim must be even");
  int64_t half = d / 2;
  auto cosTy = dyn_cast<RankedTensorType>(cos.getType());
  auto sinTy = dyn_cast<RankedTensorType>(sin.getType());
  if (!cosTy || !cosTy.hasStaticShape() || !sinTy || !sinTy.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "RoPE caches must be static");
  if (cosTy.getRank() != 2 || sinTy.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "RoPE caches must be rank 2");
  if (sinTy.getShape() != cosTy.getShape() || cosTy.getDimSize(1) != half)
    return rewriter.notifyMatchFailure(
        op, "RoPE caches must match [max_position, head_dim/2]");

  Value c, s;
  if (positionIds) {
    if (failed(gatherRopeCacheRows(cos, positionIds, b, seqLen, half, rewriter,
                                   loc, op, c)) ||
        failed(gatherRopeCacheRows(sin, positionIds, b, seqLen, half, rewriter,
                                   loc, op, s)))
      return failure();
    c = reshapeTo(c, {b, 1, seqLen, half}, rewriter);
    s = reshapeTo(s, {b, 1, seqLen, half}, rewriter);
    c = tileMultiples(c, {1, h, 1, 1}, {b, h, seqLen, half}, rewriter, loc);
    s = tileMultiples(s, {1, h, 1, 1}, {b, h, seqLen, half}, rewriter, loc);
  } else {
    if (cosTy.getDimSize(0) < seqLen)
      return rewriter.notifyMatchFailure(op, "RoPE cache shorter than seq");
    c = sliceOffsetSize(cos, {0, 0}, {seqLen, half}, rewriter, loc);
    s = sliceOffsetSize(sin, {0, 0}, {seqLen, half}, rewriter, loc);
    c = reshapeTo(c, {1, 1, seqLen, half}, rewriter);
    s = reshapeTo(s, {1, 1, seqLen, half}, rewriter);
    c = tileMultiples(c, {b, h, 1, 1}, {b, h, seqLen, half}, rewriter, loc);
    s = tileMultiples(s, {b, h, 1, 1}, {b, h, seqLen, half}, rewriter, loc);
  }
  return applyRopeExpanded(tensor, c, s, d, interleaved, rewriter, loc, op);
}

struct RopeConverter final : public OpConversionPattern<RopeOp> {
  using OpConversionPattern<RopeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RopeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto inputTy = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto cosTy = dyn_cast<RankedTensorType>(adaptor.getCosCache().getType());
    auto sinTy = dyn_cast<RankedTensorType>(adaptor.getSinCache().getType());
    if (!inputTy || !resultTy || !cosTy || !sinTy ||
        !inputTy.hasStaticShape() || !resultTy.hasStaticShape() ||
        !cosTy.hasStaticShape() || !sinTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");
    if (inputTy != resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "RoPE input/result types must match");
    if (!isa<FloatType>(inputTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "RoPE input must be floating");
    if (cosTy.getElementType() != inputTy.getElementType() ||
        sinTy.getElementType() != inputTy.getElementType())
      return rewriter.notifyMatchFailure(op,
                                         "RoPE cache types must match input");
    if (inputTy.getRank() != 3 && inputTy.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "RoPE input must be BSH or BNSH");

    int64_t b = inputTy.getDimSize(0);
    int64_t seqLen = inputTy.getDimSize(inputTy.getRank() == 3 ? 1 : 2);
    int64_t numHeads = op.getNumHeads();
    int64_t headDim;
    if (inputTy.getRank() == 3) {
      if (numHeads <= 0 || inputTy.getDimSize(2) % numHeads != 0)
        return rewriter.notifyMatchFailure(
            op, "BSH hidden size must be divisible by num_heads");
      headDim = inputTy.getDimSize(2) / numHeads;
    } else {
      if (numHeads <= 0)
        numHeads = inputTy.getDimSize(1);
      if (numHeads != inputTy.getDimSize(1))
        return rewriter.notifyMatchFailure(
            op, "num_heads disagrees with BNSH input");
      headDim = inputTy.getDimSize(3);
    }

    int64_t rotaryDim = op.getRotaryEmbeddingDim();
    if (rotaryDim == 0)
      rotaryDim = headDim;
    if (rotaryDim <= 0 || rotaryDim > headDim || rotaryDim % 2 != 0)
      return rewriter.notifyMatchFailure(
          op, "rotary dim must be positive, even, and <= head dim");
    int64_t half = rotaryDim / 2;
    if (op.getInterleaved() != 0 && op.getInterleaved() != 1)
      return rewriter.notifyMatchFailure(op, "interleaved must be 0 or 1");

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    if (inputTy.getRank() == 3) {
      input = reshapeTo(input, {b, seqLen, numHeads, headDim}, rewriter);
      input = transposePerm(input, {0, 2, 1, 3}, rewriter, loc);
    }

    Value cos = adaptor.getCosCache();
    Value sin = adaptor.getSinCache();
    if (Value positionIds = adaptor.getPositionIds()) {
      if (failed(gatherRopeCacheRows(cos, positionIds, b, seqLen, half,
                                     rewriter, loc, op, cos)) ||
          failed(gatherRopeCacheRows(sin, positionIds, b, seqLen, half,
                                     rewriter, loc, op, sin)))
        return failure();
    } else {
      if (cosTy.getRank() != 3 || sinTy.getRank() != 3 ||
          cosTy.getDimSize(0) != b || cosTy.getDimSize(1) != seqLen ||
          cosTy.getDimSize(2) != half || sinTy.getShape() != cosTy.getShape())
        return rewriter.notifyMatchFailure(
            op, "expanded RoPE caches must match [batch, seq, rotary_dim/2]");
    }
    cos = reshapeTo(cos, {b, 1, seqLen, half}, rewriter);
    sin = reshapeTo(sin, {b, 1, seqLen, half}, rewriter);

    if (failed(applyRopeExpanded(input, cos, sin, rotaryDim,
                                 op.getInterleaved() != 0, rewriter, loc, op)))
      return failure();

    if (inputTy.getRank() == 3) {
      input = transposePerm(input, {0, 2, 1, 3}, rewriter, loc);
      input = reshapeTo(input, inputTy.getShape(), rewriter);
    }
    rewriter.replaceOp(op, input);
    return success();
  }
};

LogicalResult dequantIfNeeded(Value &cache, Value scale, Type attnElem,
                              StringRef quantType,
                              ConversionPatternRewriter &rewriter, Location loc,
                              Operation *op) {
  if (quantType == "NONE")
    return success();
  auto ty = cast<RankedTensorType>(cache.getType());
  if (isa<FloatType>(ty.getElementType()))
    return success();
  if (!scale)
    return rewriter.notifyMatchFailure(op, "quantized cache missing scale");
  cache = emitTosaCast(rewriter, loc, cache, attnElem);
  Value s = scale;
  auto sTy = dyn_cast<RankedTensorType>(s.getType());
  if (!sTy || !sTy.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "scale must be a static tensor");
  if (sTy.getElementType() != attnElem)
    s = emitTosaCast(rewriter, loc, s, attnElem);
  if (failed(tosa::EqualizeRanks(rewriter, loc, cache, s)))
    return rewriter.notifyMatchFailure(op, "scale not broadcastable");
  auto outTy = cast<RankedTensorType>(cache.getType());
  cache = emitTosaMul(rewriter, loc, cache, s, outTy);
  return success();
}

LogicalResult quantIfNeeded(Value &cache, Value scale, RankedTensorType outTy,
                            StringRef quantType,
                            ConversionPatternRewriter &rewriter, Location loc,
                            Operation *op) {
  if (quantType == "NONE" || isa<FloatType>(outTy.getElementType()))
    return success();
  if (!scale)
    return rewriter.notifyMatchFailure(op, "quantized present missing scale");
  Value s = scale;
  auto sTy = dyn_cast<RankedTensorType>(s.getType());
  if (!sTy || !sTy.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "scale must be a static tensor");
  auto cacheTy = cast<RankedTensorType>(cache.getType());
  if (sTy.getElementType() != cacheTy.getElementType())
    s = emitTosaCast(rewriter, loc, s, cacheTy.getElementType());
  if (failed(tosa::EqualizeRanks(rewriter, loc, cache, s)))
    return rewriter.notifyMatchFailure(op, "scale not broadcastable");
  auto recTy = cast<RankedTensorType>(s.getType());
  Value inv = tosa::ReciprocalOp::create(rewriter, loc, recTy, s);
  Value scaled = emitTosaMul(rewriter, loc, cache, inv, cacheTy);
  cache = emitTosaCast(rewriter, loc, scaled, outTy.getElementType());
  return success();
}

// hip.gqa -> TOSA SDPA (unpack, optional RoPE, present concat, GQA broadcast,
// QK/PV matmul, mask/softmax, pack Y). Ctx and DPS inits are dropped.
struct GqaConverter final : public OpConversionPattern<GqaOp> {
  using OpConversionPattern<GqaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() < 3)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto yType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    auto pkType = dyn_cast<RankedTensorType>(op.getResult(1).getType());
    auto pvType = dyn_cast<RankedTensorType>(op.getResult(2).getType());
    if (!yType || !yType.hasStaticShape() || !pkType ||
        !pkType.hasStaticShape() || !pvType || !pvType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected static ranked tensors");

    // Dialect verify already restricts bit width to 4 or 8. INT4 is legal on
    // hip.gqa; this expansion only implements the INT8 / float cache path.
    if (op.getKvCacheBitWidth() != 8)
      return rewriter.notifyMatchFailure(op, "INT4 KV cache is unsupported");

    auto isFp8 = [](Type t) {
      return isa<FloatType>(t) && t.getIntOrFloatBitWidth() == 8;
    };
    if (isFp8(yType.getElementType()) || isFp8(pkType.getElementType()) ||
        isFp8(pvType.getElementType()))
      return rewriter.notifyMatchFailure(op, "FP8 GQA is unsupported");

    StringRef kQuant = op.getKQuantType();
    StringRef vQuant = op.getVQuantType();

    Location loc = op.getLoc();
    int64_t numHeads = op.getNumHeads();
    int64_t kvHeads = op.getKvNumHeads();
    if (numHeads <= 0 || kvHeads <= 0)
      return rewriter.notifyMatchFailure(op, "head counts must be positive");

    Value query = adaptor.getQuery();
    auto qInTy = dyn_cast<RankedTensorType>(query.getType());
    if (!qInTy || !qInTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "query must be a static tensor");
    if (!isa<FloatType>(qInTy.getElementType()) ||
        isFp8(qInTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "query must be f16/f32");

    bool packed = !adaptor.getKey() || !adaptor.getValue();
    int64_t batch = qInTy.getDimSize(0);
    int64_t seqQ =
        qInTy.getRank() == 4 ? qInTy.getDimSize(2) : qInTy.getDimSize(1);
    int64_t headDim = 0;
    Value qBnsh, kCur, vCur;

    if (packed) {
      if (qInTy.getRank() != 3)
        return rewriter.notifyMatchFailure(op, "packed QKV must be rank 3");
      int64_t packedHidden = qInTy.getDimSize(2);
      int64_t packedHeads = numHeads + 2 * kvHeads;
      if (packedHeads == 0 || packedHidden % packedHeads != 0)
        return rewriter.notifyMatchFailure(op,
                                           "packed QKV hidden size mismatch");
      headDim = packedHidden / packedHeads;
      Value qSlice = sliceOffsetSize(
          query, {0, 0, 0}, {batch, seqQ, numHeads * headDim}, rewriter, loc);
      Value kSlice =
          sliceOffsetSize(query, {0, 0, numHeads * headDim},
                          {batch, seqQ, kvHeads * headDim}, rewriter, loc);
      Value vSlice =
          sliceOffsetSize(query, {0, 0, (numHeads + kvHeads) * headDim},
                          {batch, seqQ, kvHeads * headDim}, rewriter, loc);
      if (failed(unpackToBnsh(qSlice, numHeads, headDim, rewriter, loc, op,
                              qBnsh)) ||
          failed(unpackToBnsh(kSlice, kvHeads, headDim, rewriter, loc, op,
                              kCur)) ||
          failed(
              unpackToBnsh(vSlice, kvHeads, headDim, rewriter, loc, op, vCur)))
        return failure();
    } else {
      if (qInTy.getRank() == 4)
        headDim = qInTy.getDimSize(3);
      else if (qInTy.getDimSize(2) % numHeads != 0)
        return rewriter.notifyMatchFailure(op,
                                           "query hidden is not heads * dim");
      else
        headDim = qInTy.getDimSize(2) / numHeads;
      if (failed(unpackToBnsh(query, numHeads, headDim, rewriter, loc, op,
                              qBnsh)) ||
          failed(unpackToBnsh(adaptor.getKey(), kvHeads, headDim, rewriter, loc,
                              op, kCur)) ||
          failed(unpackToBnsh(adaptor.getValue(), kvHeads, headDim, rewriter,
                              loc, op, vCur)))
        return failure();
    }

    if (op.getDoRotary() != 0) {
      if (!adaptor.getCosCache() || !adaptor.getSinCache())
        return rewriter.notifyMatchFailure(
            op, "do_rotary requires cos_cache and sin_cache");
      // Prefill without position_ids slices cache rows [0, seqQ). Decode with
      // past needs gather by position_ids (past KV is already rotated).
      if (adaptor.getPastKey() && !adaptor.getPositionIds())
        return rewriter.notifyMatchFailure(op, "decode RoPE is unsupported");
      bool interleaved = op.getRotaryInterleaved() != 0;
      if (failed(applyRope(qBnsh, adaptor.getCosCache(), adaptor.getSinCache(),
                           adaptor.getPositionIds(), seqQ, interleaved,
                           rewriter, loc, op)) ||
          failed(applyRope(kCur, adaptor.getCosCache(), adaptor.getSinCache(),
                           adaptor.getPositionIds(), seqQ, interleaved,
                           rewriter, loc, op)))
        return failure();
    }

    Type attnElem = qInTy.getElementType();
    if (failed(dequantIfNeeded(kCur, adaptor.getKScale(), attnElem, kQuant,
                               rewriter, loc, op)) ||
        failed(dequantIfNeeded(vCur, adaptor.getVScale(), attnElem, vQuant,
                               rewriter, loc, op)))
      return failure();

    Value presentK = kCur;
    Value presentV = vCur;
    int64_t seqKv = seqQ;
    Value pastK = adaptor.getPastKey();
    Value pastV = adaptor.getPastValue();
    if (pastK && pastV) {
      if (failed(dequantIfNeeded(pastK, adaptor.getKScale(), attnElem, kQuant,
                                 rewriter, loc, op)) ||
          failed(dequantIfNeeded(pastV, adaptor.getVScale(), attnElem, vQuant,
                                 rewriter, loc, op)))
        return failure();
    }
    if (failed(concatGrowingPast(kCur, vCur, pastK, pastV, batch, kvHeads,
                                 headDim, headDim, pkType.getDimSize(2),
                                 rewriter, loc, op, presentK, presentV, seqKv)))
      return failure();

    Value kSdpa, vSdpa;
    if (failed(broadcastKvHeads(presentK, numHeads, kvHeads, rewriter, loc, op,
                                kSdpa)) ||
        failed(broadcastKvHeads(presentV, numHeads, kvHeads, rewriter, loc, op,
                                vSdpa)))
      return failure();

    int64_t bh = batch * numHeads;
    Value q3 = reshapeTo(qBnsh, {bh, seqQ, headDim}, rewriter);
    Value kT = transposePerm(kSdpa, {0, 1, 3, 2}, rewriter, loc);
    Value k3 = reshapeTo(kT, {bh, headDim, seqKv}, rewriter);
    Value v3 = reshapeTo(vSdpa, {bh, seqKv, headDim}, rewriter);

    auto qkTy = RankedTensorType::get({bh, seqQ, seqKv}, attnElem);
    Value scores =
        tosa::MatMulOp::create(rewriter, loc, qkTy, q3, k3).getResult();

    double scale = op.getScale().convertToFloat();
    if (scale == 0.0)
      scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    Value scaleSplat = createSplatFloat(rewriter, loc, qkTy, scale);
    scores = emitTosaMul(rewriter, loc, scores, scaleSplat, qkTy);

    double softcap = op.getSoftcap().convertToFloat();
    if (softcap != 0.0) {
      Value cap = createSplatFloat(rewriter, loc, qkTy, softcap);
      Value invCap = tosa::ReciprocalOp::create(rewriter, loc, qkTy, cap);
      Value scaled = emitTosaMul(rewriter, loc, scores, invCap, qkTy);
      Value th = tosa::TanhOp::create(rewriter, loc, qkTy, scaled);
      scores = emitTosaMul(rewriter, loc, th, cap, qkTy);
    }

    if (adaptor.getAttentionBias() &&
        failed(addAttentionBias(scores, adaptor.getAttentionBias(), qkTy, batch,
                                numHeads, rewriter, loc, op)))
      return failure();

    Value kIdx = createArangeI32(rewriter, loc, seqKv);
    kIdx = reshapeTo(kIdx, {1, 1, seqKv}, rewriter);
    scores = applyCausalPrefill(scores, kIdx, !op.getNoCausal(), seqQ, seqKv,
                                rewriter, loc);

    int64_t window = op.getLocalWindowSize();
    if (window > 0) {
      Value qIdx = createArangeI32(rewriter, loc, seqQ);
      int64_t pastLen = seqKv - seqQ;
      if (pastLen != 0) {
        Value off = createSplatI32(rewriter, loc, {seqQ},
                                   static_cast<int32_t>(pastLen));
        qIdx = tosa::AddOp::create(
            rewriter, loc, RankedTensorType::get({seqQ}, rewriter.getI32Type()),
            qIdx, off);
      }
      qIdx = reshapeTo(qIdx, {1, seqQ, 1}, rewriter);
      Value win = createSplatI32(rewriter, loc, {1, seqQ, 1},
                                 static_cast<int32_t>(window - 1));
      auto i32Ty = RankedTensorType::get({1, seqQ, 1}, rewriter.getI32Type());
      Value qMinus = tosa::SubOp::create(rewriter, loc, i32Ty, qIdx, win);
      auto predTy =
          RankedTensorType::get({1, seqQ, seqKv}, rewriter.getI1Type());
      Value pred = tosa::GreaterOp::create(rewriter, loc, predTy, qMinus, kIdx);
      scores = applySelectNegInf(scores, pred, rewriter, loc);
    }

    Value seqlens = adaptor.getSeqlensK();
    auto seqTy = dyn_cast<RankedTensorType>(seqlens.getType());
    if (!seqTy || !seqTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "seqlens_k must be static");
    if (seqTy.getElementType() != rewriter.getI32Type())
      seqlens = emitTosaCast(rewriter, loc, seqlens, rewriter.getI32Type());
    seqTy = cast<RankedTensorType>(seqlens.getType());
    int64_t seqElems = seqTy.getNumElements();
    seqlens = reshapeTo(seqlens, {seqElems}, rewriter);
    if (seqElems == batch && batch > 1) {
      seqlens = reshapeTo(seqlens, {batch, 1, 1}, rewriter);
      seqlens = tileMultiples(seqlens, {1, numHeads, 1}, {batch, numHeads, 1},
                              rewriter, loc);
      seqlens = reshapeTo(seqlens, {bh, 1, 1}, rewriter);
    } else {
      seqlens = reshapeTo(seqlens, {1, 1, 1}, rewriter);
    }
    // ORT prefill sentinel seqlens_k=-1 means no past and total_seq=seqQ.
    // Comparing kIdx > -1 would mask every key and softmax would see an
    // all--inf row. Map the sentinel to the last current-key index first.
    {
      auto slTy = cast<RankedTensorType>(seqlens.getType());
      Value zero = createSplatI32(rewriter, loc, slTy.getShape(), 0);
      int32_t lastCurrent = seqQ > 0 ? static_cast<int32_t>(seqQ - 1) : 0;
      Value lastCur =
          createSplatI32(rewriter, loc, slTy.getShape(), lastCurrent);
      auto sentPredTy =
          RankedTensorType::get(slTy.getShape(), rewriter.getI1Type());
      Value isSentinel =
          tosa::GreaterOp::create(rewriter, loc, sentPredTy, zero, seqlens);
      seqlens = tosa::SelectOp::create(rewriter, loc, slTy, isSentinel, lastCur,
                                       seqlens);
    }
    auto padPredTy = RankedTensorType::get(
        {cast<RankedTensorType>(seqlens.getType()).getDimSize(0), 1, seqKv},
        rewriter.getI1Type());
    Value padPred =
        tosa::GreaterOp::create(rewriter, loc, padPredTy, kIdx, seqlens);
    scores = applySelectNegInf(scores, padPred, rewriter, loc);

    Value qkBeforeSoftmax = scores;
    Value extraDenom;
    if (adaptor.getHeadSink()) {
      Value sink = adaptor.getHeadSink();
      auto sinkTy = dyn_cast<RankedTensorType>(sink.getType());
      if (!sinkTy || !sinkTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "head_sink must be static");
      if (sinkTy.getElementType() != attnElem)
        sink = emitTosaCast(rewriter, loc, sink, attnElem);
      sink = reshapeTo(sink, {1, numHeads, 1, 1}, rewriter);
      sink = tileMultiples(sink, {batch, 1, 1, 1}, {batch, numHeads, 1, 1},
                           rewriter, loc);
      sink = reshapeTo(sink, {bh, 1, 1}, rewriter);
      extraDenom = tosa::ExpOp::create(
          rewriter, loc, cast<RankedTensorType>(sink.getType()), sink);
    } else if (op.getSmoothSoftmax() != 0) {
      extraDenom = createSplatFloat(
          rewriter, loc, RankedTensorType::get({bh, 1, 1}, attnElem), 1.0);
    }

    Value probs = softmaxLastDim(scores, extraDenom, rewriter, loc);
    Value qkAfterSoftmax = probs;

    auto avTy = RankedTensorType::get({bh, seqQ, headDim}, attnElem);
    Value av =
        tosa::MatMulOp::create(rewriter, loc, avTy, probs, v3).getResult();
    Value y4 = reshapeTo(av, {batch, numHeads, seqQ, headDim}, rewriter);
    Value y = packBnshToOutput(y4, yType, rewriter, loc);

    if (failed(quantIfNeeded(presentK, adaptor.getKScale(), pkType, kQuant,
                             rewriter, loc, op)) ||
        failed(quantIfNeeded(presentV, adaptor.getVScale(), pvType, vQuant,
                             rewriter, loc, op)))
      return failure();

    if (cast<RankedTensorType>(presentK.getType()).getShape() !=
        pkType.getShape()) {
      if (cast<RankedTensorType>(presentK.getType()).getElementType() ==
          pkType.getElementType())
        presentK = reshapeTo(presentK, pkType.getShape(), rewriter);
    }
    if (cast<RankedTensorType>(presentV.getType()).getShape() !=
        pvType.getShape()) {
      if (cast<RankedTensorType>(presentV.getType()).getElementType() ==
          pvType.getElementType())
        presentV = reshapeTo(presentV, pvType.getShape(), rewriter);
    }

    int64_t qkOutput = op.getQkOutput();
    SmallVector<Value> results = {y, presentK, presentV};
    if (qkOutput != 0) {
      auto qkOutTy = dyn_cast<RankedTensorType>(op.getResult(3).getType());
      if (!qkOutTy || !qkOutTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "output_qk must be static");
      Value qk = qkOutput == 1 ? qkBeforeSoftmax : qkAfterSoftmax;
      if (cast<RankedTensorType>(qk.getType()) != qkOutTy)
        qk = reshapeTo(qk, qkOutTy.getShape(), rewriter);
      results.push_back(qk);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// hip.multi_head_attention -> TOSA SDPA, matching MIGraphX's ONNX parse:
// unpack (packed QKV / packed KV / separate / BNSH cross), optional 1-D QKV
// projection bias, BNSH transpose, growing past concat, QK dot, attention
// bias + key-padding add, scale, softmax, PV dot. Ctx and DPS inits dropped.
struct MhaConverter final : public OpConversionPattern<MultiHeadAttentionOp> {
  using OpConversionPattern<MultiHeadAttentionOp>::OpConversionPattern;

  static LogicalResult slicePacked5D(Value packed, int64_t index,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, Value &bshd) {
    auto ty = cast<RankedTensorType>(packed.getType());
    int64_t b = ty.getDimSize(0);
    int64_t s = ty.getDimSize(1);
    int64_t h = ty.getDimSize(2);
    int64_t d = ty.getDimSize(4);
    Value sl = sliceOffsetSize(packed, {0, 0, 0, index, 0}, {b, s, h, 1, d},
                               rewriter, loc);
    bshd = reshapeTo(sl, {b, s, h, d}, rewriter);
    return success();
  }

  static LogicalResult addHiddenBias(Value &tensor, Value bias, int64_t start,
                                     int64_t len,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, Operation *op) {
    auto ty = cast<RankedTensorType>(tensor.getType());
    int64_t b = ty.getDimSize(0);
    int64_t s = ty.getDimSize(1);
    Value bsh = tensor;
    if (ty.getRank() == 4)
      bsh = reshapeTo(tensor, {b, s, ty.getDimSize(2) * ty.getDimSize(3)},
                      rewriter);
    auto bshTy = cast<RankedTensorType>(bsh.getType());
    if (bshTy.getDimSize(2) != len)
      return rewriter.notifyMatchFailure(op, "QKV bias hidden size mismatch");
    Value sl = sliceOffsetSize(bias, {start}, {len}, rewriter, loc);
    sl = reshapeTo(sl, {1, 1, len}, rewriter);
    if (failed(tosa::EqualizeRanks(rewriter, loc, bsh, sl)))
      return rewriter.notifyMatchFailure(op, "QKV bias not broadcastable");
    bsh = tosa::AddOp::create(rewriter, loc, bshTy, bsh, sl);
    if (ty.getRank() == 4)
      tensor = reshapeTo(bsh, ty.getShape(), rewriter);
    else
      tensor = bsh;
    return success();
  }

  static LogicalResult
  addKeyPadding(Value &scores, Value mask, RankedTensorType qkTy, int64_t batch,
                int64_t numHeads, int64_t seqQ, int64_t seqKv, Type attnElem,
                double filter, ConversionPatternRewriter &rewriter,
                Location loc, Operation *op) {
    auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskTy || !maskTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "key_padding_mask must be static");
    int64_t bh = batch * numHeads;
    if (maskTy.getRank() == 1 && maskTy.getNumElements() == 3 * batch + 2)
      return rewriter.notifyMatchFailure(op,
                                         "left-pad key mask is unsupported");

    Value additive;
    if (maskTy.getRank() == 1 && maskTy.getNumElements() == batch) {
      Value seqlens = mask;
      if (maskTy.getElementType() != rewriter.getI32Type())
        seqlens = emitTosaCast(rewriter, loc, seqlens, rewriter.getI32Type());
      seqlens = reshapeTo(seqlens, {batch, 1, 1}, rewriter);
      seqlens = tileMultiples(seqlens, {1, numHeads, 1}, {batch, numHeads, 1},
                              rewriter, loc);
      seqlens = reshapeTo(seqlens, {bh, 1, 1}, rewriter);
      Value kIdx = createArangeI32(rewriter, loc, seqKv);
      kIdx = reshapeTo(kIdx, {1, 1, seqKv}, rewriter);
      auto predTy = RankedTensorType::get({bh, 1, seqKv}, rewriter.getI1Type());
      // 1-D mask is exclusive valid length: keep k < seqlens_k.
      Value keep =
          tosa::GreaterOp::create(rewriter, loc, predTy, seqlens, kIdx);
      Value filt = createSplatFloat(rewriter, loc, qkTy, filter);
      Value zero = createSplatFloat(rewriter, loc, qkTy, 0.0);
      if (failed(tosa::EqualizeRanks(rewriter, loc, keep, scores)))
        return rewriter.notifyMatchFailure(op, "key mask not broadcastable");
      additive = tosa::SelectOp::create(rewriter, loc, qkTy, keep, zero, filt);
    } else if (maskTy.getRank() == 2 || maskTy.getRank() == 3) {
      Value m = mask;
      if (maskTy.getElementType() != rewriter.getI32Type())
        m = emitTosaCast(rewriter, loc, m, rewriter.getI32Type());
      Value zeros = createSplatI32(
          rewriter, loc, cast<RankedTensorType>(m.getType()).getShape(), 0);
      auto eqTy = RankedTensorType::get(
          cast<RankedTensorType>(m.getType()).getShape(), rewriter.getI1Type());
      Value isPad = tosa::EqualOp::create(rewriter, loc, eqTy, m, zeros);
      if (maskTy.getRank() == 2)
        isPad = reshapeTo(isPad, {batch, 1, seqKv}, rewriter);
      else
        isPad = reshapeTo(isPad, {batch, 1, seqQ, seqKv}, rewriter);
      if (maskTy.getRank() == 2)
        isPad = tileMultiples(isPad, {1, numHeads, 1}, {batch, numHeads, seqKv},
                              rewriter, loc);
      else
        isPad = tileMultiples(isPad, {1, numHeads, 1, 1},
                              {batch, numHeads, seqQ, seqKv}, rewriter, loc);
      if (maskTy.getRank() == 2)
        isPad = reshapeTo(isPad, {bh, 1, seqKv}, rewriter);
      else
        isPad = reshapeTo(isPad, {bh, seqQ, seqKv}, rewriter);
      Value filt = createSplatFloat(rewriter, loc, qkTy, filter);
      Value zero = createSplatFloat(rewriter, loc, qkTy, 0.0);
      if (failed(tosa::EqualizeRanks(rewriter, loc, isPad, filt)))
        return rewriter.notifyMatchFailure(op, "key mask not broadcastable");
      additive = tosa::SelectOp::create(rewriter, loc, qkTy, isPad, filt, zero);
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported key_padding_mask rank");
    }
    if (failed(tosa::EqualizeRanks(rewriter, loc, scores, additive)))
      return rewriter.notifyMatchFailure(op, "key mask not broadcastable");
    scores = tosa::AddOp::create(rewriter, loc, qkTy, scores, additive);
    return success();
  }

  LogicalResult
  matchAndRewrite(MultiHeadAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() < 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");
    auto yType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!yType || !yType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (adaptor.getCacheIndirection())
      return rewriter.notifyMatchFailure(op,
                                         "cache_indirection is unsupported");
    if (adaptor.getPastSequenceLength())
      return rewriter.notifyMatchFailure(
          op, "past_sequence_length share-buffer is unsupported");

    Location loc = op.getLoc();
    int64_t numHeads = op.getNumHeads();
    if (numHeads <= 0)
      return rewriter.notifyMatchFailure(op, "num_heads must be positive");

    Value query = adaptor.getQuery();
    auto qInTy = dyn_cast<RankedTensorType>(query.getType());
    if (!qInTy || !qInTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "query must be a static tensor");
    if (!isa<FloatType>(qInTy.getElementType()))
      return rewriter.notifyMatchFailure(op, "query must be float");

    int64_t batch = qInTy.getDimSize(0);
    int64_t seqQ = qInTy.getDimSize(1);
    int64_t headDim = 0;
    int64_t headDimV = 0;
    int64_t seqKv = seqQ;
    Value qBshd, kBshd, vBshd;

    if (qInTy.getRank() == 5) {
      if (qInTy.getDimSize(2) != numHeads || qInTy.getDimSize(3) != 3)
        return rewriter.notifyMatchFailure(op,
                                           "packed QKV must be [B,S,H,3,D]");
      headDim = qInTy.getDimSize(4);
      headDimV = headDim;
      if (failed(slicePacked5D(query, 0, rewriter, loc, qBshd)) ||
          failed(slicePacked5D(query, 1, rewriter, loc, kBshd)) ||
          failed(slicePacked5D(query, 2, rewriter, loc, vBshd)))
        return failure();
    } else if (qInTy.getRank() == 3) {
      if (qInTy.getDimSize(2) % numHeads != 0)
        return rewriter.notifyMatchFailure(op,
                                           "query hidden is not heads * dim");
      headDim = qInTy.getDimSize(2) / numHeads;
      qBshd = reshapeTo(query, {batch, seqQ, numHeads, headDim}, rewriter);
      if (!adaptor.getKey())
        return rewriter.notifyMatchFailure(
            op, "key is required unless QKV is packed");
      auto kTy = dyn_cast<RankedTensorType>(adaptor.getKey().getType());
      if (!kTy || !kTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "key must be a static tensor");
      if (kTy.getRank() == 5) {
        if (kTy.getDimSize(2) != numHeads || kTy.getDimSize(3) != 2 ||
            kTy.getDimSize(4) != headDim)
          return rewriter.notifyMatchFailure(op,
                                             "packed KV must be [B,S,H,2,D]");
        seqKv = kTy.getDimSize(1);
        headDimV = headDim;
        if (failed(slicePacked5D(adaptor.getKey(), 0, rewriter, loc, kBshd)) ||
            failed(slicePacked5D(adaptor.getKey(), 1, rewriter, loc, vBshd)))
          return failure();
      } else {
        if (!adaptor.getValue())
          return rewriter.notifyMatchFailure(op, "value is required");
        auto vTy = dyn_cast<RankedTensorType>(adaptor.getValue().getType());
        if (!vTy || !vTy.hasStaticShape())
          return rewriter.notifyMatchFailure(op,
                                             "value must be a static tensor");
        if (kTy.getRank() == 3) {
          seqKv = kTy.getDimSize(1);
          if (kTy.getDimSize(2) != numHeads * headDim)
            return rewriter.notifyMatchFailure(op, "key hidden mismatch");
          if (vTy.getRank() != 3)
            return rewriter.notifyMatchFailure(
                op, "value must be rank 3 for rank-3 key");
          if (vTy.getDimSize(2) % numHeads != 0)
            return rewriter.notifyMatchFailure(
                op, "value hidden is not heads * dim");
          headDimV = vTy.getDimSize(2) / numHeads;
          kBshd = reshapeTo(adaptor.getKey(), {batch, seqKv, numHeads, headDim},
                            rewriter);
          vBshd = reshapeTo(adaptor.getValue(),
                            {batch, seqKv, numHeads, headDimV}, rewriter);
        } else if (kTy.getRank() == 4) {
          seqKv = kTy.getDimSize(2);
          headDimV = vTy.getDimSize(3);
          kBshd = transposePerm(adaptor.getKey(), {0, 2, 1, 3}, rewriter, loc);
          vBshd =
              transposePerm(adaptor.getValue(), {0, 2, 1, 3}, rewriter, loc);
        } else {
          return rewriter.notifyMatchFailure(op, "unsupported key rank");
        }
      }
    } else {
      return rewriter.notifyMatchFailure(op, "query must be rank 3 or 5");
    }

    if (adaptor.getBias()) {
      auto bTy = dyn_cast<RankedTensorType>(adaptor.getBias().getType());
      if (!bTy || !bTy.hasStaticShape() || bTy.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "QKV bias must be a static 1-D tensor");
      int64_t hidden = numHeads * headDim;
      int64_t hiddenV = numHeads * headDimV;
      if (bTy.getDimSize(0) != hidden + hidden + hiddenV)
        return rewriter.notifyMatchFailure(op, "QKV bias length mismatch");
      if (failed(addHiddenBias(qBshd, adaptor.getBias(), 0, hidden, rewriter,
                               loc, op)) ||
          failed(addHiddenBias(kBshd, adaptor.getBias(), hidden, hidden,
                               rewriter, loc, op)) ||
          failed(addHiddenBias(vBshd, adaptor.getBias(), 2 * hidden, hiddenV,
                               rewriter, loc, op)))
        return failure();
    }

    Value qBnsh = transposePerm(qBshd, {0, 2, 1, 3}, rewriter, loc);
    Value kCur = transposePerm(kBshd, {0, 2, 1, 3}, rewriter, loc);
    Value vCur = transposePerm(vBshd, {0, 2, 1, 3}, rewriter, loc);

    std::optional<int64_t> presentSeq;
    if (op.getNumResults() >= 3) {
      auto pkType = dyn_cast<RankedTensorType>(op.getResult(1).getType());
      if (pkType && pkType.hasStaticShape() && pkType.getRank() == 4)
        presentSeq = pkType.getDimSize(2);
    }
    Value presentK, presentV;
    int64_t attnSeqKv = seqKv;
    if (failed(concatGrowingPast(kCur, vCur, adaptor.getPastKey(),
                                 adaptor.getPastValue(), batch, numHeads,
                                 headDim, headDimV, presentSeq, rewriter, loc,
                                 op, presentK, presentV, attnSeqKv)))
      return failure();
    seqKv = attnSeqKv;

    Type attnElem = qInTy.getElementType();
    int64_t bh = batch * numHeads;
    Value q3 = reshapeTo(qBnsh, {bh, seqQ, headDim}, rewriter);
    Value kT = transposePerm(presentK, {0, 1, 3, 2}, rewriter, loc);
    Value k3 = reshapeTo(kT, {bh, headDim, seqKv}, rewriter);
    Value v3 = reshapeTo(presentV, {bh, seqKv, headDimV}, rewriter);
    auto qkTy = RankedTensorType::get({bh, seqQ, seqKv}, attnElem);
    Value scores =
        tosa::MatMulOp::create(rewriter, loc, qkTy, q3, k3).getResult();

    double scale = op.getScale().convertToFloat();
    if (scale == 0.0)
      scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    Value scaleSplat = createSplatFloat(rewriter, loc, qkTy, scale);
    scores = emitTosaMul(rewriter, loc, scores, scaleSplat, qkTy);

    if (adaptor.getAttentionBias() &&
        failed(addAttentionBias(scores, adaptor.getAttentionBias(), qkTy, batch,
                                numHeads, rewriter, loc, op)))
      return failure();
    if (adaptor.getKeyPaddingMask() &&
        failed(addKeyPadding(scores, adaptor.getKeyPaddingMask(), qkTy, batch,
                             numHeads, seqQ, seqKv, attnElem,
                             op.getMaskFilterValue().convertToFloat(), rewriter,
                             loc, op)))
      return failure();

    Value kIdx = createArangeI32(rewriter, loc, seqKv);
    kIdx = reshapeTo(kIdx, {1, 1, seqKv}, rewriter);
    scores = applyCausalPrefill(scores, kIdx, op.getUnidirectional() != 0, seqQ,
                                seqKv, rewriter, loc);

    Value probs = softmaxLastDim(scores, Value(), rewriter, loc);
    auto avTy = RankedTensorType::get({bh, seqQ, headDimV}, attnElem);
    Value av =
        tosa::MatMulOp::create(rewriter, loc, avTy, probs, v3).getResult();
    Value y4 = reshapeTo(av, {batch, numHeads, seqQ, headDimV}, rewriter);
    Value y = packBnshToOutput(y4, yType, rewriter, loc);

    SmallVector<Value> results = {y};
    if (op.getNumResults() >= 3) {
      auto pkType = cast<RankedTensorType>(op.getResult(1).getType());
      auto pvType = cast<RankedTensorType>(op.getResult(2).getType());
      if (cast<RankedTensorType>(presentK.getType()).getShape() !=
              pkType.getShape() &&
          cast<RankedTensorType>(presentK.getType()).getElementType() ==
              pkType.getElementType())
        presentK = reshapeTo(presentK, pkType.getShape(), rewriter);
      if (cast<RankedTensorType>(presentV.getType()).getShape() !=
              pvType.getShape() &&
          cast<RankedTensorType>(presentV.getType()).getElementType() ==
              pvType.getElementType())
        presentV = reshapeTo(presentV, pvType.getShape(), rewriter);
      results.push_back(presentK);
      results.push_back(presentV);
    }
    if (op.getNumResults() == 4) {
      auto qkOutTy = dyn_cast<RankedTensorType>(op.getResult(3).getType());
      if (!qkOutTy || !qkOutTy.hasStaticShape())
        return rewriter.notifyMatchFailure(op, "qk must be a static tensor");
      Value qk = probs;
      if (cast<RankedTensorType>(qk.getType()) != qkOutTy)
        qk = reshapeTo(qk, qkOutTy.getShape(), rewriter);
      results.push_back(qk);
    } else if (op.getNumResults() == 2) {
      return rewriter.notifyMatchFailure(op, "expected 1, 3, or 4 results");
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// ---------------------------------------------------------------------------
// hip.qmoe
// ---------------------------------------------------------------------------

// TOSA cannot express the runtime's sparse expert dispatch (bucket tokens by
// expert, run each expert over just its rows, scatter-add back), because the
// per-expert row counts are data dependent. The decomposition is dense
// instead: every expert runs over every token and is folded in weighted by its
// routing weight, which the top-k mask has already zeroed for tokens that did
// not select it. That is numerically equivalent and costs E/k times the work,
// so the expert count is capped to keep both the IR and the arithmetic bounded.
constexpr int64_t kMaxQMoEExperts = 64;

struct QMoEPlan {
  int64_t tokens;
  int64_t hidden;
  int64_t inter;
  int64_t experts;
  int64_t k;
  int64_t blockSize;
  int64_t fc1Rows; // 2 * inter: gate and linear interleaved
  int64_t hiddenBlocks;
  int64_t interBlocks;
  bool normalize;
};

std::optional<QMoEPlan> planQMoE(hip::QMoEOp op) {
  if (op.getNumResults() != 1)
    return std::nullopt;

  // The runtime supports exactly this envelope: interleaved SwiGLU, no fc3.
  // activation_type is ignored there too, so it is not gated here.
  if (op.getSwigluFusion() != 1 || op.getUseSparseMixer() != 0)
    return std::nullopt;
  if (op.getFc3ExpertsWeights() || op.getFc3Scales() ||
      op.getFc3ExpertsBias() || op.getFc3ZeroPoints())
    return std::nullopt;
  // router_weights overrides router_probs and skips the softmax; not modelled.
  if (op.getRouterWeights())
    return std::nullopt;
  if (op.getExpertWeightBits() != 4)
    return std::nullopt;

  QMoEPlan plan;
  plan.blockSize = op.getBlockSize();
  if (plan.blockSize < 16 || (plan.blockSize & (plan.blockSize - 1)) != 0)
    return std::nullopt;

  auto inputTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto routerTy = dyn_cast<RankedTensorType>(op.getRouterProbs().getType());
  auto fc1wTy = dyn_cast<RankedTensorType>(op.getFc1ExpertsWeights().getType());
  auto fc1sTy = dyn_cast<RankedTensorType>(op.getFc1Scales().getType());
  auto fc2wTy = dyn_cast<RankedTensorType>(op.getFc2ExpertsWeights().getType());
  auto fc2sTy = dyn_cast<RankedTensorType>(op.getFc2Scales().getType());
  auto resultTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inputTy || !routerTy || !fc1wTy || !fc1sTy || !fc2wTy || !fc2sTy ||
      !resultTy || !inputTy.hasStaticShape() || !routerTy.hasStaticShape() ||
      !fc1wTy.hasStaticShape() || !fc1sTy.hasStaticShape() ||
      !fc2wTy.hasStaticShape() || !fc2sTy.hasStaticShape() ||
      !resultTy.hasStaticShape())
    return std::nullopt;

  Type computeElem = resultTy.getElementType();
  if (!computeElem.isF32() && !computeElem.isF16() && !computeElem.isBF16())
    return std::nullopt;
  if (inputTy.getElementType() != computeElem ||
      resultTy.getShape() != inputTy.getShape())
    return std::nullopt;
  if (!isa<FloatType>(routerTy.getElementType()) ||
      !isa<FloatType>(fc1sTy.getElementType()) ||
      !isa<FloatType>(fc2sTy.getElementType()))
    return std::nullopt;
  if (!isa<IntegerType>(fc1wTy.getElementType()) ||
      !isa<IntegerType>(fc2wTy.getElementType()))
    return std::nullopt;

  if (inputTy.getRank() < 2 || fc1wTy.getRank() != 3 || fc1sTy.getRank() != 3 ||
      fc2wTy.getRank() != 3 || fc2sTy.getRank() != 3 || routerTy.getRank() != 2)
    return std::nullopt;

  plan.hidden = inputTy.getShape().back();
  plan.tokens = 1;
  for (int64_t d : inputTy.getShape().drop_back())
    plan.tokens *= d;

  plan.experts = fc1wTy.getDimSize(0);
  plan.fc1Rows = fc1wTy.getDimSize(1);
  if (plan.experts <= 0 || plan.experts > kMaxQMoEExperts)
    return std::nullopt;
  if (routerTy.getDimSize(0) != plan.tokens ||
      routerTy.getDimSize(1) != plan.experts)
    return std::nullopt;

  plan.k = op.getK();
  if (plan.k < 1 || plan.k > plan.experts)
    return std::nullopt;
  plan.normalize = op.getNormalizeRoutingWeights() != 0;

  // fc1: [E, 2*inter, hidden/2] packed, [E, 2*inter, hidden/block] scales.
  if (fc1wTy.getDimSize(2) * 2 != plan.hidden)
    return std::nullopt;
  if (fc1sTy.getDimSize(0) != plan.experts ||
      fc1sTy.getDimSize(1) != plan.fc1Rows)
    return std::nullopt;
  plan.hiddenBlocks = fc1sTy.getDimSize(2);
  if (plan.hiddenBlocks * plan.blockSize != plan.hidden)
    return std::nullopt;

  // fc2: [E, hidden, inter/2] packed, [E, hidden, inter/block] scales.
  if (fc2wTy.getDimSize(0) != plan.experts ||
      fc2wTy.getDimSize(1) != plan.hidden)
    return std::nullopt;
  plan.inter = fc2wTy.getDimSize(2) * 2;
  if (plan.fc1Rows != 2 * plan.inter)
    return std::nullopt;
  if (fc2sTy.getDimSize(0) != plan.experts ||
      fc2sTy.getDimSize(1) != plan.hidden)
    return std::nullopt;
  plan.interBlocks = fc2sTy.getDimSize(2);
  if (plan.interBlocks * plan.blockSize != plan.inter)
    return std::nullopt;

  auto checkBias = [&](Value bias, int64_t width) {
    if (!bias)
      return true;
    auto ty = dyn_cast<RankedTensorType>(bias.getType());
    return ty && ty.hasStaticShape() && ty.getRank() == 2 &&
           ty.getDimSize(0) == plan.experts && ty.getDimSize(1) == width;
  };
  if (!checkBias(op.getFc1ExpertsBias(), plan.fc1Rows) ||
      !checkBias(op.getFc2ExpertsBias(), plan.hidden))
    return std::nullopt;

  auto checkZp = [&](Value zp, int64_t rows, int64_t blocks) {
    if (!zp)
      return true;
    auto ty = dyn_cast<RankedTensorType>(zp.getType());
    if (!ty || !ty.hasStaticShape() || ty.getRank() != 3 ||
        !isa<IntegerType>(ty.getElementType()))
      return false;
    if (ty.getDimSize(0) != plan.experts || ty.getDimSize(1) != rows)
      return false;
    // Either one zero point per block, or the MatMulNBits packed nibble
    // stream of ceil(blocks / 2) bytes.
    int64_t cols = ty.getDimSize(2);
    return cols == blocks || cols == (blocks + 1) / 2;
  };
  if (!checkZp(op.getFc1ZeroPoints(), plan.fc1Rows, plan.hiddenBlocks) ||
      !checkZp(op.getFc2ZeroPoints(), plan.hidden, plan.interBlocks))
    return std::nullopt;

  return plan;
}

bool isTosaExpressibleQMoE(hip::QMoEOp op) { return planQMoE(op).has_value(); }

// k rounds of "take the largest, break ties toward the lower expert index,
// mask the winner", which is the order the routing kernel's block argmax
// produces. Returns [tokens, experts] weights that are zero off the top-k.
Value qmoeRoutingWeights(Value probs, const QMoEPlan &plan, Type computeElem,
                         ConversionPatternRewriter &rewriter, Location loc) {
  auto probsTy =
      RankedTensorType::get({plan.tokens, plan.experts}, computeElem);
  auto boolTy =
      RankedTensorType::get({plan.tokens, plan.experts}, rewriter.getI1Type());
  auto i32Ty =
      RankedTensorType::get({plan.tokens, plan.experts}, rewriter.getI32Type());
  auto reducedF = keepdimsReduceType(probsTy, 1);
  auto reducedI = keepdimsReduceType(i32Ty, 1);
  IntegerAttr axisAttr = rewriter.getI32IntegerAttr(1);

  SmallVector<int32_t> iotaVals(plan.experts);
  for (int64_t e = 0; e < plan.experts; ++e)
    iotaVals[e] = static_cast<int32_t>(e);
  Value iota = createI32Dense(rewriter, loc, {1, plan.experts}, iotaVals);
  if (plan.tokens != 1)
    iota = tileMultiples(iota, {plan.tokens, 1}, {plan.tokens, plan.experts},
                         rewriter, loc);

  // Softmax output is in [0, 1], so any negative value loses every later round.
  Value losing = createSplatFloat(rewriter, loc, probsTy, -1.0);
  Value outOfRange = createSplatInt(rewriter, loc, i32Ty, plan.experts);

  Value selected;
  Value cur = probs;
  for (int64_t round = 0; round < plan.k; ++round) {
    Value rmax =
        tosa::ReduceMaxOp::create(rewriter, loc, reducedF, cur, axisAttr);
    Value isMax = tosa::EqualOp::create(rewriter, loc, boolTy, cur, rmax);
    // Reduce the tied positions to the smallest index so ties resolve the same
    // way the kernel's `v == bv && e < bi` comparison does.
    Value candidates =
        tosa::SelectOp::create(rewriter, loc, i32Ty, isMax, iota, outOfRange);
    Value first = tosa::ReduceMinOp::create(rewriter, loc, reducedI, candidates,
                                            axisAttr);
    Value hit = tosa::EqualOp::create(rewriter, loc, boolTy, iota, first);
    // On i1 bitwise_or and logical_or coincide; the bitwise form is the one
    // PR #1041 settled on for boolean masks in this pass.
    selected = round == 0 ? hit
                          : tosa::BitwiseOrOp::create(rewriter, loc, boolTy,
                                                      selected, hit)
                                .getResult();
    cur = tosa::SelectOp::create(rewriter, loc, probsTy, hit, losing, cur);
  }

  Value zeros = createSplatFloat(rewriter, loc, probsTy, 0.0);
  Value weights =
      tosa::SelectOp::create(rewriter, loc, probsTy, selected, probs, zeros);
  if (plan.normalize) {
    Value sum =
        tosa::ReduceSumOp::create(rewriter, loc, reducedF, weights, axisAttr);
    Value rec = tosa::ReciprocalOp::create(rewriter, loc, reducedF, sum);
    weights = emitTosaMul(rewriter, loc, weights, rec, probsTy);
  }
  return weights;
}

// Slice one expert out of [E, rows, cols/2], unpack the nibbles and apply the
// per-block scale. Packing and the implicit zero point of 8 follow the
// MatMulNBits convention the QMoE weights are stored in.
Value qmoeDequantExpert(Value packedAll, Value scalesAll, Value zpAll,
                        int64_t expert, int64_t rows, int64_t cols,
                        int64_t blocks, int64_t blockSize, Type computeElem,
                        ConversionPatternRewriter &rewriter, Location loc) {
  int64_t packedCols =
      cast<RankedTensorType>(packedAll.getType()).getDimSize(2);
  Value packed = sliceOffsetSize(packedAll, {expert, 0, 0},
                                 {1, rows, packedCols}, rewriter, loc);
  packed = reshapeTo(packed, {rows, packedCols}, rewriter);
  Value q = unpackInt4LastDim(packed, rewriter, loc);
  q = sliceLastDimTo(q, cols, rewriter, loc);

  Value scales = sliceOffsetSize(scalesAll, {expert, 0, 0}, {1, rows, blocks},
                                 rewriter, loc);
  scales = reshapeTo(scales, {rows, blocks}, rewriter);
  scales = emitTosaCast(rewriter, loc, scales, computeElem);
  scales = broadcastBlocksAlongK(scales, rows, cols, blocks, blockSize,
                                 rewriter, loc);

  Value zeroPoint;
  if (zpAll) {
    int64_t zpCols = cast<RankedTensorType>(zpAll.getType()).getDimSize(2);
    Value zp = sliceOffsetSize(zpAll, {expert, 0, 0}, {1, rows, zpCols},
                               rewriter, loc);
    zp = reshapeTo(zp, {rows, zpCols}, rewriter);
    if (zpCols != blocks) {
      zp = unpackInt4LastDim(zp, rewriter, loc);
      zp = sliceLastDimTo(zp, blocks, rewriter, loc);
    }
    zp =
        broadcastBlocksAlongK(zp, rows, cols, blocks, blockSize, rewriter, loc);
    zeroPoint = emitTosaCast(rewriter, loc, zp, computeElem);
  } else {
    zeroPoint = createSplatFloat(
        rewriter, loc, RankedTensorType::get({1, 1}, computeElem), 8.0);
  }

  auto weightTy = RankedTensorType::get({rows, cols}, computeElem);
  Value shifted = tosa::SubOp::create(
      rewriter, loc, weightTy, emitTosaCast(rewriter, loc, q, computeElem),
      zeroPoint);
  return emitTosaMul(rewriter, loc, shifted, scales, weightTy);
}

// fc1 emits gate and linear interleaved on the trailing axis: gate at even
// columns, linear at odd. De-interleave, clamp, then
// G * sigmoid(alpha * G) * (L + beta). The gate is clamped from above only;
// the linear side is clamped on both ends.
Value qmoeSwiglu(Value fc1Out, const QMoEPlan &plan, Type computeElem,
                 double alpha, double beta, double limit,
                 ConversionPatternRewriter &rewriter, Location loc) {
  Value pairs = reshapeTo(fc1Out, {plan.tokens, plan.inter, 2}, rewriter);
  Value gate = sliceOffsetSize(pairs, {0, 0, 0}, {plan.tokens, plan.inter, 1},
                               rewriter, loc);
  Value linear = sliceOffsetSize(pairs, {0, 0, 1}, {plan.tokens, plan.inter, 1},
                                 rewriter, loc);
  auto actTy = RankedTensorType::get({plan.tokens, plan.inter}, computeElem);
  gate = reshapeTo(gate, {plan.tokens, plan.inter}, rewriter);
  linear = reshapeTo(linear, {plan.tokens, plan.inter}, rewriter);

  Value hi = createSplatFloat(rewriter, loc, actTy, limit);
  Value lo = createSplatFloat(rewriter, loc, actTy, -limit);
  Value g = tosa::MinimumOp::create(rewriter, loc, actTy, gate, hi);
  Value l = tosa::MinimumOp::create(rewriter, loc, actTy, linear, hi);
  l = tosa::MaximumOp::create(rewriter, loc, actTy, l, lo);

  Value alphaC = createSplatFloat(rewriter, loc, actTy, alpha);
  Value sigmoid = tosa::SigmoidOp::create(
      rewriter, loc, actTy, emitTosaMul(rewriter, loc, g, alphaC, actTy));
  Value betaC = createSplatFloat(rewriter, loc, actTy, beta);
  Value shiftedLinear = tosa::AddOp::create(rewriter, loc, actTy, l, betaC);
  return emitTosaMul(rewriter, loc,
                     emitTosaMul(rewriter, loc, g, sigmoid, actTy),
                     shiftedLinear, actTy);
}

// hip.qmoe -> softmax routing + top-k mask, then every expert's dequantized
// FC1/SwiGLU/FC2 folded in weighted by its routing weight.
struct QMoEConverter final : public OpConversionPattern<hip::QMoEOp> {
  using OpConversionPattern<hip::QMoEOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::QMoEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<QMoEPlan> maybePlan = planQMoE(op);
    if (!maybePlan)
      return rewriter.notifyMatchFailure(op, "unsupported qmoe configuration");
    const QMoEPlan &plan = *maybePlan;

    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Type computeElem = resultTy.getElementType();

    double alpha = op.getActivationAlphaAttr().getValueAsDouble();
    double beta = op.getActivationBetaAttr().getValueAsDouble();
    double limit = op.getSwigluLimitAttr().getValueAsDouble();

    auto tokenTy =
        RankedTensorType::get({plan.tokens, plan.hidden}, computeElem);
    auto fc1Ty =
        RankedTensorType::get({plan.tokens, plan.fc1Rows}, computeElem);

    Value input =
        reshapeTo(adaptor.getInput(), {plan.tokens, plan.hidden}, rewriter);
    Value router =
        emitTosaCast(rewriter, loc, adaptor.getRouterProbs(), computeElem);
    router = reshapeTo(router, {plan.tokens, plan.experts}, rewriter);
    // The routing kernel softmaxes the logits before selecting, so the weights
    // are softmax probabilities, not raw logits.
    Value probs = softmaxLastDim(router, /*extraDenom=*/Value(), rewriter, loc);
    Value weights = qmoeRoutingWeights(probs, plan, computeElem, rewriter, loc);

    Value acc = createSplatFloat(rewriter, loc, tokenTy, 0.0);
    for (int64_t e = 0; e < plan.experts; ++e) {
      Value w1 = qmoeDequantExpert(
          adaptor.getFc1ExpertsWeights(), adaptor.getFc1Scales(),
          adaptor.getFc1ZeroPoints(), e, plan.fc1Rows, plan.hidden,
          plan.hiddenBlocks, plan.blockSize, computeElem, rewriter, loc);
      Value fc1 =
          emitUnbatchedMatmul(input, transposePerm(w1, {1, 0}, rewriter, loc),
                              fc1Ty, rewriter, loc);
      if (Value bias = adaptor.getFc1ExpertsBias()) {
        Value row =
            sliceOffsetSize(bias, {e, 0}, {1, plan.fc1Rows}, rewriter, loc);
        fc1 =
            tosa::AddOp::create(rewriter, loc, fc1Ty, fc1,
                                emitTosaCast(rewriter, loc, row, computeElem));
      }

      Value act =
          qmoeSwiglu(fc1, plan, computeElem, alpha, beta, limit, rewriter, loc);

      Value w2 = qmoeDequantExpert(
          adaptor.getFc2ExpertsWeights(), adaptor.getFc2Scales(),
          adaptor.getFc2ZeroPoints(), e, plan.hidden, plan.inter,
          plan.interBlocks, plan.blockSize, computeElem, rewriter, loc);
      Value fc2 =
          emitUnbatchedMatmul(act, transposePerm(w2, {1, 0}, rewriter, loc),
                              tokenTy, rewriter, loc);
      if (Value bias = adaptor.getFc2ExpertsBias()) {
        Value row =
            sliceOffsetSize(bias, {e, 0}, {1, plan.hidden}, rewriter, loc);
        fc2 =
            tosa::AddOp::create(rewriter, loc, tokenTy, fc2,
                                emitTosaCast(rewriter, loc, row, computeElem));
      }

      // Zero for every token that did not route to this expert, so the dense
      // sum reproduces the sparse dispatch.
      Value expertWeight =
          sliceOffsetSize(weights, {0, e}, {plan.tokens, 1}, rewriter, loc);
      acc = tosa::AddOp::create(
          rewriter, loc, tokenTy, acc,
          emitTosaMul(rewriter, loc, fc2, expertWeight, tokenTy));
    }

    rewriter.replaceOp(op, reshapeTo(acc, resultTy.getShape(), rewriter));
    return success();
  }
};

class HipToTosaPass : public impl::ConvertHipToTosaPassBase<HipToTosaPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!funcOp->hasAttr("rock.kernel"))
      return;

    MLIRContext *ctx = &getContext();

    // Mark only the ops this pass has patterns for, and use a partial
    // conversion so everything else in the kernel is left alone.
    //
    // A full conversion over an illegal hip dialect cannot work here. An
    // outlined kernel can keep an unsupported fusion anchor such as hip.pool,
    // and it carries the ops feeding the anchors' DPS operands: ub.poison for
    // the !hip.context and tensor.empty for each outs buffer.
    // Full conversion legalizes every op in the region, so each of those has
    // to be enumerated as legal or the pass fails on IR it never meant to
    // convert. Listing the illegal ops instead keeps the useful half of the
    // guarantee: a hip op this pass claims still has to convert or the pass
    // fails.
    ConversionTarget conversion(*ctx);
    conversion.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    // NonZero intentionally remains legal (unconverted): supported TOSA has
    // no ordered prefix scan or data-dependent result extent, while
    // hip.nonzero returns both compacted [rank, N] indices and the runtime N.
    conversion.addIllegalOp<
        ConvOp, CausalConvWithStateOp, MatmulOp, GemmOp, TransposeOp, AddOp,
        SubOp, MinOp, MaxOp, MulOp, DivOp, AbsOp, NegOp, CeilOp, FloorOp, ExpOp,
        LogOp, SinOp, CosOp, TanhOp, ErfOp, SigmoidOp, ReciprocalOp, SqrtOp,
        SoftplusOp, GeluOp, BiasGeluOp, FastGeluOp, SiluOp, SwishOp, WhereOp,
        LeakyReluOp, MiopenSoftmaxOp, ReduceSumOp, ReduceMeanOp, CastOp,
        QuantizeLinearOp, DequantizeLinearOp, MatMulNBitsOp, GatherOp,
        GatherElementsOp, GatherNDOp, GridSampleOp, SizeOp, OneHotOp, RangeOp,
        RopeOp, GqaOp, MultiHeadAttentionOp, RoundOp, ModOp, AtanOp, RmsNormOp,
        LayerNormOp, InstanceNormOp, SkipRmsNormOp>();
    // tosa.matmul (and other tosa ops) are not destination-passing, so
    // MatMulConverter drops each hip op's DPS `outs` operand. The
    // `tensor.empty` that fed it is then dead, but a full conversion still
    // requires every remaining op to be legal -- the framework does not DCE
    // this pre-existing op on its own. Mark it legal so conversion succeeds;
    // the canonicalizer that follows this pass removes the dead empty.
    conversion.addLegalOp<ub::PoisonOp, tensor::EmptyOp>();
    conversion.addDynamicallyLegalOp<ExpandOp>(
        [](ExpandOp op) { return !isTosaExpressibleExpand(op); });
    // The comparison, logical and sign ops are claimed by element type rather
    // than outright, so a boolean carried as ui8 or an unsigned comparison --
    // both of which OnnxToHip produces and the runtime lowering handles --
    // stays a hip op instead of failing this pass. See
    // isTosaExpressibleCompareOperand above.
    conversion.addDynamicallyLegalOp<EqualOp>(
        [](EqualOp op) { return !isTosaExpressibleCompare(op); });
    conversion.addDynamicallyLegalOp<LessOp>(
        [](LessOp op) { return !isTosaExpressibleCompare(op); });
    conversion.addDynamicallyLegalOp<AndOp>([](AndOp op) {
      return !isTosaExpressibleLogical(op, {op.getLhs(), op.getRhs()});
    });
    conversion.addDynamicallyLegalOp<OrOp>([](OrOp op) {
      return !isTosaExpressibleLogical(op, {op.getLhs(), op.getRhs()});
    });
    conversion.addDynamicallyLegalOp<NotOp>(
        [](NotOp op) { return !isTosaExpressibleLogical(op, {op.getX()}); });
    conversion.addDynamicallyLegalOp<SignOp>(
        [](SignOp op) { return !isTosaExpressibleSign(op); });
    // The four reductions this pass added are claimed by element type and
    // structure rather than outright, so an unsigned max/min, an f64 norm or a
    // multi-axis reduction -- all of which ONNX permits and OnnxToHip
    // preserves -- stays a hip op for the runtime lowering instead of failing
    // this pass and taking the fusible ops in the same function with it. The
    // predicates run matchHipReduceCore, which is the same structural match
    // the patterns run, so nothing is claimed that a pattern then refuses.
    //
    // reduce_sum and reduce_mean stay outright illegal: that is pre-existing
    // behaviour with its own coverage, and changing it is not in this change's
    // scope, though the same argument applies to them.
    conversion.addDynamicallyLegalOp<ReduceMaxOp>([](ReduceMaxOp op) {
      return !isTosaExpressibleReduce(op, isTosaReduceType);
    });
    conversion.addDynamicallyLegalOp<ReduceMinOp>([](ReduceMinOp op) {
      return !isTosaExpressibleReduce(op, isTosaReduceType);
    });
    conversion.addDynamicallyLegalOp<ReduceProdOp>([](ReduceProdOp op) {
      return !isTosaExpressibleReduce(op, isTosaReduceType);
    });
    conversion.addDynamicallyLegalOp<ReduceL2Op>([](ReduceL2Op op) {
      return !isTosaExpressibleReduce(op, isTosaExpressibleFloat);
    });
    // hip.tile and hip.pad both carry shape information as operands rather
    // than attributes, so whether they convert depends on that operand being
    // constant -- and hip.pad additionally on its mode. Neither can be claimed
    // outright: a reflect-mode pad or a computed `repeats` has to stay a hip op
    // for the runtime lowering instead of failing this pass and taking the
    // fusible ops in the same function with it.
    conversion.addDynamicallyLegalOp<hip::TileOp>(
        [](hip::TileOp op) { return !isTosaExpressibleTile(op); });
    conversion.addDynamicallyLegalOp<hip::PadOp>(
        [](hip::PadOp op) { return !isTosaExpressiblePad(op); });
    conversion.addDynamicallyLegalOp<hip::ResizeOp>(
        [](hip::ResizeOp op) { return !isTosaExpressibleResize(op); });
    conversion.addDynamicallyLegalOp<hip::ConstantOp>(
        [](hip::ConstantOp op) { return !isTosaExpressibleConstant(op); });
    conversion.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [](tensor::CollapseShapeOp op) { return !isStaticReshape(op); });
    conversion.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [](tensor::ExpandShapeOp op) { return !isStaticReshape(op); });
    conversion.addDynamicallyLegalOp<tensor::ExtractSliceOp>(
        [](tensor::ExtractSliceOp op) { return !isTosaExpressibleSlice(op); });
    conversion.addDynamicallyLegalOp<tensor::InsertSliceOp>(
        [](tensor::InsertSliceOp op) { return !matchStaticConcat(op); });
    conversion.addDynamicallyLegalOp<arith::ConstantOp>(
        [](arith::ConstantOp op) { return !isTosaExpressibleTensorConst(op); });
    conversion.addDynamicallyLegalOp<tensor::FromElementsOp>(
        [](tensor::FromElementsOp op) {
          return !isTosaExpressibleFromElements(op);
        });
    conversion.addDynamicallyLegalOp<tensor::SplatOp>(
        [](tensor::SplatOp op) { return !isTosaExpressibleSplat(op); });
    // Neither op is a FuseROCMlir anchor or a pointwise op, so neither reaches
    // this pass from buildRocMlirPipeline today -- only from a hand-written
    // rock.kernel. Declining therefore leaves the op in place rather than
    // failing the pass, which keeps the unsupported configurations testable.
    // If either becomes a fusion anchor this has to turn into a hard failure
    // like hip.div's: a hip op surviving inside a rock.kernel fails in rocMLIR,
    // it does not fall back to the HIP kernel.
    conversion.addDynamicallyLegalOp<GatherBlockQuantizedOp>(
        [](GatherBlockQuantizedOp op) {
          return !isTosaExpressibleGatherBlockQuantized(op);
        });
    conversion.addDynamicallyLegalOp<QMoEOp>(
        [](QMoEOp op) { return !isTosaExpressibleQMoE(op); });
    // Unlike the gathers, which are claimed outright, only reduction "none"
    // reaches tosa.scatter. Declining an accumulating mode is a routine
    // outcome rather than a defect, so those stay hip ops instead of failing
    // the pass.
    conversion.addDynamicallyLegalOp<ScatterElementsOp>(
        [](ScatterElementsOp op) { return op.getReduction() != "none"; });
    conversion.addDynamicallyLegalOp<ScatterNDOp>(
        [](ScatterNDOp op) { return op.getReduction() != "none"; });
    conversion.addDynamicallyLegalOp<TopKOp>(
        [](TopKOp op) { return !isTosaExpressibleTopK(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<
        ConvConverter, CausalConvWithStateConverter, MatMulConverter,
        GemmConverter, TransposeConverter, TileConverter, PadConverter,
        ResizeConverter, ConstantConverter, ExpandConverter,
        ReshapeConverter<tensor::CollapseShapeOp>,
        ReshapeConverter<tensor::ExpandShapeOp>, ExtractSliceConverter,
        InsertSliceConcatConverter, DivConverter,
        BinaryConverter<AddOp, tosa::AddOp>,
        BinaryConverter<SubOp, tosa::SubOp>,
        BinaryConverter<MinOp, tosa::MinimumOp>,
        BinaryConverter<MaxOp, tosa::MaximumOp>,
        BinaryConverter<MulOp, tosa::MulOp>,
        // Bitwise rather than logical: only the bitwise forms have a rocMLIR
        // lowering, and on i1 the two coincide.
        BinaryConverter<AndOp, tosa::BitwiseAndOp, /*BoolOnly=*/true>,
        BinaryConverter<OrOp, tosa::BitwiseOrOp, /*BoolOnly=*/true>,
        ComparisonConverter<EqualOp, tosa::EqualOp>,
        // TOSA has no `less`, so the operands are swapped into a greater.
        ComparisonConverter<LessOp, tosa::GreaterOp, /*SwapOperands=*/true>,
        UnaryConverter<AbsOp, tosa::AbsOp>,
        UnaryConverter<NegOp, tosa::NegateOp>,
        UnaryConverter<CeilOp, tosa::CeilOp, /*FloatOnly=*/true>,
        UnaryConverter<FloorOp, tosa::FloorOp, /*FloatOnly=*/true>,
        UnaryConverter<ExpOp, tosa::ExpOp, /*FloatOnly=*/true>,
        UnaryConverter<LogOp, tosa::LogOp, /*FloatOnly=*/true>,
        UnaryConverter<SinOp, tosa::SinOp, /*FloatOnly=*/true>,
        UnaryConverter<CosOp, tosa::CosOp, /*FloatOnly=*/true>,
        UnaryConverter<TanhOp, tosa::TanhOp, /*FloatOnly=*/true>,
        UnaryConverter<ErfOp, tosa::ErfOp, /*FloatOnly=*/true>,
        UnaryConverter<SigmoidOp, tosa::SigmoidOp, /*FloatOnly=*/true>,
        UnaryConverter<ReciprocalOp, tosa::ReciprocalOp,
                       /*FloatOnly=*/true>,
        LogicalNotConverter, RoundConverter, ModConverter, AtanConverter,
        SqrtConverter, SoftplusConverter, GeluConverter, BiasGeluConverter,
        FastGeluConverter, SiluConverter, SwishConverter, SignConverter,
        WhereConverter, LeakyReluConverter, SoftmaxConverter,
        ReduceConverter<ReduceSumOp, tosa::ReduceSumOp>,
        ReduceConverter<ReduceMaxOp, tosa::ReduceMaxOp>,
        ReduceConverter<ReduceMinOp, tosa::ReduceMinOp>,
        ReduceConverter<ReduceProdOp, tosa::ReduceProductOp>,
        ReduceMeanConverter, ReduceL2Converter, CastConverter,
        DequantizeLinearConverter, QuantizeLinearConverter,
        MatMulNBitsConverter, GatherConverter, GatherElementsConverter,
        GatherNDConverter, ScatterElementsConverter, ScatterNDConverter,
        GatherBlockQuantizedConverter, TopKConverter, QMoEConverter,
        GridSampleConverter, SizeConverter, TensorConstConverter,
        FromElementsConverter, SplatConverter, OneHotConverter, RangeConverter,
        RopeConverter, GqaConverter, MhaConverter, RmsNormConverter,
        LayerNormConverter, InstanceNormConverter, SkipRmsNormConverter>(ctx);

    if (failed(applyPartialConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
