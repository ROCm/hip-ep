/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
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

// Zero-pad `input` by `padding`, given as [before, after] per dimension.
static Value zeroPadTo(Value input, ArrayRef<int64_t> padding,
                       ArrayRef<int64_t> shape,
                       ConversionPatternRewriter &rewriter, Location loc) {
  auto type = cast<RankedTensorType>(input.getType());
  Type elemType = type.getElementType();
  auto zeroType = RankedTensorType::get({1}, elemType);
  Value zero = tosa::ConstOp::create(
      rewriter, loc, zeroType,
      DenseElementsAttr::get(zeroType, rewriter.getZeroAttr(elemType)));
  return tosa::PadOp::create(rewriter, loc, type.clone(shape), input,
                             createConstShape(rewriter, loc, padding), zero);
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
// element-type check rejects. hip.div does not either, since TOSA has no
// single divide; DivConverter below spells one out per element type.
//
// tosa.maximum and tosa.minimum additionally carry a nan_mode attribute, but
// ODS defaults it to PROPAGATE, which is what ONNX Max/Min do, so the
// two-operand builder below is correct for them unchanged.
template <typename HipOpTy, typename TosaOpTy>
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
FailureOr<int32_t> matchSingleReduceAxis(Value axes, int64_t rank,
                                         int64_t noopWithEmptyAxes,
                                         ConversionPatternRewriter &rewriter,
                                         Operation *op) {
  SmallVector<int64_t, 4> axesVals;
  if (!extractConstantInts(axes, axesVals)) {
    (void)rewriter.notifyMatchFailure(op, "axes must be a constant");
    return failure();
  }
  if (axesVals.empty()) {
    (void)rewriter.notifyMatchFailure(
        op, noopWithEmptyAxes ? "empty axes identity"
                              : "empty axes reduce-all is not a single tosa "
                                "reduce");
    return failure();
  }
  if (axesVals.size() != 1) {
    (void)rewriter.notifyMatchFailure(op, "tosa reduce supports a single axis");
    return failure();
  }

  int64_t axis = axesVals[0];
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    (void)rewriter.notifyMatchFailure(op, "axis out of range");
    return failure();
  }
  return static_cast<int32_t>(axis);
}

RankedTensorType keepdimsReduceType(RankedTensorType dataType, int32_t axis) {
  SmallVector<int64_t> shape(dataType.getShape().begin(),
                             dataType.getShape().end());
  shape[axis] = 1;
  return RankedTensorType::get(shape, dataType.getElementType());
}

// TOSA has no reduce_mean. Scale by 1/N then reduce_sum on one axis
// (keepdims=1). Used by hip.global_pool average mode, hip.reduce_mean, and
// RMS/LN over several axes.
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

LogicalResult matchHipReduce(Operation *op, Value data, Value axes,
                             int64_t keepdims, int64_t noopWithEmptyAxes,
                             ConversionPatternRewriter &rewriter,
                             RankedTensorType &resultType, Value &dataOut,
                             int32_t &axis, bool &keepdimsOut, bool &identity) {
  identity = false;
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected tensor mode");
  resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || !dataType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked data");
  if (dataType.getRank() < 1)
    return rewriter.notifyMatchFailure(op, "tosa reduce requires rank >= 1");

  SmallVector<int64_t, 4> axesVals;
  if (!extractConstantInts(axes, axesVals))
    return rewriter.notifyMatchFailure(op, "axes must be a constant");
  if (axesVals.empty() && noopWithEmptyAxes) {
    if (data.getType() != resultType)
      return rewriter.notifyMatchFailure(op, "identity type mismatch");
    identity = true;
    dataOut = data;
    keepdimsOut = keepdims != 0;
    return success();
  }

  FailureOr<int32_t> axisOr = matchSingleReduceAxis(
      axes, dataType.getRank(), noopWithEmptyAxes, rewriter, op);
  if (failed(axisOr))
    return failure();
  axis = *axisOr;
  keepdimsOut = keepdims != 0;
  dataOut = data;
  identity = false;
  auto reducedTy = keepdimsReduceType(dataType, axis);
  if (keepdimsOut && resultType != reducedTy)
    return rewriter.notifyMatchFailure(op, "keepdims result type mismatch");
  if (!keepdimsOut && resultType.getRank() != dataType.getRank() - 1)
    return rewriter.notifyMatchFailure(op, "keepdims=0 rank mismatch");
  return success();
}

// hip.reduce_sum -> tosa.reduce_sum. One constant axis, TOSA keepdims=1,
// optional reshape.
struct ReduceSumConverter final : public OpConversionPattern<ReduceSumOp> {
  using OpConversionPattern<ReduceSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
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
    auto reducedTy =
        keepdimsReduceType(cast<RankedTensorType>(data.getType()), axis);
    auto reduced =
        tosa::ReduceSumOp::create(rewriter, op.getLoc(), reducedTy, data,
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

// rocMLIR's rock-tosa-to-elementwise lowers these tosa.custom names; a plain
// tosa.cast float->int is illegal there.
constexpr StringLiteral kRockCustomOpDomain = "rocmlir";
constexpr StringLiteral kRockUnsignedCast = "unsigned_cast";
constexpr StringLiteral kRockFpToIntCast = "fp_to_int_cast";
constexpr int64_t kPoolAverage = 0;
constexpr int64_t kPoolMax = 1;
constexpr int64_t kPoolLp = 2;

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
    APInt first = start.getSplatValue<APInt>();
    APInt step = delta.getSplatValue<APInt>();
    SmallVector<APInt> elements;
    for (int64_t i : llvm::seq<int64_t>(length))
      elements.push_back(first + step * APInt(first.getBitWidth(), i));
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

FailureOr<Value> emitKeepdimsReduceMax(Value data, int32_t axis,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Operation *op) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || !dataType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked data");
  if (axis < 0 || axis >= dataType.getRank())
    return rewriter.notifyMatchFailure(op, "reduce axis out of range");
  auto reducedTy = keepdimsReduceType(dataType, axis);
  return tosa::ReduceMaxOp::create(rewriter, loc, reducedTy, data,
                                   rewriter.getI32IntegerAttr(axis))
      .getResult();
}

FailureOr<Value> emitKeepdimsReduceSum(Value data, int32_t axis,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, Operation *op) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || !dataType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected static ranked data");
  if (axis < 0 || axis >= dataType.getRank())
    return rewriter.notifyMatchFailure(op, "reduce axis out of range");
  auto reducedTy = keepdimsReduceType(dataType, axis);
  return tosa::ReduceSumOp::create(rewriter, loc, reducedTy, data,
                                   rewriter.getI32IntegerAttr(axis))
      .getResult();
}

// Reduce every spatial axis [2, rank) keepdims. Global pooling is NCHW with
// spatial dims collapsed to 1.
FailureOr<Value> reduceSpatialAxes(
    Value input, ConversionPatternRewriter &rewriter, Location loc,
    Operation *op,
    llvm::function_ref<FailureOr<Value>(Value, int32_t)> reduceOne) {
  auto ty = dyn_cast<RankedTensorType>(input.getType());
  if (!ty || !ty.hasStaticShape() || ty.getRank() < 3)
    return rewriter.notifyMatchFailure(
        op, "global pool expects a static rank >= 3 tensor");
  Value cur = input;
  for (int64_t a : llvm::seq<int64_t>(2, ty.getRank())) {
    FailureOr<Value> next = reduceOne(cur, static_cast<int32_t>(a));
    if (failed(next))
      return failure();
    cur = *next;
  }
  return cur;
}

// Global pooling is a keepdims reduction over every spatial axis, which TOSA
// spells out; only the element type limits it.
bool isTosaExpressibleGlobalPool(GlobalPoolOp op) {
  if (op.getNumResults() != 1)
    return false;
  auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
    return false;
  if (inTy.getElementType() != outTy.getElementType() ||
      inTy.getRank() != outTy.getRank() || inTy.getRank() < 3)
    return false;
  if (inTy.getDimSize(0) != outTy.getDimSize(0) ||
      inTy.getDimSize(1) != outTy.getDimSize(1))
    return false;
  for (int64_t i : llvm::seq<int64_t>(2, outTy.getRank()))
    if (outTy.getDimSize(i) != 1)
      return false;

  int64_t mode = op.getMode();
  if (mode == kPoolMax)
    return isa<FloatType, IntegerType>(inTy.getElementType());
  if (mode == kPoolLp && op.getP() <= 0)
    return false;
  // The reduce-mean scale and the LP powers both need a float tensor.
  return (mode == kPoolAverage || mode == kPoolLp) &&
         isa<FloatType>(inTy.getElementType());
}

struct GlobalPoolConverter final : public OpConversionPattern<GlobalPoolOp> {
  using OpConversionPattern<GlobalPoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GlobalPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isTosaExpressibleGlobalPool(op))
      return rewriter.notifyMatchFailure(op, "no TOSA reduction spelling");

    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Value input = adaptor.getInput();
    auto inTy = cast<RankedTensorType>(input.getType());
    Location loc = op.getLoc();
    int64_t mode = op.getMode();
    FailureOr<Value> y;
    if (mode == kPoolAverage) {
      y = reduceSpatialAxes(input, rewriter, loc, op, [&](Value v, int32_t a) {
        return emitTosaKeepdimsReduceMean(v, a, rewriter, loc, op);
      });
    } else if (mode == kPoolMax) {
      y = reduceSpatialAxes(input, rewriter, loc, op, [&](Value v, int32_t a) {
        return emitKeepdimsReduceMax(v, a, rewriter, loc, op);
      });
    } else {
      Value abs = tosa::AbsOp::create(rewriter, loc, inTy, input);
      Value pSplat =
          createSplatFloat(rewriter, loc, inTy, static_cast<double>(op.getP()));
      Value powered = tosa::PowOp::create(rewriter, loc, inTy, abs, pSplat);
      FailureOr<Value> summed = reduceSpatialAxes(
          powered, rewriter, loc, op, [&](Value v, int32_t a) {
            return emitKeepdimsReduceSum(v, a, rewriter, loc, op);
          });
      if (failed(summed))
        return failure();
      auto sumTy = cast<RankedTensorType>((*summed).getType());
      double invP = 1.0 / static_cast<double>(op.getP());
      Value invPSplat = createSplatFloat(rewriter, loc, sumTy, invP);
      y = tosa::PowOp::create(rewriter, loc, sumTy, *summed, invPSplat)
              .getResult();
    }
    if (failed(y))
      return failure();
    if ((*y).getType() != resultTy)
      return rewriter.notifyMatchFailure(
          op, "global pool result shape disagrees with the reduced tensor");
    rewriter.replaceOp(op, *y);
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

// Geometry of the 2D TOSA pool that implements a hip.pool, normalized to NCHW.
struct TosaPoolWindow {
  SmallVector<int64_t, 2> kernel;
  SmallVector<int64_t, 2> stride;
  SmallVector<int64_t, 2> padBefore;
  SmallVector<int64_t, 2> padAfter;
  SmallVector<int64_t, 2> crop;
  // Rank-4 NCHW shapes the pool runs on. A 1D pool is reshaped to these.
  SmallVector<int64_t, 4> inShape;
  SmallVector<int64_t, 4> outShape;
  // tosa.avg_pool2d always divides by the window's in-bounds coverage of its
  // own pad attribute, which is ONNX count_include_pad=0. The other two
  // divisors instead need the padding materialized as explicit zeros, leaving
  // the pool unpadded so it divides by the full kernel volume: that is
  // count_include_pad=1 directly, and for LP it makes the pad cells
  // contribute 0 to the window sum, as ONNX requires.
  bool materializePad = false;
};

// TOSA pooling is 2D, undilated, and has no indices output, so only part of
// hip.pool's ONNX-shaped configuration space has a TOSA spelling. The rest has
// no entry here and stays a hip.pool that ends the fusion chain.
std::optional<TosaPoolWindow> getTosaPoolWindow(PoolOp op) {
  // MaxPool's Indices result has no TOSA equivalent.
  if (op.getNumResults() != 1)
    return std::nullopt;
  auto inTy = dyn_cast<RankedTensorType>(op.getInput().getType());
  auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inTy || !outTy || !inTy.hasStaticShape() || !outTy.hasStaticShape())
    return std::nullopt;
  if (inTy.getElementType() != outTy.getElementType() ||
      inTy.getRank() != outTy.getRank())
    return std::nullopt;

  int64_t mode = op.getPoolMode();
  Type elemType = inTy.getElementType();
  if (mode == kPoolMax) {
    if (!isa<FloatType, IntegerType>(elemType))
      return std::nullopt;
  } else if (mode == kPoolAverage || mode == kPoolLp) {
    // The f32 accumulator and LP's tosa.pow both need a float tensor.
    if (!isa<FloatType>(elemType))
      return std::nullopt;
    if (mode == kPoolLp && op.getP() <= 0)
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  // ceil_mode puts the output past what `pads` describes, and TOSA has no
  // column-major indices.
  if (op.getCeilMode() != 0 || op.getStorageOrder() != 0)
    return std::nullopt;

  int64_t spatialRank = inTy.getRank() - 2;
  // TOSA has no 3D pooling.
  if (spatialRank != 1 && spatialRank != 2)
    return std::nullopt;
  SmallVector<int64_t> kernel = getI64Values(op.getKernelShape());
  SmallVector<int64_t> stride = getI64Values(op.getStrides());
  SmallVector<int64_t> pads = getI64Values(op.getPads());
  SmallVector<int64_t> dilations = getI64Values(op.getDilations());
  if (kernel.size() != size_t(spatialRank) ||
      stride.size() != size_t(spatialRank) ||
      dilations.size() != size_t(spatialRank) ||
      pads.size() != size_t(2 * spatialRank))
    return std::nullopt;
  // TOSA pooling windows are dense.
  if (llvm::any_of(dilations, [](int64_t d) { return d != 1; }))
    return std::nullopt;

  TosaPoolWindow window;
  window.crop = {0, 0};
  window.materializePad =
      mode == kPoolLp || (mode == kPoolAverage && op.getCountIncludePad() != 0);
  if (spatialRank == 1) {
    // Pool a 1D window as a 2D one over a unit leading spatial dim.
    window.kernel = {1, kernel[0]};
    window.stride = {1, stride[0]};
    window.padBefore = {0, pads[0]};
    window.padAfter = {0, pads[1]};
    window.inShape = {inTy.getDimSize(0), inTy.getDimSize(1), 1,
                      inTy.getDimSize(2)};
    window.outShape = {outTy.getDimSize(0), outTy.getDimSize(1), 1,
                       outTy.getDimSize(2)};
  } else {
    window.kernel = {kernel[0], kernel[1]};
    window.stride = {stride[0], stride[1]};
    window.padBefore = {pads[0], pads[1]};
    window.padAfter = {pads[2], pads[3]};
    window.inShape.assign(inTy.getShape().begin(), inTy.getShape().end());
    window.outShape.assign(outTy.getShape().begin(), outTy.getShape().end());
  }
  if (window.inShape[0] != window.outShape[0] ||
      window.inShape[1] != window.outShape[1])
    return std::nullopt;

  // ONNX floors the output size, dropping a trailing partial window, while
  // TOSA requires the window arithmetic to divide exactly. Absorb the
  // remainder the way ConvConverter does: shrink the trailing pad, and crop
  // the input for whatever the pad cannot cover. Either way only elements no
  // remaining window reads are removed, so the trailing window keeps the pad
  // overlap -- and therefore the AVERAGE divisor -- that ONNX gives it.
  for (int64_t dim : llvm::seq<int64_t>(2)) {
    if (window.kernel[dim] < 1 || window.stride[dim] < 1)
      return std::nullopt;
    if (window.padBefore[dim] < 0 || window.padAfter[dim] < 0)
      return std::nullopt;
    int64_t inputSize = window.inShape[dim + 2];
    int64_t span = inputSize - 1 + window.padBefore[dim] +
                   window.padAfter[dim] - (window.kernel[dim] - 1);
    if (span < 0)
      return std::nullopt;
    int64_t remainder = span % window.stride[dim];
    int64_t fromPad = std::min(window.padAfter[dim], remainder);
    window.padAfter[dim] -= fromPad;
    window.crop[dim] = remainder - fromPad;
    if (window.crop[dim] >= inputSize)
      return std::nullopt;
    if ((span - remainder) / window.stride[dim] + 1 != window.outShape[dim + 2])
      return std::nullopt;
  }
  return window;
}

bool isTosaExpressiblePool(PoolOp op) {
  return getTosaPoolWindow(op).has_value();
}

struct PoolConverter final : public OpConversionPattern<PoolOp> {
  using OpConversionPattern<PoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<TosaPoolWindow> window = getTosaPoolWindow(op);
    if (!window)
      return rewriter.notifyMatchFailure(op, "no TOSA pooling spelling");

    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Type elemType = resultTy.getElementType();
    int64_t mode = op.getPoolMode();
    ArrayRef<int64_t> inShape = window->inShape;
    ArrayRef<int64_t> outShape = window->outShape;

    Value input = adaptor.getInput();
    if (resultTy.getRank() != 4)
      input = reshapeTo(input, inShape, rewriter);

    // tosa.max_pool2d / avg_pool2d are NHWC; hip.pool is ONNX's NCHW.
    // rocMLIR folds these transposes into layout metadata.
    Value nhwc =
        transposeTo(input, {inShape[0], inShape[2], inShape[3], inShape[1]},
                    {0, 2, 3, 1}, rewriter, loc);
    SmallVector<int64_t> nhwcShape = {inShape[0], inShape[2] - window->crop[0],
                                      inShape[3] - window->crop[1], inShape[1]};
    if (window->crop[0] != 0 || window->crop[1] != 0)
      nhwc = sliceTo(nhwc, nhwcShape, rewriter, loc);

    SmallVector<int64_t, 2> padBefore = window->padBefore;
    SmallVector<int64_t, 2> padAfter = window->padAfter;
    if (window->materializePad &&
        (padBefore[0] || padBefore[1] || padAfter[0] || padAfter[1])) {
      nhwcShape[1] += padBefore[0] + padAfter[0];
      nhwcShape[2] += padBefore[1] + padAfter[1];
      nhwc = zeroPadTo(
          nhwc,
          {0, 0, padBefore[0], padAfter[0], padBefore[1], padAfter[1], 0, 0},
          nhwcShape, rewriter, loc);
      padBefore = {0, 0};
      padAfter = {0, 0};
    }

    // LP is pow(sum(pow(|x|, p)), 1/p), and TOSA's only windowed sum is the
    // average, so the window mean is scaled back up by the kernel volume.
    auto nhwcInTy = cast<RankedTensorType>(nhwc.getType());
    double p = static_cast<double>(op.getP());
    if (mode == kPoolLp) {
      Value abs = tosa::AbsOp::create(rewriter, loc, nhwcInTy, nhwc);
      nhwc = tosa::PowOp::create(rewriter, loc, nhwcInTy, abs,
                                 createSplatFloat(rewriter, loc, nhwcInTy, p));
    }

    auto nhwcOutTy =
        resultTy.clone({outShape[0], outShape[2], outShape[3], outShape[1]});
    auto padAttr = rewriter.getDenseI64ArrayAttr(
        {padBefore[0], padAfter[0], padBefore[1], padAfter[1]});
    Value pooled;
    if (mode == kPoolMax) {
      pooled =
          tosa::MaxPool2dOp::create(
              rewriter, loc, nhwcOutTy, nhwc, window->kernel, window->stride,
              ArrayRef<int64_t>{padBefore[0], padAfter[0], padBefore[1],
                                padAfter[1]},
              tosa::NanPropagationMode::PROPAGATE)
              .getResult();
    } else {
      Type accType = elemType.isF64() ? elemType : rewriter.getF32Type();
      pooled = tosa::AvgPool2dOp::create(
                   rewriter, loc, nhwcOutTy, nhwc,
                   rewriter.getDenseI64ArrayAttr(window->kernel),
                   rewriter.getDenseI64ArrayAttr(window->stride), padAttr,
                   TypeAttr::get(accType))
                   .getResult();
    }
    if (mode == kPoolLp) {
      double volume =
          static_cast<double>(window->kernel[0] * window->kernel[1]);
      Value sum = tosa::MulOp::create(
          rewriter, loc, nhwcOutTy, pooled,
          createSplatFloat(rewriter, loc, nhwcOutTy, volume),
          createZeroMulShift(rewriter, loc));
      pooled = tosa::PowOp::create(
          rewriter, loc, nhwcOutTy, sum,
          createSplatFloat(rewriter, loc, nhwcOutTy, 1.0 / p));
    }

    Value result = transposeTo(pooled, outShape, {0, 3, 1, 2}, rewriter, loc);
    if (resultTy.getRank() != 4)
      result = reshapeTo(result, resultTy.getShape(), rewriter);
    rewriter.replaceOp(op, result);
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
  if (op.getNearestMode() != 0)
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

    // ONNX permits indices in [-K, K-1]. TOSA requires non-negative in-range
    // indices, so normalize the negative half before gathering.
    auto canonicalIndicesTy =
        RankedTensorType::get({n, w}, rewriter.getI32Type());
    Value zero = createSplatInt(rewriter, loc, canonicalIndicesTy, 0);
    Value extent = createSplatInt(rewriter, loc, canonicalIndicesTy, k);
    Value isNegative = tosa::GreaterOp::create(
        rewriter, loc, RankedTensorType::get({n, w}, rewriter.getI1Type()),
        zero, indices);
    Value wrapped =
        tosa::AddOp::create(rewriter, loc, canonicalIndicesTy, indices, extent);
    indices = tosa::SelectOp::create(rewriter, loc, canonicalIndicesTy,
                                     isNegative, wrapped, indices);

    auto gatheredTy = RankedTensorType::get({n, w, c}, dataTy.getElementType());
    Value gathered =
        tosa::GatherOp::create(rewriter, loc, gatheredTy, values, indices);
    rewriter.replaceOp(op, reshapeTo(gathered, resultTy.getShape(), rewriter));
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
  Value hi =
      tosa::LogicalRightShiftOp::create(rewriter, loc, i32Ty, asI32, shift);

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
    conversion.addIllegalOp<
        ConvOp, MatmulOp, GemmOp, TransposeOp, AddOp, SubOp, MinOp, MaxOp,
        MulOp, DivOp, AbsOp, NegOp, CeilOp, FloorOp, ExpOp, LogOp, SinOp, CosOp,
        TanhOp, ErfOp, SigmoidOp, ReciprocalOp, SqrtOp, WhereOp, LeakyReluOp,
        MiopenSoftmaxOp, ReduceSumOp, ReduceMeanOp, CastOp, QuantizeLinearOp,
        DequantizeLinearOp, MatMulNBitsOp, GatherOp, RangeOp, RopeOp, GqaOp,
        MultiHeadAttentionOp, RmsNormOp, LayerNormOp, InstanceNormOp,
        SkipRmsNormOp>();
    // tosa.matmul (and other tosa ops) are not destination-passing, so
    // MatMulConverter drops each hip op's DPS `outs` operand. The
    // `tensor.empty` that fed it is then dead, but a full conversion still
    // requires every remaining op to be legal -- the framework does not DCE
    // this pre-existing op on its own. Mark it legal so conversion succeeds;
    // the canonicalizer that follows this pass removes the dead empty.
    conversion.addLegalOp<ub::PoisonOp, tensor::EmptyOp>();
    conversion.addDynamicallyLegalOp<ExpandOp>(
        [](ExpandOp op) { return !isTosaExpressibleExpand(op); });
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
    conversion.addDynamicallyLegalOp<PoolOp>(
        [](PoolOp op) { return !isTosaExpressiblePool(op); });
    conversion.addDynamicallyLegalOp<GlobalPoolOp>(
        [](GlobalPoolOp op) { return !isTosaExpressibleGlobalPool(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<
        ConvConverter, MatMulConverter, GemmConverter, TransposeConverter,
        TileConverter, PadConverter, ResizeConverter, ConstantConverter,
        ExpandConverter, ReshapeConverter<tensor::CollapseShapeOp>,
        ReshapeConverter<tensor::ExpandShapeOp>, ExtractSliceConverter,
        DivConverter, BinaryConverter<AddOp, tosa::AddOp>,
        BinaryConverter<SubOp, tosa::SubOp>,
        BinaryConverter<MinOp, tosa::MinimumOp>,
        BinaryConverter<MaxOp, tosa::MaximumOp>,
        BinaryConverter<MulOp, tosa::MulOp>, UnaryConverter<AbsOp, tosa::AbsOp>,
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
        SqrtConverter, WhereConverter, LeakyReluConverter, SoftmaxConverter,
        ReduceSumConverter, ReduceMeanConverter, CastConverter,
        DequantizeLinearConverter, QuantizeLinearConverter,
        MatMulNBitsConverter, GatherConverter, RangeConverter, RopeConverter,
        GqaConverter, MhaConverter, RmsNormConverter, LayerNormConverter,
        InstanceNormConverter, SkipRmsNormConverter, GlobalPoolConverter,
        PoolConverter>(ctx);

    if (failed(applyPartialConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
