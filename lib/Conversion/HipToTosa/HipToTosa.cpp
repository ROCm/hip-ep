/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

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
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

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
      rewriter, loc, tosa::shapeType::get(rewriter.getContext(), extents.size()),
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

    rewriter.replaceOp(
        op, reshapeTo(matmul, resultType.getShape(), rewriter));
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

// Covers the hip ops whose operands are (ctx, lhs, rhs, output) and whose TOSA
// counterpart preserves the element type. Comparisons (hip.equal, hip.less) do
// not belong here: they produce i1, which isTosaCompatibleOperand's
// element-type check rejects. hip.div does not either, since TOSA has no
// floating-point divide (tosa.intdiv is i32/i64 only, and floats decompose
// into tosa.reciprocal plus tosa.mul).
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
    auto dataType = cast<RankedTensorType>(data.getType());
    if (!isa<FloatType>(dataType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "tosa reduce_mean lowering requires a float tensor");
    int64_t n = dataType.getDimSize(axis);
    if (n <= 0)
      return rewriter.notifyMatchFailure(op, "reduce axis extent must be > 0");

    Location loc = op.getLoc();
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
    auto reduced = tosa::ReduceSumOp::create(rewriter, loc, reducedTy, scaled,
                                             rewriter.getI32IntegerAttr(axis));
    replaceWithTosaReduce(op, reduced, resultType, keepdims, rewriter);
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
      auto scalarType =
          RankedTensorType::get({}, paramType.getElementType());
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

// hip.cast is 1-1 with tosa.cast when both endpoints are signed or float.
// Float->int and any unsigned endpoint go through rocMLIR tosa.custom, because
// rock-tosa-to-elementwise rejects tosa.cast float->int.
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
    // outlined kernel keeps the fusion anchor it was built around -- hip.conv,
    // hip.pool, hip.global_pool -- which rocMLIR consumes and this pass must
    // not touch, and it carries the ops feeding the anchors' DPS operands:
    // ub.poison for the !hip.context and tensor.empty for each outs buffer.
    // Full conversion legalizes every op in the region, so each of those has
    // to be enumerated as legal or the pass fails on IR it never meant to
    // convert. Listing the illegal ops instead keeps the useful half of the
    // guarantee: a hip op this pass claims still has to convert or the pass
    // fails.
    ConversionTarget conversion(*ctx);
    conversion.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    conversion
        .addIllegalOp<MatmulOp, TransposeOp, AddOp, SubOp, MinOp, MaxOp, MulOp,
                      AbsOp, NegOp, CeilOp, FloorOp, ExpOp, LogOp, SinOp, CosOp,
                      TanhOp, ErfOp, SigmoidOp, ReciprocalOp, SqrtOp, WhereOp,
                      LeakyReluOp, MiopenSoftmaxOp, ReduceSumOp, ReduceMeanOp,
                      CastOp, QuantizeLinearOp, DequantizeLinearOp>();
    // tosa.matmul (and other tosa ops) are not destination-passing, so
    // MatMulConverter drops each hip op's DPS `outs` operand. The
    // `tensor.empty` that fed it is then dead, but a full conversion still
    // requires every remaining op to be legal -- the framework does not DCE
    // this pre-existing op on its own. Mark it legal so conversion succeeds;
    // the canonicalizer that follows this pass removes the dead empty.
    conversion.addLegalOp<ub::PoisonOp, tensor::EmptyOp>();
    conversion.addDynamicallyLegalOp<ExpandOp>(
        [](ExpandOp op) { return !isTosaExpressibleExpand(op); });
    conversion.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
        [](tensor::CollapseShapeOp op) { return !isStaticReshape(op); });
    conversion.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
        [](tensor::ExpandShapeOp op) { return !isStaticReshape(op); });
    conversion.addDynamicallyLegalOp<tensor::ExtractSliceOp>(
        [](tensor::ExtractSliceOp op) { return !isTosaExpressibleSlice(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<
        MatMulConverter, TransposeConverter, ExpandConverter,
        ReshapeConverter<tensor::CollapseShapeOp>,
        ReshapeConverter<tensor::ExpandShapeOp>, ExtractSliceConverter,
        BinaryConverter<AddOp, tosa::AddOp>,
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
        DequantizeLinearConverter, QuantizeLinearConverter>(ctx);

    if (failed(applyPartialConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
