/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Tosa/Utils/ConversionUtils.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <type_traits>

namespace mlir::hip {

#define GEN_PASS_DEF_CONVERTHIPTOTOSAPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// TOSA broadcasts size-1 dimensions only and requires every operand to carry
// the result's rank, so hip's ONNX/NumPy rank-extending broadcast does not
// always survive a 1-1 mapping.
bool isTosaCompatibleOperand(Value operand, RankedTensorType resultType) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape())
    return false;
  if (operandType.getElementType() != resultType.getElementType())
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

// tosa.matmul requires both operands to be rank-3 (batch, M/K, K/N). hip.matmul
// accepts NumPy-style broadcasting where B may be rank-2. Prepend a unit batch
// dimension to any rank-2 operand via tosa.reshape; a batch of 1 is broadcast
// against the other operand's batch by tosa.matmul.
static Value reshapeTo3D(Value input, ConversionPatternRewriter &rewriter) {
  auto type = dyn_cast<RankedTensorType>(input.getType());
  if (!type || type.getRank() != 2)
    return input;

  SmallVector<int64_t> shape(type.getShape());
  shape.insert(shape.begin(), 1);
  auto shapeConst = tosa::ConstShapeOp::create(
      rewriter, rewriter.getUnknownLoc(),
      tosa::shapeType::get(rewriter.getContext(), shape.size()),
      rewriter.getIndexTensorAttr(shape));
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
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected a static rank-3 tensor");

    Value a = reshapeTo3D(adaptor.getA(), rewriter);
    Value b = reshapeTo3D(adaptor.getB(), rewriter);
    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    if (!aType || aType.getRank() != 3 || !bType || bType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "operands not rank-3");

    // The quant-info builder appends the (zero) zero-point operands that
    // tosa.matmul requires for float inputs.
    rewriter.replaceOpWithNewOp<tosa::MatMulOp>(op, resultType, a, b);
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

class HipToTosaPass : public impl::ConvertHipToTosaPassBase<HipToTosaPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!funcOp->hasAttr("rock.kernel"))
      return;

    MLIRContext *ctx = &getContext();

    ConversionTarget conversion(*ctx);
    conversion.addIllegalDialect<HipDialect>();
    conversion.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    conversion.addLegalOp<ub::PoisonOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<MatMulConverter,
                 BinaryConverter<AddOp, tosa::AddOp>,
                 BinaryConverter<SubOp, tosa::SubOp>,
                 BinaryConverter<MinOp, tosa::MinimumOp>,
                 BinaryConverter<MaxOp, tosa::MaximumOp>,
                 BinaryConverter<MulOp, tosa::MulOp>,
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
                                /*FloatOnly=*/true>>(ctx);

    if (failed(applyFullConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
