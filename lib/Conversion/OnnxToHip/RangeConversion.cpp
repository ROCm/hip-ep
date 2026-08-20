/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include <cmath>
#include <limits>

namespace mlir {
namespace hip {
namespace {

static int64_t getRangeRuntimeDataType(Type type) {
  if (type.isF32())
    return 0; // HIPDNN_EP_DATATYPE_FLOAT
  if (type.isInteger(32))
    return 3; // HIPDNN_EP_DATATYPE_INT32
  if (type.isInteger(64))
    return 4; // HIPDNN_EP_DATATYPE_INT64
  if (type.isF64())
    return 6; // HIPDNN_EP_DATATYPE_DOUBLE
  if (type.isInteger(16))
    return 8; // HIPDNN_EP_DATATYPE_INT16
  return -1;
}

template <typename T>
static FailureOr<int64_t> computeConstantFloatCount(T start, T limit, T delta) {
  if (!std::isfinite(start) || !std::isfinite(limit) || !std::isfinite(delta) ||
      delta == T{0})
    return failure();
  if ((delta > T{0} && limit <= start) || (delta < T{0} && limit >= start))
    return 0;
  T quotient = (limit - start) / delta;
  if (!std::isfinite(quotient) || quotient < T{0})
    return failure();
  long double count = std::ceil(static_cast<long double>(quotient));
  if (!std::isfinite(count) || count < 0 ||
      count > static_cast<long double>(std::numeric_limits<int64_t>::max()))
    return failure();
  return static_cast<int64_t>(count);
}

static FailureOr<std::optional<int64_t>>
tryComputeConstantRangeCount(Value start, Value limit, Value delta,
                             Type elemTy) {
  DenseElementsAttr startDense = getConstantDense(start);
  DenseElementsAttr limitDense = getConstantDense(limit);
  DenseElementsAttr deltaDense = getConstantDense(delta);
  if (!startDense || !limitDense || !deltaDense)
    return std::optional<int64_t>();
  if (startDense.getNumElements() != 1 || limitDense.getNumElements() != 1 ||
      deltaDense.getNumElements() != 1)
    return failure();

  if (isa<IntegerType>(elemTy)) {
    APInt s = *startDense.getValues<APInt>().begin();
    APInt l = *limitDense.getValues<APInt>().begin();
    APInt d = *deltaDense.getValues<APInt>().begin();
    unsigned width = std::max<unsigned>(128, elemTy.getIntOrFloatBitWidth());
    s = s.sext(width);
    l = l.sext(width);
    d = d.sext(width);
    if (d.isZero())
      return failure();
    APInt count(width, 0, /*isSigned=*/true);
    if ((d.isStrictlyPositive() && l.sgt(s)) || (d.isNegative() && l.slt(s))) {
      APInt difference = d.isStrictlyPositive() ? l - s : s - l;
      APInt step = d.isStrictlyPositive() ? d : -d;
      APInt quotient = difference.sdiv(step);
      APInt remainder = difference.srem(step);
      count = quotient + APInt(width, !remainder.isZero());
    }
    if (!count.isSignedIntN(64) || count.isNegative())
      return failure();
    return std::optional<int64_t>(count.getSExtValue());
  }

  APFloat s = *startDense.getValues<APFloat>().begin();
  APFloat l = *limitDense.getValues<APFloat>().begin();
  APFloat d = *deltaDense.getValues<APFloat>().begin();
  FailureOr<int64_t> count =
      elemTy.isF32()
          ? computeConstantFloatCount(s.convertToFloat(), l.convertToFloat(),
                                      d.convertToFloat())
          : computeConstantFloatCount(s.convertToDouble(), l.convertToDouble(),
                                      d.convertToDouble());
  if (failed(count))
    return failure();
  return std::optional<int64_t>(*count);
}

/// Fail conversion when delta is a compile-time constant equal to zero (ORT
/// INVALID_ARGUMENT parity).
static LogicalResult
verifyConstantDeltaNonZero(Operation *op, Value deltaTensor, Type elemTy) {
  auto cst = deltaTensor.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return success();

  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense)
    return success();

  bool isZero = false;
  if (isa<IntegerType>(elemTy)) {
    for (APInt v : dense.getValues<APInt>()) {
      isZero = v.isZero();
      break;
    }
  } else if (isa<FloatType>(elemTy)) {
    for (APFloat v : dense.getValues<APFloat>()) {
      isZero = v.isZero();
      break;
    }
  } else {
    return success();
  }

  if (!isZero)
    return success();

  op->emitOpError("delta in Range operator can not be zero!");
  return failure();
}

/// Normalize a Range bound to a rank-0 scalar tensor. A rank-0 operand passes
/// through unchanged (preserving constant-foldability for
/// readbackScalarToHost); a single-element rank-1 operand is collapsed to
/// rank-0 via an empty-reassociation tensor.collapse_shape.
///
///   Before:  %v : tensor<1xi64>
///   After:   %v0 = tensor.collapse_shape %v [] : tensor<1xi64> into
///   tensor<i64>
static Value collapseRangeBoundToScalar(PatternRewriter &rewriter, Location loc,
                                        Value v) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getRank() == 0)
    return v;
  auto scalarTy = RankedTensorType::get({}, t.getElementType());
  return tensor::CollapseShapeOp::create(
      rewriter, loc, scalarTy, v, llvm::ArrayRef<mlir::ReassociationIndices>{});
}

struct RangeToHip : public RewritePattern {
  RangeToHip(MLIRContext *ctx) : RewritePattern("onnx.Range", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D ranked result");

    Type elemTy = resultType.getElementType();
    int64_t runtimeDataType = getRangeRuntimeDataType(elemTy);
    if (runtimeDataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    // ONNX Range bounds are scalars. Accept a rank-0 tensor, or a rank-1
    // tensor with a single static element (HF exports slice a length out of a
    // Shape result as tensor<1xi64>). The rank-1 form is collapsed to rank-0
    // below so the readback + hip.range path is identical either way.
    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected scalar operands matching result element type");
      if (t.getRank() == 0)
        continue;
      if (t.getRank() == 1 && t.hasStaticShape() && t.getDimSize(0) == 1)
        continue;
      return rewriter.notifyMatchFailure(
          op, "expected rank-0 or single-element rank-1 operands");
    }

    Location loc = op->getLoc();

    if (failed(verifyConstantDeltaNonZero(op, op->getOperand(2), elemTy)))
      return failure();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value ctx = *ctxOrFailure;

    // Collapse any single-element rank-1 bound to rank-0 so the readback +
    // hip.range path sees canonical scalars (see the operand check above).
    Value startS = collapseRangeBoundToScalar(rewriter, loc, op->getOperand(0));
    Value limitS = collapseRangeBoundToScalar(rewriter, loc, op->getOperand(1));
    Value deltaS = collapseRangeBoundToScalar(rewriter, loc, op->getOperand(2));

    Value init;
    FailureOr<std::optional<int64_t>> constantCount =
        tryComputeConstantRangeCount(op->getOperand(0), op->getOperand(1),
                                     op->getOperand(2), elemTy);
    if (failed(constantCount)) {
      op->emitOpError(
          "constant Range controls produce an invalid or unrepresentable "
          "result length");
      return failure();
    }
    if (*constantCount) {
      SmallVector<OpFoldResult> reifiedShape{
          rewriter.getIndexAttr(**constantCount)};
      auto constantInit = createEmptyTensorFromReifiedShape(
          rewriter, loc, resultType, reifiedShape);
      if (failed(constantInit))
        return rewriter.notifyMatchFailure(
            op, "Range result type contradicts constant trip count");
      init = *constantInit;
    }

    if (!init) {
      // One status-bearing grouped readback synchronizes all three controls.
      // The checked count op consumes `valid` before interpreting the slots,
      // initializes failure to length zero, and records the shared error state.
      SmallVector<Type> readbackTypes{
          rewriter.getI1Type(), rewriter.getI64Type(), rewriter.getI64Type(),
          rewriter.getI64Type()};
      auto readback = mlir::hip::ReadbackControlOp::create(
          rewriter, loc, readbackTypes, ctx,
          ValueRange{startS, limitS, deltaS});
      Value len = mlir::hip::CheckedRangeCountOp::create(
          rewriter, loc, rewriter.getIndexType(), ctx, readback.getValid(),
          readback.getValues()[0], readback.getValues()[1],
          readback.getValues()[2], rewriter.getI64IntegerAttr(runtimeDataType),
          rewriter.getI64IntegerAttr(resultType.getDimSize(0)));
      init = resultType.isDynamicDim(0)
                 ? Value(tensor::EmptyOp::create(rewriter, loc,
                                                 resultType.getShape(), elemTy,
                                                 ValueRange{len}))
                 : Value(tensor::EmptyOp::create(rewriter, loc,
                                                 resultType.getShape(), elemTy,
                                                 ValueRange{}));
    }

    auto rangeOp = mlir::hip::RangeOp::create(rewriter, loc, ctx, startS,
                                              limitS, deltaS, init);
    rewriter.replaceOp(op, rangeOp->getResult(0));
    return success();
  }
};

} // namespace

void populateRangeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<RangeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
