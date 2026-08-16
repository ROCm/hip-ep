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
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace hip {
namespace {

/// Empty-result condition for integer Range:
/// - delta > 0 and limit <= start
/// - delta < 0 and limit >= start
static Value buildIntRangeEmptyCheck(PatternRewriter &rewriter, Location loc,
                                     Value start, Value limit, Value delta,
                                     Value zero) {
  Value cmpPos = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       delta, zero);
  Value cmpNeg = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                       delta, zero);
  Value cmpLimLe = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sle, limit, start);
  Value cmpLimGe = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, limit, start);
  Value emptyPos = arith::AndIOp::create(rewriter, loc, cmpPos, cmpLimLe);
  Value emptyNeg = arith::AndIOp::create(rewriter, loc, cmpNeg, cmpLimGe);
  return arith::OrIOp::create(rewriter, loc, emptyPos, emptyNeg);
}

/// Empty-result condition for float Range:
/// - delta > 0 and limit <= start
/// - delta < 0 and limit >= start
static Value buildFloatRangeEmptyCheck(PatternRewriter &rewriter, Location loc,
                                       Value start, Value limit, Value delta,
                                       Value zero) {
  Value cmpPos = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                       delta, zero);
  Value cmpNeg = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                       delta, zero);
  Value cmpLimLe = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLE, limit, start);
  Value cmpLimGe = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OGE, limit, start);
  Value emptyPos = arith::AndIOp::create(rewriter, loc, cmpPos, cmpLimLe);
  Value emptyNeg = arith::AndIOp::create(rewriter, loc, cmpNeg, cmpLimGe);
  return arith::OrIOp::create(rewriter, loc, emptyPos, emptyNeg);
}

/// Dynamic length (index) for integer numpy.arange(start, limit, delta).
static Value buildIntRangeCount(PatternRewriter &rewriter, Location loc,
                                Value start, Value limit, Value delta,
                                IntegerType elemTy) {
  Value zero = arith::ConstantIntOp::create(rewriter, loc, elemTy, 0);
  Value cmpZ = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     delta, zero);
  Value empty =
      buildIntRangeEmptyCheck(rewriter, loc, start, limit, delta, zero);
  Value cmpPos = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       delta, zero);
  // Guard compile-time length expression from divide-by-zero and empty ranges.
  // Non-empty `hip_range` launches validate delta in `range_kernel`. Constant
  // delta==0 is rejected in `verifyConstantDeltaNonZero`.
  Value clampToZeroLen = arith::OrIOp::create(rewriter, loc, cmpZ, empty);

  Value diffPos = arith::SubIOp::create(rewriter, loc, limit, start);
  Value diffNeg = arith::SubIOp::create(rewriter, loc, start, limit);
  Value negDelta = arith::SubIOp::create(rewriter, loc, zero, delta);
  Value nPos = arith::CeilDivSIOp::create(rewriter, loc, diffPos, delta);
  Value nNeg = arith::CeilDivSIOp::create(rewriter, loc, diffNeg, negDelta);
  Value nInt = arith::SelectOp::create(rewriter, loc, cmpPos, nPos, nNeg);
  Value nIntSel =
      arith::SelectOp::create(rewriter, loc, clampToZeroLen, zero, nInt);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                    nIntSel);
}

/// Ceil(\p q) for \p q >= 0 using arith only (avoids math.ceil / MathDialect).
static Value buildArithCeilNonNegFloat(PatternRewriter &rewriter, Location loc,
                                       Value q, FloatType elemTy) {
  IntegerType i64 = rewriter.getI64Type();
  Value floorI = arith::FPToSIOp::create(rewriter, loc, i64, q);
  Value floorF = arith::SIToFPOp::create(rewriter, loc, elemTy, floorI);
  Value frac = arith::SubFOp::create(rewriter, loc, q, floorF);
  Value c0f = arith::ConstantFloatOp::create(
      rewriter, loc, elemTy, APFloat::getZero(elemTy.getFloatSemantics()));
  Value needBump = arith::CmpFOp::create(rewriter, loc,
                                         arith::CmpFPredicate::OGT, frac, c0f);
  Value oneI = arith::ConstantIntOp::create(rewriter, loc, i64, 1);
  Value zeroI = arith::ConstantIntOp::create(rewriter, loc, i64, 0);
  Value bumpI = arith::SelectOp::create(rewriter, loc, needBump, oneI, zeroI);
  return arith::AddIOp::create(rewriter, loc, floorI, bumpI);
}

/// Dynamic length for float ranges using ceil((limit-start)/delta) or
/// ceil((start-limit)/(-delta)).
static Value buildFloatRangeCount(PatternRewriter &rewriter, Location loc,
                                  Value start, Value limit, Value delta,
                                  FloatType elemTy) {
  Value zero = arith::ConstantFloatOp::create(
      rewriter, loc, elemTy, APFloat::getZero(elemTy.getFloatSemantics()));
  Value cmpZ = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ,
                                     delta, zero);
  Value cmpPos = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                       delta, zero);
  Value empty =
      buildFloatRangeEmptyCheck(rewriter, loc, start, limit, delta, zero);
  // Guard compile-time length expression from divide-by-zero and empty ranges.
  // Non-empty `hip_range` launches validate delta in `range_kernel`. Constant
  // delta==0 is rejected in `verifyConstantDeltaNonZero`.
  Value clampToZeroLen = arith::OrIOp::create(rewriter, loc, cmpZ, empty);

  Value diffPos = arith::SubFOp::create(rewriter, loc, limit, start);
  Value diffNeg = arith::SubFOp::create(rewriter, loc, start, limit);
  Value negDelta = arith::SubFOp::create(rewriter, loc, zero, delta);
  Value quotPos = arith::DivFOp::create(rewriter, loc, diffPos, delta);
  Value quotNeg = arith::DivFOp::create(rewriter, loc, diffNeg, negDelta);
  IntegerType i64 = rewriter.getI64Type();
  Value ceilPosI = buildArithCeilNonNegFloat(rewriter, loc, quotPos, elemTy);
  Value ceilNegI = buildArithCeilNonNegFloat(rewriter, loc, quotNeg, elemTy);
  Value nInt =
      arith::SelectOp::create(rewriter, loc, cmpPos, ceilPosI, ceilNegI);
  Value zeroI = arith::ConstantIntOp::create(rewriter, loc, i64, 0);
  Value nIntSel =
      arith::SelectOp::create(rewriter, loc, clampToZeroLen, zeroI, nInt);
  Value nNonNeg = arith::MaxSIOp::create(rewriter, loc, nIntSel, zeroI);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                    nNonNeg);
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
    if (!elemTy.isIntOrFloat())
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
    SmallVector<int64_t> startValues, limitValues, deltaValues, staticValues;
    if (isa<IntegerType>(elemTy) &&
        extractConstantIntTensor(op->getOperand(0), startValues) &&
        extractConstantIntTensor(op->getOperand(1), limitValues) &&
        extractConstantIntTensor(op->getOperand(2), deltaValues) &&
        startValues.size() == 1 && limitValues.size() == 1 &&
        deltaValues.size() == 1) {
      staticValues = {startValues[0], limitValues[0], deltaValues[0]};
      SmallVector<OpFoldResult> reifiedShape;
      if (succeeded(mlir::hip::reifyRangeShape(
              rewriter, loc, startS, limitS, deltaS, reifiedShape,
              ArrayRef<int64_t>(staticValues)))) {
        auto constantInit = createEmptyTensorFromReifiedShape(
            rewriter, loc, resultType, reifiedShape);
        if (failed(constantInit))
          return rewriter.notifyMatchFailure(
              op, "Range result type contradicts constant trip count");
        init = *constantInit;
      }
    }

    if (!init) {
      // Preserve the payload-dynamic path exactly. GPU-computed bounds use
      // synchronized scalar readback before sizing the destination.
      Value startE = readbackScalarToHost(rewriter, loc, ctx, startS);
      Value limitE = readbackScalarToHost(rewriter, loc, ctx, limitS);
      Value deltaE = readbackScalarToHost(rewriter, loc, ctx, deltaS);
      Value len = llvm::TypeSwitch<Type, Value>(elemTy)
                      .Case<IntegerType>([&](IntegerType ity) {
                        return buildIntRangeCount(rewriter, loc, startE, limitE,
                                                  deltaE, ity);
                      })
                      .Case<FloatType>([&](FloatType fty) {
                        return buildFloatRangeCount(rewriter, loc, startE,
                                                    limitE, deltaE, fty);
                      })
                      .Default([&](Type) { return Value(); });
      if (!len)
        return failure();
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
