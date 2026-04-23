/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace hip {
namespace {

/// Dynamic length (index) for integer numpy.arange(start, limit, delta).
static Value buildIntRangeCount(PatternRewriter &rewriter, Location loc,
                                Value start, Value limit, Value delta,
                                IntegerType elemTy) {
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, elemTy);
  Value cmpZ = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     delta, zero);
  Value cmpPos = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       delta, zero);
  Value cmpLimLe = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sle, limit, start);
  Value cmpLimGe = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, limit, start);
  Value emptyPos = arith::AndIOp::create(rewriter, loc, cmpPos, cmpLimLe);
  Value emptyNeg = arith::AndIOp::create(
      rewriter, loc,
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, delta,
                            zero),
      cmpLimGe);
  Value empty = arith::OrIOp::create(rewriter, loc, emptyPos, emptyNeg);
  Value bad = arith::OrIOp::create(rewriter, loc, cmpZ, empty);

  Value diffPos = arith::SubIOp::create(rewriter, loc, limit, start);
  Value diffNeg = arith::SubIOp::create(rewriter, loc, start, limit);
  Value negDelta = arith::SubIOp::create(rewriter, loc, zero, delta);
  Value nPos = arith::CeilDivSIOp::create(rewriter, loc, diffPos, delta);
  Value nNeg = arith::CeilDivSIOp::create(rewriter, loc, diffNeg, negDelta);
  Value nInt = arith::SelectOp::create(rewriter, loc, cmpPos, nPos, nNeg);
  Value nIntSel = arith::SelectOp::create(rewriter, loc, bad, zero, nInt);
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
      rewriter, loc, APFloat::getZero(elemTy.getFloatSemantics()), elemTy);
  Value needBump = arith::CmpFOp::create(rewriter, loc,
                                         arith::CmpFPredicate::OGT, frac, c0f);
  Value oneI = arith::ConstantIntOp::create(rewriter, loc, 1, i64);
  Value zeroI = arith::ConstantIntOp::create(rewriter, loc, 0, i64);
  Value bumpI = arith::SelectOp::create(rewriter, loc, needBump, oneI, zeroI);
  return arith::AddIOp::create(rewriter, loc, floorI, bumpI);
}

/// Dynamic length for float ranges using ceil((limit-start)/delta) or
/// ceil((start-limit)/(-delta)).
static Value buildFloatRangeCount(PatternRewriter &rewriter, Location loc,
                                  Value start, Value limit, Value delta,
                                  FloatType elemTy) {
  Value zero = arith::ConstantFloatOp::create(
      rewriter, loc, APFloat::getZero(elemTy.getFloatSemantics()), elemTy);
  Value cmpZ = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ,
                                     delta, zero);
  Value cmpPos = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGT,
                                       delta, zero);
  Value cmpLimLe = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLE, limit, start);
  Value cmpLimGe = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OGE, limit, start);
  Value emptyPos = arith::AndIOp::create(rewriter, loc, cmpPos, cmpLimLe);
  Value cmpNegDelta = arith::CmpFOp::create(
      rewriter, loc, arith::CmpFPredicate::OLT, delta, zero);
  Value emptyNeg = arith::AndIOp::create(rewriter, loc, cmpNegDelta, cmpLimGe);
  Value empty = arith::OrIOp::create(rewriter, loc, emptyPos, emptyNeg);
  Value bad = arith::OrIOp::create(rewriter, loc, cmpZ, empty);

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
  Value zeroI = arith::ConstantIntOp::create(rewriter, loc, 0, i64);
  Value nIntSel = arith::SelectOp::create(rewriter, loc, bad, zeroI, nInt);
  Value nNonNeg = arith::MaxSIOp::create(rewriter, loc, nIntSel, zeroI);
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                    nNonNeg);
}

struct RangeToMlir : public RewritePattern {
  RangeToMlir(MLIRContext *ctx) : RewritePattern("onnx.Range", 1, ctx) {}

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

    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 0 || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected 0-D operands matching result element type");
    }

    Location loc = op->getLoc();
    Value startE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(0),
                                             ValueRange{});
    Value limitE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(1),
                                             ValueRange{});
    Value deltaE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(2),
                                             ValueRange{});

    Value len = TypeSwitch<Type, Value>(elemTy)
                    .Case<IntegerType>([&](IntegerType ity) {
                      return buildIntRangeCount(rewriter, loc, startE, limitE,
                                                deltaE, ity);
                    })
                    .Case<FloatType>([&](FloatType fty) {
                      return buildFloatRangeCount(rewriter, loc, startE, limitE,
                                                  deltaE, fty);
                    })
                    .Default([&](Type) { return Value(); });
    if (!len)
      return failure();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value ctx = *ctxOrFailure;

    Value init;
    if (resultType.isDynamicDim(0)) {
      init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                     elemTy, ValueRange{len});
    } else {
      init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                     elemTy, ValueRange{});
    }

    auto rangeOp = mlir::hip::RangeOp::create(
        rewriter, loc, resultType, ctx, op->getOperand(0), op->getOperand(1),
        op->getOperand(2), init);
    rewriter.replaceOp(op, rangeOp.getResult());
    return success();
  }
};

} // namespace

void mlir::hip::populateRangeConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<RangeToMlir>(ctx);
}

} // namespace hip
} // namespace mlir
