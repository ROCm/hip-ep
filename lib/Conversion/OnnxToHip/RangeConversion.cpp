/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

/// ONNX Range matches numpy.arange(start, limit, delta) (exclusive limit).
template <typename T>
static void numpyArangeInt(T start, T limit, T delta,
                           llvm::SmallVectorImpl<T> &out) {
  if (delta == 0)
    return;
  constexpr int64_t kMaxElems = 1 << 20;
  if (delta > 0) {
    for (T x = start; x < limit; x += delta) {
      out.push_back(x);
      if (static_cast<int64_t>(out.size()) >= kMaxElems)
        break;
    }
  } else {
    for (T x = start; x > limit; x += delta) {
      out.push_back(x);
      if (static_cast<int64_t>(out.size()) >= kMaxElems)
        break;
    }
  }
}

static void numpyArangeDouble(double start, double limit, double delta,
                              llvm::SmallVector<double> &out) {
  if (delta == 0.0)
    return;
  constexpr int64_t kMaxElems = 1 << 20;
  if (delta > 0) {
    for (double x = start; x < limit; x += delta) {
      out.push_back(x);
      if (static_cast<int64_t>(out.size()) >= kMaxElems)
        break;
    }
  } else {
    for (double x = start; x > limit; x += delta) {
      out.push_back(x);
      if (static_cast<int64_t>(out.size()) >= kMaxElems)
        break;
    }
  }
}

static FailureOr<ElementsAttr> tryGetScalarElementsAttr(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();
  if (auto c = dyn_cast<arith::ConstantOp>(def))
    return c.getValue();
  return failure();
}

static FailureOr<APInt> getScalarIntAttr(ElementsAttr attr) {
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.getElementType().isIntOrIndex())
    return failure();
  if (dense.isSplat())
    return dense.getSplatValue<APInt>();
  if (dense.getNumElements() != 1)
    return failure();
  return *dense.value_begin<APInt>();
}

static FailureOr<APFloat> getScalarFloatAttr(ElementsAttr attr) {
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.getElementType().isFloatingPoint())
    return failure();
  if (dense.isSplat())
    return dense.getSplatValue<APFloat>();
  if (dense.getNumElements() != 1)
    return failure();
  return *dense.value_begin<APFloat>();
}

static LogicalResult tryFoldRangeConstants(Operation *op,
                                           PatternRewriter &rewriter,
                                           ElementsAttr startAttr,
                                           ElementsAttr limitAttr,
                                           ElementsAttr deltaAttr,
                                           RankedTensorType resultType) {
  Type elemTy = resultType.getElementType();
  Location loc = op->getLoc();

  if (elemTy.isInteger(16) || elemTy.isInteger(32) ||
      elemTy.isInteger(64)) {
    FailureOr<APInt> s = getScalarIntAttr(startAttr);
    FailureOr<APInt> l = getScalarIntAttr(limitAttr);
    FailureOr<APInt> d = getScalarIntAttr(deltaAttr);
    if (failed(s) || failed(l) || failed(d))
      return failure();
    if ((*d).isZero())
      return rewriter.notifyMatchFailure(op, "Range delta must be non-zero");

    int64_t start = (*s).getSExtValue();
    int64_t limit = (*l).getSExtValue();
    int64_t delta = (*d).getSExtValue();
    llvm::SmallVector<int64_t> seq;
    numpyArangeInt(start, limit, delta, seq);
    llvm::SmallVector<Attribute> attrs;
    attrs.reserve(seq.size());
    auto intTy = cast<IntegerType>(elemTy);
    for (int64_t v : seq)
      attrs.push_back(rewriter.getIntegerAttr(intTy, v));
    auto shaped = RankedTensorType::get(
        {static_cast<int64_t>(attrs.size())}, elemTy);
    auto dense = DenseElementsAttr::get(shaped, attrs);
    Value folded = rewriter.create<arith::ConstantOp>(loc, dense);
    if (folded.getType() != resultType)
      folded = rewriter.create<tensor::CastOp>(loc, resultType, folded);
    rewriter.replaceOp(op, folded);
    return success();
  }

  if (elemTy.isF32() || elemTy.isF64()) {
    FailureOr<APFloat> s = getScalarFloatAttr(startAttr);
    FailureOr<APFloat> l = getScalarFloatAttr(limitAttr);
    FailureOr<APFloat> d = getScalarFloatAttr(deltaAttr);
    if (failed(s) || failed(l) || failed(d))
      return failure();
    if ((*d).isZero())
      return rewriter.notifyMatchFailure(op, "Range delta must be non-zero");

    bool losesInfo;
    double start = (*s).convertToDouble();
    double limit = (*l).convertToDouble();
    double delta = (*d).convertToDouble();
    llvm::SmallVector<double> seq;
    numpyArangeDouble(start, limit, delta, seq);
    llvm::SmallVector<Attribute> attrs;
    attrs.reserve(seq.size());
    const llvm::fltSemantics &sem = cast<FloatType>(elemTy).getFloatSemantics();
    for (double v : seq) {
      APFloat fv(v);
      fv.convert(sem, APFloat::rmNearestTiesToEven, &losesInfo);
      attrs.push_back(rewriter.getFloatAttr(elemTy, fv));
    }
    auto shaped = RankedTensorType::get(
        {static_cast<int64_t>(attrs.size())}, elemTy);
    auto dense = DenseElementsAttr::get(shaped, attrs);
    Value folded = rewriter.create<arith::ConstantOp>(loc, dense);
    if (folded.getType() != resultType)
      folded = rewriter.create<tensor::CastOp>(loc, resultType, folded);
    rewriter.replaceOp(op, folded);
    return success();
  }

  return failure();
}

/// Dynamic length (index) for integer numpy.arange(start, limit, delta).
static Value buildIntRangeCount(PatternRewriter &rewriter, Location loc,
                                Value start, Value limit, Value delta,
                                IntegerType elemTy) {
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, elemTy);
  Value cmpZ = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     delta, zero);
  Value cmpPos = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                                       delta, zero);
  Value cmpLimLe =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, limit,
                            start);
  Value cmpLimGe =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, limit,
                            start);
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
  Value cmpLimLe =
      arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLE, limit,
                            start);
  Value cmpLimGe =
      arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE, limit,
                            start);
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
  Value ceilPos = rewriter.create<math::CeilOp>(loc, quotPos);
  Value ceilNeg = rewriter.create<math::CeilOp>(loc, quotNeg);
  Value nFloat = arith::SelectOp::create(rewriter, loc, cmpPos, ceilPos,
                                         ceilNeg);
  Value nFloatSel = arith::SelectOp::create(rewriter, loc, bad, zero, nFloat);
  IntegerType i64 = rewriter.getI64Type();
  Value nI64 = arith::FPToSIOp::create(rewriter, loc, i64, nFloatSel);
  Value nNonNeg = arith::MaxSIOp::create(
      rewriter, loc, nI64,
      arith::ConstantIntOp::create(rewriter, loc, 0, i64));
  return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                    nNonNeg);
}

static Value buildLoopFill(PatternRewriter &rewriter, Location loc,
                           RankedTensorType resultType, Value len,
                           Value startV, Value deltaV, Type elemTy) {
  Value empty;
  if (resultType.isDynamicDim(0)) {
    empty = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    elemTy, ValueRange{len});
  } else {
    empty = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    elemTy, ValueRange{});
  }

  Value ub = resultType.isDynamicDim(0)
                 ? len
                 : arith::ConstantIndexOp::create(
                       rewriter, loc, resultType.getDimSize(0));
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto loop =
      rewriter.create<scf::ForOp>(loc, c0, ub, c1, ValueRange{empty});
  Block *body = loop.getBody();
  Value iv = body->getArgument(0);
  Value iter = body->getArgument(1);
  rewriter.setInsertionPointToStart(body);

  Value val = TypeSwitch<Type, Value>(elemTy)
                  .Case<IntegerType>([&](IntegerType ity) {
                    Value ivInt =
                        arith::IndexCastOp::create(rewriter, loc, ity, iv);
                    Value prod =
                        arith::MulIOp::create(rewriter, loc, ivInt, deltaV);
                    return arith::AddIOp::create(rewriter, loc, startV, prod);
                  })
                  .Case<FloatType>([&](FloatType fty) {
                    Value ivInt =
                        arith::IndexCastOp::create(rewriter, loc,
                                                   rewriter.getI64Type(), iv);
                    Value ivFloat =
                        arith::SIToFPOp::create(rewriter, loc, fty, ivInt);
                    Value prod =
                        arith::MulFOp::create(rewriter, loc, ivFloat, deltaV);
                    return arith::AddFOp::create(rewriter, loc, startV, prod);
                  })
                  .Default([&](Type) { return Value(); });
  if (!val)
    return Value();

  Value inserted = rewriter.create<tensor::InsertOp>(loc, iter.getType(), val,
                                                      iter, ValueRange{iv});
  rewriter.create<scf::YieldOp>(loc, ValueRange{inserted});
  rewriter.setInsertionPointAfter(loop);
  return loop.getResult(0);
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

    FailureOr<ElementsAttr> startAttr =
        tryGetScalarElementsAttr(op->getOperand(0));
    FailureOr<ElementsAttr> limitAttr =
        tryGetScalarElementsAttr(op->getOperand(1));
    FailureOr<ElementsAttr> deltaAttr =
        tryGetScalarElementsAttr(op->getOperand(2));
    if (succeeded(startAttr) && succeeded(limitAttr) && succeeded(deltaAttr))
      if (succeeded(tryFoldRangeConstants(op, rewriter, *startAttr, *limitAttr,
                                            *deltaAttr, resultType)))
        return success();

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
                      return buildFloatRangeCount(rewriter, loc, startE,
                                                  limitE, deltaE, fty);
                    })
                    .Default([&](Type) { return Value(); });
    if (!len)
      return failure();

    Value filled =
        buildLoopFill(rewriter, loc, resultType, len, startE, deltaE, elemTy);
    if (!filled)
      return failure();

    rewriter.replaceOp(op, filled);
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
