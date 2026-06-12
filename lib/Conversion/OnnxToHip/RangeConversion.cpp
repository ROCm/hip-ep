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

/// Read a rank-0 scalar operand to a host SSA value for the trip-count
/// arithmetic.
///
/// onnx.Range's start/limit/delta are 0-D tensors. When an operand is a
/// compile-time constant (the common case for start=0 / delta=1) we fold it to
/// an `arith.constant` — a pure host value, no device traffic. When it is a
/// runtime value computed on the GPU (the canonical case: limit derived from
/// `image_grid_thw` via Gather/Cast/...), we MUST NOT `tensor.extract` it: that
/// lowers to a bare host `memref.load` of a device buffer with no stream
/// synchronization, which reads stale memory on targets where the pool is true
/// device memory (it accidentally works where the pool is UMA-mapped
/// host-accessible memory) and yields a zero trip count → collapsed output dim.
/// Instead we emit `hip.readback_scalar` (D2H + stream sync) so the host sees
/// the value the producing kernel actually wrote.
///
/// Before (runtime limit, incorrect):
///   %l = tensor.extract %limit[] : tensor<i64>   // host load of device mem
/// After:
///   %l = hip.readback_scalar(%ctx, %limit : tensor<i64>) -> i64
static Value readScalarOperand(PatternRewriter &rewriter, Location loc,
                               Value ctx, Value operand, Type elemTy) {
  // Compile-time constant: fold to a host constant, no readback needed. Cover
  // both `arith.constant` (already lowered) and the inline `onnx.Constant`
  // `value` attribute (before constant externalization). start=0 / delta=1 are
  // the common constant operands; folding them avoids a needless D2H sync.
  DenseElementsAttr dense;
  if (auto cst = operand.getDefiningOp<arith::ConstantOp>())
    dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  else if (Operation *def = operand.getDefiningOp())
    if (def->getName().getStringRef() == "onnx.Constant")
      dense = def->getAttrOfType<DenseElementsAttr>("value");
  if (dense && dense.getNumElements() == 1) {
    if (auto ity = dyn_cast<IntegerType>(elemTy)) {
      for (APInt v : dense.getValues<APInt>())
        return arith::ConstantIntOp::create(rewriter, loc, ity,
                                            v.getSExtValue());
    } else if (auto fty = dyn_cast<FloatType>(elemTy)) {
      for (APFloat v : dense.getValues<APFloat>())
        return arith::ConstantFloatOp::create(rewriter, loc, fty, v);
    }
  }
  // Runtime value (possibly GPU-computed): synchronized host readback.
  return ReadbackScalarOp::create(rewriter, loc, elemTy, ctx, operand)
      .getResult();
}

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

    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 0 || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected 0-D operands matching result element type");
    }

    Location loc = op->getLoc();

    if (failed(verifyConstantDeltaNonZero(op, op->getOperand(2), elemTy)))
      return failure();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value ctx = *ctxOrFailure;

    // Read start/limit/delta to host SSA values for the trip-count arithmetic.
    // Constants fold; GPU-computed operands go through a synchronized
    // hip.readback_scalar (see readScalarOperand for why a plain tensor.extract
    // is a correctness bug on true-device-memory targets).
    Value startE =
        readScalarOperand(rewriter, loc, ctx, op->getOperand(0), elemTy);
    Value limitE =
        readScalarOperand(rewriter, loc, ctx, op->getOperand(1), elemTy);
    Value deltaE =
        readScalarOperand(rewriter, loc, ctx, op->getOperand(2), elemTy);

    Value len = llvm::TypeSwitch<Type, Value>(elemTy)
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

    Value init;
    if (resultType.isDynamicDim(0)) {
      init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                     elemTy, ValueRange{len});
    } else {
      init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                     elemTy, ValueRange{});
    }

    auto rangeOp =
        mlir::hip::RangeOp::create(rewriter, loc, ctx, op->getOperand(0),
                                   op->getOperand(1), op->getOperand(2), init);
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
