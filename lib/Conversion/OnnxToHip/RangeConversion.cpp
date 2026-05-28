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

    // Check operands: ONNX Range takes scalar inputs, which can be either
    // rank-0 tensors (tensor<i64>) or rank-1 tensors with shape [1]
    // (tensor<1xi64>). Both representations are valid.
    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected operands with matching result element type");
      int64_t rank = t.getRank();
      if (rank != 0 && (rank != 1 || t.getDimSize(0) != 1))
        return rewriter.notifyMatchFailure(
            op, "expected rank-0 or rank-1 size-1 scalar operands");
    }

    Location loc = op->getLoc();

    if (failed(verifyConstantDeltaNonZero(op, op->getOperand(2), elemTy)))
      return failure();

    // Extract scalar values for Range start/limit/delta operands.
    //
    // Naively emitting `tensor.extract %v[%c0]` on a rank-1 size-1 input
    // bufferizes to `memref.load %buf[0]`, and `hip-pool-allocs` only hoists
    // pure SSA shape arithmetic (memref.dim / arith.* / index_cast) — not
    // memref.load. If %v traces back to a `tensor.from_elements(%single)`
    // (typical for Unsqueeze(scalar, [0])-style ONNX shape arithmetic), the
    // resulting `memref.load` from a shared shape scratch buffer leaves the
    // pool-size computation undefined w.r.t. block dominance and produces
    // `operand #0 does not dominate this use` later in the pipeline.
    //
    // Same playbook as ExpandConversion: peek through value-preserving
    // rank-changing ops (tensor.expand_shape / tensor.collapse_shape) and
    // `tensor.from_elements` to reach the scalar producer; only fall back to
    // `tensor.extract` when traceback is impossible (opaque function args
    // etc.).
    auto peekScalarProducer = [](Value v) -> Value {
      while (true) {
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
          return v;
        if (auto exp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
          v = exp.getSrc();
          continue;
        }
        if (auto col = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
          v = col.getSrc();
          continue;
        }
        return v;
      }
    };

    auto extractScalar = [&](Value v) -> Value {
      // Case A (NEW, hybrid externalisation path): producer is an inline
      // `onnx.Constant` with a 1-element `value` attr that the hybrid
      // Phase 1 left in place. Materialize a pure-SSA `arith.constant` of
      // the stored scalar value. This is what keeps the Range count
      // formula (sub/ceildivsi/clamp) free of `memref.load` for the very
      // common case where start/limit/delta are compile-time literals
      // -- which is exactly what `hip-pool-allocs` needs in order to
      // hoist the dynamic-size `memref.alloc` for the Range output.
      // Peek both before and after rank-changing reshape views so we
      // catch `Reshape(<1xi64>) -> <i64>`-style chains too.
      if (auto attr = getInlineScalarFromOnnxConstant(v)) {
        Value scalar = materializeScalarFromDenseAttr(rewriter, loc, *attr);
        if (scalar && scalar.getType() == elemTy)
          return scalar;
      }

      // Try traceback first: walk through rank-changing reshape views to the
      // underlying scalar producer.
      Value peek = peekScalarProducer(v);

      if (auto attr = getInlineScalarFromOnnxConstant(peek)) {
        Value scalar = materializeScalarFromDenseAttr(rewriter, loc, *attr);
        if (scalar && scalar.getType() == elemTy)
          return scalar;
      }

      // Case B: producer is a single-element tensor.from_elements.
      if (auto fromElts = peek.getDefiningOp<tensor::FromElementsOp>()) {
        if (fromElts.getNumOperands() == 1) {
          Value elem = fromElts.getOperand(0);
          if (elem.getType() == elemTy)
            return elem;
        }
      }

      // Case C: producer is itself a rank-0 tensor — extract with empty
      // indices. Bufferizes to a memref.load on a 0-D buffer, which is small
      // enough that hip-materialize-host-scalars routes it through the
      // runtime-owned host scratch buffer (not the GPU pool), so pool-allocs
      // hoist still works.
      if (auto peekTy = dyn_cast<RankedTensorType>(peek.getType())) {
        if (peekTy.getRank() == 0)
          return tensor::ExtractOp::create(rewriter, loc, peek, ValueRange{});
      }

      // Fallback: rank-0 = empty indices, rank-1 size-1 = [0].
      auto t = cast<RankedTensorType>(v.getType());
      if (t.getRank() == 0)
        return tensor::ExtractOp::create(rewriter, loc, v, ValueRange{});
      Value idx = arith::ConstantIndexOp::create(rewriter, loc, 0);
      return tensor::ExtractOp::create(rewriter, loc, v, ValueRange{idx});
    };

    Value startE = extractScalar(op->getOperand(0));
    Value limitE = extractScalar(op->getOperand(1));
    Value deltaE = extractScalar(op->getOperand(2));

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
