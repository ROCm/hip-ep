/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "hip/Dialect/IR/HipShapeInterface.h"
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

    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 0 || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected 0-D operands matching result element type");
    }

    Location loc = op->getLoc();

    if (failed(verifyConstantDeltaNonZero(op, op->getOperand(2), elemTy)))
      return failure();

    Value startE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(0),
                                             ValueRange{});
    Value limitE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(1),
                                             ValueRange{});
    Value deltaE = tensor::ExtractOp::create(rewriter, loc, op->getOperand(2),
                                             ValueRange{});

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

    // ----------------------------------------------------------------------
    // Operand-provenance dispatch for dynamic output dim 0:
    // ----------------------------------------------------------------------
    // Range with i64 element type and dynamic output dim 0: when all three
    // operands (start, limit, delta) trace to host-readable func.func
    // entry-block args (i.e. EP inputs) we can encode dim 0 as a Category
    // B DimSpec:
    //
    //     CeilDiv(Sub(limit, start), delta)
    //
    // using operand-relative InputValueI64 leaves; ComposeDimSpecs walks
    // operand → func-arg to rewrite them into EP-relative leaves. The EP
    // resolves the dim BEFORE inference_compute, allocates the ORT
    // OrtValue with the resolved shape, and the existing wrap_range path
    // is used unchanged (no slot needed).
    //
    // When any operand is an intermediate value the dim falls back to a
    // RuntimeSlot (Category C). The wrap_range path in
    // lib/Runtime/real/range.cpp will inspect the slot_id attribute to
    // decide between the static-output legacy launch and the slot-
    // publishing variant.
    mlir::ArrayAttr outputDimSpecsAttr;
    mlir::IntegerAttr slotIdAttr;
    if (resultType.isDynamicDim(0)) {
      // Category B is only legal when:
      //   1. all three operands resolve to func-arg entry-block values
      //      (host-readable in the EP marshal), AND
      //   2. the element type is i64 -- the EP resolver reads
      //      InputValueI64 leaves as int64_t (see
      //      backend-mlir-compiler/.../DimSpecResolver.cpp::readInputI64).
      //      Other element types (i32, fp16/fp32) would silently
      //      reinterpret bytes.
      bool allFuncArgs = operandIsFuncEntryBlockArg(op->getOperand(0)) &&
                         operandIsFuncEntryBlockArg(op->getOperand(1)) &&
                         operandIsFuncEntryBlockArg(op->getOperand(2));
      if (allFuncArgs && elemTy.isInteger(64)) {
        // Category B: build operand-relative DimSpec.
        // Operand indices on hip.range: 0=ctx, 1=start, 2=limit, 3=delta,
        // 4=output (DPS). The per-op-attached spec uses the OP's operand
        // indices, which matches how ComposeDimSpecs walks
        // op->getOperand(idx). The leaves get rewritten to EP-relative
        // InputValueI64 by ComposeDimSpecs.
        DimSpec startSpec =
            DimSpec::makeInputValueI64(/*input_index=*/1, /*flat_offset=*/0);
        DimSpec limitSpec =
            DimSpec::makeInputValueI64(/*input_index=*/2, 0);
        DimSpec deltaSpec =
            DimSpec::makeInputValueI64(/*input_index=*/3, 0);
        DimSpec diff =
            DimSpec::makeBinary(DimSpecKind::Sub, limitSpec, startSpec);
        DimSpec dim0 =
            DimSpec::makeBinary(DimSpecKind::CeilDiv, diff, deltaSpec);
        auto *ctxRaw = rewriter.getContext();
        outputDimSpecsAttr = rewriter.getArrayAttr(
            {rewriter.getArrayAttr({dim0.serializeAsArrayAttr(ctxRaw)})});
      } else {
        // Category C: either at least one operand is an intermediate
        // value, or the element type is not i64 (the EP resolver only
        // knows how to read i64). Allocate a unique module-level
        // slot_id, attach a RuntimeSlot leaf, and let wrap_range publish
        // into that slot at runtime. Matches the bookkeeping pattern
        // used by NonZeroConversion.
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        int32_t slot_id_v = 0;
        if (auto a = moduleOp->getAttrOfType<mlir::IntegerAttr>(
                "hipdnn.next_dyn_slot_id"))
          slot_id_v = static_cast<int32_t>(a.getInt());
        moduleOp->setAttr("hipdnn.next_dyn_slot_id",
                          rewriter.getI32IntegerAttr(slot_id_v + 1));
        slotIdAttr = rewriter.getI32IntegerAttr(slot_id_v);
        DimSpec dim0 = DimSpec::makeRuntimeSlot(slot_id_v);
        auto *ctxRaw = rewriter.getContext();
        outputDimSpecsAttr = rewriter.getArrayAttr(
            {rewriter.getArrayAttr({dim0.serializeAsArrayAttr(ctxRaw)})});
      }
    }

    auto rangeOp = mlir::hip::RangeOp::create(
        rewriter, loc, resultType, ctx, op->getOperand(0), op->getOperand(1),
        op->getOperand(2), init,
        /*output_dim_specs=*/outputDimSpecsAttr,
        /*slot_id=*/slotIdAttr);
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
