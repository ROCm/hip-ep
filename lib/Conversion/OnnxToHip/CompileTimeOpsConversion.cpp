/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cmath>
#include <limits>

#include "llvm/ADT/APFloat.h"

namespace mlir {
namespace hip {
namespace {

static mlir::FailureOr<int64_t> readIntScalarAsI64(mlir::Value v) {
  auto fromDense =
      [](mlir::DenseElementsAttr attr) -> mlir::FailureOr<int64_t> {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
    if (!type || type.getNumElements() != 1)
      return mlir::failure();
    mlir::Type et = type.getElementType();
    if (et.isInteger(64))
      return static_cast<int64_t>(*attr.getValues<int64_t>().begin());
    if (et.isInteger(32))
      return static_cast<int64_t>(*attr.getValues<int32_t>().begin());
    if (et.isInteger(16))
      return static_cast<int64_t>(*attr.getValues<int16_t>().begin());
    return mlir::failure();
  };

  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
    if (auto denseAttr =
            mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue()))
      return fromDense(denseAttr);
    return mlir::failure();
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return mlir::failure();
  if (!def->hasAttr("value"))
    return mlir::failure();
  auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      def->getAttrOfType<mlir::ElementsAttr>("value"));
  if (!valueAttr)
    return mlir::failure();
  return fromDense(valueAttr);
}

static mlir::FailureOr<llvm::APFloat> readFloatScalar(mlir::Value v,
                                                      mlir::FloatType ft) {
  auto fromDense =
      [&](mlir::DenseElementsAttr attr) -> mlir::FailureOr<llvm::APFloat> {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
    if (!type || type.getNumElements() != 1)
      return mlir::failure();
    mlir::Type et = type.getElementType();
    if (et != ft)
      return mlir::failure();
    if (ft.isF32())
      return llvm::APFloat(*attr.getValues<float>().begin());
    if (ft.isF64())
      return llvm::APFloat(*attr.getValues<double>().begin());
    return mlir::failure();
  };

  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto fAttr = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
      if (fAttr.getType() == ft)
        return fAttr.getValue();
      return mlir::failure();
    }
    if (auto denseAttr =
            mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue()))
      return fromDense(denseAttr);
    return mlir::failure();
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def || !def->hasAttr("value"))
    return mlir::failure();
  auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      def->getAttrOfType<mlir::ElementsAttr>("value"));
  if (!valueAttr)
    return mlir::failure();
  return fromDense(valueAttr);
}

static bool isOnnxRangeElementType(mlir::Type t) {
  return t.isInteger(16) || t.isInteger(32) || t.isInteger(64) || t.isF32() ||
         t.isF64();
}

/// Host-side ONNX Range element count: max(ceil((limit - start) / delta), 0).
static size_t onnxRangeNumElements(double start, double limit, double delta) {
  if (delta == 0.0 || std::isnan(delta) || std::isnan(start) ||
      std::isnan(limit))
    return 0;
  double ratio = (limit - start) / delta;
  if (std::isnan(ratio) || std::isinf(ratio))
    return 0;
  double c = std::ceil(ratio);
  if (c < 0.0 || std::isnan(c) || std::isinf(c))
    return 0;
  if (c >= static_cast<double>(std::numeric_limits<size_t>::max()))
    return 0;
  return static_cast<size_t>(c);
}

/// onnx.Shape -> arith.constant (compile-time fold).
///
/// Supports static ranked input tensors only. If any sliced dimension is
/// dynamic, this pattern intentionally does not rewrite.
struct ShapeToConstant : public mlir::RewritePattern {
  ShapeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 operand and 1 result");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op,
                                         "expected i64 output element type");

    int64_t rank = inputType.getRank();
    int64_t start = 0;
    if (auto startAttr = op->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getInt();

    // ONNX Shape supports negative indexing from the end.
    if (start < 0)
      start += rank;
    if (start < 0 || start > rank)
      return rewriter.notifyMatchFailure(op, "start is out of valid range");

    int64_t end = rank;
    if (auto endAttr = op->getAttrOfType<mlir::IntegerAttr>("end")) {
      end = endAttr.getInt();
      if (end < 0)
        end += rank;
    }
    if (end < 0 || end > rank)
      return rewriter.notifyMatchFailure(op, "end is out of valid range");
    if (start > end)
      return rewriter.notifyMatchFailure(op, "expected start <= end");

    llvm::SmallVector<int64_t> dims;
    dims.reserve(static_cast<size_t>(end - start));
    for (int64_t i = start; i < end; ++i) {
      if (inputType.isDynamicDim(i))
        return rewriter.notifyMatchFailure(
            op, "dynamic input dimension cannot be folded for onnx.Shape");
      dims.push_back(inputType.getDimSize(i));
    }

    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D output tensor");
    if (!resultType.isDynamicDim(0) &&
        resultType.getDimSize(0) != static_cast<int64_t>(dims.size()))
      return rewriter.notifyMatchFailure(op, "output length mismatch");

    auto signlessI64 = mlir::IntegerType::get(
        rewriter.getContext(), 64,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto constType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(dims.size())}, signlessI64);
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(dims.size());
    for (int64_t d : dims)
      elements.push_back(
          mlir::arith::ConstantIntOp::create(rewriter, op->getLoc(), d, 64));
    auto folded = mlir::tensor::FromElementsOp::create(rewriter, op->getLoc(),
                                                       constType, elements);
    rewriter.replaceOp(op, folded.getResult());
    return mlir::success();
  }
};

/// onnx.Size -> tensor.from_elements(arith.constant i64)
///
/// Compile-time fold only when all dimensions are static.
struct SizeToConstant : public mlir::RewritePattern {
  SizeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Size", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 operand and 1 result");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "dynamic input dimension cannot be folded for onnx.Size");
    if (resultType.getRank() != 0)
      return rewriter.notifyMatchFailure(op, "expected scalar tensor result");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op,
                                         "expected i64 output element type");

    int64_t numElements = inputType.getNumElements();
    auto signlessI64 = mlir::IntegerType::get(
        rewriter.getContext(), 64,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto constType = mlir::RankedTensorType::get({}, signlessI64);
    auto scalar = mlir::arith::ConstantIntOp::create(rewriter, op->getLoc(),
                                                     numElements, 64);
    auto folded = mlir::tensor::FromElementsOp::create(
        rewriter, op->getLoc(), constType, scalar.getResult());
    rewriter.replaceOp(op, folded.getResult());
    return mlir::success();
  }
};

/// onnx.Range static fold: ONNX types i16/i32/i64/f32/f64 scalar constants.
struct RangeToConstant : public mlir::RewritePattern {
  RangeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "expected 3 operands and 1 result");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D tensor result");

    mlir::Type elemTy = resultType.getElementType();
    if (!isOnnxRangeElementType(elemTy))
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    mlir::Location loc = op->getLoc();
    llvm::SmallVector<mlir::Value> elements;

    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      auto startOr = readIntScalarAsI64(op->getOperand(0));
      auto limitOr = readIntScalarAsI64(op->getOperand(1));
      auto deltaOr = readIntScalarAsI64(op->getOperand(2));
      if (mlir::failed(startOr) || mlir::failed(limitOr) ||
          mlir::failed(deltaOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.Range fold needs constant integer start/limit/delta");

      int64_t start = *startOr;
      int64_t limit = *limitOr;
      int64_t delta = *deltaOr;
      if (delta == 0) {
        op->emitError("onnx.Range requires non-zero delta");
        return mlir::failure();
      }

      llvm::SmallVector<int64_t> values;
      if ((delta > 0 && start < limit) || (delta < 0 && start > limit)) {
        for (int64_t v = start; (delta > 0) ? (v < limit) : (v > limit);
             v += delta) {
          values.push_back(v);
        }
      }

      if (!resultType.isDynamicDim(0) &&
          resultType.getDimSize(0) != static_cast<int64_t>(values.size()))
        return rewriter.notifyMatchFailure(
            op, "output length mismatch after onnx.Range fold");

      auto foldedTy = mlir::RankedTensorType::get(
          {static_cast<int64_t>(values.size())}, elemTy);
      elements.reserve(values.size());
      for (int64_t v : values)
        elements.push_back(mlir::arith::ConstantIntOp::create(
            rewriter, loc, v, intTy.getWidth()));
      auto folded = mlir::tensor::FromElementsOp::create(rewriter, loc,
                                                         foldedTy, elements);
      rewriter.replaceOp(op, folded.getResult());
      return mlir::success();
    }

    auto floatTy = mlir::cast<mlir::FloatType>(elemTy);
    auto startOr = readFloatScalar(op->getOperand(0), floatTy);
    auto limitOr = readFloatScalar(op->getOperand(1), floatTy);
    auto deltaOr = readFloatScalar(op->getOperand(2), floatTy);
    if (mlir::failed(startOr) || mlir::failed(limitOr) || mlir::failed(deltaOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.Range fold needs constant float start/limit/delta");

    llvm::APFloat start = *startOr;
    llvm::APFloat limit = *limitOr;
    llvm::APFloat delta = *deltaOr;
    if (delta.isZero()) {
      op->emitError("onnx.Range requires non-zero delta");
      return mlir::failure();
    }

    double sd = start.convertToDouble();
    double ld = limit.convertToDouble();
    double dd = delta.convertToDouble();
    size_t n = onnxRangeNumElements(sd, ld, dd);
    llvm::SmallVector<llvm::APFloat> values;
    values.reserve(n);
    llvm::APFloat acc = start;
    llvm::APFloat step = delta;
    for (size_t i = 0; i < n; ++i) {
      (void)i;
      values.push_back(acc);
      acc.add(step, llvm::APFloat::rmNearestTiesToEven);
    }

    if (!resultType.isDynamicDim(0) &&
        resultType.getDimSize(0) != static_cast<int64_t>(values.size()))
      return rewriter.notifyMatchFailure(
          op, "output length mismatch after onnx.Range fold");

    auto foldedTy = mlir::RankedTensorType::get(
        {static_cast<int64_t>(values.size())}, elemTy);
    elements.reserve(values.size());
    for (const llvm::APFloat &apf : values)
      elements.push_back(
          mlir::arith::ConstantFloatOp::create(rewriter, loc, floatTy, apf));
    auto folded =
        mlir::tensor::FromElementsOp::create(rewriter, loc, foldedTy, elements);
    rewriter.replaceOp(op, folded.getResult());
    return mlir::success();
  }
};

/// onnx.Range dynamic lowering when operands are not all constants.
/// Emits scf.for + tensor.empty/tensor.insert; output leading dim must be
/// dynamic (cannot prove static length from non-constant inputs here).
struct RangeToScf : public mlir::RewritePattern {
  RangeToScf(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "expected 3 operands and 1 result");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D ranked result");
    if (!resultType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op,
          "dynamic onnx.Range lowering expects a dynamic leading dimension");

    mlir::Type elemTy = resultType.getElementType();
    if (!isOnnxRangeElementType(elemTy))
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    for (mlir::Value v : op->getOperands()) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
      if (!t || t.getRank() != 0 || t.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "expected 0-D scalar tensor operands matching result element "
                "type");
    }

    mlir::Location loc = op->getLoc();

    auto extractScalar = [&](mlir::Value tensor) -> mlir::Value {
      return rewriter.create<mlir::tensor::ExtractOp>(loc, tensor,
                                                      mlir::ValueRange{});
    };

    mlir::Value startS = extractScalar(op->getOperand(0));
    mlir::Value limitS = extractScalar(op->getOperand(1));
    mlir::Value deltaS = extractScalar(op->getOperand(2));

    mlir::Type calcTy =
        elemTy.isF64() ? rewriter.getF64Type() : rewriter.getF32Type();

    auto promote = [&](mlir::Value v) -> mlir::Value {
      if (mlir::isa<mlir::FloatType>(elemTy)) {
        if (v.getType() == calcTy)
          return v;
        if (elemTy.isF32() && calcTy.isF64())
          return rewriter.create<mlir::arith::ExtFOp>(loc, calcTy, v);
        return rewriter.create<mlir::arith::TruncFOp>(loc, calcTy, v);
      }
      return rewriter.create<mlir::arith::SIToFPOp>(loc, calcTy, v);
    };

    mlir::Value startF = promote(startS);
    mlir::Value limitF = promote(limitS);
    mlir::Value deltaF = promote(deltaS);

    mlir::Value diff =
        rewriter.create<mlir::arith::SubFOp>(loc, limitF, startF);
    mlir::Value ratio = rewriter.create<mlir::arith::DivFOp>(loc, diff, deltaF);
    mlir::Value ceiled = rewriter.create<mlir::math::CeilOp>(loc, ratio);
    auto calcFloatTy = mlir::cast<mlir::FloatType>(calcTy);
    llvm::APFloat zeroApf =
        llvm::APFloat::getZero(calcFloatTy.getFloatSemantics(),
                               /*negative=*/false);
    mlir::Value zeroF = rewriter.create<mlir::arith::ConstantFloatOp>(
        loc, calcFloatTy, zeroApf);
    mlir::Value isNeg = rewriter.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OLT, ceiled, zeroF);
    mlir::Value nF =
        rewriter.create<mlir::arith::SelectOp>(loc, isNeg, zeroF, ceiled);

    mlir::Value nI64 =
        rewriter.create<mlir::arith::FPToSIOp>(loc, rewriter.getI64Type(), nF);
    mlir::Value nIdx = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIndexType(), nI64);

    auto dynamic1d =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, elemTy);
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        loc, dynamic1d, mlir::ValueRange{nIdx});

    mlir::Value lb = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

    auto forOp = rewriter.create<mlir::scf::ForOp>(loc, lb, nIdx, step,
                                                   mlir::ValueRange{empty});
    {
      mlir::PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      mlir::Value iv = forOp.getInductionVar();
      mlir::Value cur = forOp.getRegionIterArg(0);

      mlir::Value elemVal;
      if (mlir::isa<mlir::IntegerType>(elemTy)) {
        mlir::Value ivInt =
            rewriter.create<mlir::arith::IndexCastOp>(loc, elemTy, iv);
        mlir::Value prod =
            rewriter.create<mlir::arith::MulIOp>(loc, deltaS, ivInt);
        elemVal = rewriter.create<mlir::arith::AddIOp>(loc, startS, prod);
      } else {
        mlir::Value ivI64 = rewriter.create<mlir::arith::IndexCastOp>(
            loc, rewriter.getI64Type(), iv);
        mlir::Value ivFloat = rewriter.create<mlir::arith::SIToFPOp>(
            loc, mlir::cast<mlir::FloatType>(elemTy), ivI64);
        mlir::Value prod =
            rewriter.create<mlir::arith::MulFOp>(loc, deltaS, ivFloat);
        elemVal = rewriter.create<mlir::arith::AddFOp>(loc, startS, prod);
      }

      mlir::Value inserted = rewriter.create<mlir::tensor::InsertOp>(
          loc, elemVal, cur, mlir::ValueRange{iv});
      rewriter.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{inserted});
    }

    rewriter.replaceOp(op, forOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateCompileTimeOpsConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<ShapeToConstant, SizeToConstant, RangeToConstant, RangeToScf>(
      ctx);
}

} // namespace hip
} // namespace mlir
