/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static mlir::FailureOr<int64_t> getI64ScalarConstant(mlir::Value v) {
  auto fromElements =
      [](mlir::DenseElementsAttr attr) -> mlir::FailureOr<int64_t> {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(attr.getType());
    if (!type || type.getNumElements() != 1)
      return mlir::failure();
    auto elemType = type.getElementType();
    if (!elemType.isInteger(64))
      return mlir::failure();
    return (*attr.getValues<int64_t>().begin());
  };

  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
    if (auto denseAttr =
            mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue()))
      return fromElements(denseAttr);
    return mlir::failure();
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return mlir::failure();
  auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      def->getAttrOfType<mlir::ElementsAttr>("value"));
  if (!valueAttr)
    return mlir::failure();
  return fromElements(valueAttr);
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

/// onnx.Range -> tensor.from_elements(arith.constant ...)
///
/// Compile-time fold only for i64 scalar constants.
struct RangeToConstant : public mlir::RewritePattern {
  RangeToConstant(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/1, ctx) {}

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
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op,
                                         "expected i64 output element type");

    auto startOr = getI64ScalarConstant(op->getOperand(0));
    auto limitOr = getI64ScalarConstant(op->getOperand(1));
    auto deltaOr = getI64ScalarConstant(op->getOperand(2));
    if (mlir::failed(startOr) || mlir::failed(limitOr) || mlir::failed(deltaOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.Range requires i64 scalar constants for start/limit/delta");

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

    auto signlessI64 = mlir::IntegerType::get(
        rewriter.getContext(), 64,
        mlir::IntegerType::SignednessSemantics::Signless);
    auto constType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(values.size())}, signlessI64);
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(values.size());
    for (int64_t v : values)
      elements.push_back(
          mlir::arith::ConstantIntOp::create(rewriter, op->getLoc(), v, 64));
    auto folded = mlir::tensor::FromElementsOp::create(rewriter, op->getLoc(),
                                                       constType, elements);
    rewriter.replaceOp(op, folded.getResult());
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateCompileTimeOpsConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ShapeToConstant, SizeToConstant, RangeToConstant>(ctx);
}

} // namespace hip
} // namespace mlir
