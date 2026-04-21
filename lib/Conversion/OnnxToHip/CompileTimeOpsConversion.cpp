/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

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
      return rewriter.notifyMatchFailure(op, "expected i64 output element type");

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

    auto signlessI64 =
        mlir::IntegerType::get(rewriter.getContext(), 64,
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

} // namespace

void mlir::hip::populateCompileTimeOpsConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ShapeToConstant>(ctx);
}

} // namespace hip
} // namespace mlir
