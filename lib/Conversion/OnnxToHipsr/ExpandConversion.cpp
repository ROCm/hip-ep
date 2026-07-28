/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include <algorithm>

namespace mlir {
namespace hipsr {
namespace {

struct ExpandToHipsr : public ::mlir::RewritePattern {
  ExpandToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected two operands and a single result");
    }

    ::mlir::Value input = op->getOperand(0);
    ::mlir::Value shape = op->getOperand(1);
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor input");
    }

    auto shapeType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(shape.getType());
    if (!shapeType || shapeType.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "expected rank-1 shape tensor");
    }
    if (!shapeType.getElementType().isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "expected i64 shape elements");
    }
    if (shapeType.isDynamicDim(0)) {
      return rewriter.notifyMatchFailure(op, "expected static shape length");
    }

    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    if (inputType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching input and result element types");
    }
    int64_t expectedRank =
        std::max(inputType.getRank(), shapeType.getDimSize(0));
    if (resultType.getRank() != expectedRank) {
      return rewriter.notifyMatchFailure(
          op, "result rank must equal max(input rank, shape length)");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init =
        PlaceholderOp::create(rewriter, loc, ::mlir::TypeRange{resultType})
            .getResult(0);
    auto expandOp =
        ExpandOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                         input, shape, init, ::mlir::DenseI64ArrayAttr{});
    rewriter.replaceOp(op, expandOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx) {
  patterns.add<ExpandToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
