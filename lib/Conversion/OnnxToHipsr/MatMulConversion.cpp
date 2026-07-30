/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

struct MatMulToHipsr : public ::mlir::RewritePattern {
  MatMulToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Matching is by name on an (unregistered) ONNX op, so guard the shape:
    // onnx.MatMul is two-input / single-result.
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected two operands and a single result");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value a = op->getOperand(0);
    ::mlir::Value b = op->getOperand(1);
    ::mlir::FailureOr<::mlir::Value> placeholderA =
        getPlaceholderDependency(a, op, rewriter);
    ::mlir::FailureOr<::mlir::Value> placeholderB =
        getPlaceholderDependency(b, op, rewriter);
    if (::mlir::failed(placeholderA) || ::mlir::failed(placeholderB)) {
      return ::mlir::failure();
    }
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }

    ::mlir::Value init = PlaceholderOp::create(
                             rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                             ::mlir::ValueRange{*placeholderA, *placeholderB},
                             PlaceholderType::Normal)
                             .getResult(0);

    auto matmulOp = MatMulOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx, a, b, init);

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateMatMulConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx) {
  patterns.add<MatMulToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
