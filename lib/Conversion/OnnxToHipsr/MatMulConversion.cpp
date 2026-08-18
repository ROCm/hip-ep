/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hipsr {
namespace {

struct MatMulToHipsr : public ::mlir::ConversionPattern {
  MatMulToHipsr(const ::mlir::TypeConverter &typeConverter,
                ::mlir::MLIRContext *ctx)
      : ConversionPattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::ArrayRef<::mlir::Value> operands,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
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
    ::mlir::Value a = operands[0];
    ::mlir::Value b = operands[1];
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);

    ::mlir::Value init = PlaceholderOp::create(
                             rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                             ::mlir::ValueRange{a, b}, PlaceholderType::Normal)
                             .getResult(0);

    auto matmulOp = MatMulOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx, a, b, init);

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateMatMulConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx) {
  patterns.add<MatMulToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
