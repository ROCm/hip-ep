/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

/// onnx.Cast -> hipsr.cast. The output shape equals the input shape; the shape
/// region is filled by CastOp::populateShapeRegion() (single source of truth).
struct CastToHipsr : public ::mlir::RewritePattern {
  CastToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Location loc = op->getLoc();
    ::mlir::Value input = op->getOperand(0);
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    // DPS init: an empty tensor of the result type. Cast preserves the shape,
    // so dynamic dims are taken from the input.
    ::llvm::SmallVector<::mlir::Value> dynDims;
    for (int64_t i = 0; i < resultType.getRank(); ++i)
      if (resultType.isDynamicDim(i))
        dynDims.push_back(rewriter.create<::mlir::tensor::DimOp>(loc, input, i));
    ::mlir::Value init = rewriter.create<::mlir::tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), dynDims);

    auto castOp = rewriter.create<CastOp>(loc, ::mlir::TypeRange{resultType},
                                          input, init);
    // Single source of truth: emit the shape region from the op itself.
    castOp.populateShapeRegion(rewriter, castOp.getShapeRegion());

    rewriter.replaceOp(op, castOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateCastConversionPatterns(::mlir::RewritePatternSet &patterns,
                                    ::mlir::MLIRContext *ctx) {
  patterns.add<CastToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
