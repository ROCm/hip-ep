/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hipsr {
namespace {

struct MatMulToHipsr
    : public ::mlir::OpConversionPattern<::mlir::onnx::MatMulOp> {
  MatMulToHipsr(const ::mlir::TypeConverter &typeConverter,
                ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::MatMulOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op.getLoc();
    ::mlir::Value a = adaptor.getA();
    ::mlir::Value b = adaptor.getB();
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op.getY().getType());
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
