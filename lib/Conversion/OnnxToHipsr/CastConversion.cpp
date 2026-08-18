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

/// The ONNX `saturate` attribute (clamp-vs-wrap when casting to float8) is not
/// modeled on hipsr.cast yet, so it is currently ignored. Revisit once float8
/// cast targets are supported.
struct CastToHipsr : public ::mlir::OpConversionPattern<::mlir::onnx::CastOp> {
  CastToHipsr(const ::mlir::TypeConverter &typeConverter,
              ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::CastOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op.getLoc();
    ::mlir::Value input = adaptor.getInput();
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op.getOutput().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);

    ::mlir::Value init =
        rewriter
            .create<PlaceholderOp>(loc, ::mlir::TypeRange{resultType}, *ctx,
                                   ::mlir::ValueRange{input},
                                   PlaceholderType::Normal)
            .getResult(0);

    auto castOp = rewriter.create<CastOp>(loc, ::mlir::TypeRange{resultType},
                                          *ctx, input, init);
    rewriter.replaceOp(op, castOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateCastConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                    ::mlir::RewritePatternSet &patterns,
                                    ::mlir::MLIRContext *ctx) {
  patterns.add<CastToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
