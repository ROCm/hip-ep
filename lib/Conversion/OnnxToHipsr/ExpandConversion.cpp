/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>

namespace mlir {
namespace hipsr {
namespace {

struct ExpandToHipsr
    : public ::mlir::OpConversionPattern<::mlir::onnx::ExpandOp> {
  ExpandToHipsr(const ::mlir::TypeConverter &typeConverter,
                ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::ExpandOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    ::mlir::Value input = adaptor.getInput();
    ::mlir::Value shape = adaptor.getShape();
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
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op.getOutput().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);
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

    ::mlir::Location loc = op.getLoc();
    ::mlir::Value init =
        PlaceholderOp::create(rewriter, loc, ::mlir::TypeRange{resultType},
                              *ctx, ::mlir::ValueRange{input, shape},
                              PlaceholderType::Barrier)
            .getResult(0);
    auto expandOp =
        ExpandOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                         input, shape, init, ::mlir::DenseI64ArrayAttr{});
    rewriter.replaceOp(op, expandOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns, ::mlir::MLIRContext *ctx) {
  patterns.add<ExpandToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
