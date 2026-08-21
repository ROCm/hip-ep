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

struct MinToHipsr : public ::mlir::OpConversionPattern<::mlir::onnx::MinOp> {
  MinToHipsr(const ::mlir::TypeConverter &typeConverter,
             ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::MinOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    ::mlir::ValueRange inputs = adaptor.getData_0();
    if (inputs.empty()) {
      return rewriter.notifyMatchFailure(op, "expected at least one input");
    }

    if (inputs.size() == 1) {
      rewriter.replaceOp(op, inputs[0]);
      return ::mlir::success();
    }

    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op.getMin().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op.getLoc();
    ::mlir::Value accumulate = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
      ::mlir::Value rhs = inputs[i];
      auto accumulateType =
          ::mlir::dyn_cast<::mlir::RankedTensorType>(accumulate.getType());
      if (!accumulateType) {
        return rewriter.notifyMatchFailure(op, "expected ranked tensor inputs");
      }
      ::mlir::RankedTensorType stepType =
          (i + 1 == inputs.size()) ? resultType : accumulateType;

      ::mlir::Value init =
          PlaceholderOp::create(rewriter, loc, ::mlir::TypeRange{stepType},
                                *ctx, ::mlir::ValueRange{accumulate, rhs},
                                PlaceholderType::Normal)
              .getResult(0);
      accumulate = MinOp::create(rewriter, loc, ::mlir::TypeRange{stepType},
                                 *ctx, accumulate, rhs, init)
                       .getResult(0);
    }

    rewriter.replaceOp(op, accumulate);
    return ::mlir::success();
  }
};

} // namespace

void populateMinConversionPatterns(const ::mlir::TypeConverter &typeConverter,
                                   ::mlir::RewritePatternSet &patterns,
                                   ::mlir::MLIRContext *ctx) {
  patterns.add<MinToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
