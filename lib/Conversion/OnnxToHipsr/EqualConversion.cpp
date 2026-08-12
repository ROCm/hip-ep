/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- EqualConversion.cpp - Convert onnx.Equal to hipsr.equal -----------===//
//
// onnx.Equal compares two device tensors elementwise with ONNX broadcasting,
// which hipsr.equal models directly. The placeholder's shape region is left
// empty for hipsr-populate-shape-region, as for every DPS operation.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

struct EqualToHipsr : public ::mlir::RewritePattern {
  EqualToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Equal", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected two operands and a single result");
    }

    ::mlir::Value lhs = op->getOperand(0);
    ::mlir::Value rhs = op->getOperand(1);
    auto lhsType = ::mlir::dyn_cast<::mlir::RankedTensorType>(lhs.getType());
    auto rhsType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rhs.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (lhsType.getElementType() != rhsType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching lhs and rhs element types");
    }
    // ONNX types the mask as bool, which importers spell as i1 or ui8.
    ::mlir::Type resultElementType = resultType.getElementType();
    if (!resultElementType.isInteger(1) && !resultElementType.isInteger(8)) {
      return rewriter.notifyMatchFailure(
          op, "expected an i1 or 8-bit integer result");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init =
        PlaceholderOp::create(rewriter, loc, ::mlir::TypeRange{resultType},
                              *ctx, ::mlir::ValueRange{lhs, rhs},
                              PlaceholderType::Normal)
            .getResult(0);
    auto equalOp = EqualOp::create(rewriter, loc, ::mlir::TypeRange{resultType},
                                   *ctx, lhs, rhs, init);
    rewriter.replaceOp(op, equalOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateEqualConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx) {
  patterns.add<EqualToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
