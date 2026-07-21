/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

/// onnx.MatMul -> hipsr.matmul. The shape region is left empty here; a later
/// pass populates every op's shape region uniformly (see populateShapeRegion).
/// The `!hipsr.context` threaded onto the op comes from function arg 0 (added
/// in the ONNX phase).
///
/// Before:
///   %0 = "onnx.MatMul"(%A, %B) : (tensor<?x?xf16>, tensor<?x?xf16>)
///                                 -> tensor<?x?xf16>
/// After:
///   %init = hipsr.placeholder : tensor<?x?xf16>
///   %0 = hipsr.matmul(%ctx) ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>)
///                           outs(%init : tensor<?x?xf16>) : tensor<?x?xf16>
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
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }

    // A hipsr.placeholder mirroring the result type satisfies the DPS verifier
    // without computing the output shape here; a later pass fills that in.
    ::mlir::Value init =
        rewriter.create<PlaceholderOp>(loc, ::mlir::TypeRange{resultType})
            .getResult(0);

    auto matmulOp = rewriter.create<MatMulOp>(
        loc, ::mlir::TypeRange{resultType}, *ctx, a, b, init);

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
