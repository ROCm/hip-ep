/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

/// onnx.Cast -> hipsr.cast. The op's shape region is left empty (zero blocks)
/// here; a dedicated later pass populates every op's shape region uniformly, so
/// this stage does not call populateShapeRegion.
///
/// The ONNX `saturate` attribute (clamp-vs-wrap when casting to float8) is not
/// modeled on hipsr.cast yet, so it is currently ignored. Revisit once float8
/// cast targets are supported.
struct CastToHipsr : public ::mlir::RewritePattern {
  CastToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // Matching is by name on an (unregistered) ONNX op, so guard the shape:
    // onnx.Cast is single-input / single-result.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected a single operand and result");

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx))
      return ::mlir::failure();

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value input = op->getOperand(0);
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    // DPS init: a hipsr.placeholder with the cast result type. Its type matches
    // the result, which is all the DPS verifier needs, and it saves computing
    // the output shape here (no tensor.empty + tensor.dim). A later pass fills
    // in the shape region.
    //
    // Old way (built the init from the input shape):
    //   %d0   = tensor.dim %input, %c0 : tensor<?x8xf32>
    //   %init = tensor.empty(%d0)      : tensor<?x8xf16>
    //   %0    = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
    // New way (placeholder mirrors the result type):
    //   %init = hipsr.placeholder      : tensor<?x8xf16>
    //   %0    = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
    ::mlir::Value init =
        rewriter.create<PlaceholderOp>(loc, ::mlir::TypeRange{resultType})
            .getResult(0);

    auto castOp = rewriter.create<CastOp>(loc, ::mlir::TypeRange{resultType},
                                          *ctx, input, init);
    // The shape region is optional: leave it empty (zero blocks) here. A later
    // dedicated pass populates the shape computation.

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
