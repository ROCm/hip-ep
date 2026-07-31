/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Lower variadic `onnx.Min` to pairwise `hip.min` operations.
///
/// Before:
///   %r = "onnx.Min"(%a, %b, %c) : (...) -> tensor<2x3x4xf32>
/// After:
///   %ab_init = tensor.empty() : tensor<3x4xf32>
///   %ab = hip.min ... outs(%ab_init : tensor<3x4xf32>)
///   %abc_init = tensor.empty() : tensor<2x3x4xf32>
///   %r = hip.min ... outs(%abc_init : tensor<2x3x4xf32>)
struct MinToHip : public mlir::RewritePattern {
  MinToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Min", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return lowerVariadicBroadcastChain<mlir::hip::MinOp>(op, rewriter);
  }
};

} // namespace

void populateMinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<MinToHip>(ctx);
}

} // namespace hip
} // namespace mlir
