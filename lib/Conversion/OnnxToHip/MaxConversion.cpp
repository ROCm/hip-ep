/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Lower variadic `onnx.Max` to pairwise `hip.max` operations.
///
/// Before:
///   %r = "onnx.Max"(%a, %b, %c) : (...) -> tensor<2x3xf32>
/// After:
///   %ab_init = tensor.empty() : tensor<2x3xf32>
///   %ab = hip.max ... outs(%ab_init : tensor<2x3xf32>)
///   %abc_init = tensor.empty() : tensor<2x3xf32>
///   %r = hip.max ... outs(%abc_init : tensor<2x3xf32>)
struct MaxToHip : public mlir::RewritePattern {
  MaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Max", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return lowerVariadicBroadcastChain<mlir::hip::MaxOp>(op, rewriter);
  }
};

} // namespace

void populateMaxConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<MaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
