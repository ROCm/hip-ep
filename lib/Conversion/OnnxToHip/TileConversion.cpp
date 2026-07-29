/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Tile -> hip.tile
struct TileToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileToHip)
  TileToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tile", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value repeats = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Trust shape inference: output rank == input rank, dims may be dynamic.
    // Use input as the source for any dynamic dim sizes.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

    auto hipOp =
        mlir::hip::TileOp::create(rewriter, loc, context, input, repeats, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTileConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<TileToHip>(ctx);
}

} // namespace hip
} // namespace mlir
