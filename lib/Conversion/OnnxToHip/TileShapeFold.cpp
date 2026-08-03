/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TileShapeFold.cpp - Preserve constant Tile repeats ----------------===//
//
// Capture inline Tile repeats before lowerOnnxConstants externalizes the
// tensor. Dynamic result extents can then be computed exactly from input dims
// and this attribute without a runtime D2H readback.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "tile-shape-fold"

STATISTIC(NumTileConstStamps,
          "Number of onnx.Tile ops whose constant repeats were stamped");

namespace mlir {
namespace hip {
namespace {

struct TileStampConstRepeats : public RewritePattern {
  TileStampConstRepeats(MLIRContext *ctx)
      : RewritePattern("onnx.Tile", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("hipdnn.tile_repeats"))
      return rewriter.notifyMatchFailure(op, "tile.already_stamped");
    if (op->getNumOperands() < 2)
      return rewriter.notifyMatchFailure(op, "tile.arity");
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "tile.static_result");

    SmallVector<int64_t> repeats;
    if (!extractConstantIntVector(op->getOperand(1), repeats,
                                  CompileTimeConstantScope::InlineOnly))
      return rewriter.notifyMatchFailure(op, "tile.repeats_not_inline_i64");

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("hipdnn.tile_repeats",
                  rewriter.getDenseI64ArrayAttr(repeats));
    });
    ++NumTileConstStamps;
    return success();
  }
};

} // namespace

void populateTileShapeFoldPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<TileStampConstRepeats>(ctx);
}

} // namespace hip
} // namespace mlir
