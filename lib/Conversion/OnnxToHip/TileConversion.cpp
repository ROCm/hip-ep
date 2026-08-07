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

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto repeatsType =
        mlir::dyn_cast<mlir::RankedTensorType>(repeats.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !repeatsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Tile requires ranked input, repeats, and result");
    if (repeatsType.getRank() != 1 ||
        !repeatsType.getElementType().isInteger(64) ||
        repeatsType.isDynamicDim(0) ||
        repeatsType.getDimSize(0) != inputType.getRank() ||
        resultType.getRank() != inputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Tile repeats must be a static-length rank-1 i64 tensor whose "
              "length equals the input/result rank");

    DenseI64ArrayAttr staticRepeats =
        op->getAttrOfType<DenseI64ArrayAttr>("hipdnn.tile_repeats");
    SmallVector<int64_t> extractedRepeats;
    if (!staticRepeats && extractConstantIntVector(repeats, extractedRepeats))
      staticRepeats = rewriter.getDenseI64ArrayAttr(extractedRepeats);
    std::optional<ArrayRef<int64_t>> staticRepeatsValues;
    if (staticRepeats)
      staticRepeatsValues = staticRepeats.asArrayRef();

    SmallVector<OpFoldResult> resultShape;
    if (staticRepeatsValues) {
      if (failed(inferTileShape(inputType.getShape(), *staticRepeatsValues)))
        return rewriter.notifyMatchFailure(
            op, "Tile repeats must match input rank and be non-negative");
      if (failed(reifyTileShape(rewriter, loc, input, repeats,
                                staticRepeatsValues, resultShape)))
        return failure();
    } else {
      // Runtime repeats are needed only for result dimensions that remain
      // dynamic after ONNX shape inference. Imported static extents stay
      // authoritative; this is common when a repeats tensor is assembled at
      // runtime from one dynamic entry and one constant 1.
      //
      // When at least one result extent is dynamic, one bulk readback produces
      // every repeat with a single stream synchronization. Fully static
      // results need no host readback at all.
      mlir::hip::ReadbackShapeOp readback;
      SmallVector<OpFoldResult> inputSizes;
      if (resultType.getNumDynamicDims() != 0) {
        SmallVector<Type> dimTypes(inputType.getRank(),
                                   rewriter.getIndexType());
        readback = mlir::hip::ReadbackShapeOp::create(
            rewriter, loc, dimTypes, context, repeats,
            rewriter.getI64IntegerAttr(inputType.getRank()));
        inputSizes = tensor::getMixedSizes(rewriter, loc, input);
      }
      resultShape.reserve(inputType.getRank());
      for (int64_t dim : llvm::seq<int64_t>(inputType.getRank())) {
        if (!resultType.isDynamicDim(dim)) {
          resultShape.push_back(
              rewriter.getIndexAttr(resultType.getDimSize(dim)));
          continue;
        }
        OpFoldResult inputSize = inputSizes[dim];
        Value inputExtent =
            getValueOrCreateConstantIndexOp(rewriter, loc, inputSize);
        resultShape.push_back(arith::MulIOp::create(rewriter, loc, inputExtent,
                                                    readback.getDims()[dim])
                                  .getResult());
      }
    }

    FailureOr<Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, resultType, resultShape);
    if (failed(init))
      return rewriter.notifyMatchFailure(
          op, "Tile result type is incompatible with inferred extents");

    auto hipOp = mlir::hip::TileOp::create(rewriter, loc, context, input,
                                           repeats, *init, staticRepeats);
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
