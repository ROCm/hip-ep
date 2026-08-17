/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

static bool hasRepresentableStaticElementCount(ArrayRef<int64_t> shape) {
  APInt count(128, 1, /*isSigned=*/true);
  for (int64_t extent : shape) {
    if (ShapedType::isDynamic(extent))
      return true;
    count *= APInt(128, extent, /*isSigned=*/true);
    if (!count.isSignedIntN(64) || count.isNegative())
      return false;
  }
  return true;
}

/// onnx.Tile -> hip.tile
struct TileToHip : public mlir::RewritePattern {
  TileToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tile", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 2 inputs, 1 output");

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

    SmallVector<int64_t> extractedRepeats;
    DenseI64ArrayAttr staticRepeats;
    if (extractConstantIntVector(repeats, extractedRepeats))
      staticRepeats = rewriter.getDenseI64ArrayAttr(extractedRepeats);
    std::optional<ArrayRef<int64_t>> staticRepeatsValues;
    if (staticRepeats)
      staticRepeatsValues = staticRepeats.asArrayRef();

    SmallVector<OpFoldResult> resultShape;
    SmallVector<int64_t> inferredStaticShape;
    if (staticRepeatsValues) {
      auto inferredShape =
          inferTileShape(inputType.getShape(), *staticRepeatsValues);
      if (failed(inferredShape)) {
        op->emitOpError(
            "constant Tile repeats or extent products are invalid or "
            "unrepresentable");
        return failure();
      }
      if (!isResultTypeCompatibleWithPayloadShape(resultType, *inferredShape))
        return rewriter.notifyMatchFailure(
            op, "Tile result type contradicts constant repeats");
      if (!hasRepresentableStaticElementCount(*inferredShape)) {
        op->emitOpError(
            "constant Tile output element count is unrepresentable");
        return failure();
      }
      inferredStaticShape.assign(inferredShape->begin(), inferredShape->end());
    } else if (!hasRepresentableStaticElementCount(resultType.getShape())) {
      op->emitOpError("Tile output element count is unrepresentable");
      return failure();
    }

    Value repeatsValid =
        arith::ConstantIntOp::create(rewriter, loc, rewriter.getI1Type(), 1);
    SmallVector<Value> repeatValues;
    if (staticRepeatsValues) {
      repeatValues.reserve(staticRepeatsValues->size());
      for (int64_t repeat : *staticRepeatsValues)
        repeatValues.push_back(arith::ConstantIntOp::create(
            rewriter, loc, rewriter.getI64Type(), repeat));
    } else {
      SmallVector<Type> readbackTypes(1 + inputType.getRank(),
                                      rewriter.getI64Type());
      readbackTypes.front() = rewriter.getI1Type();
      auto readback = mlir::hip::ReadbackControlOp::create(
          rewriter, loc, readbackTypes, context, ValueRange{repeats});
      repeatsValid = readback.getValid();
      llvm::append_range(repeatValues, readback.getValues());
    }

    SmallVector<OpFoldResult> inputSizes =
        tensor::getMixedSizes(rewriter, loc, input);
    Value shapeValid = repeatsValid;
    Value priorElements = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> checkedExtents;
    checkedExtents.reserve(inputType.getRank());
    bool needsRuntimeCheck =
        !staticRepeatsValues || inputType.getNumDynamicDims() != 0;
    if (needsRuntimeCheck) {
      SmallVector<Type> checkedTypes{rewriter.getI1Type(),
                                     rewriter.getIndexType(),
                                     rewriter.getIndexType()};
      for (int64_t dim : llvm::seq<int64_t>(inputType.getRank())) {
        Value inputExtent =
            getValueOrCreateConstantIndexOp(rewriter, loc, inputSizes[dim]);
        auto checked = mlir::hip::CheckedTileExtentOp::create(
            rewriter, loc, checkedTypes, context, shapeValid, inputExtent,
            repeatValues[dim], priorElements,
            rewriter.getI64IntegerAttr(resultType.getDimSize(dim)));
        shapeValid = checked.getValid();
        checkedExtents.push_back(checked.getExtent());
        priorElements = checked.getElements();
      }
    }

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    resultShape.reserve(inputType.getRank());
    for (int64_t dim : llvm::seq<int64_t>(inputType.getRank())) {
      if (resultType.isDynamicDim(dim)) {
        if (!needsRuntimeCheck)
          resultShape.push_back(
              rewriter.getIndexAttr(inferredStaticShape[dim]));
        else
          resultShape.push_back(
              arith::SelectOp::create(rewriter, loc, shapeValid,
                                      checkedExtents[dim], zero)
                  .getResult());
      } else {
        resultShape.push_back(
            rewriter.getIndexAttr(resultType.getDimSize(dim)));
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
