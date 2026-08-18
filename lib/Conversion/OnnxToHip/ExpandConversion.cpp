/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Expand -> hip.expand
///
/// Constant targets validate through `inferExpandShape`. Runtime targets use
/// one grouped readback, then checked right-aligned extent merges. Once runtime
/// IR emission begins, failures become `shape_valid = false` and safe zero
/// extents rather than a failed partial rewrite.
struct ExpandToHip : public mlir::RewritePattern {
  ExpandToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

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
    mlir::Value shape = op->getOperand(1);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto shapeType = mlir::dyn_cast<mlir::RankedTensorType>(shape.getType());
    if (!resultType || !inputType || !shapeType)
      return rewriter.notifyMatchFailure(
          op, "Expand input, shape, and result must be ranked tensors");

    int64_t resultRank = resultType.getRank();
    int64_t inputRank = inputType.getRank();

    // Result rank is determined by right-aligning the input and target shapes.
    // A dynamic target length cannot define a ranked destination.
    if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0) ||
        (!shapeType.getElementType().isInteger(32) &&
         !shapeType.getElementType().isInteger(64)))
      return rewriter.notifyMatchFailure(
          op, "Expand shape must be a static-length rank-1 i32 or i64 tensor");
    int64_t shapeLen = shapeType.getDimSize(0);
    if (resultRank != std::max(inputRank, shapeLen))
      return rewriter.notifyMatchFailure(
          op, "Expand result rank must match the broadcast rank");

    llvm::SmallVector<int64_t> targetShape;
    std::optional<llvm::ArrayRef<int64_t>> staticTarget;
    if (shapeLen == 0 || extractConstantIntVector(shape, targetShape))
      staticTarget = targetShape;

    llvm::SmallVector<int64_t> inferredStaticShape;
    bool useStaticPlan = staticTarget && inputType.hasStaticShape();
    if (staticTarget) {
      auto inferred =
          mlir::hip::inferExpandShape(inputType.getShape(), *staticTarget);
      if (mlir::failed(inferred))
        return rewriter.notifyMatchFailure(
            op, "constant Expand target is negative or broadcast-incompatible");
      if (!isResultTypeCompatibleWithPayloadShape(resultType, *inferred))
        return rewriter.notifyMatchFailure(
            op, "Expand result type contradicts constant target");
      if (useStaticPlan) {
        int64_t elementCount = 1;
        for (int64_t extent : *inferred) {
          if (llvm::MulOverflow(elementCount, extent, elementCount)) {
            op->emitOpError(
                "constant Expand output element count is unrepresentable");
            return mlir::failure();
          }
        }
        inferredStaticShape.assign(inferred->begin(), inferred->end());
      }
    }
    mlir::Value shapeValid = mlir::arith::ConstantIntOp::create(
        rewriter, loc, rewriter.getI1Type(), 1);
    llvm::SmallVector<mlir::OpFoldResult> exactShape;
    exactShape.reserve(resultRank);
    if (useStaticPlan) {
      for (int64_t extent : inferredStaticShape)
        exactShape.push_back(rewriter.getIndexAttr(extent));
    } else {
      llvm::SmallVector<mlir::OpFoldResult> targetExtents;
      targetExtents.reserve(shapeLen);
      if (staticTarget) {
        for (int64_t extent : *staticTarget)
          targetExtents.push_back(rewriter.getIndexAttr(extent));
      } else {
        llvm::SmallVector<mlir::Type> readbackTypes(1 + shapeLen,
                                                    rewriter.getI64Type());
        readbackTypes.front() = rewriter.getI1Type();
        auto readback = mlir::hip::ReadbackControlOp::create(
            rewriter, loc, readbackTypes, context, mlir::ValueRange{shape},
            rewriter.getBoolAttr(true));
        shapeValid = readback.getValid();
        for (mlir::Value extent : readback.getValues()) {
          targetExtents.push_back(
              mlir::arith::IndexCastOp::create(rewriter, loc,
                                               rewriter.getIndexType(), extent)
                  .getResult());
        }
      }

      llvm::SmallVector<mlir::OpFoldResult> inputExtents =
          mlir::tensor::getMixedSizes(rewriter, loc, input);
      mlir::OpFoldResult one = rewriter.getIndexAttr(1);
      mlir::Value priorElements =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      llvm::SmallVector<mlir::Type> checkedTypes{rewriter.getI1Type(),
                                                 rewriter.getIndexType(),
                                                 rewriter.getIndexType()};
      for (int64_t axis : llvm::seq<int64_t>(0, resultRank)) {
        int64_t inputIdx = axis - (resultRank - inputRank);
        int64_t targetIdx = axis - (resultRank - shapeLen);
        mlir::Value inputExtent = mlir::getValueOrCreateConstantIndexOp(
            rewriter, loc, inputIdx < 0 ? one : inputExtents[inputIdx]);
        mlir::Value targetExtent = mlir::getValueOrCreateConstantIndexOp(
            rewriter, loc, targetIdx < 0 ? one : targetExtents[targetIdx]);
        auto checked = mlir::hip::CheckedExpandExtentOp::create(
            rewriter, loc, checkedTypes, context, shapeValid, inputExtent,
            targetExtent, priorElements,
            rewriter.getI64IntegerAttr(resultType.getDimSize(axis)));
        shapeValid = checked.getValid();
        priorElements = checked.getElements();
        exactShape.push_back(checked.getExtent());
      }
    }

    llvm::SmallVector<mlir::Value> dynSizes;
    dynSizes.reserve(resultType.getNumDynamicDims());
    for (int64_t axis : llvm::seq<int64_t>(0, resultRank))
      if (resultType.isDynamicDim(axis))
        dynSizes.push_back(mlir::getValueOrCreateConstantIndexOp(
            rewriter, loc, exactShape[axis]));
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp = mlir::hip::ExpandOp::create(rewriter, loc, context, shapeValid,
                                             input, shape, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ExpandToHip>(ctx);
}

} // namespace hip
} // namespace mlir
