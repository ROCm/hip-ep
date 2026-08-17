/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipShapeUtilsShapeOps.h"

namespace mlir {
namespace hip {
namespace {

/// Read element \p idx of the 1-D `shape` operand to a host `index`. Folds a
/// constant shape; otherwise reads it back with a stream sync (see
/// ReadbackScalar.h) -- a bare host load of the GPU-computed extent races the
/// producing kernel and sizes the output from stale memory (a zero collapses to
/// a null buffer and `wrap_expand` fails; a wrong size collapses the result).
static mlir::Value readShapeEntryToIndex(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc, mlir::Value ctx,
                                         mlir::Value shape, int64_t idx) {
  mlir::Value scalar = readbackShapeEntryToHost(rewriter, loc, ctx, shape, idx);
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), scalar);
}

/// onnx.Expand -> hip.expand
///
/// We trust the result type produced by ONNX shape inference. For dynamic
/// dims in the result we extract the corresponding entry from the `shape`
/// input tensor (right-aligned with the result rank, NumPy-style); leading
/// dims that are absent from `shape` fall back to the matching input dim.
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

    // shape is a 1-D int tensor; for dynamic result dims we extract entries
    // from it (right-aligned with the result rank). When `shape` itself has
    // a dynamic length we cannot reason about it here, so bail out.
    if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0) ||
        (!shapeType.getElementType().isInteger(32) &&
         !shapeType.getElementType().isInteger(64)))
      return rewriter.notifyMatchFailure(
          op, "Expand shape must be a static-length rank-1 i32 or i64 tensor");
    int64_t shapeLen = shapeType.getDimSize(0);
    if (resultRank != std::max(inputRank, shapeLen))
      return rewriter.notifyMatchFailure(
          op, "Expand result rank must match the broadcast rank");

    mlir::Value init;
    llvm::SmallVector<mlir::OpFoldResult> constantShape;
    llvm::SmallVector<int64_t> shapeValues;
    std::optional<llvm::ArrayRef<int64_t>> staticShape;
    if (extractConstantIntVector(shape, shapeValues))
      staticShape = shapeValues;
    if (staticShape) {
      auto inferredShape =
          mlir::hip::inferExpandShape(inputType.getShape(), *staticShape);
      if (mlir::failed(inferredShape))
        return rewriter.notifyMatchFailure(
            op, "constant Expand shape is not broadcast-compatible");
      if (!isResultTypeCompatibleWithPayloadShape(resultType, *inferredShape))
        return rewriter.notifyMatchFailure(
            op, "Expand result type contradicts constant shape");
    }
    if (mlir::succeeded(mlir::hip::reifyExpandShape(
            rewriter, loc, input, shape, constantShape, staticShape))) {
      auto constantInit = createEmptyTensorFromReifiedShape(
          rewriter, loc, resultType, constantShape);
      if (mlir::failed(constantInit))
        return rewriter.notifyMatchFailure(
            op, "Expand result type contradicts constant broadcast shape");
      init = *constantInit;
    } else {
      // Preserve the payload-dynamic path exactly: every shape entry is read
      // back with synchronization before it sizes the destination.
      for (int64_t i = 0; i < resultRank; ++i) {
        if (!resultType.isDynamicDim(i))
          continue;
        int64_t shapeIdx = i - (resultRank - shapeLen);
        int64_t inputIdx = i - (resultRank - inputRank);
        if (shapeIdx < 0 && inputIdx < 0)
          return rewriter.notifyMatchFailure(
              op, "cannot resolve dynamic dim from input or shape");
      }

      llvm::SmallVector<mlir::Value> dynSizes;
      for (int64_t i = 0; i < resultRank; ++i) {
        if (!resultType.isDynamicDim(i))
          continue;
        int64_t shapeIdx = i - (resultRank - shapeLen);
        mlir::Value dim;
        if (shapeIdx >= 0) {
          dim = readShapeEntryToIndex(rewriter, loc, context, shape, shapeIdx);
        } else {
          int64_t inputIdx = i - (resultRank - inputRank);
          dim = mlir::tensor::DimOp::create(rewriter, loc, input, inputIdx);
        }
        dynSizes.push_back(dim);
      }
      init =
          mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        resultType.getElementType(), dynSizes);
    }

    auto hipOp =
        mlir::hip::ExpandOp::create(rewriter, loc, context, input, shape, init);
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
