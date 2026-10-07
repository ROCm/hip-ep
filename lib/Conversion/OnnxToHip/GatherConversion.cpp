/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipShapeUtilsGather.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX Gather -> HIP Gather
//===----------------------------------------------------------------------===//

struct GatherToHip : public mlir::RewritePattern {
  GatherToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    // Get axis attribute from ONNX Gather operation
    int64_t axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    // Get result type
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Output shape is `data[:axis] ++ indices.shape ++ data[axis+1:]`. Use the
    // same helper that backs `GatherOp::reifyResultShapes` so the destination
    // and the shape consumers observe cannot disagree.
    mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> resultShape =
        mlir::hip::reifyGatherWithAxis(rewriter, loc, data, indices, axis);
    if (mlir::failed(resultShape))
      return rewriter.notifyMatchFailure(op, "Gather axis is not reifiable");
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, resultType, *resultShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Gather result type is incompatible with the gathered shape");

    // Create hip.gather operation
    auto gatherOp = mlir::hip::GatherOp::create(rewriter, loc, context, data,
                                                indices, *init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGatherConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<GatherToHip>(ctx);
}

} // namespace hip
} // namespace mlir
