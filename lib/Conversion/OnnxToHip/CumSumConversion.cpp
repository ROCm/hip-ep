/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.CumSum -> hip.cumsum
struct CumSumToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CumSumToHip)
  CumSumToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.CumSum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);
    mlir::Value axis = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Output shape mirrors the data input shape.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

    int64_t exclusive = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("exclusive"))
      exclusive = attr.getValue().getSExtValue();
    int64_t reverse = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("reverse"))
      reverse = attr.getValue().getSExtValue();

    auto hipOp =
        mlir::hip::CumSumOp::create(rewriter, loc, context, x, axis, init,
                                    rewriter.getI64IntegerAttr(exclusive),
                                    rewriter.getI64IntegerAttr(reverse));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateCumSumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<CumSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
