/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Mod -> hip.mod
struct ModToHip : public mlir::RewritePattern {
  ModToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mod", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    mlir::FailureOr<mlir::Value> initOrFailure =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Mod: no ranked operand spans dynamic result dim");

    int64_t fmod = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("fmod"))
      fmod = attr.getValue().getSExtValue();

    auto hipOp =
        mlir::hip::ModOp::create(rewriter, loc, context, a, b, *initOrFailure,
                                 rewriter.getI64IntegerAttr(fmod));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateModConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<ModToHip>(ctx);
}

} // namespace hip
} // namespace mlir
