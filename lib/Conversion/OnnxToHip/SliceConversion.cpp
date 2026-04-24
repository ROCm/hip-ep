/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX Slice -> HIP Slice
//===----------------------------------------------------------------------===//
//
// ONNX Slice (opset 13+) operand order: data, starts, ends, [axes], [steps].
// Optional `axes` and `steps` may be absent or materialized as `onnx.NoValue`
// (a NoneType value). This pattern normalizes both cases into a hip.slice op
// whose optional operands are simply omitted.
//===----------------------------------------------------------------------===//

struct SliceToHip : public mlir::RewritePattern {
  SliceToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    unsigned numOperands = op->getNumOperands();
    if (numOperands < 3 || numOperands > 5)
      return rewriter.notifyMatchFailure(
          op, "expected 3-5 operands for ONNX Slice");

    mlir::Value data = op->getOperand(0);
    mlir::Value starts = op->getOperand(1);
    mlir::Value ends = op->getOperand(2);

    auto isNoneOperand = [](mlir::Value v) {
      return mlir::isa<mlir::NoneType>(v.getType());
    };

    mlir::Value axes = nullptr;
    if (numOperands > 3 && !isNoneOperand(op->getOperand(3)))
      axes = op->getOperand(3);

    mlir::Value steps = nullptr;
    if (numOperands > 4 && !isNoneOperand(op->getOperand(4)))
      steps = op->getOperand(4);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "Slice result must be a ranked tensor");

    // Build DPS init tensor. Dynamic output dimensions cannot be inferred from
    // the input tensor alone (they depend on starts/ends/steps), so only
    // patterns with static output shapes are supported here.
    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "dynamic output shapes are not yet supported by hip.slice");

    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{});

    auto hipOp = mlir::hip::SliceOp::create(rewriter, loc, resultType, context,
                                            data, starts, ends, axes, steps,
                                            init);

    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateSliceConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<SliceToHip>(ctx);
}

} // namespace hip
} // namespace mlir
