/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// onnx.NonZero -> hip.nonzero
//
// ONNX NonZero (opset 13) takes any-rank input and returns
//   tensor<RANKx?xi64>
// whose columns are the row-major N-D coords of every nonzero element of
// the input.  The trailing dim is data-dependent.
//
// Our pipeline runs against pre-allocated DPS buffers, so we MUST publish a
// static K bound at compile time.  Strategy: require the parent pass (or
// upstream shape inference) to have stamped the result type with a static
// K.  When K is dynamic we bail; the caller can use static-shape inference
// (polygraphy + onnx-shape-inference) to fix it up before invoking the EP.
//
// The runtime kernel writes up to K coords; trailing slots are zero-filled
// so downstream Transpose+Gather/Slice users see well-defined data.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct NonzeroToHip : public RewritePattern {
  NonzeroToHip(MLIRContext *ctx)
      : RewritePattern("onnx.NonZero", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(op, "onnx.NonZero needs 1 operand");
    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.NonZero requires ranked tensor operand and result");

    int64_t rank = inputType.getRank();
    if (resultType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.NonZero result must be rank-2 (N, K)");
    int64_t resN = resultType.getDimSize(0);
    if (resN != rank)
      return rewriter.notifyMatchFailure(
          op, "onnx.NonZero result dim 0 must equal input rank");

    // K must be static.  If it's not, we can't allocate a DPS buffer.  Give
    // up early so a higher-level pre-pass (e.g. polygraphy --override-input-
    // shapes followed by onnxruntime shape inference) can be re-run with the
    // correct K bound.
    if (resultType.isDynamicDim(1))
      return rewriter.notifyMatchFailure(
          op, "onnx.NonZero requires static K (output dim 1)");

    Location loc = op->getLoc();
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        ValueRange{});
    auto hipOp =
        NonzeroOp::create(rewriter, loc, resultType, context, input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateNonzeroConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<NonzeroToHip>(ctx);
}

} // namespace hip
} // namespace mlir
