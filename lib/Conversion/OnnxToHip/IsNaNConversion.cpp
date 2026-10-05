/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.IsNaN -> hip.isnan
///
/// Y = isnan(X). The input is floating point and the output is a 1-byte
/// boolean of the same shape. The ORT/morphizen frontend prints ONNX bool as
/// ui8; hand-written IR uses i1. Both are accepted.
///
/// Before:
///   %y = "onnx.IsNaN"(%x) : (tensor<24x20x1x1xf16>) -> tensor<24x20x1x1xui8>
/// After:
///   %init = tensor.empty() : tensor<24x20x1x1xui8>
///   %y = hip.isnan(%ctx) ins(%x : tensor<24x20x1x1xf16>)
///                        outs(%init : tensor<24x20x1x1xui8>)
struct IsNaNToHip : public mlir::RewritePattern {
  IsNaNToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.IsNaN", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.IsNaN expects 1 operand and 1 result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.IsNaN lowering expects ranked tensors");

    mlir::Type inputElem = inputType.getElementType();
    if (!inputElem.isF16() && !inputElem.isBF16() && !inputElem.isF32() &&
        !inputElem.isF64())
      return rewriter.notifyMatchFailure(
          op, "onnx.IsNaN input must be f16, bf16, f32, or f64");

    mlir::Type resultElem = resultType.getElementType();
    if (!resultElem.isInteger(1) && !resultElem.isInteger(8))
      return rewriter.notifyMatchFailure(
          op, "onnx.IsNaN result must be a 1-byte boolean (i1 or i8/ui8)");

    if (inputType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(
          op, "onnx.IsNaN input and result ranks must match");
    for (int64_t dim : llvm::seq<int64_t>(inputType.getRank())) {
      if (inputType.isDynamicDim(dim) || resultType.isDynamicDim(dim))
        continue;
      if (inputType.getDimSize(dim) != resultType.getDimSize(dim))
        return rewriter.notifyMatchFailure(
            op, "onnx.IsNaN input and result shapes must match");
    }

    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp =
        mlir::hip::IsNaNOp::create(rewriter, loc, context, input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateIsNaNConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<IsNaNToHip>(ctx);
}

} // namespace hip
} // namespace mlir
