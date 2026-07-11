/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cstdlib>

namespace mlir {
namespace hip {
namespace {

/// HIPDNN_EP_FUSE_ELEMENTWISE: eliminate a redundant same-dtype onnx.Cast
/// (to == input element type), which is semantically the identity. The
/// Qwen3.5-VL exporter emits these on every transformer block (see the
/// same-dtype fast path in lib/Runtime/real/cast.cpp); folding them here
/// removes the hip.cast op, its runtime D2D copy, and the per-op cache-flush
/// barrier entirely. Pure identity forwarding -> bit-exact. Higher benefit than
/// CastToHip so it runs first.
struct CastElide : public mlir::RewritePattern {
  CastElide(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    const char *f = std::getenv("HIPDNN_EP_FUSE_ELEMENTWISE");
    if (!f || f[0] != '1')
      return rewriter.notifyMatchFailure(op, "cast-elide disabled");
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected cast shape");

    auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy)
      return rewriter.notifyMatchFailure(op, "need ranked tensors");
    if (inTy.getElementType() != outTy.getElementType())
      return rewriter.notifyMatchFailure(op, "type-changing cast (keep)");
    if (inTy != outTy)
      return rewriter.notifyMatchFailure(op, "same elem type but type mismatch");

    // Same-dtype cast == identity: forward the input.
    rewriter.replaceOp(op, op->getOperand(0));
    return mlir::success();
  }
};

/// onnx.Cast -> hip.cast
struct CastToHip : public mlir::RewritePattern {
  CastToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
CastToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Map MLIR element type to ONNX DataType enum
  mlir::Type targetType = resultType.getElementType();
  int64_t onnxDataType = 0;
  if (targetType.isF16())
    onnxDataType = 10;
  else if (targetType.isBF16())
    onnxDataType = 16;
  else if (targetType.isF32())
    onnxDataType = 1;
  else if (targetType.isF64())
    onnxDataType = 11;
  else if (targetType.isInteger(8))
    onnxDataType = 3;
  else if (targetType.isInteger(16))
    onnxDataType = 5;
  else if (targetType.isInteger(32))
    onnxDataType = 6;
  else if (targetType.isInteger(64))
    onnxDataType = 7;
  else if (targetType.isInteger(1))
    onnxDataType = 9;

  // Validate that we have a supported type
  if (onnxDataType == 0)
    return rewriter.notifyMatchFailure(op, "unsupported cast target type");

  auto toAttr = rewriter.getI64IntegerAttr(onnxDataType);

  auto hipOp =
      mlir::hip::CastOp::create(rewriter, loc, context, input, init, toAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateCastConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<CastElide>(ctx);
  patterns.add<CastToHip>(ctx);
}

} // namespace hip
} // namespace mlir
