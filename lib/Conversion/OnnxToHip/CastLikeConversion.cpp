/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.CastLike -> hip.cast (reuses existing Cast infrastructure)
///
/// CastLike casts the first input to the element type of the second input.
/// In MLIR the target element type is statically known from the result type,
/// so this is resolved entirely at compile time — no new Hip op, HipToLLVM
/// lowering, or runtime function needed.
///
/// Special case: when input and output element types are identical, the op
/// is a pure identity and is eliminated without generating any hip.cast.
struct CastLikeToHip : public mlir::RewritePattern {
  CastLikeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.CastLike", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
CastLikeToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(op, "expected at least 2 operands");

  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

  // Identity cast: input and output have the same element type.
  if (inputType.getElementType() == resultType.getElementType()) {
    rewriter.replaceOp(op, input);
    return mlir::success();
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

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

  if (onnxDataType == 0)
    return rewriter.notifyMatchFailure(op, "unsupported cast target type");

  auto toAttr = rewriter.getI64IntegerAttr(onnxDataType);

  auto hipOp = mlir::hip::CastOp::create(rewriter, loc, resultType, context,
                                         input, init, toAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void mlir::hip::populateCastLikeConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<CastLikeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
