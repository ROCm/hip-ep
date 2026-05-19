/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Add -> hip.miopen.add
struct AddToHip : public mlir::RewritePattern {
  AddToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Add", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Mul -> hip.mul
struct MulToHip : public mlir::RewritePattern {
  MulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Sub -> hip.sub
struct SubToHip : public mlir::RewritePattern {
  SubToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sub", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
AddToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
  mlir::Value source = (aType.getRank() == resultType.getRank()) ? a : b;
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp =
      mlir::hip::AddOp::create(rewriter, loc, resultType, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
MulToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Use the operand whose rank matches the result for dim extraction
  // (handles scalar * tensor broadcasting).
  auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
  mlir::Value source = (aType.getRank() == resultType.getRank()) ? a : b;
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp =
      mlir::hip::MulOp::create(rewriter, loc, resultType, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
SubToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  // Use the operand whose rank matches the result for dynamic dim extraction
  // (the other may be a broadcast scalar with fewer dims).
  auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
  mlir::Value source = (lhsType.getRank() == resultType.getRank()) ? lhs : rhs;
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);
  auto hipOp = mlir::hip::SubOp::create(rewriter, loc, resultType, context, lhs,
                                        rhs, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateElementwiseConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<AddToHip, MulToHip, SubToHip>(ctx);
}

} // namespace hip
} // namespace mlir
