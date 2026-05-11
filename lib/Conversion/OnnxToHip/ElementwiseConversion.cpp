//===- ElementwiseConversion.cpp - ONNX-to-HIP Elementwise conversion - *- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Add -> hip.miopen.add
struct AddToHip : public RewritePattern {
  AddToHip(MLIRContext *ctx) : RewritePattern("onnx.Add", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// onnx.Mul -> hip.mul
struct MulToHip : public RewritePattern {
  MulToHip(MLIRContext *ctx) : RewritePattern("onnx.Mul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// onnx.Sub -> hip.sub
struct SubToHip : public RewritePattern {
  SubToHip(MLIRContext *ctx) : RewritePattern("onnx.Sub", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult AddToHip::matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value a = op->getOperand(0);
  Value b = op->getOperand(1);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  auto aType = cast<RankedTensorType>(a.getType());
  Value source = (aType.getRank() == resultType.getRank()) ? a : b;
  Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp =
      mlir::hip::AddOp::create(rewriter, loc, resultType, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

LogicalResult MulToHip::matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value a = op->getOperand(0);
  Value b = op->getOperand(1);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // Use the operand whose rank matches the result for dim extraction
  // (handles scalar * tensor broadcasting).
  auto aType = cast<RankedTensorType>(a.getType());
  Value source = (aType.getRank() == resultType.getRank()) ? a : b;
  Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp =
      mlir::hip::MulOp::create(rewriter, loc, resultType, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

LogicalResult SubToHip::matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, lhs);
  auto hipOp = mlir::hip::SubOp::create(rewriter, loc, resultType, context, lhs,
                                        rhs, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateElementwiseConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<AddToHip, MulToHip, SubToHip>(ctx);
}

} // namespace hip
} // namespace mlir
