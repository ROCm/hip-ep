//===- PowerConversion.cpp - ONNX-to-HIP Power conversion ----- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Reciprocal -> hip.reciprocal
/// Converts ONNX reciprocal (y = 1/x, full signed IEEE domain) to HIP.
/// Runtime uses a HIP elementwise kernel (not MIOpen POWER activation).
struct ReciprocalToHip : public RewritePattern {
  ReciprocalToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Reciprocal", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

/// onnx.Sqrt -> hip.sqrt
/// ONNX element-wise sqrt; lowered to @wrap_power(0, 1, 0.5) and executed
/// with hip_elementwise_sqrt at runtime (not MIOpen POWER).
struct SqrtToHip : public RewritePattern {
  SqrtToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Sqrt", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult
ReciprocalToHip::matchAndRewrite(Operation* op,
                                 PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  if (!isa<RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor input");
  if (!isa<RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor result");
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::ReciprocalOp::create(rewriter, loc, resultType,
                                               context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

LogicalResult SqrtToHip::matchAndRewrite(Operation* op,
                                         PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  if (!isa<RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor input");
  if (!isa<RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor result");
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SqrtOp::create(rewriter, loc, resultType, context,
                                         input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populatePowerConversionPatterns(RewritePatternSet& patterns,
                                                MLIRContext* ctx) {
  patterns.add<ReciprocalToHip, SqrtToHip>(ctx);
}

} // namespace hip
} // namespace mlir
