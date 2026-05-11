//===- ActivationConversion.cpp - ONNX-to-HIP Activation conversion - *- C++
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

/// onnx.Softmax -> hip.miopen.softmax
struct SoftmaxToHip : public RewritePattern {
  SoftmaxToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Softmax", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult SoftmaxToHip::matchAndRewrite(Operation* op,
                                            PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, resultType,
                                                  context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

/// onnx.Sigmoid -> hip.sigmoid
struct SigmoidToHip : public RewritePattern {
  SigmoidToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Sigmoid", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult SigmoidToHip::matchAndRewrite(Operation* op,
                                            PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SigmoidOp::create(rewriter, loc, resultType, context,
                                            input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

/// onnx.Softplus -> hip.softplus
struct SoftplusToHip : public RewritePattern {
  SoftplusToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Softplus", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult SoftplusToHip::matchAndRewrite(Operation* op,
                                             PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SoftplusOp::create(rewriter, loc, resultType, context,
                                             input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

/// onnx.Gelu -> hip.gelu
struct GeluToHip : public RewritePattern {
  GeluToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Gelu", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult GeluToHip::matchAndRewrite(Operation* op,
                                         PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Extract and validate approximate attribute from ONNX op (default to "none")
  StringAttr approximateAttr = rewriter.getStringAttr("none");
  if (auto attr = op->getAttrOfType<StringAttr>("approximate")) {
    llvm::StringRef approxValue = attr.getValue();
    // Only "none" and "tanh" are valid per ONNX Gelu spec
    if (approxValue != "none" && approxValue != "tanh") {
      return rewriter.notifyMatchFailure(op, "invalid approximate attribute '" +
                                                 approxValue.str() +
                                                 "', must be 'none' or 'tanh'");
    }
    approximateAttr = attr;
  }

  auto hipOp = mlir::hip::GeluOp::create(rewriter, loc, resultType, context,
                                         input, init, approximateAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateActivationConversionPatterns(
    RewritePatternSet& patterns, MLIRContext* ctx) {
  patterns.add<SoftmaxToHip, SigmoidToHip, SoftplusToHip, GeluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
