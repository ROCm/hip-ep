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

  mlir::FailureOr<mlir::Value> initOrFailure =
      createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "Add: no ranked operand spans dynamic result dim");

  auto hipOp =
      mlir::hip::AddOp::create(rewriter, loc, context, a, b, *initOrFailure);
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

  mlir::FailureOr<mlir::Value> initOrFailure =
      createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "Mul: no ranked operand spans dynamic result dim");

  auto hipOp =
      mlir::hip::MulOp::create(rewriter, loc, context, a, b, *initOrFailure);
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
  mlir::FailureOr<mlir::Value> initOrFailure =
      createBroadcastEmptyTensor(rewriter, loc, resultType, {lhs, rhs});
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "Sub: no ranked operand spans dynamic result dim");
  auto hipOp = mlir::hip::SubOp::create(rewriter, loc, context, lhs, rhs,
                                        *initOrFailure);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Custom(com.microsoft.GatedAdd) -> hip.mul + hip.add.
/// The contract is X + round_T(Y * gate), the rounding of a separate Mul and
/// Add, so the decomposition matches it exactly.
struct MicrosoftGatedAddToHip : public mlir::RewritePattern {
  MicrosoftGatedAddToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MicrosoftGatedAddToHip::matchAndRewrite(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GatedAdd") {
    return rewriter.notifyMatchFailure(op, "not a GatedAdd operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for GatedAdd");
  }
  if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(
        op, "com.microsoft GatedAdd expects 3 inputs and 1 result");
  }

  mlir::Value x = op->getOperand(0);
  mlir::Value y = op->getOperand(1);
  mlir::Value gate = op->getOperand(2);
  auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
  auto yType = mlir::dyn_cast<mlir::RankedTensorType>(y.getType());
  auto gateType = mlir::dyn_cast<mlir::RankedTensorType>(gate.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!xType || !yType || !gateType || !resultType) {
    return rewriter.notifyMatchFailure(op, "GatedAdd requires ranked tensors");
  }
  mlir::Type elemType = xType.getElementType();
  if (!elemType.isF16() && !elemType.isF32()) {
    return rewriter.notifyMatchFailure(op,
                                       "GatedAdd supports only f16 and f32");
  }
  if (yType != xType || resultType != xType ||
      gateType.getElementType() != elemType) {
    return rewriter.notifyMatchFailure(
        op, "GatedAdd X, Y and output must share one type");
  }
  int64_t rank = xType.getRank();
  if (rank < 1 || rank > 4 || gateType.getRank() != rank) {
    return rewriter.notifyMatchFailure(
        op, "GatedAdd requires rank 1-4 and gate of the same rank");
  }
  if (gateType.isDynamicDim(rank - 1) || gateType.getDimSize(rank - 1) != 1) {
    return rewriter.notifyMatchFailure(
        op, "GatedAdd gate last dimension must be 1");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();

  mlir::Value productInit = createEmptyTensor(rewriter, loc, resultType, y);
  mlir::Value product =
      mlir::hip::MulOp::create(rewriter, loc, context, y, gate, productInit)
          ->getResult(0);
  mlir::Value sumInit = createEmptyTensor(rewriter, loc, resultType, x);
  auto sum =
      mlir::hip::AddOp::create(rewriter, loc, context, x, product, sumInit);
  rewriter.replaceOp(op, sum->getResult(0));
  return mlir::success();
}

} // namespace

void populateElementwiseConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<AddToHip, MulToHip, SubToHip, MicrosoftGatedAddToHip>(ctx);
}

} // namespace hip
} // namespace mlir
