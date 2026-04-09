/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX MatMulNBits -> HIP MatMulNBits (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct MatMulNBitsToHip : public mlir::RewritePattern {
  MatMulNBitsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulNBitsToHip::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MatMulNBits") {
    return rewriter.notifyMatchFailure(op, "not a MatMulNBits custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 3) {
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs for MatMulNBits");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for MatMulNBits");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  mlir::Value A = op->getOperand(0);
  mlir::Value B = op->getOperand(1);
  mlir::Value scales = op->getOperand(2);

  auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
    if (idx >= op->getNumOperands()) {
      return mlir::Value{};
    }
    mlir::Value v = op->getOperand(idx);
    if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
      return mlir::Value{};
    }
    return v;
  };
  mlir::Value zeroPoints = getOptionalInput(3);
  mlir::Value gIdx = getOptionalInput(4);
  mlir::Value bias = getOptionalInput(5);

  auto KAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("K").getSInt());
  auto NAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("N").getSInt());

  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  auto bitsAttr =
      rewriter.getI64IntegerAttr(bitsIntAttr ? bitsIntAttr.getSInt() : 4);

  auto blockSizeAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("block_size").getSInt());

  auto accuracyIntAttr = op->getAttrOfType<mlir::IntegerAttr>("accuracy_level");
  auto accuracyLevelAttr = rewriter.getI64IntegerAttr(
      accuracyIntAttr ? accuracyIntAttr.getSInt() : 0);

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, A);

  auto hipOp = mlir::hip::MatMulNBitsOp::create(
      rewriter, loc, mlir::TypeRange{rt}, context, A, B, scales, zeroPoints,
      gIdx, bias, init, KAttr, NAttr, bitsAttr, blockSizeAttr,
      accuracyLevelAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void mlir::hip::populateMatMulNBitsConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MatMulNBitsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
