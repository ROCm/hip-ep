/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.matmul_nbits -> hip.matmul_nbits
///
/// This matches a custom torch op injected during model export for
/// int4-quantized linear layers. Signature:
///
///   %out = "torch.aten.matmul_nbits"(%A, %B, %scales, %zero_points, %bias)
///     attrs: K, N, bits, block_size, accuracy_level
///
/// Where:
///   A: activation tensor [..., K] (f16)
///   B: packed weight tensor [N, K/block_size, blob_size] (uint8)
///   scales: dequant scales [N, K/block_size] (f16)
///   zero_points: optional per-block zero points
///   bias: optional output bias [N]
struct TorchMatMulNBitsToHip : public mlir::RewritePattern {
  TorchMatMulNBitsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.matmul_nbits", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    if (op->getNumOperands() < 3)
      return rewriter.notifyMatchFailure(
          op, "expected at least 3 operands (A, B, scales)");

    mlir::Value A = op->getOperand(0);
    mlir::Value B = op->getOperand(1);
    mlir::Value scales = op->getOperand(2);

    // Optional operands
    mlir::Value zeroPoints;
    if (op->getNumOperands() > 3 && !isTorchNone(op->getOperand(3)))
      zeroPoints = op->getOperand(3);

    mlir::Value gIdx; // Not used in torch path

    mlir::Value bias;
    if (op->getNumOperands() > 4 && !isTorchNone(op->getOperand(4)))
      bias = op->getOperand(4);

    // Required attributes
    auto KAttr = op->getAttrOfType<mlir::IntegerAttr>("K");
    auto NAttr = op->getAttrOfType<mlir::IntegerAttr>("N");
    if (!KAttr || !NAttr)
      return rewriter.notifyMatchFailure(op, "missing K or N attribute");

    auto kI64 = rewriter.getI64IntegerAttr(KAttr.getValue().getSExtValue());
    auto nI64 = rewriter.getI64IntegerAttr(NAttr.getValue().getSExtValue());

    // Optional attributes with defaults
    auto bitsAttrOrig = op->getAttrOfType<mlir::IntegerAttr>("bits");
    auto bitsAttr =
        bitsAttrOrig
            ? rewriter.getI64IntegerAttr(bitsAttrOrig.getValue().getSExtValue())
            : rewriter.getI64IntegerAttr(4);

    auto blockSizeAttrOrig = op->getAttrOfType<mlir::IntegerAttr>("block_size");
    auto blockSizeAttr = blockSizeAttrOrig
                             ? rewriter.getI64IntegerAttr(
                                   blockSizeAttrOrig.getValue().getSExtValue())
                             : rewriter.getI64IntegerAttr(32);

    auto accAttrOrig = op->getAttrOfType<mlir::IntegerAttr>("accuracy_level");
    auto accuracyAttr =
        accAttrOrig
            ? rewriter.getI64IntegerAttr(accAttrOrig.getValue().getSExtValue())
            : rewriter.getI64IntegerAttr(4);

    auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensorForTorch(rewriter, loc, rt, A);

    auto hipOp = mlir::hip::MatMulNBitsOp::create(
        rewriter, loc, mlir::TypeRange{rt}, context, A, B, scales, zeroPoints,
        gIdx, bias, init, kI64, nI64, bitsAttr, blockSizeAttr, accuracyAttr);

    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populateTorchMatMulNBitsConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<TorchMatMulNBitsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
