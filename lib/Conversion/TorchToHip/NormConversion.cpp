/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.rms_norm -> hip.rms_norm
///
/// torch.aten.rms_norm signature:
///   %out = "torch.aten.rms_norm"(%input, %normalized_shape, %weight, %eps)
///     : (tensor, !torch.list<int>, tensor, !torch.float) -> tensor
///
/// hip.rms_norm signature:
///   %out = hip.rms_norm(%context, %input, %scale, %init,
///                       {axis, epsilon, stash_type})
struct TorchRmsNormToHip : public mlir::RewritePattern {
  TorchRmsNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.rms_norm", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Operands: input, normalized_shape, weight, eps
    if (op->getNumOperands() < 3)
      return rewriter.notifyMatchFailure(op, "expected at least 3 operands");

    mlir::Value input = op->getOperand(0);
    // operand 1 = normalized_shape (list<int>, used to determine axis)
    mlir::Value weight = op->getOperand(2);

    // Extract epsilon from operand 3 (torch.constant.float or attribute)
    double epsilon = 1e-6;
    if (op->getNumOperands() > 3) {
      mlir::Value epsVal = op->getOperand(3);
      auto *epsDefOp = epsVal.getDefiningOp();
      if (epsDefOp) {
        if (auto floatAttr =
                epsDefOp->getAttrOfType<mlir::FloatAttr>("value")) {
          epsilon = floatAttr.getValueAsDouble();
        }
      }
    }

    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Axis: RMSNorm normalizes over the last N dimensions specified by
    // normalized_shape. For transformer models this is always the last dim
    // (axis = -1 or rank-1).
    int64_t axis = inputType.getRank() - 1;

    // stash_type=1 means accumulate in float32 (standard practice)
    int64_t stashType = 1;

    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, input);

    auto axisAttr = rewriter.getI64IntegerAttr(axis);
    auto epsAttr = rewriter.getF32FloatAttr(static_cast<float>(epsilon));
    auto stashTypeAttr = rewriter.getI64IntegerAttr(stashType);

    auto hipOp = mlir::hip::RmsNormOp::create(rewriter, loc, resultType,
                                              context, input, weight, init,
                                              axisAttr, epsAttr, stashTypeAttr);

    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchNormConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<TorchRmsNormToHip>(ctx);
}

} // namespace hip
} // namespace mlir
