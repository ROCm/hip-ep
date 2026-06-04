/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ClipConversion.cpp - onnx.Clip -> hip.max + hip.min ---------------===//
//
// ONNX Clip (opset 11+): takes the input plus two optional 0-rank scalar
// operands `min` and `max`. When either is absent the spec defaults it to
// `numeric_limits::lowest()` / `numeric_limits::max()` of the element type.
//
// Decompose:
//   clip(x, lo, hi) == min(max(x, lo), hi)
// We emit the HIP-dialect ops directly (hip.max then hip.min) rather than
// going through onnx.Max / onnx.Min, because the convert-onnx-to-hip driver
// runs with GreedyRewriteStrictness::ExistingOps — any onnx.* op we
// synthesize here would not be picked up by MaxToHip / MinToHip and would
// trip "op was not bufferized" downstream. Omitted bounds simply skip the
// corresponding step; if both are missing the Clip is pure identity.
//
//   Before:
//     %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<...xT>, tensor<T>, tensor<T>)
//                                          -> tensor<...xT>
//
//   After:
//     %i0 = tensor.empty(...) : tensor<...xT>
//     %t  = hip.max(%ctx) ins(%x, %lo : ..., tensor<T>) outs(%i0 : ...)
//     %i1 = tensor.empty(...) : tensor<...xT>
//     %y  = hip.min(%ctx) ins(%t, %hi : ..., tensor<T>) outs(%i1 : ...)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

/// `onnx.Clip` may carry an `onnx.NoValue` as a placeholder for an omitted
/// optional operand. Treat such operands as missing.
static bool isNoValue(mlir::Value v) {
  if (!v)
    return true;
  if (mlir::Operation *def = v.getDefiningOp())
    return def->getName().getStringRef() == "onnx.NoValue";
  return false;
}

struct ClipToHipMinMax : public mlir::RewritePattern {
  ClipToHipMinMax(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Clip", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "onnx.Clip expects 1 result");
    unsigned n = op->getNumOperands();
    if (n < 1 || n > 3)
      return rewriter.notifyMatchFailure(op, "onnx.Clip expects 1..3 operands");

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);

    mlir::Value loV = (n >= 2 && !isNoValue(op->getOperand(1)))
                          ? op->getOperand(1)
                          : nullptr;
    mlir::Value hiV = (n >= 3 && !isNoValue(op->getOperand(2)))
                          ? op->getOperand(2)
                          : nullptr;

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Both bounds missing: pure identity.
    if (!loV && !hiV) {
      rewriter.replaceOp(op, x);
      return mlir::success();
    }

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Value cur = x;
    if (loV) {
      mlir::Value init = createEmptyTensor(rewriter, loc, resultType, cur);
      cur = mlir::hip::MaxOp::create(rewriter, loc, resultType, context, cur,
                                     loV, init)
                ->getResult(0);
    }
    if (hiV) {
      mlir::Value init = createEmptyTensor(rewriter, loc, resultType, cur);
      cur = mlir::hip::MinOp::create(rewriter, loc, resultType, context, cur,
                                     hiV, init)
                ->getResult(0);
    }

    rewriter.replaceOp(op, cur);
    return mlir::success();
  }
};

} // namespace

void populateClipConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ClipToHipMinMax>(ctx);
}

} // namespace hip
} // namespace mlir
