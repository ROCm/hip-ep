/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ClipConversion.cpp - onnx.Clip -> onnx.Max + onnx.Min -------------===//
//
// ONNX Clip (opset 11+) takes the input plus two optional 0-rank scalar
// operands `min` and `max`. When either is absent the spec defaults it to
// `numeric_limits::lowest()` / `numeric_limits::max()` of the element type.
//
// We decompose at compile time:
//   clip(x, lo, hi) == min(max(x, lo), hi)
// Omitted bounds collapse: a missing `lo` makes the outer `max` an identity
// (skip it); a missing `hi` makes the outer `min` an identity. The resulting
// `onnx.Max` / `onnx.Min` ops are picked up by their own conversion patterns
// (which delegate to MIOpen's miopenOpTensor and support broadcasting).
//
//   Before:
//     %y = "onnx.Clip"(%x, %lo, %hi) : (tensor<...xT>, tensor<T>, tensor<T>)
//                                         -> tensor<...xT>
//
//   After:
//     %t = "onnx.Max"(%x, %lo)       : (tensor<...xT>, tensor<T>)
//                                         -> tensor<...xT>
//     %y = "onnx.Min"(%t, %hi)       : (tensor<...xT>, tensor<T>)
//                                         -> tensor<...xT>
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

/// Build an onnx op with the given name and a single result of `resultType`.
static mlir::Value buildOnnxBinary(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::StringRef opName,
                                   mlir::Type resultType, mlir::Value lhs,
                                   mlir::Value rhs) {
  mlir::OperationState state(loc, opName);
  state.addTypes(resultType);
  state.addOperands({lhs, rhs});
  return rewriter.create(state)->getResult(0);
}

struct ClipToOnnxMinMax : public mlir::RewritePattern {
  ClipToOnnxMinMax(mlir::MLIRContext *ctx)
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

    mlir::Value loV =
        (n >= 2 && !isNoValue(op->getOperand(1))) ? op->getOperand(1) : nullptr;
    mlir::Value hiV =
        (n >= 3 && !isNoValue(op->getOperand(2))) ? op->getOperand(2) : nullptr;

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Both bounds missing: pure identity.
    if (!loV && !hiV) {
      rewriter.replaceOp(op, x);
      return mlir::success();
    }

    mlir::Value cur = x;
    if (loV)
      cur = buildOnnxBinary(rewriter, loc, "onnx.Max", resultType, cur, loV);
    if (hiV)
      cur = buildOnnxBinary(rewriter, loc, "onnx.Min", resultType, cur, hiV);

    rewriter.replaceOp(op, cur);
    return mlir::success();
  }
};

} // namespace

void populateClipConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ClipToOnnxMinMax>(ctx);
}

} // namespace hip
} // namespace mlir
