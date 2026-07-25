/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Create a tensor.empty for a DPS init operand whose shape is the result of
/// ONNX-style multidirectional (NumPy) broadcasting over multiple inputs.
///
/// Operand shapes are right-aligned with the result. For each dynamic
/// dimension of \p resultType, the size is taken from the first operand that
/// truly contributes to the broadcast extent at that axis -- i.e., an operand
/// whose corresponding dim is not statically 1. Operands whose rank does not
/// span the dimension (shorter rank, conceptually padded with 1 on the left)
/// are skipped, as are operands whose dim is statically 1. If every operand
/// is statically 1 at the axis (degenerate case for a dynamic result), we
/// fall back to the first operand spanning the dim.
///
/// Returns failure if no ranked operand spans a dynamic result dim (e.g.
/// every operand is unranked while the result is ranked-and-dynamic). We
/// surface this through `FailureOr` rather than `assert` so that the check
/// remains active in Release builds (assertions are stripped under NDEBUG)
/// and the pattern fails cleanly via `notifyMatchFailure` instead of
/// dereferencing a null Value at the next builder call.
mlir::FailureOr<mlir::Value>
createBroadcastEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::ValueRange operands) {
  int64_t resultRank = resultType.getRank();
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultRank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;

    mlir::Value chosen;
    int64_t chosenDim = -1;
    mlir::Value fallback;
    int64_t fallbackDim = -1;
    for (mlir::Value operand : operands) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (!t)
        continue;
      int64_t offset = resultRank - t.getRank();
      if (dimIdx < offset)
        continue; // operand is broadcast (padded to 1) at this axis
      int64_t operandDim = dimIdx - offset;
      if (!fallback) {
        fallback = operand;
        fallbackDim = operandDim;
      }
      if (!t.isDynamicDim(operandDim) && t.getDimSize(operandDim) == 1)
        continue; // statically broadcast, does not define the extent
      chosen = operand;
      chosenDim = operandDim;
      break;
    }
    if (!chosen) {
      chosen = fallback;
      chosenDim = fallbackDim;
    }
    if (!chosen)
      return mlir::failure();
    dynSizes.push_back(
        mlir::tensor::DimOp::create(builder, loc, chosen, chosenDim));
  }
  return mlir::Value(
      mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes));
}

/// onnx.Where -> hip.where
/// ONNX Where: output[i] = condition[i] ? X[i] : Y[i].
/// Supports multidirectional (NumPy-style) broadcasting between condition,
/// X and Y. The condition tensor is bool (i1); X and Y share the result
/// element type.
struct WhereToHip : public mlir::RewritePattern {
  WhereToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Where", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
WhereToHip::matchAndRewrite(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "onnx.Where expects 3 operands and 1 result");

  mlir::Location loc = op->getLoc();
  mlir::Value condition = op->getOperand(0);
  mlir::Value x = op->getOperand(1);
  mlir::Value y = op->getOperand(2);

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(
        op, "onnx.Where lowering expects a ranked tensor result");

  // ONNX Where supports multidirectional (NumPy-style) broadcasting, so any
  // given output dim may be contributed by a different operand. Resolve each
  // dynamic result dim by scanning all three operands rather than relying on
  // a single "source" tensor.
  mlir::FailureOr<mlir::Value> initOrFailure =
      createBroadcastEmptyTensor(rewriter, loc, resultType, {condition, x, y});
  if (mlir::failed(initOrFailure))
    return rewriter.notifyMatchFailure(
        op, "onnx.Where: no ranked operand spans dynamic result dim");
  auto hipOp = mlir::hip::WhereOp::create(rewriter, loc, context, condition, x,
                                          y, *initOrFailure);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateWhereConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<WhereToHip>(ctx);
}

} // namespace hip
} // namespace mlir
