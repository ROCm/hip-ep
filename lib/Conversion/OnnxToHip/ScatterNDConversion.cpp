/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Trace the SSA chain from a value back through ops that don't change
/// element identity (Transpose, Reshape, Squeeze, Unsqueeze, Cast, etc.)
/// to find a hip.nonzero result. If found, return the NonZeroOp's count_buf
/// result (result[1]). Otherwise return nullptr.
///
/// Returns: {count_value, should_defer}
///   count_value: non-null if hip.NonZeroOp found upstream
///   should_defer: true if onnx.NonZero found (not yet converted) — caller
///                 should return failure() to let the greedy rewriter retry
///                 after NonZero conversion
static std::pair<mlir::Value, bool> traceToNonZeroCount(mlir::Value indices) {
  mlir::Value current = indices;
  for (int depth = 0; depth < 16; ++depth) {
    auto *defOp = current.getDefiningOp();
    if (!defOp)
      return {nullptr, false};

    // Found a hip.nonzero — its result[1] is the count_buf
    if (auto nonzeroOp = llvm::dyn_cast<mlir::hip::NonZeroOp>(defOp)) {
      if (nonzeroOp->getNumResults() >= 2)
        return {nonzeroOp->getResult(1), false};
      return {nullptr, false};
    }

    // Unconverted onnx.NonZero: signal caller to defer so the greedy rewriter
    // retries this op after NonZero conversion (which enables count_buf).
    if (defOp->getName().getStringRef() == "onnx.NonZero")
      return {nullptr, true};

    // ONNX ops that pass indices through without changing count
    llvm::StringRef opName = defOp->getName().getStringRef();
    if (opName == "onnx.Transpose" || opName == "onnx.Reshape" ||
        opName == "onnx.Squeeze" || opName == "onnx.Unsqueeze" ||
        opName == "onnx.Cast" || opName == "onnx.Gather") {
      if (defOp->getNumOperands() >= 1) {
        current = defOp->getOperand(0);
        continue;
      }
    }

    // HIP dialect transparent ops
    if (opName == "hip.transpose" || opName == "hip.cast") {
      if (defOp->getNumOperands() >= 2) {
        current = defOp->getOperand(1);
        continue;
      }
    }

    // tensor.cast or other tensor ops that don't change element identity
    if (opName == "tensor.cast" || opName == "tensor.collapse_shape" ||
        opName == "tensor.expand_shape") {
      if (defOp->getNumOperands() >= 1) {
        current = defOp->getOperand(0);
        continue;
      }
    }

    // Can't trace further
    return {nullptr, false};
  }
  return {nullptr, false};
}

/// onnx.ScatterND -> hip.scatter_nd
///
/// Output shape matches `data` exactly (per ONNX spec), so the destination
/// buffer can be allocated from the data shape directly.
///
/// The conversion also performs a backtrace on the `indices` operand: if
/// the indices originated from a NonZero op (possibly through Transpose),
/// the NonZero's count_buf is passed to hip.scatter_nd as `valid_count`.
/// This lets the kernel process only the valid (non-padded) rows.
struct ScatterNDToHip : public mlir::RewritePattern {
  ScatterNDToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ScatterND", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected 3 inputs (data, indices, updates), 1 output");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);
    mlir::Value updates = op->getOperand(2);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    if (!resultType || !dataType)
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor types for data and output");
    if (resultType.getRank() != dataType.getRank())
      return rewriter.notifyMatchFailure(
          op, "ScatterND requires result rank == data rank");

    // Output shape == data shape
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    mlir::StringAttr reductionAttr;
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("reduction"))
      reductionAttr = attr;
    else
      reductionAttr = rewriter.getStringAttr("none");

    // Backtrace: try to find a NonZero count_buf upstream of indices.
    // If an unconverted onnx.NonZero is found, defer this conversion so
    // the greedy rewriter retries after NonZero is converted.
    auto [validCount, shouldDefer] = traceToNonZeroCount(indices);
    if (shouldDefer)
      return rewriter.notifyMatchFailure(
          op, "deferring: upstream onnx.NonZero not yet converted");
    bool hasValidCount = (validCount != nullptr);

    if (!validCount) {
      // No NonZero upstream — create a dummy 1xi32 tensor (won't be read)
      auto countType = mlir::RankedTensorType::get({1}, rewriter.getI32Type());
      validCount = mlir::tensor::EmptyOp::create(
          rewriter, loc, countType.getShape(), countType.getElementType());
    }

    auto hipOp = mlir::hip::ScatterNDOp::create(
        rewriter, loc, resultType, context, data, indices, updates, validCount,
        init, reductionAttr, rewriter.getBoolAttr(hasValidCount));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateScatterNDConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ScatterNDToHip>(ctx);
}

} // namespace hip
} // namespace mlir
