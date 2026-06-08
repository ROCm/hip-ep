/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Backtrace the SSA chain from `indices` through ops that preserve element
/// identity (Transpose/Reshape/Squeeze/Unsqueeze/Cast/Gather and their HIP /
/// tensor equivalents) to find a `hip.nonzero`. If found, return its count_buf
/// (result[1]) — the runtime row count to bound ScatterND's writes.
///
/// Returns {count_value, should_defer}:
///   count_value : non-null if a converted `hip.nonzero` is found upstream.
///   should_defer: true if an UNCONVERTED `onnx.NonZero` is found — the caller
///                 returns failure() so the greedy rewriter retries this op
///                 after NonZero converts (and exposes its count_buf result).
static std::pair<mlir::Value, bool> traceToNonZeroCount(mlir::Value indices) {
  mlir::Value current = indices;
  // Bounded walk: the NonZero->ScatterND index chain in real exports is short
  // (Transpose + a reshape or two); 16 hops is generous slack.
  for (int depth = 0; depth < 16; ++depth) {
    auto *defOp = current.getDefiningOp();
    if (!defOp)
      return {nullptr, false};

    if (auto nonzeroOp = llvm::dyn_cast<mlir::hip::NonZeroOp>(defOp)) {
      if (nonzeroOp->getNumResults() >= 2)
        return {nonzeroOp->getResult(1), false};
      return {nullptr, false};
    }

    // Unconverted onnx.NonZero: defer so the rewriter retries post-conversion.
    if (defOp->getName().getStringRef() == "onnx.NonZero")
      return {nullptr, true};

    llvm::StringRef opName = defOp->getName().getStringRef();
    // ONNX shape/identity ops carry the index data through operand 0.
    if (opName == "onnx.Transpose" || opName == "onnx.Reshape" ||
        opName == "onnx.Squeeze" || opName == "onnx.Unsqueeze" ||
        opName == "onnx.Cast" || opName == "onnx.Gather") {
      if (defOp->getNumOperands() >= 1) {
        current = defOp->getOperand(0);
        continue;
      }
    }
    // HIP DPS equivalents take ctx as operand 0; the data input is operand 1.
    if (opName == "hip.transpose" || opName == "hip.cast") {
      if (defOp->getNumOperands() >= 2) {
        current = defOp->getOperand(1);
        continue;
      }
    }
    // tensor reshape/cast descriptor ops carry data through operand 0.
    if (opName == "tensor.cast" || opName == "tensor.collapse_shape" ||
        opName == "tensor.expand_shape") {
      if (defOp->getNumOperands() >= 1) {
        current = defOp->getOperand(0);
        continue;
      }
    }
    return {nullptr, false};
  }
  return {nullptr, false};
}

/// onnx.ScatterND -> hip.scatter_nd
///
/// Output shape matches `data` exactly (per ONNX spec), so the destination
/// buffer can be allocated from the data shape directly. We do NOT split the
/// op into `copy(data -> output)` + `scatter_inplace(output, indices,
/// updates)` at this level — the runtime takes the original `data` as an
/// input operand and is responsible for both the initial copy and the
/// per-index writes. This keeps the IR symmetric with `hip.gather_nd` and
/// avoids spawning extra `linalg.copy` ops that the buffer-pooling pass
/// would have to fuse back together.
///
/// Dynamic shape support: any dynamic dim of the output is sourced from
/// the corresponding `data` dim via `tensor.dim` at runtime. ScatterND's
/// `out.rank == data.rank` invariant means the mapping is identity.
///
/// `reduction` is forwarded as a string attribute (`"none"` by default).
/// The runtime stub today only logs its parameters, so any reduction mode
/// is accepted at compile time.
///
/// The conversion also backtraces `indices` (see traceToNonZeroCount): if it
/// originates from a NonZero (possibly through Transpose/Reshape/...), that
/// op's count_buf is passed as `valid_count` with `has_valid_count = true`, so
/// the kernel processes only the runtime-valid rows. Otherwise a dummy 1xi32
/// init is passed with `has_valid_count = false` (never read).
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
    // ScatterND's defining invariant: output and data have identical
    // shapes (and therefore identical ranks). Reject any mismatch loudly
    // rather than silently producing a malformed `tensor.empty`.
    if (resultType.getRank() != dataType.getRank())
      return rewriter.notifyMatchFailure(
          op, "ScatterND requires result rank == data rank");

    // Output shape == data shape; reuse data's dim values for any dynamic
    // dims of the destination empty.
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    // `reduction` is optional (default "none" per ONNX spec). Preserve the
    // exact string so the runtime side can switch on it.
    mlir::StringAttr reductionAttr;
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("reduction"))
      reductionAttr = attr;
    else
      reductionAttr = rewriter.getStringAttr("none");

    // Backtrace indices to an upstream NonZero count_buf. If an unconverted
    // onnx.NonZero is found, defer so the rewriter retries after it converts.
    auto [validCount, shouldDefer] = traceToNonZeroCount(indices);
    if (shouldDefer)
      return rewriter.notifyMatchFailure(
          op, "deferring: upstream onnx.NonZero not yet converted");
    bool hasValidCount = (validCount != nullptr);
    if (!validCount) {
      // No NonZero upstream — dummy 1xi32 init (has_valid_count=false; never
      // read by the kernel, present only to satisfy the operand list).
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
