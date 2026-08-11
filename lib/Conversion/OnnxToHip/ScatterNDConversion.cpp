/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

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
struct ScatterNDToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScatterNDToHip)
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

    auto hipOp = mlir::hip::ScatterNDOp::create(
        rewriter, loc, context, data, indices, updates, init, reductionAttr);
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
