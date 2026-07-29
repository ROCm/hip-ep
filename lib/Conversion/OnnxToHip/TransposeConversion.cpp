/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Transpose -> hip.transpose
///
/// Supports arbitrary perm permutations (full ONNX Transpose semantics).  When
/// perm is omitted on the ONNX op, the default reverse permutation
/// [rank-1, ..., 0] is materialized so hip.transpose always carries an
/// explicit perm attribute.
struct TransposeToHip : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeToHip)
  TransposeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TransposeToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected single-input single-output");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor input");

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

  const int64_t rank = inputType.getRank();
  if (resultType.getRank() != rank)
    return op->emitOpError("result rank must match input rank");

  // Materialize the perm vector. Default per ONNX spec is the reverse
  // permutation when the attribute is absent.
  llvm::SmallVector<int64_t> perm;
  perm.reserve(rank);
  if (auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
    if (static_cast<int64_t>(permAttr.size()) != rank)
      return op->emitOpError("perm length must match input rank");
    for (mlir::Attribute attr : permAttr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
      if (!intAttr)
        return op->emitOpError("perm entries must be integers");
      perm.push_back(intAttr.getValue().getSExtValue());
    }
  } else {
    for (int64_t i = rank - 1; i >= 0; --i)
      perm.push_back(i);
  }

  // Validate that perm is a permutation of [0, rank).
  llvm::SmallVector<bool> seen(rank, false);
  for (int64_t p : perm) {
    if (p < 0 || p >= rank || seen[p])
      return op->emitOpError("perm must be a permutation of [0, rank)");
    seen[p] = true;
  }

  // Build dynamic sizes for the DPS init tensor: output dim i = input dim
  // perm[i].
  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto [outDimIdx, srcDim] : llvm::enumerate(perm)) {
    if (resultType.isDynamicDim(outDimIdx))
      dynSizes.push_back(mlir::tensor::DimOp::create(
          rewriter, loc, data, static_cast<int64_t>(srcDim)));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  mlir::ArrayAttr permArrayAttr = rewriter.getI64ArrayAttr(perm);
  auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, context, data,
                                              init, permArrayAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateTransposeConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<TransposeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
