/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.hipblaslt.matmul
struct MatMulToHip : public mlir::RewritePattern {
  MatMulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  int64_t transA = 0;
  int64_t transB = 0;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transA"))
    transA = attr.getSInt();
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transB"))
    transB = attr.getSInt();

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<mlir::Value> dynSizes;
  const int64_t rank = resultType.getRank();
  const auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
  const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
  const int64_t aRank = aType.getRank();
  const int64_t bRank = bType.getRank();
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx == rank - 1) {
      int64_t bDim = transB ? bRank - 2 : bRank - 1;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, b, bDim));
    } else if (dimIdx == rank - 2) {
      int64_t aDim = transA ? aRank - 1 : aRank - 2;
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, aDim));
    } else {
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(transA)));
  attrs.push_back(
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)));
  llvm::SmallVector<mlir::Value> operands = {context, a, b, init};
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
