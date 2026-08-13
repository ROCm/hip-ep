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

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<mlir::Value> dynSizes;
  const int64_t rank = resultType.getRank();
  const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx == rank - 1) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
    } else {
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);
  // Inferred-type Op::create overload: result type is read from the typed
  // outs operand by MatmulOp's inference family. This is equivalent to passing
  // `resultType` explicitly -- outs.getType() == resultType by construction
  // here -- but keeps `result_type == outs_operand_type` closed by ODS rather
  // than restated at the callsite.
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
