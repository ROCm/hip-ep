/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Transpose -> hip.transpose
struct TransposeToHip : public mlir::RewritePattern {
  TransposeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TransposeToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm");
  if (!permAttr)
    return op->emitOpError("hip.transpose requires explicit perm attribute");

  int64_t dim0 = -1, dim1 = -1;
  int64_t mismatchCount = 0;
  for (auto [permIdx, attr] : llvm::enumerate(permAttr)) {
    int64_t p = mlir::cast<mlir::IntegerAttr>(attr).getValue().getSExtValue();
    if (p != static_cast<int64_t>(permIdx)) {
      ++mismatchCount;
      if (dim0 < 0)
        dim0 = static_cast<int64_t>(permIdx);
      else if (dim1 < 0)
        dim1 = static_cast<int64_t>(permIdx);
    }
  }
  if (mismatchCount != 2 || dim0 < 0 || dim1 < 0)
    return op->emitOpError("perm must swap exactly two dimensions");
  int64_t p0 =
      mlir::cast<mlir::IntegerAttr>(permAttr[dim0]).getValue().getSExtValue();
  int64_t p1 =
      mlir::cast<mlir::IntegerAttr>(permAttr[dim1]).getValue().getSExtValue();
  if (p0 != dim1 || p1 != dim0)
    return op->emitOpError("perm must swap exactly two dimensions");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Transpose: output dim i corresponds to input dim perm[i].
  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto [outDimIdx, attr] : llvm::enumerate(permAttr)) {
    if (resultType.isDynamicDim(outDimIdx)) {
      const int64_t srcDim =
          mlir::cast<mlir::IntegerAttr>(attr).getValue().getSExtValue();
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, data, srcDim));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim0);
  mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim1);

  auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, resultType,
                                              context, d0, d1, data, init);
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
