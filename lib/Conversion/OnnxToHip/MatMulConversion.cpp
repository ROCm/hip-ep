//===- MatMulConversion.cpp - ONNX-to-HIP MatMul conversion --- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.hipblaslt.matmul
struct MatMulToHip : public RewritePattern {
  MatMulToHip(MLIRContext* ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult MatMulToHip::matchAndRewrite(Operation* op,
                                           PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value a = op->getOperand(0);
  Value b = op->getOperand(1);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<Value> dynSizes;
  const int64_t rank = resultType.getRank();
  const auto bType = cast<RankedTensorType>(b.getType());
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx == rank - 1) {
      dynSizes.push_back(
          tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
    } else {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, a, dimIdx));
    }
  }

  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType, context,
                                           a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateMatMulConversionPatterns(RewritePatternSet& patterns,
                                                 MLIRContext* ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
