/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.mm -> hip.matmul (2D x 2D matrix multiply)
struct TorchMmToHip : public mlir::RewritePattern {
  TorchMmToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.mm", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Result[M, N] = A[M, K] @ B[K, N]
    // M from A dim 0, N from B dim 1
    const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
      if (!resultType.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == resultType.getRank() - 1) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
      } else {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
      }
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);
    auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType, context,
                                             a, b, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.bmm -> hip.matmul (3D batched matrix multiply)
struct TorchBmmToHip : public mlir::RewritePattern {
  TorchBmmToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.bmm", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Result[B, M, N] = A[B, M, K] @ B[B, K, N]
    const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
    const int64_t rank = resultType.getRank();
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == rank - 1) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
      } else {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
      }
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);
    auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType, context,
                                             a, b, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.matmul -> hip.matmul (general N-D matmul)
struct TorchMatmulToHip : public mlir::RewritePattern {
  TorchMatmulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.matmul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N]
    // Batch and M dims come from A; N comes from B's last dim.
    const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
    const int64_t rank = resultType.getRank();
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == rank - 1) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
      } else {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
      }
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);
    auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType, context,
                                             a, b, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// torch.aten.linear -> hip.matmul + optional hip.add for bias
/// PyTorch linear: output = input @ weight^T + bias
/// Weight is transposed in PyTorch convention; the runtime handles the
/// transpose so we pass operands directly to hip.matmul.
struct TorchLinearToHip : public mlir::RewritePattern {
  TorchLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.linear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value weight = op->getOperand(1);
    mlir::Value bias = op->getOperand(2);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Create init tensor for matmul result
    // Result shape: [..., M, N] where M from input, N from weight dim 0
    // (since weight is [N, K] in PyTorch linear convention)
    const auto weightType =
        mlir::cast<mlir::RankedTensorType>(weight.getType());
    const int64_t rank = resultType.getRank();
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == rank - 1) {
        // N dimension comes from weight dim 0
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, weight, 0));
      } else {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx));
      }
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    // Create hip.matmul: the runtime handles the W^T transpose
    auto matmulOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType,
                                                context, input, weight, init);
    mlir::Value result = matmulOp->getResult(0);

    // Add bias if present (not torch.constant.none)
    if (!isTorchNone(bias)) {
      mlir::Value addInit =
          createEmptyTensorForTorch(rewriter, loc, resultType, result);
      auto addOp = mlir::hip::AddOp::create(rewriter, loc, resultType, context,
                                            result, bias, addInit);
      result = addOp->getResult(0);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateTorchMatMulConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx) {
  patterns.add<TorchMmToHip, TorchBmmToHip, TorchMatmulToHip, TorchLinearToHip>(
      ctx);
}

} // namespace hip
} // namespace mlir
