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

/// Build a tensor.empty for the result of a hip.transpose whose target shape
/// is derived from the source by swapping `dim0` and `dim1`.  Dynamic dims
/// of the result are filled in from `srcDimSizes`, which holds tensor.dim
/// values for the original input.
static mlir::Value buildSwapInit(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Value data,
                                 mlir::RankedTensorType targetType,
                                 llvm::ArrayRef<mlir::Value> srcDimSizes,
                                 llvm::ArrayRef<int64_t> srcDimMap) {
  // srcDimMap[i] == j means: target dim i comes from source dim j.
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < targetType.getRank(); ++i) {
    if (targetType.isDynamicDim(i)) {
      mlir::Value dimVal = srcDimSizes[srcDimMap[i]];
      dynSizes.push_back(dimVal);
    }
  }
  return mlir::tensor::EmptyOp::create(rewriter, loc, targetType.getShape(),
                                       targetType.getElementType(), dynSizes);
}

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
    return rewriter.notifyMatchFailure(
        op, "hip.transpose requires explicit perm attribute");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::cast<mlir::RankedTensorType>(data.getType());
  int64_t rank = inputType.getRank();
  // Rank-0 transpose: forward the input unchanged regardless of the perm
  // size.  Kokoro's iSTFTNet noise path emits a rank-0 onnx.Transpose with
  // perm=[0,2,1,3] (dead code, but the conversion has to accept it).
  if (rank == 0 && resultType.getRank() == 0) {
    rewriter.replaceOp(op, data);
    return mlir::success();
  }
  if (rank != static_cast<int64_t>(permAttr.size()) ||
      rank != resultType.getRank())
    return rewriter.notifyMatchFailure(op, "perm size != tensor rank");

  // Materialize the original input dim sizes once; we pull from this when
  // building intermediate empties.
  llvm::SmallVector<mlir::Value> srcDimSizes(rank);
  for (int64_t i = 0; i < rank; ++i)
    srcDimSizes[i] = mlir::tensor::DimOp::create(rewriter, loc, data, i);

  // Compute final perm[].
  llvm::SmallVector<int64_t> perm(rank);
  for (int64_t i = 0; i < rank; ++i)
    perm[i] = mlir::cast<mlir::IntegerAttr>(permAttr[i]).getValue().getSExtValue();

  // Cycle-decomposition of `perm` into a sequence of 2-element swaps.
  // Algorithm: starting from the identity, for each i find the position j
  // that currently holds perm[i] and swap (i, j) in the running state.
  //
  // Each (i, j) is appended as a hip.transpose dim swap to the IR.  Final
  // hip.transpose's result type matches resultType; intermediate result
  // types are derived by mutating the running shape.
  llvm::SmallVector<int64_t> state(rank);
  for (int64_t i = 0; i < rank; ++i)
    state[i] = i;

  llvm::SmallVector<std::pair<int64_t, int64_t>> swaps;
  for (int64_t i = 0; i < rank; ++i) {
    if (state[i] == perm[i])
      continue;
    int64_t j = -1;
    for (int64_t k = i + 1; k < rank; ++k) {
      if (state[k] == perm[i]) {
        j = k;
        break;
      }
    }
    if (j < 0) // perm is not a permutation -- malformed onnx.Transpose
      return rewriter.notifyMatchFailure(op, "perm is not a valid permutation");
    std::swap(state[i], state[j]);
    swaps.push_back({i, j});
  }

  // No-op transpose: just forward the input.
  if (swaps.empty()) {
    rewriter.replaceOp(op, data);
    return mlir::success();
  }

  // Apply each swap as one hip.transpose.  After applying all of them, the
  // tensor lives in `resultType`.  For each intermediate step, we build a
  // tensor.empty with the partially-permuted shape.
  mlir::Value cur = data;

  // Track the dim-index mapping: targetType dim i comes from original
  // input dim `currentMap[i]`.  Starts as identity.
  llvm::SmallVector<int64_t> currentMap(rank);
  for (int64_t i = 0; i < rank; ++i)
    currentMap[i] = i;

  for (size_t step = 0; step < swaps.size(); ++step) {
    auto [a, b] = swaps[step];
    // After this swap: target[a] = currentMap[b], target[b] = currentMap[a].
    llvm::SmallVector<int64_t> nextMap = currentMap;
    std::swap(nextMap[a], nextMap[b]);

    bool isLast = (step + 1 == swaps.size());
    mlir::RankedTensorType targetType;
    if (isLast) {
      targetType = resultType;
    } else {
      llvm::SmallVector<int64_t> shape(rank);
      for (int64_t i = 0; i < rank; ++i)
        shape[i] = inputType.getDimSize(nextMap[i]);
      targetType = mlir::RankedTensorType::get(shape, inputType.getElementType());
    }

    mlir::Value init =
        buildSwapInit(rewriter, loc, data, targetType, srcDimSizes, nextMap);
    mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, a);
    mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, b);
    auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, targetType,
                                                context, d0, d1, cur, init);
    cur = hipOp->getResult(0);
    currentMap = nextMap;
  }

  rewriter.replaceOp(op, cur);
  return mlir::success();
}

} // namespace

void mlir::hip::populateTransposeConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.add<TransposeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
