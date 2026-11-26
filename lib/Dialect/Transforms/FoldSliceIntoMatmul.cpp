/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipConstantData.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/STLExtras.h>

#include <cstring>
#include <optional>

namespace mlir::hip {
#define GEN_PASS_DEF_FOLDSLICEINTOMATMULPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// A slice of the last (N) dimension of a matmul or gemm result only needs the
/// matching output columns of the weight, so the weight (and a gemm bias) can
/// be pre-sliced at compile time and the full-width product never computed.
///
/// Before:
///   %y = hip.matmul(%ctx) ins(%x, %w : tensor<8x64xf32>, tensor<64x96xf32>)
///            outs(%e : tensor<8x96xf32>) : tensor<8x96xf32>
///   %s = tensor.extract_slice %y[0, 32] [8, 32] [1, 1]
///            : tensor<8x96xf32> to tensor<8x32xf32>
/// After:
///   %w2 = hip.constant {value = dense<w[:, 32:64]>} : tensor<64x32xf32>
///   %y2 = hip.matmul(%ctx) ins(%x, %w2 : tensor<8x64xf32>, tensor<64x32xf32>)
///             outs(%e2 : tensor<8x32xf32>) : tensor<8x32xf32>
///
/// The matmul result must have the slice as its only consumer, otherwise the
/// full product is still needed (e.g. a fused QKV projection split into three
/// slices is left alone).
struct FoldSliceIntoMatmul : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override;
};

/// The slice of the last dimension: `[start, start + size * stride)`.
struct LastDimSlice {
  int64_t start;
  int64_t size;
  int64_t stride;
};

std::optional<LastDimSlice> matchLastDimSlice(tensor::ExtractSliceOp op) {
  auto srcType = dyn_cast<RankedTensorType>(op.getSourceType());
  auto resType = dyn_cast<RankedTensorType>(op.getType());
  if (!srcType || !resType || !srcType.hasStaticShape() ||
      !resType.hasStaticShape() || srcType.getRank() != resType.getRank() ||
      srcType.getRank() < 2 || !op.getSource().hasOneUse())
    return std::nullopt;
  const int64_t last = srcType.getRank() - 1;
  for (int64_t d : llvm::seq<int64_t>(last)) {
    if (op.getStaticOffsets()[d] != 0 || op.getStaticStrides()[d] != 1 ||
        op.getStaticSizes()[d] != srcType.getDimSize(d))
      return std::nullopt;
  }
  LastDimSlice slice{op.getStaticOffsets()[last], op.getStaticSizes()[last],
                     op.getStaticStrides()[last]};
  if (ShapedType::isDynamic(slice.start) || ShapedType::isDynamic(slice.size) ||
      ShapedType::isDynamic(slice.stride) || slice.stride < 1)
    return std::nullopt;
  return slice;
}

int64_t product(ArrayRef<int64_t> dims) {
  int64_t n = 1;
  for (int64_t d : dims)
    n *= d;
  return n;
}

/// `constant` with `dim` restricted to the given slice, as a new inline
/// constant. Fails when the payload is unreadable or has sub-byte elements.
FailureOr<Value> sliceConstant(PatternRewriter &rewriter, Value value,
                               int64_t dim, const LastDimSlice &slice) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!constant || !type || !type.hasStaticShape() || dim < 0 ||
      dim >= type.getRank())
    return failure();
  Type elementType = type.getElementType();
  if (!elementType.isIntOrFloat() ||
      elementType.getIntOrFloatBitWidth() % 8 != 0)
    return failure();
  FailureOr<DenseElementsAttr> data = readConstantData(constant, type);
  if (failed(data))
    return failure();

  SmallVector<int64_t> newShape(type.getShape());
  newShape[dim] = slice.size;
  auto newType = RankedTensorType::get(newShape, elementType);

  DenseElementsAttr newAttr;
  if (data->isSplat()) {
    newAttr = data->resizeSplat(newType);
  } else {
    const int64_t dimSize = type.getDimSize(dim);
    const int64_t outer = product(type.getShape().take_front(dim));
    const int64_t inner = product(type.getShape().drop_front(dim + 1));
    const int64_t chunk = inner * (elementType.getIntOrFloatBitWidth() / 8);
    ArrayRef<char> raw = data->getRawData();
    SmallVector<char> out(outer * slice.size * chunk);
    char *dst = out.data();
    for (int64_t o : llvm::seq<int64_t>(outer)) {
      for (int64_t j : llvm::seq<int64_t>(slice.size)) {
        const int64_t src = (o * dimSize + slice.start + j * slice.stride);
        std::memcpy(dst, raw.data() + src * chunk, chunk);
        dst += chunk;
      }
    }
    newAttr = DenseElementsAttr::getFromRawBuffer(newType, out);
  }
  return ConstantOp::create(rewriter, constant.getLoc(), newType, newAttr)
      .getResult();
}

/// The tensor dimension of the weight `b` (rank `rank`) that is the output's N.
int64_t weightNDim(int64_t rank, bool transposed) {
  return transposed ? rank - 2 : rank - 1;
}

LogicalResult
FoldSliceIntoMatmul::matchAndRewrite(tensor::ExtractSliceOp op,
                                     PatternRewriter &rewriter) const {
  std::optional<LastDimSlice> slice = matchLastDimSlice(op);
  if (!slice)
    return rewriter.notifyMatchFailure(op, "not a static last-dim slice");

  Operation *producer = op.getSource().getDefiningOp();
  if (!producer || producer->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "source is not a tensor-mode op");
  auto matmul = dyn_cast<MatmulOp>(producer);
  auto gemm = dyn_cast<GemmOp>(producer);
  if (!matmul && !gemm)
    return rewriter.notifyMatchFailure(op, "source is not matmul or gemm");

  auto resultType = cast<RankedTensorType>(op.getType());
  Value weights = matmul ? matmul.getB() : gemm.getInputB();
  const bool transB = matmul ? matmul.getTransB() != 0 : gemm.getTransB() != 0;
  auto weightType = dyn_cast<RankedTensorType>(weights.getType());
  if (!weightType || !weightType.hasStaticShape() || weightType.getRank() < 2)
    return rewriter.notifyMatchFailure(op, "weights are not static");
  const int64_t srcN =
      cast<RankedTensorType>(op.getSourceType()).getShape().back();
  const int64_t weightDim = weightNDim(weightType.getRank(), transB);
  if (weightType.getDimSize(weightDim) != srcN)
    return rewriter.notifyMatchFailure(op, "weights do not span the N dim");

  rewriter.setInsertionPoint(producer);
  FailureOr<Value> newWeights =
      sliceConstant(rewriter, weights, weightDim, *slice);
  if (failed(newWeights))
    return rewriter.notifyMatchFailure(op, "weights are not readable");

  Value newBias;
  if (gemm && gemm.getInputC()) {
    Value bias = gemm.getInputC();
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || !biasType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "bias is not static");
    // A bias that is broadcast along N is unchanged by the slice.
    if (biasType.getRank() > 0 && biasType.getShape().back() != 1) {
      if (biasType.getShape().back() != srcN)
        return rewriter.notifyMatchFailure(op, "bias does not span N");
      FailureOr<Value> sliced =
          sliceConstant(rewriter, bias, biasType.getRank() - 1, *slice);
      if (failed(sliced))
        return rewriter.notifyMatchFailure(op, "bias is not readable");
      newBias = *sliced;
    } else {
      newBias = bias;
    }
  }

  Location loc = producer->getLoc();
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());
  SmallVector<Value> operands;
  if (matmul) {
    operands = {matmul.getCtx(), matmul.getA(), *newWeights, init};
  } else {
    operands = {gemm.getCtx(), gemm.getInputA(), *newWeights};
    if (newBias)
      operands.push_back(newBias);
    operands.push_back(init);
  }
  Operation *folded = rewriter.create(
      loc, producer->getName().getIdentifier(), operands,
      TypeRange{resultType}, producer->getAttrs());
  rewriter.replaceOp(op, folded->getResult(0));
  rewriter.eraseOp(producer);
  return success();
}

struct FoldSliceIntoMatmulPass
    : public impl::FoldSliceIntoMatmulPassBase<FoldSliceIntoMatmulPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldSliceIntoMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
