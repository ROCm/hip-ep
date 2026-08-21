/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "gather-conversion"

STATISTIC(NumGatherAxis0ExtractSlice,
          "Number of axis-0 scalar-index onnx.Gather ops lowered to "
          "tensor.extract_slice");

namespace mlir {
namespace hip {
namespace {

static bool dimsCompatible(int64_t expected, int64_t actual) {
  if (ShapedType::isDynamic(expected) || ShapedType::isDynamic(actual))
    return true;
  return expected == actual;
}

static bool outputMatchesAxis0Gather(RankedTensorType dataTy,
                                     RankedTensorType indicesTy,
                                     RankedTensorType outputTy) {
  int64_t dataRank = dataTy.getRank();
  int64_t indicesRank = indicesTy.getRank();
  if (outputTy.getRank() != dataRank - 1 + indicesRank)
    return false;

  ArrayRef<int64_t> dataShape = dataTy.getShape();
  ArrayRef<int64_t> indicesShape = indicesTy.getShape();
  ArrayRef<int64_t> outShape = outputTy.getShape();

  int64_t outDim = 0;
  for (int64_t i = 0; i < indicesRank; ++i, ++outDim)
    if (!dimsCompatible(indicesShape[i], outShape[outDim]))
      return false;
  for (int64_t i = 1; i < dataRank; ++i, ++outDim)
    if (!dimsCompatible(dataShape[i], outShape[outDim]))
      return false;
  return true;
}

/// Axis-0 extract_slice requires a compile-time index so the offset folds to
/// a static `memref.subview` (zero-copy). Dynamic offsets would bufferize to a
/// host load from GPU-resident external constants and crash at inference.
static std::optional<OpFoldResult>
axisZeroStaticOffset(PatternRewriter &rewriter, Value indices) {
  if (auto idx = getCompileTimeScalarInt(indices))
    return OpFoldResult(rewriter.getIndexAttr(*idx));
  return std::nullopt;
}

/// Swin-style axis=0 Gather with a single scalar (or len-1) index lowers to
/// tensor.extract_slice when the index is known at compile time, not
/// hip.gather.
struct GatherAxis0ExtractSlice : public RewritePattern {
  GatherAxis0ExtractSlice(MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    int64_t axis = op->getAttrOfType<IntegerAttr>("axis").getSInt();
    auto dataTy = cast<RankedTensorType>(op->getOperand(0).getType());
    auto indicesTy = cast<RankedTensorType>(op->getOperand(1).getType());
    auto outputTy = cast<RankedTensorType>(op->getResult(0).getType());

    int64_t normalizedAxis = axis < 0 ? axis + dataTy.getRank() : axis;
    if (normalizedAxis != 0)
      return failure();

    if (indicesTy.getRank() > 1)
      return failure();
    if (indicesTy.getRank() == 1 && indicesTy.getDimSize(0) != 1 &&
        !indicesTy.isDynamicDim(0))
      return failure();

    if (!outputMatchesAxis0Gather(dataTy, indicesTy, outputTy))
      return failure();

    Location loc = op->getLoc();
    Value data = op->getOperand(0);
    Value indices = op->getOperand(1);

    std::optional<OpFoldResult> axisOffset =
        axisZeroStaticOffset(rewriter, indices);
    if (!axisOffset)
      return failure();

    int64_t rank = dataTy.getRank();
    SmallVector<OpFoldResult, 4> offsets(rank, rewriter.getIndexAttr(0));
    offsets[0] = *axisOffset;

    SmallVector<OpFoldResult, 4> sizes;
    SmallVector<OpFoldResult, 4> strides(rank, rewriter.getIndexAttr(1));
    for (int64_t i = 0; i < rank; ++i) {
      if (i == 0) {
        sizes.push_back(rewriter.getIndexAttr(1));
        continue;
      }
      if (dataTy.isDynamicDim(i)) {
        Value dimVal = tensor::DimOp::create(rewriter, loc, data, i);
        sizes.push_back(dimVal);
      } else
        sizes.push_back(rewriter.getIndexAttr(dataTy.getDimSize(i)));
    }

    Value slice = tensor::ExtractSliceOp::create(rewriter, loc, outputTy, data,
                                                 offsets, sizes, strides);

    ++NumGatherAxis0ExtractSlice;
    rewriter.replaceOp(op, slice);
    return success();
  }
};

struct GatherToHip : public RewritePattern {
  GatherToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    int64_t axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
    auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

    int64_t normalizedAxis = axis < 0 ? axis + dataType.getRank() : axis;

    llvm::SmallVector<mlir::Value> dynSizes;
    int64_t outDimIdx = 0;

    for (auto i : llvm::seq<int64_t>(0, normalizedAxis)) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }
    for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, indices, i));
      outDimIdx++;
    }
    for (auto i : llvm::seq<int64_t>(normalizedAxis + 1, dataType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto gatherOp = mlir::hip::GatherOp::create(rewriter, loc, context, data,
                                                indices, init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGatherConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<GatherAxis0ExtractSlice, GatherToHip>(ctx);
}

} // namespace hip
} // namespace mlir
