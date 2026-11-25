/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- BasicConversion.cpp - Basic direct GGML-to-HIP patterns -----------===//

#include "GgmlToHipUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

namespace mlir {
namespace hip {
namespace {

template <typename SourceOp, typename TargetOp>
struct BinaryElementwiseToHip final : OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    RankedTensorType resultType = op.getResult().getType();

    // GGML defines ADD and MUL as an elementwise operation over the lhs
    // shape.  The rhs may repeat whenever each of its dimensions divides
    // the corresponding lhs dimension.  hip.add and hip.mul instead use
    // NumPy broadcasting, where a differing rhs dimension must be one.
    // Accept only the intersection of those contracts.  In particular,
    // do not silently translate a valid GGML 3 -> 6 repeat into an
    // incompatible HIP broadcast.
    if (resultType.getShape() != lhsType.getShape()) {
      return op.emitOpError(
                 "cannot lower to HIP NumPy broadcasting: result shape ")
             << resultType << " must match the GGML lhs shape " << lhsType;
    }
    for (int64_t dimension = 0; dimension < lhsType.getRank(); ++dimension) {
      if (lhsType.isDynamicDim(dimension) &&
          !haveEquivalentGgmlDimensionExtent(op.getLhs(), dimension,
                                             op.getResult(), dimension)) {
        return op.emitOpError(
                   "cannot lower to HIP NumPy broadcasting: lhs/result "
                   "dimension ")
               << dimension
               << " must be statically equal or carry identical GGML shape "
                  "provenance";
      }
    }
    if (rhsType.getRank() > lhsType.getRank()) {
      return op.emitOpError("cannot lower to HIP NumPy broadcasting: rhs rank ")
             << rhsType.getRank() << " exceeds lhs rank " << lhsType.getRank();
    }

    const int64_t rankOffset = lhsType.getRank() - rhsType.getRank();
    for (int64_t rhsDim = 0; rhsDim < rhsType.getRank(); ++rhsDim) {
      const int64_t lhsDim = rhsDim + rankOffset;
      const int64_t lhsExtent = lhsType.getDimSize(lhsDim);
      const int64_t rhsExtent = rhsType.getDimSize(rhsDim);
      const bool singletonRhs = rhsExtent == 1;
      if (!singletonRhs && !haveEquivalentGgmlDimensionExtent(
                               op.getLhs(), lhsDim, op.getRhs(), rhsDim)) {
        InFlightDiagnostic diagnostic =
            op.emitOpError("cannot lower GGML repeat semantics to HIP NumPy "
                           "broadcasting at aligned "
                           "dimension ");
        diagnostic << lhsDim << ": lhs extent is ";
        if (ShapedType::isDynamic(lhsExtent)) {
          diagnostic << "?";
        } else {
          diagnostic << lhsExtent;
        }
        diagnostic << " and rhs extent is ";
        if (ShapedType::isDynamic(rhsExtent)) {
          diagnostic << "?";
        } else {
          diagnostic << rhsExtent;
        }
        diagnostic << "; the rhs extent must be 1 or have matching static or "
                      "symbolic GGML shape provenance with the lhs";
        return failure();
      }
    }

    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    FailureOr<Value> init = createGgmlBroadcastEmptyTensor(
        rewriter, op.getLoc(), resultType, ValueRange{lhs, rhs});
    if (failed(init)) {
      return rewriter.notifyMatchFailure(
          op, "no ranked operand supplies a dynamic result dimension");
    }

    auto converted =
        TargetOp::create(rewriter, op.getLoc(), *context, lhs, rhs, *init);
    rewriter.replaceOp(op, converted.getResultTensors());
    return success();
  }
};

using AddToHip = BinaryElementwiseToHip<ggml::AddOp, hip::AddOp>;
using MulToHip = BinaryElementwiseToHip<ggml::MulOp, hip::MulOp>;

/// Validate and build the canonical GGML GET_ROWS output shape:
///
///   data    [batch_1, batch_0, rows, width]
///   indices [1,       batch_1, batch_0, selected_rows]
///   result  [batch_1, batch_0, selected_rows, width]
static FailureOr<SmallVector<Value>>
getGetRowsDynamicSizes(ggml::GetRowsOp op, PatternRewriter &rewriter,
                       RankedTensorType dataType, RankedTensorType indicesType,
                       RankedTensorType resultType) {
  if (dataType.getRank() != 4 || indicesType.getRank() != 4 ||
      resultType.getRank() != 4) {
    op.emitOpError("requires rank-4 data, indices, and result tensors");
    return failure();
  }

  if (indicesType.isDynamicDim(0) || indicesType.getDimSize(0) != 1) {
    op.emitOpError(
        "requires indices dimension 0 to be statically 1 in the raw GGML "
        "rank-4 carrier");
    return failure();
  }

  if (!haveEquivalentGgmlDimensionExtent(op.getData(), 0, op.getIndices(), 1) ||
      !haveEquivalentGgmlDimensionExtent(op.getData(), 0, op.getResult(), 0) ||
      !haveEquivalentGgmlDimensionExtent(op.getData(), 1, op.getIndices(), 2) ||
      !haveEquivalentGgmlDimensionExtent(op.getData(), 1, op.getResult(), 1) ||
      !haveEquivalentGgmlDimensionExtent(op.getIndices(), 3, op.getResult(),
                                         2) ||
      !haveEquivalentGgmlDimensionExtent(op.getData(), 3, op.getResult(), 3)) {
    op.emitOpError(
        "requires data [B1, B0, rows, width], indices [1, B1, B0, "
        "selected_rows], and result [B1, B0, selected_rows, width] with "
        "matching static or symbolic GGML shape provenance");
    return failure();
  }

  SmallVector<Value> dynamicSizes;
  for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension) {
    if (!resultType.isDynamicDim(dimension)) {
      continue;
    }
    Value source = dimension == 2 ? op.getIndices() : op.getData();
    const int64_t sourceDimension = dimension == 2 ? 3 : dimension;
    dynamicSizes.push_back(
        tensor::DimOp::create(rewriter, op.getLoc(), source, sourceDimension));
  }
  return dynamicSizes;
}

struct GetRowsToHip final : OpRewritePattern<ggml::GetRowsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ggml::GetRowsOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> context = getHipContext(op, rewriter);
    if (failed(context)) {
      return failure();
    }

    RankedTensorType dataType = op.getData().getType();
    RankedTensorType indicesType = op.getIndices().getType();
    RankedTensorType resultType = op.getResult().getType();
    if (!indicesType.getElementType().isInteger(32)) {
      return op.emitOpError(
          "requires i32 row indices, matching GGML semantics");
    }
    if (!dataType.getElementType().isF32() ||
        !resultType.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(op, "supports only f32 rows");
    }

    FailureOr<SmallVector<Value>> dynamicSizes =
        getGetRowsDynamicSizes(op, rewriter, dataType, indicesType, resultType);
    if (failed(dynamicSizes)) {
      return failure();
    }
    if (dataType.getDimSize(0) != 1 || dataType.getDimSize(1) != 1 ||
        indicesType.getDimSize(1) != 1 || indicesType.getDimSize(2) != 1) {
      return rewriter.notifyMatchFailure(
          op, "supports only one row table and one index vector");
    }

    // hip.gather applies one flattened index vector to each outer data slice,
    // which is GGML GET_ROWS when both GGML batch dimensions are singleton.
    auto flattenedIndicesType = RankedTensorType::get(
        {indicesType.getDimSize(3)}, indicesType.getElementType());
    Value flattenedIndices = tensor::CollapseShapeOp::create(
        rewriter, op.getLoc(), flattenedIndicesType, op.getIndices(),
        SmallVector<ReassociationIndices>{{0, 1, 2, 3}});
    Value gatherInit =
        tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(),
                                resultType.getElementType(), *dynamicSizes);
    auto gathered = hip::GatherOp::create(
        rewriter, op.getLoc(), *context, op.getData(), flattenedIndices,
        gatherInit, rewriter.getI64IntegerAttr(/*axis=*/2));
    rewriter.replaceOp(op, gathered.getResultTensors());
    return success();
  }
};

} // namespace

void populateGgmlBasicConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddToHip, MulToHip, GetRowsToHip>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
