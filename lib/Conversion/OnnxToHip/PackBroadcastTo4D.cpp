/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PackBroadcastTo4D.cpp - Pre-lower binary broadcasts to 4-D ---------===//
//
// Pre-lowering pattern set inside convert-onnx-to-hip, alongside
// GatherShapeFold / ReshapeShapeFold / PadShapeFold / SliceShapeFold. It
// rewrites static high-rank onnx.Add/Sub/Mul/Div/Less/Greater operations into
// collapse_shape -> rank-<=4 ONNX op -> expand_shape before compute conversion
// creates the corresponding HIP op.
//
// Before:
//   %r = "onnx.Add"(%a, %b) : (...) -> tensor<6x2500x8x1x2x4x2xf32>
//
// After, using reassociation [[0], [1], [2,3,4], [5,6]]:
//   %pa = tensor.collapse_shape %a    ... into tensor<6x2500x1x8xf32>
//   %pb = tensor.collapse_shape %b    ... into tensor<6x2500x16x8xf32>
//   %pr = "onnx.Add"(%pa, %pb) : (...) -> tensor<6x2500x16x8xf32>
//   %r  = tensor.expand_shape %pr ... into tensor<6x2500x8x1x2x4x2xf32>
//
// A contiguous group is safe only when each operand is either fully broadcast
// in the group (product 1) or fully present (product equals the output group
// product). Unsafe and dynamic cases are deliberately left unchanged for the
// HIP-to-LLVM backend to diagnose or support.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#include <functional>
#include <limits>

#define DEBUG_TYPE "onnx-pack-broadcast-to-4d"

STATISTIC(NumBroadcastsPacked,
          "Number of high-rank ONNX binary broadcasts packed to rank <= 4");

namespace mlir {
namespace hip {
namespace {

struct PackedBroadcast {
  Value lhs;
  Value rhs;
  RankedTensorType packedResultType;
  RankedTensorType originalResultType;
  SmallVector<ReassociationIndices> reassociation;
};

static bool multiplyDim(int64_t &product, int64_t dim) {
  if (dim < 0)
    return false;
  if (dim != 0 && product > std::numeric_limits<int64_t>::max() / dim)
    return false;
  product *= dim;
  return true;
}

static FailureOr<Value> rightAlignTensor(OpBuilder &builder, Location loc,
                                         Value value, int64_t targetRank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() == 0 ||
      type.getRank() > targetRank)
    return failure();
  if (type.getRank() == targetRank)
    return value;

  int64_t leadingOnes = targetRank - type.getRank();
  SmallVector<int64_t> alignedShape(leadingOnes, 1);
  llvm::append_range(alignedShape, type.getShape());
  auto alignedType = RankedTensorType::get(alignedShape, type.getElementType(),
                                           type.getEncoding());

  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices firstGroup;
  for (int64_t i = 0; i <= leadingOnes; ++i)
    firstGroup.push_back(i);
  reassociation.push_back(std::move(firstGroup));
  for (int64_t i = 1; i < type.getRank(); ++i)
    reassociation.push_back({leadingOnes + i});

  SmallVector<OpFoldResult> outputShape;
  for (int64_t dim : alignedShape)
    outputShape.push_back(builder.getIndexAttr(dim));
  return Value(tensor::ExpandShapeOp::create(builder, loc, alignedType, value,
                                             reassociation, outputShape));
}

static FailureOr<Value>
collapseTensor(OpBuilder &builder, Location loc, Value value,
               ArrayRef<ReassociationIndices> reassociation) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> packedShape;
  for (const ReassociationIndices &group : reassociation) {
    int64_t product = 1;
    for (int64_t dim : group)
      if (!multiplyDim(product, type.getDimSize(dim)))
        return failure();
    packedShape.push_back(product);
  }
  auto packedType = RankedTensorType::get(packedShape, type.getElementType(),
                                          type.getEncoding());
  return Value(tensor::CollapseShapeOp::create(builder, loc, packedType, value,
                                               reassociation));
}

static FailureOr<PackedBroadcast> packBroadcast(OpBuilder &builder,
                                                Location loc, Value lhs,
                                                Value rhs,
                                                RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  if (rank <= 4 || !resultType.hasStaticShape())
    return failure();

  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || lhsType.getRank() > rank ||
      rhsType.getRank() > rank)
    return failure();

  SmallVector<int64_t> lhsShape(rank, 1);
  SmallVector<int64_t> rhsShape(rank, 1);
  llvm::copy(lhsType.getShape(), lhsShape.begin() + rank - lhsType.getRank());
  llvm::copy(rhsType.getShape(), rhsShape.begin() + rank - rhsType.getRank());
  ArrayRef<int64_t> outShape = resultType.getShape();

  for (int64_t dim : llvm::seq<int64_t>(rank)) {
    if ((lhsShape[dim] != 1 && lhsShape[dim] != outShape[dim]) ||
        (rhsShape[dim] != 1 && rhsShape[dim] != outShape[dim]) ||
        (outShape[dim] != lhsShape[dim] && outShape[dim] != rhsShape[dim]))
      return failure();
  }

  auto groupIsSafe = [&](int64_t begin, int64_t end) {
    int64_t lhsProduct = 1;
    int64_t rhsProduct = 1;
    int64_t outProduct = 1;
    for (int64_t axis = begin; axis < end; ++axis) {
      if (!multiplyDim(lhsProduct, lhsShape[axis]) ||
          !multiplyDim(rhsProduct, rhsShape[axis]) ||
          !multiplyDim(outProduct, outShape[axis]))
        return false;
    }
    return (lhsProduct == 1 || lhsProduct == outProduct) &&
           (rhsProduct == 1 || rhsProduct == outProduct);
  };

  SmallVector<ReassociationIndices> chosen;
  for (int64_t wantedGroups = 4; wantedGroups >= 1 && chosen.empty();
       --wantedGroups) {
    SmallVector<ReassociationIndices> candidate;
    std::function<bool(int64_t, int64_t)> chooseGroups =
        [&](int64_t begin, int64_t groupsLeft) {
          if (groupsLeft == 1) {
            if (!groupIsSafe(begin, rank))
              return false;
            candidate.emplace_back();
            for (int64_t axis = begin; axis < rank; ++axis)
              candidate.back().push_back(axis);
            return true;
          }

          int64_t lastEnd = rank - (groupsLeft - 1);
          for (int64_t end = begin + 1; end <= lastEnd; ++end) {
            if (!groupIsSafe(begin, end))
              continue;
            candidate.emplace_back();
            for (int64_t axis = begin; axis < end; ++axis)
              candidate.back().push_back(axis);
            if (chooseGroups(end, groupsLeft - 1))
              return true;
            candidate.pop_back();
          }
          return false;
        };
    if (chooseGroups(0, wantedGroups))
      chosen = std::move(candidate);
  }
  if (chosen.empty())
    return failure();

  auto packOperand = [&](Value operand) -> FailureOr<Value> {
    auto operandType = cast<RankedTensorType>(operand.getType());
    // Rank-0 tensors already broadcast through every packed rank.
    if (operandType.getRank() == 0)
      return operand;
    FailureOr<Value> aligned = rightAlignTensor(builder, loc, operand, rank);
    if (failed(aligned))
      return failure();
    return collapseTensor(builder, loc, *aligned, chosen);
  };

  FailureOr<Value> packedLhs = packOperand(lhs);
  FailureOr<Value> packedRhs = packOperand(rhs);
  if (failed(packedLhs) || failed(packedRhs))
    return failure();

  SmallVector<int64_t> packedResultShape;
  for (const ReassociationIndices &group : chosen) {
    int64_t product = 1;
    for (int64_t axis : group)
      if (!multiplyDim(product, resultType.getDimSize(axis)))
        return failure();
    packedResultShape.push_back(product);
  }
  auto packedResultType = RankedTensorType::get(
      packedResultShape, resultType.getElementType(), resultType.getEncoding());
  return PackedBroadcast{*packedLhs, *packedRhs, packedResultType, resultType,
                         std::move(chosen)};
}

struct PackBroadcastPattern : public RewritePattern {
  PackBroadcastPattern(StringRef opName, MLIRContext *ctx)
      : RewritePattern(opName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() <= 4)
      return failure();

    FailureOr<PackedBroadcast> packing =
        packBroadcast(rewriter, op->getLoc(), op->getOperand(0),
                      op->getOperand(1), resultType);
    if (failed(packing))
      return failure();

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands({packing->lhs, packing->rhs});
    state.addTypes(packing->packedResultType);
    state.addAttributes(op->getAttrs());
    Operation *packedOp = rewriter.create(state);

    SmallVector<OpFoldResult> outputShape;
    for (int64_t dim : packing->originalResultType.getShape())
      outputShape.push_back(rewriter.getIndexAttr(dim));
    Value restored =
        tensor::ExpandShapeOp::create(
            rewriter, op->getLoc(), packing->originalResultType,
            packedOp->getResult(0), packing->reassociation, outputShape)
            .getResult();
    rewriter.replaceOp(op, restored);
    ++NumBroadcastsPacked;
    return success();
  }
};

} // namespace

void populatePackBroadcastTo4DPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  for (StringRef opName : {"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div",
                           "onnx.Less", "onnx.Greater"})
    patterns.add<PackBroadcastPattern>(opName, ctx);
}

} // namespace hip
} // namespace mlir
