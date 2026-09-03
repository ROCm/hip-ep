/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PackBroadcastTo4D.cpp - Pre-lower binary broadcasts to 4-D ---------===//
//
// Pre-lowering pattern set inside convert-onnx-to-hip, alongside
// GatherShapeFold / ReshapeShapeFold / PadShapeFold / SliceShapeFold. It
// rewrites static high-rank onnx.Add/Sub/Mul/Div/Less/Greater/Max/Min/And
// operations into collapse_shape -> rank-<=4 ONNX op -> expand_shape before
// compute conversion creates the corresponding HIP op. Only the two-operand
// form is matched; variadic onnx.Max/onnx.Min (three or more inputs) are left
// unchanged.
//
// onnx.Clip and onnx.Relu are packed the same way. They are not broadcasts,
// but both expand into hip.max/hip.min during conversion and so hit the same
// rank-4 ceiling in the elementwise ABI; their broadcast lives entirely on the
// first operand, and Clip's optional min/max bounds are scalars that are
// forwarded unchanged.
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
  SmallVector<Value> operands;
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
                                                Location loc,
                                                ArrayRef<Value> dataOperands,
                                                RankedTensorType resultType) {
  int64_t rank = resultType.getRank();
  if (rank <= 4 || !resultType.hasStaticShape())
    return failure();
  if (dataOperands.empty())
    return failure();

  // Right-align every operand against the result rank so the broadcast checks
  // below can compare them axis by axis.
  SmallVector<SmallVector<int64_t>> operandShapes;
  for (Value operand : dataOperands) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type || !type.hasStaticShape() || type.getRank() > rank)
      return failure();
    SmallVector<int64_t> shape(rank, 1);
    llvm::copy(type.getShape(), shape.begin() + rank - type.getRank());
    operandShapes.push_back(std::move(shape));
  }
  ArrayRef<int64_t> outShape = resultType.getShape();

  for (int64_t dim : llvm::seq<int64_t>(rank)) {
    bool someOperandCoversOut = false;
    for (const SmallVector<int64_t> &shape : operandShapes) {
      if (shape[dim] != 1 && shape[dim] != outShape[dim])
        return failure();
      if (shape[dim] == outShape[dim])
        someOperandCoversOut = true;
    }
    if (!someOperandCoversOut)
      return failure();
  }

  auto groupIsSafe = [&](int64_t begin, int64_t end) {
    int64_t outProduct = 1;
    for (int64_t axis = begin; axis < end; ++axis)
      if (!multiplyDim(outProduct, outShape[axis]))
        return false;
    for (const SmallVector<int64_t> &shape : operandShapes) {
      int64_t product = 1;
      for (int64_t axis = begin; axis < end; ++axis)
        if (!multiplyDim(product, shape[axis]))
          return false;
      if (product != 1 && product != outProduct)
        return false;
    }
    return true;
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

  SmallVector<Value> packedOperands;
  for (Value operand : dataOperands) {
    FailureOr<Value> packed = packOperand(operand);
    if (failed(packed))
      return failure();
    packedOperands.push_back(*packed);
  }

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
  return PackedBroadcast{std::move(packedOperands), packedResultType,
                         resultType, std::move(chosen)};
}

struct PackBroadcastPattern : public RewritePattern {
  // The first `numDataOperands` operands carry the broadcast and get packed.
  // `maxOperands` bounds the total operand count; it exceeds
  // `numDataOperands` only for onnx.Clip, whose optional min/max bounds ride
  // along unchanged and so must be rank-0 tensors or `none`. Every other op
  // is matched at an exact count, which keeps variadic onnx.Max/onnx.Min out
  // of the pattern.
  PackBroadcastPattern(StringRef opName, MLIRContext *ctx,
                       unsigned numDataOperands, unsigned maxOperands)
      : RewritePattern(opName, /*benefit=*/1, ctx),
        numDataOperands(numDataOperands), maxOperands(maxOperands) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    unsigned numOperands = op->getNumOperands();
    if (numOperands < numDataOperands || numOperands > maxOperands ||
        op->getNumResults() != 1)
      return failure();
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() <= 4)
      return failure();

    // A pass-through operand that carried shape would silently disagree with
    // the packed result, so only scalars and `none` may ride along. This is a
    // whitelist on purpose: an unranked operand carries no rank to check, so
    // rejecting merely the ranked-and-non-scalar case would let it through.
    OperandRange trailing = op->getOperands().drop_front(numDataOperands);
    for (Value extra : trailing) {
      if (isa<NoneType>(extra.getType()))
        continue;
      auto extraType = dyn_cast<RankedTensorType>(extra.getType());
      if (!extraType || extraType.getRank() != 0)
        return failure();
    }

    // An op whose optional operands are all absent is an identity: onnx.Clip
    // with neither bound does nothing, so packing it would churn reshapes and
    // inflate the statistic for no gain.
    auto isAbsent = [](Value v) { return isa<NoneType>(v.getType()); };
    if (maxOperands > numDataOperands && llvm::all_of(trailing, isAbsent))
      return failure();

    SmallVector<Value> dataOperands(
        op->getOperands().take_front(numDataOperands));
    FailureOr<PackedBroadcast> packing =
        packBroadcast(rewriter, op->getLoc(), dataOperands, resultType);
    if (failed(packing))
      return failure();

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(packing->operands);
    state.addOperands(op->getOperands().drop_front(numDataOperands));
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

private:
  unsigned numDataOperands;
  unsigned maxOperands;
};

} // namespace

void populatePackBroadcastTo4DPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  for (StringRef opName :
       {"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Less",
        "onnx.Greater", "onnx.Max", "onnx.Min", "onnx.And"})
    patterns.add<PackBroadcastPattern>(opName, ctx, /*numDataOperands=*/2,
                                       /*maxOperands=*/2);
  // hip.max and hip.min are also produced by onnx.Clip and onnx.Relu, which
  // carry the whole broadcast on their first operand; Clip's min/max bounds
  // are scalars that pack through untouched.
  patterns.add<PackBroadcastPattern>("onnx.Relu", ctx, /*numDataOperands=*/1,
                                     /*maxOperands=*/1);
  patterns.add<PackBroadcastPattern>("onnx.Clip", ctx, /*numDataOperands=*/1,
                                     /*maxOperands=*/3);
}

} // namespace hip
} // namespace mlir
