/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SumConversion.cpp - onnx.Sum -> onnx.Add chain --------------------===//
//
// onnx.Sum is the variadic form of elementwise addition. hip.add already
// lowers the binary case, including NumPy broadcast and the rank>4 packing
// pass, so Sum becomes a left fold of onnx.Add. Gelu fusion matches Sum
// before this pattern runs.
//
//   Before:
//     %y = "onnx.Sum"(%a, %b, %c)
//          : (tensor<1x4x8x8xf32>, tensor<1x4x1x1xf32>, tensor<1x4x8x8xf32>)
//            -> tensor<1x4x8x8xf32>
//   After:
//     %t = "onnx.Add"(%a, %b)
//          : (tensor<1x4x8x8xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x8x8xf32>
//     %y = "onnx.Add"(%t, %c)
//          : (tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include <algorithm>

namespace mlir {
namespace hip {
namespace {

// Right-align two ranked shapes. A static dim broadcasts when it is 1 or
// equal to the other dim. A dynamic dim takes the other static extent.
FailureOr<RankedTensorType> broadcastRankedTypes(RankedTensorType lhs,
                                                 RankedTensorType rhs) {
  if (lhs.getElementType() != rhs.getElementType())
    return failure();
  int64_t rank = std::max(lhs.getRank(), rhs.getRank());
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t lhsDim = i < rank - lhs.getRank()
                         ? 1
                         : lhs.getDimSize(i - (rank - lhs.getRank()));
    int64_t rhsDim = i < rank - rhs.getRank()
                         ? 1
                         : rhs.getDimSize(i - (rank - rhs.getRank()));
    if (lhsDim == 1)
      shape[i] = rhsDim;
    else if (rhsDim == 1 || lhsDim == rhsDim)
      shape[i] = lhsDim;
    else if (ShapedType::isDynamic(lhsDim))
      shape[i] = rhsDim;
    else if (ShapedType::isDynamic(rhsDim))
      shape[i] = lhsDim;
    else
      return failure();
  }
  return RankedTensorType::get(shape, lhs.getElementType());
}

// The declared ONNX result may be more specific than the pairwise broadcast
// (a static dim where the fold still has a dynamic one) but must not disagree
// on a static extent.
bool broadcastMatchesResult(RankedTensorType broadcast,
                            RankedTensorType result) {
  if (broadcast.getRank() != result.getRank() ||
      broadcast.getElementType() != result.getElementType())
    return false;
  for (int64_t i = 0; i < result.getRank(); ++i) {
    if (result.isDynamicDim(i) || broadcast.isDynamicDim(i))
      continue;
    if (result.getDimSize(i) != broadcast.getDimSize(i))
      return false;
  }
  return true;
}

Value emitOnnxAdd(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
                  Type resultType) {
  OperationState state(loc, "onnx.Add");
  state.addOperands({lhs, rhs});
  state.addTypes(resultType);
  return rewriter.create(state)->getResult(0);
}

struct SumToAdd : public RewritePattern {
  SumToAdd(MLIRContext *ctx) : RewritePattern("onnx.Sum", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Sum expects at least one operand and one result");

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "onnx.Sum result must be ranked");

    SmallVector<Value> operands(op->getOperands());
    SmallVector<RankedTensorType> types;
    types.reserve(operands.size());
    for (Value operand : operands) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      if (!type || type.getElementType() != resultType.getElementType())
        return rewriter.notifyMatchFailure(
            op, "onnx.Sum operands must be ranked tensors of the result type");
      types.push_back(type);
    }

    // A single input is already the sum.
    if (operands.size() == 1) {
      if (types.front() != resultType)
        return rewriter.notifyMatchFailure(
            op, "onnx.Sum of one tensor must keep that tensor's type");
      rewriter.replaceOp(op, operands.front());
      return success();
    }

    Location loc = op->getLoc();
    Value acc = operands.front();
    RankedTensorType accType = types.front();
    for (size_t i = 1, n = operands.size(); i < n; ++i) {
      FailureOr<RankedTensorType> broadcast =
          broadcastRankedTypes(accType, types[i]);
      if (failed(broadcast))
        return rewriter.notifyMatchFailure(
            op, "onnx.Sum operands are not broadcast-compatible");
      bool last = i + 1 == n;
      if (last && !broadcastMatchesResult(*broadcast, resultType))
        return rewriter.notifyMatchFailure(
            op, "onnx.Sum result shape is not the broadcast of its inputs");
      Type addType = last ? Type(resultType) : Type(*broadcast);
      acc = emitOnnxAdd(rewriter, loc, acc, operands[i], addType);
      accType = cast<RankedTensorType>(addType);
    }
    rewriter.replaceOp(op, acc);
    return success();
  }
};

} // namespace

void populateSumConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<SumToAdd>(ctx);
}

} // namespace hip
} // namespace mlir
