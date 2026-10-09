/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipConversionUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace mlir::hip {

bool isResultTypeCompatibleWithInferredShape(
    RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape) {
  if (resultType.getRank() != static_cast<int64_t>(inferredShape.size()))
    return false;
  for (auto [actual, inferred] :
       llvm::zip_equal(resultType.getShape(), inferredShape)) {
    if (!ShapedType::isDynamic(actual) && !ShapedType::isDynamic(inferred) &&
        actual != inferred)
      return false;
  }
  return true;
}

FailureOr<Value> createBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            ValueRange operands) {
  llvm::SmallVector<llvm::ArrayRef<int64_t>> staticShapes;
  staticShapes.reserve(operands.size());
  for (Value operand : operands) {
    auto ranked = dyn_cast<RankedTensorType>(operand.getType());
    if (!ranked)
      return failure();
    staticShapes.push_back(ranked.getShape());
  }

  FailureOr<llvm::SmallVector<int64_t>> inferredShape =
      inferBroadcastShape(staticShapes, [&] { return emitError(loc); });
  if (failed(inferredShape) ||
      !isResultTypeCompatibleWithInferredShape(resultType, *inferredShape))
    return failure();

  // Preserve conversion's existing dynamic extent selection. Shape interfaces
  // use reifyBroadcastResultShape for exact runtime broadcast extents.
  int64_t resultRank = resultType.getRank();
  llvm::SmallVector<Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultRank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;

    Value chosen;
    int64_t chosenDim = -1;
    Value fallback;
    int64_t fallbackDim = -1;
    for (Value operand : operands) {
      auto type = dyn_cast<RankedTensorType>(operand.getType());
      int64_t offset = resultRank - type.getRank();
      if (dimIdx < offset)
        continue;
      int64_t operandDim = dimIdx - offset;
      if (!fallback) {
        fallback = operand;
        fallbackDim = operandDim;
      }
      if (!type.isDynamicDim(operandDim) && type.getDimSize(operandDim) == 1)
        continue;
      chosen = operand;
      chosenDim = operandDim;
      break;
    }
    if (!chosen) {
      chosen = fallback;
      chosenDim = fallbackDim;
    }
    if (!chosen)
      return failure();
    dynSizes.push_back(tensor::DimOp::create(builder, loc, chosen, chosenDim));
  }
  return Value(tensor::EmptyOp::create(builder, loc, resultType, dynSizes));
}

FailureOr<Value> getContextArg(Operation *op, PatternRewriter &rewriter) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return rewriter.notifyMatchFailure(op, "not inside a function");
  auto &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0)
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  Value ctx = entry.getArgument(0);
  if (!isa<hip::ContextType>(ctx.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hip.context");
  return ctx;
}

} // namespace mlir::hip
