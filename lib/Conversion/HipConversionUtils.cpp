/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipConversionUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
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

  // Use the same extents for the destination and result shape queries.
  // Before: empty(dim(a, 0)) for a: tensor<?xf32>, b: tensor<?xf32>.
  // After:  empty(select(dim(a, 0) == 1, dim(b, 0), dim(a, 0))).
  // A folded extent can contradict the imported result type. Keep all new
  // operations detached until these checks pass.
  Block pending;
  OpBuilder shapeBuilder(builder.getContext());
  shapeBuilder.setInsertionPointToEnd(&pending);
  auto shape = reifyBroadcastResultShape(shapeBuilder, loc, operands,
                                         [&] { return emitError(loc); });
  if (failed(shape))
    return failure();

  llvm::SmallVector<Value> dynSizes;
  for (auto [extent, inferred] :
       llvm::zip_equal(resultType.getShape(), *shape)) {
    auto constant = getConstantIntValue(inferred);
    if (constant && (*constant < 0 ||
                     (!ShapedType::isDynamic(extent) && extent != *constant)))
      return failure();
    if (ShapedType::isDynamic(extent))
      dynSizes.push_back(
          getValueOrCreateConstantIndexOp(shapeBuilder, loc, inferred));
  }
  Value result =
      tensor::EmptyOp::create(shapeBuilder, loc, resultType, dynSizes);
  // Insert through the caller's builder to notify its listener.
  while (!pending.empty()) {
    Operation *op = &pending.front();
    op->remove();
    builder.insert(op);
  }
  return result;
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
