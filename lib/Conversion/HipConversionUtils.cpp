/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipConversionUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace mlir::hip {

bool isLosslessShapeIndexCast(arith::IndexCastOp op) {
  constexpr unsigned kIndexBitWidth = 64;
  Type sourceType = op.getIn().getType();
  Type resultType = op.getType();
  if (sourceType.isIndex()) {
    auto integerType = dyn_cast<IntegerType>(resultType);
    return integerType && integerType.getWidth() >= kIndexBitWidth;
  }
  if (resultType.isIndex()) {
    auto integerType = dyn_cast<IntegerType>(sourceType);
    return integerType && integerType.getWidth() <= kIndexBitWidth;
  }
  return false;
}

DenseElementsAttr matchHipCompileTimeConstantTensor(Value value) {
  if (!value)
    return {};
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return {};

  if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    return dense && dense.getType() == value.getType() ? dense
                                                       : DenseElementsAttr();
  }

  auto carrier = dyn_cast<hip::ConstantOp>(defOp);
  if (!carrier || carrier.getResult() != value)
    return {};
  auto dense = dyn_cast_or_null<DenseElementsAttr>(carrier.getValueAttr());
  return dense && dense.getType() == carrier.getResult().getType()
             ? dense
             : DenseElementsAttr();
}

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

FailureOr<Value>
createEmptyTensorFromReifiedShape(OpBuilder &builder, Location loc,
                                  RankedTensorType resultType,
                                  llvm::ArrayRef<OpFoldResult> reifiedShape) {
  if (static_cast<int64_t>(reifiedShape.size()) != resultType.getRank())
    return failure();

  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      continue;
    std::optional<int64_t> reifiedStatic =
        getConstantIntValue(reifiedShape[dimIdx]);
    if (reifiedStatic && *reifiedStatic != resultType.getDimSize(dimIdx))
      return failure();
  }

  llvm::SmallVector<Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank()))
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, reifiedShape[dimIdx]));
  return Value(tensor::EmptyOp::create(builder, loc, resultType, dynSizes));
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

  FailureOr<llvm::SmallVector<OpFoldResult>> shape = reifyBroadcastResultShape(
      builder, loc, operands, [&] { return emitError(loc); });
  if (failed(shape))
    return failure();
  return createEmptyTensorFromReifiedShape(builder, loc, resultType, *shape);
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
