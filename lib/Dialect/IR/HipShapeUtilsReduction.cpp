/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsReduction.cpp - Reduction shape helpers --------------===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hip;

namespace {

/// Map each output dimension of an ONNX reduction to the input dimension it
/// takes its extent from; `std::nullopt` marks a reduced axis retained by
/// `keepdims`, whose extent is the literal 1 rather than an input extent.
///
/// `axes` holds reduced axis indices in the ONNX negative-axis convention; an
/// empty list means no reduction. Returns failure when an axis is out of range.
///
/// This is the single source of truth behind `inferReductionShape` and
/// `reifyReductionResultShape`, so the static and mixed forms cannot disagree.
FailureOr<SmallVector<std::optional<int64_t>>>
computeReductionDimMap(int64_t dataRank, ArrayRef<int64_t> axes,
                       int64_t keepdims) {
  if (keepdims != 0 && keepdims != 1)
    return failure();
  FailureOr<SmallVector<int64_t>> normalizedAxes =
      normalizeReductionAxes(dataRank, axes);
  if (failed(normalizedAxes))
    return failure();

  // Axis membership over the closed domain [0, dataRank).
  llvm::SmallBitVector reduced(dataRank);
  for (int64_t axis : *normalizedAxes)
    reduced.set(axis);

  SmallVector<std::optional<int64_t>> dimMap;
  dimMap.reserve(dataRank);
  for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
    if (!reduced.test(i))
      dimMap.push_back(i);
    else if (keepdims)
      dimMap.push_back(std::nullopt);
    // keepdims=0: the reduced axis leaves the output rank entirely.
  }
  return dimMap;
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::normalizeReductionAxes(int64_t dataRank, ArrayRef<int64_t> axes) {
  if (dataRank < 0)
    return failure();

  llvm::SmallBitVector seen(dataRank);
  SmallVector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t axis : axes) {
    int64_t value = axis < 0 ? axis + dataRank : axis;
    if (value < 0 || value >= dataRank || seen.test(value))
      return failure();
    seen.set(value);
    normalized.push_back(value);
  }
  llvm::sort(normalized);
  for (size_t i = 1; i < normalized.size(); ++i)
    if (normalized[i] != normalized[i - 1] + 1)
      return failure();
  return normalized;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::resolveConstantReductionAxes(Value axes, int64_t dataRank,
                                        int64_t noopWithEmptyAxes) {
  if (noopWithEmptyAxes != 0 && noopWithEmptyAxes != 1)
    return failure();
  SmallVector<int64_t> rawAxes;
  if (!matchConstantIntTensor(axes, rawAxes))
    return failure();
  if (rawAxes.empty() && noopWithEmptyAxes == 0)
    rawAxes = llvm::to_vector(llvm::seq<int64_t>(0, dataRank));
  return normalizeReductionAxes(dataRank, rawAxes);
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferReductionShape(ArrayRef<int64_t> dataShape,
                               ArrayRef<int64_t> axes, int64_t keepdims) {
  FailureOr<SmallVector<std::optional<int64_t>>> dimMap =
      computeReductionDimMap(dataShape.size(), axes, keepdims);
  if (failed(dimMap))
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(dimMap->size());
  for (std::optional<int64_t> sourceDim : *dimMap)
    shape.push_back(sourceDim ? dataShape[*sourceDim] : 1);
  return shape;
}

LogicalResult mlir::hip::verifyReductionDpsOp(Operation *op, Value data,
                                              Value axes, int64_t keepdims,
                                              int64_t noopWithEmptyAxes) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "reduction verification requires DestinationStyleOpInterface");
  SmallVector<Value> operands = {data, axes};
  llvm::append_range(operands, dpsOp.getDpsInits());
  if (failed(verifyDpsComputeOp(op, operands, /*numInits=*/1)))
    return failure();

  auto dataType = cast<ShapedType>(data.getType());
  auto axesType = cast<ShapedType>(axes.getType());
  if ((axesType.getRank() != 0 && axesType.getRank() != 1) ||
      !axesType.getElementType().isInteger(64))
    return op->emitOpError(
        "axes must be a rank-0 or rank-1 i64 tensor or memref");
  if (keepdims != 0 && keepdims != 1)
    return op->emitOpError("keepdims must be 0 or 1");
  if (noopWithEmptyAxes != 0 && noopWithEmptyAxes != 1)
    return op->emitOpError("noop_with_empty_axes must be 0 or 1");

  SmallVector<int64_t> rawAxes;
  if (!matchConstantIntTensor(axes, rawAxes))
    return op->emitOpError(
        "axes must have a structurally-proven compile-time constant source");
  FailureOr<SmallVector<int64_t>> axesList =
      resolveConstantReductionAxes(axes, dataType.getRank(), noopWithEmptyAxes);
  if (failed(axesList))
    return op->emitOpError(
        "constant axes must be unique, in range, and form one contiguous span");
  auto normalizedAttr = op->getAttrOfType<DenseI64ArrayAttr>("normalized_axes");
  if (!normalizedAttr || !llvm::equal(normalizedAttr.asArrayRef(), *axesList))
    return op->emitOpError(
        "normalized_axes must exactly match the normalized constant axes "
        "source");

  FailureOr<SmallVector<int64_t>> expected =
      inferReductionShape(dataType.getShape(), *axesList, keepdims);
  if (failed(expected))
    return op->emitOpError("invalid reduction shape contract");
  return verifyHipOpShape(
      op, [&]() -> FailureOr<SmallVector<int64_t>> { return *expected; });
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyReductionResultShape(OpBuilder &b, Location loc, Value data,
                                     ArrayRef<int64_t> axes, int64_t keepdims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h).
  FailureOr<SmallVector<std::optional<int64_t>>> dimMap =
      computeReductionDimMap(dataType.getRank(), axes, keepdims);
  if (failed(dimMap))
    return failure();

  ArrayRef<int64_t> dataShape = dataType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(dimMap->size());
  for (std::optional<int64_t> sourceDim : *dimMap)
    dims.push_back(sourceDim ? reifyDimOrConstant(b, loc, dataShape[*sourceDim],
                                                  data, *sourceDim)
                             : OpFoldResult(b.getIndexAttr(1)));
  return dims;
}

namespace {

/// Reduction result shape recovered from a structurally-proven constant
/// `axes` operand.
///
/// Accepts inline tensor constants and constant globals after bufferization,
/// resolves ONNX's empty-axes semantics, normalizes and validates the
/// contiguous span, then delegates to `reifyReductionResultShape`.
///
/// Returns failure when `axes` is not a recognised constant, which is why this
/// returns `LogicalResult` and writes through `out`: a valid rank-0 reduction
/// result is a successful *empty* dim list and would otherwise be
/// indistinguishable from the bail path.
LogicalResult reifyReductionWithKeepdims(OpBuilder &b, Location loc, Value data,
                                         Value axes, int64_t keepdims,
                                         int64_t noopWithEmptyAxes,
                                         Operation *op,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();

  FailureOr<SmallVector<int64_t>> axesList =
      resolveConstantReductionAxes(axes, dataType.getRank(), noopWithEmptyAxes);
  if (failed(axesList))
    return failure();
  auto normalizedAttr = op->getAttrOfType<DenseI64ArrayAttr>("normalized_axes");
  if (!normalizedAttr || !llvm::equal(normalizedAttr.asArrayRef(), *axesList))
    return failure();

  FailureOr<SmallVector<OpFoldResult>> dims =
      mlir::hip::reifyReductionResultShape(b, loc, data, *axesList, keepdims);
  if (failed(dims))
    return failure();
  out.assign(dims->begin(), dims->end());
  return success();
}

} // namespace

LogicalResult
mlir::hip::reifyReductionShape(OpBuilder &b, Location loc, Value data,
                               Value axes, int64_t keepdims,
                               int64_t noopWithEmptyAxes, Operation *op,
                               ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(data.getType()))
    return failure();

  // Reify from the validated structural constant source with keepdims
  // awareness. Failure occurs before any dimension IR is emitted.
  SmallVector<OpFoldResult> dims;
  if (succeeded(reifyReductionWithKeepdims(b, loc, data, axes, keepdims,
                                           noopWithEmptyAxes, op, dims))) {
    reified.assign({std::move(dims)});
    return success();
  }

  return failure();
}
