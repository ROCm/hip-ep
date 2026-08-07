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
  // Axis membership over the closed domain [0, dataRank).
  llvm::SmallBitVector reduced(dataRank);
  for (int64_t axis : axes) {
    int64_t normalized = axis < 0 ? axis + dataRank : axis;
    if (normalized < 0 || normalized >= dataRank)
      return failure();
    if (reduced.test(normalized))
      return failure();
    reduced.set(normalized);
  }

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

  SmallVector<int64_t> axesList;
  if (!matchConstantIntTensor(axes, axesList))
    return success();
  if (axesList.empty() && noopWithEmptyAxes == 0)
    axesList = llvm::to_vector(llvm::seq<int64_t>(0, dataType.getRank()));

  FailureOr<SmallVector<int64_t>> expected =
      inferReductionShape(dataType.getShape(), axesList, keepdims);
  if (failed(expected))
    return op->emitOpError(
        "constant axes must be unique and in the range [-rank, rank)");
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

/// Reduction result shape recovered from a constant `axes` operand.
///
/// Introspects `axes` as an `arith.constant` (the typical case after the
/// OnnxToHip converter materializes it from the ONNX attribute), resolves
/// ONNX's empty-axes semantics against `noop_with_empty_axes` — reduce every
/// axis when 0, reduce nothing when 1 — and delegates the shape rule to
/// `reifyReductionResultShape`.
///
/// Returns failure when `axes` is not a recognised constant, which is why this
/// returns `LogicalResult` and writes through `out`: a valid rank-0 reduction
/// result is a successful *empty* dim list and would otherwise be
/// indistinguishable from the bail path.
LogicalResult reifyReductionWithKeepdims(OpBuilder &b, Location loc, Value data,
                                         Value axes, int64_t keepdims,
                                         int64_t noopWithEmptyAxes,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();

  SmallVector<int64_t> axesList;
  if (!matchConstantIntTensor(axes, axesList))
    return failure();
  if (axesList.empty() && noopWithEmptyAxes == 0) {
    // Reduce every axis. `noop_with_empty_axes = 1` instead means "reduce
    // nothing", which the empty list already expresses.
    axesList = llvm::to_vector(llvm::seq<int64_t>(0, dataType.getRank()));
  }

  FailureOr<SmallVector<OpFoldResult>> dims =
      mlir::hip::reifyReductionResultShape(b, loc, data, axesList, keepdims);
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

  // Tier-1: introspect `axes` as an `arith.constant` and reify per-dim
  // from the input shape with keepdims awareness.
  SmallVector<OpFoldResult> dims;
  if (succeeded(reifyReductionWithKeepdims(b, loc, data, axes, keepdims,
                                           noopWithEmptyAxes, dims))) {
    reified.assign({std::move(dims)});
    return success();
  }

  // Fallback: lift the DPS `outs` operand's own shape via the shared
  // `HipDpsOp` default. `cast<HipDpsOp>(op).reifyResultShapes` dispatches
  // through the `HipDpsOp` interface concept and lands on the
  // interface-default body in `HipDpsOpInterface.cpp` — it walks
  // `getDpsInits()` and lifts each via `tensor::getMixedSizes` /
  // `memref::getMixedSizes`. Reduction ops override the SEPARATE
  // `ReifyRankedShapedTypeOpInterface::reifyResultShapes` (auto-emitted
  // by `Hip_DpsOp_Reduction` and the body of THIS function), but do not
  // override the `HipDpsOp` interface method, so the call below resolves
  // to the default body and does not recurse.
  return cast<HipDpsOp>(op).reifyResultShapes(b, reified);
}
