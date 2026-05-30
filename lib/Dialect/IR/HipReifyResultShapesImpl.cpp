/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipReifyResultShapesImpl.cpp ---------------------------------------===//
//
// Per-op `ReifyRankedShapedTypeOpInterface` impls for HIP dialect ops. One
// section per op below. See `docs/design/hip-shape-inference.md` for the
// recipe to wire a new op.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "llvm/ADT/Sequence.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

/// Read the shape of `v` if shaped, else return empty (treated as a graceful
/// bail-out by callers below). Duplicated in `HipDialect.cpp`; the two
/// callers may diverge later (verify rejects non-shaped, reify bails).
ArrayRef<int64_t> getShapeOf(Value v) {
  if (auto t = dyn_cast<RankedTensorType>(v.getType()))
    return t.getShape();
  if (auto m = dyn_cast<MemRefType>(v.getType()))
    return m.getShape();
  return {};
}

} // namespace

//===----------------------------------------------------------------------===//
// MatmulOp
//
// Reify uses `inferContractionShape` to recompute the result shape, then
// lifts each dim to an OpFoldResult: static -> IndexAttr, dynamic ->
// tensor.dim of the operand that contributes the runtime size.
// Source: M = A[-2], N = B[-1], batch dim = the broadcast-canonical side.
//
// Before:
//   %m = hip.matmul ins(%a, %b : tensor<?x4096xf16>, tensor<4096x4096xf16>)
//                   outs(%out : tensor<?x4096xf16>) -> tensor<?x4096xf16>
// After (reified result shape):
//   dim 0 (dynamic M) -> %d0 = tensor.dim %a, %c0
//   dim 1 (static N)  -> 4096 : index
//===----------------------------------------------------------------------===//

LogicalResult
MatmulOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  // memref-mode has no SSA results; reify is only called on tensor mode
  // per interface contract, but bail defensively if invoked anyway.
  if (getNumResults() == 0)
    return failure();

  ArrayRef<int64_t> aShape = getShapeOf(getA());
  ArrayRef<int64_t> bShape = getShapeOf(getB());
  if (aShape.empty() || bShape.empty())
    return failure();

  // Re-run the contraction-shape helper. verify() has already passed by reify
  // time, but bail on empty() in case a pre-verify call sneaks in.
  SmallVector<int64_t> outShape = mlir::hip::inferContractionShape(
      aShape, bShape, [&]() { return this->emitOpError(); });
  if (outShape.empty())
    return failure();

  Location loc = getLoc();
  Value A = getA();
  Value B = getB();
  size_t outRank = outShape.size();
  size_t aRank = aShape.size();
  size_t bRank = bShape.size();

  // Loop-invariant: right-alignment padding for A's and B's batch dims.
  size_t batchRank = outRank - 2;
  size_t aPad = batchRank - (aRank >= 2 ? aRank - 2 : 0);
  size_t bPad = batchRank - (bRank >= 2 ? bRank - 2 : 0);

  SmallVector<OpFoldResult> dims;
  dims.reserve(outRank);
  for (size_t i : llvm::seq<size_t>(0, outRank)) {
    // M dim: A[-2].
    if (i + 2 == outRank) {
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], A, aRank - 2));
      continue;
    }
    // N dim: B[-1].
    if (i + 1 == outRank) {
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], B, bRank - 1));
      continue;
    }
    // Batch dim: prefer the side that contributes the size (in range, not 1).
    // When neither contributes, prefer A in range so folds see a stable source.
    int64_t aDim = i < aPad ? 1 : aShape[i - aPad];
    int64_t bDim = i < bPad ? 1 : bShape[i - bPad];
    bool aCanonical = i >= aPad && aDim != 1;
    bool bCanonical = i >= bPad && bDim != 1;
    bool pickA = aCanonical || (!bCanonical && i >= aPad);
    Value src = pickA ? A : B;
    size_t srcDim = pickA ? i - aPad : i - bPad;
    dims.push_back(
        mlir::hip::reifyDimOrConstant(b, loc, outShape[i], src, srcDim));
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}
