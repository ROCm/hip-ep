/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===----------------------------------------------------------------------===//
//
// Per-op `ReifyRankedShapedTypeOpInterface` implementations for HIP dialect.
//
// `HipDialect.cpp` hosts each op's verify(), getEffects(),
// getDpsInitsMutable(), and any custom builders / printers / parsers. The
// reify implementations live here so that the per-op shape arithmetic for
// every dynamic-shape op can be read alongside each other, and so that
// `HipDialect.cpp` stays focused on op identity as the dialect picks up
// shape-inference support for more ops.
//
// Each op's reify lives under its own `// <OpName>` section banner below.
// To wire shape inference for a new op, follow the recipe in
// `docs/design/hip-shape-inference.md` — the `// <OpName>` section here is
// step 4 of that recipe.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "llvm/ADT/Sequence.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

/// Read the shape of `v` if it is a `RankedTensorType` or `MemRefType`;
/// returns an empty `ArrayRef` otherwise. Reify entries below treat the
/// empty case as a graceful bail-out (`failure()`); verify() guards the
/// same property up front.
///
/// Duplicated as a file-local static here AND in `HipDialect.cpp` rather
/// than promoted to `HipShapeUtils.h` — the helper is small enough that a
/// local copy in each TU is clearer than expanding the public API surface,
/// and the two callers may legitimately diverge in the future (e.g. the
/// verify-side helper might want to reject non-shaped types up front,
/// while the reify-side wants the silent bail-out it has today).
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
//===----------------------------------------------------------------------===//
//
// Reify for `hip.matmul`. The inferred shape is recomputed from the operand
// shapes via `inferContractionShape` (same helper that drives verify()) and
// then lifted into `OpFoldResult`s: each static dim becomes an `IndexAttr`,
// each dynamic dim becomes a `tensor.dim` / `memref.dim` of the operand
// whose runtime size determines that dim.
//
// Source-of-dim contract:
//   - last-2 dims (M, N): `M = A[-2]`, `N = B[-1]`
//   - batch dims (right-aligned, broadcast over leading dims): prefer the
//     side that contributes the dim (in-range and not 1). If both sides
//     are dynamic the choice is fixed (A first, then B) so downstream folds
//     see a stable shape source.
//
// Before:
//   %m = hip.matmul ins(%ctx, %a : !hip.context, tensor<?x4096xf16>),
//                   ins(%b   : tensor<4096x4096xf16>),
//                   outs(%out : tensor<?x4096xf16>) -> tensor<?x4096xf16>
//
// After (reified shapes for the result of %m):
//   //   dim 0 (dynamic M)  -> %d0 = tensor.dim %a, %c0 : tensor<?x4096xf16>
//   //   dim 1 (static N)   -> 4096 : index
//
//===----------------------------------------------------------------------===//

LogicalResult
MatmulOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  // memref-mode matmul has no SSA results — by interface contract
  // `reifyResultShapes` is only called on ops with `RankedTensorType`
  // results (see `InferTypeOpInterface.td`). Bail safely if invoked
  // anyway.
  if (getNumResults() == 0)
    return failure();

  ArrayRef<int64_t> aShape = getShapeOf(getA());
  ArrayRef<int64_t> bShape = getShapeOf(getB());
  if (aShape.empty() || bShape.empty())
    return failure();

  // Recompute the static shape vector via the shared helper. By the time
  // reify runs, verify() has already approved the shapes, so the error
  // callback is unreachable in practice — but bail safely on empty()
  // anyway, in case a future caller invokes reify on an op that has not
  // yet been through verification.
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

  SmallVector<OpFoldResult> dims;
  dims.reserve(outRank);
  for (size_t i : llvm::seq<size_t>(0, outRank)) {
    if (i + 2 == outRank) {
      // M dim: comes from A's [-2].
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], A, aRank - 2));
      continue;
    }
    if (i + 1 == outRank) {
      // N dim: comes from B's [-1].
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], B, bRank - 1));
      continue;
    }
    // Batch dim: right-aligned over A's and B's batch shapes.
    size_t batchRank = outRank - 2;
    size_t aBatchRank = aRank >= 2 ? aRank - 2 : 0;
    size_t bBatchRank = bRank >= 2 ? bRank - 2 : 0;
    size_t aPad = batchRank - aBatchRank;
    size_t bPad = batchRank - bBatchRank;
    int64_t aDim = i < aPad ? 1 : aShape[i - aPad];
    int64_t bDim = i < bPad ? 1 : bShape[i - bPad];

    // Static result -> IntegerAttr (no IR).
    if (!ShapedType::isDynamic(outShape[i])) {
      dims.push_back(b.getIndexAttr(outShape[i]));
      continue;
    }
    // Dynamic result -> dim of whichever input is the actual source. If A's
    // side is 1 or out-of-range, B is the source; otherwise A. This keeps
    // the emitted dim op tied to the operand whose runtime size determines
    // the result, which is the invariant downstream folds rely on.
    bool aSourceCanonical =
        i >= aPad && aDim != 1; // A contributes when it's in range and not 1
    bool bSourceCanonical = i >= bPad && bDim != 1;
    if (aSourceCanonical) {
      dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, ShapedType::kDynamic,
                                                   A, i - aPad));
    } else if (bSourceCanonical) {
      dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, ShapedType::kDynamic,
                                                   B, i - bPad));
    } else if (i >= aPad) {
      // Both sides are 1 / dynamic — A is in range, prefer it.
      dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, ShapedType::kDynamic,
                                                   A, i - aPad));
    } else {
      dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, ShapedType::kDynamic,
                                                   B, i - bPad));
    }
  }
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}
