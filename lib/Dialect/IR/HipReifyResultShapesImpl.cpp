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
  // results. Bail safely if invoked anyway.
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
    // Batch dim: pick whichever input contributes the runtime size. A
    // contributes when it's in range AND its dim is not 1; B contributes
    // symmetrically. When neither is canonical (both 1 / out-of-range /
    // dynamic), prefer A when in range so downstream folds see a stable
    // source. `reifyDimOrConstant` handles the static-vs-dynamic dispatch
    // on `outShape[i]` -- no separate static-result fast path needed.
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
