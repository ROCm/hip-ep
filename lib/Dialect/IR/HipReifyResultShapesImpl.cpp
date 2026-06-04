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

/// Read the shape of `v` if shaped, else return empty. Callers below treat
/// empty as a graceful bail-out (return failure() and let the caller of
/// reifyResultShapes fall back to using the existing result type's shape).
/// `HipDialect.cpp` carries a near-twin used in `verify()`; keeping the
/// two distinct lets verify reject non-shaped values while reify bails
/// silently if the contract ever loosens.
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
// Reify recomputes the result shape via `inferMatmulShape`, then lifts
// each dim to an OpFoldResult: static dims become `IndexAttr`; dynamic
// dims become `tensor.dim` of whichever operand contributes the runtime
// size — M from A[-2], N from B[-1], batch from the broadcast-canonical
// side.
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

  // Re-run the matmul-shape helper. verify() has already passed by reify
  // time, but bail on empty() in case a pre-verify call sneaks in.
  SmallVector<int64_t> outShape = mlir::hip::inferMatmulShape(
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

//===----------------------------------------------------------------------===//
// RopeOp
//
// Result shape == input data tensor's shape (rotary embedding rotates
// values within each head; output rank/extents match `$input` exactly).
//
// Before:
//   %y = hip.rope(%ctx) ins(%x, %pos, %cos, %sin :
//                            tensor<?x?x4096xf16>, ...)
//                       outs(%out : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 (dynamic) -> %d0 = tensor.dim %x, %c0
//   dim 1 (dynamic) -> %d1 = tensor.dim %x, %c1
//   dim 2 (static)  -> 4096 : index
//===----------------------------------------------------------------------===//

LogicalResult
RopeOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

//===----------------------------------------------------------------------===//
// RmsNormOp
//
// Result shape == input data tensor's shape (per-element normalization;
// `$scale` broadcasts over leading dims and does not contribute extents).
//
// Before:
//   %y = hip.rms_norm(%ctx) ins(%x, %scale : tensor<?x?x4096xf16>,
//                                            tensor<4096xf16>)
//                            outs(%out : tensor<?x?x?xf16>)
//                            : tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 (dynamic) -> %d0 = tensor.dim %x, %c0
//   dim 1 (dynamic) -> %d1 = tensor.dim %x, %c1
//   dim 2 (static)  -> 4096 : index
//===----------------------------------------------------------------------===//

LogicalResult
RmsNormOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

//===----------------------------------------------------------------------===//
// QMoEOp
//
// Result shape == input data tensor's shape. Top-k expert routing happens
// inside the kernel and produces per-token outputs that are accumulated
// back into the original token slot — output rank/extents match `$input`.
// Verified against `lib/Runtime/real/qmoe.cpp`'s output buffer sizing
// (num_tokens * hidden_size * elem_size).
//
// Before:
//   %y = hip.qmoe(%ctx) ins(%x, %router, %fc1w, ... :
//                            tensor<?x?x2880xf16>, ...)
//                       outs(%out : tensor<?x?x?xf16>)
//                       : tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 (dynamic) -> %d0 = tensor.dim %x, %c0
//   dim 1 (dynamic) -> %d1 = tensor.dim %x, %c1
//   dim 2 (static)  -> 2880 : index
//===----------------------------------------------------------------------===//

LogicalResult
QMoEOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

//===----------------------------------------------------------------------===//
// MatMulNBitsOp
//
// `$A` has shape `[..., K]`. Output has shape `[..., N]` where leading
// dims are taken from `$A` and the final dim is the integer attribute
// `$N` (always static, hence resolved as `IndexAttr`). Packed-B layout
// `(N, k_blocks, blob_size)` is irrelevant to shape inference — the
// attribute carries the logical N.
//
// Before:
//   %y = hip.matmul_nbits(%ctx)
//          ins(%a, %b, %scales : tensor<?x?x2880xf16>,
//                                tensor<5120x90x16xui8>,
//                                tensor<5120x90xf16>)
//          outs(%out : tensor<?x?x?xf16>)
//          {K = 2880, N = 5120, ...} : tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 (dynamic) -> %d0 = tensor.dim %a, %c0
//   dim 1 (dynamic) -> %d1 = tensor.dim %a, %c1
//   dim 2 (static)  -> 5120 : index
//===----------------------------------------------------------------------===//

LogicalResult MatMulNBitsOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  ArrayRef<int64_t> aShape = getShapeOf(getA());
  if (aShape.empty())
    return failure();

  Location loc = getLoc();
  Value A = getA();
  size_t aRank = aShape.size();
  SmallVector<OpFoldResult> dims;
  dims.reserve(aRank);
  // Leading dims (rank-1 of them) from A; final dim is the static N attr.
  for (size_t i : llvm::seq<size_t>(0, aRank - 1))
    dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, aShape[i], A, i));
  dims.push_back(b.getIndexAttr(getN()));
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

//===----------------------------------------------------------------------===//
// GemmOp
//
// 2D output `[M, N]`:
//   M = transA ? A.shape[1] : A.shape[0]
//   N = transB ? B.shape[0] : B.shape[1]
// Optional `$input_c` is broadcast against `[M, N]` and does not
// contribute extents. transA/transB are integer attributes (0/1).
//
// Before:
//   %y = hip.gemm(%ctx) ins(%a, %b : tensor<?x256xf32>, tensor<256x?xf32>)
//                       outs(%out : tensor<?x?xf32>)
//                       {transA = 0, transB = 0, ...} : tensor<?x?xf32>
// After (reified result shape):
//   dim 0 (M) -> %dM = tensor.dim %a, %c0
//   dim 1 (N) -> %dN = tensor.dim %b, %c1
//===----------------------------------------------------------------------===//

LogicalResult
GemmOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  ArrayRef<int64_t> aShape = getShapeOf(getInputA());
  ArrayRef<int64_t> bShape = getShapeOf(getInputB());
  if (aShape.size() != 2 || bShape.size() != 2)
    return failure();

  Location loc = getLoc();
  Value A = getInputA();
  Value B = getInputB();
  bool transA = getTransA() != 0;
  bool transB = getTransB() != 0;

  size_t mDim = transA ? 1 : 0;
  size_t nDim = transB ? 0 : 1;
  SmallVector<OpFoldResult> dims;
  dims.reserve(2);
  dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, aShape[mDim], A, mDim));
  dims.push_back(mlir::hip::reifyDimOrConstant(b, loc, bShape[nDim], B, nDim));
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}
