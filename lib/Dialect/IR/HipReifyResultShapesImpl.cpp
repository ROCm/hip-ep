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

//===----------------------------------------------------------------------===//
// SiluOp / SigmoidOp / SoftplusOp / GeluOp / ReciprocalOp / SqrtOp /
// NotOp / CosOp / SinOp / NegOp / CastOp / SignOp / CumSumOp / ScatterNDOp
//
// Same-shape ops: result shape == primary input tensor's shape. CumSum /
// CastOp / Sign / etc. all preserve the input rank/extents — only the
// element type (Cast) or values (everything else) changes. ScatterND
// preserves `$data`'s shape (writes happen back into the same
// rank/extents).
//
// Before:
//   %y = hip.<op>(%ctx) ins(%x : tensor<?x?x4096xf16>)
//                       outs(%out : tensor<?x?x?xf16>) : tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 (dynamic) -> %d0 = tensor.dim %x, %c0
//   dim 1 (dynamic) -> %d1 = tensor.dim %x, %c1
//   dim 2 (static)  -> 4096 : index
//===----------------------------------------------------------------------===//

LogicalResult
SiluOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

LogicalResult
SigmoidOp::reifyResultShapes(OpBuilder &b,
                             ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult SoftplusOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
GeluOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

LogicalResult ReciprocalOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
SqrtOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
NotOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
CosOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
SinOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
NegOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult
CastOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getInput())});
  return success();
}

LogicalResult
SignOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult CumSumOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getX().getType()))
    return failure();
  // cumsum reads `$x` along `$axis`; the output rank/extents match `$x` —
  // the axis operand only selects WHICH dim is accumulated, not how big
  // the result is.
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getX())});
  return success();
}

LogicalResult ScatterNDOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getData().getType()))
    return failure();
  // Output shape == `$data` shape. ScatterND copies `$data` into output
  // and overwrites entries at `$indices`; rank/extents are preserved.
  reifiedReturnShapes.assign(
      {mlir::hip::reifyElementwiseSameShape(b, getLoc(), getData())});
  return success();
}

//===----------------------------------------------------------------------===//
// SizeOp
//
// Result is rank-0 (scalar i64 element count). Reify returns one outer
// shape entry with an empty inner dim list — there are no dims to walk.
// `$x`'s shape is irrelevant to the result shape; the runtime computes
// `prod(x.shape)` into the rank-0 output buffer.
//
// Before:
//   %y = hip.size(%ctx) ins(%x : tensor<?x?xf16>)
//                       outs(%out : tensor<i64>) : tensor<i64>
// After (reified result shape):
//   rank-0 -> {} (empty dim list)
//===----------------------------------------------------------------------===//

LogicalResult
SizeOp::reifyResultShapes(OpBuilder & /*b*/,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  reifiedReturnShapes.assign({SmallVector<OpFoldResult>{}});
  return success();
}

//===----------------------------------------------------------------------===//
// Broadcast elementwise ops: miopen.add, mul, add, min, div, equal, and,
// sub, where, less, mod
//
// All these ops take 2-3 broadcast-compatible input operands and write
// their NumPy-broadcast result into `outs`. The output rank equals the
// max input rank and each output dim is determined by whichever operand
// contributes a non-1 value at that position (right-aligned).
// `reifyBroadcastShape` lifts each output dim to an `OpFoldResult`:
// static dims become `IndexAttr`; dynamic dims become `tensor.dim`
// against whichever operand actually contributes the runtime extent.
//
// Output element type is independent of this reify path — comparisons
// (equal, less) emit i1 outs while their operands are typically f32/f16,
// and `where` mixes an i1 condition with f32/f16 x/y. Reify only walks
// shapes; the dtype lives on `outs` and is set by the converter.
//
// Before (typical canonical case):
//   %add = hip.add(%ctx) ins(%lhs, %rhs : tensor<1x?x4096xf16>,
//                                          tensor<?x1x4096xf16>)
//                        outs(%out : tensor<?x?x4096xf16>) : tensor<?x?x4096xf16>
//   %d0 = tensor.dim %add, %c0
//   %d1 = tensor.dim %add, %c1
// After (reified, then folded by tensor.dim folder):
//   dim 0 dynamic -> tensor.dim %rhs, %c0  (rhs.shape[0]=? != 1, canonical)
//   dim 1 dynamic -> tensor.dim %lhs, %c1  (lhs.shape[1]=? != 1, canonical)
//   dim 2 static  -> arith.constant 4096
//
// Before (3-operand `where`):
//   %sel = hip.where(%ctx) ins(%cond, %x, %y : tensor<1x4xi1>,
//                                              tensor<2x1xf32>,
//                                              tensor<2x4xf32>)
//                          outs(%out : tensor<2x4xf32>) : tensor<2x4xf32>
// After (reified result shape):
//   dim 0 static -> 2 (canonical: from %x or %y)
//   dim 1 static -> 4 (canonical: from %cond or %y)
//===----------------------------------------------------------------------===//

namespace {

// Common shape: every broadcast op below differs only in (a) the input
// operand list and (b) which getter names that list. Centralizing the
// guard + `reifyBroadcastShape` call keeps each op's reify thunk a
// one-liner against a `ValueRange` slice.
LogicalResult
reifyBroadcastShapeFor(OpBuilder &b, Location loc, ValueRange operands,
                       Operation *op,
                       ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (op->getNumResults() == 0)
    return failure();
  for (Value v : operands)
    if (!isa<RankedTensorType>(v.getType()))
      return failure();
  SmallVector<OpFoldResult> dims =
      mlir::hip::reifyBroadcastShape(b, loc, operands);
  if (dims.empty())
    return failure();
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

} // namespace

LogicalResult MiopenAddOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getA(), getB()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
MulOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
AddOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
MinOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
DivOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
EqualOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
AndOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
SubOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
WhereOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(),
                                {getCondition(), getX(), getY()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
LessOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

LogicalResult
ModOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyBroadcastShapeFor(b, getLoc(), {getLhs(), getRhs()}, *this,
                                reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// Shape-changing ops (PR #263 commit 3):
// transpose, gather, gather_nd, reduce_sum, reduce_max, reduce_prod,
// pad, tile, expand, slice, range
//
// Two-tier strategy:
//
//  Tier 1 — real per-input-dim mapping. Helpers in `HipShapeUtils`
//  walk the input's static shape and emit an `OpFoldResult` per output
//  dim that points back at the contributing input dim (or a static
//  `IndexAttr`). This is the chained-refinement case where a
//  preceding op's reify already tightened the input type.
//    transpose, gather, gather_nd: shape pieces come directly from
//    static input shapes + structural attrs (perm / axis / batch_dims).
//    reduce_sum/max/prod: same idea, but the axes-list arrives via a
//    `Value` operand rather than an attr; the helper introspects it as
//    an `arith.constant` (which is what the OnnxToHip converter
//    materialises). On non-constant axes the op falls through to the
//    Tier-2 fallback so the reify interface still succeeds.
//
//  Tier 2 — no-op fallback. For ops whose output dims are arithmetic
//  functions of operand values (pad: data+pads_begin+pads_end; tile:
//  input*repeats; expand: max(input, target_shape); slice:
//  per-axis slice arithmetic; range: ceil((limit-start)/delta)),
//  reify lifts the DPS `outs` operand's own shape via
//  `reifyElementwiseSameShape(getOutput())`. Static dims become
//  `IndexAttr` (the common case after ONNX shape inference); dynamic
//  dims emit `tensor.dim %output, %i` — a no-op tightening that
//  doesn't add signal beyond what's already on the result type, but
//  keeps every Hip_DpsOp wired with a non-failing reify implementation
//  (the contract a future backward dim-origin composer relies on).
//  This mirrors the upstream IREE convention: `LinalgExtOp`'s default
//  `reifyResultShapes` walks `getDpsInits()` and lifts each output's
//  shape — the same fallback we use here for ops without a specialized
//  arithmetic helper. Proper per-op arithmetic for these five (e.g.
//  `tensor::PadOp`'s `affine.apply (d0 + d1 + d2)` recipe) is deferred
//  to a follow-up PR.
//
// Before (transpose, perm-driven mapping, Tier 1):
//   %t = hip.transpose(%ctx) ins(%x : tensor<2x?x4096xf16>)
//                            outs(%out : tensor<?x?x?xf16>)
//                            {perm = [2, 0, 1]} : tensor<?x?x?xf16>
// After (reified result shape):
//   dim 0 -> 4096 : index            (static, from %x.shape[2])
//   dim 1 -> 2 : index               (static, from %x.shape[0])
//   dim 2 -> tensor.dim %x, %c1      (dynamic, from %x.shape[1])
//
// Before (reduce_sum with constant axes, keepdims=1, Tier 1):
//   %a = arith.constant dense<[1]> : tensor<1xi64>
//   %r = hip.reduce_sum(%ctx) ins(%x, %a : tensor<?x4096xf16>,
//                                            tensor<1xi64>)
//                              outs(%out : tensor<?x?xf16>)
//                              {keepdims = 1, noop_with_empty_axes = 0}
//                            : tensor<?x?xf16>
// After (reified result shape):
//   dim 0 -> tensor.dim %x, %c0     (passes through from input)
//   dim 1 -> 1 : index              (axes-listed dim → keepdims=1 → 1)
//
// Before (pad, Tier-2 no-op fallback — pad arithmetic deferred):
//   %p = hip.pad(%ctx) ins(%x, %pads : tensor<?x4096xf16>,
//                                       tensor<4xi64>)
//                      outs(%out : tensor<?x4100xf16>)
//                      {mode = "constant"} : tensor<?x4100xf16>
// After (reified result shape — lifted from `%out`, not computed):
//   dim 0 -> tensor.dim %out, %c0   (dynamic dim of `outs`; folds away
//                                    if a downstream consumer already
//                                    knows the result type)
//   dim 1 -> 4100 : index           (static dim of `outs`)
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getInput().getType()))
    return failure();

  // Decode the I64ArrayAttr `perm`. Verifier already rejects non-int
  // entries; we still bail defensively if any entry is not an IntegerAttr.
  SmallVector<int64_t> perm;
  perm.reserve(getPerm().size());
  for (Attribute a : getPerm()) {
    auto ia = dyn_cast<IntegerAttr>(a);
    if (!ia)
      return failure();
    perm.push_back(ia.getInt());
  }

  SmallVector<OpFoldResult> dims =
      mlir::hip::reifyTransposeByPerm(b, getLoc(), getInput(), perm);
  if (dims.empty())
    return failure();
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

LogicalResult GatherOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getData().getType()) ||
      !isa<RankedTensorType>(getIndices().getType()))
    return failure();

  SmallVector<OpFoldResult> dims = mlir::hip::reifyGatherWithAxis(
      b, getLoc(), getData(), getIndices(), getAxis());
  if (dims.empty())
    return failure();
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

LogicalResult GatherNDOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(getData().getType()) ||
      !isa<RankedTensorType>(getIndices().getType()))
    return failure();

  SmallVector<OpFoldResult> dims = mlir::hip::reifyGatherND(
      b, getLoc(), getData(), getIndices(), getBatchDims());
  if (dims.empty())
    return failure();
  reifiedReturnShapes.assign({std::move(dims)});
  return success();
}

namespace {

// Shared Tier-2 fallback for shape-changing ops without per-op
// arithmetic helpers: just lift the DPS `outs` operand's own shape.
// Static dims fall out as `IndexAttr`; dynamic dims emit
// `tensor.dim %output, %i` — a no-op tightening that nonetheless
// keeps the reify interface non-failing for downstream consumers.
LogicalResult reifyDpsOutShape(OpBuilder &b, Location loc, Value output,
                               Operation *op,
                               ReifiedRankedShapedTypeDims &out) {
  if (op->getNumResults() == 0)
    return failure();
  auto outputType = dyn_cast<RankedTensorType>(output.getType());
  if (!outputType)
    return failure();
  SmallVector<OpFoldResult> dims =
      mlir::hip::reifyElementwiseSameShape(b, loc, output);
  // `reifyElementwiseSameShape` returns empty for rank-0 too; rank-0 is
  // a valid result so don't conflate it with bail.
  if (dims.empty() && outputType.getRank() != 0)
    return failure();
  out.assign({std::move(dims)});
  return success();
}

// Shared reduction reify body. Tries `reifyReductionWithKeepdims`; if
// that fails (axes is not a recognised constant), falls back to the
// Tier-2 `reifyDpsOutShape` so the reify interface still succeeds.
LogicalResult reifyReductionShape(OpBuilder &b, Location loc, Value data,
                                  Value axes, int64_t keepdims,
                                  int64_t noopWithEmptyAxes, Value output,
                                  Operation *op,
                                  ReifiedRankedShapedTypeDims &out) {
  if (op->getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(data.getType()) ||
      !isa<RankedTensorType>(output.getType()))
    return failure();

  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifyReductionWithKeepdims(
          b, loc, data, axes, keepdims, noopWithEmptyAxes, dims))) {
    out.assign({std::move(dims)});
    return success();
  }
  return reifyDpsOutShape(b, loc, output, op, out);
}

} // namespace

LogicalResult ReduceSumOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyReductionShape(b, getLoc(), getData(), getAxes(), getKeepdims(),
                             getNoopWithEmptyAxes(), getOutput(), *this,
                             reifiedReturnShapes);
}

LogicalResult ReduceMaxOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyReductionShape(b, getLoc(), getData(), getAxes(), getKeepdims(),
                             getNoopWithEmptyAxes(), getOutput(), *this,
                             reifiedReturnShapes);
}

LogicalResult ReduceProdOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyReductionShape(b, getLoc(), getData(), getAxes(), getKeepdims(),
                             getNoopWithEmptyAxes(), getOutput(), *this,
                             reifiedReturnShapes);
}

LogicalResult
PadOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyDpsOutShape(b, getLoc(), getOutput(), *this, reifiedReturnShapes);
}

LogicalResult
TileOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyDpsOutShape(b, getLoc(), getOutput(), *this, reifiedReturnShapes);
}

LogicalResult ExpandOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyDpsOutShape(b, getLoc(), getOutput(), *this, reifiedReturnShapes);
}

LogicalResult
SliceOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyDpsOutShape(b, getLoc(), getOutput(), *this, reifiedReturnShapes);
}

LogicalResult
RangeOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  return reifyDpsOutShape(b, getLoc(), getOutput(), *this, reifiedReturnShapes);
}
