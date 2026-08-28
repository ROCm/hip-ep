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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
      aShape, bShape, [&]() { return this->emitOpError(); }, getTransA(),
      getTransB());
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
    // M dim: A[-2], or A[-1] when transA.
    if (i + 2 == outRank) {
      size_t aDim = getTransA() ? aRank - 1 : aRank - 2;
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], A, aDim));
      continue;
    }
    // N dim: B[-1], or B[-2] when transB.
    if (i + 1 == outRank) {
      size_t bDim = getTransB() ? bRank - 2 : bRank - 1;
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], B, bDim));
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
// Shape-changing ops with bespoke per-input-dim reify:
//   transpose, gather, gather_nd
//
// These ops own per-input-dim mapping. Helpers in `HipShapeUtils` walk
// the input's static shape and emit an `OpFoldResult` per output dim
// that points back at the contributing input dim (or a static
// `IndexAttr`). This is the chained-refinement case where a preceding
// op's reify already tightened the input type. Shape pieces come
// directly from static input shapes + structural attrs (perm / axis
// / batch_dims).
//
// Sibling reduction ops (reduce_sum, reduce_max, reduce_prod) follow
// the same per-input-dim pattern, but the axes-list arrives via a
// `Value` operand rather than an attr — these are wired via the
// `Hip_DpsOp_Reduction` sub-base in `HipOps.td`, which auto-emits a
// reify body that calls `mlir::hip::reifyReductionShape`. On
// non-constant axes that helper falls back to the shared `HipDpsOp`
// outs-lift default, so the reify interface always succeeds.
//
// Sibling broadcast ops (miopen.add, mul, add, min, div, equal, and,
// sub, where, less, mod) use NumPy-broadcast over their inputs and
// are wired via the `Hip_DpsOp_Broadcast` sub-base, which auto-emits
// a reify body calling `mlir::hip::reifyBroadcastShapeFor`. Each
// leaf op contributes only the list of operand getter names; no
// per-op `.cpp` thunk is needed.
//
// Other ops whose output dims are arithmetic functions of operand
// values (pad, tile, expand, slice, range) have per-op Tier-1 reify
// thunks below: each calls a dedicated `reifyPadShape` /
// `reifyTileShape` / `reifyExpandShape` / `reifySliceShape` /
// `reifyRangeShape` helper (`HipShapeUtils.cpp`) that computes the
// output shape from the INPUT operands using a fold-or-bail strategy.
// On bail (non-constant operands, dynamic input dims) the thunk falls
// back to `cast<HipDpsOp>(getOperation()).reifyResultShapes` — i.e. the
// shared `HipDpsOp` outs-lift default. This avoids emitting per-dim
// `arith.addi(tensor.dim, const)` / `arith.divsi(...)` chains that
// don't fold and would clutter the IR (particularly important for
// slice's per-axis chain and range's count-only output).
//
// Before (transpose, perm-driven mapping):
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
// Before (pad, Tier-1 with constant pads, full fold):
//   %pads = arith.constant dense<[1, 2, 1, 2]> : tensor<4xi64>
//   %p = hip.pad(%ctx) ins(%x, %pads : tensor<3x4xf16>, tensor<4xi64>)
//                      outs(%out : tensor<?x?xf16>)
//                      {mode = "constant"} : tensor<?x?xf16>
// After (reified result shape — computed from data.shape + pads):
//   dim 0 -> 6 : index    (3 + pads[0]=1 + pads[2]=1)
//   dim 1 -> 8 : index    (4 + pads[1]=2 + pads[3]=2)
//
// Before (slice, non-foldable starts -> outs-lift fallback):
//   %p = hip.slice(%ctx) ins(%data, %starts, %ends : ...)
//        outs(%out : tensor<?x4xf32>) : tensor<?x4xf32>
// After (reified result shape — outs-lift via HipDpsOp default, no IR bloat):
//   dim 0 -> tensor.dim %out, %c0   (passes through from outs)
//   dim 1 -> 4 : index              (static dim of `outs`)
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

LogicalResult
GatherOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
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

LogicalResult
GatherElementsOp::reifyResultShapes(OpBuilder &b,
                                    ReifiedRankedShapedTypeDims &reified) {
  if (getNumResults() == 0)
    return failure();
  auto indicesType = dyn_cast<RankedTensorType>(getIndices().getType());
  if (!indicesType)
    return failure();

  SmallVector<OpFoldResult> dims;
  for (auto i : llvm::seq<int64_t>(0, indicesType.getRank()))
    dims.push_back(tensor::getMixedSize(b, getLoc(), getIndices(), i));
  reified.assign({std::move(dims)});
  return success();
}

LogicalResult
OneHotOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reified) {
  if (getNumResults() == 0)
    return failure();
  auto indicesType = dyn_cast<RankedTensorType>(getIndices().getType());
  auto depthType = dyn_cast<RankedTensorType>(getDepth().getType());
  if (!indicesType || !depthType)
    return failure();

  int64_t outRank = indicesType.getRank() + 1;
  int64_t axis = getAxis();
  if (axis < 0)
    axis += outRank;

  // The one-hot axis extent is the runtime *value* of the `depth` scalar
  // (data-dependent), NOT any static dim of the `depth` tensor. A scalar
  // depth's only "dim" is its element count (always 1), so reading
  // `depth`'s shape here -- a rank-0 attr of 1, or dim(depth, 0) == 1 for a
  // single-element rank-1 export -- both wrongly report an axis extent of 1.
  // --hip-infer-shapes would then narrow the (dynamic) axis dim to a static
  // 1 and the scatter drops every index >= 1, collapsing the axis to one
  // row. Lift the axis extent from the DPS `outs` init instead: the
  // ONNX->HIP converter sizes that init to the real depth via
  // hip.readback_scalar (dynamic, so infer-shapes leaves it alone), or to a
  // static extent when the depth folded at compile time (so infer-shapes
  // narrows it correctly). Non-axis dims still come from `indices`, which
  // may carry tighter static extents than the init.
  //
  // Before (buggy): rank-0 depth -> depthDim = 1 -> infer-shapes forces
  //                 tensor<?x?x?> to tensor<?x?x1>.
  // After:          depthDim = size(outs, axis) -> stays dynamic (readback)
  //                 or narrows only to a genuine compile-time depth.
  //
  // Lift the axis extent from the init WITHOUT materializing a fresh
  // `tensor.dim` on it: read the init's static dim directly, and for a
  // dynamic axis reuse the init producer's own extent operand. A
  // materialized `tensor.dim` would add a SECOND use to the init
  // `tensor.empty`, tripping the single-use guard in `--hip-infer-shapes`
  // (refineOneResult) that gates the whole result refinement -- so the
  // static non-axis dims (from `indices`) would ALSO fail to narrow.
  Value initVal = getOutput();
  OpFoldResult axisExtent;
  auto initTy = dyn_cast<RankedTensorType>(initVal.getType());
  if (initTy && !initTy.isDynamicDim(axis))
    axisExtent = b.getIndexAttr(initTy.getDimSize(axis));
  else if (auto emptyOp = initVal.getDefiningOp<tensor::EmptyOp>())
    axisExtent = emptyOp.getMixedSizes()[axis];
  else
    axisExtent = tensor::getMixedSize(b, getLoc(), getResult(0), axis);

  SmallVector<OpFoldResult> dims;
  int64_t inDim = 0;
  for (int64_t outDim : llvm::seq<int64_t>(0, outRank)) {
    if (outDim == axis)
      dims.push_back(axisExtent);
    else
      dims.push_back(tensor::getMixedSize(b, getLoc(), getIndices(), inDim++));
  }
  reified.assign({std::move(dims)});
  return success();
}

LogicalResult
CompressOp::reifyResultShapes(OpBuilder &b,
                              ReifiedRankedShapedTypeDims &reified) {
  if (getNumResults() == 0)
    return failure();
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto conditionType = dyn_cast<RankedTensorType>(getCondition().getType());
  if (!inputType || !conditionType || conditionType.getRank() != 1)
    return failure();

  int64_t rank = inputType.getRank();
  int64_t axis = getAxis();
  if (axis < 0)
    axis += rank;
  // The flattened result is rank-1, so the selection always shrinks dim 0.
  int64_t selectDim = getFlatten() ? 0 : axis;

  // The selected extent is the number of TRUE entries in `condition`, which is
  // data-dependent -- the condition LENGTH is only its upper bound. Reporting
  // that upper bound here would let --hip-infer-shapes narrow the result to
  // the padded extent and would size a graph output's hip.alloc_output above
  // the row count the consumer expects. Lift it from the DPS `outs` init
  // instead: the ONNX->HIP converter sizes that init to the true count via
  // hip.readback_dim (dynamic, so infer-shapes leaves it alone), or to a
  // static extent when ONNX shape inference already knew it.
  //
  // Read the init's extent WITHOUT materializing a fresh `tensor.dim` on it:
  // that would add a second use to the init `tensor.empty` and trip the
  // single-use guard in `--hip-infer-shapes` (refineOneResult), which gates
  // the whole result refinement -- so the static non-selection dims would
  // also stop narrowing.
  Value initVal = getOutput();
  OpFoldResult selectedExtent;
  auto initTy = dyn_cast<RankedTensorType>(initVal.getType());
  if (initTy && !initTy.isDynamicDim(selectDim))
    selectedExtent = b.getIndexAttr(initTy.getDimSize(selectDim));
  else if (auto emptyOp = initVal.getDefiningOp<tensor::EmptyOp>())
    selectedExtent = emptyOp.getMixedSizes()[selectDim];
  else
    // Not an empty, so a `tensor.dim` here cannot trip that guard.
    selectedExtent = tensor::getMixedSize(b, getLoc(), initVal, selectDim);

  if (getFlatten()) {
    reified.assign({SmallVector<OpFoldResult>{selectedExtent}});
    return success();
  }

  SmallVector<OpFoldResult> dims;
  for (int64_t i : llvm::seq<int64_t>(0, rank)) {
    if (i == axis)
      dims.push_back(selectedExtent);
    else
      dims.push_back(tensor::getMixedSize(b, getLoc(), getInput(), i));
  }
  reified.assign({std::move(dims)});
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

// Per-op Tier-1 thunks for pad / tile / expand / slice / range. Each
// calls its dedicated `reify*Shape` helper (HipShapeUtils.cpp); on
// failure it falls back to the shared `HipDpsOp::reifyResultShapes`
// default body in `HipDpsOpInterface.cpp`, which walks `getDpsInits()`
// and lifts each init's runtime shape via `tensor::getMixedSizes` /
// `memref::getMixedSizes`. The fallback dispatches through the
// `HipDpsOp` interface concept and resolves to the default body (these
// ops override only `ReifyRankedShapedTypeOpInterface::reifyResultShapes`
// — the body of THIS function — and do not override the `HipDpsOp`
// interface method, so the call below does not recurse).
LogicalResult
PadOp::reifyResultShapes(OpBuilder &b,
                         ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifyPadShape(b, getLoc(), getData(), getPads(),
                                         getAxes(), dims))) {
    reifiedReturnShapes.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

LogicalResult
TileOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifyTileShape(b, getLoc(), getInput(), getRepeats(),
                                          dims))) {
    reifiedReturnShapes.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

LogicalResult
ExpandOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifyExpandShape(b, getLoc(), getInput(), getShape(),
                                            dims))) {
    reifiedReturnShapes.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

LogicalResult
SliceOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifySliceShape(b, getLoc(), getData(), getStarts(),
                                           getEnds(), getAxes(), getSteps(),
                                           dims))) {
    reifiedReturnShapes.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

LogicalResult
RangeOp::reifyResultShapes(OpBuilder &b,
                           ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  SmallVector<OpFoldResult> dims;
  if (succeeded(mlir::hip::reifyRangeShape(b, getLoc(), getStart(), getLimit(),
                                           getDelta(), dims))) {
    reifiedReturnShapes.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// TopKOp
//
// Two DPS results (values, indices) share the same extents; the axis dim is K.
// Lift each init's runtime shape via the shared HipDpsOp default body.
//
// Before:
//   %v, %idx = hip.top_k(%ctx) ins(%x, %k : ...)
//                            outs(%values, %indices : ...)
// After (reified result shapes):
//   values dim i  -> tensor.dim %values, %ci  (or memref.dim)
//   indices dim i -> tensor.dim %indices, %ci
//===----------------------------------------------------------------------===//

LogicalResult
TopKOp::reifyResultShapes(OpBuilder &b,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure();
  return cast<HipDpsOp>(getOperation())
      .reifyResultShapes(b, reifiedReturnShapes);
}
