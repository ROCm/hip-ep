/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dims. Last two dims of `aShape` are `[M, K]`; last two
/// dims of `bShape` are `[K, N]`. Leading dims are broadcast (right-aligned,
/// missing dims treated as 1).
///
/// Returns the inferred shape on success. Returns an empty `SmallVector` and
/// emits a diagnostic via `emitError` on rank-, K-, or batch-broadcast
/// mismatch.
///
/// `ShapedType::kDynamic` is treated as a wildcard:
///   - K_a or K_b dynamic -> K match passes (result K is dropped anyway).
///   - Batch dim broadcast follows NumPy / TF / ONNX MatMul semantics
///     (delegated to `mlir::OpTrait::util::getBroadcastedShape`):
///       * 1 broadcasts against any dim.
///       * dynamic + static>1 -> static (the dynamic side must be 1 or
///         match the static side at runtime per the broadcast contract;
///         taking the static side is the strictly-correct tightening).
///       * dynamic + dynamic -> dynamic.
///       * static + static, equal -> static; unequal and neither is 1
///         -> error.
SmallVector<int64_t>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError);

/// Verify that the actual `outs` operand shapes of a DPS HIP op match the
/// shapes returned by `computeExpected`. `op` must implement
/// `DestinationStyleOpInterface`.
///
/// `computeExpected` is invoked once and must return one shape per init
/// operand (== one per `OpResult` for tensor mode; same count for memref
/// mode, just no SSA result). An empty outer vector signals that the
/// shape-arithmetic helper already emitted a diagnostic — this function
/// returns `failure()` without re-emitting.
///
/// Element-type checks are intentionally not handled here: dtype-changing
/// ops (cast, equal, less, not, and) keep their own element-type checks in
/// their op-local verifiers.
LogicalResult verifyHipOpShape(
    Operation *op,
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected);

/// Build an `OpFoldResult` for one dimension of a reify-callable op's
/// result:
///   - if `staticDim` is not `kDynamic`, returns `b.getIndexAttr(staticDim)`
///     (no IR emitted).
///   - otherwise emits `tensor.dim` against `source` at `sourceDim`. The
///     dim op folds to an `arith.constant` automatically when
///     `source.getType()` has a static size at `sourceDim`.
///
/// `source` is required to have `RankedTensorType` -- the
/// `ReifyRankedShapedTypeOpInterface` contract restricts its callers to
/// ops with tensor results, so memref-typed sources cannot reach a reify
/// path today.
OpFoldResult reifyDimOrConstant(OpBuilder &b, Location loc, int64_t staticDim,
                                Value source, int64_t sourceDim);

/// Reify the result shape of a shape-preserving DPS op as the runtime
/// shape of `source`: each static dim becomes `IndexAttr`, each dynamic
/// dim becomes `tensor.dim %source, %i`. Used by ops whose result has
/// the same shape as one designated input (e.g. rope, rms_norm, qmoe).
///
/// `source` must be a `RankedTensorType`-typed Value -- this helper is
/// called from `reifyResultShapes` impls, which are invoked only in
/// tensor mode per the interface contract.
SmallVector<OpFoldResult> reifyElementwiseSameShape(OpBuilder &b, Location loc,
                                                    Value source);

/// Compute the NumPy-broadcast result shape over `operands` and lift each
/// output dim to an `OpFoldResult`. Static result dims become `IndexAttr`
/// (no IR emitted); dynamic result dims become `tensor.dim` against
/// whichever operand contributes the runtime extent — right-aligned, and
/// preferring the canonical side (in-range and != 1) when multiple
/// operands could contribute. The canonical-side preference matches the
/// batch-dim contract in `MatmulOp::reifyResultShapes` and ensures that
/// a future `tensor.dim` of the result folds back to the operand that
/// actually determines the size at runtime.
///
/// Used by elementwise ops that take broadcast-shape operands and write
/// the broadcast result into their `outs` (add, mul, sub, div, min, mod,
/// equal, less, and, where, ...). The output dtype is taken from the
/// op's `outs` operand and is independent of this helper — comparisons
/// (equal, less) emit i1 outs while the operands are typically f32/f16,
/// and the helper handles both cases identically (it only looks at
/// shapes).
///
/// All operands must be `RankedTensorType`-typed Values (the interface
/// contract for `reifyResultShapes` callers). Returns an empty vector
/// if broadcast fails — verifiers should already have caught this, but
/// reify bails defensively to avoid materializing nonsense IR.
SmallVector<OpFoldResult>
reifyBroadcastShape(OpBuilder &b, Location loc, ValueRange operands);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
