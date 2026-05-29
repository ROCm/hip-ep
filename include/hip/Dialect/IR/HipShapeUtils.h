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

/// Compute the shape of `A @ B` for a matmul / contraction with NumPy-style
/// batch broadcast over the leading dims. Last two dims of `aShape` are
/// `[M, K]`; last two dims of `bShape` are `[K, N]`. Leading dims are
/// broadcast (right-aligned, missing dims treated as 1).
///
/// Returns the inferred shape on success. Returns an empty `SmallVector` and
/// emits a diagnostic via `emitError` on rank-, contraction-K-, or
/// batch-broadcast mismatch.
///
/// `ShapedType::kDynamic` is treated as a wildcard:
///   - K_a or K_b dynamic -> contraction match passes (result K is dropped
///     anyway).
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
inferContractionShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
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

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
