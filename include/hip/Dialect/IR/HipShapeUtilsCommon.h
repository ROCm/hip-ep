/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_COMMON_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_COMMON_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hip {

/// Static grouping information for `hip.readback_control`. Sources are
/// flattened in operand-major order; `resultOffsets[i]` is the first i64 result
/// produced for source `i`, and the final entry equals `totalCount`.
struct ReadbackControlLayout {
  SmallVector<int64_t> sourceLengths;
  SmallVector<int64_t> resultOffsets;
  int64_t totalCount = 0;
};

/// Validate rank-0/rank-1 statically-sized i32/i64 source types and compute the
/// operand-major result grouping used by conversion, verification, and
/// lowering. At least one source is required.
FailureOr<ReadbackControlLayout>
getReadbackControlLayout(TypeRange sourceTypes);

/// Parse the payload of a rank-0 or rank-1 dense integer tensor into signed
/// i64 values. `expectedRank` may restrict callers to scalar or vector form.
/// This is the dialect-layer parsing core shared by inline constant matching
/// and ONNX conversion's broader pre-/post-externalization recognition.
bool parseDenseIntElements(DenseElementsAttr dense,
                           SmallVectorImpl<int64_t> &out,
                           std::optional<int64_t> expectedRank = std::nullopt);

/// Match a rank-0/rank-1 integer constant and parse it with
/// `parseDenseIntElements`. Recognized structural sources are an inline
/// `arith.constant` tensor, a dense-value `hip.constant` carrier whose
/// attribute type exactly matches its result type, and a `memref.get_global`
/// referencing a constant global with a dense initializer. Location-only
/// `hip.constant` carriers are deliberately not treated as payload values. The
/// global form preserves constant provenance across tensor-to-memref
/// bufferization. Generic HIP dialect code does not inspect frontend
/// operations.
bool matchConstantIntTensor(Value value, SmallVectorImpl<int64_t> &out,
                            std::optional<int64_t> expectedRank = std::nullopt);

//===----------------------------------------------------------------------===//
// Contract shared by every shape helper
//
// The `infer*` helpers are pure functions of static shapes: they take no
// builder and emit no IR. The `reify*` helpers may materialize index SSA, and
// each one validates every precondition through its `infer*` counterpart
// BEFORE touching the builder. A `reify*` failure therefore never leaves
// stray ops behind, which both the pattern-rewrite contract and
// `ReifyRankedShapedTypeOpInterface` require (see
// `GreedyPatternRewriteDriver`'s expensive checks and
// `ResolveShapedTypeResultDims`).
//
// Converter destination construction and `reifyResultShapes` call the same
// helper for a given op, so the DPS `outs` shape and the shape observed by
// consumers cannot disagree. See `docs/design/hip-shape-inference.md`.
//===----------------------------------------------------------------------===//

/// Verify that one `outs` shape of a DPS HIP op matches the shape returned by
/// `inferShape`. `op` must implement `DestinationStyleOpInterface`;
/// `initIndex` selects the destination (zero for single-destination ops).
///
/// `inferShape` is invoked once and returns `failure()` when the underlying
/// `infer*` helper already emitted a diagnostic, which this function
/// propagates without re-emitting. Pair it directly with an `infer*` helper:
///
///   return verifyHipOpShape(*this, [&] {
///     return inferMatmulShape(aShape, bShape,
///                             [&] { return this->emitOpError(); });
///   });
///
/// Element-type checks are intentionally not handled here: dtype-changing
/// ops (cast, equal, less, not, and) keep their own element-type checks in
/// their op-local verifiers.
/// Verify one DPS destination against a pure `infer*` shape rule.
LogicalResult
verifyHipOpShape(Operation *op,
                 function_ref<FailureOr<SmallVector<int64_t>>()> inferShape,
                 unsigned initIndex = 0);

/// Verify the cross-cutting HIP DPS compute contract:
///   * every listed data operand is a ranked tensor or memref;
///   * all listed operands use the same tensor/memref mode;
///   * the DestinationStyleOpInterface init count equals `numInits`;
///   * tensor mode has one result per init with exactly matching types;
///   * memref mode has no SSA results.
///
/// `dataOperands` must include the DPS init operands as well as the semantic
/// inputs. Keeping this helper public lets generated TableGen verifier bodies
/// compose the same structural contract as dedicated op verifiers.
LogicalResult verifyDpsComputeOp(Operation *op, ArrayRef<Value> dataOperands,
                                 unsigned numInits);

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
/// Returns failure when `source` is not a ranked tensor. The `FailureOr`
/// distinguishes that failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source);

/// One-shot reify body for a single-result DPS op whose result shape equals
/// one named source operand. This deliberately does not inspect operand order
/// or lift the DPS init: `source` is the semantic shape authority selected by
/// the op's TableGen definition.
LogicalResult
reifyElementwiseSameShapeFor(OpBuilder &b, Location loc, Value source,
                             Operation *op,
                             ReifiedRankedShapedTypeDims &reified);

/// Verify the single-result structural DPS contract, then verify that DPS init
/// `initIndex` has a shape statically compatible with the named semantic
/// `source`. Dynamic extents on either side are wildcards. All ranked
/// tensor/memref DPS inputs participate in the structural check, including
/// non-source operands; non-shaped inputs such as !hip.context are ignored.
///
/// This is intentionally named-source based rather than first-operand based:
/// ops such as CumSum and Scatter carry other shaped inputs that must not
/// influence the result shape.
LogicalResult verifySameShapeDpsOp(Operation *op, Value source,
                                   unsigned initIndex = 0);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_COMMON_H
