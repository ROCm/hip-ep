/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_H

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

/// Parse the payload of a rank-0 or rank-1 dense integer tensor into signed
/// i64 values. `expectedRank` may restrict callers to scalar or vector form.
/// This is the dialect-layer parsing core shared by inline constant matching
/// and ONNX conversion's broader pre-/post-externalization recognition.
bool parseDenseIntElements(DenseElementsAttr dense,
                           SmallVectorImpl<int64_t> &out,
                           std::optional<int64_t> expectedRank = std::nullopt);

/// Match an inline `arith.constant` rank-0/rank-1 integer tensor and parse it
/// with `parseDenseIntElements`. Generic HIP dialect code intentionally does
/// not inspect ONNX ops or conversion-side externalized globals.
bool matchConstantIntTensor(Value value, SmallVectorImpl<int64_t> &out,
                            std::optional<int64_t> expectedRank = std::nullopt);

//===----------------------------------------------------------------------===//
// Contract shared by every helper in this header
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

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dimensions. The matrix dimensions are `A[..., M, K]` and
/// `B[..., K, N]`.
///
/// Dynamic contraction dimensions are treated as compatible. Batch dimensions
/// use `OpTrait::util::getBroadcastedShape`. On failure, emits a diagnostic
/// through `emitError`.
FailureOr<SmallVector<int64_t>>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError);

/// Verify that MatMul's broadcasted batches are representable by one constant
/// strided-batch offset per operand. A stride can only express "one matrix
/// broadcast across every output batch" (stride 0) or "one matrix per output
/// batch" (stride == matrix size), so an operand is rejected only when it
/// provably needs something in between: a partial broadcast that pads some
/// batch axes up to the output extent while carrying batches on others.
///
/// This helper rejects layouts that are provably partial from static types.
/// Dynamic extents can conceal a partial broadcast, so the HIP-to-LLVM lowering
/// also passes each operand's runtime batch count to the wrapper. The wrapper
/// dispatches only when that count is 1 or equals the output batch count.
LogicalResult
verifyStridedBatchMatmul(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                         function_ref<InFlightDiagnostic()> emitError);

/// Compute ONNX Gemm's rank-2 `{M, N}` result shape from static extents.
/// Validates that A and B are rank 2, that `transA`/`transB` are 0 or 1, that
/// the transpose-aware contraction extents agree, and that the optional C is
/// unidirectionally broadcastable onto `{M, N}`. `cShape` is `std::nullopt`
/// when C is absent; C never contributes M or N.
FailureOr<SmallVector<int64_t>>
inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
               std::optional<ArrayRef<int64_t>> cShape, int64_t transA,
               int64_t transB, function_ref<InFlightDiagnostic()> emitError);

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

/// Pure NumPy right-aligned broadcast over static shapes. Dynamic dimensions
/// are honest wildcards. This is the common static rule used by conversion,
/// reification, and verification.
FailureOr<SmallVector<int64_t>>
inferBroadcastShape(ArrayRef<ArrayRef<int64_t>> shapes,
                    function_ref<InFlightDiagnostic()> emitError);

/// Generated-verifier target for `Hip_DpsOp_Broadcast`.
LogicalResult verifyBroadcastDpsOp(Operation *op, ValueRange operands);

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

/// Compute the NumPy-broadcast result shape over ranked tensor `operands`,
/// using `tensor::getMixedSizes` for each. Static extents remain `IndexAttr`;
/// dynamic/dynamic pairs materialize the runtime broadcast rule as index SSA.
///
/// Before (choosing either dynamic operand is incorrect when it is 1):
///   %lhs_dim = tensor.dim %lhs, %c0
///   %init = tensor.empty(%lhs_dim) : tensor<?xf32>
/// After:
///   %lhs_dim = tensor.dim %lhs, %c0
///   %rhs_dim = tensor.dim %rhs, %c0
///   %lhs_is_one = arith.cmpi eq, %lhs_dim, %c1 : index
///   %extent = arith.select %lhs_is_one, %rhs_dim, %lhs_dim : index
///   %init = tensor.empty(%extent) : tensor<?xf32>
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
///
/// Used by elementwise ops that take broadcast-shape operands and write
/// the broadcast result into their `outs` (add, mul, sub, div, min, mod,
/// equal, less, and, where, ...). The output dtype is taken from the
/// op's `outs` operand and is independent of this helper — comparisons
/// (equal, less) emit i1 outs while the operands are typically f32/f16,
/// and the helper handles both cases identically (it only looks at
/// shapes).
///
/// All operands must be `RankedTensorType`-typed Values.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastResultShape(OpBuilder &b, Location loc, ValueRange operands,
                          function_ref<InFlightDiagnostic()> emitError);

/// Reify ONNX MatMul's result shape: broadcast the leading batch dimensions,
/// append M from `A[-2]`, and append N from `B[-1]`. Rank-1 MatMul is outside
/// the current HIP op contract.
FailureOr<SmallVector<OpFoldResult>>
reifyMatmulResultShape(OpBuilder &b, Location loc, Value A, Value B,
                       function_ref<InFlightDiagnostic()> emitError);

/// Pure MatMulNBits result shape: A's leading dimensions followed by N.
FailureOr<SmallVector<int64_t>> inferMatMulNBitsShape(ArrayRef<int64_t> aShape,
                                                      int64_t N);

/// Reify MatMulNBits' result shape: `A`'s leading dimensions followed by the
/// static `N` attribute. The quantized weight is stored transposed, so the
/// contraction dimension never appears in the result -- which is why the last
/// dimension comes from `N` and not from `A`.
///
/// `A` must be a `RankedTensorType`-typed Value of rank at least 1.
FailureOr<SmallVector<OpFoldResult>>
reifyMatMulNBitsResultShape(OpBuilder &b, Location loc, Value A, int64_t N);

/// Reify ONNX Gemm's rank-2 `{M, N}` result using transpose-aware dimensions.
/// Optional C is checked for static unidirectional broadcast compatibility but
/// never supplies M or N.
FailureOr<SmallVector<OpFoldResult>>
reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                     Value optionalC, int64_t transA, int64_t transB,
                     function_ref<InFlightDiagnostic()> emitError);

/// Compute the ONNX forward-convolution result shape for rank-3 NCL or rank-4
/// NCHW operands. The result is `{input[0], weights[0], spatial...}` and each
/// spatial extent follows the signed-floor window formula.
///
/// An empty `kernelShape` means the ONNX attribute was omitted. In that case
/// the kernel is derived from the static spatial dimensions of `weightShape`;
/// dynamic weight spatial dimensions require an explicit kernel.
FailureOr<SmallVector<int64_t>>
inferConvShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
               ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
               ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
               int64_t group, function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvShape`. Validates through the static helper
/// before materializing any dimension arithmetic.
FailureOr<SmallVector<OpFoldResult>>
reifyConvResultShape(OpBuilder &b, Location loc, Value input, Value weights,
                     ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
                     int64_t group,
                     function_ref<InFlightDiagnostic()> emitError);

/// Compute an ONNX MaxPool/AveragePool/LpPool result shape for spatial rank
/// 1..3. N/C pass through from the input; spatial extents use signed floor
/// division, or signed ceil division when `ceilMode` is 1.
FailureOr<SmallVector<int64_t>>
inferPoolShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> kernelShape,
               ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
               ArrayRef<int64_t> dilations, int64_t ceilMode,
               function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferPoolShape`. The returned shape is shared by the
/// values result and optional MaxPool indices result.
FailureOr<SmallVector<OpFoldResult>>
reifyPoolResultShape(OpBuilder &b, Location loc, Value input,
                     ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
                     int64_t ceilMode,
                     function_ref<InFlightDiagnostic()> emitError);

/// Compute the supported rank-4 NCHW ONNX ConvTranspose result shape:
///   N = input[0]
///   M = weights[1] * group
///   spatial[i] = stride[i] * (input[i+2] - 1) + output_padding[i]
///                + ((kernel[i] - 1) * dilation[i] + 1)
///                - pads[i] - pads[i+2]
///
/// Dynamic input/weight dimensions remain dynamic. All attribute vectors are
/// validated before a result is produced.
FailureOr<SmallVector<int64_t>> inferConvTransposeShape(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> outputPadding, int64_t group,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvTransposeShape`. Validates through the static
/// helper before materializing any dimension arithmetic.
FailureOr<SmallVector<OpFoldResult>>
reifyConvTransposeResultShape(OpBuilder &b, Location loc, Value input,
                              Value weights, ArrayRef<int64_t> kernelShape,
                              ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
                              ArrayRef<int64_t> dilations,
                              ArrayRef<int64_t> outputPadding, int64_t group,
                              function_ref<InFlightDiagnostic()> emitError);

/// GlobalPool shape: preserve N/C and replace every spatial extent with 1.
FailureOr<SmallVector<int64_t>>
inferGlobalPoolShape(ArrayRef<int64_t> inputShape,
                     function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferGlobalPoolShape`.
FailureOr<SmallVector<OpFoldResult>>
reifyGlobalPoolResultShape(OpBuilder &b, Location loc, Value input,
                           function_ref<InFlightDiagnostic()> emitError);

/// Compute the supported spatial Resize result shape. The HIP op deliberately
/// does not carry ONNX `sizes` or `scales`: N/C therefore come from `input`,
/// while every spatial extent must already be static in `outputTemplate`.
///
/// A dynamic input N/C extent requires the corresponding template extent to
/// remain dynamic; a static template cannot promise equality to an unknown
/// runtime input extent. A static input N/C extent may refine a dynamic
/// template extent.
FailureOr<SmallVector<int64_t>>
inferResizeShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputTemplate,
                 function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferResizeShape`. Validation completes before any
/// `tensor.dim` is emitted. Dynamic N/C extents come from `input`; spatial
/// extents are constants from `outputTemplate`.
FailureOr<SmallVector<OpFoldResult>>
reifyResizeShape(OpBuilder &b, Location loc, Value input,
                 ArrayRef<int64_t> outputTemplate,
                 function_ref<InFlightDiagnostic()> emitError);

/// Compute the only output shape implemented by the default
/// `hip.multi_head_attention` runtime: separate rank-3 fp16 Q/K/V with equal
/// batch and hidden extents, and equal K/V sequence extents. The result is
/// exactly `[query.B, query.S, query.hidden]`.
FailureOr<SmallVector<int64_t>> inferMultiHeadAttentionOutputShape(
    ArrayRef<int64_t> queryShape, ArrayRef<int64_t> keyShape,
    ArrayRef<int64_t> valueShape, int64_t numHeads,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferMultiHeadAttentionOutputShape`. Validation
/// completes before dimensions are materialized; all result extents come from
/// `query`.
FailureOr<SmallVector<OpFoldResult>> reifyMultiHeadAttentionOutputShape(
    OpBuilder &b, Location loc, Value query, Value key, Value value,
    int64_t numHeads, function_ref<InFlightDiagnostic()> emitError);

/// Pure transpose shape rule: `output[i] = input[perm[i]]`.
FailureOr<SmallVector<int64_t>>
inferTransposeShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> perm);

/// Reify the result shape of a transpose op as `output[i] = input[perm[i]]`.
/// `perm` must be a permutation of `[0, rank)` and have the same length as
/// `input`'s rank. Returns failure on malformed input, before emitting IR.
///
/// Each output dim `i`:
///   - emits `IndexAttr(input.shape[perm[i]])` when that dim is static,
///   - emits `tensor.dim %input, perm[i]` otherwise.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                     ArrayRef<int64_t> perm);

/// Pure Gather shape rule:
/// `data[:axis] ++ indices.shape ++ data[axis+1:]`.
FailureOr<SmallVector<int64_t>> inferGatherShape(ArrayRef<int64_t> dataShape,
                                                 ArrayRef<int64_t> indicesShape,
                                                 int64_t axis);

/// Reify the result shape of a gather op as
/// `output = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.
/// `axis` is normalized into `[0, data.rank)` (negative axis follows ONNX
/// convention). Returns failure on a malformed axis.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyGatherWithAxis(OpBuilder &b, Location loc, Value data, Value indices,
                    int64_t axis);

/// Compute com.microsoft GatherBlockQuantized's dequantized output shape.
///
/// The logical data shape equals the storage shape except that
/// `quantizeAxis` is multiplied by two for byte-packed 4-bit storage. Gather
/// then replaces `gatherAxis` in that logical shape with `indicesShape`:
///
///   logicalData[quantizeAxis] = dataShape[quantizeAxis] * 2  (packed int4)
///   output = logicalData[:gatherAxis] ++ indicesShape
///            ++ logicalData[gatherAxis + 1:]
///
/// If `gatherAxis == quantizeAxis`, the packed axis is removed by Gather and
/// no output extent is multiplied. `uint8Storage` carries the schema's
/// gather-axis-zero restriction independently of whether the storage contains
/// packed 4-bit values.
///
/// Also validates the runtime-supported ranks, attributes, data/scales block
/// grid, and optional zero-point shape. Dynamic extents are treated as
/// statically unknown.
FailureOr<SmallVector<int64_t>> inferGatherBlockQuantizedShape(
    ArrayRef<int64_t> dataShape, ArrayRef<int64_t> indicesShape,
    ArrayRef<int64_t> scalesShape,
    std::optional<ArrayRef<int64_t>> zeroPointsShape, int64_t bits,
    int64_t blockSize, int64_t gatherAxis, int64_t quantizeAxis,
    bool bytePackedInt4, bool uint8Storage,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferGatherBlockQuantizedShape`. Validation completes
/// before any `tensor.dim` or arithmetic operation is emitted.
FailureOr<SmallVector<OpFoldResult>> reifyGatherBlockQuantizedShape(
    OpBuilder &b, Location loc, Value data, Value indices, Value scales,
    Value zeroPoints, int64_t bits, int64_t blockSize, int64_t gatherAxis,
    int64_t quantizeAxis, bool bytePackedInt4, bool uint8Storage,
    function_ref<InFlightDiagnostic()> emitError);

/// Pure GatherND shape rule. A dynamic trailing tuple width returns failure
/// because the output rank is not statically knowable; callers that support an
/// outs-authoritative fallback must distinguish that case before diagnosing.
FailureOr<SmallVector<int64_t>>
inferGatherNDShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> indicesShape,
                   int64_t batchDims);

/// Reify the result shape of a `gather_nd` op as
/// `batch_dims_from_data ++ indices.shape[batch_dims:-1] ++
///  data.shape[batch_dims + indices.shape[-1]:]`.
/// Per ONNX GatherND semantics, output rank =
/// `q + r - indices.shape[-1] - 1 - batch_dims`, where `q = rank(indices)`
/// and `r = rank(data)`. Returns failure when the
/// trailing index-tuple width (`indices.shape[-1]`) is dynamic — the
/// output rank itself is then unknown and reify cannot run.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>> reifyGatherND(OpBuilder &b, Location loc,
                                                   Value data, Value indices,
                                                   int64_t batchDims);

/// ONNX reduction result shape over `axes`, from static extents only.
///
/// `axes` holds the already-resolved reduced axis indices (ONNX negative-axis
/// convention); an empty list means no reduction. `keepdims = 0` drops reduced
/// axes from the output rank, so the output dimension order is *not* positional
/// in the input: reducing axes `[1, 2]` of a rank-4 input maps output dimension
/// 1 to input dimension 3.
///
/// This and `reifyReductionResultShape` share one internal output-to-input
/// dimension mapping, so a destination built from either cannot disagree with
/// the shape `reifyResultShapes` reports. Returns failure when an axis is out
/// of range for `dataShape`.
FailureOr<SmallVector<int64_t>> inferReductionShape(ArrayRef<int64_t> dataShape,
                                                    ArrayRef<int64_t> axes,
                                                    int64_t keepdims);

/// Generated-verifier target for `Hip_DpsOp_Reduction`. Exact result shape is
/// checked only when `axes` is an inline constant. Payload-dynamic axes retain
/// the converter-selected outs shape after structural and attribute checks.
LogicalResult verifyReductionDpsOp(Operation *op, Value data, Value axes,
                                   int64_t keepdims, int64_t noopWithEmptyAxes);

/// ONNX Size always produces one rank-zero i64 result.
FailureOr<SmallVector<int64_t>> inferSizeShape();

/// ONNX reduction result shape over `axes` as mixed extents, emitting
/// `tensor.dim` only for dimensions that are dynamic in `data`. Same mapping
/// as `inferReductionShape`; see it for the `keepdims` semantics. `data` must
/// be a `RankedTensorType`-typed Value.
FailureOr<SmallVector<OpFoldResult>>
reifyReductionResultShape(OpBuilder &b, Location loc, Value data,
                          ArrayRef<int64_t> axes, int64_t keepdims);

/// ONNX LayerNormalization output shapes. Output 0 (Y) equals the input;
/// optional Mean and InvStdDev outputs use the keepdims reduction shape over
/// `[axis, rank)`.
FailureOr<SmallVector<SmallVector<int64_t>>>
inferLayerNormOutputShapes(ArrayRef<int64_t> inputShape, int64_t axis,
                           unsigned numOutputs);

/// Mixed-shape form of `inferLayerNormOutputShapes`.
FailureOr<ReifiedRankedShapedTypeDims>
reifyLayerNormOutputShapes(OpBuilder &b, Location loc, Value input,
                           int64_t axis, unsigned numOutputs);

/// Runtime-supported LayerNormalization stats element type for ONNX
/// `stash_type` (TensorProto enum): 0/1 -> f32, 10 -> f16.
FailureOr<Type> inferLayerNormStatsType(MLIRContext *ctx, int64_t stashType);

/// LinearAttention result shapes:
///   output        = [B, T, max(Hq, Hkv) * Dv]
///   present_state = [B, Hkv, Dk, Dv]
/// where Dk = query[-1] / Hq and Dv = value[-1] / Hkv.
///
/// Also validates the statically knowable head-count and key-sharing
/// constraints from the op contract.
FailureOr<SmallVector<SmallVector<int64_t>>> inferLinearAttentionOutputShapes(
    ArrayRef<int64_t> queryShape, ArrayRef<int64_t> keyShape,
    ArrayRef<int64_t> valueShape, int64_t qNumHeads, int64_t kvNumHeads,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferLinearAttentionOutputShapes`.
FailureOr<ReifiedRankedShapedTypeDims>
reifyLinearAttentionOutputShapes(OpBuilder &b, Location loc, Value query,
                                 Value key, Value value, int64_t qNumHeads,
                                 int64_t kvNumHeads,
                                 function_ref<InFlightDiagnostic()> emitError);

/// CausalConvWithState output shapes for the runtime-supported 1D layout:
///   output        = input = [B, C, L]
///   present_state = [B, C, weight[2] - 1]
///
/// Also validates the depthwise weight layout `[C, 1, K]`, optional bias
/// `[C]`, and optional past state `[B, C, K - 1]`. Dynamic extents are treated
/// as compatible when equality cannot be decided statically.
FailureOr<SmallVector<SmallVector<int64_t>>>
inferCausalConvWithStateOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    std::optional<ArrayRef<int64_t>> biasShape,
    std::optional<ArrayRef<int64_t>> pastStateShape, int64_t ndim,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferCausalConvWithStateOutputShapes`.
FailureOr<ReifiedRankedShapedTypeDims> reifyCausalConvWithStateOutputShapes(
    OpBuilder &b, Location loc, Value input, Value weight, Value bias,
    Value pastState, int64_t ndim,
    function_ref<InFlightDiagnostic()> emitError);

/// SkipSimplifiedLayerNormalization output shapes. The normalized output and
/// optional `input_skip_bias_sum` both equal the input shape. The runtime
/// flattens every non-final input axis into rows, requires skip with the same
/// shape, and rank-1 gamma/optional bias whose length equals the input's final
/// extent.
FailureOr<SmallVector<SmallVector<int64_t>>> inferSkipRmsNormOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> skipShape,
    ArrayRef<int64_t> gammaShape, std::optional<ArrayRef<int64_t>> biasShape,
    unsigned numOutputs, function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferSkipRmsNormOutputShapes`.
FailureOr<ReifiedRankedShapedTypeDims>
reifySkipRmsNormOutputShapes(OpBuilder &b, Location loc, Value input,
                             Value skip, Value gamma, Value bias,
                             unsigned numOutputs,
                             function_ref<InFlightDiagnostic()> emitError);

/// One-shot reify body for ONNX-style reduction ops (reduce_sum,
/// reduce_max, reduce_prod). Tries `reifyReductionWithKeepdims` first
/// to recover per-input-dim mappings from a constant `axes` operand.
/// When `axes` is not a recognised constant, falls back to the shared
/// `HipDpsOp` outs-lift default so the reify interface always
/// succeeds (the only honest answer when we cannot decide which dims
/// were reduced is the type of the `outs` operand the converter
/// already picked).
///
/// `op` must implement both `HipDpsOp` (so the fallback can walk
/// `getDpsInits()`) and have a `RankedTensorType` `data` operand.
/// Returns `failure()` only on the no-tensor-results / non-tensor
/// `data` defensive paths; otherwise always returns `success()`.
///
/// Used as the body of `Hip_DpsOp_Reduction`'s auto-emitted reify
/// dispatcher; see `Hip_DpsOp_Reduction` in `HipOps.td`.
LogicalResult reifyReductionShape(OpBuilder &b, Location loc, Value data,
                                  Value axes, int64_t keepdims,
                                  int64_t noopWithEmptyAxes, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified);

/// One-shot reify body for elementwise NumPy-broadcast ops (add, mul,
/// sub, div, min, mod, equal, less, and, where, ...). Wraps
/// `reifyBroadcastResultShape` with the per-op guards (no-results bail,
/// every operand must be `RankedTensorType`) and writes the lifted
/// dim list into `reified`.
///
/// `operands` is the list of broadcast input operands in the order
/// they should be aligned (right-aligned for NumPy broadcast).
/// Returns `failure()` on any defensive bail or when broadcast itself fails.
/// A successful rank-zero result writes one empty shape into `reified`.
///
/// Used as the body of `Hip_DpsOp_Broadcast`'s auto-emitted reify
/// dispatcher; see `Hip_DpsOp_Broadcast` in `HipOps.td`.
LogicalResult reifyBroadcastShapeFor(OpBuilder &b, Location loc,
                                     ValueRange operands, Operation *op,
                                     ReifiedRankedShapedTypeDims &reified);

/// Compute the result shape of an ONNX-style `pad` op:
///   `output[d] = data.shape[d] + pre_pad[d] + post_pad[d]`
/// where `pre_pad[d]` / `post_pad[d]` come from `pads[axes.find(d)]` /
/// `pads[axes.find(d) + N]` (default `axes = [0, rank)`, `N = num_axes`).
///
/// The pure form validates axes, lengths, duplicate axes, and statically
/// computable extents. Dynamic data dimensions remain dynamic.
FailureOr<SmallVector<int64_t>>
inferPadShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> pads,
              std::optional<ArrayRef<int64_t>> axes);
///
/// The mixed form first resolves compile-time pads/axes from the optional
/// stamped attributes or constant operands and validates through
/// `inferPadShape`. Only then does it emit exact affine index SSA for dynamic
/// input dimensions. Payload-dynamic pads/axes return failure so the caller
/// can preserve its synchronized-readback (converter) or outs-lift (reify)
/// policy; dialect reification never reads tensor payload.
///
/// `axes` may be null (ONNX "pad all axes" default).
LogicalResult reifyPadShape(OpBuilder &b, Location loc, Value data, Value pads,
                            Value axes,
                            std::optional<ArrayRef<int64_t>> staticPads,
                            std::optional<ArrayRef<int64_t>> staticAxes,
                            SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `tile` op:
///   `output[d] = input.shape[d] * repeats[d]`
/// where `repeats` is a rank-1 i64 tensor of length `input.rank`.
///
/// The pure static form validates rank/length/non-negative repeats.
FailureOr<SmallVector<int64_t>> inferTileShape(ArrayRef<int64_t> inputShape,
                                               ArrayRef<int64_t> repeats);

/// Mixed form for constant repeats. Dynamic input dims produce exact index SSA;
/// payload-dynamic repeats return failure so the caller can use bulk readback
/// (converter) or outs-lift (reify).
LogicalResult reifyTileShape(OpBuilder &b, Location loc, Value input,
                             Value repeats,
                             std::optional<ArrayRef<int64_t>> staticRepeats,
                             SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `expand` op:
///   broadcast `input.shape` against `shape`'s constant values
///   (right-aligned, leading-1 padded on whichever side is shorter).
///
/// Same fold-or-bail strategy. `shape` is the target-shape operand
/// (rank-1 i64 tensor); when it is an `arith.constant` with a
/// `DenseIntElementsAttr`, the helper runs MLIR's
/// `OpTrait::util::getBroadcastedShape` and lifts the result shape.
/// Returns `failure()` when the broadcast result has any dynamic dim
/// (so the caller falls back to outs-lifting).
LogicalResult
reifyExpandShape(OpBuilder &b, Location loc, Value input, Value shape,
                 SmallVectorImpl<OpFoldResult> &out,
                 std::optional<ArrayRef<int64_t>> staticShape = std::nullopt);

/// Compute the result shape of an ONNX-style `slice` op. `output[axis]`
/// for each `axis` in `axes` is `ceil_div(end - start, step)` (negative
/// indices and steps clamp per ONNX rules); for axes not in `axes`,
/// `output[d] = data.shape[d]`.
///
/// Exact static clamping requires a static input extent on every sliced axis.
/// Dynamic untouched axes remain dynamic and pass through from `data`.
FailureOr<SmallVector<int64_t>>
inferSliceShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> starts,
                ArrayRef<int64_t> ends, std::optional<ArrayRef<int64_t>> axes,
                std::optional<ArrayRef<int64_t>> steps);
///
/// The mixed form folds all four parameter tensors, validates through
/// `inferSliceShape` before emitting IR, and materializes `tensor.dim` only for
/// untouched dynamic axes. A payload-dynamic parameter or dynamic sliced-axis
/// clamp returns failure, preserving the native Slice capacity/outs fallback.
///
/// `axes` and `steps` may be null (ONNX defaults: `[0, rank)` and
/// all-ones respectively).
LogicalResult reifySliceShape(OpBuilder &b, Location loc, Value data,
                              Value starts, Value ends, Value axes, Value steps,
                              SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `range` op:
///   `output.shape[0] = ceil_div(max(0, limit - start), delta)` (limit
///   exclusive; clamp to 0 when `limit < start && delta > 0`, etc.).
///
/// `start`, `limit`, `delta` are scalar (rank-0) tensors. Pure
/// fold-or-bail: when ALL three are `arith.constant` with a single int
/// value, computes the static output count and returns a one-element
/// `IndexAttr` vector. Else `failure()` -- the dim-arith chain to
/// compute the count at runtime is rarely useful for refinement and
/// would persist in IR.
LogicalResult
reifyRangeShape(OpBuilder &b, Location loc, Value start, Value limit,
                Value delta, SmallVectorImpl<OpFoldResult> &out,
                std::optional<ArrayRef<int64_t>> staticValues = std::nullopt);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
