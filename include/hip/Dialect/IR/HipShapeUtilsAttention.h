/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_ATTENTION_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_ATTENTION_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

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
///   output        = input = [B, C, L] or [B, L, C] when channels-last
///   present_state = [B, C, weight[2] - 1]
///
/// Also validates the depthwise weight layout `[C, 1, K]`, optional bias
/// `[C]`, and optional past state `[B, C, K - 1]`. The state layout remains
/// channels-first regardless of the input/output layout. Dynamic extents are
/// compatible when equality cannot be decided statically.
FailureOr<SmallVector<SmallVector<int64_t>>>
inferCausalConvWithStateOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    std::optional<ArrayRef<int64_t>> biasShape,
    std::optional<ArrayRef<int64_t>> pastStateShape, int64_t ndim,
    bool channelsLast, function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferCausalConvWithStateOutputShapes`.
FailureOr<ReifiedRankedShapedTypeDims> reifyCausalConvWithStateOutputShapes(
    OpBuilder &b, Location loc, Value input, Value weight, Value bias,
    Value pastState, int64_t ndim, bool channelsLast,
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

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_ATTENTION_H
