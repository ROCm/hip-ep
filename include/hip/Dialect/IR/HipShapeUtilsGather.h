/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_GATHER_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_GATHER_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

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

/// Pure OneHot shape rule: insert the semantic depth extent at `axis` in the
/// indices shape. `depth` is absent when the scalar payload is not statically
/// known, in which case the inserted extent remains dynamic.
FailureOr<SmallVector<int64_t>> inferOneHotShape(ArrayRef<int64_t> indicesShape,
                                                 std::optional<int64_t> depth,
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
/// no output extent is multiplied. `uint8Storage` means unsigned `bits == 8`
/// logical storage and enforces its gather-axis-zero, last-quantize-axis
/// policy. It is false for signed int8 and for byte-packed INT4/UINT4.
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
/// The indices tensor must have i64 elements, matching the width-less runtime
/// ABI that interprets its pointer as `int64_t *`.
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

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_GATHER_H
