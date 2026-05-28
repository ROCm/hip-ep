/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInference.h - Pure result-type rules for onnx.* ops -===//
//
// Pure result-type inference for `onnx.*` ops, shared between the
// `InferOnnxShapes` pass (which uses them to refine existing result
// types in-place) and the pre-lowering rewriters `ProjectorOpsRewrites`
// / `FastGeluFusion` (which use them to type newly-emitted ops without
// manually constructing `RankedTensorType` at every emission site).
//
// Convention:
//   * Each helper computes the most-refined `RankedTensorType` derivable
//     from the operand types + attributes + (for Reshape) the shape
//     operand's SSA chain.
//   * The element type of the result matches the relevant input's
//     element type (or `targetElemType` for Cast), following ONNX
//     semantics.
//   * On precondition failure (rank mismatch, perm out of bounds, etc.)
//     the helper returns a default-constructed `RankedTensorType` (i.e.
//     `!result`). Callers must null-check.
//   * For `Reshape`, an unresolvable shape operand degrades the result
//     to an all-dynamic tensor of the requested rank (still useful as a
//     placeholder type for the rewriter case).
//
// These helpers DO NOT mutate the IR and DO NOT consult any existing
// result type. The decision of whether to apply the proposal (e.g. via
// the `InferOnnxShapes` pass's `isStrictlyTighter` guard) is the
// caller's.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_RESULTTYPEINFERENCE_H
#define HIP_CONVERSION_ONNXTOHIP_RESULTTYPEINFERENCE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace hip {

/// Reshape: traces the `shapeOperand` SSA chain backward through
/// `onnx.Concat / Slice / Shape / Gather / Unsqueeze / Cast / Constant`,
/// and computes the resulting shape. Handles ONNX `0` (copy-from-input,
/// gated by `allowzero`) and `-1` (compute-from-total-size, when dyn
/// counts on both sides match). Element type taken from `inputType`.
RankedTensorType inferReshapeResultType(RankedTensorType inputType,
                                        Value shapeOperand, int64_t outputRank,
                                        int64_t allowzero);

/// Transpose: `output[i] = input[perm[i]]`. Element type from input.
RankedTensorType inferTransposeResultType(RankedTensorType inputType,
                                          ArrayRef<int64_t> perm);

/// MatMul: `out = [...broadcast outer-batch, lhs[-2], rhs[-1]]`. Element
/// type from `lhsType`.
///
/// IMPORTANT (CLAUDE.md "MatMul outer-batch dim alignment" gotcha): the
/// outer-batch dims of lhs/rhs are right-aligned over the OUTER slice
/// only (the `lhsRank-2` / `rhsRank-2` window) — NOT right-aligned over
/// the full operand rank. Right-aligning over the full rank silently
/// traces output dims to the K dim of an operand instead of the batch
/// dim, with no compile error and wrong output shapes at runtime.
RankedTensorType inferMatMulResultType(RankedTensorType lhsType,
                                       RankedTensorType rhsType);

/// Cast: same shape, element type replaced by `targetElemType`.
RankedTensorType inferCastResultType(RankedTensorType inputType,
                                     Type targetElemType);

/// Unary same-shape (Tanh / Softmax / LayerNorm / Sqrt / Gelu / Sigmoid /
/// Neg / Erf / ...): output type == input type. Trivially returns
/// `inputType` after rank sanity-check.
RankedTensorType inferUnarySameShapeResultType(RankedTensorType inputType);

/// Binary broadcast (Add / Sub / Mul / Div / Pow / ...): numpy-style
/// right-aligned broadcast. Element type from `lhsType`.
RankedTensorType inferBinaryBroadcastResultType(RankedTensorType lhsType,
                                                RankedTensorType rhsType);

/// Concat: along `axis`, sizes sum (or dynamic if any input is dynamic at
/// `axis`); other axes tighten to the first operand that is static at
/// that position (matches the most-refined view of the shared shape).
/// Rank / element type taken from the first ranked operand. `axis` is
/// the raw ONNX attribute (may be negative).
RankedTensorType inferConcatResultType(ValueRange operands, int64_t axis);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_RESULTTYPEINFERENCE_H
