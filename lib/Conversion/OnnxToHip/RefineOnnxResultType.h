/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RefineOnnxResultType.h - ONNX result-type rules + meet -------------===//
//
// Two free functions used by the `--onnx-infer-shapes` pass:
//
//   * `refineResultTypeFromOperands(op, resultIdx)` -- the rules library.
//     Switches on `op->getName()` and returns a candidate
//     `RankedTensorType` derived from operand types for the named ONNX
//     op. Returns null Type when no rule matches; the caller treats
//     null as "leave the op alone". Ops that change rank in ways the
//     library does not yet model (e.g. `onnx.ReduceSum` with
//     `keepdims = 0`) are intentionally absent so a stale rank-0
//     placeholder cannot be silently promoted.
//
//   * `meetRankedTypes(old, candidate)` -- lattice-meet helper.
//     Combines a rule's candidate output with the op's existing result
//     type, preserving any static dim already present on either side.
//     Returns null Type on conflict (rank mismatch, element-type
//     mismatch, or two static dims that disagree). The shape-inference
//     pass treats null as no-op. Modeled after torch-mlir's
//     `Torch::meetTensorTypes` (lib/Dialect/Torch/IR/TorchTypes.cpp);
//     same semantics, narrower API surface (RankedTensorType only,
//     since onnx.* ops only ever produce ranked tensors).
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_REFINEONNXRESULTTYPE_H
#define HIP_CONVERSION_ONNXTOHIP_REFINEONNXRESULTTYPE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace hip {

/// Apply the rules library to `op` and return a candidate result type
/// derived from operand types. Returns null Type when no rule matches
/// or the rule cannot make progress (e.g. operand still unranked).
///
/// Rule groups (op-name keyed):
///   1. Pointwise (numpy-style align-right broadcast over operand
///      shapes; unary ops are the degenerate N=1 case): Identity,
///      Cast, CastLike, Tanh, Sigmoid, Relu, Gelu, Erf, Softmax,
///      Add, Sub, Mul, Div, Min, Max, Where, Equal, Greater, Less.
///      Output rank = max operand rank, per-dim merge with leading-1
///      padding. Element type is read from the EXISTING result type
///      so element-type-changing ops (Cast / CastLike) and i1-result
///      comparison ops (Equal / Greater / Less) are handled without
///      inspecting attributes.
///   2. Concat: result rank == operand rank; axis dim is the sum of
///      per-operand axis dims (kDynamic if any is dynamic); non-axis
///      dims agree-or-dynamic; negative axis wrap-corrected.
///   3. Slice: rank-preserving, all dims kDynamic.
///   4. LayerNormalization: result 0 has operand[0]'s shape (reuses
///      the pointwise rule); results 1, 2 (mean / inv-std) are not
///      handled (rule returns null).
RankedTensorType refineResultTypeFromOperands(Operation *op,
                                              unsigned resultIdx);

/// Lattice meet over (rank, per-dim, elem-type). Returns the strictly-
/// more-refined type combining `old` and `candidate`, or null on any
/// conflict. Accepted refinements:
///   * Rank promotion: rank-0 `old` + ranked `candidate` -> candidate.
///     This is the canonical refinement applied to body ops in an
///     outlined `onnx.Loop` (which arrive with rank-0 placeholder
///     result types).
///   * Per-dim refinement: kDynamic + static -> static (either side);
///     equal-static stays; both kDynamic stays kDynamic.
///
/// Conflict cases (return null):
///   * Element types disagree.
///   * Ranks differ AND `old` is not rank-0.
///   * Two static dims at the same position disagree.
///
/// `old` and `candidate` must be RankedTensorType (caller filters for
/// this); non-ranked operands are not touched by the rules library.
RankedTensorType meetRankedTypes(RankedTensorType old,
                                 RankedTensorType candidate);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_REFINEONNXRESULTTYPE_H
