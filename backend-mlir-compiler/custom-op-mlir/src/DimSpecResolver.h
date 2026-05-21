/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_DIM_SPEC_RESOLVER_H
#define HIPDNN_EP_DIM_SPEC_RESOLVER_H

// EP-side evaluator for per-output DimSpec trees.
//
// The compiler emits a FlatBuffers HipModelMetaInfo blob inside the model.dll
// that the EP loads via inference_get_metadata_json (see InferenceState). For
// each output tensor, the blob carries one DimSpec tree per dim. This header
// walks that tree to resolve a concrete int64 size at:
//
//   * Pre-compute time (Category A / B / D): the value only depends on input
//     shapes, host-readable input values, and integer arithmetic. We resolve
//     these BEFORE inference_compute so we can call ctx.GetOutput with the
//     final shape and avoid a post-compute D2H stall.
//
//   * Post-compute time (Category C): the value comes from a RuntimeSlot
//     published by a wrap_* during inference_compute. We resolve these AFTER
//     inference_compute returns and the EP stream has been synced.
//
// The caller (MlirCustomOp::Compute) decides which phase to invoke based on
// `containsRuntimeSlot`. A future post-compute path will need to call
// inference_state.read_dim(slot_id) and then re-evaluate.
//
// Bool return semantics:
//   resolve returns false when the spec is "not resolvable in this phase"
//   (typically because a RuntimeSlot leaf was unpublished at the time of the
//   call). The caller must distinguish "skipped" from "failed":
//     * pre-compute, slot present, resolved == false -> defer the output
//     * post-compute, slot present, resolved == false -> LOG(FATAL)
//
// Errors during resolution (out-of-range slot, ill-formed tree, etc.) are
// LOG(FATAL) -- the compiler emitted the tree, so an unresolvable spec is a
// compiler/runtime invariant violation, not a recoverable runtime condition.

#include "model_metadata_generated.h"
#include <cstdint>
#include <vector>

namespace mlir_compilation::customop {

class InferenceState;

// True iff any leaf in `spec` is a RuntimeSlot (Category C). Used to gate
// pre-compute resolution -- if the tree has any RuntimeSlot leaf, we must
// wait for inference_compute to publish it. Pure-arithmetic trees can be
// resolved before compute.
bool containsRuntimeSlot(const mlir::hip::DimSpecT &spec);

// Resolve the DimSpec to a concrete dimension value (int64).
//
//   spec:           the tree to evaluate
//   input_shapes:   per-input dim arrays as the EP marshalled them
//                   (input_shapes[input_index][dim_index] is the int64
//                   extent). MUST be in the same order as the model's
//                   declared inputs (the EP-side marshalling already maps
//                   ORT order -> compiler order).
//   input_data:    for InputValueI64 leaves -- per-input host pointer to
//                   the i64 elements of that input tensor. nullptr entries
//                   are allowed for inputs that aren't referenced.
//   state:          for RuntimeSlot leaves -- the inference state holding
//                   the slot table. May be null pre-compute (returns false
//                   on RuntimeSlot).
//   out_value:      receives the resolved value on success.
//
// Returns true iff resolution produced a value. Returns false ONLY when a
// RuntimeSlot leaf is encountered and `state` is null OR the slot is
// unpublished. All other failure modes (bad tree, out-of-range indices)
// LOG(FATAL).
bool resolve(const mlir::hip::DimSpecT &spec,
             const std::vector<std::vector<int64_t>> &input_shapes,
             const std::vector<const void *> &input_data,
             const InferenceState *state, int64_t &out_value);

} // namespace mlir_compilation::customop

#endif // HIPDNN_EP_DIM_SPEC_RESOLVER_H
