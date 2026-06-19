/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- op_state.cpp - Per-op state slot runtime ABI ---------------------===//
//
// Runtime side of the op-state-slots design (see
// docs/design/op-state-slots-design.md):
//   * hipdnn_ep_op_states_alloc / _set / _get manage the per-session
//     RuntimeState::op_states array. They keep the RuntimeState layout opaque
//     to generated IR, which only ever calls these by symbol.
//
// The per-op constructors (hipdnn_ep_op_state_construct_<class>) live in each
// op's runtime TU (e.g. the reference op `conv` in real/miopen.cpp), since
// their state usually owns device resources. This TU is host-only so the same
// bitcode serves both the real and mock runtimes.
//
//===----------------------------------------------------------------------===//

#include "op_state.h"

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdlib>

extern "C" {

// Allocate the per-session op-state array (one entry per assigned slot).
// Zero-fills so unconstructed slots are null. `state` is always a valid
// RuntimeState* (the generated init passes inference_init's own state), so the
// only real failure is the allocation itself. Returns true on success.
bool hipdnn_ep_op_states_alloc(RuntimeState *state, int64_t n) {
  state->op_states = nullptr;
  state->num_op_states = 0;
  if (n <= 0)
    return true;
  state->op_states = static_cast<OpState **>(
      calloc(static_cast<size_t>(n), sizeof(OpState *)));
  if (!state->op_states)
    return false;
  state->num_op_states = static_cast<int>(n);
  return true;
}

// Store a constructed state into its slot. Bounds- and null-checked so it is
// memory-safe even if allocation failed (num_op_states == 0) or the compiler
// emits a bad slot/value. Returns true on success, false (no write) otherwise.
bool hipdnn_ep_op_state_set(RuntimeState *state, int32_t slot, OpState *value) {
  if (!state->op_states)
    return false;
  if (slot < 0 || slot >= state->num_op_states)
    return false;
  if (!value)
    return false;
  state->op_states[slot] = value;
  return true;
}

void *hipdnn_ep_op_state_get(RuntimeState *state, int slot) {
  if (!state || !state->op_states)
    return nullptr;
  if (slot < 0 || slot >= state->num_op_states)
    return nullptr;
  return state->op_states[slot];
}

} // extern "C"

// NOTE: per-op constructors (hipdnn_ep_op_state_construct_<class>) live in each
// op's own runtime translation unit, not here, because their state typically
// owns device resources (HIP). For example `matmul` defines MatmulState +
// hipdnn_ep_op_state_construct_matmul in lib/Runtime/real/matmul.cpp (with a
// mock stub in lib/Runtime/mock/op_state_stubs.cpp). This TU stays host-only so
// it can be shared verbatim by the real and mock runtimes.
