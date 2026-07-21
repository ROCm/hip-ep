// ============================================================
// Minimal EP-runtime substrate for the standalone GQA dispatch harness.
//
// The real GQA dispatchers (lib/Runtime/real/gqa.cpp and gqa_back.cpp) call a
// handful of RuntimeState accessors + op-state + op-profile entry points that
// normally live in the compiled EP runtime. Here we provide trivial,
// self-contained implementations over the real RuntimeState struct so each
// flow can be linked into an isolated .dll and driven without the full EP.
//
// One copy is compiled into each flow's DLL (gqa_new.dll / gqa_back.dll); the
// symbols are internal to each module and never cross the DLL boundary.
// ============================================================

#include "op_profile.h"             // op_profile_* + OpProfileState fwd
#include "op_state.h"               // OpState
#include "runtime_state_internal.h" // full RuntimeState layout

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>

//===----------------------------------------------------------------------===//
// RuntimeState accessors
//===----------------------------------------------------------------------===//
extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state ? static_cast<void *>(state->stream) : nullptr;
}

extern "C" void *hipdnn_ep_state_get_hipblas_handle(RuntimeState *state) {
  return state ? static_cast<void *>(state->hipblas_handle) : nullptr;
}

extern "C" void *hipdnn_ep_state_get_workspace(RuntimeState *state) {
  return state ? state->workspace : nullptr;
}

extern "C" size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state) {
  return state ? state->workspace_size : 0;
}

// Grow-on-demand device workspace; never shrinks. Mirrors the real runtime
// contract (free + malloc on grow, no data preservation).
extern "C" int hipdnn_ep_state_ensure_workspace(RuntimeState *state,
                                                 size_t needed_size) {
  if (!state)
    return -1;
  if (state->workspace_size >= needed_size)
    return 0;
  if (state->workspace) {
    hipFree(state->workspace);
    state->workspace = nullptr;
    state->workspace_size = 0;
  }
  if (hipMalloc(&state->workspace, needed_size) != hipSuccess) {
    state->workspace = nullptr;
    state->workspace_size = 0;
    return -1;
  }
  state->workspace_size = needed_size;
  return 0;
}

//===----------------------------------------------------------------------===//
// Op-state slots
//===----------------------------------------------------------------------===//
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *state, int slot) {
  if (!state || slot < 0 || slot >= state->num_op_states)
    return nullptr;
  return state->op_states[slot];
}

void hipdnn_ep_op_state_set(RuntimeState *state, int32_t slot, OpState *value) {
  if (!state || slot < 0 || slot >= state->num_op_states)
    return;
  state->op_states[slot] = value;
}

//===----------------------------------------------------------------------===//
// Op-profile (always inactive in the harness; symbols still need to link
// because OP_PROFILE expands to guarded calls into them).
//===----------------------------------------------------------------------===//
extern "C" void *hipdnn_ep_state_get_op_profile(RuntimeState *) {
  return nullptr;
}

bool op_profile_is_active(OpProfileState *) { return false; }
void op_profile_ensure_epoch(OpProfileState *, hipStream_t) {}
int op_profile_acquire_marker(OpProfileState *) { return 0; }
hipEvent_t op_profile_get_marker_event(OpProfileState *, int) {
  return nullptr;
}
void op_profile_add_pending(OpProfileState *, const std::string &,
                            const std::string &, int, double, int64_t, double) {
}
void op_profile_add_cpu(OpProfileState *, const std::string &, double) {}
