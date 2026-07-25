/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Output allocator runtime contract.
//
// Two functions connect the EP and the generated model.dll:
//   * hipdnn_ep_set_output_allocator - the EP calls this (it is EXPORTED from
//     the model.dll) to set the allocator on RuntimeState before
//     inference_compute.
//   * hipdnn_ep_alloc_output - the generated main_graph calls this (it is what
//     hip.alloc_output lowers to). It forwards to the set callback.
//
// Kept in its own .cpp file (not hipdnn_ep_runtime_state.cpp) so the GPU-free
// unit test can build it against the mock runtime types - nothing here touches
// HIP/MIOpen/hipBLASLt. See test/runtime/.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdio>

// EP -> model.dll. Exported so the EP can GetProcAddress it (see
// HIPDNN_EP_RT_EXPORT in hipdnn_ep_runtime.h for why the attribute is on both
// the decl and this def, and why it is also listed in export_symbols).
extern "C" HIPDNN_EP_RT_EXPORT void
hipdnn_ep_set_output_allocator(RuntimeState *state,
                               const hipdnn_output_allocator_t *allocator) {
  if (!state)
    return;
  // A null allocator clears the slot ("none set"). The struct layout is a fixed
  // ABI contract (see hipdnn_ep_runtime.h), so a plain copy works on both sides
  // of the model.dll <-> EP boundary.
  if (!allocator) {
    state->output_allocator.self = nullptr;
    state->output_allocator.allocate = nullptr;
    return;
  }
  state->output_allocator = *allocator;
}

// generated main_graph -> runtime. Internal (not exported): the generated code
// calls it directly inside the DLL. Forwards to the EP callback; returns null
// (and logs) when no allocator is set - which only happens if the generated
// code asks for an output buffer but the EP never set a callback.
extern "C" void *hipdnn_ep_alloc_output(RuntimeState *state, int64_t out_idx,
                                        const int64_t *shape, int64_t rank,
                                        int64_t elem_size) {
  if (!state || !state->output_allocator.allocate) {
    fprintf(stderr,
            "hipdnn_ep_alloc_output: no output allocator installed "
            "(out_idx=%lld)\n",
            (long long)out_idx);
    return nullptr;
  }
  return state->output_allocator.allocate(state->output_allocator.self, out_idx,
                                          shape, rank, elem_size);
}
