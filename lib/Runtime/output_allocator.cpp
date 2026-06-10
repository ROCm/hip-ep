/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Output allocator runtime contract.
//
// Two entry points bridge the EP and the generated model.dll:
//   * hipdnn_ep_set_output_allocator - EP-called, EXPORTED from model.dll.
//     Installs the allocator onto RuntimeState before inference_compute.
//   * hipdnn_ep_alloc_output - called by generated main_graph (lowered from
//     hip.alloc_output). Forwards to the installed callback.
//
// Kept in its own translation unit (not hipdnn_ep_runtime_state.cpp) so the
// GPU-free unit test can compile it natively against the mock runtime types -
// nothing here touches HIP/MIOpen/hipBLASLt. See test/runtime/.
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
  // A null allocator clears the slot ("none installed"). The struct layout is a
  // fixed ABI contract (see hipdnn_ep_runtime.h), so a plain copy is correct on
  // both sides of the model.dll <-> EP boundary.
  if (!allocator) {
    state->output_allocator.self = nullptr;
    state->output_allocator.allocate = nullptr;
    return;
  }
  state->output_allocator = *allocator;
}

// generated main_graph -> runtime. Internal (not exported): resolved within the
// DLL at bitcode-link time by the generated call site. Forwards to the EP
// callback; returns null (and logs) when no allocator is installed - which can
// only happen if the allocator codegen path runs without the EP installing one.
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
