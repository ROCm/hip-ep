/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Output allocator runtime contract (Phase 3).
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
#include <cstring>

// EP -> model.dll. Exported so the EP can GetProcAddress it from the compiled
// model.dll (the symbol is also added to export_symbols in
// lib/Compiler/CompilerDriver.cpp). HIPDNN_EP_RT_EXPORT is dllexport in the
// bitcode build -- belt-and-suspenders with that list: it keeps the symbol
// alive through LLVM optimization, which the linker-stage export list cannot --
// and is empty in the unit-test build via HIPDNN_EP_RT_NO_EXPORT.
extern "C" HIPDNN_EP_RT_EXPORT void
hipdnn_ep_set_output_allocator(RuntimeState *state,
                               const hipdnn_output_allocator_t *allocator) {
  if (!state)
    return;
  // Reset to "none installed": self-describing size, null context + callback.
  state->output_allocator.struct_size = sizeof(hipdnn_output_allocator_t);
  state->output_allocator.self = nullptr;
  state->output_allocator.allocate = nullptr;
  if (!allocator)
    return;
  // Forward/backward compatible copy: take only the prefix both sides agree
  // on. A caller built against an older struct passes a smaller struct_size
  // (new fields stay at the defaults above); a newer caller's unknown tail
  // beyond our sizeof is ignored. struct_size is then normalized to OUR sizeof
  // so the stored value always describes this build's layout.
  size_t n = allocator->struct_size;
  if (n > sizeof(hipdnn_output_allocator_t))
    n = sizeof(hipdnn_output_allocator_t);
  memcpy(&state->output_allocator, allocator, n);
  state->output_allocator.struct_size = sizeof(hipdnn_output_allocator_t);
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
