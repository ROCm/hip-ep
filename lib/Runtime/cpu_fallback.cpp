/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Debug CPU fallback setter (EP -> model.dll).
//
// Mirrors output_allocator.cpp: exported from model.dll, EP resolves by name
// and installs before inference_compute. See docs/design/debug-cpu-fallback-
// plan.md.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdio>

extern "C" HIPDNN_EP_RT_EXPORT void
hipdnn_ep_set_cpu_fallback(RuntimeState *state,
                           const hipdnn_cpu_fallback_iface_t *iface) {
  if (!state)
    return;
  if (!iface) {
    state->cpu_fallback.user = nullptr;
    state->cpu_fallback.invoke = nullptr;
    return;
  }
  state->cpu_fallback = *iface;
}
