/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- runtime_plugin_api.cpp - Plugin-visible runtime state accessors ----===//
//
// Definitions of the C API in hip/Runtime/PluginAPI.h: the session stream,
// the shared workspace, the DP4A matmul scratch and the device error flag.
// Runtime helpers built outside lib/Runtime use these instead of reaching
// into RuntimeState.
//
//===----------------------------------------------------------------------===//

#include "hip/Runtime/PluginAPI.h"

#include "debug_log.h"
#include "hip_cleanup.h"
#include "runtime_state_internal.h"

#include <cstdio>
#include <cstdlib>

extern "C" {

void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state ? static_cast<void *>(state->stream) : nullptr;
}

void *hipdnn_ep_state_get_workspace(RuntimeState *state) {
  return state ? state->workspace : nullptr;
}

size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state) {
  return state ? state->workspace_size : 0;
}

int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->workspace_size >= needed_size)
    return 0;

  // Amortize growth: when enlarging an existing workspace, round the new
  // size up to at least 1.5x the current buffer. Callers whose request
  // size grows monotonically by a small increment per inference (e.g. the
  // GQA decode path, which sizes S buffers to B*H*total_seq and adds B*H
  // elements per token) would otherwise trigger a hipStreamSynchronize +
  // hipFree + hipMalloc cycle on every decode step; with the 1.5x factor
  // that drops to O(log N) reallocations over the whole generation.
  // Cold-start (no existing workspace) keeps the exact requested size so
  // warmup doesn't silently double large initial allocations.
  size_t alloc_size = needed_size;
  if (state->workspace_size > 0) {
    size_t grown = state->workspace_size + state->workspace_size / 2; // 1.5x
    if (grown > alloc_size)
      alloc_size = grown;
  }

  // Grow: free old, allocate new.
  // Sync the stream first to ensure no in-flight kernel is still using the
  // old workspace buffer (prevents use-after-free on async GPU execution).
  if (state->workspace) {
    if (state->stream) {
      HIP_CLEANUP(hipStreamSynchronize(state->stream));
    }
    HIP_CLEANUP(hipFree(state->workspace));
    state->workspace = nullptr;
    state->workspace_size = 0;
  }

  if (hipMalloc(&state->workspace, alloc_size) != hipSuccess) {
    fprintf(
        stderr,
        "hipdnn_ep_state_ensure_workspace: hipMalloc failed for %zu bytes\n",
        alloc_size);
    std::abort();
  }

  state->workspace_size = alloc_size;
  RUNTIME_DEBUG_LOG(
      "[workspace] Allocated shared workspace: %zu bytes (requested %zu)\n",
      alloc_size, needed_size);
  return 0;
}

void *hipdnn_ep_state_get_matmul_dp4a_scratch(RuntimeState *state) {
  return state ? state->matmul_dp4a_scratch : nullptr;
}

size_t hipdnn_ep_state_get_matmul_dp4a_scratch_size(RuntimeState *state) {
  return state ? state->matmul_dp4a_scratch_size : 0;
}

int hipdnn_ep_state_ensure_matmul_dp4a_scratch(RuntimeState *state,
                                               size_t needed_size) {
  if (!state)
    return -1;
  if (needed_size == 0)
    return 0;
  if (state->matmul_dp4a_scratch_size >= needed_size)
    return 0;

  // 1.5x growth amortisation mirrors conv_scratch / qmoe_scratch.
  size_t alloc_size = needed_size;
  if (state->matmul_dp4a_scratch_size > 0) {
    size_t grown =
        state->matmul_dp4a_scratch_size + state->matmul_dp4a_scratch_size / 2;
    if (grown > alloc_size)
      alloc_size = grown;
  }

  if (state->matmul_dp4a_scratch) {
    // Drain any in-flight dp4a gemv that may still be reading the old buffer
    // before we free it. Growth is rare (only when a larger K is first seen).
    if (state->stream) {
      hipStreamSynchronize(state->stream);
    }
    HIP_CLEANUP(hipFree(state->matmul_dp4a_scratch));
    state->matmul_dp4a_scratch = nullptr;
    state->matmul_dp4a_scratch_size = 0;
  }

  if (hipMalloc(&state->matmul_dp4a_scratch, alloc_size) != hipSuccess) {
    fprintf(stderr,
            "hipdnn_ep_state_ensure_matmul_dp4a_scratch: hipMalloc failed for "
            "%zu bytes\n",
            alloc_size);
    return -1;
  }
  state->matmul_dp4a_scratch_size = alloc_size;
  return 0;
}

void *hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state) {
  return state ? static_cast<void *>(state->device_error_flag) : nullptr;
}

int hipdnn_ep_state_reset_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    fprintf(stderr, "hipdnn_ep_state_reset_error_flag: invalid state\n");
    return -1;
  }
  hipError_t err =
      hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  return (err == hipSuccess) ? 0 : -1;
}

int hipdnn_ep_state_read_and_clear_error_flag(RuntimeState *state) {
  if (!state || !state->device_error_flag || !state->stream) {
    fprintf(stderr,
            "hipdnn_ep_state_read_and_clear_error_flag: invalid state\n");
    return -1;
  }

  int host_error = 0;
  hipError_t err =
      hipMemcpyAsync(&host_error, state->device_error_flag, sizeof(int),
                     hipMemcpyDeviceToHost, state->stream);
  if (err != hipSuccess)
    return -1;
  err = hipStreamSynchronize(state->stream);
  if (err != hipSuccess)
    return -1;
  if (host_error != 0)
    return host_error;

  err = hipMemsetAsync(state->device_error_flag, 0, sizeof(int), state->stream);
  return (err == hipSuccess) ? 0 : -1;
}

} // extern "C"
