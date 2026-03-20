/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// HIP memory copy wrapper (GPU-to-GPU using hipMemcpyAsync)
// Follows opaque RuntimeState pattern - extracts stream internally
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes) {
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null pointer\n");
    return -1;
  }
  if (size_bytes == 0) {
    return 0; // No-op for zero-sized copy
  }

  // Extract stream from opaque RuntimeState using accessor function
  // (Maintains abstraction barrier - no direct field access)
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err = hipMemcpyAsync(dst_ptr, src_ptr, size_bytes,
                                  hipMemcpyDeviceToDevice, stream);

  if (err != hipSuccess) {
    fprintf(stderr, "wrap_hipMemcpyAsync: copy failed (%zu bytes): %s\n",
            size_bytes, hipGetErrorString(err));
    return -1;
  }

  return 0;
}
