/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>

// ONNX Shape runtime wrapper (dynamic-input path only).
//
// Statically-shaped Shape ops are folded to arith.constant in
// ShapeConversion.cpp and never reach this symbol — every output element
// is a compile-time constant in that case.
//
// This wrapper is only called when at least one input dim is dynamic
// (typically because the input came from a Category-C producer such as
// NonZero). The HipToLLVM lowering materialises each output element as
// an i64 SSA value (constant or slot read), packs them on the host
// stack, and hands the buffer to us. Our only job: blit the i64 vector
// H2D into the destination GPU memref.
//
// `num_elements` is the rank-1 length (compile-time constant — equal to
// `end - start` of the ONNX Shape op, clamped against the input rank).
// `host_values` is a host-resident int64_t[num_elements] living in the
// caller's stack frame; pageable host memory + H2D in HIP semantics is
// effectively synchronous from the host's POV (the driver stages the
// bytes into a pinned scratch buffer before returning), so the stack
// slot is safe to read across the call.
int wrap_shape(RuntimeState *state, void *output, int64_t num_elements,
               const int64_t *host_values) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_shape\n");
    return -1;
  }
  if (!output) {
    fprintf(stderr, "Invalid output pointer in wrap_shape\n");
    return -1;
  }
  if (!host_values) {
    fprintf(stderr, "Invalid host_values pointer in wrap_shape\n");
    return -1;
  }
  if (num_elements < 0) {
    fprintf(stderr, "Invalid num_elements (%lld) in wrap_shape\n",
            (long long)num_elements);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_shape(num_elements=%lld) values=[",
                    (long long)num_elements);
  for (int64_t i = 0; i < num_elements && i < 8; ++i) {
    RUNTIME_DEBUG_LOG("%lld%s", (long long)host_values[i],
                      i + 1 < num_elements ? ", " : "");
  }
  if (num_elements > 8) {
    RUNTIME_DEBUG_LOG("...");
  }
  RUNTIME_DEBUG_LOG("]\n");

  if (num_elements == 0)
    return 0;

  void *stream = hipdnn_ep_state_get_stream(state);
  int64_t total_bytes = num_elements * static_cast<int64_t>(sizeof(int64_t));
  return wrap_hipMemcpyH2D(output, host_values, total_bytes, stream);
}
