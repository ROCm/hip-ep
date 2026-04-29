/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_concat.  The compiler stack-allocates two arrays
// (input pointers, per-input inner sizes) at lowering time and passes them
// straight through to the kernel.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

extern "C" int wrap_concat(RuntimeState *state, void *output,
                           int64_t element_size_bytes, int64_t outer,
                           int64_t output_inner, int64_t num_inputs,
                           const void *const *inputs,
                           const int64_t *input_inner_sizes) {
  if (!state || !output || !inputs || !input_inner_sizes) {
    fprintf(stderr, "wrap_concat: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_concat: elem=%lld, outer=%lld, out_inner=%lld, n_in=%lld\n",
      (long long)element_size_bytes, (long long)outer,
      (long long)output_inner, (long long)num_inputs);

  int64_t num_out = outer * output_inner;
  int rc = hip_concat(stream, output, static_cast<int>(element_size_bytes),
                      outer, output_inner, static_cast<int>(num_inputs), inputs,
                      input_inner_sizes);
  return rc;
}
