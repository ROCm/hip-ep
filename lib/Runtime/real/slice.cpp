/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for the strided slice HIP kernel.  The compiler computes
// per-axis input strides and start offsets at lowering time and passes them
// in as i64 arrays.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

extern "C" int wrap_slice(RuntimeState *state, void *input, void *output,
                          int64_t num_elements, int64_t element_size_bytes,
                          int64_t rank, const int64_t *out_shape,
                          const int64_t *in_strides_elems,
                          const int64_t *starts_elems) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_slice: null argument\n");
    return -1;
  }
  if (!out_shape || !in_strides_elems || !starts_elems) {
    fprintf(stderr, "wrap_slice: null shape/stride array\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_slice: n=%lld, elem=%lld, rank=%lld\n",
      (long long)num_elements, (long long)element_size_bytes, (long long)rank);

  int rc = hip_slice(stream, input, output, num_elements,
                     static_cast<int>(element_size_bytes),
                     static_cast<int>(rank), out_shape, in_strides_elems,
                     starts_elems);
  if (rc == 0)
    nan_trace_check("slice", output, num_elements);
  return rc;
}
