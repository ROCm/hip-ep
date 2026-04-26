/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_where (cond ? x : y).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

extern "C" int wrap_where(RuntimeState *state, void *cond, void *x, void *y,
                          void *out, int64_t num_elements,
                          int64_t element_size_bytes, int64_t rank,
                          const int64_t *out_shape,
                          const int64_t *cond_strides_elems,
                          const int64_t *x_strides_elems,
                          const int64_t *y_strides_elems) {
  if (!state || !cond || !x || !y || !out) {
    fprintf(stderr, "wrap_where: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_where: elem=%lld, n=%lld, rank=%lld\n",
      (long long)element_size_bytes, (long long)num_elements, (long long)rank);

  int rc = hip_where(stream, cond, x, y, out, num_elements,
                     static_cast<int>(element_size_bytes),
                     static_cast<int>(rank), out_shape, cond_strides_elems,
                     x_strides_elems, y_strides_elems);
  if (rc == 0)
    nan_trace_check("where", out, num_elements);
  return rc;
}
