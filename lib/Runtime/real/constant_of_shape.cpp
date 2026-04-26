/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_constant_of_shape.  The compiler embeds the scalar
// fill value in the lowered call as a 64-bit blob so the runtime doesn't
// need to introspect tensor types.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>

extern "C" int wrap_constant_of_shape(RuntimeState *state, void *output,
                                      int64_t num_elements,
                                      int64_t element_size_bytes,
                                      uint64_t scalar_bits) {
  if (!state || !output) {
    fprintf(stderr, "wrap_constant_of_shape: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_constant_of_shape: n=%lld, elem=%lld, bits=0x%llx\n",
      (long long)num_elements, (long long)element_size_bytes,
      (unsigned long long)scalar_bits);

  int rc = hip_constant_of_shape(stream, output, num_elements,
                                 static_cast<int>(element_size_bytes),
                                 scalar_bits);
  if (rc == 0)
    nan_trace_check("const_of_shape", output, num_elements);
  return rc;
}
