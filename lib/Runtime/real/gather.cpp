/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

int wrap_gather(RuntimeState *state, void *data, void *indices, void *output,
                int64_t axis, int64_t data_num_elements,
                int64_t indices_num_elements, int64_t output_num_elements,
                int64_t axis_size, int64_t inner_size,
                int64_t element_size_bytes, int64_t indices_element_size_bytes,
                const int64_t *data_shape, int64_t data_rank,
                const int64_t *indices_shape, int64_t indices_rank,
                const int64_t *output_shape, int64_t output_rank,
                int64_t data_hip_dtype, int64_t indices_hip_dtype) {
  OP_PROFILE(
      "gather",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld", (long long)output_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !data || !indices || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_gather: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  if (state->cpu_fallback.invoke &&
      hipdnn_ep_debug_cpu_fallback_ops_contains("Gather")) {
    auto shape_dims_ok = [](const int64_t *shape, int64_t rank) -> bool {
      if (rank <= 0)
        return true;
      if (!shape)
        return false;
      for (int64_t i = 0; i < rank; ++i) {
        if (shape[i] < 0)
          return false;
      }
      return true;
    };
    if (shape_dims_ok(data_shape, data_rank) &&
        shape_dims_ok(indices_shape, indices_rank) &&
        shape_dims_ok(output_shape, output_rank)) {
      const int fb = hipdnn_cpu_fb_try_gather(
          state, stream, data, indices, output, axis, data_rank, data_shape,
          data_num_elements, indices_rank, indices_shape, indices_num_elements,
          output_rank, output_shape, output_num_elements, data_hip_dtype,
          indices_hip_dtype);
      if (fb == 0) {
        return 0;
      }
      if (fb < 0) {
        return -1;
      }
    }
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather: axis=%lld, data_num=%lld, indices_num=%lld, "
      "output_num=%lld, axis_size=%lld, inner=%lld, elem_size=%lld, "
      "idx_elem_size=%lld -> calling hip_gather\n",
      (long long)axis, (long long)data_num_elements,
      (long long)indices_num_elements, (long long)output_num_elements,
      (long long)axis_size, (long long)inner_size,
      (long long)element_size_bytes, (long long)indices_element_size_bytes);

  return hip_gather(stream, data, indices, output, axis, data_num_elements,
                    indices_num_elements, output_num_elements, axis_size,
                    inner_size, static_cast<int>(element_size_bytes),
                    static_cast<int>(indices_element_size_bytes));
}
