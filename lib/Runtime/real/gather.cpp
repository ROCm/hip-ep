/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>
#include <vector>

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

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_gather: axis=%lld, data_num=%lld, indices_num=%lld, "
      "output_num=%lld, axis_size=%lld, inner=%lld, elem_size=%lld, "
      "idx_elem_size=%lld -> calling hip_gather\n",
      (long long)axis, (long long)data_num_elements,
      (long long)indices_num_elements, (long long)output_num_elements,
      (long long)axis_size, (long long)inner_size,
      (long long)element_size_bytes, (long long)indices_element_size_bytes);

  const bool want_cpu_fb =
      state->cpu_fallback.invoke &&
      hipdnn_ep_debug_cpu_fallback_ops_contains("Gather");

  if (want_cpu_fb) {
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
    if (!shape_dims_ok(data_shape, data_rank) ||
        !shape_dims_ok(indices_shape, indices_rank) ||
        !shape_dims_ok(output_shape, output_rank)) {
      RUNTIME_DEBUG_LOG(
          "[REAL] wrap_gather: invalid shape (negative dim); skip CPU "
          "fallback\n");
    } else if (wrap_hipStreamSynchronize(stream) != 0) {
      return -1;
    } else {
      const int64_t data_bytes = data_num_elements * element_size_bytes;
      const int64_t indices_bytes =
          indices_num_elements * indices_element_size_bytes;
      const int64_t output_bytes = output_num_elements * element_size_bytes;
      if (data_bytes < 0 || indices_bytes < 0 || output_bytes < 0) {
        RUNTIME_DEBUG_LOG("[REAL] wrap_gather: negative byte size (cpu fb)\n");
        return -1;
      }

      std::vector<char> host_data(static_cast<size_t>(data_bytes));
      std::vector<char> host_output(static_cast<size_t>(output_bytes));
      // ORT CreateTensor reads indices with native element alignment; staging
      // int32/int64 in typed vectors avoids UB from std::vector<char>::data().
      std::vector<int64_t> host_indices_i64;
      std::vector<int32_t> host_indices_i32;
      std::vector<char> host_indices_raw;
      void *indices_host = nullptr;
      if (indices_element_size_bytes == 8) {
        host_indices_i64.resize(
            static_cast<size_t>(indices_num_elements));
        indices_host = host_indices_i64.data();
      } else if (indices_element_size_bytes == 4) {
        host_indices_i32.resize(
            static_cast<size_t>(indices_num_elements));
        indices_host = host_indices_i32.data();
      } else {
        host_indices_raw.resize(static_cast<size_t>(indices_bytes));
        indices_host = host_indices_raw.data();
      }

      if (wrap_hipMemcpyD2H(host_data.data(), data, data_bytes, stream) != 0)
        return -1;
      if (wrap_hipMemcpyD2H(indices_host, indices, indices_bytes, stream) != 0)
        return -1;
      if (wrap_hipStreamSynchronize(stream) != 0)
        return -1;

      HipdnnCpuFbGatherDesc desc{};
      desc.axis = axis;
      desc.data_hip_dtype = data_hip_dtype;
      desc.indices_hip_dtype = indices_hip_dtype;
      desc.indices_element_size_bytes = indices_element_size_bytes;
      desc.data_rank = data_rank;
      desc.data_shape = data_shape;
      desc.data_host = host_data.data();
      desc.indices_rank = indices_rank;
      desc.indices_shape = indices_shape;
      desc.indices_host = indices_host;
      desc.output_rank = output_rank;
      desc.output_shape = output_shape;
      desc.output_host = host_output.data();
      desc.data_num_elements = data_num_elements;
      desc.indices_num_elements = indices_num_elements;
      desc.output_num_elements = output_num_elements;

      const int fb_rc = state->cpu_fallback.invoke(
          state->cpu_fallback.user, state, HIPDNN_CPU_FB_OP_GATHER, &desc,
          sizeof(desc));
      if (fb_rc != 0) {
        fprintf(stderr,
                "[REAL] wrap_gather: CPU fallback invoke failed rc=%d; "
                "falling back to GPU kernel\n",
                fb_rc);
      } else {
        if (wrap_hipMemcpyH2D(output, host_output.data(), output_bytes,
                              stream) != 0)
          return -1;
        return 0;
      }
    }
  }

  return hip_gather(stream, data, indices, output, axis, data_num_elements,
                    indices_num_elements, output_num_elements, axis_size,
                    inner_size, static_cast<int>(element_size_bytes),
                    static_cast<int>(indices_element_size_bytes));
}
