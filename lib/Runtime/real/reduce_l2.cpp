/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static int hipdnn_to_hip_dtype_l2(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  default:
    return -1;
  }
}

int wrap_reduce_l2(RuntimeState *state, void *data, void *axes, void *output,
                   int64_t data_num_elements, int64_t output_num_elements,
                   int64_t axes_num_elements, int64_t data_type,
                   int64_t keepdims, int64_t noop_with_empty_axes,
                   int64_t inner_size) {
  OP_PROFILE(
      "reduce_l2",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lld->%lld", (long long)data_num_elements,
                 (long long)output_num_elements);
        return std::string(b);
      },
      state);
  if (!state || !data || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_l2: null argument\n");
    return -1;
  }

  if (axes_num_elements == 0 && noop_with_empty_axes == 1) {
    if (data_num_elements != output_num_elements) {
      fprintf(stderr,
              "[REAL] wrap_reduce_l2: noop path shape mismatch "
              "data_num_elements=%lld output_num_elements=%lld\n",
              (long long)data_num_elements, (long long)output_num_elements);
      return -1;
    }
    void *stream = hipdnn_ep_state_get_stream(state);
    int64_t element_size_bytes = hipdnn_ep_datatype_size(data_type);
    if (element_size_bytes < 0) {
      fprintf(stderr,
              "[REAL] wrap_reduce_l2: unsupported data_type=%lld for noop "
              "memcpy\n",
              (long long)data_type);
      return -1;
    }
    int64_t total_bytes = output_num_elements * element_size_bytes;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_reduce_l2: noop_with_empty_axes=1 with empty axes, "
        "copying %lld bytes (data_type=%s)\n",
        (long long)total_bytes, hipdnn_ep_datatype_name(data_type));
    HIP_CHECK(hipMemcpyAsync(output, data, total_bytes, hipMemcpyDeviceToDevice,
                             static_cast<hipStream_t>(stream)));
    return 0;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  int hip_dtype = hipdnn_to_hip_dtype_l2(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_reduce_l2: unsupported data_type=%s(%lld) "
            "(supported: f16, f32)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_reduce_l2: data_num=%lld, output_num=%lld, "
      "axes_num=%lld, data_type=%s(%lld), keepdims=%lld, "
      "noop_with_empty_axes=%lld, hip_dtype=%d -> calling hip_reduce_l2\n",
      (long long)data_num_elements, (long long)output_num_elements,
      (long long)axes_num_elements, hipdnn_ep_datatype_name(data_type),
      (long long)data_type, (long long)keepdims,
      (long long)noop_with_empty_axes, hip_dtype);

  return hip_reduce_l2(stream, data, output, data_num_elements,
                       output_num_elements, inner_size, hip_dtype);
}
