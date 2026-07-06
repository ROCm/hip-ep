/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ReduceProd: y = prod(x) over the reduce axes.
//
// Same block-per-output structure as wrap_reduce_max / hip_reduce_max; only
// the init value (1) and the operator (*) differ. Both are encoded in
// reduce_sum_kernel.hip's templated `reduce_*_kernel<T, OP=OP_PROD>` path
// added by the ReduceMax commit.
//
// Source: onnxruntime/core/providers/cuda/reduction/reduction_ops.cc /
//         reduction_functions.cu @ v1.22.2.
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <hip/hip_runtime.h>

static int reduce_prod_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_reduce_prod(RuntimeState *state, void *data, void *axes, void *output,
                     int64_t data_num_elements, int64_t output_num_elements,
                     int64_t axes_num_elements, int64_t data_type,
                     int64_t keepdims, int64_t noop_with_empty_axes,
                     int64_t inner_size) {
  OP_PROFILE(
      "reduce_prod",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lld->%lld:%s", (long long)data_num_elements,
                 (long long)output_num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)axes;
  (void)keepdims;

  if (!state || !data || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_prod: null argument\n");
    return -1;
  }

  if (axes_num_elements == 0 && noop_with_empty_axes == 1) {
    void *stream = hipdnn_ep_state_get_stream(state);
    int64_t element_size_bytes = hipdnn_ep_datatype_size(data_type);
    if (element_size_bytes < 0) {
      fprintf(stderr,
              "[REAL] wrap_reduce_prod: unsupported data_type=%lld for noop\n",
              (long long)data_type);
      return -1;
    }
    int64_t total_bytes = data_num_elements * element_size_bytes;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_reduce_prod: noop_with_empty_axes=1, copying %lld bytes\n",
        (long long)total_bytes);
    hipError_t err =
        hipMemcpyAsync(output, data, total_bytes, hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(stream));
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_reduce_prod: noop hipMemcpyAsync failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    return 0;
  }

  int hip_dtype = reduce_prod_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_reduce_prod: unsupported data_type=%s(%lld) "
            "(supported: f16, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  {
    const int fb_rc = hipdnn_cpu_fb_try_reduce_flat(
        state, stream, "ReduceProd", data, axes, output, data_num_elements,
        axes_num_elements, output_num_elements, data_type, keepdims,
        noop_with_empty_axes);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_prod: data_num=%lld, output_num=%lld, "
                    "data_type=%s, hip_dtype=%d -> hip_reduce_prod\n",
                    (long long)data_num_elements,
                    (long long)output_num_elements,
                    hipdnn_ep_datatype_name(data_type), hip_dtype);

  return hip_reduce_prod(stream, data, output, data_num_elements,
                         output_num_elements, inner_size, hip_dtype);
}
