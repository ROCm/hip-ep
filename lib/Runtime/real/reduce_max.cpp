/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ReduceMax: y = max(x) over the reduce axes.
//
// Port note: the HipToLLVM lowering for hip.reduce_* passes only
// `data_num_elements`, `output_num_elements` and the `axes` buffer pointer.
// The actual reduce axes are NOT inspected here -- the kernel assumes the
// upstream lowering has arranged for the reduce dims to occupy the trailing
// portion of `data` so that
//      reduce_size = data_num_elements / output_num_elements
// is correct. This mirrors how wrap_reduce_sum / hip_reduce_sum already work.
//
// Source: onnxruntime/core/providers/cuda/reduction/reduction_ops.cc /
//         reduction_functions.cu @ v1.22.2 (CudaT type-min initializer +
//         max operator) -- simplified to the block-per-output pattern that
//         reduce_sum_kernel.hip already establishes.
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <hip/hip_runtime.h>

static int reduce_max_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_reduce_max(RuntimeState *state, void *data, void *axes, void *output,
                    int64_t data_num_elements, int64_t output_num_elements,
                    int64_t axes_num_elements, int64_t data_type,
                    int64_t keepdims, int64_t noop_with_empty_axes,
                    int64_t inner_size) {
  OP_PROFILE(
      "reduce_max",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lld->%lld:%s", (long long)data_num_elements,
                 (long long)output_num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)keepdims;

  if (!state || !data || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_max: null argument\n");
    return -1;
  }

  // Empty-axes shortcut: copy input -> output, matching reduce_sum.
  if (axes_num_elements == 0 && noop_with_empty_axes == 1) {
    void *stream = hipdnn_ep_state_get_stream(state);
    int64_t element_size_bytes = hipdnn_ep_datatype_size(data_type);
    if (element_size_bytes < 0) {
      fprintf(stderr,
              "[REAL] wrap_reduce_max: unsupported data_type=%lld for noop\n",
              (long long)data_type);
      return -1;
    }
    int64_t total_bytes = data_num_elements * element_size_bytes;
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_reduce_max: noop_with_empty_axes=1, copying %lld bytes\n",
        (long long)total_bytes);
    hipError_t err =
        hipMemcpyAsync(output, data, total_bytes, hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(stream));
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_reduce_max: noop hipMemcpyAsync failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    return 0;
  }

  int hip_dtype = reduce_max_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_reduce_max: unsupported data_type=%s(%lld) "
            "(supported: f16, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  {
    const int fb_rc = hipdnn_cpu_fb_try_reduce_flat(
        state, stream, "ReduceMax", data, axes, output, data_num_elements,
        axes_num_elements, output_num_elements, data_type, keepdims,
        noop_with_empty_axes);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_reduce_max: data_num=%lld, output_num=%lld, "
                    "data_type=%s, hip_dtype=%d -> hip_reduce_max\n",
                    (long long)data_num_elements,
                    (long long)output_num_elements,
                    hipdnn_ep_datatype_name(data_type), hip_dtype);

  return hip_reduce_max(stream, data, output, data_num_elements,
                        output_num_elements, inner_size, hip_dtype);
}
