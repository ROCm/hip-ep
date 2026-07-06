/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

static int64_t range_kernel_dtype_to_hipdnn(int64_t hip_dtype) {
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return HIPDNN_EP_DATATYPE_FLOAT;
  case HIP_DTYPE_FLOAT16:
    return HIPDNN_EP_DATATYPE_HALF;
  case HIP_DTYPE_INT64:
    return HIPDNN_EP_DATATYPE_INT64;
  case HIP_DTYPE_INT32:
    return HIPDNN_EP_DATATYPE_INT32;
  case HIP_DTYPE_FLOAT64:
    return HIPDNN_EP_DATATYPE_DOUBLE;
  default:
    return -1;
  }
}

int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype) {
  if (!state || !start || !limit || !delta || !output) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_range: null argument (state=%p start=%p limit=%p "
        "delta=%p output=%p output_num_elements=%lld hip_dtype=%lld)\n",
        (void *)state, start, limit, delta, output,
        (long long)output_num_elements, (long long)hip_dtype);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_range: output_num_elements=%lld, hip_dtype=%lld\n",
      (long long)output_num_elements, (long long)hip_dtype);

  {
    const int64_t fb_dtype = range_kernel_dtype_to_hipdnn(hip_dtype);
    if (fb_dtype < 0) {
      fprintf(stderr,
              "[REAL] wrap_range: unsupported hip_dtype=%lld for CPU fallback\n",
              (long long)hip_dtype);
      return -1;
    }
    int64_t out_shape[1] = {output_num_elements};
    HipdnnCpuFbGenericDesc fb{};
    fb.op_name = "Range";
    fb.opset = 11;
    fb.num_inputs = 3;
    fb.num_outputs = 1;
    fb.inputs[0] = {start, 0, nullptr, 1, fb_dtype};
    fb.inputs[1] = {limit, 0, nullptr, 1, fb_dtype};
    fb.inputs[2] = {delta, 0, nullptr, 1, fb_dtype};
    fb.outputs[0] = {output, 1, out_shape, output_num_elements, fb_dtype};
    const int fb_rc =
        hipdnn_cpu_fallback_try_generic(state, stream, "Range", &fb);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }

  void *deviceErrorFlag = hipdnn_ep_state_get_error_flag_device_ptr(state);
  return hip_range(stream, start, limit, delta, output, output_num_elements,
                   hip_dtype, deviceErrorFlag);
}
