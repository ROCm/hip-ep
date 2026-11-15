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
#include <string>

// wrap_roi_align(state, input, rois, batch_indices, output,
//                data_type, N, C, inH, inW, num_rois, outH, outW,
//                sampling_ratio, spatial_scale, half_pixel, mode)
//
// mode:       0 = avg, 1 = max
// half_pixel: 1 = "half_pixel", 0 = "output_half_pixel"

static int hipdnn_ep_to_hip_dtype_roi_align(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  default:
    return -1;
  }
}

HIPDNN_EP_RT_EXPORT int
wrap_roi_align(RuntimeState *state, void *input, void *rois,
               void *batch_indices, void *output, int64_t data_type, int64_t n,
               int64_t c, int64_t in_h, int64_t in_w, int64_t num_rois,
               int64_t out_h, int64_t out_w, int64_t sampling_ratio,
               float spatial_scale, int64_t half_pixel, int64_t mode) {
  OP_PROFILE(
      "roialign",
      [&] {
        char b[192];
        snprintf(b, sizeof(b),
                 "N=%lld C=%lld in=%lldx%lld rois=%lld out=%lldx%lld "
                 "sr=%lld mode=%lld",
                 (long long)n, (long long)c, (long long)in_h, (long long)in_w,
                 (long long)num_rois, (long long)out_h, (long long)out_w,
                 (long long)sampling_ratio, (long long)mode);
        return std::string(b);
      },
      state);

  if (num_rois <= 0 || c <= 0 || out_h <= 0 || out_w <= 0)
    return 0;
  if (!state || !input || !rois || !batch_indices || !output) {
    fprintf(stderr, "[REAL] wrap_roi_align: null argument\n");
    return -1;
  }
  int hip_dtype = hipdnn_ep_to_hip_dtype_roi_align(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_roi_align: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_roi_align: dtype=%lld N=%lld C=%lld in=%lldx%lld "
      "rois=%lld out=%lldx%lld sr=%lld scale=%f half_pixel=%lld mode=%lld\n",
      (long long)data_type, (long long)n, (long long)c, (long long)in_h,
      (long long)in_w, (long long)num_rois, (long long)out_h, (long long)out_w,
      (long long)sampling_ratio, (double)spatial_scale, (long long)half_pixel,
      (long long)mode);

  int rc = hip_roi_align(
      stream, input, rois, batch_indices, output, n, c, in_h, in_w, num_rois,
      out_h, out_w, static_cast<int>(sampling_ratio), spatial_scale,
      static_cast<int>(half_pixel), static_cast<int>(mode), hip_dtype);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_roi_align: kernel launch failed (%d)\n", rc);
    return -1;
  }
  return 0;
}
