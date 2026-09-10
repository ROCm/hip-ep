/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(X) = Q(Sigmoid(DQ(X)))
//
// x = (X - x_zero_point) * x_scale
// y = 1 / (1 + exp(-x))
// Y = saturate(round(y / y_scale) + y_zero_point)
// --->
// let out_recip_scale = 1 / y_scale   (folded once by HipToLLVM lowering)
// Y = saturate(round(y * out_recip_scale) + y_zero_point)
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int hipdnn_to_hip_dtype_qact(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  default:
    return -1;
  }
}

// Notes on function reuse: every future family member (qtanh, qsoftplus,
// qgelu, ...) reuses this declaration unchanged -- only `kind` and the
// device-side math differ, so the ABI shape stays fixed as the family grows.
int wrap_qactivation(RuntimeState *state, void *input, void *output,
                     int64_t kind, int64_t num_elements, int64_t data_type,
                     float x_scale, int64_t x_zero_point, float out_recip_scale,
                     int64_t y_zero_point) {
  OP_PROFILE(
      hipdnn_ep_qactivation_kind_name(kind),
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%s:%lld", hipdnn_ep_datatype_name(data_type),
                 (long long)num_elements);
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    fprintf(stderr, "wrap_qactivation: null tensor argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  if (kind != HIPDNN_EP_QACTIVATION_SIGMOID) {
    fprintf(stderr, "wrap_qactivation: unsupported kind=%lld\n",
            (long long)kind);
    return -1;
  }

  int hip_dtype = hipdnn_to_hip_dtype_qact(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_qactivation: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qactivation: kind=%lld dtype=%s(%lld) num_elements=%lld "
      "x_scale=%g x_zp=%lld out_recip_scale=%g y_zp=%lld\n",
      (long long)kind, hipdnn_ep_datatype_name(data_type), (long long)data_type,
      (long long)num_elements, (double)x_scale, (long long)x_zero_point,
      (double)out_recip_scale, (long long)y_zero_point);

  return hip_qactivation(stream, input, output, kind, num_elements, hip_dtype,
                         x_scale, x_zero_point, out_recip_scale, y_zero_point);
}
