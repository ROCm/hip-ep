/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A,B) = Q(DQ(A) [OP] DQ(B))
// OP belongs to {+,-,*,/}
//
// a = (A - z_a) * s_a
// b = (B - z_b) * s_b
// OUT = saturate(round((a [OP] b) / s_out) + z_out)
// --->
// let M_a = s_a/s_out, M_b = s_b/s_out
// OUT = saturate(round(M_a * (A - z_a) [OP] M_b * (B - z_b)) + z_out)
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int hipdnn_to_hip_dtype_qelem(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  default:
    return -1;
  }
}

int wrap_qelementwise(RuntimeState *state, void *lhs, void *rhs, void *output,
                      int64_t kind, const int64_t *lhs_shape, int64_t lhs_rank,
                      const int64_t *rhs_shape, int64_t rhs_rank,
                      const int64_t *out_shape, int64_t out_rank,
                      int64_t data_type, float M_a, int64_t lhs_zp, float M_b,
                      int64_t rhs_zp, int64_t output_zp) {
  if (!state || !lhs || !rhs || !output) {
    fprintf(stderr, "wrap_qelementwise: null tensor argument\n");
    return -1;
  }
  if ((lhs_rank > 0 && !lhs_shape) || (rhs_rank > 0 && !rhs_shape) ||
      (out_rank > 0 && !out_shape)) {
    fprintf(stderr,
            "wrap_qelementwise: null shape argument with non-zero rank\n");
    return -1;
  }

  int hip_dtype = hipdnn_to_hip_dtype_qelem(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_qelementwise: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  if (hipdnn_ep_debug_enabled()) {
    auto dump_shape = [](const char *name, const int64_t *shape, int64_t rank) {
      fprintf(stderr, "[REAL] wrap_qelementwise: %s rank=%lld shape=[", name,
              (long long)rank);
      for (int64_t i = 0; i < rank; ++i)
        fprintf(stderr, "%s%lld", i ? "," : "", (long long)shape[i]);
      fprintf(stderr, "]\n");
    };
    fprintf(stderr,
            "[REAL] wrap_qelementwise: kind=%lld dtype=%s(%lld) "
            "M_a=%g M_b=%g zp=(%lld,%lld,%lld)\n",
            (long long)kind, hipdnn_ep_datatype_name(data_type),
            (long long)data_type, (double)M_a, (double)M_b, (long long)lhs_zp,
            (long long)rhs_zp, (long long)output_zp);
    dump_shape("lhs", lhs_shape, lhs_rank);
    dump_shape("rhs", rhs_shape, rhs_rank);
    dump_shape("output", out_shape, out_rank);
  }

  return hip_qelementwise(stream, lhs, rhs, output, kind, lhs_shape, lhs_rank,
                          rhs_shape, rhs_rank, out_shape, out_rank, hip_dtype,
                          M_a, lhs_zp, M_b, rhs_zp, output_zp);
}
