/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

// onnx.QLinearMatMul. Rank-2 8-bit matrix product, per-tensor scales and
// zero points. Not wrap_qmatmul: that entry point is the QDQ fusion.

static int hipdnn_to_hip_dtype_qlinear_mm(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  default:
    return -1;
  }
}

static int check_per_tensor(const char *name, int64_t count) {
  if (count == 1)
    return 0;
  fprintf(stderr,
          "[REAL] wrap_qlinear_matmul: %s count %lld is not per-tensor\n", name,
          (long long)count);
  return -1;
}

int wrap_qlinear_matmul(RuntimeState *state, const void *a, const void *a_scale,
                        const void *a_zero_point, const void *b,
                        const void *b_scale, const void *b_zero_point,
                        const void *y_scale, const void *y_zero_point, void *y,
                        int64_t m, int64_t k, int64_t n, int64_t a_dtype,
                        int64_t b_dtype, int64_t y_dtype, int64_t a_scale_count,
                        int64_t b_scale_count, int64_t y_scale_count,
                        int64_t a_zp_count, int64_t b_zp_count,
                        int64_t y_zp_count) {
  OP_PROFILE(
      "qlinear_matmul",
      [&] {
        char buf[96];
        snprintf(buf, sizeof(buf), "%lldx%lldx%lld,%s", (long long)m,
                 (long long)k, (long long)n, hipdnn_ep_datatype_name(a_dtype));
        return std::string(buf);
      },
      state);

  if (!state || !a || !a_scale || !a_zero_point || !b || !b_scale ||
      !b_zero_point || !y_scale || !y_zero_point || !y) {
    fprintf(stderr, "[REAL] wrap_qlinear_matmul: null argument\n");
    return -1;
  }
  if (m < 1 || k < 1 || n < 1) {
    fprintf(stderr, "[REAL] wrap_qlinear_matmul: non-positive geometry\n");
    return -1;
  }

  int hip_a = hipdnn_to_hip_dtype_qlinear_mm(a_dtype);
  int hip_b = hipdnn_to_hip_dtype_qlinear_mm(b_dtype);
  int hip_y = hipdnn_to_hip_dtype_qlinear_mm(y_dtype);
  if (hip_a < 0 || hip_b < 0 || hip_y < 0) {
    fprintf(stderr,
            "[REAL] wrap_qlinear_matmul: unsupported dtypes a=%lld b=%lld "
            "y=%lld\n",
            (long long)a_dtype, (long long)b_dtype, (long long)y_dtype);
    return -1;
  }
  if (check_per_tensor("a scale", a_scale_count) ||
      check_per_tensor("b scale", b_scale_count) ||
      check_per_tensor("y scale", y_scale_count) ||
      check_per_tensor("a zero point", a_zp_count) ||
      check_per_tensor("b zero point", b_zp_count) ||
      check_per_tensor("y zero point", y_zp_count))
    return -1;

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_qlinear_matmul: M=%lld K=%lld N=%lld\n",
                    (long long)m, (long long)k, (long long)n);

  int rc = hip_qlinear_matmul(stream, a, a_scale, a_zero_point, b, b_scale,
                              b_zero_point, y_scale, y_zero_point, y, m, k, n,
                              hip_a, hip_b, hip_y);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qlinear_matmul: kernel launch failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}
