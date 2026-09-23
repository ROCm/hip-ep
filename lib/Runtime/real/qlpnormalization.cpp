/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// wrap_qlpnormalization: Q(LpNormalization(DQ(x))) for UINT16, p=2, last axis.
//
//   dq = (x - zp_in) * scale_in
//   y  = dq / ||dq||_2  ==  RMS(dq, scale=1/sqrt(N), epsilon=0)
//   out = saturate(round(y / scale_out) + zp_out)

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

int wrap_qlpnormalization(RuntimeState *state, const void *input, void *output,
                          int64_t num_elements, int64_t norm_num_elements,
                          int64_t data_type, float input_scale,
                          int64_t input_zp, float output_scale,
                          int64_t output_zp, int64_t axis, int64_t p) {
  OP_PROFILE(
      "qlpnormalization",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(norm_num_elements > 0
                                 ? num_elements / norm_num_elements
                                 : 0),
                 (long long)norm_num_elements);
        return std::string(b);
      },
      state);

  (void)axis;
  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: null argument\n");
    return -1;
  }
  if (p != 2) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: only p=2 is supported\n");
    return -1;
  }
  if (data_type != HIPDNN_EP_DATATYPE_UINT16) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: expected UINT16, got %s\n",
            hipdnn_ep_datatype_name(data_type));
    return -1;
  }
  if (num_elements <= 0 || norm_num_elements <= 0 ||
      num_elements % norm_num_elements != 0) {
    fprintf(stderr,
            "[REAL] wrap_qlpnormalization: bad extents numel=%lld N=%lld\n",
            (long long)num_elements, (long long)norm_num_elements);
    return -1;
  }
  if (output_scale == 0.0f) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: output_scale is 0\n");
    return -1;
  }
  if (input_zp < 0 || input_zp > 65535 || output_zp < 0 || output_zp > 65535) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: zp out of UINT16 range\n");
    return -1;
  }

  const int64_t num_rows = num_elements / norm_num_elements;
  void *stream = hipdnn_ep_state_get_stream(state);

  int rc = hip_qlpnormalization(
      stream, input, output, num_rows, norm_num_elements,
      1.0f / std::sqrt(static_cast<float>(norm_num_elements)), input_scale,
      static_cast<int32_t>(input_zp), output_scale,
      static_cast<int32_t>(output_zp), HIP_DTYPE_UINT16);
  if (rc != 0) {
    fprintf(stderr,
            "[REAL] wrap_qlpnormalization: qlpnormalization failed "
            "(%d)\n",
            rc);
    return rc;
  }
  return 0;
}
