/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// wrap_qsigmoid: Q(sigmoid(DQ(x))) for UINT16 per-tensor QDQ.
//
//   dq  = (x - zp_in) * scale_in
//   y   = 1 / (1 + exp(-dq))
//   out = saturate(round(y / scale_out) + zp_out)

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

int wrap_qsigmoid(RuntimeState *state, const void *input, void *output,
                  int64_t num_elements, int64_t data_type, float input_scale,
                  int64_t input_zp, float output_scale, int64_t output_zp) {
  OP_PROFILE(
      "qsigmoid",
      [&] {
        char b[48];
        snprintf(b, sizeof(b), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: null argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  if (data_type != HIPDNN_EP_DATATYPE_UINT16) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: expected UINT16, got %s\n",
            hipdnn_ep_datatype_name(data_type));
    return -1;
  }
  if (output_scale == 0.0f) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: output_scale is 0\n");
    return -1;
  }
  if (input_zp < 0 || input_zp > 65535 || output_zp < 0 || output_zp > 65535) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: zp out of UINT16 range\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int rc = hip_qsigmoid(stream, input, output, num_elements, HIP_DTYPE_UINT16,
                        input_scale, static_cast<int32_t>(input_zp),
                        output_scale, static_cast<int32_t>(output_zp));
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: qsigmoid failed (%d)\n", rc);
    return rc;
  }
  return 0;
}
