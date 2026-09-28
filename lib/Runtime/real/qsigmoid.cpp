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
  const size_t f32_bytes = static_cast<size_t>(num_elements) * sizeof(float);
  const size_t scratch_bytes =
      f32_bytes + 2 * sizeof(float) + 2 * sizeof(uint16_t);
  if (hipdnn_ep_state_ensure_qsigmoid_scratch(state, scratch_bytes) != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: scratch allocation failed\n");
    return -1;
  }

  auto *scratch =
      static_cast<uint8_t *>(hipdnn_ep_state_get_qsigmoid_scratch(state));
  void *dq = scratch;
  void *in_scale = scratch + f32_bytes;
  void *out_scale = static_cast<uint8_t *>(in_scale) + sizeof(float);
  void *in_zp = static_cast<uint8_t *>(out_scale) + sizeof(float);
  void *out_zp = static_cast<uint8_t *>(in_zp) + sizeof(uint16_t);

  int rc = hip_qsigmoid_prepare_params(stream, in_scale, input_scale, out_scale,
                                       output_scale, in_zp,
                                       static_cast<uint16_t>(input_zp), out_zp,
                                       static_cast<uint16_t>(output_zp));
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: parameter preparation failed (%d)\n",
            rc);
    return rc;
  }

  const int64_t flat_shape[1] = {num_elements};
  const int64_t scalar_shape[1] = {1};

  rc = hip_dequantize_linear(
      stream, input, in_scale, in_zp, dq, flat_shape, 1, scalar_shape, 0,
      /*axis=*/0, /*block_size=*/0, HIP_DTYPE_UINT16, HIP_DTYPE_FLOAT32,
      HIP_DTYPE_FLOAT32, /*in_bits=*/16);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: dequant failed (%d)\n", rc);
    return rc;
  }

  rc = hip_elementwise_sigmoid(stream, dq, dq, num_elements, HIP_DTYPE_FLOAT32);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: sigmoid failed (%d)\n", rc);
    return rc;
  }

  rc = hip_quantize_linear(
      stream, dq, out_scale, out_zp, output, flat_shape, 1, scalar_shape, 0,
      /*axis=*/0, /*block_size=*/0, /*precision=*/0, HIP_DTYPE_FLOAT32,
      HIP_DTYPE_FLOAT32, HIP_DTYPE_UINT16, /*out_bits=*/16);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qsigmoid: quant failed (%d)\n", rc);
    return rc;
  }
  return 0;
}
