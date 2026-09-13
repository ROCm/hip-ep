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

  const size_t f32_bytes = static_cast<size_t>(num_elements) * sizeof(float);
  const size_t scale_bytes =
      static_cast<size_t>(norm_num_elements) * sizeof(float);
  const size_t scratch_bytes =
      2 * f32_bytes + scale_bytes + 2 * sizeof(float) + 2 * sizeof(uint16_t);
  if (hipdnn_ep_state_ensure_qlpnormalization_scratch(state, scratch_bytes) !=
      0) {
    fprintf(stderr,
            "[REAL] wrap_qlpnormalization: scratch allocation failed\n");
    return -1;
  }

  auto *scratch = static_cast<uint8_t *>(
      hipdnn_ep_state_get_qlpnormalization_scratch(state));
  void *dq = scratch;
  void *rms = scratch + f32_bytes;
  void *scale_vec = scratch + 2 * f32_bytes;
  void *in_scale = scratch + 2 * f32_bytes + scale_bytes;
  void *out_scale = static_cast<uint8_t *>(in_scale) + sizeof(float);
  void *in_zp = static_cast<uint8_t *>(out_scale) + sizeof(float);
  void *out_zp = static_cast<uint8_t *>(in_zp) + sizeof(uint16_t);

  int rc = hip_qlpnormalization_prepare_params(
      stream, scale_vec, norm_num_elements,
      1.0f / std::sqrt(static_cast<float>(norm_num_elements)), in_scale,
      input_scale, out_scale, output_scale, in_zp,
      static_cast<uint16_t>(input_zp), out_zp,
      static_cast<uint16_t>(output_zp));
  if (rc != 0) {
    fprintf(stderr,
            "[REAL] wrap_qlpnormalization: parameter preparation failed "
            "(%d)\n",
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
    fprintf(stderr, "[REAL] wrap_qlpnormalization: dequant failed (%d)\n", rc);
    return rc;
  }

  rc = hip_rms_norm(stream, dq, scale_vec, rms, num_rows, norm_num_elements,
                    /*scale_rows=*/1, /*epsilon=*/0.0f, HIP_DTYPE_FLOAT32);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: rms_norm failed (%d)\n", rc);
    return rc;
  }

  rc = hip_quantize_linear(
      stream, rms, out_scale, out_zp, output, flat_shape, 1, scalar_shape, 0,
      /*axis=*/0, /*block_size=*/0, /*precision=*/0, HIP_DTYPE_FLOAT32,
      HIP_DTYPE_FLOAT32, HIP_DTYPE_UINT16, /*out_bits=*/16);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qlpnormalization: quant failed (%d)\n", rc);
    return rc;
  }
  return 0;
}
