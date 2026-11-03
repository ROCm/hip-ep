/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// wrap_qswiglu / wrap_fake_quantize: per-tensor Q/DQ chains over f32
// activations collapsed into one launch each. See hipdnn_ep_runtime.h.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

bool inU16(int64_t v) { return v >= 0 && v <= 65535; }

} // namespace

int wrap_qswiglu(RuntimeState *state, const void *gate, const void *up,
                 void *output, int64_t num_elements, float gate_scale,
                 int64_t gate_zp, float sigmoid_scale, int64_t sigmoid_zp,
                 float m_silu, int64_t silu_zp, float up_scale, int64_t up_zp,
                 float m_mul, int64_t mul_zp, float dq_scale, int64_t dq_zp) {
  OP_PROFILE(
      "qswiglu",
      [&] { return std::to_string(static_cast<long long>(num_elements)); },
      state);

  if (!state || !gate || !up || !output) {
    fprintf(stderr, "[REAL] wrap_qswiglu: null argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  if (!inU16(gate_zp) || !inU16(sigmoid_zp) || !inU16(silu_zp) ||
      !inU16(up_zp) || !inU16(mul_zp) || !inU16(dq_zp)) {
    fprintf(stderr, "[REAL] wrap_qswiglu: zero point out of UINT16 range\n");
    return -1;
  }
  if (gate_scale == 0.0f || sigmoid_scale == 0.0f || up_scale == 0.0f) {
    fprintf(stderr, "[REAL] wrap_qswiglu: zero quantization scale\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qswiglu: n=%lld gate=(%g,%lld) "
                    "sigmoid=(%g,%lld) M_silu=%g silu_zp=%lld up=(%g,%lld) "
                    "M_mul=%g mul_zp=%lld dq=(%g,%lld)\n",
                    (long long)num_elements, (double)gate_scale,
                    (long long)gate_zp, (double)sigmoid_scale,
                    (long long)sigmoid_zp, (double)m_silu, (long long)silu_zp,
                    (double)up_scale, (long long)up_zp, (double)m_mul,
                    (long long)mul_zp, (double)dq_scale, (long long)dq_zp);

  return hip_qswiglu(
      hipdnn_ep_state_get_stream(state), static_cast<const float *>(gate),
      static_cast<const float *>(up), static_cast<float *>(output),
      num_elements, gate_scale, static_cast<int32_t>(gate_zp), sigmoid_scale,
      static_cast<int32_t>(sigmoid_zp), m_silu, static_cast<int32_t>(silu_zp),
      up_scale, static_cast<int32_t>(up_zp), m_mul,
      static_cast<int32_t>(mul_zp), dq_scale, static_cast<int32_t>(dq_zp));
}

int wrap_fake_quantize(RuntimeState *state, const void *input, void *output,
                       int64_t num_elements, float scale, int64_t zero_point,
                       int64_t qmin, int64_t qmax) {
  OP_PROFILE(
      "fake_quantize",
      [&] { return std::to_string(static_cast<long long>(num_elements)); },
      state);

  if (!state || !input || !output) {
    fprintf(stderr, "[REAL] wrap_fake_quantize: null argument\n");
    return -1;
  }
  if (num_elements <= 0)
    return 0;
  if (scale == 0.0f || qmin > qmax || qmin < INT32_MIN || qmax > INT32_MAX ||
      zero_point < qmin || zero_point > qmax) {
    fprintf(stderr,
            "[REAL] wrap_fake_quantize: invalid parameters scale=%g zp=%lld "
            "range=[%lld,%lld]\n",
            (double)scale, (long long)zero_point, (long long)qmin,
            (long long)qmax);
    return -1;
  }

  return hip_fake_quantize(
      hipdnn_ep_state_get_stream(state), static_cast<const float *>(input),
      static_cast<float *>(output), num_elements, scale,
      static_cast<int32_t>(zero_point), static_cast<int32_t>(qmin),
      static_cast<int32_t>(qmax));
}
