/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime for com.microsoft.DequantizeLinear and com.microsoft.QuantizeLinear.
//
// ORCA 2-bit model uses W2A8: weights are 2-bit (handled by MatMulNBits),
// activations are 8-bit quantized with per-tensor scale/zp (these ops).
//
// DequantizeLinear:  out[i] = (float(in[i]) - float(zp)) * scale
// QuantizeLinear:    out[i] = clamp(round(in[i] / scale) + zp, -128, 127)
//
// Per-tensor (1D scale, axis ignored) is the common case for activation quant.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

extern "C" int wrap_ms_dequantize_linear(
    RuntimeState *state,
    const void *input,
    const void *scale,
    const void *zero_point,   // nullable
    void *output,
    int64_t n_elements,
    int64_t axis,
    int64_t n_channels,       // scale/zp length (1 for per-tensor)
    int64_t input_elem_size,  // bytes: 1=int8, 2=fp16, 4=fp32
    int64_t scale_elem_size)  // bytes: 2=fp16, 4=fp32
{
  OP_PROFILE(
      "ms_dequantize_linear",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld ch=%lld in=%lld sc=%lld",
                 (long long)n_elements, (long long)n_channels,
                 (long long)input_elem_size, (long long)scale_elem_size);
        return std::string(b);
      },
      state);

  if (!state || !input || !scale || !output) {
    fprintf(stderr, "[REAL] wrap_ms_dequantize_linear: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "[REAL] wrap_ms_dequantize_linear: null stream\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_ms_dequantize_linear: n=%lld axis=%lld ch=%lld "
      "in_bytes=%lld sc_bytes=%lld has_zp=%d\n",
      (long long)n_elements, (long long)axis, (long long)n_channels,
      (long long)input_elem_size, (long long)scale_elem_size,
      zero_point ? 1 : 0);

  return hip_ms_dequantize_linear(
      stream, input, scale, zero_point, output,
      n_elements, n_channels,
      input_elem_size, scale_elem_size);
}

extern "C" int wrap_ms_quantize_linear(
    RuntimeState *state,
    const void *input,
    const void *scale,
    const void *zero_point,    // nullable
    void *output,
    int64_t n_elements,
    int64_t axis,
    int64_t n_channels,
    int64_t input_elem_size,
    int64_t output_elem_size)
{
  OP_PROFILE(
      "ms_quantize_linear",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld ch=%lld in=%lld out=%lld",
                 (long long)n_elements, (long long)n_channels,
                 (long long)input_elem_size, (long long)output_elem_size);
        return std::string(b);
      },
      state);

  if (!state || !input || !scale || !output) {
    fprintf(stderr, "[REAL] wrap_ms_quantize_linear: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "[REAL] wrap_ms_quantize_linear: null stream\n");
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_ms_quantize_linear: n=%lld axis=%lld ch=%lld "
      "in_bytes=%lld out_bytes=%lld has_zp=%d\n",
      (long long)n_elements, (long long)axis, (long long)n_channels,
      (long long)input_elem_size, (long long)output_elem_size,
      zero_point ? 1 : 0);

  return hip_ms_quantize_linear(
      stream, input, scale, zero_point, output,
      n_elements, n_channels,
      input_elem_size, output_elem_size);
}
