/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

int wrap_gated_delta_net(RuntimeState *state, const void *qkv,
                         const float *decay, const float *beta,
                         const float *initial_state, const float *a_log,
                         const float *dt_bias, void *output, float *final_state,
                         int64_t batch, int64_t seq_len, int64_t num_heads_qk,
                         int64_t num_heads_v, int64_t head_size_qk,
                         int64_t head_size_v, float scale,
                         int64_t element_size_bytes) {
  if (!state || !qkv || !decay || !beta || !initial_state || !a_log ||
      !dt_bias || !output || !final_state) {
    fprintf(stderr, "wrap_gated_delta_net: null required argument\n");
    return -1;
  }
  if (element_size_bytes != 2 && element_size_bytes != 4) {
    fprintf(stderr,
            "wrap_gated_delta_net: unsupported element_size %lld (expect 2 "
            "for fp16 or 4 for fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (batch < 1 || seq_len < 1 || head_size_qk < 1) {
    fprintf(stderr,
            "wrap_gated_delta_net: invalid shape batch=%lld seq_len=%lld "
            "dk=%lld\n",
            (long long)batch, (long long)seq_len, (long long)head_size_qk);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  if (!stream) {
    fprintf(stderr, "wrap_gated_delta_net: null stream\n");
    return -1;
  }
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(head_size_qk));
  }

  OP_PROFILE(
      "gated_delta_net",
      [&] {
        char b[80];
        snprintf(b, sizeof(b),
                 "b=%lld,sq=%lld,hqk=%lld,hv=%lld,dk=%lld,dv=%lld",
                 (long long)batch, (long long)seq_len, (long long)num_heads_qk,
                 (long long)num_heads_v, (long long)head_size_qk,
                 (long long)head_size_v);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG("[REAL] wrap_gated_delta_net: B=%lld L=%lld Hqk=%lld "
                    "Hv=%lld dk=%lld dv=%lld scale=%f elem=%lld\n",
                    (long long)batch, (long long)seq_len,
                    (long long)num_heads_qk, (long long)num_heads_v,
                    (long long)head_size_qk, (long long)head_size_v,
                    (double)scale, (long long)element_size_bytes);

  int rc = hip_gated_delta_net(
      stream, qkv, decay, beta, initial_state, a_log, dt_bias, output,
      final_state, batch, seq_len, num_heads_qk, num_heads_v, head_size_qk,
      head_size_v, scale, static_cast<int>(element_size_bytes));
  if (rc != 0) {
    fprintf(stderr, "wrap_gated_delta_net: hip_gated_delta_net failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}

int wrap_gated_delta_net_conv_norm(
    RuntimeState *state, void *proj, void *conv_weight, void *conv_bias,
    void *conv_state, void *gates, void *recurrent_state, void *a_log,
    void *dt_bias, void *norm_weight, void *output, void *conv_state_out,
    void *recurrent_state_out, void *conv_in, void *conv_out, void *decay,
    void *beta, void *gdn_out, void *gate_z, int64_t proj_num_elements,
    int64_t proj_stride, int64_t conv_state_num_elements, int64_t gate_stride,
    int64_t conv_channels, int64_t kernel_size, int64_t decay_column,
    int64_t beta_column, int64_t z_column, int64_t num_heads_qk,
    int64_t num_heads_v, int64_t head_size_qk, int64_t head_size_v, float scale,
    float epsilon) {
  if (!state || !proj || !conv_weight || !conv_bias || !conv_state || !gates ||
      !recurrent_state || !a_log || !dt_bias || !norm_weight || !output ||
      !conv_state_out || !recurrent_state_out || !conv_in || !conv_out ||
      !decay || !beta || !gdn_out || !gate_z) {
    fprintf(stderr, "wrap_gated_delta_net_conv_norm: null argument\n");
    return -1;
  }
  const int64_t value_width = num_heads_v * head_size_v;
  const int64_t state_plane = conv_channels * (kernel_size - 1);
  if (proj_stride <= 0 || kernel_size < 2 || state_plane <= 0 ||
      conv_state_num_elements % state_plane != 0 ||
      proj_num_elements % proj_stride != 0 ||
      conv_channels != 2 * num_heads_qk * head_size_qk + value_width ||
      conv_channels > proj_stride || z_column < 0 ||
      z_column + value_width > proj_stride || decay_column < 0 ||
      beta_column < 0 || decay_column + num_heads_v > gate_stride ||
      beta_column + num_heads_v > gate_stride) {
    fprintf(stderr,
            "wrap_gated_delta_net_conv_norm: proj=%lld stride=%lld "
            "conv_state=%lld C=%lld K=%lld z_column=%lld\n",
            (long long)proj_num_elements, (long long)proj_stride,
            (long long)conv_state_num_elements, (long long)conv_channels,
            (long long)kernel_size, (long long)z_column);
    return -1;
  }
  const int64_t batch = conv_state_num_elements / state_plane;
  const int64_t rows = proj_num_elements / proj_stride;
  if (batch < 1 || rows % batch != 0) {
    fprintf(stderr, "wrap_gated_delta_net_conv_norm: rows=%lld batch=%lld\n",
            (long long)rows, (long long)batch);
    return -1;
  }
  const int64_t seq_len = rows / batch;
  if (scale == 0.0f)
    scale = 1.0f / std::sqrt(static_cast<float>(head_size_qk));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  const int64_t state_bytes = conv_state_num_elements * 2;
  const auto *past = static_cast<const char *>(conv_state);
  const auto *present = static_cast<const char *>(conv_state_out);
  // The fused kernel handles an in-place conv state, not a partial overlap.
  const bool state_overlaps = past != present && past < present + state_bytes &&
                              present < past + state_bytes;
  if (hipdnn_ep_gdn_conv_norm_fused_enabled() && !state_overlaps &&
      hip_gated_delta_net_conv_norm_ok(batch, seq_len, kernel_size,
                                       num_heads_qk, num_heads_v, head_size_qk,
                                       head_size_v)) {
    OP_PROFILE(
        "gated_delta_net",
        [&] {
          char b[80];
          snprintf(b, sizeof(b), "conv_norm b=%lld,sq=%lld,hqk=%lld,hv=%lld",
                   (long long)batch, (long long)seq_len,
                   (long long)num_heads_qk, (long long)num_heads_v);
          return std::string(b);
        },
        state);
    int rc = hip_gated_delta_net_conv_norm(
        stream, proj, proj_stride, conv_weight, conv_bias, conv_state,
        conv_state_out, gates, gate_stride, decay_column, beta_column, z_column,
        static_cast<const float *>(recurrent_state),
        static_cast<float *>(recurrent_state_out),
        static_cast<const float *>(a_log), static_cast<const float *>(dt_bias),
        norm_weight, output, batch, seq_len, kernel_size, num_heads_qk,
        num_heads_v, scale, epsilon);
    if (rc != 0)
      fprintf(stderr,
              "wrap_gated_delta_net_conv_norm: fused kernel failed (%d)\n", rc);
    return rc;
  }

  // Separate kernels, bit-exact with the unfused conv, cast, recurrence and
  // norm sequence.
  if (hipMemcpy2DAsync(conv_in, conv_channels * 2, proj, proj_stride * 2,
                       conv_channels * 2, rows, hipMemcpyDeviceToDevice,
                       stream) != hipSuccess)
    return -1;
  int rc = wrap_causal_conv_with_state(
      state, /*op_state_slot=*/-1, conv_in, conv_weight, conv_bias, conv_state,
      conv_out, conv_state_out, batch, conv_channels, seq_len, kernel_size,
      /*ndim=*/1, /*activation=silu*/ 1, /*element_size_bytes=*/2,
      /*channels_last=*/1, /*dilation=*/1, /*state_channels_last=*/0);
  if (rc != 0)
    return rc;
  rc = hip_gated_delta_net_split_gates(stream, gates, gate_stride, decay_column,
                                       beta_column, static_cast<float *>(decay),
                                       static_cast<float *>(beta), rows,
                                       num_heads_v);
  if (rc != 0)
    return rc;
  rc = wrap_gated_delta_net(
      state, conv_out, static_cast<const float *>(decay),
      static_cast<const float *>(beta),
      static_cast<const float *>(recurrent_state),
      static_cast<const float *>(a_log), static_cast<const float *>(dt_bias),
      gdn_out, static_cast<float *>(recurrent_state_out), batch, seq_len,
      num_heads_qk, num_heads_v, head_size_qk, head_size_v, scale,
      /*element_size_bytes=*/2);
  if (rc != 0)
    return rc;
  if (hipMemcpy2DAsync(gate_z, value_width * 2,
                       static_cast<const char *>(proj) + z_column * 2,
                       proj_stride * 2, value_width * 2, rows,
                       hipMemcpyDeviceToDevice, stream) != hipSuccess)
    return -1;
  return wrap_gated_rms_norm(state, gdn_out, norm_weight, gate_z, output,
                             rows * value_width, head_size_v,
                             /*element_size_bytes=*/2, /*activation=sigmoid*/ 0,
                             epsilon);
}
