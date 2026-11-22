/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cmath>
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
