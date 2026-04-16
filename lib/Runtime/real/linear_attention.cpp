/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

// Update rule enum values (must match compiler lowering in
// LinearAttentionLowering.cpp)
static constexpr int64_t kUpdateRuleLinear = 0;
static constexpr int64_t kUpdateRuleGated = 1;
static constexpr int64_t kUpdateRuleDelta = 2;
static constexpr int64_t kUpdateRuleGatedDelta = 3;

static const char *updateRuleName(int64_t rule) {
  switch (rule) {
  case kUpdateRuleLinear:
    return "linear";
  case kUpdateRuleGated:
    return "gated";
  case kUpdateRuleDelta:
    return "delta";
  case kUpdateRuleGatedDelta:
    return "gated_delta";
  default:
    return "unknown";
  }
}

extern "C" int wrap_linear_attention(
    RuntimeState *state, const void *query, const void *key, const void *value,
    const void *past_state, const void *decay, const void *beta, void *output,
    void *present_state, int64_t q_num_heads, int64_t kv_num_heads,
    float scale, int64_t chunk_size, int64_t update_rule, int64_t batch_size,
    int64_t seq_len, int64_t head_dim_k, int64_t head_dim_v,
    int64_t element_size_bytes) {

  RUNTIME_DEBUG_LOG("[linear_attention] enter: B=%lld T=%lld "
                    "q_heads=%lld kv_heads=%lld d_k=%lld d_v=%lld "
                    "scale=%.6f chunk=%lld rule=%s elem_size=%lld\n",
                    (long long)batch_size, (long long)seq_len,
                    (long long)q_num_heads, (long long)kv_num_heads,
                    (long long)head_dim_k, (long long)head_dim_v,
                    (double)scale, (long long)chunk_size,
                    updateRuleName(update_rule),
                    (long long)element_size_bytes);

  RUNTIME_DEBUG_LOG("[linear_attention] ptrs: query=%p key=%p value=%p "
                    "past_state=%p decay=%p beta=%p output=%p "
                    "present_state=%p\n",
                    query, key, value, past_state, decay, beta, output,
                    present_state);

  // Validate required inputs
  if (!query || !key || !value || !output || !present_state) {
    fprintf(stderr,
            "[linear_attention] ERROR: null required pointer "
            "(query=%p key=%p value=%p output=%p present_state=%p)\n",
            query, key, value, output, present_state);
    return -1;
  }

  // Validate update rule constraints
  if ((update_rule == kUpdateRuleGated ||
       update_rule == kUpdateRuleGatedDelta) &&
      !decay) {
    fprintf(stderr,
            "[linear_attention] ERROR: decay is required for %s mode\n",
            updateRuleName(update_rule));
    return -1;
  }
  if ((update_rule == kUpdateRuleDelta ||
       update_rule == kUpdateRuleGatedDelta) &&
      !beta) {
    fprintf(stderr,
            "[linear_attention] ERROR: beta is required for %s mode\n",
            updateRuleName(update_rule));
    return -1;
  }

  void *hip_stream = hipdnn_ep_state_get_stream(state);
  if (!hip_stream) {
    fprintf(stderr, "[linear_attention] ERROR: failed to get HIP stream\n");
    return -1;
  }

  hipblasLtHandle_t blaslt_handle =
      (hipblasLtHandle_t)hipdnn_ep_state_get_hipblas_handle(state);
  if (!blaslt_handle) {
    fprintf(stderr,
            "[linear_attention] ERROR: failed to get hipBLASLt handle\n");
    return -1;
  }

  // Auto-compute scale if sentinel zero
  if (scale == 0.0f && head_dim_k > 0)
    scale = 1.0f / sqrtf((float)head_dim_k);

  // Dimension aliases
  const int64_t B = batch_size;
  const int64_t Hkv = kv_num_heads;
  const int64_t dk = head_dim_k;
  const int64_t dv = head_dim_v;

  // State size per batch per head: dk * dv elements
  const int64_t state_bytes = dk * dv * element_size_bytes;
  int64_t total_state_bytes = B * Hkv * state_bytes;

  int result = 0;

  // Initialize present_state from past_state (or zeros)
  if (past_state) {
    HIP_CHECK(hipMemcpyAsync(present_state, past_state, total_state_bytes,
                              hipMemcpyDeviceToDevice,
                              (hipStream_t)hip_stream));
  } else {
    HIP_CHECK(
        hipMemsetAsync(present_state, 0, total_state_bytes,
                       (hipStream_t)hip_stream));
  }

  // TODO: Implement the full linear attention recurrence kernel.
  //
  // The recurrence for each token t (per batch b, per KV head g):
  //   "linear":       S = S + k_t outer v_t
  //   "gated":        S = exp(g_t) * S + k_t outer v_t
  //   "delta":        S = S + beta_t * k_t outer (v_t - S^T k_t)
  //   "gated_delta":  S = exp(g_t) * S + beta_t * k_t outer (v_t - exp(g_t) * S^T k_t)
  //   o_t = scale * q_t^T S    (for each query head h mapped to this KV group)
  //
  // For T=1 (decode): sequential recurrence is efficient.
  // For T>1 (prefill): chunk-parallel WY decomposition is preferred.
  //
  // Current implementation: initialize state, placeholder for kernel dispatch.
  // Full GPU kernel integration is pending custom_kernels library extension.

  RUNTIME_DEBUG_LOG("[linear_attention] state initialized (%s), "
                    "kernel dispatch pending\n",
                    past_state ? "from past" : "zeros");

cleanup:
  RUNTIME_DEBUG_LOG("[linear_attention] exit result=%d\n", result);
  return result;
}
