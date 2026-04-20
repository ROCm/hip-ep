/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
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
    void *present_state, int64_t q_num_heads, int64_t kv_num_heads, float scale,
    int64_t chunk_size, int64_t update_rule, int64_t batch_size,
    int64_t seq_len, int64_t head_dim_k, int64_t head_dim_v,
    int64_t element_size_bytes) {

  RUNTIME_DEBUG_LOG("[linear_attention] enter: B=%lld T=%lld "
                    "q_heads=%lld kv_heads=%lld d_k=%lld d_v=%lld "
                    "scale=%.6f chunk=%lld rule=%s elem_size=%lld\n",
                    (long long)batch_size, (long long)seq_len,
                    (long long)q_num_heads, (long long)kv_num_heads,
                    (long long)head_dim_k, (long long)head_dim_v, (double)scale,
                    (long long)chunk_size, updateRuleName(update_rule),
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
    fprintf(stderr, "[linear_attention] ERROR: decay is required for %s mode\n",
            updateRuleName(update_rule));
    return -1;
  }
  if ((update_rule == kUpdateRuleDelta ||
       update_rule == kUpdateRuleGatedDelta) &&
      !beta) {
    fprintf(stderr, "[linear_attention] ERROR: beta is required for %s mode\n",
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
  const int64_t Hq = q_num_heads;
  const int64_t Hkv = kv_num_heads;
  const int64_t dk = head_dim_k;
  const int64_t dv = head_dim_v;

  // State size per batch per head: dk * dv elements
  const int64_t state_bytes = dk * dv * element_size_bytes;
  int64_t total_state_bytes = B * Hkv * state_bytes;

  int result = 0;

  // Temp buffers for T>1 token-by-token processing (nullptr when T=1)
  void *q_t = nullptr, *k_t = nullptr, *v_t = nullptr, *o_t = nullptr;
  void *decay_t = nullptr, *beta_t = nullptr;

  // Initialize present_state from past_state (or zeros)
  if (past_state) {
    HIP_CHECK(hipMemcpyAsync(present_state, past_state, total_state_bytes,
                             hipMemcpyDeviceToDevice, (hipStream_t)hip_stream));
  } else {
    HIP_CHECK(hipMemsetAsync(present_state, 0, total_state_bytes,
                             (hipStream_t)hip_stream));
  }

  if (seq_len == 1) {
    int kern_result = hip_linear_attention_decode(
        hip_stream, query, key, value, decay, beta, present_state, output, B,
        q_num_heads, Hkv, dk, dv, scale, update_rule, element_size_bytes);
    if (kern_result != 0) {
      fprintf(stderr, "[linear_attention] ERROR: decode kernel failed (%d)\n",
              kern_result);
      result = -1;
      goto cleanup;
    }
    RUNTIME_DEBUG_LOG("[linear_attention] T=1 decode kernel dispatched\n");
  } else {
    // T>1 prefill: loop over tokens, reusing the T=1 decode kernel.
    // Each iteration extracts one token from the packed [B,T,H*D] inputs,
    // runs one recurrence step (updating present_state in-place), and
    // writes the output back to the correct position. Correct but O(T)
    // kernel launches; a chunk-parallel WY kernel would be faster.
    const int64_t q_token_bytes = Hq * dk * element_size_bytes;
    const int64_t k_token_bytes = Hkv * dk * element_size_bytes;
    const int64_t v_token_bytes = Hkv * dv * element_size_bytes;
    const int64_t o_token_bytes = Hq * dv * element_size_bytes;
    const int64_t decay_token_bytes = Hkv * dk * element_size_bytes;
    const int64_t beta_token_bytes = Hkv * element_size_bytes;

    HIP_CHECK(hipMalloc(&q_t, B * q_token_bytes));
    HIP_CHECK(hipMalloc(&k_t, B * k_token_bytes));
    HIP_CHECK(hipMalloc(&v_t, B * v_token_bytes));
    HIP_CHECK(hipMalloc(&o_t, B * o_token_bytes));
    if (decay)
      HIP_CHECK(hipMalloc(&decay_t, B * decay_token_bytes));
    if (beta)
      HIP_CHECK(hipMalloc(&beta_t, B * beta_token_bytes));

    RUNTIME_DEBUG_LOG(
        "[linear_attention] T>1 prefill: looping %lld tokens\n",
        (long long)seq_len);

    for (int64_t t = 0; t < seq_len; ++t) {
      // Gather token t from packed [B,T,X] into contiguous [B,1,X].
      // hipMemcpy2DAsync handles the stride between batches.
      HIP_CHECK(hipMemcpy2DAsync(
          q_t, q_token_bytes,
          (const char *)query + t * q_token_bytes, seq_len * q_token_bytes,
          q_token_bytes, B, hipMemcpyDeviceToDevice,
          (hipStream_t)hip_stream));
      HIP_CHECK(hipMemcpy2DAsync(
          k_t, k_token_bytes,
          (const char *)key + t * k_token_bytes, seq_len * k_token_bytes,
          k_token_bytes, B, hipMemcpyDeviceToDevice,
          (hipStream_t)hip_stream));
      HIP_CHECK(hipMemcpy2DAsync(
          v_t, v_token_bytes,
          (const char *)value + t * v_token_bytes, seq_len * v_token_bytes,
          v_token_bytes, B, hipMemcpyDeviceToDevice,
          (hipStream_t)hip_stream));
      if (decay) {
        HIP_CHECK(hipMemcpy2DAsync(
            decay_t, decay_token_bytes,
            (const char *)decay + t * decay_token_bytes,
            seq_len * decay_token_bytes, decay_token_bytes, B,
            hipMemcpyDeviceToDevice, (hipStream_t)hip_stream));
      }
      if (beta) {
        HIP_CHECK(hipMemcpy2DAsync(
            beta_t, beta_token_bytes,
            (const char *)beta + t * beta_token_bytes,
            seq_len * beta_token_bytes, beta_token_bytes, B,
            hipMemcpyDeviceToDevice, (hipStream_t)hip_stream));
      }

      int kern_result = hip_linear_attention_decode(
          hip_stream, q_t, k_t, v_t, decay_t, beta_t, present_state, o_t,
          B, q_num_heads, Hkv, dk, dv, scale, update_rule,
          element_size_bytes);
      if (kern_result != 0) {
        fprintf(stderr,
                "[linear_attention] ERROR: decode kernel failed at t=%lld "
                "(%d)\n",
                (long long)t, kern_result);
        result = -1;
        goto cleanup;
      }

      // Scatter output token back into packed [B,T,X] output buffer.
      HIP_CHECK(hipMemcpy2DAsync(
          (char *)output + t * o_token_bytes, seq_len * o_token_bytes, o_t,
          o_token_bytes, o_token_bytes, B, hipMemcpyDeviceToDevice,
          (hipStream_t)hip_stream));
    }

    RUNTIME_DEBUG_LOG(
        "[linear_attention] T>1 prefill completed (%lld tokens)\n",
        (long long)seq_len);
  }

cleanup:
  hipFree(q_t);
  hipFree(k_t);
  hipFree(v_t);
  hipFree(o_t);
  hipFree(decay_t);
  hipFree(beta_t);
  RUNTIME_DEBUG_LOG("[linear_attention] exit result=%d\n", result);
  return result;
}
