/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
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

static size_t la_align_up(size_t bytes) { return (bytes + 255) & ~size_t(255); }

extern "C" int wrap_linear_attention(
    RuntimeState *state, const void *query, const void *key, const void *value,
    const void *past_state, const void *decay, const void *beta,
    const void *decay_bias, const void *decay_scale,
    const void *qk_norm_weight, void *output, void *present_state, int64_t Hq,
    int64_t Hkv, int64_t Nk, int64_t decay_per_key_dim, int64_t beta_per_head,
    float scale, int64_t chunk_size, int64_t update_rule, int64_t B,
    int64_t seq_len, int64_t dk, int64_t dv, int64_t type,
    float qk_norm_epsilon, float q_scale, int64_t beta_sigmoid) {
  OP_PROFILE(
      "linear_attention",
      [&] {
        char b[64];
        snprintf(b, sizeof(b),
                 "b=%lld,sq=%lld,hq=%lld,hkv=%lld,dk=%lld,dv=%lld",
                 (long long)B, (long long)seq_len, (long long)Hq,
                 (long long)Hkv, (long long)dk, (long long)dv);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG("[linear_attention] enter: B=%lld seq_len=%lld "
                    "q_heads=%lld kv_heads=%lld n_k=%lld d_k=%lld d_v=%lld "
                    "scale=%.6f chunk=%lld rule=%lld type=%s(%lld)\n",
                    (long long)B, (long long)seq_len, (long long)Hq,
                    (long long)Hkv, (long long)Nk, (long long)dk, (long long)dv,
                    (double)scale, (long long)chunk_size,
                    (long long)update_rule, hipdnn_ep_datatype_name(type),
                    (long long)type);

  RUNTIME_DEBUG_LOG("[linear_attention] layout: decay_per_key_dim=%lld "
                    "beta_per_head=%lld\n",
                    (long long)decay_per_key_dim, (long long)beta_per_head);

  RUNTIME_DEBUG_LOG("[linear_attention] ptrs: query=%p key=%p value=%p "
                    "past_state=%p decay=%p beta=%p output=%p "
                    "present_state=%p\n",
                    query, key, value, past_state, decay, beta, output,
                    present_state);

  if (type != HIPDNN_EP_DATATYPE_FLOAT && type != HIPDNN_EP_DATATYPE_HALF &&
      type != HIPDNN_EP_DATATYPE_BFLOAT16) {
    fprintf(stderr,
            "[linear_attention] ERROR: unsupported element type %lld "
            "(expect HIPDNN_EP_DATATYPE_FLOAT/HALF/BFLOAT16)\n",
            (long long)type);
    return -1;
  }

  const int64_t elem_size = hipdnn_ep_datatype_size(type);
  if (elem_size <= 0) {
    fprintf(stderr,
            "[linear_attention] ERROR: invalid elem_size from type %lld\n",
            (long long)type);
    return -1;
  }

  // Validate required inputs (null key and value: packed QKV in query)
  if (!query || !key != !value || !output || !present_state) {
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
            "[linear_attention] ERROR: decay is required for rule=%lld\n",
            (long long)update_rule);
    return -1;
  }
  if ((update_rule == kUpdateRuleDelta ||
       update_rule == kUpdateRuleGatedDelta) &&
      !beta) {
    fprintf(stderr,
            "[linear_attention] ERROR: beta is required for rule=%lld\n",
            (long long)update_rule);
    return -1;
  }

  // The HIP decode kernel supports all four decay/beta layout combinations:
  //   decay_per_key_dim=1 -> [B, T, H_kv * d_k]
  //   decay_per_key_dim=0 -> [B, T, H_kv]        (broadcast across d_k)
  //   beta_per_head=1     -> [B, T, H_kv]
  //   beta_per_head=0     -> [B, T, 1]           (broadcast across H_kv)
  // The flags are forwarded to hip_linear_attention_decode below.

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

  // When 0.0 (default), derives d_k = query.shape[-1] / q_num_heads and uses
  // 1/sqrt(d_k). Set explicitly to override.
  if (scale == 0.0f && dk > 0)
    scale = 1.0f / sqrtf((float)dk);

  // Output head count = max(Hq, Hkv).  In standard GQA (Hq >= Hkv) the output
  // is packed in Q-head order; in inverse GQA it is packed in KV-head order.
  const int64_t Hout = Hq >= Hkv ? Hq : Hkv;

  // The chunk-parallel prefill (below) needs a device scratch arena sized to
  // this shape and rule; the two rules run different kernels and need
  // different layouts. It lives in the per-session RuntimeState::la_scratch
  // pool (grow-on-demand, freed in hipdnn_ep_state_cleanup) -- same policy as
  // qmoe/conv scratch -- behind the prepared inputs, if any. Size both before
  // preparing: growing frees the old arena.
  const bool try_chunked = (update_rule == kUpdateRuleGated ||
                            update_rule == kUpdateRuleGatedDelta) &&
                           seq_len > 1;
  const size_t chunk_bytes =
      try_chunked ? hip_linear_attention_prefill_scratch_bytes(
                        (int)B, (int)seq_len, (int)Hkv, (int)dk, (int)dv,
                        update_rule)
                  : 0;

  // Input preparation: packed QKV, q/k head norms, q scale, softplus decay
  // and sigmoid beta run as one launch writing contiguous inputs into the
  // front of la_scratch; everything below then sees the plain layout.
  const bool packed = !key;
  const bool prep_q = packed || qk_norm_weight || q_scale != 1.f;
  const bool prep_k = packed || qk_norm_weight;
  const bool prep_decay = decay && decay_bias && decay_scale;
  const bool prep_beta = beta && beta_sigmoid;
  const int64_t tokens = B * seq_len;
  const int64_t beta_width = beta_per_head ? Hkv : 1;
  size_t prep_bytes = 0;
  size_t q_off = 0, k_off = 0, v_off = 0, decay_off = 0, beta_off = 0;
  if (prep_q || prep_k || prep_decay || prep_beta) {
    if (type != HIPDNN_EP_DATATYPE_HALF ||
        (prep_decay && decay_per_key_dim)) {
      fprintf(stderr,
              "[linear_attention] ERROR: input preparation needs fp16 and "
              "per-head decay (type=%lld decay_per_key_dim=%lld)\n",
              (long long)type, (long long)decay_per_key_dim);
      return -1;
    }
    auto reserve = [&](bool used, int64_t elems) {
      size_t off = prep_bytes;
      if (used)
        prep_bytes += la_align_up((size_t)(tokens * elems * elem_size));
      return off;
    };
    q_off = reserve(prep_q, Hq * dk);
    k_off = reserve(prep_k, Nk * dk);
    v_off = reserve(packed, Hkv * dv);
    decay_off = reserve(prep_decay, Hkv);
    beta_off = reserve(prep_beta, beta_width);
  }
  char *la_scratch = nullptr;
  if (prep_bytes + chunk_bytes > 0 &&
      hipdnn_ep_state_ensure_la_scratch(state, prep_bytes + chunk_bytes) == 0)
    la_scratch = (char *)hipdnn_ep_state_get_la_scratch(state);
  if (prep_bytes > 0) {
    if (!la_scratch) {
      fprintf(stderr, "[linear_attention] ERROR: no scratch for input "
                      "preparation (%zu bytes)\n",
              prep_bytes);
      return -1;
    }
    const int64_t width = (Hq + Nk) * dk + Hkv * dv;
    const char *src = (const char *)query;
    const void *k_src = packed ? src + Hq * dk * elem_size : key;
    const void *v_src = packed ? src + (Hq + Nk) * dk * elem_size : value;
    void *q_out = prep_q ? la_scratch + q_off : nullptr;
    void *k_out = prep_k ? la_scratch + k_off : nullptr;
    void *v_out = packed ? la_scratch + v_off : nullptr;
    void *decay_out = prep_decay ? la_scratch + decay_off : nullptr;
    void *beta_out = prep_beta ? la_scratch + beta_off : nullptr;
    const int rc = hip_linear_attention_prepare(
        hip_stream, query, k_src, v_src, packed ? width : Hq * dk,
        packed ? width : Nk * dk, packed ? width : Hkv * dv, decay, beta,
        decay_bias, decay_scale, q_out, k_out, v_out, decay_out, beta_out,
        tokens, Hq, Nk, Hkv, dk, dv, qk_norm_weight, qk_norm_epsilon, q_scale,
        beta_width);
    if (rc != 0) {
      fprintf(stderr, "[linear_attention] ERROR: input preparation failed "
                      "(%d)\n",
              rc);
      return -1;
    }
    if (q_out)
      query = q_out;
    if (k_out)
      key = k_out;
    if (v_out)
      value = v_out;
    if (decay_out)
      decay = decay_out;
    if (beta_out)
      beta = beta_out;
    RUNTIME_DEBUG_LOG("[linear_attention] prepared inputs: packed=%d "
                      "qk_norm=%d q_scale=%g decay_softplus=%d "
                      "beta_sigmoid=%d\n",
                      (int)packed, (int)(qk_norm_weight != nullptr),
                      (double)q_scale,
                      (int)prep_decay, (int)prep_beta);
  }

  // State size per batch per head: dk * dv elements
  const int64_t state_bytes = dk * dv * elem_size;
  int64_t total_state_bytes = B * Hkv * state_bytes;

  // Per-token stride (in bytes) for each packed [B, T, H*D] tensor. The
  // kernel indexes into the packed layout directly using seq_len as the
  // per-batch stride, so we only need to advance the base pointer by one
  // token between iterations -- no intermediate gather/scatter buffers.
  //
  //   query  : Hq  * dk      key    : Nk   * dk (may be < Hkv*dk)
  //   value  : Hkv * dv      output : Hout * dv (Hout = max(Hq, Hkv))
  //   decay_per_key_dim==1 -> Hkv * dk;   ==0 -> Hkv
  //   beta_per_head==1     -> Hkv;        ==0 -> 1
  //
  // Declared BEFORE the first HIP_CHECK below so the `goto cleanup` paths
  // inside HIP_CHECK don't bypass their initialization. C++ forbids jumping
  // past a variable's initialization to a later label (the "cannot jump from
  // this goto statement to its label" error clang emits on linux; MSVC is
  // historically lax about this and accepted the previous ordering).
  const int64_t q_token_bytes = Hq * dk * elem_size;
  const int64_t k_token_bytes = Nk * dk * elem_size;
  const int64_t v_token_bytes = Hkv * dv * elem_size;
  const int64_t o_token_bytes = Hout * dv * elem_size;
  const int64_t decay_token_bytes =
      (decay_per_key_dim ? Hkv * dk : Hkv) * elem_size;
  const int64_t beta_token_bytes = (beta_per_head ? Hkv : 1) * elem_size;

  int result = 0;

  // Initialize present_state from past_state (or zeros) before dispatching any
  // update rule or execution path. Use a compute-stream kernel rather than
  // hipMemcpyAsync(D2D): LA state may live in host-mapped memory, where the
  // copy-engine path degenerates into a synchronous host-staged transfer and
  // drains the compute queue. outer_rank=0 expresses one contiguous row, so
  // this single launch covers the entire state for decode, per-token prefill,
  // chunked prefill, and the chunked-path fallback.
  //
  // Skip initialization when the buffers alias. Under a shared-buffer binding
  // the recurrent state is already in present_state and all paths update it in
  // place, so copying it to itself would be a semantic no-op.
  if (past_state) {
    if (past_state != present_state) {
      const int copy_result = hip_strided_copy(
          hip_stream, present_state, past_state, elem_size,
          /*outer_rank=*/0, /*outer_sizes=*/nullptr,
          /*src_outer_strides=*/nullptr, /*dst_outer_strides=*/nullptr,
          /*row_elems=*/B * Hkv * dk * dv, /*outer_total=*/1);
      if (copy_result != 0) {
        fprintf(stderr,
                "[linear_attention] ERROR: state copy kernel failed (%d)\n",
                copy_result);
        result = -1;
        goto cleanup;
      }
    }
  } else {
    HIP_CHECK(hipMemsetAsync(present_state, 0, total_state_bytes,
                             (hipStream_t)hip_stream));
  }

  // Fast path: prefill (seq_len > 1) of the gated and gated_delta rules is
  // handled by a chunked-parallel affine scan. This replaces ~seq_len
  // per-token launches per layer. The launcher returns 1 when it declines an
  // unsupported config; we then fall back to the per-token loop below.
  if (try_chunked) {
    // If sizing/growth failed we pass a null scratch and the launcher
    // declines (rc=1 -> per-token loop below).
    void *chunk_scratch =
        la_scratch && chunk_bytes > 0 ? la_scratch + prep_bytes : nullptr;
    int rc = hip_linear_attention_prefill_chunked(
        hip_stream, query, key, value, decay, beta, present_state, output, B,
        seq_len, Hq, Hkv, Nk, dk, dv, scale, update_rule, decay_per_key_dim,
        beta_per_head, type, chunk_scratch, chunk_bytes);
    if (rc == 0) {
      RUNTIME_DEBUG_LOG(
          "[linear_attention] prefill via chunked-parallel kernel (%lld "
          "token(s))\n",
          (long long)seq_len);
      goto cleanup;
    }
    if (rc < 0) {
      fprintf(stderr,
              "[linear_attention] ERROR: chunked prefill kernel failed (%d)\n",
              rc);
      result = -1;
      goto cleanup;
    }
    RUNTIME_DEBUG_LOG(
        "[linear_attention] chunked prefill declined (rc=%d); falling back to "
        "per-token loop\n",
        rc);
  }

  RUNTIME_DEBUG_LOG(
      "[linear_attention] dispatching %lld token(s) via per-step decode "
      "kernel\n",
      (long long)seq_len);

  // Unified path for decode (seq_len==1) and prefill (seq_len>1):
  // advance the base pointers by t * token_bytes and let the kernel use
  // seq_len as the per-batch stride. decay/beta layout flags are passed
  // through so the kernel can select the appropriate access pattern.
  for (int64_t t = 0; t < seq_len; ++t) {
    const void *q_t = (const char *)query + t * q_token_bytes;
    const void *k_t = (const char *)key + t * k_token_bytes;
    const void *v_t = (const char *)value + t * v_token_bytes;
    const void *decay_t =
        decay ? (const void *)((const char *)decay + t * decay_token_bytes)
              : nullptr;
    const void *beta_t =
        beta ? (const void *)((const char *)beta + t * beta_token_bytes)
             : nullptr;
    void *o_t = (char *)output + t * o_token_bytes;

    int kern_result = hip_linear_attention_decode(
        hip_stream, q_t, k_t, v_t, decay_t, beta_t, present_state, o_t, B,
        seq_len, Hq, Hkv, Nk, dk, dv, scale, update_rule, decay_per_key_dim,
        beta_per_head, type);
    if (kern_result != 0) {
      fprintf(stderr,
              "[linear_attention] ERROR: decode kernel failed at t=%lld "
              "(%d)\n",
              (long long)t, kern_result);
      result = -1;
      goto cleanup;
    }
  }

  RUNTIME_DEBUG_LOG("[linear_attention] completed %lld token(s)\n",
                    (long long)seq_len);

cleanup:
  RUNTIME_DEBUG_LOG("[linear_attention] exit result=%d\n", result);
  return result;
}
