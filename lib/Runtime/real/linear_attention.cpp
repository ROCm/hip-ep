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

// --- RGP capture fence (diagnostics only) -----------------------------------
// Isolating one sparse kernel (linear_attention: 1 dispatch/layer buried in
// ~800 MoE dispatches) in an RGP dispatch capture is unreliable via delay- or
// dispatch-index triggers because of run-to-run jitter. This fence removes the
// guesswork: when HIPDNN_RGP_FENCE names an op, the runtime drains the GPU
// queue and idles (Sleep) for a WIDE window right before that op's kernels
// launch -- ONCE, after HIPDNN_RGP_FENCE_SKIP matching calls (skip warmup).
// An RGP capture armed with --rgp-auto-capture dispatch:1:N and an
// --rgp-auto-capture-delay landing anywhere inside that idle window arms while
// the GPU is idle, so the very next dispatch is this op -> captured cleanly.
// Uses only raw char ops + Win32/HIP imports that resolve in the JIT-linked
// runtime bitcode (same constraints as op_profile_ablated); no STL string/IO.
#ifdef _WIN32
extern "C" __declspec(dllimport) unsigned long __stdcall GetEnvironmentVariableA(
    const char *, char *, unsigned long);
extern "C" __declspec(dllimport) void __stdcall Sleep(unsigned long);
#endif

static bool rgp_getenv(const char *name, char *buf, unsigned bufsz) {
#ifdef _WIN32
  unsigned long n = GetEnvironmentVariableA(name, buf, bufsz);
  if (n == 0 || n >= bufsz) {
    buf[0] = '\0';
    return false;
  }
  return true;
#else
  const char *v = getenv(name);
  if (!v) {
    buf[0] = '\0';
    return false;
  }
  unsigned i = 0;
  for (; v[i] && i + 1 < bufsz; ++i)
    buf[i] = v[i];
  buf[i] = '\0';
  return i > 0;
#endif
}

static long rgp_atol(const char *s) {
  long r = 0;
  while (*s >= '0' && *s <= '9') {
    r = r * 10 + (*s - '0');
    ++s;
  }
  return r;
}

static bool rgp_str_eq(const char *a, const char *b) {
  while (*a && *b && *a == *b) {
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

// Fences the given op if HIPDNN_RGP_FENCE matches it. Fires exactly once, after
// HIPDNN_RGP_FENCE_SKIP matching calls; idles HIPDNN_RGP_FENCE_MS (default 45s).
static void rgp_capture_fence(const char *opname, hipStream_t stream) {
  static char target[128];
  static const bool has =
      rgp_getenv("HIPDNN_RGP_FENCE", target, sizeof(target));
  if (!has || !rgp_str_eq(target, opname))
    return;
  static const long skip = [] {
    char b[32];
    return rgp_getenv("HIPDNN_RGP_FENCE_SKIP", b, sizeof(b)) ? rgp_atol(b) : 3L;
  }();
  static const long fence_ms = [] {
    char b[32];
    return rgp_getenv("HIPDNN_RGP_FENCE_MS", b, sizeof(b)) ? rgp_atol(b)
                                                           : 45000L;
  }();
  static long count = 0;
  static bool fired = false;
  if (fired || count++ < skip)
    return;
  fired = true;
  fprintf(stderr,
          "[rgp_fence] '%s' call #%ld: draining queue + idling %ld ms so RGP "
          "can arm on the next dispatch...\n",
          opname, count, fence_ms);
  fflush(stderr);
  (void)hipStreamSynchronize(stream);
  (void)hipDeviceSynchronize();
#ifdef _WIN32
  Sleep((unsigned long)fence_ms);
#endif
  fprintf(stderr, "[rgp_fence] resume '%s' -> launching now (capture window).\n",
          opname);
  fflush(stderr);
}

extern "C" int wrap_linear_attention(
    RuntimeState *state, const void *query, const void *key, const void *value,
    const void *past_state, const void *decay, const void *beta, void *output,
    void *present_state, int64_t Hq, int64_t Hkv, int64_t Nk,
    int64_t decay_per_key_dim, int64_t beta_per_head, float scale,
    int64_t chunk_size, int64_t update_rule, int64_t B, int64_t seq_len,
    int64_t dk, int64_t dv, int64_t type) {
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

  // Initialize present_state from past_state (or zeros).
  //
  // Skip the copy when the two buffers alias (past_state == present_state).
  // Under a shared-buffer binding (OGA past_present_share_buffer / EP GPU
  // aliasing) the recurrent state is one buffer used for both past input and
  // present output, so present already holds past's data and the decode/prefill
  // kernel updates it in place -- the copy is a semantic no-op. It is NOT free
  // to issue, though: the buffer comes from the EP's host-mapped allocator
  // (hipHostMallocMapped), and on that memory hipMemcpyAsync D2D degenerates to
  // a synchronous host-staged copy, costing host (CPU) time per call. At
  // decode this dominates the op (profiled ~54.6 ms CPU vs ~5.6 ms GPU across
  // 30 single-token LA calls; ~1.8 ms CPU per layer per token), so skipping the
  // self-copy removes the per-token recurrent-state staging overhead.
  if (past_state) {
    if (past_state != present_state) {
      HIP_CHECK(hipMemcpyAsync(present_state, past_state, total_state_bytes,
                               hipMemcpyDeviceToDevice,
                               (hipStream_t)hip_stream));
    }
    // else: aliased shared buffer -> present already == past, copy is a no-op.
  } else {
    HIP_CHECK(hipMemsetAsync(present_state, 0, total_state_bytes,
                             (hipStream_t)hip_stream));
  }

  // Fast path: prefill (seq_len > 1) of the gated_delta rule is handled by a
  // single chunked-parallel launch that processes the whole sequence (keeps
  // the recurrent state fp32 across each chunk -> more accurate than the
  // per-token kernel, and replaces ~seq_len launches with one). The launcher
  // returns 1 when it declines an unsupported config; we then fall back to the
  // per-token loop below.
  if (update_rule == kUpdateRuleGatedDelta && seq_len > 1) {
    // Optional: idle here so an RGP capture can arm right before this launch.
    // No-op unless HIPDNN_RGP_FENCE=linear_attention.
    rgp_capture_fence("linear_attention", (hipStream_t)hip_stream);
    int rc = hip_linear_attention_prefill_chunked(
        hip_stream, query, key, value, decay, beta, present_state, output, B,
        seq_len, Hq, Hkv, Nk, dk, dv, scale, update_rule, decay_per_key_dim,
        beta_per_head, type);
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
