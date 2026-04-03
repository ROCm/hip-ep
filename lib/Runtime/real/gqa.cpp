/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "cache_utils.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include "error_check_macros.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

// =============================================================================
// hipBLASLt layout helper
// =============================================================================

static hipblasStatus_t setLayoutBatch(hipblasLtMatrixLayout_t layout,
                                      int32_t batchCount, int64_t stride) {
  hipblasStatus_t status;
  status = hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount,
      sizeof(batchCount));
  if (status != HIPBLAS_STATUS_SUCCESS)
    return status;
  status = hipblasLtMatrixLayoutSetAttribute(
      layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
      sizeof(stride));
  return status;
}

//===----------------------------------------------------------------------===//
// GQA GEMM descriptor cache
//===----------------------------------------------------------------------===//
//
// hipBLASLt descriptors and heuristic-selected algorithms are created once per
// unique (m, n, k, batch, transA) shape and reused for the process lifetime.
// This avoids repeated descriptor creation and heuristic queries on every
// GQA inference call in the decomposed (prefill) path.

struct GqaGemmKey {
  int64_t m, n, k, batch;
  bool transA; // true for Score GEMM (K^T * Q), false for Value GEMM (V * S)
  bool operator==(const GqaGemmKey &o) const {
    return m == o.m && n == o.n && k == o.k && batch == o.batch &&
           transA == o.transA;
  }
};

struct GqaGemmKeyHash {
  size_t operator()(const GqaGemmKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.m);
    hash_combine_val(h, k.n);
    hash_combine_val(h, k.k);
    hash_combine_val(h, k.batch);
    hash_combine_val(h, k.transA);
    return h;
  }
};

/// Cached hipBLASLt state for a single GEMM shape.
/// Ownership: descriptors are created in queryOrCreateGemmState() and live
/// for the process lifetime (never destroyed individually).
struct GqaGemmCacheEntry {
  hipblasLtMatmulDesc_t desc;                     // matmul operation descriptor
  hipblasLtMatrixLayout_t layA, layB, layC, layD; // matrix layouts
  hipblasLtMatmulAlgo_t algo; // heuristic-selected algorithm
  size_t workspace_size;      // workspace bytes required by algo
};

static std::unordered_map<GqaGemmKey, GqaGemmCacheEntry, GqaGemmKeyHash>
    g_gqa_gemm_cache;

/// Look up or create cached hipBLASLt descriptors + algorithm for a GEMM shape.
/// On first call for a given key, creates all descriptors, queries the
/// heuristic, and caches the result. Returns nullptr on any API failure
/// (partially created descriptors are cleaned up).
static const GqaGemmCacheEntry *queryOrCreateGemmState(hipblasLtHandle_t handle,
                                                       const GqaGemmKey &key) {
  auto it = g_gqa_gemm_cache.find(key);
  if (it != g_gqa_gemm_cache.end())
    return &it->second;

  int64_t m = key.m, n = key.n, k = key.k;
  int32_t batch = (int32_t)key.batch;

  GqaGemmCacheEntry entry = {};

  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasStatus_t st;

#define GQA_CACHE_CHECK(call)                                                  \
  do {                                                                         \
    st = (call);                                                               \
    if (st != HIPBLAS_STATUS_SUCCESS)                                          \
      goto cache_fail;                                                         \
  } while (0)

  GQA_CACHE_CHECK(
      hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  {
    hipblasOperation_t opA = key.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opN = HIPBLAS_OP_N;
    GQA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    GQA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  }

  {
    int64_t a_rows = key.transA ? k : m;
    int64_t a_cols = key.transA ? m : k;
    GQA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layA, HIP_R_16F, a_rows,
                                                a_cols, a_rows));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layA, batch, m * k));

    GQA_CACHE_CHECK(
        hipblasLtMatrixLayoutCreate(&entry.layB, HIP_R_16F, k, n, k));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layB, batch, n * k));

    GQA_CACHE_CHECK(
        hipblasLtMatrixLayoutCreate(&entry.layC, HIP_R_16F, m, n, m));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layC, batch, n * m));
    GQA_CACHE_CHECK(
        hipblasLtMatrixLayoutCreate(&entry.layD, HIP_R_16F, m, n, m));
    GQA_CACHE_CHECK(setLayoutBatch(entry.layD, batch, n * m));
  }

  GQA_CACHE_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  {
    const size_t max_ws = kMaxWorkspaceBytes;
    GQA_CACHE_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
        sizeof(max_ws)));
  }

  {
    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    GQA_CACHE_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, entry.desc, entry.layA, entry.layB, entry.layC, entry.layD,
        pref, 1, &heur, &returned));
    hipblasLtMatmulPreferenceDestroy(pref);
    pref = nullptr;

    if (returned == 0) {
      fprintf(stderr,
              "GQA: no algorithm found for GEMM m=%lld n=%lld k=%lld "
              "batch=%lld\n",
              (long long)m, (long long)n, (long long)k, (long long)key.batch);
      goto cache_fail;
    }

    entry.algo = heur.algo;
    entry.workspace_size = heur.workspaceSize;
  }

#undef GQA_CACHE_CHECK
  goto cache_done;

cache_fail:
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (entry.layD)
    hipblasLtMatrixLayoutDestroy(entry.layD);
  if (entry.layC)
    hipblasLtMatrixLayoutDestroy(entry.layC);
  if (entry.layB)
    hipblasLtMatrixLayoutDestroy(entry.layB);
  if (entry.layA)
    hipblasLtMatrixLayoutDestroy(entry.layA);
  if (entry.desc)
    hipblasLtMatmulDescDestroy(entry.desc);
  return nullptr;

cache_done:
  auto [ins, _] = g_gqa_gemm_cache.emplace(key, entry);
  return &ins->second;
}

// Shared KV cache update: concat past+new or append new tokens.
// Returns 0 on success, non-zero on failure.
static int update_kv_cache(hipStream_t stream, const void *past_key,
                           const void *past_value, const void *new_key,
                           const void *new_value, void *present_key,
                           void *present_value, int B, int past_len, int sq,
                           int G, int d, int present_seq) {
  if (past_key && past_len > 0 && past_key != present_key) {
    if (hip_gqa_kv_cache_concat(stream, past_key, new_key, present_key, B,
                                past_len, sq, G, d, past_len, present_seq) != 0)
      return -1;
    if (hip_gqa_kv_cache_concat(stream, past_value, new_value, present_value, B,
                                past_len, sq, G, d, past_len, present_seq) != 0)
      return -1;
  } else {
    if (hip_gqa_kv_cache_append(stream, new_key, present_key, B, sq, G, d,
                                present_seq, past_len) != 0)
      return -1;
    if (hip_gqa_kv_cache_append(stream, new_value, present_value, B, sq, G, d,
                                present_seq, past_len) != 0)
      return -1;
  }
  return 0;
}

// =============================================================================
// 12-step hipBLASLt GQA pipeline (Step 0 + Steps 1-11, FP16 only)
// =============================================================================

static int gqa_forward_hipblaslt(
    RuntimeState *state, hipStream_t stream, hipblasLtHandle_t ltHandle,
    const void *query,    // BSHD [B, sq, H, d] or packed [B, sq, (H+2G)*d]
    const void *key,      // BSHD [B, sq, G, d] or null (packed QKV)
    const void *value,    // BSHD [B, sq, G, d] or null (packed QKV)
    const void *past_key, // BNSD [B, G, past_seq, d] or null
    const void *past_value, const void *cos_cache, const void *sin_cache,
    void *head_sink,         // [H] smooth softmax factor or null
    bool use_smooth_softmax, // true when smooth softmax is enabled (ORT:
                             // head_sink || smooth_softmax attr)
    void *output,            // BSHD [B, sq, H, d]
    void *present_key,       // BNSD [B, G, present_seq, d]
    void *present_value, int64_t B, int64_t sq, int64_t skv, int64_t H,
    int64_t G, int64_t d, float scale, int64_t do_rotary,
    int64_t local_window_size) {

  int64_t HPG = H / G;
  int64_t past_len = skv - sq;
  if (past_len < 0)
    past_len = 0;
  int64_t present_seq = skv;
  size_t elem_sz = 2; // FP16

  bool need_rope = do_rotary && cos_cache && sin_cache;

  // =========================================================================
  // Fused GQA decode path (sq == 1, d in {64, 128, 256}, KV cache enabled)
  //
  // Replaces steps 3, 6-11 of the decomposed pipeline with a single kernel
  // that reads Q in BSHD and KV from the BNSD cache, producing O in BSHD.
  // Steps 1-2 (RoPE) and 4-5 (KV cache update) still run as separate
  // kernels before the fused dispatch.
  //
  // All prefill (sq > 1) goes through the decomposed hipBLASLt path where
  // auto-tuned GEMMs outperform fixed WMMA tiling and all ORT GQA features
  // (sliding window, smooth softmax, head sink) are supported.
  // =========================================================================
  bool fused_d = (d == 64 || d == 128 || d == 256);
  if (fused_d && sq == 1 && key && value && present_key && present_value &&
      local_window_size == 0 && !head_sink && !use_smooth_softmax) {
    const void *qSrc = query;
    const void *kSrc = key;

    if (need_rope) {
      size_t Q_bytes = (size_t)B * sq * H * d * elem_sz;
      size_t K_bytes = (size_t)B * sq * G * d * elem_sz;
      if (hipdnn_ep_state_ensure_workspace(state, Q_bytes + K_bytes) != 0)
        return -1;
      char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
      void *d_Qroped = ws;
      void *d_Kroped = ws + Q_bytes;

      int half_rot = (int)(d / 2);
      if (hip_gqa_rope(stream, query, d_Qroped, cos_cache, sin_cache, (int)B,
                       (int)sq, (int)H, (int)d, half_rot, (int)past_len) != 0)
        return -1;
      if (hip_gqa_rope(stream, key, d_Kroped, cos_cache, sin_cache, (int)B,
                       (int)sq, (int)G, (int)d, half_rot, (int)past_len) != 0)
        return -1;

      qSrc = d_Qroped;
      kSrc = d_Kroped;
    }

    if (update_kv_cache(stream, past_key, past_value, kSrc, value, present_key,
                        present_value, (int)B, (int)past_len, (int)sq, (int)G,
                        (int)d, (int)present_seq) != 0)
      return -1;

    if (hip_gqa_fused_decode(stream, qSrc, present_key, present_value, output,
                             (int)B, (int)H, (int)G, (int)d, (int)skv,
                             (int)present_seq, scale) != 0)
      return -1;

    RUNTIME_DEBUG_LOG(
        "[REAL] fused GQA decode: B=%lld sq=%lld skv=%lld H=%lld G=%lld "
        "d=%lld\n",
        (long long)B, (long long)sq, (long long)skv, (long long)H, (long long)G,
        (long long)d);
    return 0;
  }

  // =========================================================================
  // Decomposed hipBLASLt pipeline (all prefill sq > 1, unsupported d, or
  // features requiring sliding window / smooth softmax / head sink)
  // =========================================================================

  bool packed_qkv = (key == nullptr && value == nullptr);

  // ---- Workspace layout ----
  // All temp buffers are packed contiguously into the shared workspace,
  // followed by the GEMM workspace region. This eliminates per-call
  // hipMalloc/hipFree -- after the first inference the workspace is
  // already large enough and reuse is zero-cost.
  //
  // Layout: [Qtrans | Kexp | Vexp | S | O | Qroped? | Kroped? | Qsplit? |
  // Ksplit? | Vsplit? | GEMM ws]

  size_t Qtrans_bytes = (size_t)B * H * sq * d * elem_sz;
  size_t Kexp_bytes = (size_t)B * H * skv * d * elem_sz;
  size_t S_bytes = (size_t)B * H * sq * skv * elem_sz;
  size_t O_bytes = (size_t)B * H * sq * d * elem_sz;

  size_t off_Qtrans = 0;
  size_t off_Kexp = off_Qtrans + Qtrans_bytes;
  size_t off_Vexp = off_Kexp + Kexp_bytes;
  size_t off_S = off_Vexp + Kexp_bytes;
  size_t off_O = off_S + S_bytes;
  size_t temp_end = off_O + O_bytes;

  // Optional RoPE buffers: allocated only when do_rotary is enabled.
  size_t off_Qroped = 0, off_Kroped = 0;
  if (need_rope) {
    size_t Q_bytes = (size_t)B * sq * H * d * elem_sz;
    size_t K_bytes = (size_t)B * sq * G * d * elem_sz;
    off_Qroped = temp_end;
    off_Kroped = off_Qroped + Q_bytes;
    temp_end = off_Kroped + K_bytes;
  }

  // Optional packed-QKV split buffers: allocated only when key/value are
  // null (GPT-OSS style packed input).  These must be placed AFTER the RoPE
  // buffers in the layout so that split outputs remain live while RoPE reads
  // from them and writes to the RoPE region (no overlap).
  size_t off_Qsplit = 0, off_Ksplit = 0, off_Vsplit = 0;
  if (packed_qkv) {
    size_t Q_bytes = (size_t)B * sq * H * d * elem_sz;
    size_t K_bytes = (size_t)B * sq * G * d * elem_sz;
    off_Qsplit = temp_end;
    off_Ksplit = off_Qsplit + Q_bytes;
    off_Vsplit = off_Ksplit + K_bytes;
    temp_end = off_Vsplit + K_bytes;
  }

  // Query or create cached GEMM descriptors + algorithms. On first call for
  // a given shape the descriptors and layouts are created and the heuristic
  // is queried; subsequent calls reuse the cached state.
  GqaGemmKey scoreKey{skv, sq, d, B * H, true};
  GqaGemmKey valueKey{d, sq, skv, B * H, false};
  const GqaGemmCacheEntry *scoreState =
      queryOrCreateGemmState(ltHandle, scoreKey);
  if (!scoreState)
    return -1;
  const GqaGemmCacheEntry *valueState =
      queryOrCreateGemmState(ltHandle, valueKey);
  if (!valueState)
    return -1;

  int result = 0;

  // Single workspace allocation: temp buffers + GEMM workspace
  {
    size_t gemm_ws =
        std::max(scoreState->workspace_size, valueState->workspace_size);
    size_t total_needed = temp_end + gemm_ws;
    HIP_CHECK(hipdnn_ep_state_ensure_workspace(state, total_needed));
  }

  {
    char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    size_t ws_total = hipdnn_ep_state_get_workspace_size(state);

    void *d_Qtrans = ws + off_Qtrans;
    void *d_Kexp = ws + off_Kexp;
    void *d_Vexp = ws + off_Vexp;
    void *d_S = ws + off_S;
    void *d_O = ws + off_O;

    void *gemm_ws_ptr = ws + temp_end;
    size_t gemm_ws_bytes = ws_total - temp_end;

    // Mutable source pointers: initially point to the raw inputs, but get
    // redirected to workspace buffers as pipeline steps (split, RoPE) produce
    // intermediate results.  Downstream steps always read through these so
    // they automatically pick up the latest transformed data.
    const void *qSrc = query;
    const void *kSrc = key;
    const void *vSrc = value;

    // ---- Step 0: Split packed QKV (if needed) ----
    // When key/value are null, query is a packed [B, S, (H+2G)*d] tensor.
    // Split into separate Q [B*S, H*d], K [B*S, G*d], V [B*S, G*d].
    if (packed_qkv) {
      void *d_Qsplit = ws + off_Qsplit;
      void *d_Ksplit = ws + off_Ksplit;
      void *d_Vsplit = ws + off_Vsplit;
      HIP_CHECK(hip_gqa_split_qkv(stream, query, d_Qsplit, d_Ksplit, d_Vsplit,
                                  (int)B, (int)sq, (int)H, (int)G, (int)d));
      qSrc = d_Qsplit;
      kSrc = d_Ksplit;
      vSrc = d_Vsplit;
    }

    // ---- Steps 1-2: RoPE (optional) ----
    // Uses qSrc/kSrc (not raw query/key) so that packed-QKV split buffers
    // are correctly fed into RoPE when both features are active.
    if (need_rope) {
      int half_rot = (int)(d / 2);
      void *d_Qroped = ws + off_Qroped;
      void *d_Kroped = ws + off_Kroped;

      HIP_CHECK(hip_gqa_rope(stream, qSrc, d_Qroped, cos_cache, sin_cache,
                             (int)B, (int)sq, (int)H, (int)d, half_rot,
                             (int)past_len));
      HIP_CHECK(hip_gqa_rope(stream, kSrc, d_Kroped, cos_cache, sin_cache,
                             (int)B, (int)sq, (int)G, (int)d, half_rot,
                             (int)past_len));

      qSrc = d_Qroped;
      kSrc = d_Kroped;
      // vSrc is intentionally NOT updated: V is never RoPE'd.
    }

    // ---- Step 3: Q Transpose BSHD [B,S,H,d] -> BNSD [B,H,S,d] ----
    HIP_CHECK(hip_gqa_transpose_mid_dims(stream, qSrc, d_Qtrans, (int)B,
                                         (int)sq, (int)H, (int)d));

    // ---- Steps 4-5: KV Cache Update ----
    if (present_key && present_value) {
      HIP_CHECK(update_kv_cache(
          stream, past_key, past_value, kSrc, vSrc, present_key, present_value,
          (int)B, (int)past_len, (int)sq, (int)G, (int)d, (int)present_seq));
    }

    // ---- Steps 6-7: KV Expand [B*G, present_seq, d] -> [B*H, skv, d] ----
    {
      const void *kCache = present_key ? present_key : key;
      const void *vCache = present_value ? present_value : value;
      int kvSrcStride = (int)(present_seq * d);
      int kvDstStride = (int)(skv * d);
      int expandCopy = (int)(skv * d);

      HIP_CHECK(hip_gqa_expand_kv(stream, kCache, d_Kexp, (int)(B * H),
                                  (int)HPG, kvSrcStride, kvDstStride,
                                  expandCopy));
      HIP_CHECK(hip_gqa_expand_kv(stream, vCache, d_Vexp, (int)(B * H),
                                  (int)HPG, kvSrcStride, kvDstStride,
                                  expandCopy));
    }

    // ---- Step 8: Score GEMM ----
    float scoreAlpha = scale;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t sAlgo = scoreState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, scoreState->desc, &scoreAlpha, d_Kexp, scoreState->layA,
        d_Qtrans, scoreState->layB, &beta, d_S, scoreState->layC, d_S,
        scoreState->layD, &sAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 9: Causal Mask + Softmax ----
    int scoreBatchStride = (int)(sq * skv);
    if (sq > 1) {
      HIP_CHECK(hip_gqa_causal_mask(stream, d_S, (int)(B * H), (int)skv,
                                    (int)sq, scoreBatchStride, (int)past_len,
                                    (int)local_window_size));
    }
    HIP_CHECK(hip_gqa_softmax_inplace(stream, d_S, (int)(B * H * sq), (int)skv,
                                      (int)sq, scoreBatchStride, head_sink,
                                      (int)H, (int)use_smooth_softmax));

    // ---- Step 10: Value GEMM ----
    float valAlpha = 1.0f;
    hipblasLtMatmulAlgo_t vAlgo = valueState->algo;

    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueState->desc, &valAlpha, d_Vexp, valueState->layA, d_S,
        valueState->layB, &beta, d_O, valueState->layC, d_O, valueState->layD,
        &vAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 11: O Transpose BNSD [B,H,S,d] -> BSHD [B,S,H,d] ----
    HIP_CHECK(hip_gqa_transpose_mid_dims(stream, d_O, output, (int)B, (int)H,
                                         (int)sq, (int)d));
  }

cleanup:
  return result;
}

// =============================================================================
// Public wrapper called by generated IR
// =============================================================================

int wrap_group_query_attention(
    RuntimeState *state,
    // Inputs 1-7 (core GQA)
    void *query, void *key, void *value, void *past_key, void *past_value,
    void *seqlens_k, void *total_seq_len,
    // Inputs 8-10 (RoPE)
    void *cos_cache, void *sin_cache, void *position_ids,
    // Inputs 11-14 (advanced features)
    void *attention_bias, void *head_sink, void *k_scale, void *v_scale,
    // Outputs
    void *output, void *present_key, void *present_value, void *output_qk,
    // Attributes (12)
    int64_t num_heads, int64_t kv_num_heads, float scale, int64_t do_rotary,
    int64_t rotary_interleaved, float softcap, int64_t local_window_size,
    int64_t smooth_softmax, int64_t qk_output, int64_t k_quant_type,
    int64_t v_quant_type, int64_t kv_cache_bit_width,
    // Shape values (5)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv, int64_t head_dim,
    int64_t element_size_bytes) {

  if (!state) {
    fprintf(stderr, "wrap_group_query_attention: null state\n");
    return -1;
  }
  if (!query || !output) {
    fprintf(stderr, "wrap_group_query_attention: null required argument\n");
    return -1;
  }
  if (num_heads % kv_num_heads != 0) {
    fprintf(stderr,
            "wrap_group_query_attention: num_heads (%lld) must be "
            "divisible by kv_num_heads (%lld)\n",
            (long long)num_heads, (long long)kv_num_heads);
    return -1;
  }
  // The hipBLASLt GQA pipeline is currently FP16-only: all matrix layouts are
  // created with HIP_R_16F and every custom HIP kernel (rope_kernel,
  // kv_cache_append_kernel, softmax_inplace_kernel, etc.) operates on __half.
  // This matches requirements where GQA runs in FP16.
  //
  // Future FP32 (or BF16) support would require:
  //   1. Parameterizing all hipblasLtMatrixLayoutCreate() calls with the
  //      runtime data type instead of hard-coded HIP_R_16F.
  //   2. Templating the custom HIP kernels on the element type so they can
  //      handle float / __hip_bfloat16 in addition to __half.
  //   3. Adjusting workspace layout sizing for 4-byte (or other) elements.
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "wrap_group_query_attention: hipBLASLt pipeline requires "
            "FP16 (elem_size=2), got %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  hipblasLtHandle_t ltHandle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  if (!stream || !ltHandle) {
    fprintf(stderr, "wrap_group_query_attention: null stream or hipblas "
                    "handle\n");
    return -1;
  }

  // Reject features not yet implemented
  if (position_ids != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: position_ids not yet implemented\n");
    return -1;
  }
  if (attention_bias != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: attention_bias not yet implemented\n");
    return -1;
  }
  if (k_scale != nullptr || v_scale != nullptr) {
    fprintf(stderr, "wrap_group_query_attention: KV cache quantization not yet "
                    "implemented\n");
    return -1;
  }
  if (output_qk != nullptr) {
    fprintf(stderr,
            "wrap_group_query_attention: output_qk not yet implemented\n");
    return -1;
  }
  if (qk_output != 0) {
    fprintf(stderr,
            "wrap_group_query_attention: qk_output not yet implemented\n");
    return -1;
  }
  if (k_quant_type != 0 || v_quant_type != 0) {
    fprintf(
        stderr,
        "wrap_group_query_attention: quantization types not yet implemented\n");
    return -1;
  }
  if (kv_cache_bit_width != 8) {
    fprintf(stderr,
            "wrap_group_query_attention: non-8bit cache not yet implemented\n");
    return -1;
  }

  // ORT uses scale == 0.0 as sentinel for "auto-compute 1/√head_size"
  // (gqa_attention_base.h line 185: scale_ == 0.0f ? 1/sqrt(head_size) :
  // scale_).
  if (scale == 0.0f && head_dim > 0) {
    scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  }

  // Smooth softmax: activated when head_sink is provided OR smooth_softmax
  // attribute is explicitly 1, matching ORT behaviour (gqa_attention_base.h
  // line 354: use_smooth_softmax_ || head_sink != nullptr).
  // Compare == 1 following ORT's GetAttrOrDefault("smooth_softmax", 0) == 1
  // pattern (default 0 is set in HipOps.td and OnnxToHip.cpp).
  bool has_smooth_softmax = (head_sink != nullptr || smooth_softmax == 1);

  bool is_packed_qkv = (key == nullptr && value == nullptr);
  bool has_rope =
      (do_rotary == 1 && cos_cache != nullptr && sin_cache != nullptr);
  bool has_local_window = (local_window_size > 0);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_group_query_attention:\n"
      "  shapes: batch=%lld, seq_q=%lld, seq_kv=%lld, num_heads=%lld, "
      "kv_heads=%lld, head_dim=%lld\n"
      "  attrs:  scale=%f, do_rotary=%lld, rotary_interleaved=%lld, "
      "softcap=%f, local_window_size=%lld, smooth_softmax=%lld\n"
      "  inputs: query=%p, key=%p, value=%p, past_key=%p, past_value=%p\n"
      "          cos_cache=%p, sin_cache=%p, head_sink=%p\n"
      "  outputs: output=%p, present_key=%p, present_value=%p\n"
      "  mode:   packed_qkv=%d, rope=%d, local_window=%d, smooth_softmax=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (double)scale, (long long)do_rotary, (long long)rotary_interleaved,
      (double)softcap, (long long)local_window_size, (long long)smooth_softmax,
      query, key, value, past_key, past_value, cos_cache, sin_cache, head_sink,
      output, present_key, present_value, (int)is_packed_qkv, (int)has_rope,
      (int)has_local_window, (int)has_smooth_softmax);

  int rc = gqa_forward_hipblaslt(
      state, stream, ltHandle, query, key, value, past_key, past_value,
      cos_cache, sin_cache, head_sink, has_smooth_softmax, output, present_key,
      present_value, batch_size, seq_len_q, seq_len_kv, num_heads, kv_num_heads,
      head_dim, scale, do_rotary, local_window_size);

  if (rc != 0) {
    fprintf(stderr, "wrap_group_query_attention: gqa_forward failed (rc=%d)\n",
            rc);
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_query_attention: completed successfully\n");
  }

  return rc;
}
