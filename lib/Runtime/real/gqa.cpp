/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include "error_check_macros.h"

#include <algorithm>
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

// =============================================================================
// GQA GEMM algorithm cache: query heuristic once per shape, reuse after.
// =============================================================================

struct GqaGemmKey {
  int64_t m, n, k, batch;
  bool transA;
  bool operator==(const GqaGemmKey &o) const {
    return m == o.m && n == o.n && k == o.k && batch == o.batch &&
           transA == o.transA;
  }
};

struct GqaGemmKeyHash {
  size_t operator()(const GqaGemmKey &k) const {
    size_t h = std::hash<int64_t>{}(k.m);
    h ^= std::hash<int64_t>{}(k.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.batch) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.transA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct GqaGemmCacheEntry {
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
};

static std::unordered_map<GqaGemmKey, GqaGemmCacheEntry, GqaGemmKeyHash>
    g_gqa_gemm_cache;

static int queryGemmAlgo(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
                         hipblasLtMatrixLayout_t layA,
                         hipblasLtMatrixLayout_t layB,
                         hipblasLtMatrixLayout_t layC,
                         hipblasLtMatrixLayout_t layD, const GqaGemmKey &key,
                         hipblasLtMatmulAlgo_t *out_algo, size_t *out_ws) {

  auto it = g_gqa_gemm_cache.find(key);
  if (it != g_gqa_gemm_cache.end()) {
    *out_algo = it->second.algo;
    *out_ws = it->second.workspace_size;
    return 0;
  }

  hipblasLtMatmulPreference_t pref = nullptr;
  int result = 0;

  HIPBLAS_CHECK_GOTO(hipblasLtMatmulPreferenceCreate(&pref), cleanup_pref);
  {
    const size_t max_ws = 256ULL << 20;
    HIPBLAS_CHECK_GOTO(hipblasLtMatmulPreferenceSetAttribute(
                           pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                           &max_ws, sizeof(max_ws)),
                       cleanup_pref);
  }

  {
    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    HIPBLAS_CHECK_GOTO(hipblasLtMatmulAlgoGetHeuristic(handle, desc, layA, layB,
                                                       layC, layD, pref, 1,
                                                       &heur, &returned),
                       cleanup_pref);

    if (returned == 0) {
      fprintf(stderr,
              "GQA: no algorithm found for GEMM m=%lld n=%lld k=%lld "
              "batch=%lld\n",
              (long long)key.m, (long long)key.n, (long long)key.k,
              (long long)key.batch);
      result = -1;
      goto cleanup_pref;
    }

    GqaGemmCacheEntry entry;
    entry.algo = heur.algo;
    entry.workspace_size = heur.workspaceSize;
    g_gqa_gemm_cache[key] = entry;

    *out_algo = heur.algo;
    *out_ws = heur.workspaceSize;
  }

cleanup_pref:
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  return result;
}

// =============================================================================
// 11-step hipBLASLt GQA pipeline (FP16 only)
// =============================================================================

static int gqa_forward_hipblaslt(
    RuntimeState *state, hipStream_t stream, hipblasLtHandle_t ltHandle,
    const void *query,    // BSHD [B, sq, H, d]
    const void *key,      // BSHD [B, sq, G, d]
    const void *value,    // BSHD [B, sq, G, d]
    const void *past_key, // BNSD [B, G, past_seq, d] or null
    const void *past_value, const void *cos_cache, const void *sin_cache,
    void *output,      // BSHD [B, sq, H, d]
    void *present_key, // BNSD [B, G, present_seq, d]
    void *present_value, int64_t B, int64_t sq, int64_t skv, int64_t H,
    int64_t G, int64_t d, float scale, int64_t do_rotary) {

  int64_t HPG = H / G;
  int64_t past_len = skv - sq;
  if (past_len < 0)
    past_len = 0;
  // present_seq is the sequence dimension of the present KV buffer.
  // In the separate-buffer case this may differ from skv (the total active
  // token count used for GEMM dimensions) if the buffer is pre-allocated.
  int64_t present_seq = skv;

  // ---- Workspace layout ----
  // All temp buffers are packed contiguously into the shared workspace,
  // followed by the GEMM workspace region. This eliminates per-call
  // hipMalloc/hipFree -- after the first inference the workspace is
  // already large enough and reuse is zero-cost.
  //
  // Layout: [Qtrans | Kexp | Vexp | S | O | Qroped? | Kroped? | GEMM ws]

  size_t elem_sz = 2; // FP16
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

  size_t off_Qroped = 0, off_Kroped = 0;
  bool need_rope = do_rotary && cos_cache && sin_cache;
  if (need_rope) {
    size_t Q_bytes = (size_t)B * sq * H * d * elem_sz;
    size_t K_bytes = (size_t)B * sq * G * d * elem_sz;
    off_Qroped = temp_end;
    off_Kroped = off_Qroped + Q_bytes;
    temp_end = off_Kroped + K_bytes;
  }

  // Query GEMM algorithms first (cached) to know the GEMM workspace size,
  // then ensure a single workspace allocation covering everything.
  int32_t batchCount = (int32_t)(B * H);
  hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;

  // Score GEMM: S[skv, sq] = K_exp^T[skv,d] * Q_trans[d,sq] * scale
  hipblasLtMatmulDesc_t scoreDesc = nullptr;
  hipblasLtMatrixLayout_t sLayA = nullptr, sLayB = nullptr, sLayC = nullptr,
                          sLayD = nullptr;
  hipblasLtMatmulDesc_t valueDesc = nullptr;
  hipblasLtMatrixLayout_t vLayA = nullptr, vLayB = nullptr, vLayC = nullptr,
                          vLayD = nullptr;
  GqaGemmKey scoreKey{skv, sq, d, B * H, true};
  GqaGemmKey valueKey{d, sq, skv, B * H, false};
  hipblasLtMatmulAlgo_t scoreAlgo, valueAlgo;
  size_t scoreWs = 0, valueWs = 0;
  int result = 0;

  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&scoreDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      scoreDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      scoreDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&sLayA, HIP_R_16F, d, skv, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&sLayB, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&sLayC, HIP_R_16F, skv, sq, skv));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&sLayD, HIP_R_16F, skv, sq, skv));
  HIPBLAS_CHECK(setLayoutBatch(sLayA, batchCount, (int64_t)skv * d));
  HIPBLAS_CHECK(setLayoutBatch(sLayB, batchCount, (int64_t)sq * d));
  HIPBLAS_CHECK(setLayoutBatch(sLayC, batchCount, (int64_t)sq * skv));
  HIPBLAS_CHECK(setLayoutBatch(sLayD, batchCount, (int64_t)sq * skv));

  // Value GEMM: O[d, sq] = V_exp[d,skv] * softmax(S)[skv,sq]
  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&valueDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      valueDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      valueDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayA, HIP_R_16F, d, skv, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayB, HIP_R_16F, skv, sq, skv));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayC, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayD, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(setLayoutBatch(vLayA, batchCount, (int64_t)skv * d));
  HIPBLAS_CHECK(setLayoutBatch(vLayB, batchCount, (int64_t)sq * skv));
  HIPBLAS_CHECK(setLayoutBatch(vLayC, batchCount, (int64_t)sq * d));
  HIPBLAS_CHECK(setLayoutBatch(vLayD, batchCount, (int64_t)sq * d));

  if (queryGemmAlgo(ltHandle, scoreDesc, sLayA, sLayB, sLayC, sLayD, scoreKey,
                    &scoreAlgo, &scoreWs) != 0) {
    result = -1;
    goto cleanup;
  }
  if (queryGemmAlgo(ltHandle, valueDesc, vLayA, vLayB, vLayC, vLayD, valueKey,
                    &valueAlgo, &valueWs) != 0) {
    result = -1;
    goto cleanup;
  }

  // Single workspace allocation: temp buffers + GEMM workspace
  {
    size_t gemm_ws = std::max(scoreWs, valueWs);
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

    const void *qSrc = query;
    const void *kSrc = key;

    // ---- Steps 1-2: RoPE (optional) ----
    if (need_rope) {
      int half_rot = (int)(d / 2);
      void *d_Qroped = ws + off_Qroped;
      void *d_Kroped = ws + off_Kroped;

      HIP_CHECK(hip_gqa_rope(stream, query, d_Qroped, cos_cache, sin_cache,
                             (int)B, (int)sq, (int)H, (int)d, half_rot,
                             (int)past_len));
      HIP_CHECK(hip_gqa_rope(stream, key, d_Kroped, cos_cache, sin_cache,
                             (int)B, (int)sq, (int)G, (int)d, half_rot,
                             (int)past_len));

      qSrc = d_Qroped;
      kSrc = d_Kroped;
    }

    // ---- Step 3: Q Transpose BSHD [B,S,H,d] -> BNSD [B,H,S,d] ----
    HIP_CHECK(hip_gqa_transpose_mid_dims(stream, qSrc, d_Qtrans, (int)B,
                                         (int)sq, (int)H, (int)d));

    // ---- Steps 4-5: KV Cache Update ----
    if (present_key && present_value) {
      if (past_key && past_len > 0 && past_key != present_key) {
        // Separate buffers: past [B,G,past_len,d] and present
        // [B,G,present_seq,d] have different strides. A single kernel copies
        // past data at [0,past_len) and transposes new tokens from BSHD into
        // [past_len,past_len+sq).
        HIP_CHECK(hip_gqa_kv_cache_concat(
            stream, past_key, kSrc, present_key, (int)B, (int)past_len, (int)sq,
            (int)G, (int)d, (int)past_len, (int)present_seq));
        HIP_CHECK(hip_gqa_kv_cache_concat(
            stream, past_value, value, present_value, (int)B, (int)past_len,
            (int)sq, (int)G, (int)d, (int)past_len, (int)present_seq));
      } else {
        // Same buffer (aliased / in-place): past data already at correct
        // offsets, only append new tokens at [past_len, past_len+sq).
        HIP_CHECK(hip_gqa_kv_cache_append(stream, kSrc, present_key, (int)B,
                                          (int)sq, (int)G, (int)d,
                                          (int)present_seq, (int)past_len));
        HIP_CHECK(hip_gqa_kv_cache_append(stream, value, present_value, (int)B,
                                          (int)sq, (int)G, (int)d,
                                          (int)present_seq, (int)past_len));
      }
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

    HIPBLAS_CHECK(hipblasLtMatmul(ltHandle, scoreDesc, &scoreAlpha, d_Kexp,
                                  sLayA, d_Qtrans, sLayB, &beta, d_S, sLayC,
                                  d_S, sLayD, &scoreAlgo, gemm_ws_ptr,
                                  gemm_ws_bytes, stream));

    // ---- Step 9: Causal Mask + Softmax ----
    int scoreBatchStride = (int)(sq * skv);
    if (sq > 1) {
      HIP_CHECK(hip_gqa_causal_mask(stream, d_S, (int)(B * H), (int)skv,
                                    (int)sq, scoreBatchStride, (int)past_len));
    }
    HIP_CHECK(hip_gqa_softmax_inplace(stream, d_S, (int)(B * H * sq), (int)skv,
                                      (int)sq, scoreBatchStride));

    // ---- Step 10: Value GEMM ----
    float valAlpha = 1.0f;
    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueDesc, &valAlpha, d_Vexp, vLayA, d_S, vLayB, &beta, d_O,
        vLayC, d_O, vLayD, &valueAlgo, gemm_ws_ptr, gemm_ws_bytes, stream));

    // ---- Step 11: O Transpose BNSD [B,H,S,d] -> BSHD [B,S,H,d] ----
    HIP_CHECK(hip_gqa_transpose_mid_dims(stream, d_O, output, (int)B, (int)H,
                                         (int)sq, (int)d));
  }

cleanup:
  if (sLayA)
    hipblasLtMatrixLayoutDestroy(sLayA);
  if (sLayB)
    hipblasLtMatrixLayoutDestroy(sLayB);
  if (sLayC)
    hipblasLtMatrixLayoutDestroy(sLayC);
  if (sLayD)
    hipblasLtMatrixLayoutDestroy(sLayD);
  if (vLayA)
    hipblasLtMatrixLayoutDestroy(vLayA);
  if (vLayB)
    hipblasLtMatrixLayoutDestroy(vLayB);
  if (vLayC)
    hipblasLtMatrixLayoutDestroy(vLayC);
  if (vLayD)
    hipblasLtMatrixLayoutDestroy(vLayD);
  if (scoreDesc)
    hipblasLtMatmulDescDestroy(scoreDesc);
  if (valueDesc)
    hipblasLtMatmulDescDestroy(valueDesc);

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
  if (!query || !key || !value || !output) {
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

  // Phase 1: Warn about unimplemented features (Phase 2 will implement)
  if (position_ids != nullptr) {
    RUNTIME_DEBUG_LOG("[WARN] position_ids not yet implemented (Phase 2)\n");
  }
  if (attention_bias != nullptr) {
    RUNTIME_DEBUG_LOG("[WARN] attention_bias not yet implemented (Phase 2)\n");
  }
  if (head_sink != nullptr) {
    RUNTIME_DEBUG_LOG("[WARN] head_sink not yet implemented (Phase 2)\n");
  }
  if (k_scale != nullptr || v_scale != nullptr) {
    RUNTIME_DEBUG_LOG(
        "[WARN] KV cache quantization not yet implemented (Phase 2)\n");
  }
  if (output_qk != nullptr) {
    RUNTIME_DEBUG_LOG("[WARN] output_qk not yet implemented (Phase 2)\n");
  }
  if (local_window_size != -1) {
    RUNTIME_DEBUG_LOG(
        "[WARN] local_window_size not yet implemented (Phase 2)\n");
  }
  if (smooth_softmax != 0) {
    RUNTIME_DEBUG_LOG("[WARN] smooth_softmax not yet implemented (Phase 2)\n");
  }
  if (qk_output != 0) {
    RUNTIME_DEBUG_LOG("[WARN] qk_output not yet implemented (Phase 2)\n");
  }
  if (k_quant_type != 0 || v_quant_type != 0) {
    RUNTIME_DEBUG_LOG(
        "[WARN] quantization types not yet implemented (Phase 2)\n");
  }
  if (kv_cache_bit_width != 8) {
    RUNTIME_DEBUG_LOG("[WARN] non-8bit cache not yet implemented (Phase 2)\n");
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_group_query_attention: batch=%lld, seq_q=%lld, "
      "seq_kv=%lld, num_heads=%lld, kv_heads=%lld, head_dim=%lld, "
      "scale=%f, do_rotary=%lld, elem_size=%lld\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)kv_num_heads, (long long)head_dim,
      (double)scale, (long long)do_rotary, (long long)element_size_bytes);

  int rc = gqa_forward_hipblaslt(state, stream, ltHandle, query, key, value,
                                 past_key, past_value, cos_cache, sin_cache,
                                 output, present_key, present_value, batch_size,
                                 seq_len_q, seq_len_kv, num_heads, kv_num_heads,
                                 head_dim, scale, do_rotary);

  if (rc != 0) {
    fprintf(stderr, "wrap_group_query_attention: gqa_forward failed (rc=%d)\n",
            rc);
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_query_attention: completed successfully\n");
  }

  return rc;
}
