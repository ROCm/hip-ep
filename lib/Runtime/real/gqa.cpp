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
// 12-step hipBLASLt GQA pipeline (Step 0 + Steps 1-11, FP16 only)
// =============================================================================

static int gqa_forward_hipblaslt(
    RuntimeState *state, hipStream_t stream, hipblasLtHandle_t ltHandle,
    const void *query,    // BSHD [B, sq, H, d] or packed [B, sq, (H+2G)*d]
    const void *key,      // BSHD [B, sq, G, d] or null (packed QKV)
    const void *value,    // BSHD [B, sq, G, d] or null (packed QKV)
    const void *past_key, // BNSD [B, G, past_buf_seq, d] or null
    const void *past_value,
    const void *seqlens_k_ptr, // GPU pointer to seqlens_k [B] int32
    const void *cos_cache, const void *sin_cache,
    void *head_sink,         // [H] smooth softmax factor or null
    bool use_smooth_softmax, // true when smooth softmax is enabled (ORT:
                             // head_sink || smooth_softmax attr)
    void *output,            // BSHD [B, sq, H, d]
    void *present_key,       // BNSD [B, G, present_buf_seq, d]
    void *present_value, int64_t B, int64_t sq, int64_t skv,
    int64_t past_buf_seq, int64_t H, int64_t G, int64_t d, float scale,
    int64_t do_rotary, int64_t local_window_size) {

  int64_t HPG = H / G;
  // present_seq is the buffer dimension of present_key (may be max_length
  // for pre-allocated caches, larger than the actual valid token count).
  int64_t present_seq = skv;

  // Determine actual total sequence length from seqlens_k (ORT convention:
  // seqlens_k[b] = total_valid_tokens - 1).  When seqlens_k is not provided,
  // fall back to the buffer dimension (non-pre-allocated case).
  int64_t total_seq = skv;
  if (seqlens_k_ptr) {
    int32_t seqlens_k_val = 0;
    hipError_t memErr =
        hipMemcpyAsync(&seqlens_k_val, seqlens_k_ptr, sizeof(int32_t),
                       hipMemcpyDeviceToHost, stream);
    if (memErr == hipSuccess)
      memErr = hipStreamSynchronize(stream);
    if (memErr != hipSuccess) {
      fprintf(stderr, "GQA: failed to read seqlens_k from GPU\n");
      return -1;
    }
    total_seq = (int64_t)seqlens_k_val + 1;
  }

  int64_t past_len = total_seq - sq;
  if (past_len < 0)
    past_len = 0;

  bool packed_qkv = (key == nullptr && value == nullptr);

  // ---- Workspace layout ----
  // All temp buffers are packed contiguously into the shared workspace,
  // followed by the GEMM workspace region. This eliminates per-call
  // hipMalloc/hipFree -- after the first inference the workspace is
  // already large enough and reuse is zero-cost.
  //
  // Layout: [Qtrans | Kexp | Vexp | S | O | Qroped? | Kroped? | Qsplit? |
  // Ksplit? | Vsplit? | GEMM ws]

  size_t elem_sz = 2; // FP16
  size_t Qtrans_bytes = (size_t)B * H * sq * d * elem_sz;
  size_t Kexp_bytes = (size_t)B * H * total_seq * d * elem_sz;
  size_t S_bytes = (size_t)B * H * sq * total_seq * elem_sz;
  size_t O_bytes = (size_t)B * H * sq * d * elem_sz;

  size_t off_Qtrans = 0;
  size_t off_Kexp = off_Qtrans + Qtrans_bytes;
  size_t off_Vexp = off_Kexp + Kexp_bytes;
  size_t off_S = off_Vexp + Kexp_bytes;
  size_t off_O = off_S + S_bytes;
  size_t temp_end = off_O + O_bytes;

  // Optional RoPE buffers: allocated only when do_rotary is enabled.
  size_t off_Qroped = 0, off_Kroped = 0;
  bool need_rope = (do_rotary == 1) && cos_cache && sin_cache;
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

  // Query GEMM algorithms first (cached) to know the GEMM workspace size,
  // then ensure a single workspace allocation covering everything.
  int32_t batchCount = (int32_t)(B * H);
  hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;

  // Score GEMM: S[T, sq] = K_exp^T[T,d] * Q_trans[d,sq] * scale
  // where T = total_seq (actual valid tokens, not buffer dimension)
  hipblasLtMatmulDesc_t scoreDesc = nullptr;
  hipblasLtMatrixLayout_t sLayA = nullptr, sLayB = nullptr, sLayC = nullptr,
                          sLayD = nullptr;
  hipblasLtMatmulDesc_t valueDesc = nullptr;
  hipblasLtMatrixLayout_t vLayA = nullptr, vLayB = nullptr, vLayC = nullptr,
                          vLayD = nullptr;
  GqaGemmKey scoreKey{total_seq, sq, d, B * H, true};
  GqaGemmKey valueKey{d, sq, total_seq, B * H, false};
  hipblasLtMatmulAlgo_t scoreAlgo, valueAlgo;
  size_t scoreWs = 0, valueWs = 0;
  int result = 0;

  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&scoreDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      scoreDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      scoreDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  HIPBLAS_CHECK(
      hipblasLtMatrixLayoutCreate(&sLayA, HIP_R_16F, d, total_seq, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&sLayB, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(
      hipblasLtMatrixLayoutCreate(&sLayC, HIP_R_16F, total_seq, sq, total_seq));
  HIPBLAS_CHECK(
      hipblasLtMatrixLayoutCreate(&sLayD, HIP_R_16F, total_seq, sq, total_seq));
  HIPBLAS_CHECK(setLayoutBatch(sLayA, batchCount, (int64_t)total_seq * d));
  HIPBLAS_CHECK(setLayoutBatch(sLayB, batchCount, (int64_t)sq * d));
  HIPBLAS_CHECK(setLayoutBatch(sLayC, batchCount, (int64_t)sq * total_seq));
  HIPBLAS_CHECK(setLayoutBatch(sLayD, batchCount, (int64_t)sq * total_seq));

  // Value GEMM: O[d, sq] = V_exp[d,T] * softmax(S)[T,sq]
  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&valueDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      valueDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      valueDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

  HIPBLAS_CHECK(
      hipblasLtMatrixLayoutCreate(&vLayA, HIP_R_16F, d, total_seq, d));
  HIPBLAS_CHECK(
      hipblasLtMatrixLayoutCreate(&vLayB, HIP_R_16F, total_seq, sq, total_seq));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayC, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&vLayD, HIP_R_16F, d, sq, d));
  HIPBLAS_CHECK(setLayoutBatch(vLayA, batchCount, (int64_t)total_seq * d));
  HIPBLAS_CHECK(setLayoutBatch(vLayB, batchCount, (int64_t)sq * total_seq));
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
    // Zero present buffers when past != present (separate buffers) so that
    // the unused region [total_seq, present_seq) is deterministic, matching
    // ORT CPU which memsets present to zero before populating valid entries.
    if (present_key && present_value && past_key != present_key) {
      size_t present_bytes = (size_t)B * G * present_seq * d * elem_sz;
      HIP_CHECK(hipMemsetAsync(present_key, 0, present_bytes, stream));
      HIP_CHECK(hipMemsetAsync(present_value, 0, present_bytes, stream));
    }

    if (present_key && present_value) {
      if (past_key && past_len > 0 && past_key != present_key) {
        // Separate buffers: past [B,G,past_buf_seq,d] and present
        // [B,G,present_seq,d] may have different strides. A single kernel
        // copies past data at [0,past_len) and transposes new tokens from
        // BSHD into [past_len,past_len+sq).  past_buf_seq is the actual
        // buffer dimension of past_key (may be larger than past_len when the
        // cache is pre-allocated at max_length).
        HIP_CHECK(hip_gqa_kv_cache_concat(
            stream, past_key, kSrc, present_key, (int)B, (int)past_len, (int)sq,
            (int)G, (int)d, (int)past_buf_seq, (int)present_seq));
        HIP_CHECK(hip_gqa_kv_cache_concat(
            stream, past_value, vSrc, present_value, (int)B, (int)past_len,
            (int)sq, (int)G, (int)d, (int)past_buf_seq, (int)present_seq));
      } else {
        // Same buffer (aliased / in-place): past data already at correct
        // offsets, only append new tokens at [past_len, past_len+sq).
        HIP_CHECK(hip_gqa_kv_cache_append(stream, kSrc, present_key, (int)B,
                                          (int)sq, (int)G, (int)d,
                                          (int)present_seq, (int)past_len));
        HIP_CHECK(hip_gqa_kv_cache_append(stream, vSrc, present_value, (int)B,
                                          (int)sq, (int)G, (int)d,
                                          (int)present_seq, (int)past_len));
      }
    }

    // ---- Steps 6-7: KV Expand [B*G, present_seq, d] -> [B*H, total_seq, d]
    // Source reads from present_key with buffer stride present_seq*d; dest
    // uses total_seq*d since GEMM only operates on total_seq tokens.
    {
      const void *kCache = present_key ? present_key : key;
      const void *vCache = present_value ? present_value : value;
      int kvSrcStride = (int)(present_seq * d);
      int kvDstStride = (int)(total_seq * d);
      int expandCopy = (int)(total_seq * d);

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
    // Causal mask (prefill only): masks future tokens, and when
    // local_window_size > 0 also masks distant past outside the window.
    // Smooth softmax: activated when head_sink is non-null (per-head sink
    // factors in the denominator) or when smooth_softmax attr == 1 (uses 0
    // as the sink value, matching ORT default).
    int scoreBatchStride = (int)(sq * total_seq);
    if (sq > 1) {
      HIP_CHECK(hip_gqa_causal_mask(stream, d_S, (int)(B * H), (int)total_seq,
                                    (int)sq, scoreBatchStride, (int)past_len,
                                    (int)local_window_size));
    }
    HIP_CHECK(hip_gqa_softmax_inplace(
        stream, d_S, (int)(B * H * sq), (int)total_seq, (int)sq,
        scoreBatchStride, head_sink, (int)H, (int)use_smooth_softmax));

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
    // Shape values (6)
    int64_t batch_size, int64_t seq_len_q, int64_t seq_len_kv,
    int64_t past_seq_len, int64_t head_dim, int64_t element_size_bytes) {

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
      "  shapes: batch=%lld, seq_q=%lld, seq_kv=%lld, past_seq=%lld, "
      "num_heads=%lld, kv_heads=%lld, head_dim=%lld\n"
      "  attrs:  scale=%f, do_rotary=%lld, rotary_interleaved=%lld, "
      "softcap=%f, local_window_size=%lld, smooth_softmax=%lld\n"
      "  inputs: query=%p, key=%p, value=%p, past_key=%p, past_value=%p\n"
      "          cos_cache=%p, sin_cache=%p, head_sink=%p\n"
      "  outputs: output=%p, present_key=%p, present_value=%p\n"
      "  mode:   packed_qkv=%d, rope=%d, local_window=%d, smooth_softmax=%d\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)past_seq_len, (long long)num_heads, (long long)kv_num_heads,
      (long long)head_dim, (double)scale, (long long)do_rotary,
      (long long)rotary_interleaved, (double)softcap,
      (long long)local_window_size, (long long)smooth_softmax, query, key,
      value, past_key, past_value, cos_cache, sin_cache, head_sink, output,
      present_key, present_value, (int)is_packed_qkv, (int)has_rope,
      (int)has_local_window, (int)has_smooth_softmax);

  int rc = gqa_forward_hipblaslt(
      state, stream, ltHandle, query, key, value, past_key, past_value,
      seqlens_k, cos_cache, sin_cache, head_sink, has_smooth_softmax, output,
      present_key, present_value, batch_size, seq_len_q, seq_len_kv,
      past_seq_len, num_heads, kv_num_heads, head_dim, scale, do_rotary,
      local_window_size);

  if (rc != 0) {
    fprintf(stderr, "wrap_group_query_attention: gqa_forward failed (rc=%d)\n",
            rc);
  } else {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_group_query_attention: completed successfully\n");
  }

  return rc;
}
