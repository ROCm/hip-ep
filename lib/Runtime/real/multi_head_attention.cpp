/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "cache_utils.h"
#include "error_check_macros.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <unordered_map>

#define HIP_CHECK(cmd) HIP_CHECK_GOTO(cmd, cleanup)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

//===----------------------------------------------------------------------===//
// com.microsoft.MultiHeadAttention runtime
//===----------------------------------------------------------------------===//
//
// Implements the standard "Q_K_V_BSNH" path of MS MultiHeadAttention (ORT
// contrib op): self-/cross-attention with three separate rank-3 inputs
//
//   query : [B, S_q,  N * H]   fp16
//   key   : [B, S_kv, N * H]   fp16
//   value : [B, S_kv, N * H]   fp16
//   output: [B, S_q,  N * H]   fp16
//
// Algorithm (decomposed hipBLASLt pipeline, same shape as GQA's decomposed
// path but simpler because N_q == N_kv so no KV-expansion step is needed):
//
//   1. Q transpose [B, S_q,  N, H] -> [B, N, S_q,  H]   (skipped when S_q==1)
//   2. K transpose [B, S_kv, N, H] -> [B, N, S_kv, H]   (skipped when S_kv==1)
//   3. V transpose [B, S_kv, N, H] -> [B, N, S_kv, H]   (skipped when S_kv==1)
//   4. Score GEMM: S_f32 = (Q @ K^T) * scale   (fp16->fp32, batched B*N)
//   5. Causal mask  (only if unidirectional == 1)
//   6. Softmax:    P_f16 = softmax(S_f32)      (fp32->fp16)
//   7. Value GEMM: O_BNSH = P_f16 @ V          (fp16, batched B*N)
//   8. O transpose [B, N, S_q, H] -> [B, S_q, N, H]    (skipped when S_q==1)
//
// All hipBLASLt descriptors and heuristic-selected algorithms are cached in
// a per-session unordered_map keyed on (m, n, k, batch, strides, transA,
// outputFp32). The matmul workspace and the transpose / score / softmax /
// output scratch all share the per-session growable device buffer obtained
// via hipdnn_ep_state_(ensure|get)_workspace.
//
// Unsupported features (any of which currently return -1 with an error log):
//   * Packed QKV (rank-5 query [B, S, N, 3, H] or rank-3 [B, S, 3*N*H])
//   * Packed KV  (rank-5 key [B, S_kv, N, 2, H])
//   * BNSH past_key / past_value (KV cache)
//   * bias (per-head input projection bias - q_bias|k_bias|v_bias)
//   * key_padding_mask
//   * attention_bias
//   * cache_indirection (beam search)
//   * present_key / present_value / qk outputs
//   * Non-fp16 element types
//
// These would each be straightforward incremental extensions (most reuse
// kernels we already have for GQA), but they are deliberately left out so
// that any test exercising an unsupported case fails loudly with a clear
// log instead of silently producing wrong numbers.

namespace {

//===----------------------------------------------------------------------===//
// MHA hipBLASLt GEMM descriptor cache (per RuntimeState session)
//===----------------------------------------------------------------------===//

struct MhaGemmKey {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch;
  int64_t strideA;
  int64_t strideB;
  int64_t strideC;
  bool transA;
  bool outputFp32;

  bool operator==(const MhaGemmKey &o) const {
    return m == o.m && n == o.n && k == o.k && batch == o.batch &&
           strideA == o.strideA && strideB == o.strideB &&
           strideC == o.strideC && transA == o.transA &&
           outputFp32 == o.outputFp32;
  }
};

struct MhaGemmKeyHash {
  template <typename T> static void hash_combine_val(size_t &h, const T &v) {
    h ^= std::hash<T>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  size_t operator()(const MhaGemmKey &k) const noexcept {
    size_t h = 0;
    hash_combine_val(h, k.m);
    hash_combine_val(h, k.n);
    hash_combine_val(h, k.k);
    hash_combine_val(h, k.batch);
    hash_combine_val(h, k.strideA);
    hash_combine_val(h, k.strideB);
    hash_combine_val(h, k.strideC);
    hash_combine_val(h, (int)k.transA);
    hash_combine_val(h, (int)k.outputFp32);
    return h;
  }
};

struct MhaGemmCacheEntry {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t layA;
  hipblasLtMatrixLayout_t layB;
  hipblasLtMatrixLayout_t layC;
  hipblasLtMatrixLayout_t layD;
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
};

struct MhaGemmCache {
  std::unordered_map<MhaGemmKey, MhaGemmCacheEntry, MhaGemmKeyHash> entries;
  // Destroys every cached hipBLASLt descriptor/layout entry. Defined
  // out-of-line below. Runs when the owning op-state slot is torn down
  // (MhaState's deletor).
  ~MhaGemmCache();
};

// Per-instance MultiHeadAttention op-state (see op-state-slots-design.md):
// owns this instance's per-GEMM-shape hipBLASLt descriptor/algorithm cache.
// Replaces the former shared RuntimeState::mha_gemm_cache, so concurrent
// sessions (and distinct MHA layers) no longer share one descriptor map.
struct MhaState : OpStateT<MhaState> {
  MhaGemmCache cache;
};

// Resolve this MHA instance's descriptor cache from its op-state slot. Returns
// nullptr when the slot is unconstructed (init failure) — callers propagate the
// error rather than lazily allocating, since the slot is built at session init.
MhaGemmCache *get_mha_gemm_cache(RuntimeState *state, int op_state_slot) {
  MhaState *ms = MhaState::get_op_state(state, op_state_slot);
  return ms ? &ms->cache : nullptr;
}

//===----------------------------------------------------------------------===//
// hipBLASLt layout helper (same convention as GQA's setLayoutBatch)
//===----------------------------------------------------------------------===//

hipblasStatus_t setLayoutBatch(hipblasLtMatrixLayout_t layout,
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
// Build / look up a hipBLASLt GEMM descriptor for a particular shape
//===----------------------------------------------------------------------===//

const MhaGemmCacheEntry *queryOrCreateMhaGemm(RuntimeState *state,
                                              hipblasLtHandle_t handle,
                                              const MhaGemmKey &key,
                                              int op_state_slot) {
  assert(handle && "queryOrCreateMhaGemm: null handle");
  auto *cache = get_mha_gemm_cache(state, op_state_slot);
  if (!cache) {
    fprintf(stderr, "queryOrCreateMhaGemm: no MhaState at slot %d\n",
            op_state_slot);
    return nullptr;
  }
  auto it = cache->entries.find(key);
  if (it != cache->entries.end())
    return &it->second;

  int64_t m = key.m, n = key.n, k = key.k;
  int32_t batch = static_cast<int32_t>(key.batch);

  MhaGemmCacheEntry entry = {};

  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasStatus_t st_status;

#define MHA_CACHE_CHECK(call)                                                  \
  do {                                                                         \
    st_status = (call);                                                        \
    if (st_status != HIPBLAS_STATUS_SUCCESS)                                   \
      goto cache_fail;                                                         \
  } while (0)

  MHA_CACHE_CHECK(
      hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  {
    hipblasOperation_t opA = key.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opN = HIPBLAS_OP_N;
    MHA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    MHA_CACHE_CHECK(hipblasLtMatmulDescSetAttribute(
        entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  }
  {
    int64_t a_rows = key.transA ? k : m;
    int64_t a_cols = key.transA ? m : k;
    MHA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layA, HIP_R_16F, a_rows,
                                                a_cols, a_rows));
    MHA_CACHE_CHECK(setLayoutBatch(entry.layA, batch, key.strideA));

    MHA_CACHE_CHECK(
        hipblasLtMatrixLayoutCreate(&entry.layB, HIP_R_16F, k, n, k));
    MHA_CACHE_CHECK(setLayoutBatch(entry.layB, batch, key.strideB));

    hipDataType outType = key.outputFp32 ? HIP_R_32F : HIP_R_16F;
    MHA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layC, outType, m, n, m));
    MHA_CACHE_CHECK(setLayoutBatch(entry.layC, batch, key.strideC));
    MHA_CACHE_CHECK(hipblasLtMatrixLayoutCreate(&entry.layD, outType, m, n, m));
    MHA_CACHE_CHECK(setLayoutBatch(entry.layD, batch, key.strideC));
  }
  MHA_CACHE_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  {
    const size_t max_ws = kMaxWorkspaceBytes;
    MHA_CACHE_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
        sizeof(max_ws)));
  }
  {
    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    MHA_CACHE_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, entry.desc, entry.layA, entry.layB, entry.layC, entry.layD,
        pref, 1, &heur, &returned));
    hipblasLtMatmulPreferenceDestroy(pref);
    pref = nullptr;
    if (returned == 0) {
      fprintf(stderr,
              "MHA: no hipBLASLt algorithm found for GEMM m=%lld n=%lld "
              "k=%lld batch=%lld transA=%d outputFp32=%d\n",
              (long long)m, (long long)n, (long long)k, (long long)batch,
              (int)key.transA, (int)key.outputFp32);
      goto cache_fail;
    }
    entry.algo = heur.algo;
    entry.workspace_size = heur.workspaceSize;
  }
#undef MHA_CACHE_CHECK
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
  auto [ins, _] = cache->entries.emplace(key, entry);
  return &ins->second;
}

MhaGemmCache::~MhaGemmCache() {
  for (auto &kv : entries) {
    auto &e = kv.second;
    if (e.layD)
      hipblasLtMatrixLayoutDestroy(e.layD);
    if (e.layC)
      hipblasLtMatrixLayoutDestroy(e.layC);
    if (e.layB)
      hipblasLtMatrixLayoutDestroy(e.layB);
    if (e.layA)
      hipblasLtMatrixLayoutDestroy(e.layA);
    if (e.desc)
      hipblasLtMatmulDescDestroy(e.desc);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Per-op state construction (called from the generated op-states init).
//===----------------------------------------------------------------------===//

// Construct this MHA instance's op-state slot (see op-state-slots-design.md).
// No compile-time params: the descriptor cache fills lazily per GEMM shape. The
// construct fn stores the state into op_states[slot] itself.
extern "C" int8_t
hipdnn_ep_op_state_construct_multi_head_attention(RuntimeState *state,
                                                  int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, MhaState::create().release());
  return 0;
}

//===----------------------------------------------------------------------===//
// Public runtime wrapper - signature must match the declaration in
// hipdnn_ep_runtime.h and the lowering in MultiHeadAttentionLowering.cpp.
//===----------------------------------------------------------------------===//

extern "C" int wrap_multi_head_attention(
    RuntimeState *state,
    // Per-instance op-state slot (MhaState: this layer's GEMM descriptor cache)
    int op_state_slot, void *query, void *key, void *value, void *bias,
    void *key_padding_mask, void *attention_bias, void *past_key,
    void *past_value, void *past_sequence_length, void *cache_indirection,
    void *output, void *present_key, void *present_value, void *qk,
    int64_t num_heads, float mask_filter_value, float scale,
    int64_t unidirectional, int64_t batch_size, int64_t seq_len_q,
    int64_t seq_len_kv, int64_t query_hidden, int64_t v_hidden,
    int64_t head_size, int64_t query_rank, int64_t element_size_bytes) {
  (void)mask_filter_value; // currently unused (causal mask uses the -inf/-65504
                           // sentinel from hip_gqa_causal_mask_f32)

  OP_PROFILE(
      "multi_head_attention",
      [&] {
        char b[80];
        snprintf(b, sizeof(b), "b=%lld,sq=%lld,skv=%lld,n=%lld,h=%lld%s",
                 (long long)batch_size, (long long)seq_len_q,
                 (long long)seq_len_kv, (long long)num_heads,
                 (long long)head_size, unidirectional ? ",causal" : "");
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG(
      "[multi_head_attention] enter: B=%lld sq=%lld skv=%lld N=%lld "
      "Hq=%lld Hv=%lld H=%lld query_rank=%lld elem=%lld scale=%.6f "
      "unidir=%lld\n",
      (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
      (long long)num_heads, (long long)query_hidden, (long long)v_hidden,
      (long long)head_size, (long long)query_rank,
      (long long)element_size_bytes, (double)scale, (long long)unidirectional);

  RUNTIME_DEBUG_LOG(
      "[multi_head_attention] ptrs: q=%p k=%p v=%p bias=%p mask=%p "
      "attn_bias=%p past_k=%p past_v=%p past_seq=%p cache_ind=%p "
      "out=%p present_k=%p present_v=%p qk=%p\n",
      query, key, value, bias, key_padding_mask, attention_bias, past_key,
      past_value, past_sequence_length, cache_indirection, output, present_key,
      present_value, qk);

  // ---- Validate inputs against the implemented (Q_K_V_BSNH fp16) subset ----
  if (!query || !output) {
    fprintf(stderr, "[multi_head_attention] ERROR: query or output is null\n");
    return -1;
  }
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: only fp16 (elem=2) is supported "
            "(got elem=%lld)\n",
            (long long)element_size_bytes);
    return -1;
  }
  if (query_rank != 3) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: only rank-3 query [B,S,N*H] is "
            "supported (got rank=%lld - packed QKV / BSN3H not yet "
            "implemented)\n",
            (long long)query_rank);
    return -1;
  }
  if (!key || !value) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: only separate Q/K/V (Q_K_V_BSNH) "
            "is supported; packed QKV in query is not yet implemented "
            "(key=%p value=%p)\n",
            key, value);
    return -1;
  }
  if (bias || key_padding_mask || attention_bias || past_key || past_value ||
      past_sequence_length || cache_indirection) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: bias / key_padding_mask / "
            "attention_bias / past KV / cache_indirection are not yet "
            "implemented (bias=%p mask=%p attn_bias=%p past_k=%p past_v=%p "
            "past_seq=%p cache_ind=%p)\n",
            bias, key_padding_mask, attention_bias, past_key, past_value,
            past_sequence_length, cache_indirection);
    return -1;
  }
  if (present_key || present_value || qk) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: present_key / present_value / qk "
            "outputs are not yet implemented (present_k=%p present_v=%p "
            "qk=%p)\n",
            present_key, present_value, qk);
    return -1;
  }
  if (num_heads <= 0 || batch_size <= 0 || seq_len_q <= 0 || seq_len_kv <= 0 ||
      query_hidden <= 0) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: invalid shape (B=%lld sq=%lld "
            "skv=%lld N=%lld query_hidden=%lld)\n",
            (long long)batch_size, (long long)seq_len_q, (long long)seq_len_kv,
            (long long)num_heads, (long long)query_hidden);
    return -1;
  }

  // Derive head_size when the lowering passed the sentinel 0 (rank-3 path).
  // For Q_K_V_BSNH: Q hidden = N * H. We require Hq == Hv (same head_size)
  // - independent Hq / Hv is future work that needs separate Score/Value GEMM
  // shape paths.
  int64_t H = head_size;
  if (H <= 0) {
    if (query_hidden % num_heads != 0) {
      fprintf(stderr,
              "[multi_head_attention] ERROR: query_hidden (%lld) not "
              "divisible by num_heads (%lld); cannot derive head_size\n",
              (long long)query_hidden, (long long)num_heads);
      return -1;
    }
    H = query_hidden / num_heads;
  }
  // v_hidden==0 sentinel means "value matches Q"; otherwise require equal Hv.
  if (v_hidden > 0 && v_hidden != query_hidden) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: v_hidden (%lld) != query_hidden "
            "(%lld); different Hq and Hv is not yet implemented\n",
            (long long)v_hidden, (long long)query_hidden);
    return -1;
  }

  const int64_t B = batch_size;
  const int64_t N = num_heads;
  const int64_t Sq = seq_len_q;
  const int64_t Skv = seq_len_kv;
  const int64_t hidden = N * H;
  if (hidden != query_hidden) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: derived hidden (N*H=%lld*%lld="
            "%lld) != query_hidden (%lld)\n",
            (long long)N, (long long)H, (long long)hidden,
            (long long)query_hidden);
    return -1;
  }

  // ONNX spec default: scale = 1/sqrt(head_size) when not supplied. Compiler
  // lowers a missing attribute to 0.0f (sentinel meaning "auto-compute at
  // runtime", same convention as GQA).
  if (scale == 0.0f) {
    scale = 1.0f / sqrtf(static_cast<float>(H));
  }

  // ---- Runtime resources ----
  hipStream_t stream =
      reinterpret_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream) {
    fprintf(stderr, "[multi_head_attention] ERROR: failed to get HIP stream\n");
    return -1;
  }
  hipblasLtHandle_t ltHandle = reinterpret_cast<hipblasLtHandle_t>(
      hipdnn_ep_state_get_hipblas_handle(state));
  if (!ltHandle) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: failed to get hipBLASLt handle\n");
    return -1;
  }

  // ---- Fused flash-attention prefill fast path -------------------------
  // Non-causal self-attention prefill (the Qwen VLM vision encoder: B=1,
  // N=16, Sq=Skv=7296, H=72). The decomposed path below materializes the fp32
  // score matrix S[B,N,Sq,Skv] (~3.4 GB here) in DRAM and is entirely
  // HBM-bandwidth bound. hip_mha_flash_prefill keeps the running softmax state
  // in registers and never writes S: only K/V need transposing to BNSD; Q and
  // O stay in their native BSND layout (no Q/O transpose). Always on for
  // eligible shapes; the decomposed path below still handles the ineligible
  // cases (decode Sq==1, causal/unidirectional, head-dim>256).
  if (Sq > 1 && unidirectional != 1 && (((H + 15) / 16) * 16) <= 256) {
    const size_t fa_align = 64;
    auto fa_align_up = [&](size_t v) {
      return (v + fa_align - 1) & ~(fa_align - 1);
    };
    const size_t sz_kv = fa_align_up((size_t)B * N * Skv * H * 2);
    const size_t fa_ws = sz_kv * 2;
    if (hipdnn_ep_state_ensure_workspace(state, fa_ws) != 0) {
      fprintf(stderr,
              "[multi_head_attention] ERROR: flash path failed to ensure "
              "workspace of %zu bytes\n",
              fa_ws);
      return -1;
    }
    char *fa = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
    if (!fa) {
      fprintf(
          stderr,
          "[multi_head_attention] ERROR: flash workspace pointer is null\n");
      return -1;
    }
    void *d_Kbnsh = fa;
    void *d_Vbnsh = fa + sz_kv;

    // Transpose K/V from BSND [B,Skv,N,H] to BNSD [B,N,Skv,H] (fp16).
    if (hip_gqa_transpose_mid_dims(stream, key, d_Kbnsh, static_cast<int>(B),
                                   static_cast<int>(Skv), static_cast<int>(N),
                                   static_cast<int>(H), 2) != 0 ||
        hip_gqa_transpose_mid_dims(stream, value, d_Vbnsh, static_cast<int>(B),
                                   static_cast<int>(Skv), static_cast<int>(N),
                                   static_cast<int>(H), 2) != 0) {
      fprintf(
          stderr,
          "[multi_head_attention] ERROR: flash path K/V transpose failed\n");
      return -1;
    }

    // Q is BSND [B,Sq,N,H] == the kernel's expected [B,sq,N,d]; O written in
    // place (BSND). K/V cache is the just-transposed BNSD with max_seq = Skv.
    int rc = hip_mha_flash_prefill(
        stream, query, d_Kbnsh, d_Vbnsh, output, static_cast<int>(B),
        static_cast<int>(N), static_cast<int>(Sq), static_cast<int>(Skv),
        static_cast<int>(H), static_cast<int>(Skv), scale);
    if (rc != 0) {
      fprintf(stderr,
              "[multi_head_attention] ERROR: hip_mha_flash_prefill failed "
              "(rc=%d) for B=%lld N=%lld Sq=%lld Skv=%lld H=%lld\n",
              rc, (long long)B, (long long)N, (long long)Sq, (long long)Skv,
              (long long)H);
      return -1;
    }
    RUNTIME_DEBUG_LOG(
        "[multi_head_attention] flash success: B=%lld N=%lld Sq=%lld "
        "Skv=%lld H=%lld scale=%.6f\n",
        (long long)B, (long long)N, (long long)Sq, (long long)Skv, (long long)H,
        (double)scale);
    return 0;
  }

  // ---- Scratch layout (all backed by the shared workspace buffer) ----
  // 64-byte aligned sub-buffers in a single contiguous allocation:
  //   [ Q_BNSH (fp16, B*N*Sq *H)        -- skipped when S_q==1
  //   | K_BNSH (fp16, B*N*Skv*H)        -- skipped when S_kv==1
  //   | V_BNSH (fp16, B*N*Skv*H)        -- skipped when S_kv==1
  //   | S_f32  (fp32, B*N*Sq *Skv)
  //   | P_f16  (fp16, B*N*Sq *Skv)
  //   | O_BNSH (fp16, B*N*Sq *H)        -- skipped when S_q==1
  //   | gemm_workspace (kMaxWorkspaceBytes from hipBLASLt heuristic)
  //   ]
  // When a transpose is skipped, the source memref is BSHD [B,1,N,H] which is
  // bit-identical to BNSH [B,N,1,H] (different logical reshape, same memory
  // order), so we can point straight at the input buffer instead of copying.
  const size_t align = 64;
  auto align_up = [&](size_t v) { return (v + align - 1) & ~(align - 1); };
  const bool need_q_trans = (Sq > 1);
  const bool need_kv_trans = (Skv > 1);
  const bool need_o_trans = (Sq > 1);
  const size_t sz_q_bnsh = need_q_trans ? align_up(B * N * Sq * H * 2) : 0;
  const size_t sz_k_bnsh = need_kv_trans ? align_up(B * N * Skv * H * 2) : 0;
  const size_t sz_v_bnsh = need_kv_trans ? align_up(B * N * Skv * H * 2) : 0;
  const size_t sz_s_f32 = align_up(B * N * Sq * Skv * 4);
  const size_t sz_p_f16 = align_up(B * N * Sq * Skv * 2);
  const size_t sz_o_bnsh = need_o_trans ? align_up(B * N * Sq * H * 2) : 0;
  const size_t sz_gemm_ws = align_up(kMaxWorkspaceBytes);
  const size_t total_ws = sz_q_bnsh + sz_k_bnsh + sz_v_bnsh + sz_s_f32 +
                          sz_p_f16 + sz_o_bnsh + sz_gemm_ws;

  if (hipdnn_ep_state_ensure_workspace(state, total_ws) != 0) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: failed to ensure workspace of "
            "%zu bytes\n",
            total_ws);
    return -1;
  }
  char *ws = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
  if (!ws) {
    fprintf(stderr,
            "[multi_head_attention] ERROR: workspace pointer is null after "
            "ensure_workspace\n");
    return -1;
  }
  size_t off = 0;
  void *d_Qbnsh = nullptr;
  void *d_Kbnsh = nullptr;
  void *d_Vbnsh = nullptr;
  void *d_O_bnsh = nullptr;
  if (sz_q_bnsh) {
    d_Qbnsh = ws + off;
    off += sz_q_bnsh;
  } else {
    d_Qbnsh = query;
  }
  if (sz_k_bnsh) {
    d_Kbnsh = ws + off;
    off += sz_k_bnsh;
  } else {
    d_Kbnsh = key;
  }
  if (sz_v_bnsh) {
    d_Vbnsh = ws + off;
    off += sz_v_bnsh;
  } else {
    d_Vbnsh = value;
  }
  void *d_S_f32 = ws + off;
  off += sz_s_f32;
  void *d_P_f16 = ws + off;
  off += sz_p_f16;
  if (sz_o_bnsh) {
    d_O_bnsh = ws + off;
    off += sz_o_bnsh;
  } else {
    d_O_bnsh = output;
  }
  void *gemm_ws = ws + off;
  const size_t gemm_ws_bytes = sz_gemm_ws;

  RUNTIME_DEBUG_LOG(
      "[multi_head_attention] workspace=%zu bytes (q=%zu k=%zu v=%zu "
      "s=%zu p=%zu o=%zu gemm=%zu); need_q_trans=%d need_kv_trans=%d "
      "need_o_trans=%d\n",
      total_ws, sz_q_bnsh, sz_k_bnsh, sz_v_bnsh, sz_s_f32, sz_p_f16, sz_o_bnsh,
      sz_gemm_ws, (int)need_q_trans, (int)need_kv_trans, (int)need_o_trans);

  int result = 0;

  // ---- Step 1-3: Transpose Q / K / V from BSHD [B,S,N,H] to BNSD [B,N,S,H]
  // hip_gqa_transpose_mid_dims swaps the middle two dims: input is treated
  // as [batch_size, dim1, dim2, D]. For Q with dim1=Sq, dim2=N this produces
  // [B, N, Sq, H]; same shape with Skv for K and V.
  if (need_q_trans) {
    // MHA path is fp16-only; element_size_bytes=2.
    HIP_CHECK(hip_gqa_transpose_mid_dims(
        stream, query, d_Qbnsh, static_cast<int>(B), static_cast<int>(Sq),
        static_cast<int>(N), static_cast<int>(H), /*element_size_bytes=*/2));
  }
  if (need_kv_trans) {
    HIP_CHECK(hip_gqa_transpose_mid_dims(
        stream, key, d_Kbnsh, static_cast<int>(B), static_cast<int>(Skv),
        static_cast<int>(N), static_cast<int>(H), /*element_size_bytes=*/2));
    HIP_CHECK(hip_gqa_transpose_mid_dims(
        stream, value, d_Vbnsh, static_cast<int>(B), static_cast<int>(Skv),
        static_cast<int>(N), static_cast<int>(H), /*element_size_bytes=*/2));
  }

  // ---- Step 4: Score GEMM: S = Q @ K^T * scale, fp16->fp32 ---------------
  // Following GQA's score-GEMM convention (hipBLASLt is column-major):
  //   A = K, transA=true; B = Q, transA=false
  //   m = Skv, n = Sq, k = H
  //   per-batch (b*N + n) strides: A=Skv*H, B=Sq*H, C=Skv*Sq
  // Produces fp32 S of shape [B*N, Sq, Skv] with row-stride Skv per query.
  {
    MhaGemmKey scoreKey = {};
    scoreKey.m = Skv;
    scoreKey.n = Sq;
    scoreKey.k = H;
    scoreKey.batch = B * N;
    scoreKey.transA = true;
    scoreKey.outputFp32 = true;
    scoreKey.strideA = Skv * H;
    scoreKey.strideB = Sq * H;
    scoreKey.strideC = Skv * Sq;

    const MhaGemmCacheEntry *scoreState =
        queryOrCreateMhaGemm(state, ltHandle, scoreKey, op_state_slot);
    if (!scoreState) {
      fprintf(stderr,
              "[multi_head_attention] ERROR: failed to build Score GEMM "
              "state for B*N=%lld Skv=%lld Sq=%lld H=%lld\n",
              (long long)(B * N), (long long)Skv, (long long)Sq, (long long)H);
      result = -1;
      goto cleanup;
    }

    float scoreAlpha = scale;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t sAlgo = scoreState->algo;
    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, scoreState->desc, &scoreAlpha, d_Kbnsh, scoreState->layA,
        d_Qbnsh, scoreState->layB, &beta, d_S_f32, scoreState->layC, d_S_f32,
        scoreState->layD, &sAlgo, gemm_ws, gemm_ws_bytes, stream));
  }

  // ---- Step 5: Optional causal mask (fp32) ------------------------------
  // S has logical shape [B*N, Sq, Skv]; head_stride = Sq*Skv.
  if (unidirectional == 1) {
    HIP_CHECK(hip_gqa_causal_mask_f32(
        stream, d_S_f32, static_cast<int>(B * N), static_cast<int>(Skv),
        static_cast<int>(Sq), static_cast<int>(Sq * Skv),
        /*past_len=*/0, /*local_window_size=*/0));
  }

  // ---- Step 6: Softmax (fp32 -> fp16) -----------------------------------
  // total_head_queries = B*N*Sq (one block per (head, query row)).
  HIP_CHECK(hip_gqa_softmax_f32_to_f16(
      stream, d_S_f32, d_P_f16, static_cast<int>(B * N * Sq),
      static_cast<int>(Skv), static_cast<int>(Sq),
      /*input_batch_stride=*/static_cast<int>(Sq * Skv),
      /*output_batch_stride=*/static_cast<int>(Sq * Skv),
      /*head_sink=*/nullptr, /*num_heads=*/static_cast<int>(N),
      /*use_smooth_softmax=*/0));

  // ---- Step 7: Value GEMM: O = P @ V, fp16 ------------------------------
  // hipBLASLt column-major: A = V (transA=false), B = P (transA=false)
  //   m = H, n = Sq, k = Skv
  //   per-batch (b*N + n) strides: A=Skv*H, B=Sq*Skv, C=Sq*H
  // Produces fp16 O of shape [B*N, Sq, H] (BNSH layout).
  {
    MhaGemmKey valueKey = {};
    valueKey.m = H;
    valueKey.n = Sq;
    valueKey.k = Skv;
    valueKey.batch = B * N;
    valueKey.transA = false;
    valueKey.outputFp32 = false;
    valueKey.strideA = Skv * H;
    valueKey.strideB = Sq * Skv;
    valueKey.strideC = Sq * H;

    const MhaGemmCacheEntry *valueState =
        queryOrCreateMhaGemm(state, ltHandle, valueKey, op_state_slot);
    if (!valueState) {
      fprintf(stderr,
              "[multi_head_attention] ERROR: failed to build Value GEMM "
              "state for B*N=%lld Sq=%lld Skv=%lld H=%lld\n",
              (long long)(B * N), (long long)Sq, (long long)Skv, (long long)H);
      result = -1;
      goto cleanup;
    }

    float valAlpha = 1.0f;
    float beta = 0.0f;
    hipblasLtMatmulAlgo_t vAlgo = valueState->algo;
    HIPBLAS_CHECK(hipblasLtMatmul(
        ltHandle, valueState->desc, &valAlpha, d_Vbnsh, valueState->layA,
        d_P_f16, valueState->layB, &beta, d_O_bnsh, valueState->layC, d_O_bnsh,
        valueState->layD, &vAlgo, gemm_ws, gemm_ws_bytes, stream));
  }

  // ---- Step 8: Transpose O from BNSH [B,N,Sq,H] back to BSHD [B,Sq,N,H]
  if (need_o_trans) {
    HIP_CHECK(hip_gqa_transpose_mid_dims(
        stream, d_O_bnsh, output, static_cast<int>(B), static_cast<int>(N),
        static_cast<int>(Sq), static_cast<int>(H), /*element_size_bytes=*/2));
  }

  RUNTIME_DEBUG_LOG(
      "[multi_head_attention] success: B=%lld N=%lld Sq=%lld Skv=%lld "
      "H=%lld scale=%.6f causal=%d\n",
      (long long)B, (long long)N, (long long)Sq, (long long)Skv, (long long)H,
      (double)scale, (int)(unidirectional == 1));

cleanup:
  return result;
}
