/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- mla.cpp - hip.mla_attention runtime ------------------------------===//
//
// Causal attention over a latent (MLA) cache. A cache row is
// [c (L = kv_lora_rank), k_r (R = qk_rope_dim)]; head h sees
//   K_h = [c @ W_k,h^T, k_r]   V_h = c @ W_v,h^T
// with W_k [H * dn, L] and W_v [H * dv, L] the up-projections.
//
// Prefill (S > 1) expands K/V for all valid rows once (three batched GEMMs
// into a BNSD scratch) and runs the fused dn + R attention over that past.
// Decode (S == 1) moves the query into the latent space instead,
//   q'_h = [q_nope,h @ W_k,h, q_rope,h]                (L + R)
// attends over the cache itself as one KV head of width L + R (the decode
// kernel reads V from the K cache), and maps the result back,
//   out_h = o'_h[:L] @ W_v,h^T.
// Decode reads T x (L + R) cache elements instead of T x H x (dn + R + dv).
//
//===----------------------------------------------------------------------===//

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "cache_utils.h"
#include "gqa_fused.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstdio>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

// One up-projection as GEMM operands. Logically W [H * n, L]; `rows_n` means
// it is stored that way (row h * n + j at (h * n + j) * ld), otherwise it is
// the transposed MatMul weight [L, H * n] (element (j, l) at l * ld + j).
struct WeightView {
  const void *ptr = nullptr;
  int64_t ld = 0;
  int64_t head_stride = 0;
  bool rows_n = false;
};

struct GemmEntry {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t ws = 0;
  bool batched = true; // false: no algo for the batched form, loop per batch
};

using GemmKey = std::array<int64_t, 11>;

// Hashed and compared element by element: std::map's lexicographic
// std::less on an int64 array lowers on MSVC to the vectorized
// __std_mismatch_8 helper, which the Windows LLVM_IR JIT cannot resolve.
struct GemmKeyHash {
  size_t operator()(const GemmKey &k) const {
    size_t h = 0;
    for (int64_t v : k)
      hash_combine_val(h, v);
    return h;
  }
};

struct GemmKeyEq {
  bool operator()(const GemmKey &a, const GemmKey &b) const {
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i])
        return false;
    return true;
  }
};

struct MlaState : OpStateT<MlaState> {
  std::mutex mu;
  const void *k_src = nullptr, *v_src = nullptr;
  void *k_deq = nullptr, *v_deq = nullptr; // fp16 [H * n, L]
  void *identity = nullptr;                // fp16 [R, R]
  WeightView wk, wv;
  std::unordered_map<GemmKey, GemmEntry, GemmKeyHash, GemmKeyEq> gemms;

  ~MlaState() {
    for (auto &[key, e] : gemms) {
      if (e.a)
        hipblasLtMatrixLayoutDestroy(e.a);
      if (e.b)
        hipblasLtMatrixLayoutDestroy(e.b);
      if (e.c)
        hipblasLtMatrixLayoutDestroy(e.c);
      if (e.desc)
        hipblasLtMatmulDescDestroy(e.desc);
    }
    for (void *p : {k_deq, v_deq, identity})
      if (p)
        hipFree(p);
  }
};

constexpr size_t kGemmMaxWorkspace = 32ull << 20;

hipblasStatus_t set_batch(hipblasLtMatrixLayout_t l, int32_t count,
                          int64_t stride) {
  hipblasStatus_t s = hipblasLtMatrixLayoutSetAttribute(
      l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count));
  if (s != HIPBLAS_STATUS_SUCCESS)
    return s;
  return hipblasLtMatrixLayoutSetAttribute(
      l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
}

// Descriptors for row-major C[M, N] = A[M, K] @ op(B), fp16 in and out, fp32
// accumulation; op(B) is B [K, N] or, with trans_b, B^T for B [N, K].
// hipBLASLt is column-major, so this is C^T = op(B)^T A^T.
bool build_gemm(hipblasLtHandle_t handle, const GemmKey &k, int32_t batch,
                GemmEntry &e) {
  const int64_t M = k[0], N = k[1], K = k[2], lda = k[3], ldb = k[4],
                ldc = k[5], sA = k[6], sB = k[7], sC = k[8];
  const bool trans_b = k[9] != 0;
  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  const hipblasOperation_t opA = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t opB = HIPBLAS_OP_N;
  const size_t max_ws = kGemmMaxWorkspace;
  bool ok =
      hipblasLtMatmulDescCreate(&e.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmulDescSetAttribute(e.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &opA,
                                      sizeof(opA)) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatmulDescSetAttribute(e.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &opB,
                                      sizeof(opB)) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&e.a, HIP_R_16F, trans_b ? K : N,
                                  trans_b ? N : K,
                                  ldb) == HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&e.b, HIP_R_16F, K, M, lda) ==
          HIPBLAS_STATUS_SUCCESS &&
      hipblasLtMatrixLayoutCreate(&e.c, HIP_R_16F, N, M, ldc) ==
          HIPBLAS_STATUS_SUCCESS;
  if (ok && batch > 1)
    ok = set_batch(e.a, batch, sB) == HIPBLAS_STATUS_SUCCESS &&
         set_batch(e.b, batch, sA) == HIPBLAS_STATUS_SUCCESS &&
         set_batch(e.c, batch, sC) == HIPBLAS_STATUS_SUCCESS;
  ok = ok && hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
       hipblasLtMatmulPreferenceSetAttribute(
           pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
           sizeof(max_ws)) == HIPBLAS_STATUS_SUCCESS &&
       hipblasLtMatmulAlgoGetHeuristic(handle, e.desc, e.a, e.b, e.c, e.c, pref,
                                       1, &heur,
                                       &found) == HIPBLAS_STATUS_SUCCESS &&
       found > 0;
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (ok) {
    e.algo = heur.algo;
    e.ws = heur.workspaceSize;
  }
  return ok;
}

void destroy_gemm(GemmEntry &e) {
  for (hipblasLtMatrixLayout_t l : {e.a, e.b, e.c})
    if (l)
      hipblasLtMatrixLayoutDestroy(l);
  if (e.desc)
    hipblasLtMatmulDescDestroy(e.desc);
  e = GemmEntry{};
}

// Row-major strided-batched C_i = A_i @ op(B_i) over `batch` heads; a zero
// stride shares that operand across the batch.
int gemm(RuntimeState *state, MlaState *ms, hipStream_t stream, int64_t M,
         int64_t N, int64_t K, const void *A, int64_t lda, int64_t sA,
         const void *B, int64_t ldb, int64_t sB, bool trans_b, void *C,
         int64_t ldc, int64_t sC, int64_t batch) {
  auto handle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  if (!handle)
    return -1;
  const GemmKey key{M, N, K, lda, ldb, ldc, sA, sB, sC, trans_b, batch};
  auto it = ms->gemms.find(key);
  if (it == ms->gemms.end()) {
    GemmEntry e;
    if (!build_gemm(handle, key, static_cast<int32_t>(batch), e)) {
      destroy_gemm(e);
      e.batched = false;
      if (!build_gemm(handle, key, 1, e)) {
        destroy_gemm(e);
        fprintf(stderr,
                "mla_attention: no GEMM algorithm for M=%lld N=%lld K=%lld "
                "batch=%lld\n",
                (long long)M, (long long)N, (long long)K, (long long)batch);
        return -1;
      }
      e.batched = false;
    }
    it = ms->gemms.emplace(key, e).first;
  }
  const GemmEntry &e = it->second;
  void *ws = nullptr;
  if (e.ws) {
    if (hipdnn_ep_state_ensure_workspace(state, e.ws) != 0)
      return -1;
    ws = hipdnn_ep_state_get_workspace(state);
  }
  const float one = 1.0f, zero = 0.0f;
  const int64_t loops = e.batched ? 1 : batch;
  for (int64_t i = 0; i < loops; ++i) {
    const char *a = static_cast<const char *>(A) + 2 * i * sA;
    const char *b = static_cast<const char *>(B) + 2 * i * sB;
    char *c = static_cast<char *>(C) + 2 * i * sC;
    hipblasLtMatmulAlgo_t algo = e.algo;
    if (hipblasLtMatmul(handle, e.desc, &one, b, e.a, a, e.b, &zero, c, e.c, c,
                        e.c, &algo, ws, e.ws, stream) != HIPBLAS_STATUS_SUCCESS)
      return -1;
  }
  return 0;
}

// C[rows, n] per head = X[rows, L] @ W_h^T (X shared or strided per head).
int project_out(RuntimeState *state, MlaState *ms, hipStream_t stream,
                const WeightView &w, int64_t rows, int64_t n, int64_t L,
                const void *X, int64_t ldx, int64_t sX, void *C, int64_t ldc,
                int64_t sC, int64_t heads) {
  return gemm(state, ms, stream, rows, n, L, X, ldx, sX, w.ptr, w.ld,
              w.head_stride, /*trans_b=*/w.rows_n, C, ldc, sC, heads);
}

int ensure_scratch(RuntimeState *state, size_t bytes) {
  if (bytes <= state->mla_scratch_size)
    return 0;
  if (state->mla_scratch &&
      hipdnn_ep_state_device_free(state, state->mla_scratch) != hipSuccess)
    return -1;
  state->mla_scratch = nullptr;
  state->mla_scratch_size = 0;
  constexpr size_t kGrain = 64ull << 20;
  const size_t size = (bytes + kGrain - 1) / kGrain * kGrain;
  if (hipdnn_ep_state_device_alloc(state, &state->mla_scratch, size) !=
      hipSuccess) {
    fprintf(stderr, "mla_attention: scratch alloc of %zu bytes failed\n", size);
    return -1;
  }
  state->mla_scratch_size = size;
  return 0;
}

// Valid cache rows for prefill, which needs them on the host. One length
// for all batches, as the fused prefill requires.
int64_t valid_rows(RuntimeState *state, hipStream_t stream,
                   const void *kv_seqlen, const void *seqlens, int64_t B) {
  const int32_t pre = gqa_read_seqlens_k(state, stream, seqlens, B);
  if (pre >= 0)
    return static_cast<int64_t>(pre) + 1;
  if (B == 1)
    return -1;
  std::vector<int64_t> lengths(static_cast<size_t>(B));
  if (hipMemcpyAsync(lengths.data(), kv_seqlen, lengths.size() * 8,
                     hipMemcpyDeviceToHost, stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess)
    return -1;
  for (int64_t b = 1; b < B; ++b) {
    if (lengths[static_cast<size_t>(b)] != lengths[0]) {
      fprintf(stderr,
              "wrap_mla_attention: prefill needs one kv_seqlen for all "
              "batches (batch %lld has %lld, batch 0 has %lld)\n",
              (long long)b, (long long)lengths[static_cast<size_t>(b)],
              (long long)lengths[0]);
      return -1;
    }
  }
  return lengths[0];
}

int prepare_weights(MlaState *ms, hipStream_t stream, const void *k_b,
                    const void *v_b, const void *k_s, const void *v_s,
                    int64_t H, int64_t L, int64_t dn, int64_t R, int64_t dv,
                    int64_t block) {
  if (ms->k_src == k_b && ms->v_src == v_b)
    return 0;
  if (block > 0) {
    for (auto [src, sc, dst, n] : {std::make_tuple(k_b, k_s, &ms->k_deq, dn),
                                   std::make_tuple(v_b, v_s, &ms->v_deq, dv)}) {
      if (*dst)
        hipFree(*dst);
      *dst = nullptr;
      if (hipMalloc(dst, static_cast<size_t>(H * n * L) * 2) != hipSuccess)
        return -1;
      hip_matmul_nbits_dequant_b_fp16(
          stream, src, sc, nullptr, *dst, static_cast<int>(H * n),
          static_cast<int>(L), static_cast<int>(block));
    }
    ms->wk = {ms->k_deq, L, dn * L, true};
    ms->wv = {ms->v_deq, L, dv * L, true};
  } else {
    ms->wk = {k_b, H * dn, dn, false};
    ms->wv = {v_b, H * dv, dv, false};
  }
  if (!ms->identity) {
    std::vector<uint16_t> eye(static_cast<size_t>(R * R), 0);
    for (int64_t i = 0; i < R; ++i)
      eye[static_cast<size_t>(i * R + i)] = 0x3C00; // fp16 1.0
    if (hipMalloc(&ms->identity, eye.size() * 2) != hipSuccess ||
        hipMemcpy(ms->identity, eye.data(), eye.size() * 2,
                  hipMemcpyHostToDevice) != hipSuccess)
      return -1;
  }
  ms->k_src = k_b;
  ms->v_src = v_b;
  return 0;
}

} // namespace

extern "C" int8_t
hipdnn_ep_op_state_construct_mla_attention(RuntimeState *state, int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, MlaState::create().release());
  return 0;
}

int wrap_mla_attention(RuntimeState *state, int op_state_slot, void *query,
                       void *cache, void *k_b, void *v_b, void *k_b_scales,
                       void *v_b_scales, void *kv_seqlen, void *output,
                       int64_t num_heads, int64_t kv_lora_rank,
                       int64_t qk_nope_dim, int64_t qk_rope_dim, int64_t v_dim,
                       float scale, int64_t block_size, int64_t batch,
                       int64_t seq_len_q, int64_t cache_seq) {
  OP_PROFILE(
      "mla",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "b=%lld,sq=%lld,t=%lld", (long long)batch,
                 (long long)seq_len_q, (long long)cache_seq);
        return std::string(b);
      },
      state);
  const int64_t H = num_heads, L = kv_lora_rank, dn = qk_nope_dim,
                R = qk_rope_dim, dv = v_dim, B = batch, S = seq_len_q,
                T = cache_seq;
  MlaState *ms = state ? MlaState::get_op_state(state, op_state_slot) : nullptr;
  if (!ms || !query || !cache || !k_b || !v_b || !output ||
      (block_size > 0 && (!k_b_scales || !v_b_scales))) {
    fprintf(stderr, "wrap_mla_attention: null state or argument\n");
    return -1;
  }
  if (S <= 0 || B <= 0)
    return 0;
  if (S > T) {
    fprintf(stderr, "wrap_mla_attention: %lld queries over %lld cache rows\n",
            (long long)S, (long long)T);
    return -1;
  }
  auto stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (!stream)
    return -1;
  std::lock_guard<std::mutex> lock(ms->mu);
  if (prepare_weights(ms, stream, k_b, v_b, k_b_scales, v_b_scales, H, L, dn, R,
                      dv, block_size) != 0) {
    fprintf(stderr, "wrap_mla_attention: weight preparation failed\n");
    return -1;
  }
  const int64_t dq = dn + R, dl = L + R;
  const char *cache_c = static_cast<const char *>(cache);
  const void *seqlens = nullptr;
  if (kv_seqlen) {
    seqlens = gqa_seqlens_from_lengths(state, stream, kv_seqlen, B);
    if (!seqlens)
      return -1;
  }

  if (S == 1) {
    // Decode: q' [B, H, L + R] and o' [B, H, L + R] in scratch.
    const size_t part = static_cast<size_t>(B * H * dl) * 2;
    if (ensure_scratch(state, 2 * part) != 0)
      return -1;
    char *q_lat = static_cast<char *>(state->mla_scratch);
    char *o_lat = q_lat + part;
    // q_nope,h [B, dn] @ W_k,h [dn, L]: the transposed projection.
    const WeightView &w = ms->wk;
    if (gemm(state, ms, stream, B, L, dn, query, H * dq, dq, w.ptr, w.ld,
             w.head_stride, /*trans_b=*/!w.rows_n, q_lat, H * dl, dl, H) != 0)
      return -1;
    if (hipMemcpy2DAsync(q_lat + 2 * L, 2 * dl,
                         static_cast<const char *>(query) + 2 * dn, 2 * dq,
                         2 * R, B * H, hipMemcpyDeviceToDevice,
                         stream) != hipSuccess)
      return -1;
    if (gqa_fused_attend_past(state, stream, q_lat, cache, cache, seqlens,
                              o_lat, B, 1, T, H, 1, dl, scale) != 0)
      return -1;
    return project_out(state, ms, stream, ms->wv, B, dv, L, o_lat, H * dl, dl,
                       output, H * dv, dv, H);
  }

  // Prefill: K [B, H, T, dn + R] and V [B, H, T, dv] in scratch.
  if (dv != dq) {
    fprintf(stderr,
            "wrap_mla_attention: prefill needs v_dim == qk_nope_dim + "
            "qk_rope_dim (got %lld vs %lld)\n",
            (long long)dv, (long long)dq);
    return -1;
  }
  // Only the first Tv rows of a cache buffer are expanded.
  const int64_t Tv =
      seqlens ? valid_rows(state, stream, kv_seqlen, seqlens, B) : T;
  if (Tv < S || Tv > T) {
    fprintf(stderr,
            "wrap_mla_attention: %lld valid cache rows for %lld queries in a "
            "%lld-row cache\n",
            (long long)Tv, (long long)S, (long long)T);
    return -1;
  }
  const size_t k_bytes = static_cast<size_t>(B * H * Tv * dq) * 2;
  if (ensure_scratch(state, 2 * k_bytes) != 0)
    return -1;
  char *K = static_cast<char *>(state->mla_scratch);
  char *V = K + k_bytes;
  for (int64_t b = 0; b < B; ++b) {
    const char *c_b = cache_c + 2 * b * T * dl;
    char *K_b = K + 2 * b * H * Tv * dq;
    char *V_b = V + 2 * b * H * Tv * dv;
    if (project_out(state, ms, stream, ms->wk, Tv, dn, L, c_b, dl, 0, K_b, dq,
                    Tv * dq, H) != 0 ||
        gemm(state, ms, stream, Tv, R, R, c_b + 2 * L, dl, 0, ms->identity, R,
             0, false, K_b + 2 * dn, dq, Tv * dq, H) != 0 ||
        project_out(state, ms, stream, ms->wv, Tv, dv, L, c_b, dl, 0, V_b, dv,
                    Tv * dv, H) != 0)
      return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] mla_attention prefill: B=%lld S=%lld T=%lld "
                    "Tv=%lld\n",
                    (long long)B, (long long)S, (long long)T, (long long)Tv);
  return gqa_fused_attend_past(state, stream, query, K, V, nullptr, output, B,
                               S, Tv, H, H, dq, scale);
}
