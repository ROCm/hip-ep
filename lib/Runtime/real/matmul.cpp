/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "cache_utils.h"
#include "runtime_types.h"

#include "hip_custom_kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>

// =============================================================================
// Per-shape descriptor cache with multi-algorithm auto-tune.
//
// On the first call for each unique (M, N, K, batch, elem_size) shape we
// create descriptors, request up to MAX_ALGO_CANDIDATES algorithms from the
// heuristic, and store them. On the first actual matmul call (when real GPU
// pointers are available), we benchmark each candidate and cache the winner.
// Subsequent calls reuse the tuned algorithm with zero overhead.
//
// Auto-tune is enabled by default. Set HIPDNN_EP_AUTOTUNE=0 to disable.
// =============================================================================

static constexpr int MAX_ALGO_CANDIDATES = 60;
static constexpr int AUTOTUNE_TIMING_ITERS = 3;

static bool autotune_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *env = getenv("HIPDNN_EP_AUTOTUNE");
    enabled = (env && strcmp(env, "0") != 0) ? 1 : 0;
  }
  return enabled != 0;
}

struct MatmulCacheKey {
  int64_t M, N, K, batch_count, elem_size;
  bool operator==(const MatmulCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && batch_count == o.batch_count &&
           elem_size == o.elem_size;
  }
};

struct MatmulCacheKeyHash {
  size_t operator()(const MatmulCacheKey &k) const {
    size_t h = 0;
    hash_combine_val(h, k.M);
    hash_combine_val(h, k.N);
    hash_combine_val(h, k.K);
    hash_combine_val(h, k.batch_count);
    hash_combine_val(h, k.elem_size);
    return h;
  }
};

struct MatmulCacheEntry {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t layA, layB, layC;
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
  bool tuned;
  int num_candidates;
  size_t max_candidate_workspace;
  hipblasLtMatmulHeuristicResult_t candidates[MAX_ALGO_CANDIDATES];
};

static std::unordered_map<MatmulCacheKey, MatmulCacheEntry, MatmulCacheKeyHash>
    g_matmul_cache;

static MatmulCacheEntry *queryOrCreateMatmul(hipblasLtHandle_t handle,
                                             const MatmulCacheKey &key) {
  auto it = g_matmul_cache.find(key);
  if (it != g_matmul_cache.end())
    return &it->second;

  hipDataType dt = (key.elem_size == 2) ? HIP_R_16F : HIP_R_32F;
  int64_t M = key.M, N = key.N, K = key.K;

  MatmulCacheEntry entry{};
  entry.tuned = false;
  entry.num_candidates = 0;
  entry.max_candidate_workspace = 0;

  if (hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS)
    return nullptr;

  hipblasOperation_t opN = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescSetAttribute(entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &opN, sizeof(opN)) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                      &opN, sizeof(opN)) !=
          HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }

  // Row-major -> col-major trick: BLAS sees m=N, k=K, n=M with ld = first dim
  if (hipblasLtMatrixLayoutCreate(&entry.layA, dt, N, K, N) !=
      HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }
  if (hipblasLtMatrixLayoutCreate(&entry.layB, dt, K, M, K) !=
      HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(entry.layA);
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }
  if (hipblasLtMatrixLayoutCreate(&entry.layC, dt, N, M, N) !=
      HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(entry.layB);
    hipblasLtMatrixLayoutDestroy(entry.layA);
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }

  if (key.batch_count > 1) {
    int64_t bc = key.batch_count;
    int64_t sA = K * N, sB = M * K, sC = M * N;
    if (hipblasLtMatrixLayoutSetAttribute(
            entry.layA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
            sizeof(bc)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutSetAttribute(
            entry.layA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA,
            sizeof(sA)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutSetAttribute(
            entry.layB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
            sizeof(bc)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutSetAttribute(
            entry.layB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB,
            sizeof(sB)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutSetAttribute(
            entry.layC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
            sizeof(bc)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutSetAttribute(
            entry.layC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC,
            sizeof(sC)) != HIPBLAS_STATUS_SUCCESS) {
      hipblasLtMatrixLayoutDestroy(entry.layC);
      hipblasLtMatrixLayoutDestroy(entry.layB);
      hipblasLtMatrixLayoutDestroy(entry.layA);
      hipblasLtMatmulDescDestroy(entry.desc);
      return nullptr;
    }
  }

  hipblasLtMatmulPreference_t pref;
  if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(entry.layC);
    hipblasLtMatrixLayoutDestroy(entry.layB);
    hipblasLtMatrixLayoutDestroy(entry.layA);
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }
  const size_t max_ws = kMaxWorkspaceBytes;
  hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

  bool do_autotune = autotune_enabled();
  int request_count = do_autotune ? MAX_ALGO_CANDIDATES : 1;

  int returned = 0;
  hipblasLtMatmulAlgoGetHeuristic(handle, entry.desc, entry.layA, entry.layB,
                                  entry.layC, entry.layC, pref, request_count,
                                  entry.candidates, &returned);
  hipblasLtMatmulPreferenceDestroy(pref);

  if (returned == 0) {
    fprintf(stderr,
            "queryOrCreateMatmul: no algo for M=%lld N=%lld K=%lld "
            "batch=%lld\n",
            (long long)M, (long long)N, (long long)K,
            (long long)key.batch_count);
    hipblasLtMatrixLayoutDestroy(entry.layC);
    hipblasLtMatrixLayoutDestroy(entry.layB);
    hipblasLtMatrixLayoutDestroy(entry.layA);
    hipblasLtMatmulDescDestroy(entry.desc);
    return nullptr;
  }

  entry.num_candidates = returned;
  entry.algo = entry.candidates[0].algo;
  entry.workspace_size = entry.candidates[0].workspaceSize;

  if (do_autotune) {
    for (int i = 0; i < returned; i++)
      entry.max_candidate_workspace = std::max(
          entry.max_candidate_workspace, entry.candidates[i].workspaceSize);
    entry.tuned = (returned <= 1);
  } else {
    entry.tuned = true;
  }

  auto [ins, _] = g_matmul_cache.emplace(key, entry);

  RUNTIME_DEBUG_LOG("[MATMUL] cached M=%lld N=%lld K=%lld batch=%lld: "
                    "%d algo(s), autotune=%s\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)key.batch_count, returned,
                    entry.tuned ? "skipped" : "pending");

  return &ins->second;
}

// =============================================================================
// Auto-tune: benchmark all candidate algorithms and select the fastest.
// Called once per shape on the first matmul invocation with real GPU pointers.
// =============================================================================

static void autotuneMatmul(hipblasLtHandle_t handle, hipStream_t stream,
                           MatmulCacheEntry *entry, const void *blas_A,
                           const void *blas_B, void *blas_C, void *ws_ptr,
                           size_t ws_size, const MatmulCacheKey &key) {
  float alpha = 1.0f, beta = 0.0f;
  hipEvent_t ev_start = nullptr, ev_stop = nullptr;
  if (hipEventCreate(&ev_start) != hipSuccess ||
      hipEventCreate(&ev_stop) != hipSuccess) {
    if (ev_start)
      hipEventDestroy(ev_start);
    fprintf(stderr, "[AUTOTUNE] WARNING: hipEventCreate failed, skipping\n");
    entry->tuned = true;
    return;
  }

  float best_ms = std::numeric_limits<float>::max();
  int best_idx = 0;
  int tested = 0;

  for (int i = 0; i < entry->num_candidates; i++) {
    auto &cand = entry->candidates[i];

    if (cand.workspaceSize > ws_size)
      continue;

    // Warm-up
    hipblasStatus_t st =
        hipblasLtMatmul(handle, entry->desc, &alpha, blas_A, entry->layA,
                        blas_B, entry->layB, &beta, blas_C, entry->layC, blas_C,
                        entry->layC, &cand.algo, ws_ptr, ws_size, stream);
    if (st != HIPBLAS_STATUS_SUCCESS)
      continue;

    // Timed iterations
    hipEventRecord(ev_start, stream);
    for (int t = 0; t < AUTOTUNE_TIMING_ITERS; t++) {
      hipblasLtMatmul(handle, entry->desc, &alpha, blas_A, entry->layA, blas_B,
                      entry->layB, &beta, blas_C, entry->layC, blas_C,
                      entry->layC, &cand.algo, ws_ptr, ws_size, stream);
    }
    hipEventRecord(ev_stop, stream);
    hipEventSynchronize(ev_stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, ev_start, ev_stop);
    tested++;

    if (ms < best_ms) {
      best_ms = ms;
      best_idx = i;
    }
  }

  hipEventDestroy(ev_start);
  hipEventDestroy(ev_stop);

  if (tested == 0) {
    fprintf(stderr,
            "[AUTOTUNE] WARNING: M=%lld N=%lld K=%lld batch=%lld: "
            "0/%d candidates passed, keeping heuristic #0\n",
            (long long)key.M, (long long)key.N, (long long)key.K,
            (long long)key.batch_count, entry->num_candidates);
    return;
  }

  entry->algo = entry->candidates[best_idx].algo;
  entry->workspace_size = entry->candidates[best_idx].workspaceSize;
  entry->tuned = true;

  RUNTIME_DEBUG_LOG("[AUTOTUNE] M=%lld N=%lld K=%lld batch=%lld: "
                    "tested %d/%d algos, best=#%d (%.3f ms/%d iters)\n",
                    (long long)key.M, (long long)key.N, (long long)key.K,
                    (long long)key.batch_count, tested, entry->num_candidates,
                    best_idx, best_ms, AUTOTUNE_TIMING_ITERS);
}

// =============================================================================
// Batched MatMul via hipBLASLt
// =============================================================================
//
// ONNX MatMul semantics: output = A @ B (row-major)
//   A: [batch_count x M x K]
//   B: [batch_count x K x N]  (or [K x N] with broadcast)
//   output: [batch_count x M x N]
//
// hipBLASLt expects column-major storage. To avoid explicit transposition we
// use the identity:  C_row = (B^T * A^T)^T
//
// In column-major terms, row-major matrix A(M,K) looks like A^T(K,M).
// So we tell hipBLASLt:
//   m = N, n = M, k = K
//   A_ptr = B (column-major view is B^T: N rows, K cols, ld = N)
//   B_ptr = A (column-major view is A^T: K rows, M cols, ld = K)
//   C_ptr = output (column-major view is C^T: N rows, M cols, ld = N)
//
// Both fp16 and fp32 use HIPBLAS_COMPUTE_32F for accumulation precision.
// =============================================================================

int wrap_hipblasLtMatmul(RuntimeState *state, const void *A, const void *B,
                         void *output, int64_t M, int64_t N, int64_t K,
                         int64_t batch_count, int64_t elem_size) {
  if (!state || !A || !B || !output) {
    fprintf(stderr, "Invalid arguments to wrap_hipblasLtMatmul\n");
    return -1;
  }

  hipblasLtHandle_t handle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  if (!handle || !stream) {
    fprintf(stderr, "wrap_hipblasLtMatmul: null handle or stream\n");
    return -1;
  }

  if (elem_size != 2 && elem_size != 4) {
    fprintf(stderr, "wrap_hipblasLtMatmul: unsupported elem_size %lld\n",
            (long long)elem_size);
    return -1;
  }

  const char *type_name = (elem_size == 2) ? "f16" : "f32";
  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, elem_size=%lld (%s), "
                    "total_bytes=%lld\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)elem_size, type_name,
                    (long long)(batch_count * M * N * elem_size));

  MatmulCacheKey key{M, N, K, batch_count, elem_size};
  MatmulCacheEntry *cached = queryOrCreateMatmul(handle, key);
  if (!cached) {
    fprintf(stderr,
            "wrap_hipblasLtMatmul: failed to create/find cached "
            "descriptors for M=%lld N=%lld K=%lld batch=%lld\n",
            (long long)M, (long long)N, (long long)K, (long long)batch_count);
    return -1;
  }

  // Ensure workspace is large enough for auto-tune candidates (if pending)
  // or the selected algorithm.
  size_t needed_ws =
      cached->tuned ? cached->workspace_size : cached->max_candidate_workspace;
  if (needed_ws > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, needed_ws) != 0)
      return -1;
  }

  void *ws_ptr = hipdnn_ep_state_get_workspace(state);
  size_t ws_size = hipdnn_ep_state_get_workspace_size(state);

  // Auto-tune on first call: benchmark all candidates with real GPU data
  if (!cached->tuned) {
    autotuneMatmul(handle, stream, cached, B, A, output, ws_ptr, ws_size, key);
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  hipblasStatus_t st =
      hipblasLtMatmul(handle, cached->desc, &alpha, B,
                      cached->layA,    // "A" = B (row->col trick)
                      A, cached->layB, // "B" = A (row->col trick)
                      &beta, output, cached->layC, output, cached->layC,
                      const_cast<hipblasLtMatmulAlgo_t *>(&cached->algo),
                      ws_ptr, ws_size, stream);

  if (st != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "wrap_hipblasLtMatmul: hipblasLtMatmul failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: completed successfully\n");
  return 0;
}
