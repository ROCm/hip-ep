/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>

// =============================================================================
// Full-object cache: desc, layouts, algo, and workspace size are created once
// per unique problem shape and reused for the process lifetime.
// Modeled on gqa.cpp's queryOrCreateGemm pattern.
// =============================================================================

struct MatmulCacheKey {
  int64_t M, N, K, batch_count, elem_size;
  bool operator==(const MatmulCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && batch_count == o.batch_count &&
           elem_size == o.elem_size;
  }
};

struct MatmulCacheKeyHash {
  size_t operator()(const MatmulCacheKey &k) const {
    size_t h = std::hash<int64_t>{}(k.M);
    h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.batch_count) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.elem_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct MatmulCacheEntry {
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t layA, layB, layC;
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
};

static std::unordered_map<MatmulCacheKey, MatmulCacheEntry, MatmulCacheKeyHash>
    g_matmul_cache;

static const MatmulCacheEntry *
queryOrCreateMatmul(hipblasLtHandle_t handle, const MatmulCacheKey &key) {
  auto it = g_matmul_cache.find(key);
  if (it != g_matmul_cache.end())
    return &it->second;

  hipDataType dt = (key.elem_size == 2) ? HIP_R_16F : HIP_R_32F;
  int64_t M = key.M, N = key.N, K = key.K;

  MatmulCacheEntry entry{};

  if (hipblasLtMatmulDescCreate(&entry.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
      HIPBLAS_STATUS_SUCCESS)
    return nullptr;

  hipblasOperation_t opN = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(entry.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                  &opN, sizeof(opN));
  hipblasLtMatmulDescSetAttribute(entry.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                  &opN, sizeof(opN));

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
    hipblasLtMatrixLayoutSetAttribute(
        entry.layA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        entry.layA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA,
        sizeof(sA));
    hipblasLtMatrixLayoutSetAttribute(
        entry.layB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        entry.layB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB,
        sizeof(sB));
    hipblasLtMatrixLayoutSetAttribute(
        entry.layC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    hipblasLtMatrixLayoutSetAttribute(
        entry.layC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC,
        sizeof(sC));
  }

  hipblasLtMatmulPreference_t pref;
  hipblasLtMatmulPreferenceCreate(&pref);
  const size_t max_ws = 1ULL << 30; // 1 GB
  hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
      sizeof(max_ws));

  hipblasLtMatmulHeuristicResult_t heur;
  int returned = 0;
  hipblasLtMatmulAlgoGetHeuristic(handle, entry.desc, entry.layA, entry.layB,
                                  entry.layC, entry.layC, pref, 1, &heur,
                                  &returned);
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

  entry.algo = heur.algo;
  entry.workspace_size = heur.workspaceSize;
  auto [ins, _] = g_matmul_cache.emplace(key, entry);

  RUNTIME_DEBUG_LOG("[REAL] queryOrCreateMatmul: cached desc+layouts+algo for "
                    "M=%lld N=%lld K=%lld batch=%lld (ws=%zu)\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)key.batch_count, entry.workspace_size);

  return &ins->second;
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
  const MatmulCacheEntry *cached = queryOrCreateMatmul(handle, key);
  if (!cached) {
    fprintf(stderr, "wrap_hipblasLtMatmul: failed to create/find cached "
                    "descriptors for M=%lld N=%lld K=%lld batch=%lld\n",
            (long long)M, (long long)N, (long long)K,
            (long long)batch_count);
    return -1;
  }

  if (cached->workspace_size > 0) {
    if (hipdnn_ep_state_ensure_workspace(state, cached->workspace_size) != 0)
      return -1;
  }

  void *ws_ptr = hipdnn_ep_state_get_workspace(state);
  size_t ws_size = hipdnn_ep_state_get_workspace_size(state);

  float alpha = 1.0f;
  float beta = 0.0f;

  hipblasStatus_t st = hipblasLtMatmul(
      handle, cached->desc, &alpha, B,
      cached->layA,    // "A" = B (row->col trick)
      A, cached->layB, // "B" = A (row->col trick)
      &beta, output, cached->layC, output, cached->layC,
      const_cast<hipblasLtMatmulAlgo_t *>(&cached->algo), ws_ptr, ws_size,
      stream);

  if (st != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "wrap_hipblasLtMatmul: hipblasLtMatmul failed (%d)\n", st);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: completed successfully\n");
  return 0;
}
