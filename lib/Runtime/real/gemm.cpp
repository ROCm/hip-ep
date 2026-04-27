/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../operator_profile.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)
#define MIOPEN_CHECK(cmd) MIOPEN_CHECK_GOTO(cmd, cleanup)

// Type codes — must match the lowering in HipToLLVM.cpp GemmOpLowering
static constexpr int64_t kTypeFloat16 = 0;
static constexpr int64_t kTypeFloat32 = 1;
static constexpr int64_t kTypeFloat64 = 2;
static constexpr int64_t kTypeBFloat16 = 3;

static bool resolveGemmTypes(int64_t typeCode, hipDataType &dataType,
                             hipblasComputeType_t &computeType,
                             hipDataType &scaleType) {
  switch (typeCode) {
  case kTypeFloat16:
    dataType = HIP_R_16F;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  case kTypeFloat32:
    dataType = HIP_R_32F;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  case kTypeFloat64:
    dataType = HIP_R_64F;
    computeType = HIPBLAS_COMPUTE_64F;
    scaleType = HIP_R_64F;
    return true;
  case kTypeBFloat16:
    dataType = HIP_R_16BF;
    computeType = HIPBLAS_COMPUTE_32F;
    scaleType = HIP_R_32F;
    return true;
  default:
    return false;
  }
}

// =============================================================================
// Algorithm cache: query heuristic once per unique problem shape, reuse after.
// =============================================================================

struct GemmCacheKey {
  int64_t M, N, K, transA, transB, typeCode;
  bool operator==(const GemmCacheKey &o) const {
    return M == o.M && N == o.N && K == o.K && transA == o.transA &&
           transB == o.transB && typeCode == o.typeCode;
  }
};

struct GemmCacheKeyHash {
  size_t operator()(const GemmCacheKey &k) const {
    size_t h = std::hash<int64_t>{}(k.M);
    h ^= std::hash<int64_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.transA) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.transB) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int64_t>{}(k.typeCode) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

struct GemmCacheEntry {
  hipblasLtMatmulAlgo_t algo;
  size_t workspace_size;
};

static std::unordered_map<GemmCacheKey, GemmCacheEntry, GemmCacheKeyHash>
    g_gemm_algo_cache;

// =============================================================================
// Broadcast helper: write beta * C_broadcast into output using MIOpen
// =============================================================================
// Uses miopenOpTensor(Add) with broadcasting:
//   output = beta * C_broadcast
// C is normalized to 2D [cDim0, cDim1], broadcastable to [M, N].
// After this, hipblasLtMatmul accumulates with effective_beta=1.0:
//   output = alpha * A * B + 1.0 * output  (= alpha*A*B + beta*C_broadcast)

static bool resolveGemmMiopenType(int64_t typeCode, miopenDataType_t &dt) {
  switch (typeCode) {
  case kTypeFloat16:
    dt = miopenHalf;
    return true;
  case kTypeFloat32:
    dt = miopenFloat;
    return true;
  case kTypeFloat64:
    dt = miopenDouble;
    return true;
  case kTypeBFloat16:
    dt = miopenBFloat16;
    return true;
  default:
    return false;
  }
}

static int broadcastBiasToOutput(RuntimeState *state, const void *C,
                                 void *output, int64_t M, int64_t N,
                                 int64_t cDim0, int64_t cDim1, float beta,
                                 int64_t typeCode) {
  miopenHandle_t handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  if (!handle) {
    fprintf(stderr, "wrap_gemm: broadcastBiasToOutput: null MIOpen handle\n");
    return -1;
  }

  miopenDataType_t dt;
  if (!resolveGemmMiopenType(typeCode, dt)) {
    fprintf(stderr,
            "wrap_gemm: broadcastBiasToOutput: unsupported typeCode %lld\n",
            (long long)typeCode);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: broadcastBiasToOutput C[%lld,%lld] -> "
                    "[%lld,%lld], beta=%f\n",
                    (long long)cDim0, (long long)cDim1, (long long)M,
                    (long long)N, beta);

  miopenTensorDescriptor_t cDesc = nullptr;
  miopenTensorDescriptor_t outDesc = nullptr;
  int result = 0;

  MIOPEN_CHECK(miopenCreateTensorDescriptor(&cDesc));
  MIOPEN_CHECK(miopenCreateTensorDescriptor(&outDesc));

  MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      cDesc, dt, 1, 1, static_cast<int>(cDim0), static_cast<int>(cDim1)));
  MIOPEN_CHECK(miopenSet4dTensorDescriptor(
      outDesc, dt, 1, 1, static_cast<int>(M), static_cast<int>(N)));

  // output = beta * (C + 0 * C) + 0 * output = beta * C_broadcast
  if (typeCode == kTypeFloat64) {
    double alpha1 = static_cast<double>(beta), alpha2 = 0.0, beta_c = 0.0;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, cDesc, C,
                                &alpha2, cDesc, C, &beta_c, outDesc, output));
  } else {
    float alpha1 = beta, alpha2 = 0.0f, beta_c = 0.0f;
    MIOPEN_CHECK(miopenOpTensor(handle, miopenTensorOpAdd, &alpha1, cDesc, C,
                                &alpha2, cDesc, C, &beta_c, outDesc, output));
  }

cleanup:
  if (cDesc)
    miopenDestroyTensorDescriptor(cDesc);
  if (outDesc)
    miopenDestroyTensorDescriptor(outDesc);
  return result;
}

// =============================================================================
// ONNX Gemm via hipBLASLt
// =============================================================================
//
// ONNX Gemm semantics (row-major):
//   Y = alpha * op(A) * op(B) + beta * C
//   op(A) = A^T if transA else A  →  always [M, K] after op
//   op(B) = B^T if transB else B  →  always [K, N] after op
//   C is optional, broadcastable to [M, N]
//   Y has shape [M, N]
//
// C broadcast shapes (ONNX unidirectional broadcastable to [M, N]):
//   []      → scalar      → cDim0=1, cDim1=1
//   [N]     → row vector   → cDim0=1, cDim1=N   (most common: FC bias)
//   [1, N]  → row vector   → cDim0=1, cDim1=N
//   [M, 1]  → col vector   → cDim0=M, cDim1=1
//   [M, N]  → no broadcast  → cDim0=M, cDim1=N
//
// hipBLASLt uses column-major. Using the transpose identity:
//   Y^T = alpha * op(B)^T * op(A)^T + beta * C^T
//
// So we swap A↔B in the hipBLASLt call with m=N, n=M, k=K:
//   hipBLASLt "A" = B buffer, TRANSA = transB ? OP_T : OP_N
//   hipBLASLt "B" = A buffer, TRANSB = transA ? OP_T : OP_N
//
// Matrix layouts (col-major view of row-major data):
//   transB=0: B_rm[K,N] → col-major [N,K] ld=N
//   transB=1: B_rm[N,K] → col-major [K,N] ld=K
//   transA=0: A_rm[M,K] → col-major [K,M] ld=K
//   transA=1: A_rm[K,M] → col-major [M,K] ld=M
//   C/Y:      [M,N] rm  → col-major [N,M] ld=N
// =============================================================================

int wrap_gemm(RuntimeState *state, const void *A, const void *B, const void *C,
              void *output, int64_t M, int64_t N, int64_t K, float alpha,
              float beta, int64_t transA, int64_t transB, int64_t typeCode,
              int64_t cDim0, int64_t cDim1) {
  if (!state || !A || !B || !output) {
    fprintf(stderr, "wrap_gemm: invalid arguments\n");
    return -1;
  }
  HIPDNN_EP_OP_PROFILE_SCOPE(state);

  hipblasLtHandle_t handle =
      static_cast<hipblasLtHandle_t>(hipdnn_ep_state_get_hipblas_handle(state));
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  if (!handle || !stream) {
    fprintf(stderr, "wrap_gemm: null handle or stream\n");
    return -1;
  }

  hipDataType dataType;
  hipblasComputeType_t computeType;
  hipDataType scaleType;
  if (!resolveGemmTypes(typeCode, dataType, computeType, scaleType)) {
    fprintf(stderr, "wrap_gemm: unsupported typeCode %lld\n",
            (long long)typeCode);
    return -1;
  }

  // Determine if C needs broadcasting.
  // When C is [M, N], hipblasLtMatmul handles it directly (single-pass).
  // Otherwise, we first broadcast beta*C into output via MIOpen, then let
  // hipblasLtMatmul accumulate with effective_beta=1.0 on top of it.
  bool needsBroadcast = C && !(cDim0 == M && cDim1 == N);

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: M=%lld, N=%lld, K=%lld, transA=%lld, "
                    "transB=%lld, alpha=%f, beta=%f, typeCode=%lld, C=%p, "
                    "cDim0=%lld, cDim1=%lld, needsBroadcast=%d\n",
                    (long long)M, (long long)N, (long long)K, (long long)transA,
                    (long long)transB, alpha, beta, (long long)typeCode, C,
                    (long long)cDim0, (long long)cDim1, (int)needsBroadcast);

  // Pre-broadcast: write beta * C_broadcast into output before matmul.
  if (needsBroadcast) {
    int bc_result = broadcastBiasToOutput(state, C, output, M, N, cDim0, cDim1,
                                          beta, typeCode);
    if (bc_result != 0)
      return bc_result;
  }

  // Select effective beta and C pointer for hipblasLtMatmul.
  //   C absent:     beta=0, C_ptr=output (placeholder, content irrelevant)
  //   C is [M,N]:   beta=beta, C_ptr=C  (direct single-pass, no broadcast)
  //   C broadcast:  beta=1.0, C_ptr=output (already holds beta*C_broadcast)
  float effective_beta;
  const void *effective_C;
  if (!C) {
    effective_beta = 0.0f;
    effective_C = output;
  } else if (!needsBroadcast) {
    effective_beta = beta;
    effective_C = C;
  } else {
    effective_beta = 1.0f;
    effective_C = output;
  }

  hipblasLtMatrixLayout_t matA_layout = nullptr;
  hipblasLtMatrixLayout_t matB_layout = nullptr;
  hipblasLtMatrixLayout_t matC_layout = nullptr;
  hipblasLtMatmulDesc_t matmul_desc = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  int result = 0;

  GemmCacheKey key{M, N, K, transA, transB, typeCode};
  auto it = g_gemm_algo_cache.find(key);

  // hipBLASLt "A" = B buffer
  int64_t hblA_rows, hblA_cols, hblA_ld;
  if (!transB) {
    hblA_rows = N;
    hblA_cols = K;
    hblA_ld = N;
  } else {
    hblA_rows = K;
    hblA_cols = N;
    hblA_ld = K;
  }

  // hipBLASLt "B" = A buffer
  int64_t hblB_rows, hblB_cols, hblB_ld;
  if (!transA) {
    hblB_rows = K;
    hblB_cols = M;
    hblB_ld = K;
  } else {
    hblB_rows = M;
    hblB_cols = K;
    hblB_ld = M;
  }

  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA_layout, dataType, hblA_rows,
                                            hblA_cols, hblA_ld));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB_layout, dataType, hblB_rows,
                                            hblB_cols, hblB_ld));
  // C and output: [M,N] row-major → col-major [N,M] ld=N
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC_layout, dataType, N, M, N));

  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&matmul_desc, computeType, scaleType));

  {
    hipblasOperation_t opA = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
  }

  // Algorithm selection with caching
  if (it == g_gemm_algo_cache.end()) {
    HIPBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    const size_t max_ws = 256ULL << 20; // 256 MB
    HIPBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
        sizeof(max_ws)));

    hipblasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    HIPBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc, matA_layout, matB_layout, matC_layout, matC_layout,
        pref, 1, &heur, &returned));
    hipblasLtMatmulPreferenceDestroy(pref);
    pref = nullptr;

    if (returned == 0) {
      fprintf(stderr,
              "wrap_gemm: no algorithm found for M=%lld N=%lld K=%lld "
              "transA=%lld transB=%lld typeCode=%lld\n",
              (long long)M, (long long)N, (long long)K, (long long)transA,
              (long long)transB, (long long)typeCode);
      result = -1;
      goto cleanup;
    }

    GemmCacheEntry entry;
    entry.algo = heur.algo;
    entry.workspace_size = heur.workspaceSize;
    it = g_gemm_algo_cache.emplace(key, entry).first;

    RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: cached algo for M=%lld N=%lld K=%lld "
                      "transA=%lld transB=%lld (ws=%zu)\n",
                      (long long)M, (long long)N, (long long)K,
                      (long long)transA, (long long)transB,
                      entry.workspace_size);
  }

  {
    const GemmCacheEntry &cached = it->second;
    if (cached.workspace_size > 0) {
      if (hipdnn_ep_state_ensure_workspace(state, cached.workspace_size) != 0) {
        result = -1;
        goto cleanup;
      }
    }

    void *ws_ptr = hipdnn_ep_state_get_workspace(state);
    size_t ws_size = hipdnn_ep_state_get_workspace_size(state);

    if (typeCode == kTypeFloat64) {
      double alpha_d = static_cast<double>(alpha);
      double beta_d = static_cast<double>(effective_beta);
      HIPBLAS_CHECK(hipblasLtMatmul(
          handle, matmul_desc, &alpha_d, B, matA_layout, A, matB_layout,
          &beta_d, effective_C, matC_layout, output, matC_layout,
          const_cast<hipblasLtMatmulAlgo_t *>(&cached.algo), ws_ptr, ws_size,
          stream));
    } else {
      HIPBLAS_CHECK(hipblasLtMatmul(
          handle, matmul_desc, &alpha, B, matA_layout, A, matB_layout,
          &effective_beta, effective_C, matC_layout, output, matC_layout,
          const_cast<hipblasLtMatmulAlgo_t *>(&cached.algo), ws_ptr, ws_size,
          stream));
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: completed successfully\n");

cleanup:
  if (pref)
    hipblasLtMatmulPreferenceDestroy(pref);
  if (matA_layout)
    hipblasLtMatrixLayoutDestroy(matA_layout);
  if (matB_layout)
    hipblasLtMatrixLayoutDestroy(matB_layout);
  if (matC_layout)
    hipblasLtMatrixLayoutDestroy(matC_layout);
  if (matmul_desc)
    hipblasLtMatmulDescDestroy(matmul_desc);

  return result;
}
