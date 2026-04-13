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
 #include <vector>
 
 #define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)
 
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
               float beta, int64_t transA, int64_t transB, int64_t typeCode) {
   if (!state || !A || !B || !output) {
     fprintf(stderr, "wrap_gemm: invalid arguments\n");
     return -1;
   }
 
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
 
   RUNTIME_DEBUG_LOG("[REAL] wrap_gemm: M=%lld, N=%lld, K=%lld, transA=%lld, "
                     "transB=%lld, alpha=%f, beta=%f, typeCode=%lld, C=%p\n",
                     (long long)M, (long long)N, (long long)K,
                     (long long)transA, (long long)transB, alpha, beta,
                     (long long)typeCode, C);
 
   float beta_f = C ? beta : 0.0f;
 
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
     hblA_rows = N; hblA_cols = K; hblA_ld = N;
   } else {
     hblA_rows = K; hblA_cols = N; hblA_ld = K;
   }
 
   // hipBLASLt "B" = A buffer
   int64_t hblB_rows, hblB_cols, hblB_ld;
   if (!transA) {
     hblB_rows = K; hblB_cols = M; hblB_ld = K;
   } else {
     hblB_rows = M; hblB_cols = K; hblB_ld = M;
   }
 
   HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA_layout, dataType, hblA_rows,
                                             hblA_cols, hblA_ld));
   HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB_layout, dataType, hblB_rows,
                                             hblB_cols, hblB_ld));
   // C and output: [M,N] row-major → col-major [N,M] ld=N
   HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC_layout, dataType, N, M, N));
 
   HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&matmul_desc, computeType, scaleType));
 
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
 
     // When C is NULL, use output as the C argument (beta=0 so content is
     // irrelevant).
     const void *C_ptr = C ? C : output;
 
     if (typeCode == kTypeFloat64) {
       double alpha_d = static_cast<double>(alpha);
       double beta_d = static_cast<double>(beta_f);
       HIPBLAS_CHECK(hipblasLtMatmul(
           handle, matmul_desc, &alpha_d, B, matA_layout, A, matB_layout,
           &beta_d, C_ptr, matC_layout, output, matC_layout,
           const_cast<hipblasLtMatmulAlgo_t *>(&cached.algo), ws_ptr, ws_size,
           stream));
     } else {
       HIPBLAS_CHECK(hipblasLtMatmul(
           handle, matmul_desc, &alpha, B, matA_layout, A, matB_layout, &beta_f,
           C_ptr, matC_layout, output, matC_layout,
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