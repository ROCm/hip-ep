//===- hipblaslt_matmul.cpp - hip.hipblaslt.matmul runtime -----------------===//
//
// Runtime for hip.hipblaslt.matmul(handle) ins(A, B) outs(C).
// The MLIR lowering extracts device pointers and dimensions from
// the memref descriptors:
//   hip_hipblaslt_matmul(handle, A_ptr, B_ptr, C_ptr, M, K, N)
// where A is [M,K], B is [K,N], C is [M,N].
//
//===----------------------------------------------------------------------===//

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>

#define HIPBLASLT_CHECK(call)                                             \
  do {                                                                    \
    hipblasStatus_t status = (call);                                      \
    if (status != HIPBLAS_STATUS_SUCCESS) {                               \
      fprintf(stderr, "hipBLAS-LT error at %s:%d (status=%d)\n",         \
              __FILE__, __LINE__, status);                                \
      return;                                                             \
    }                                                                     \
  } while (0)

extern "C" void hip_hipblaslt_matmul(void * /*handle*/,
                                      void *A, void *B, void *C,
                                      int64_t M, int64_t K, int64_t N) {
  // hipBLAS-LT uses column-major.  For row-major C = A @ B we compute
  // column-major C' = B' @ A' by swapping operands and transposing dims.
  const int64_t blas_M = N, blas_N = M, blas_K = K;
  const int64_t lda = N, ldb = K, ldc = N;
  float alpha = 1.0f, beta = 0.0f;

  hipblasLtHandle_t handle = nullptr;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  hipblasLtMatmulDesc_t desc = nullptr;
  HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op = HIPBLAS_OP_N;
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));

  hipblasLtMatrixLayout_t la, lb, lc, ld;
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&la, HIP_R_32F, blas_M, blas_K, lda));
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&lb, HIP_R_32F, blas_K, blas_N, ldb));
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, blas_M, blas_N, ldc));
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, blas_M, blas_N, ldc));

  hipblasLtMatmulPreference_t pref = nullptr;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  const size_t max_ws = 32 * 1024 * 1024;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws)));

  hipblasLtMatmulHeuristicResult_t result = {};
  int returned = 0;
  HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, pref, 1, &result, &returned));

  void *ws = nullptr;
  if (returned > 0 && result.workspaceSize > 0)
    hipMalloc(&ws, result.workspaceSize);

  if (returned > 0) {
    HIPBLASLT_CHECK(hipblasLtMatmul(handle, desc, &alpha, B, la, A, lb, &beta, C, lc, C, ld,
                                     &result.algo, ws, result.workspaceSize, nullptr));
    hipDeviceSynchronize();
  } else {
    fprintf(stderr, "[hipblaslt.matmul] no algorithm found for M=%lld K=%lld N=%lld\n",
            (long long)M, (long long)K, (long long)N);
  }

  if (ws) hipFree(ws);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(ld);
  hipblasLtMatrixLayoutDestroy(lc);
  hipblasLtMatrixLayoutDestroy(lb);
  hipblasLtMatrixLayoutDestroy(la);
  hipblasLtMatmulDescDestroy(desc);
  hipblasLtDestroy(handle);
}
