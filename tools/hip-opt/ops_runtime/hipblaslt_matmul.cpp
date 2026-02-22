/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hipblaslt_matmul.cpp - hip.hipblaslt.matmul runtime
//-----------------===//
//
// Rank-generic matmul: C = A @ B  (row-major)
//
// Signature from MLIR lowering:
//   hip_hipblaslt_matmul(handle, A, B, C, rankA, rankB, batch, M, K, N)
//
// - batch from A: if A is 3D, batch = A.dim[0]; else batch = 1
// - M = A.dim[-2], K = A.dim[-1], N = B.dim[-1]
// - B broadcast: if rankB < rankA, stride_B = 0 (same W for all batches)
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <cstdio>

#define HIPBLASLT_CHECK(call)                                              \
  do {                                                                     \
    hipblasStatus_t status = (call);                                       \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                \
      fprintf(stderr, "hipBLAS-LT error at %s:%d (status=%d)\n", __FILE__, \
              __LINE__, status);                                           \
      return;                                                              \
    }                                                                      \
  } while (0)

extern "C" void hip_hipblaslt_matmul(void* /*handle*/, void* A, void* B,
                                     void* C, int64_t rankA, int64_t rankB,
                                     int64_t batch, int64_t M, int64_t K,
                                     int64_t N) {
  fprintf(stderr,
          "[hipblaslt.matmul] rankA=%lld rankB=%lld batch=%lld M=%lld K=%lld "
          "N=%lld\n",
          (long long)rankA, (long long)rankB, (long long)batch, (long long)M,
          (long long)K, (long long)N);

  // hipBLAS-LT column-major: for row-major C = A @ B, compute C' = B' @ A'
  const int64_t blas_M = N, blas_N = M, blas_K = K;
  const int64_t lda = N, ldb = K, ldc = N;
  float alpha = 1.0f, beta = 0.0f;

  // Strides for batched GEMM
  // blas_A = B (row-major), blas_B = A (row-major) -- swapped for col-major
  // trick
  int64_t stride_blas_a =
      (rankB < rankA) ? 0 : K * N;  // B broadcast if lower rank
  int64_t stride_blas_b = M * K;    // A always has batch stride
  int64_t stride_c = M * N;

  hipblasLtHandle_t handle = nullptr;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  hipblasLtMatmulDesc_t desc = nullptr;
  HIPBLASLT_CHECK(
      hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op = HIPBLAS_OP_N;
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));

  hipblasLtMatrixLayout_t la, lb, lc, ld;
  HIPBLASLT_CHECK(
      hipblasLtMatrixLayoutCreate(&la, HIP_R_32F, blas_M, blas_K, lda));
  HIPBLASLT_CHECK(
      hipblasLtMatrixLayoutCreate(&lb, HIP_R_32F, blas_K, blas_N, ldb));
  HIPBLASLT_CHECK(
      hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F, blas_M, blas_N, ldc));
  HIPBLASLT_CHECK(
      hipblasLtMatrixLayoutCreate(&ld, HIP_R_32F, blas_M, blas_N, ldc));

  if (batch > 1) {
    // blas_A = B (row-major), blas_B = A (row-major)
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        la, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        la, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_blas_a,
        sizeof(stride_blas_a)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lb, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lb, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_blas_b,
        sizeof(stride_blas_b)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lc, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lc, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c,
        sizeof(stride_c)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        ld, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        ld, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c,
        sizeof(stride_c)));
  }

  hipblasLtMatmulPreference_t pref = nullptr;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  const size_t max_ws = 32 * 1024 * 1024;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
      sizeof(max_ws)));

  hipblasLtMatmulHeuristicResult_t result = {};
  int returned = 0;
  HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld,
                                                  pref, 1, &result, &returned));

  void* ws = nullptr;
  if (returned > 0 && result.workspaceSize > 0)
    hipMalloc(&ws, result.workspaceSize);

  if (returned > 0) {
    HIPBLASLT_CHECK(hipblasLtMatmul(handle, desc, &alpha, B, la, A, lb, &beta,
                                    C, lc, C, ld, &result.algo, ws,
                                    result.workspaceSize, nullptr));
    hipDeviceSynchronize();
  } else {
    fprintf(stderr, "[hipblaslt.matmul] no algorithm found\n");
  }

  if (ws)
    hipFree(ws);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(ld);
  hipblasLtMatrixLayoutDestroy(lc);
  hipblasLtMatrixLayoutDestroy(lb);
  hipblasLtMatrixLayoutDestroy(la);
  hipblasLtMatmulDescDestroy(desc);
  hipblasLtDestroy(handle);
}
