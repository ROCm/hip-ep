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
#include <cstdlib>
#include <cstring>

#define HIPBLASLT_CHECK(call)                                              \
  do {                                                                     \
    hipblasStatus_t status = (call);                                       \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                \
      fprintf(stderr, "hipBLAS-LT error at %s:%d (status=%d)\n", __FILE__, \
              __LINE__, status);                                           \
      return;                                                              \
    }                                                                      \
  } while (0)

static void* ensure_device(void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return ptr;
  hipPointerAttribute_t attrs = {};
  if (hipPointerGetAttributes(&attrs, ptr) == hipSuccess &&
      (attrs.type == hipMemoryTypeDevice ||
       attrs.type == hipMemoryTypeUnified))
    return ptr;
  // Non-HIP pointer (e.g. DLL .rdata constant).  Stage through a heap
  // buffer because HIP PAL's hipMemcpy can crash on read-only sections.
  void* staging = malloc(bytes);
  if (!staging) return ptr;
  memcpy(staging, ptr, bytes);
  void* d = nullptr;
  hipMalloc(&d, bytes);
  hipMemcpy(d, staging, bytes, hipMemcpyHostToDevice);
  free(staging);
  fprintf(stderr, "[hipblaslt] staged %zu bytes host->device\n", bytes);
  return d;
}

extern "C" void hip_hipblaslt_matmul(void* /*handle*/, void* A, void* B,
                                     void* C, int64_t rankA, int64_t rankB,
                                     int64_t batch, int64_t M, int64_t K,
                                     int64_t N) {
  fprintf(stderr,
          "[hipblaslt.matmul] rankA=%lld rankB=%lld batch=%lld M=%lld K=%lld "
          "N=%lld\n",
          (long long)rankA, (long long)rankB, (long long)batch, (long long)M,
          (long long)K, (long long)N);

  size_t sizeA = batch * M * K * sizeof(float);
  size_t sizeB = (rankB < rankA ? K * N : batch * K * N) * sizeof(float);
  size_t sizeC = batch * M * N * sizeof(float);
  void* devA = ensure_device(A, sizeA);
  void* devB = ensure_device(B, sizeB);
  void* devC = ensure_device(C, sizeC);
  bool allocA = (devA != A), allocB = (devB != B), allocC = (devC != C);

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
    HIPBLASLT_CHECK(hipblasLtMatmul(handle, desc, &alpha, devB, la, devA, lb,
                                    &beta, devC, lc, devC, ld, &result.algo, ws,
                                    result.workspaceSize, nullptr));
    hipDeviceSynchronize();
  } else {
    fprintf(stderr, "[hipblaslt.matmul] no algorithm found\n");
  }

  if (allocC)
    hipMemcpy(C, devC, sizeC, hipMemcpyDeviceToHost);

  if (ws)
    hipFree(ws);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(ld);
  hipblasLtMatrixLayoutDestroy(lc);
  hipblasLtMatrixLayoutDestroy(lb);
  hipblasLtMatrixLayoutDestroy(la);
  hipblasLtMatmulDescDestroy(desc);
  hipblasLtDestroy(handle);

  if (allocA) hipFree(devA);
  if (allocB) hipFree(devB);
  if (allocC) hipFree(devC);
}
