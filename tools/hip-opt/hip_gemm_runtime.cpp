//===- hip_gemm_runtime.cpp - Runtime wrapper for hipBLAS-LT GEMM ---------===//
//
// This file implements the C function hip_gemm_f32() that the MLIR-compiled
// code calls. It uses hipBLAS-LT to perform C = A @ B on device.
//
// The MLIR lowering layer calls this function; swapping this file for a
// hipDNN-based implementation in the future requires no MLIR changes.
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

// Stub implementations for hip.create_handle / hip.destroy_handle.
// These are declared by the MLIR-generated code but are no-ops in this demo
// because hipBLAS-LT manages its own handle internally.
extern "C" void* hipCreateHandle() { return nullptr; }
extern "C" void  hipDestroyHandle(void*) {}

extern "C" void hip_gemm_f32(float* A, float* B, float* C,
                              int64_t M, int64_t K, int64_t N) {
  // hipBLAS-LT uses column-major layout.
  // For row-major C = A @ B, we compute column-major C' = B' @ A'
  // by swapping operands and transposing the dimensions:
  //   blas_A = B,  blas_B = A
  //   blas_M = N,  blas_N = M,  blas_K = K
  //   lda = N,     ldb = K,     ldc = N

  const int64_t blas_M = N;
  const int64_t blas_N = M;
  const int64_t blas_K = K;
  const int64_t lda = N;   // leading dim of blas_A (= B in row-major)
  const int64_t ldb = K;   // leading dim of blas_B (= A in row-major)
  const int64_t ldc = N;   // leading dim of C

  float alpha = 1.0f;
  float beta = 0.0f;

  // 1. Create handle
  hipblasLtHandle_t handle = nullptr;
  HIPBLASLT_CHECK(hipblasLtCreate(&handle));

  // 2. Create matmul descriptor
  hipblasLtMatmulDesc_t matmul_desc = nullptr;
  HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(
      &matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  hipblasOperation_t op = HIPBLAS_OP_N;
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op)));
  HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
      matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op)));

  // 3. Create matrix layouts (column-major)
  hipblasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr;
  hipblasLtMatrixLayout_t layout_c = nullptr, layout_d = nullptr;

  // blas_A = B (row-major) viewed as column-major: (blas_M x blas_K) with lda
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layout_a, HIP_R_32F, blas_M, blas_K, lda));
  // blas_B = A (row-major) viewed as column-major: (blas_K x blas_N) with ldb
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layout_b, HIP_R_32F, blas_K, blas_N, ldb));
  // C: (blas_M x blas_N) with ldc
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layout_c, HIP_R_32F, blas_M, blas_N, ldc));
  // D = C (in-place output)
  HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(
      &layout_d, HIP_R_32F, blas_M, blas_N, ldc));

  // 4. Select algorithm via heuristic
  hipblasLtMatmulPreference_t preference = nullptr;
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));

  const size_t max_workspace = 32 * 1024 * 1024;  // 32 MB
  HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace, sizeof(max_workspace)));

  hipblasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_results = 0;
  HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
      handle, matmul_desc,
      layout_a, layout_b, layout_c, layout_d,
      preference, 1, &heuristic_result, &returned_results));

  if (returned_results == 0) {
    fprintf(stderr, "hip_gemm_f32: no algorithm found\n");
    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(layout_d);
    hipblasLtMatrixLayoutDestroy(layout_c);
    hipblasLtMatrixLayoutDestroy(layout_b);
    hipblasLtMatrixLayoutDestroy(layout_a);
    hipblasLtMatmulDescDestroy(matmul_desc);
    hipblasLtDestroy(handle);
    return;
  }

  // 5. Allocate workspace on device
  void* workspace_ptr = nullptr;
  if (heuristic_result.workspaceSize > 0) {
    hipMalloc(&workspace_ptr, heuristic_result.workspaceSize);
  }

  // 6. Execute GEMM
  //    D = alpha * blas_A @ blas_B + beta * C
  //    blas_A = B (row-major), blas_B = A (row-major)
  //    This computes row-major C = A @ B
  HIPBLASLT_CHECK(hipblasLtMatmul(
      handle, matmul_desc,
      &alpha,
      B, layout_a,          // blas_A = B
      A, layout_b,          // blas_B = A
      &beta,
      C, layout_c,          // C input (unused since beta=0)
      C, layout_d,          // D output = C
      &heuristic_result.algo,
      workspace_ptr, heuristic_result.workspaceSize,
      nullptr));            // default stream

  hipDeviceSynchronize();

  // 7. Cleanup
  if (workspace_ptr) hipFree(workspace_ptr);
  hipblasLtMatmulPreferenceDestroy(preference);
  hipblasLtMatrixLayoutDestroy(layout_d);
  hipblasLtMatrixLayoutDestroy(layout_c);
  hipblasLtMatrixLayoutDestroy(layout_b);
  hipblasLtMatrixLayoutDestroy(layout_a);
  hipblasLtMatmulDescDestroy(matmul_desc);
  hipblasLtDestroy(handle);
}
