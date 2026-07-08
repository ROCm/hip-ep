/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HIPDNN_EP_DISABLE_VENDOR_BLAS: keep the wrapper symbol (so model code links)
// but return an error and pull no vendor header. See vendor_blas_stub.cpp.
#ifdef HIPDNN_EP_DISABLE_VENDOR_BLAS

#include "../hipdnn_ep_runtime.h"

#include <cstdint>
#include <cstdio>

int wrap_hipblasLtGemm(void *handle, void *stream, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *A,
                       const void *B, const void *beta, void *C) {
  (void)handle;
  (void)stream;
  (void)m;
  (void)n;
  (void)k;
  (void)alpha;
  (void)A;
  (void)B;
  (void)beta;
  (void)C;
  fprintf(stderr, "wrap_hipblasLtGemm: hipBLASLt disabled at build time "
                  "(HIPDNN_EP_DISABLE_VENDOR_BLAS); GEMM is unavailable\n");
  return -1;
}

#else // !HIPDNN_EP_DISABLE_VENDOR_BLAS

#include "../hipdnn_ep_runtime.h"
#include "error_check_macros.h"
#include "runtime_types.h"

#include <cstdio>

// Convenience wrappers for goto cleanup pattern (all functions use 'cleanup'
// label)
#define HIPBLAS_CHECK(cmd) HIPBLAS_CHECK_GOTO(cmd, cleanup)

// hipBLASLt GEMM wrapper implementation

int wrap_hipblasLtGemm(void *handle, void *stream, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *A,
                       const void *B, const void *beta, void *C) {
  if (!handle || !stream || !alpha || !A || !B || !beta || !C) {
    fprintf(stderr, "Invalid arguments to wrap_hipblasLtGemm\n");
    return -1;
  }

  hipblasLtHandle_t hipblas_handle = static_cast<hipblasLtHandle_t>(handle);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Initialize all resource pointers to nullptr for safe cleanup
  hipblasLtMatrixLayout_t matA = nullptr;
  hipblasLtMatrixLayout_t matB = nullptr;
  hipblasLtMatrixLayout_t matC = nullptr;
  hipblasLtMatmulDesc_t matmul_desc = nullptr;
  int result = 0;

  // Create matrix descriptors (assuming float32, column-major)
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, m));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, k));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m));

  // Create operation descriptor
  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  // Perform GEMM
  HIPBLAS_CHECK(hipblasLtMatmul(hipblas_handle, matmul_desc, alpha, A, matA, B,
                                matB, beta, C, matC, C, matC,
                                nullptr, // algo
                                nullptr, // workspace
                                0,       // workspaceSize
                                hip_stream));

cleanup:
  // Best-effort cleanup: free all allocated resources
  // Continue cleanup even if individual operations fail
  if (matA) {
    hipblasLtMatrixLayoutDestroy(matA);
  }
  if (matB) {
    hipblasLtMatrixLayoutDestroy(matB);
  }
  if (matC) {
    hipblasLtMatrixLayoutDestroy(matC);
  }
  if (matmul_desc) {
    hipblasLtMatmulDescDestroy(matmul_desc);
  }

  return result;
}

#endif // HIPDNN_EP_DISABLE_VENDOR_BLAS
