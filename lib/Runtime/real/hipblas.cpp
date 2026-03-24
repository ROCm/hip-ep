/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "error_check_macros.h"

#include <cstdio>

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
  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, m), cleanup);
  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, k), cleanup);
  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m), cleanup);

  // Create operation descriptor
  HIPBLAS_CHECK_GOTO(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), cleanup);

  // Perform GEMM
  HIPBLAS_CHECK_GOTO(hipblasLtMatmul(hipblas_handle, matmul_desc, alpha, A, matA, B, matB,
                                     beta, C, matC, C, matC,
                                     nullptr, // algo
                                     nullptr, // workspace
                                     0,       // workspaceSize
                                     hip_stream), cleanup);

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
