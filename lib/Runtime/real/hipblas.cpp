/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// Error checking macro
#define HIPBLAS_CHECK(cmd)                                                     \
  do {                                                                         \
    hipblasStatus_t status = (cmd);                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "hipBLAS error at %s:%d: %d\n", __FILE__, __LINE__,      \
              status);                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

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

  // Create matrix descriptors (assuming float32, column-major)
  hipblasLtMatrixLayout_t matA, matB, matC;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, m));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, k));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m));

  // Create operation descriptor
  hipblasLtMatmulDesc_t matmul_desc;
  HIPBLAS_CHECK(
      hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  // Perform GEMM
  HIPBLAS_CHECK(hipblasLtMatmul(hipblas_handle, matmul_desc, alpha, A, matA, B,
                                matB, beta, C, matC, C, matC,
                                nullptr, // algo
                                nullptr, // workspace
                                0,       // workspaceSize
                                hip_stream));

  // Cleanup
  hipblasLtMatrixLayoutDestroy(matA);
  hipblasLtMatrixLayoutDestroy(matB);
  hipblasLtMatrixLayoutDestroy(matC);
  hipblasLtMatmulDescDestroy(matmul_desc);

  return 0;
}
