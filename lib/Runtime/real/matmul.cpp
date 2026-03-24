/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"
#include "error_check_macros.h"

#include <cstdio>
#include <cstring>

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define HIPBLAS_CHECK(cmd)                                                     \
  do {                                                                         \
    hipblasStatus_t status = (cmd);                                            \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "hipBLASLt error at %s:%d: %d\n", __FILE__, __LINE__,    \
              status);                                                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

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

  const char *type_name = (elem_size == 2)   ? "f16"
                          : (elem_size == 4) ? "f32"
                                             : "?";
  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: M=%lld, N=%lld, K=%lld, "
                    "batch=%lld, elem_size=%lld (%s), "
                    "total_bytes=%lld\n",
                    (long long)M, (long long)N, (long long)K,
                    (long long)batch_count, (long long)elem_size, type_name,
                    (long long)(batch_count * M * N * elem_size));

  hipDataType data_type;
  if (elem_size == 2)
    data_type = HIP_R_16F;
  else if (elem_size == 4)
    data_type = HIP_R_32F;
  else {
    fprintf(stderr, "wrap_hipblasLtMatmul: unsupported elem_size %lld\n",
            (long long)elem_size);
    return -1;
  }

  // Initialize all resource pointers to nullptr for safe cleanup
  hipblasLtMatrixLayout_t matA_layout = nullptr;
  hipblasLtMatrixLayout_t matB_layout = nullptr;
  hipblasLtMatrixLayout_t matC_layout = nullptr;
  hipblasLtMatmulDesc_t matmul_desc = nullptr;
  int result = 0;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Row-major → column-major trick: swap A/B and M/N
  int64_t ld_A_hipblas = N; // leading dim of B viewed as col-major
  int64_t ld_B_hipblas = K; // leading dim of A viewed as col-major
  int64_t ld_C_hipblas = N; // leading dim of output viewed as col-major

  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matA_layout, data_type, N, K, ld_A_hipblas), cleanup);
  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matB_layout, data_type, K, M, ld_B_hipblas), cleanup);
  HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutCreate(&matC_layout, data_type, N, M, ld_C_hipblas), cleanup);

  if (batch_count > 1) {
    int64_t stride_A_hipblas = K * N; // stride over B batches
    int64_t stride_B_hipblas = M * K; // stride over A batches
    int64_t stride_C_hipblas = M * N; // stride over output batches

    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matA_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)), cleanup);
    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matA_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_A_hipblas, sizeof(stride_A_hipblas)), cleanup);

    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matB_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)), cleanup);
    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matB_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_B_hipblas, sizeof(stride_B_hipblas)), cleanup);

    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matC_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)), cleanup);
    HIPBLAS_CHECK_GOTO(hipblasLtMatrixLayoutSetAttribute(
        matC_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_C_hipblas, sizeof(stride_C_hipblas)), cleanup);
  }

  HIPBLAS_CHECK_GOTO(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), cleanup);

  HIPBLAS_CHECK_GOTO(hipblasLtMatmul(handle, matmul_desc, &alpha, B,
                                     matA_layout,    // "A" = B (row→col trick)
                                     A, matB_layout, // "B" = A (row→col trick)
                                     &beta, output, matC_layout, output, matC_layout, nullptr,
                                     nullptr, 0, stream), cleanup);

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipblasLtMatmul: completed successfully\n");

cleanup:
  // Best-effort cleanup: free all allocated resources
  // Continue cleanup even if individual operations fail
  if (matA_layout) {
    hipblasLtMatrixLayoutDestroy(matA_layout);
  }
  if (matB_layout) {
    hipblasLtMatrixLayoutDestroy(matB_layout);
  }
  if (matC_layout) {
    hipblasLtMatrixLayoutDestroy(matC_layout);
  }
  if (matmul_desc) {
    hipblasLtMatmulDescDestroy(matmul_desc);
  }

  return result;
}
