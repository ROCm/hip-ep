/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "dense_gemm.h"

// Retain the public column-major fp32 ABI. The custom row-major view computes
// C^T = B^T * A^T. The kernel resolver already skips online candidate sweeps
// for C==Y and beta!=0, retaining this in-place ABI without device allocation.
int wrap_hipblasLtGemm(void *handle, void *stream, int64_t m, int64_t n,
                       int64_t k, const void *alpha, const void *A,
                       const void *B, const void *beta, void *C) {
  (void)handle;
  if (!alpha || !beta)
    return -1;
  return hipdnn_ep::dense_gemm::launch(
      stream, B, A, C, C, n, m, k, *static_cast<const float *>(alpha),
      *static_cast<const float *>(beta), 0, 0, 1, n, m);
}
