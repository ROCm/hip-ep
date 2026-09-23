/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_CK_GEMM_SELECT_H
#define HIPDNN_EP_CK_GEMM_SELECT_H

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>

// Time every CK instance that accepts this problem and return the fastest, or
// -1 when none does. CK ships nothing like AlgoGetHeuristic, so measuring is
// the only way to pick. Arguments mirror hip_ck_gemm_run minus the instance,
// i.e. hipBLASLt's column-major convention.
//
// Timing iterations overwrite `output`, so the caller must not have seeded it
// with anything the real call still needs.
inline int ckSelectGemmInstance(hipStream_t stream, const void *A,
                                const void *B, const void *bias, void *output,
                                int64_t m, int64_t n, int64_t k, int64_t batch,
                                int transA, int transB, int abDtype, int dDtype,
                                float alpha, int64_t lda, int64_t ldb,
                                int64_t ldd, int64_t strideA, int64_t strideB,
                                int64_t strideD) {
  auto launch = [&](int instance) {
    return hip_ck_gemm_run(stream, instance, A, B, bias, output, m, n, k, batch,
                           transA, transB, abDtype, dDtype, alpha, lda, ldb,
                           ldd, strideA, strideB, strideD);
  };

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess ||
      hipEventCreate(&stop) != hipSuccess) {
    if (start) {
      (void)hipEventDestroy(start);
    }
    if (stop) {
      (void)hipEventDestroy(stop);
    }
    return -1;
  }

  int best = -1;
  double best_ms = 1e30;
  const int count = hip_ck_gemm_num_instances();
  for (int i = 0; i < count; ++i) {
    if (launch(i) != 0) {
      continue;
    }
    // An instance CK accepted can still fault; clear the sticky error so the
    // next one is not blamed for it.
    if (hipStreamSynchronize(stream) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    if (hipEventRecord(start, stream) != hipSuccess) {
      continue;
    }
    for (int r = 0; r < 3; ++r) {
      launch(i);
    }
    if (hipEventRecord(stop, stream) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
      continue;
    }
    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
      continue;
    }
    if (ms < best_ms) {
      best_ms = ms;
      best = i;
    }
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return best;
}

#endif // HIPDNN_EP_CK_GEMM_SELECT_H
