/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_CK_GEMM_SELECT_H
#define HIPDNN_EP_CK_GEMM_SELECT_H

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>

// Return the CK instance to serve this problem, or -1 when none accepts it.
// The offline table proposes up to three instances; a single accepted one wins
// outright, several are timed against each other. Without an accepted proposal
// every accepting instance is timed and the fastest wins. CK ships nothing like
// AlgoGetHeuristic, so measuring is the only way to pick. Arguments mirror
// hip_ck_gemm_run minus the instance, i.e. hipBLASLt's column-major convention.
//
// Probe and timing launches overwrite `output`, so the caller must not have
// seeded it with anything the real call still needs.
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
  // A launch CK accepted can still leave an error; consume it so the next
  // candidate is not blamed for it.
  auto accepted = [&](int instance) {
    if (launch(instance) != 0) {
      return false;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
    return true;
  };

  int proposed[3];
  const int num_proposed =
      hip_ck_gemm_lut_candidates(m, n, k, batch, transA, abDtype, dDtype,
                                 bias != nullptr ? 1 : 0, proposed, 3);
  int candidates[3];
  int num_candidates = 0;
  for (int c = 0; c < num_proposed; ++c) {
    if (accepted(proposed[c])) {
      candidates[num_candidates++] = proposed[c];
    }
  }
  if (num_candidates == 1) {
    return candidates[0];
  }

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
    (void)hipGetLastError();
    return num_candidates > 0 ? candidates[0] : -1;
  }

  // Fastest of two rounds: one 3-iteration sample is noisy enough to rank a
  // slower instance first. Returns false when no round could be timed.
  auto timeInstance = [&](int instance, float &out_ms) {
    bool timed = false;
    for (int round = 0; round < 2; ++round) {
      if (hipEventRecord(start, stream) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      for (int r = 0; r < 3; ++r) {
        launch(instance);
      }
      if (hipEventRecord(stop, stream) != hipSuccess ||
          hipEventSynchronize(stop) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, start, stop) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      if (!timed || ms < out_ms) {
        out_ms = ms;
        timed = true;
      }
    }
    return timed;
  };

  int best = -1;
  float best_ms = 0.0f;
  auto consider = [&](int instance) {
    float ms = 0.0f;
    if (timeInstance(instance, ms) && (best < 0 || ms < best_ms)) {
      best = instance;
      best_ms = ms;
    }
  };
  if (num_candidates > 1) {
    for (int c = 0; c < num_candidates; ++c) {
      consider(candidates[c]);
    }
    if (best < 0) {
      best = candidates[0];
    }
  } else {
    const int count = hip_ck_gemm_num_instances();
    for (int i = 0; i < count; ++i) {
      if (accepted(i)) {
        consider(i);
      }
    }
  }
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return best;
}

#endif // HIPDNN_EP_CK_GEMM_SELECT_H
