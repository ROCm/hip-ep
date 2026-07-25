/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_rms_norm.cpp - Main driver for two-rms_norm DPS test (3D) -----===//
//
// B[B,S,D] = RMSNorm(RMSNorm(A[B,S,D], W0[D]), W1[D])
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <vector>

extern "C" __declspec(dllimport) void two_rms_norms(
    float *A_a, float *A_al, int64_t A_o, int64_t A_s0, int64_t A_s1,
    int64_t A_s2, int64_t A_st0, int64_t A_st1, int64_t A_st2, float *W0_a,
    float *W0_al, int64_t W0_o, int64_t W0_s0, int64_t W0_st0, float *W1_a,
    float *W1_al, int64_t W1_o, int64_t W1_s0, int64_t W1_st0, float *B_a,
    float *B_al, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_s2,
    int64_t B_st0, int64_t B_st1, int64_t B_st2);

static void cpu_rms_norm(const float *in, const float *w, float *out,
                         int64_t rows, int64_t D, float eps) {
  for (int64_t n = 0; n < rows; ++n) {
    float sq = 0;
    for (int64_t d = 0; d < D; ++d) {
      float v = in[n * D + d];
      sq += v * v;
    }
    float rms = std::sqrt(sq / float(D) + eps);
    for (int64_t d = 0; d < D; ++d)
      out[n * D + d] = in[n * D + d] / rms * w[d];
  }
}

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t e = (call);                                                     \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,             \
              hipGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  const int64_t Bat = 1, S = 4, D = 8;
  const int64_t n = Bat * S * D;
  const float eps = 1e-5f;
  printf("3D rms_norm test: [%lld,%lld,%lld]\n", Bat, S, D);

  std::vector<float> h_A(n), h_W0(D), h_W1(D), h_B(n, 0), h_ref(n);
  for (int64_t i = 0; i < n; ++i)
    h_A[i] = float(i + 1) * 0.1f;
  for (int64_t d = 0; d < D; ++d) {
    h_W0[d] = 1.0f;
    h_W1[d] = 0.5f;
  }

  std::vector<float> h_tmp(n);
  cpu_rms_norm(h_A.data(), h_W0.data(), h_tmp.data(), Bat * S, D, eps);
  cpu_rms_norm(h_tmp.data(), h_W1.data(), h_ref.data(), Bat * S, D, eps);

  float *d_A, *d_W0, *d_W1, *d_B;
  HIP_CHECK(hipMalloc(&d_A, n * 4));
  HIP_CHECK(hipMalloc(&d_W0, D * 4));
  HIP_CHECK(hipMalloc(&d_W1, D * 4));
  HIP_CHECK(hipMalloc(&d_B, n * 4));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), n * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W0, h_W0.data(), D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), D * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_B, 0, n * 4));

  printf("Running on GPU...\n");
  two_rms_norms(d_A, d_A, 0, Bat, S, D, S * D, D, 1, d_W0, d_W0, 0, D, 1, d_W1,
                d_W1, 0, D, 1, d_B, d_B, 0, Bat, S, D, S * D, D, 1);

  HIP_CHECK(hipMemcpy(h_B.data(), d_B, n * 4, hipMemcpyDeviceToHost));

  float mx = 0;
  for (int64_t i = 0; i < n; ++i) {
    float d = std::fabs(h_B[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-3f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_W0));
  HIP_CHECK(hipFree(d_W1));
  HIP_CHECK(hipFree(d_B));
  return (mx < 1e-3f) ? 0 : 1;
}
