/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_gemm.cpp - Main driver for two-matmul DPS test (3D) -----------===//
//
// A[B,S,K] @ B0[K,N] -> tmp[B,S,N] -> tmp @ B1[N,P] -> C[B,S,P]
// B0, B1 are 2D (broadcast across batch).
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <vector>

extern "C" __declspec(dllimport) void two_matmuls(
    float *A_a, float *A_al, int64_t A_o, int64_t A_s0, int64_t A_s1,
    int64_t A_s2, int64_t A_st0, int64_t A_st1, int64_t A_st2, float *B0_a,
    float *B0_al, int64_t B0_o, int64_t B0_s0, int64_t B0_s1, int64_t B0_st0,
    int64_t B0_st1, float *B1_a, float *B1_al, int64_t B1_o, int64_t B1_s0,
    int64_t B1_s1, int64_t B1_st0, int64_t B1_st1, float *C_a, float *C_al,
    int64_t C_o, int64_t C_s0, int64_t C_s1, int64_t C_s2, int64_t C_st0,
    int64_t C_st1, int64_t C_st2);

static void cpu_matmul(const float *A, const float *B, float *C, int64_t M,
                       int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float s = 0;
      for (int64_t k = 0; k < K; ++k)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
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
  const int64_t B = 1, S = 4, K = 8, N = 4, P = 4;
  printf("3D matmul test: A[%lld,%lld,%lld] @ B0[%lld,%lld] -> tmp, tmp @ "
         "B1[%lld,%lld] -> C\n",
         B, S, K, K, N, N, P);

  std::vector<float> h_A(B * S * K), h_B0(K * N), h_B1(N * P);
  std::vector<float> h_C(B * S * P, 0), h_ref(B * S * P);
  for (int64_t i = 0; i < B * S * K; ++i)
    h_A[i] = float(i % 10) / 10.0f;
  for (int64_t i = 0; i < K * N; ++i)
    h_B0[i] = float((i + 3) % 10) / 10.0f;
  for (int64_t i = 0; i < N * P; ++i)
    h_B1[i] = float((i + 7) % 10) / 10.0f;

  std::vector<float> h_tmp(B * S * N, 0);
  for (int64_t b = 0; b < B; b++) {
    cpu_matmul(&h_A[b * S * K], h_B0.data(), &h_tmp[b * S * N], S, K, N);
    cpu_matmul(&h_tmp[b * S * N], h_B1.data(), &h_ref[b * S * P], S, N, P);
  }

  float *d_A, *d_B0, *d_B1, *d_C;
  HIP_CHECK(hipMalloc(&d_A, B * S * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B0, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B1, N * P * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, B * S * P * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), B * S * K * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B0, h_B0.data(), K * N * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), N * P * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, B * S * P * sizeof(float)));

  printf("Running on GPU...\n");
  two_matmuls(d_A, d_A, 0, B, S, K, S * K, K, 1, d_B0, d_B0, 0, K, N, N, 1,
              d_B1, d_B1, 0, N, P, P, 1, d_C, d_C, 0, B, S, P, S * P, P, 1);

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, B * S * P * sizeof(float),
                      hipMemcpyDeviceToHost));

  float mx = 0;
  for (int64_t i = 0; i < B * S * P; ++i) {
    float d = std::fabs(h_C[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-4f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B0));
  HIP_CHECK(hipFree(d_B1));
  HIP_CHECK(hipFree(d_C));
  return (mx < 1e-4f) ? 0 : 1;
}
