/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_gemm.cpp - Two-matmul test via inference interface -------------===//
//
// A[B,S,K] @ B0[K,N] -> tmp[B,S,N] -> tmp @ B1[N,P] -> C[B,S,P]
// B0, B1 are 2D (broadcast across batch).
//
//===----------------------------------------------------------------------===//

#include "hip_inference.h"
#include <cmath>
#include <cstdio>
#include <vector>

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

  void *state = nullptr;
  inference_init(&state);

  int64_t a_shape[] = {B, S, K};
  int64_t b0_shape[] = {K, N};
  int64_t b1_shape[] = {N, P};
  int64_t c_shape[] = {B, S, P};
  tensor_t inputs[] = {
      {h_A.data(), a_shape, 3, sizeof(float)},
      {h_B0.data(), b0_shape, 2, sizeof(float)},
      {h_B1.data(), b1_shape, 2, sizeof(float)},
  };
  tensor_t outputs[] = {
      {h_C.data(), c_shape, 3, sizeof(float)},
  };
  span_t in_span = {inputs, 3};
  span_t out_span = {outputs, 1};

  printf("Running on GPU...\n");
  int ret = inference_compute(state, &in_span, &out_span);
  if (ret != 0) {
    fprintf(stderr, "inference_compute failed: %d\n", ret);
    inference_cleanup(state);
    return 1;
  }

  float mx = 0;
  for (int64_t i = 0; i < B * S * P; ++i) {
    float d = std::fabs(h_C[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-4f ? "PASS" : "FAIL");

  inference_cleanup(state);
  return (mx < 1e-4f) ? 0 : 1;
}
