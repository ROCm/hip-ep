/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_softmax.cpp - Main driver for two-softmax DPS test (3D) -------===//
//
// B[B,S,D] = softmax(softmax(A[B,S,D]))
// Two chained softmaxes: verifies the op works correctly on 3D tensors.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <vector>

extern "C" __declspec(dllimport) void two_softmaxes(
    float *A_a, float *A_al, int64_t A_o, int64_t A_s0, int64_t A_s1,
    int64_t A_s2, int64_t A_st0, int64_t A_st1, int64_t A_st2, float *B_a,
    float *B_al, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_s2,
    int64_t B_st0, int64_t B_st1, int64_t B_st2);

static void cpu_softmax(const float *in, float *out, int64_t rows,
                        int64_t cols) {
  for (int64_t n = 0; n < rows; ++n) {
    float mx = in[n * cols];
    for (int64_t d = 1; d < cols; ++d)
      mx = std::fmax(mx, in[n * cols + d]);
    float sum = 0;
    for (int64_t d = 0; d < cols; ++d) {
      out[n * cols + d] = std::exp(in[n * cols + d] - mx);
      sum += out[n * cols + d];
    }
    for (int64_t d = 0; d < cols; ++d)
      out[n * cols + d] /= sum;
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
  const int64_t rows = Bat * S;
  printf("3D softmax test: B = softmax(softmax(A))  [%lld,%lld,%lld]\n", Bat, S,
         D);

  std::vector<float> h_A(n), h_B(n, 0), h_ref(n);
  for (int64_t i = 0; i < n; ++i)
    h_A[i] = float(i % 7) * 0.5f - 1.5f;

  // CPU reference: two chained softmaxes
  std::vector<float> h_tmp(n);
  cpu_softmax(h_A.data(), h_tmp.data(), rows, D);
  cpu_softmax(h_tmp.data(), h_ref.data(), rows, D);

  float *d_A, *d_B;
  HIP_CHECK(hipMalloc(&d_A, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, n * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_A, h_A.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_B, 0, n * sizeof(float)));

  printf("Running on GPU...\n");
  two_softmaxes(d_A, d_A, 0, Bat, S, D, S * D, D, 1, d_B, d_B, 0, Bat, S, D,
                S * D, D, 1);

  HIP_CHECK(
      hipMemcpy(h_B.data(), d_B, n * sizeof(float), hipMemcpyDeviceToHost));

  printf("\nGPU Result (first 2 rows):\n");
  for (int64_t i = 0; i < 2; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j)
      printf(" %7.4f", h_B[i * D + j]);
    printf(" ]\n");
  }
  printf("CPU Reference (first 2 rows):\n");
  for (int64_t i = 0; i < 2; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j)
      printf(" %7.4f", h_ref[i * D + j]);
    printf(" ]\n");
  }

  float mx = 0;
  for (int64_t i = 0; i < n; ++i) {
    float d = std::fabs(h_B[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("\nMax diff: %e  %s\n", mx, mx < 1e-4f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  return (mx < 1e-4f) ? 0 : 1;
}
