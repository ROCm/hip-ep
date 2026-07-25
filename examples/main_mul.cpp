/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_mul.cpp - Main driver for two-mul DPS test (3D) ---------------===//
//
// D[B,S,D] = A * B * C   (two chained muls with 3D tensors)
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime_api.h>
#include <vector>

extern "C" __declspec(dllimport) void two_muls(
    float *A_a, float *A_al, int64_t A_o, int64_t A_s0, int64_t A_s1,
    int64_t A_s2, int64_t A_st0, int64_t A_st1, int64_t A_st2, float *B_a,
    float *B_al, int64_t B_o, int64_t B_s0, int64_t B_s1, int64_t B_s2,
    int64_t B_st0, int64_t B_st1, int64_t B_st2, float *C_a, float *C_al,
    int64_t C_o, int64_t C_s0, int64_t C_s1, int64_t C_s2, int64_t C_st0,
    int64_t C_st1, int64_t C_st2, float *D_a, float *D_al, int64_t D_o,
    int64_t D_s0, int64_t D_s1, int64_t D_s2, int64_t D_st0, int64_t D_st1,
    int64_t D_st2);

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
  const int64_t B = 1, S = 4, D = 4;
  const int64_t n = B * S * D;
  printf("3D mul test: D = A * B * C  [%lld,%lld,%lld]\n", B, S, D);

  std::vector<float> h_A(n), h_B(n), h_C(n), h_D(n, 0), h_ref(n);
  for (int64_t i = 0; i < n; ++i) {
    h_A[i] = float(i + 1) * 0.1f;
    h_B[i] = float(i + 1) * 0.2f;
    h_C[i] = 0.5f;
    h_ref[i] = h_A[i] * h_B[i] * h_C[i];
  }

  float *d_A, *d_B, *d_C, *d_D;
  HIP_CHECK(hipMalloc(&d_A, n * 4));
  HIP_CHECK(hipMalloc(&d_B, n * 4));
  HIP_CHECK(hipMalloc(&d_C, n * 4));
  HIP_CHECK(hipMalloc(&d_D, n * 4));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), n * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), n * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_C, h_C.data(), n * 4, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_D, 0, n * 4));

  printf("Running on GPU...\n");
  two_muls(d_A, d_A, 0, B, S, D, S * D, D, 1, d_B, d_B, 0, B, S, D, S * D, D, 1,
           d_C, d_C, 0, B, S, D, S * D, D, 1, d_D, d_D, 0, B, S, D, S * D, D,
           1);

  HIP_CHECK(hipMemcpy(h_D.data(), d_D, n * 4, hipMemcpyDeviceToHost));

  float mx = 0;
  for (int64_t i = 0; i < n; ++i) {
    float d = std::fabs(h_D[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-4f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipFree(d_D));
  return (mx < 1e-4f) ? 0 : 1;
}
