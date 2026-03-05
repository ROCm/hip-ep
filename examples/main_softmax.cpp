/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_softmax.cpp - Two-softmax test via inference interface ---------===//
//
// B[B,S,D] = softmax(softmax(A[B,S,D]))
//
//===----------------------------------------------------------------------===//

#include "hip_inference.h"
#include <cmath>
#include <cstdio>
#include <vector>

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

int main() {
  const int64_t Bat = 1, S = 4, D = 8;
  const int64_t n = Bat * S * D;
  const int64_t rows = Bat * S;
  printf("3D softmax test: B = softmax(softmax(A))  [%lld,%lld,%lld]\n", Bat, S,
         D);

  std::vector<float> h_A(n), h_B(n, 0), h_ref(n);
  for (int64_t i = 0; i < n; ++i)
    h_A[i] = float(i % 7) * 0.5f - 1.5f;

  std::vector<float> h_tmp(n);
  cpu_softmax(h_A.data(), h_tmp.data(), rows, D);
  cpu_softmax(h_tmp.data(), h_ref.data(), rows, D);

  void *state = nullptr;
  inference_init(&state);

  int64_t in_shape[] = {Bat, S, D};
  int64_t out_shape[] = {Bat, S, D};
  tensor_t inputs[] = {
      {h_A.data(), in_shape, 3, sizeof(float)},
  };
  tensor_t outputs[] = {
      {h_B.data(), out_shape, 3, sizeof(float)},
  };
  span_t in_span = {inputs, 1};
  span_t out_span = {outputs, 1};

  printf("Running on GPU...\n");
  int ret = inference_compute(state, &in_span, &out_span);
  if (ret != 0) {
    fprintf(stderr, "inference_compute failed: %d\n", ret);
    inference_cleanup(state);
    return 1;
  }

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

  inference_cleanup(state);
  return (mx < 1e-4f) ? 0 : 1;
}
