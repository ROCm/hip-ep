/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_rms_norm.cpp - Two-RMSNorm test via inference interface --------===//
//
// B[B,S,D] = RMSNorm(RMSNorm(A[B,S,D], W0[D]), W1[D])
//
//===----------------------------------------------------------------------===//

#include "hip_inference.h"
#include <cmath>
#include <cstdio>
#include <vector>

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

  void *state = nullptr;
  inference_init(&state);

  int64_t in_shape[] = {Bat, S, D};
  int64_t w_shape[] = {D};
  int64_t out_shape[] = {Bat, S, D};
  tensor_t inputs[] = {
      {h_A.data(), in_shape, 3, sizeof(float)},
      {h_W0.data(), w_shape, 1, sizeof(float)},
      {h_W1.data(), w_shape, 1, sizeof(float)},
  };
  tensor_t outputs[] = {
      {h_B.data(), out_shape, 3, sizeof(float)},
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
  for (int64_t i = 0; i < n; ++i) {
    float d = std::fabs(h_B[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-3f ? "PASS" : "FAIL");

  inference_cleanup(state);
  return (mx < 1e-3f) ? 0 : 1;
}
