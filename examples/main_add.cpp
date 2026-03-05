/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_add.cpp - Two-add test via inference interface -----------------===//
//
// D[B,S,D] = A + B + C   (two chained adds with 3D tensors)
//
//===----------------------------------------------------------------------===//

#include "hip_inference.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const int64_t B = 1, S = 4, D = 4;
  const int64_t n = B * S * D;
  printf("3D add test: D = A + B + C  [%lld,%lld,%lld]\n", B, S, D);

  std::vector<float> h_A(n), h_B(n), h_C(n), h_D(n, 0), h_ref(n);
  for (int64_t i = 0; i < n; ++i) {
    h_A[i] = float(i) * 0.1f;
    h_B[i] = float(i) * 0.2f;
    h_C[i] = float(i) * 0.3f;
    h_ref[i] = h_A[i] + h_B[i] + h_C[i];
  }

  void *state = nullptr;
  inference_init(&state);

  int64_t shape[] = {B, S, D};
  tensor_t inputs[] = {
      {h_A.data(), shape, 3, sizeof(float)},
      {h_B.data(), shape, 3, sizeof(float)},
      {h_C.data(), shape, 3, sizeof(float)},
  };
  tensor_t outputs[] = {
      {h_D.data(), shape, 3, sizeof(float)},
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
    float d = std::fabs(h_D[i] - h_ref[i]);
    if (d > mx)
      mx = d;
  }
  printf("Max diff: %e  %s\n", mx, mx < 1e-4f ? "PASS" : "FAIL");

  inference_cleanup(state);
  return (mx < 1e-4f) ? 0 : 1;
}
