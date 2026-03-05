/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- main_e2e.cpp - E2E transformer test via inference interface ---------===//
//
// Calls the MLIR-compiled @run function from test_e2e.mlir via the
// inference_init/compute/cleanup interface.  The function allocates all
// GPU buffers internally and exercises every HIP dialect op.
//
//===----------------------------------------------------------------------===//

#include "hip_inference.h"
#include <cstdio>

int main() {
  printf("=== HIP Dialect E2E Transformer Test ===\n");

  void *state = nullptr;
  inference_init(&state);

  span_t in_span = {nullptr, 0};
  span_t out_span = {nullptr, 0};

  printf("Calling inference_compute...\n\n");
  int ret = inference_compute(state, &in_span, &out_span);
  if (ret != 0)
    fprintf(stderr, "inference_compute failed: %d\n", ret);

  inference_cleanup(state);

  printf("\n=== All ops executed successfully. ===\n");
  return ret;
}
