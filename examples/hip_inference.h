/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Shared header for example drivers.
 * Defines the C ABI types and imports the three DLL entry points generated
 * by hip-compiler: inference_init, inference_compute, inference_cleanup.
 */
#ifndef HIP_INFERENCE_H
#define HIP_INFERENCE_H

#include <cstddef>
#include <cstdint>

struct tensor_t {
  void *data;
  int64_t *shape;
  size_t rank;
  size_t element_size; // bytes per element (4 = float32)
};

struct span_t {
  tensor_t *data;
  size_t count;
};

extern "C" __declspec(dllimport) int inference_init(void **out_state);
extern "C" __declspec(dllimport) int inference_compute(void *state,
                                                       span_t *inputs,
                                                       span_t *outputs);
extern "C" __declspec(dllimport) int inference_cleanup(void *state);

#endif // HIP_INFERENCE_H
