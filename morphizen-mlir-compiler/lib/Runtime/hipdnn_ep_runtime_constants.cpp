/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// Internal runtime state structure (must match state.cpp)
struct RuntimeState {
  void* stream;
  void* miopen_handle;
  void* hipblas_handle;
  void** gpu_constants;
  size_t num_constants;
};

// Error checking macro for mock runtime (no-op)
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    (void)(cmd);                                                               \
  } while (0)

// Constant management implementation

int hipdnn_ep_constant_upload(RuntimeState* state, int64_t index,
                              const void* data, int64_t size) {
  if (!state || !data || size <= 0) {
    fprintf(stderr, "Invalid arguments to hipdnn_ep_constant_upload\n");
    return -1;
  }

  // Validate index range
  if (index < 0 || (size_t)index >= state->num_constants) {
    fprintf(stderr, "Constant index %lld out of range [0, %zu)\n",
            (long long)index, state->num_constants);
    return -1;
  }

  // Check if constant already exists (shouldn't happen, but defensive)
  if (state->gpu_constants[index] != nullptr) {
    fprintf(stderr, "Constant %lld already uploaded\n", (long long)index);
    return -1;
  }

  // Allocate GPU memory
  void* gpu_ptr = nullptr;
  HIP_CHECK(hipMalloc(&gpu_ptr, size));

  // Copy data from DLL .data section to GPU
  HIP_CHECK(hipMemcpyAsync(gpu_ptr, data, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(state->stream)));

  // Store in array
  state->gpu_constants[index] = gpu_ptr;

  return 0;
}

void* hipdnn_ep_constant_get(RuntimeState* state, int64_t index) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state\n");
    return nullptr;
  }

  // Validate index range
  if (index < 0 || (size_t)index >= state->num_constants) {
    fprintf(stderr, "Constant index %lld out of range [0, %zu)\n",
            (long long)index, state->num_constants);
    return nullptr;
  }

  return state->gpu_constants[index];
}

int hipdnn_ep_constant_release(RuntimeState* state, int64_t index) {
  if (!state) {
    fprintf(stderr, "Invalid runtime state\n");
    return -1;
  }

  // Validate index range
  if (index < 0 || (size_t)index >= state->num_constants) {
    fprintf(stderr, "Constant index %lld out of range [0, %zu)\n",
            (long long)index, state->num_constants);
    return -1;
  }

  // Check if constant exists
  if (state->gpu_constants[index] == nullptr) {
    fprintf(stderr, "Constant %lld not found\n", (long long)index);
    return -1;
  }

  // Free GPU memory
  HIP_CHECK(hipFree(state->gpu_constants[index]));

  // Clear array entry
  state->gpu_constants[index] = nullptr;

  return 0;
}
