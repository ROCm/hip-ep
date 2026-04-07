/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// Note: These simple wrapper functions don't need cleanup, so we use a local
// HIP_CHECK that returns directly instead of the goto-based HIP_CHECK_GOTO
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// HIP memory wrappers

int wrap_hipMalloc(void **ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int wrap_hipFree(void *ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipStreamSynchronize(void *stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipDeviceSynchronize(void) {
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_hipDeviceSynchronize: %s (code=%d)\n",
            hipGetErrorString(err), static_cast<int>(err));
    return static_cast<int>(err);
  }
  return 0;
}

int wrap_hipGetLastError(void) {
  return static_cast<int>(hipGetLastError());
}

int wrap_hipPeekAtLastError(void) {
  return static_cast<int>(hipPeekAtLastError());
}

int wrap_hipMemGetInfo(size_t *free_bytes, size_t *total_bytes) {
  HIP_CHECK(hipMemGetInfo(free_bytes, total_bytes));
  return 0;
}
