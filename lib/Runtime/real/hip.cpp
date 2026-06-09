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

// Direct device alloc/free entrypoints for the bufferization-produced
// memref.alloc / memref.dealloc lowering in MemoryLowering.cpp (kHipMalloc /
// kHipFree). Static-shape models pool every buffer into the runtime byte pool,
// so a model.dll never references these symbols. They are first reached on
// dynamic-shape models, where a runtime-sized buffer (e.g. a dynamic-bound
// Range's element buffer) escapes pooling and lowers to a direct
// hip_device_malloc/free call; without these definitions lld-link fails with
// "undefined symbol: hip_device_free" when linking such a model.dll.
//
// Failure is reported via the return value (nullptr) / a no-op, not HIP_CHECK,
// because aborting the host process from inside a loaded model DLL is hostile;
// every caller null-checks the result.
extern "C" void *hip_device_malloc(int64_t size) {
  void *ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, size);
  if (err != hipSuccess) {
    fprintf(stderr, "hip_device_malloc(%lld) failed: %s\n", (long long)size,
            hipGetErrorString(err));
    return nullptr;
  }
  return ptr;
}

extern "C" void hip_device_free(void *ptr) {
  if (!ptr)
    return;
  hipError_t err = hipFree(ptr);
  if (err != hipSuccess) {
    fprintf(stderr, "hip_device_free(%p) failed: %s\n", ptr,
            hipGetErrorString(err));
  }
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
