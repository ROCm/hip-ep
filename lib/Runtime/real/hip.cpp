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

// MemoryLowering.cpp emits llvm.call @hip_device_malloc(size) -> ptr for
// hip.alloc (and the fallback memref.alloc path).  Static-shape models pool
// every alloc into the runtime byte pool so this entrypoint is never hit;
// PR #254 first exercises it on dynamic-Range models (alloc(seq*batch) for
// position_ids), and lld-link fails with "undefined symbol: hip_device_free"
// at the EP fallback point if the runtime doesn't ship these symbols.
//
// Signature: (i64) -> ptr.  Returns nullptr on failure (callers check), no
// HIP_CHECK that would terminate the host process from inside the model DLL.
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
