/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Range: out[i] = start + i*delta for i in [0, n).
//
// On gfx1201, hipLaunchKernelGGL for kernels compiled in this TU (or
// even kernels from other TUs called within this context) returns
// hipErrorInvalidValue due to stale HIP runtime error state from
// NAN_TRACE's hipMemcpy(D2H) calls.  Instead, the range values are
// generated and uploaded via hipMemcpy(H2D) — the same DMA mechanism
// used for model weight loading and constant tensor initialization.
// hipMemcpy(H2D) is handled by the GPU's copy engine and implicitly
// drains stale async errors, leaving a clean state for downstream
// kernel launches.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" int hip_range_i64(void *stream, int64_t start, int64_t delta,
                              int64_t n, void *output) {
  if (n <= 0)
    return 0;
  int64_t *h = static_cast<int64_t *>(malloc(n * sizeof(int64_t)));
  if (!h)
    return -1;
  for (int64_t i = 0; i < n; i++)
    h[i] = start + i * delta;
  (void)hipGetLastError();
  hipError_t err =
      hipMemcpy(output, h, n * sizeof(int64_t), hipMemcpyHostToDevice);
  free(h);
  (void)hipGetLastError();
  return static_cast<int>(err);
}

extern "C" int hip_range_f32(void *stream, float start, float delta, int64_t n,
                              void *output) {
  if (n <= 0)
    return 0;
  float *h = static_cast<float *>(malloc(n * sizeof(float)));
  if (!h)
    return -1;
  for (int64_t i = 0; i < n; i++)
    h[i] = start + static_cast<float>(i) * delta;
  (void)hipGetLastError();
  hipError_t err =
      hipMemcpy(output, h, n * sizeof(float), hipMemcpyHostToDevice);
  free(h);
  (void)hipGetLastError();
  return static_cast<int>(err);
}

extern "C" int hip_range_dyn(void *stream, const void *start_dev,
                              const void *limit_dev, const void *delta_dev,
                              void *output, int64_t output_capacity,
                              int hip_dtype) {
  if (hip_dtype == /*HIP_DTYPE_INT64*/ 2 ||
      hip_dtype == /*HIP_DTYPE_INT32*/ 3) {
    int64_t s = 0, l = 0, d = 1;
    if (hip_dtype == 2) {
      hipMemcpy(&s, start_dev, 8, hipMemcpyDeviceToHost);
      hipMemcpy(&l, limit_dev, 8, hipMemcpyDeviceToHost);
      hipMemcpy(&d, delta_dev, 8, hipMemcpyDeviceToHost);
    } else {
      int32_t s32 = 0, l32 = 0, d32 = 1;
      hipMemcpy(&s32, start_dev, 4, hipMemcpyDeviceToHost);
      hipMemcpy(&l32, limit_dev, 4, hipMemcpyDeviceToHost);
      hipMemcpy(&d32, delta_dev, 4, hipMemcpyDeviceToHost);
      s = s32; l = l32; d = d32;
    }
    (void)hipGetLastError();
    if (d == 0) d = 1;
    int64_t n = (l - s) / d;
    if (n < 0) n = 0;
    if (n > output_capacity) n = output_capacity;

    int rc = hip_range_i64(stream, s, d, n, output);
    if (rc != 0) return rc;
    if (output_capacity > n) {
      int64_t fill = output_capacity - n;
      int64_t *sentinel = static_cast<int64_t *>(
          malloc(fill * sizeof(int64_t)));
      if (sentinel) {
        for (int64_t i = 0; i < fill; i++)
          sentinel[i] = -1;
        hipMemcpy(static_cast<int64_t *>(output) + n,
                  sentinel, fill * sizeof(int64_t), hipMemcpyHostToDevice);
        free(sentinel);
      }
    }
    return 0;
  }
  if (hip_dtype == /*HIP_DTYPE_FLOAT32*/ 0) {
    float s = 0, l = 0, d = 1;
    hipMemcpy(&s, start_dev, 4, hipMemcpyDeviceToHost);
    hipMemcpy(&l, limit_dev, 4, hipMemcpyDeviceToHost);
    hipMemcpy(&d, delta_dev, 4, hipMemcpyDeviceToHost);
    (void)hipGetLastError();
    if (d == 0.f) d = 1.f;
    int64_t n = (int64_t)((l - s) / d);
    if (n < 0) n = 0;
    if (n > output_capacity) n = output_capacity;
    return hip_range_f32(stream, s, d, n, output);
  }
  fprintf(stderr, "[custom_kernels] hip_range_dyn: unsupported dtype %d\n",
          hip_dtype);
  return -1;
}
