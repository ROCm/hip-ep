/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Range: write `start + i*delta` for i in [0, n).  The full ONNX
// op accepts arbitrary numeric scalars; this kernel only handles i64
// and f32 (the two element types Kokoro uses).  The host launcher
// caps grid.x at 65535 and tiles across grid.y for very long ranges.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

template <typename T>
__global__ void range_kernel(T start, T delta, int64_t n, T *__restrict__ out) {
  int64_t flat_y = (int64_t)blockIdx.y * gridDim.x * blockDim.x;
  int64_t flat_x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t i = flat_y + flat_x;
  if (i >= n)
    return;
  out[i] = start + (T)i * delta;
}

// Host-side fallback: compute the range on the CPU and upload via hipMemcpy.
// Uses synchronous hipMemcpy only -- hipMemcpyAsync on the Morphizen bridge
// stream triggers "bad variant access" crashes.
//
// When capacity > n, fills positions [n..capacity) with `sentinel` so that
// downstream consumers (e.g. alignment comparisons) don't see stale data.
template <typename T>
int host_range_fallback(hipStream_t /*stream*/, T start, T delta, int64_t n,
                        T *dev_out, int64_t capacity, T sentinel) {
  int64_t total = (capacity > n) ? capacity : n;
  T *host_buf = static_cast<T *>(malloc(total * sizeof(T)));
  if (!host_buf)
    return -1;
  for (int64_t i = 0; i < n; i++)
    host_buf[i] = start + static_cast<T>(i) * delta;
  for (int64_t i = n; i < total; i++)
    host_buf[i] = sentinel;
  (void)hipGetLastError();
  hipError_t err =
      hipMemcpy(dev_out, host_buf, total * sizeof(T), hipMemcpyHostToDevice);
  free(host_buf);
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] host_range_fallback: hipMemcpy H2D failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

template <typename T>
int launch_range(hipStream_t stream, T start, T delta, int64_t n, T *out,
                 int64_t capacity = 0, T sentinel = T(0)) {
  if (n <= 0)
    return 0;
  (void)hipGetLastError(); // clear stale errors
  constexpr int kBlock = 128;
  int64_t blocks = (n + kBlock - 1) / kBlock;
  int64_t grid_x = (blocks > 65535) ? 65535 : blocks;
  int64_t grid_y = (blocks + 65535 - 1) / 65535;
  hipLaunchKernelGGL((range_kernel<T>),
                     dim3((unsigned)grid_x, (unsigned)grid_y, 1),
                     dim3(kBlock, 1, 1), 0, stream, start, delta, n, out);
  int rc = static_cast<int>(hipGetLastError());
  if (rc != 0) {
    fprintf(stderr,
            "[custom_kernels] range GPU launch failed (rc=%d), "
            "using host fallback for n=%lld elements\n",
            rc, (long long)n);
    return host_range_fallback<T>(stream, start, delta, n, out, capacity,
                                  sentinel);
  }
  // GPU launch OK. If capacity > n, zero-fill remaining via hipMemset.
  if (capacity > n) {
    (void)hipMemset(out + n, 0xFF, (capacity - n) * sizeof(T));
  }
  return 0;
}

} // namespace

extern "C" int hip_range_i64(void *stream, int64_t start, int64_t delta,
                              int64_t n, void *output) {
  return launch_range<int64_t>(static_cast<hipStream_t>(stream), start, delta,
                                 n, static_cast<int64_t *>(output));
}

extern "C" int hip_range_f32(void *stream, float start, float delta, int64_t n,
                              void *output) {
  return launch_range<float>(static_cast<hipStream_t>(stream), start, delta, n,
                              static_cast<float *>(output));
}

// Variant that takes GPU pointers to start/limit/delta scalars.  Reads
// them back to host via hipMemcpy, computes n = max((limit - start) /
// delta, 0), then dispatches to the appropriate launch_range<>.  Used
// when MLIR can't extract the scalar at compile time (e.g. start /
// limit are runtime-computed values living in device memory).
extern "C" int hip_range_dyn(void *stream, const void *start_dev,
                              const void *limit_dev, const void *delta_dev,
                              void *output, int64_t output_capacity,
                              int hip_dtype) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // hipMemcpy(D2H) is synchronous -- it implicitly waits for all
  // pending work that touches these device addresses to complete.
  // No explicit sync is needed; avoid hipDeviceSynchronize which
  // surfaces deferred errors and can poison subsequent launches.

  // Read scalars from device memory.
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
    if (d == 0) d = 1;
    int64_t n = (l - s) / d;
    if (n < 0) n = 0;
    if (n > output_capacity) n = output_capacity;
    fprintf(stderr,
        "[custom_kernels] hip_range_dyn: i64 s=%lld l=%lld d=%lld n=%lld cap=%lld\n",
        (long long)s, (long long)l, (long long)d, (long long)n,
        (long long)output_capacity);
    // Sentinel = -1: fails all `>= 0` alignment comparisons for unused slots.
    return launch_range<int64_t>(hip_stream, s, d, n,
                                 static_cast<int64_t *>(output),
                                 output_capacity, int64_t(-1));
  }
  if (hip_dtype == /*HIP_DTYPE_FLOAT32*/ 0) {
    float s = 0, l = 0, d = 1;
    hipMemcpy(&s, start_dev, 4, hipMemcpyDeviceToHost);
    hipMemcpy(&l, limit_dev, 4, hipMemcpyDeviceToHost);
    hipMemcpy(&d, delta_dev, 4, hipMemcpyDeviceToHost);
    if (d == 0.f) d = 1.f;
    int64_t n = (int64_t)((l - s) / d);
    if (n < 0) n = 0;
    if (n > output_capacity) n = output_capacity;
    return launch_range<float>(hip_stream, s, d, n,
                                 static_cast<float *>(output),
                                 output_capacity, -1.0f);
  }
  fprintf(stderr, "[custom_kernels] hip_range_dyn: unsupported dtype %d\n",
          hip_dtype);
  return -1;
}
