/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Range: write `start + i*delta` for i in [0, n).  The full ONNX
// op accepts arbitrary numeric scalars; this kernel only handles i64
// and f32 (the two element types Kokoro uses).  The host launcher
// caps grid.x at 65535 and tiles across grid.y for very long ranges.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

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

template <typename T>
int launch_range(hipStream_t stream, T start, T delta, int64_t n, T *out) {
  if (n <= 0)
    return 0;
  constexpr int kBlock = 128;
  int64_t blocks = (n + kBlock - 1) / kBlock;
  int64_t grid_x = (blocks > 65535) ? 65535 : blocks;
  int64_t grid_y = (blocks + 65535 - 1) / 65535;
  hipLaunchKernelGGL((range_kernel<T>),
                     dim3((unsigned)grid_x, (unsigned)grid_y, 1),
                     dim3(kBlock, 1, 1), 0, stream, start, delta, n, out);
  return static_cast<int>(hipGetLastError());
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
  // Sync to ensure any prior writes to the scalar buffers complete.
  hipError_t syn = hipStreamSynchronize(hip_stream);
  if (syn != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_range_dyn: hipStreamSynchronize: %s\n",
            hipGetErrorString(syn));
    return static_cast<int>(syn);
  }
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
    return launch_range<int64_t>(hip_stream, s, d, n,
                                   static_cast<int64_t *>(output));
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
                                 static_cast<float *>(output));
  }
  fprintf(stderr, "[custom_kernels] hip_range_dyn: unsupported dtype %d\n",
          hip_dtype);
  return -1;
}
