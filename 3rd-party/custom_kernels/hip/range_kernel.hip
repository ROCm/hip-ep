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
