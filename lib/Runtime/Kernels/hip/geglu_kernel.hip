/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GEGLU: output[r, j] = input[r, j] * gelu(input[r, F + j]) for j < F, with
// the exact (erf) GELU. The value and gate halves are the two halves of one
// [rows, 2F] row, as diffusers' feed-forward chunks its projection, so the
// kernel reads each projection element once and never materialises either
// half. Math in fp32.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

template <typename T> struct GegluTraits;

template <> struct GegluTraits<float> {
  __device__ static float load(float v) { return v; }
  __device__ static float store(float v) { return v; }
};

template <> struct GegluTraits<__half> {
  __device__ static float load(__half v) { return __half2float(v); }
  __device__ static __half store(float v) { return __float2half(v); }
};

__device__ __forceinline__ float gegluValue(float x, float g) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  return x * (0.5f * g * (1.0f + erff(g * kInvSqrt2)));
}

// One thread per 16-byte vector of the output. Rows are walked by the grid's
// y extent so the column index needs no division.
template <typename T, int V>
__global__ __launch_bounds__(256) void gegluVecKernel(
    const T *__restrict__ input, T *__restrict__ output, int64_t rows,
    int64_t halfCols) {
  union Vec {
    uint4 raw;
    T e[V];
  };
  const int64_t c = (static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x) * V;
  if (c >= halfCols)
    return;
  for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
    const T *row = input + r * 2 * halfCols;
    Vec x, g, y;
    x.raw = *reinterpret_cast<const uint4 *>(row + c);
    g.raw = *reinterpret_cast<const uint4 *>(row + halfCols + c);
#pragma unroll
    for (int i = 0; i < V; ++i)
      y.e[i] = GegluTraits<T>::store(gegluValue(GegluTraits<T>::load(x.e[i]),
                                                GegluTraits<T>::load(g.e[i])));
    *reinterpret_cast<uint4 *>(output + r * halfCols + c) = y.raw;
  }
}

template <typename T>
__global__ __launch_bounds__(256) void gegluKernel(const T *__restrict__ input,
                                                   T *__restrict__ output,
                                                   int64_t rows,
                                                   int64_t halfCols) {
  const int64_t c = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= halfCols)
    return;
  for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
    const T *row = input + r * 2 * halfCols;
    output[r * halfCols + c] = GegluTraits<T>::store(gegluValue(
        GegluTraits<T>::load(row[c]), GegluTraits<T>::load(row[halfCols + c])));
  }
}

template <typename T, int V>
void launchGeglu(hipStream_t stream, const void *input, void *output,
                 int64_t rows, int64_t halfCols) {
  constexpr int kBlock = 256;
  constexpr int64_t kMaxGridY = 65535;
  const unsigned gridY =
      static_cast<unsigned>(rows < kMaxGridY ? rows : kMaxGridY);
  const T *in = static_cast<const T *>(input);
  T *out = static_cast<T *>(output);
  // A 16-byte vector per thread needs every row start 16-byte aligned, which
  // the half width being a multiple of V gives once the bases are.
  const bool vec = halfCols % V == 0 &&
                   reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(output) % 16 == 0;
  if (vec) {
    const int64_t vecs = halfCols / V;
    const dim3 grid(static_cast<unsigned>((vecs + kBlock - 1) / kBlock), gridY);
    gegluVecKernel<T, V><<<grid, kBlock, 0, stream>>>(in, out, rows, halfCols);
    return;
  }
  const dim3 grid(static_cast<unsigned>((halfCols + kBlock - 1) / kBlock),
                  gridY);
  gegluKernel<T><<<grid, kBlock, 0, stream>>>(in, out, rows, halfCols);
}

} // namespace

extern "C" int hip_geglu(void *stream, const void *input, void *output,
                         int64_t rows, int64_t half_cols, int hip_dtype) {
  if (rows <= 0 || half_cols <= 0)
    return 0;
  if (!input || !output)
    return -1;
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    launchGeglu<__half, 8>(s, input, output, rows, half_cols);
    break;
  case HIP_DTYPE_FLOAT32:
    launchGeglu<float, 4>(s, input, output, rows, half_cols);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_geglu: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_geglu: dtype=%d rows=%lld half_cols=%lld\n",
      hip_dtype, (long long)rows, (long long)half_cols);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_geglu launch failed: %s\n",
            hipGetErrorString(err));
  return static_cast<int>(err);
}
