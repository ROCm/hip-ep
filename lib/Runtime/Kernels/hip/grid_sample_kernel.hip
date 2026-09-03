/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX GridSample (4-D NCHW). One thread per output element.
 *
 * Grid last dim is (x, y) in [-1, 1]. Coordinate unnormalization:
 *   align_corners=1: ((g + 1) * (size - 1)) / 2
 *   align_corners=0: ((g + 1) * size - 1) / 2
 *
 * mode: 0 nearest, 1 bilinear
 * padding_mode: 0 zeros, 1 border, 2 reflection over [0, size-1]
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace {

template <typename T> __device__ inline float to_float(T v);
template <> __device__ inline float to_float<float>(float v) { return v; }
template <> __device__ inline float to_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ inline float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ inline float to_float<double>(double v) {
  return static_cast<float>(v);
}

template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline float from_float<float>(float v) { return v; }
template <> __device__ inline __half from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline __hip_bfloat16 from_float<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ inline double from_float<double>(float v) {
  return static_cast<double>(v);
}

__device__ inline float unnormalize(float coord, int size, int align_corners) {
  if (size <= 1)
    return 0.0f;
  if (align_corners)
    return ((coord + 1.0f) * static_cast<float>(size - 1)) * 0.5f;
  return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
}

__device__ inline int reflect_idx(int x, int size) {
  if (size <= 1)
    return 0;
  int period = 2 * size - 2;
  int m = x % period;
  if (m < 0)
    m += period;
  if (m >= size)
    m = period - m;
  return m;
}

template <typename T>
__device__ inline float sample_at(const T *__restrict__ img, int h, int w,
                                  int H, int W, int padding_mode) {
  if (padding_mode == 0) {
    if (h < 0 || h >= H || w < 0 || w >= W)
      return 0.0f;
    return to_float<T>(img[static_cast<int64_t>(h) * W + w]);
  }
  if (padding_mode == 1) {
    if (h < 0)
      h = 0;
    if (h >= H)
      h = H - 1;
    if (w < 0)
      w = 0;
    if (w >= W)
      w = W - 1;
    return to_float<T>(img[static_cast<int64_t>(h) * W + w]);
  }
  h = reflect_idx(h, H);
  w = reflect_idx(w, W);
  return to_float<T>(img[static_cast<int64_t>(h) * W + w]);
}

template <typename T>
__global__ void grid_sample_kernel(const T *__restrict__ input,
                                   const T *__restrict__ grid,
                                   T *__restrict__ output, int64_t N, int64_t C,
                                   int64_t inH, int64_t inW, int64_t outH,
                                   int64_t outW, int mode, int padding_mode,
                                   int align_corners, int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t plane = outH * outW;
  int64_t batch_stride = C * plane;
  int64_t n = tid / batch_stride;
  int64_t rem = tid - n * batch_stride;
  int64_t c = rem / plane;
  rem -= c * plane;
  int64_t oh = rem / outW;
  int64_t ow = rem - oh * outW;

  const T *g = grid + ((n * outH + oh) * outW + ow) * 2;
  float gx = to_float<T>(g[0]);
  float gy = to_float<T>(g[1]);
  float x = unnormalize(gx, static_cast<int>(inW), align_corners);
  float y = unnormalize(gy, static_cast<int>(inH), align_corners);

  const T *img =
      input + (n * C + c) * (inH * inW);
  int H = static_cast<int>(inH);
  int W = static_cast<int>(inW);
  float val;
  if (mode == 0) {
    int iy = static_cast<int>(floorf(y + 0.5f));
    int ix = static_cast<int>(floorf(x + 0.5f));
    val = sample_at(img, iy, ix, H, W, padding_mode);
  } else {
    int y0 = static_cast<int>(floorf(y));
    int x0 = static_cast<int>(floorf(x));
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    float dy = y - static_cast<float>(y0);
    float dx = x - static_cast<float>(x0);
    float v00 = sample_at(img, y0, x0, H, W, padding_mode);
    float v01 = sample_at(img, y0, x1, H, W, padding_mode);
    float v10 = sample_at(img, y1, x0, H, W, padding_mode);
    float v11 = sample_at(img, y1, x1, H, W, padding_mode);
    val = v00 * (1.0f - dx) * (1.0f - dy) + v01 * dx * (1.0f - dy) +
          v10 * (1.0f - dx) * dy + v11 * dx * dy;
  }
  output[tid] = from_float<T>(val);
}

} // namespace

extern "C" int hip_grid_sample(void *stream, const void *input, const void *grid,
                               void *output, int64_t n, int64_t c, int64_t in_h,
                               int64_t in_w, int64_t out_h, int64_t out_w,
                               int mode, int padding_mode, int align_corners,
                               int hip_dtype) {
  if (n <= 0 || c <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0)
    return 0;
  if (mode != 0 && mode != 1) {
    fprintf(stderr, "[custom_kernels] hip_grid_sample: bad mode=%d\n", mode);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  int64_t total = n * c * out_h * out_w;
  dim3 block(256);
  dim3 grid_dim(static_cast<unsigned int>((total + 255) / 256));

  (void)hipGetLastError();
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_grid_sample: dtype=%d N=%lld C=%lld in=%lldx%lld "
      "out=%lldx%lld mode=%d pad=%d align=%d\n",
      hip_dtype, (long long)n, (long long)c, (long long)in_h, (long long)in_w,
      (long long)out_h, (long long)out_w, mode, padding_mode, align_corners);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<__half>), grid_dim,
                       block, 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<const __half *>(grid),
                       static_cast<__half *>(output), n, c, in_h, in_w, out_h,
                       out_w, mode, padding_mode, align_corners, total);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<__hip_bfloat16>),
                       grid_dim, block, 0, hip_stream,
                       static_cast<const __hip_bfloat16 *>(input),
                       static_cast<const __hip_bfloat16 *>(grid),
                       static_cast<__hip_bfloat16 *>(output), n, c, in_h, in_w,
                       out_h, out_w, mode, padding_mode, align_corners, total);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<float>), grid_dim,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(grid),
                       static_cast<float *>(output), n, c, in_h, in_w, out_h,
                       out_w, mode, padding_mode, align_corners, total);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<double>), grid_dim,
                       block, 0, hip_stream,
                       static_cast<const double *>(input),
                       static_cast<const double *>(grid),
                       static_cast<double *>(output), n, c, in_h, in_w, out_h,
                       out_w, mode, padding_mode, align_corners, total);
  } else {
    fprintf(stderr, "[custom_kernels] hip_grid_sample: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_grid_sample launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
