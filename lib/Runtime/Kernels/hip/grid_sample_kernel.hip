/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX GridSample (4-D NCHW). One thread per output pixel: the grid read,
 * the source coordinates and the padded corner offsets do not depend on the
 * channel, so each thread computes them once and then samples every channel.
 *
 * Grid last dim is (x, y) in [-1, 1]. Coordinate unnormalization:
 *   align_corners=1: ((g + 1) * (size - 1)) / 2
 *   align_corners=0: ((g + 1) * size - 1) / 2
 *
 * mode: 0 nearest (round half to even), 1 bilinear
 * padding_mode: 0 zeros, 1 border, 2 reflection over [0, size-1] with
 * align_corners, else over the pixel edges [-0.5, size-0.5]
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

// Reflect a source coordinate into [lo, hi]: [0, size - 1] with align_corners,
// else the pixel edges [-0.5, size - 0.5], as the ONNX reference does.
__device__ inline float reflect_coord(float x, float lo, float hi) {
  float range = hi - lo;
  if (range <= 0.0f)
    return lo;
  if (x < lo) {
    float dx = lo - x;
    int n = static_cast<int>(dx / range);
    float r = dx - static_cast<float>(n) * range;
    return (n % 2 == 0) ? lo + r : hi - r;
  }
  if (x > hi) {
    float dx = x - hi;
    int n = static_cast<int>(dx / range);
    float r = dx - static_cast<float>(n) * range;
    return (n % 2 == 0) ? hi - r : lo + r;
  }
  return x;
}

__device__ inline int reflect_index(int i, int size, float lo, float hi) {
  int r = static_cast<int>(reflect_coord(static_cast<float>(i), lo, hi));
  return r < 0 ? 0 : (r >= size ? size - 1 : r);
}

// Offset of source pixel (h, w) within a plane after padding, or -1 when zero
// padding puts it outside.
// `lo` is the reflection bound below 0 (0 or -0.5); the upper bound is
// size - 1 - lo.
template <int PAD>
__device__ inline int64_t padded_offset(int h, int w, int H, int W, float lo) {
  if (PAD == 0) {
    if (h < 0 || h >= H || w < 0 || w >= W)
      return -1;
  } else if (PAD == 1) {
    h = h < 0 ? 0 : (h >= H ? H - 1 : h);
    w = w < 0 ? 0 : (w >= W ? W - 1 : w);
  } else {
    h = reflect_index(h, H, lo, static_cast<float>(H - 1) - lo);
    w = reflect_index(w, W, lo, static_cast<float>(W - 1) - lo);
  }
  return static_cast<int64_t>(h) * W + w;
}

template <typename T>
__device__ inline float load_or_zero(const T *__restrict__ img, int64_t off) {
  return off < 0 ? 0.0f : to_float<T>(img[off]);
}

// One thread per output pixel (n, oh, ow); loops over the C channels.
template <typename T, int MODE, int PAD>
__global__ void grid_sample_kernel(const T *__restrict__ input,
                                   const T *__restrict__ grid,
                                   T *__restrict__ output, int64_t C,
                                   int64_t inH, int64_t inW, int64_t outH,
                                   int64_t outW, int align_corners,
                                   int64_t pixels) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= pixels)
    return;

  int64_t out_plane = outH * outW;
  int64_t n = tid / out_plane;
  int64_t pix = tid - n * out_plane;

  const T *g = grid + tid * 2;
  float gx = to_float<T>(g[0]);
  float gy = to_float<T>(g[1]);
  int H = static_cast<int>(inH);
  int W = static_cast<int>(inW);
  float x = unnormalize(gx, W, align_corners);
  float y = unnormalize(gy, H, align_corners);
  const float lo = align_corners ? 0.0f : -0.5f;
  if (PAD == 2) {
    x = reflect_coord(x, lo, static_cast<float>(W - 1) - lo);
    y = reflect_coord(y, lo, static_cast<float>(H - 1) - lo);
  }

  int64_t in_plane = inH * inW;
  const T *img = input + n * C * in_plane;
  T *out = output + n * C * out_plane + pix;
  if (MODE == 0) {
    // Round half to even, as the ONNX reference does.
    int iy = static_cast<int>(rintf(y));
    int ix = static_cast<int>(rintf(x));
    int64_t off = padded_offset<PAD>(iy, ix, H, W, lo);
    for (int64_t c = 0; c < C; ++c, img += in_plane, out += out_plane)
      *out = from_float<T>(load_or_zero(img, off));
    return;
  }
  int y0 = static_cast<int>(floorf(y));
  int x0 = static_cast<int>(floorf(x));
  int y1 = y0 + 1;
  int x1 = x0 + 1;
  float dy = y - static_cast<float>(y0);
  float dx = x - static_cast<float>(x0);
  int64_t o00 = padded_offset<PAD>(y0, x0, H, W, lo);
  int64_t o01 = padded_offset<PAD>(y0, x1, H, W, lo);
  int64_t o10 = padded_offset<PAD>(y1, x0, H, W, lo);
  int64_t o11 = padded_offset<PAD>(y1, x1, H, W, lo);
  for (int64_t c = 0; c < C; ++c, img += in_plane, out += out_plane) {
    float v00 = load_or_zero(img, o00);
    float v01 = load_or_zero(img, o01);
    float v10 = load_or_zero(img, o10);
    float v11 = load_or_zero(img, o11);
    float val = v00 * (1.0f - dx) * (1.0f - dy) + v01 * dx * (1.0f - dy) +
                v10 * (1.0f - dx) * dy + v11 * dx * dy;
    *out = from_float<T>(val);
  }
}

template <typename T, int MODE>
static void launch_grid_sample_pad(int padding_mode, dim3 grid_dim, dim3 block,
                                   hipStream_t stream, const void *input,
                                   const void *grid, void *output, int64_t c,
                                   int64_t in_h, int64_t in_w, int64_t out_h,
                                   int64_t out_w, int align_corners,
                                   int64_t pixels) {
  auto in = static_cast<const T *>(input);
  auto g = static_cast<const T *>(grid);
  auto out = static_cast<T *>(output);
  if (padding_mode == 0)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<T, MODE, 0>),
                       grid_dim, block, 0, stream, in, g, out, c, in_h, in_w,
                       out_h, out_w, align_corners, pixels);
  else if (padding_mode == 1)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<T, MODE, 1>),
                       grid_dim, block, 0, stream, in, g, out, c, in_h, in_w,
                       out_h, out_w, align_corners, pixels);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(grid_sample_kernel<T, MODE, 2>),
                       grid_dim, block, 0, stream, in, g, out, c, in_h, in_w,
                       out_h, out_w, align_corners, pixels);
}

template <typename T>
static void launch_grid_sample(int mode, int padding_mode, dim3 grid_dim,
                               dim3 block, hipStream_t stream,
                               const void *input, const void *grid,
                               void *output, int64_t c, int64_t in_h,
                               int64_t in_w, int64_t out_h, int64_t out_w,
                               int align_corners, int64_t pixels) {
  if (mode == 0)
    launch_grid_sample_pad<T, 0>(padding_mode, grid_dim, block, stream, input,
                                 grid, output, c, in_h, in_w, out_h, out_w,
                                 align_corners, pixels);
  else
    launch_grid_sample_pad<T, 1>(padding_mode, grid_dim, block, stream, input,
                                 grid, output, c, in_h, in_w, out_h, out_w,
                                 align_corners, pixels);
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
  int64_t pixels = n * out_h * out_w;
  dim3 block(256);
  dim3 grid_dim(static_cast<unsigned int>((pixels + 255) / 256));

  (void)hipGetLastError();
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_grid_sample: dtype=%d N=%lld C=%lld in=%lldx%lld "
      "out=%lldx%lld mode=%d pad=%d align=%d\n",
      hip_dtype, (long long)n, (long long)c, (long long)in_h, (long long)in_w,
      (long long)out_h, (long long)out_w, mode, padding_mode, align_corners);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    launch_grid_sample<__half>(mode, padding_mode, grid_dim, block, hip_stream,
                               input, grid, output, c, in_h, in_w, out_h,
                               out_w, align_corners, pixels);
  } else if (hip_dtype == HIP_DTYPE_BFLOAT16) {
    launch_grid_sample<__hip_bfloat16>(mode, padding_mode, grid_dim, block,
                                       hip_stream, input, grid, output, c, in_h,
                                       in_w, out_h, out_w, align_corners,
                                       pixels);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    launch_grid_sample<float>(mode, padding_mode, grid_dim, block, hip_stream,
                              input, grid, output, c, in_h, in_w, out_h, out_w,
                              align_corners, pixels);
  } else if (hip_dtype == HIP_DTYPE_FLOAT64) {
    launch_grid_sample<double>(mode, padding_mode, grid_dim, block, hip_stream,
                               input, grid, output, c, in_h, in_w, out_h,
                               out_w, align_corners, pixels);
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
