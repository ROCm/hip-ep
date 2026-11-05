/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>

// =============================================================================
// Resize (1D / 2D / 3D spatial)
// =============================================================================
//
// One thread per output element.  Output is laid out row-major over
// (N, C, O_1, O_2, O_3) where unused trailing axes are size 1.  Each
// thread maps its (o_i) coords back to the input domain via the configured
// coordinate_transformation_mode, then picks the nearest input sample
// (mode 0), N-linearly interpolates over the surrounding 2^k samples
// (mode 1), with HIP_RESIZE_FLAG_ANTIALIAS averages the samples under a
// widened triangle filter, or N-cubically interpolates over the surrounding
// 4^k samples (mode 2).  (N, C) prefix is pass-through.  An axis samples with
// its ONNX scale when hip_resize_scaled supplies one (> 0), else with
// `out_dim / in_dim`.

namespace {

template <typename T> __device__ inline float to_float(T v);
template <> __device__ inline float to_float<float>(float v) { return v; }
template <> __device__ inline float to_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ inline float to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}
template <> __device__ inline float to_float<double>(double v) {
  return static_cast<float>(v);
}

template <typename T> __device__ inline T from_float(float v);
template <> __device__ inline float from_float<float>(float v) { return v; }
template <> __device__ inline __half from_float<__half>(float v) {
  return __float2half(v);
}
template <> __device__ inline hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}
template <> __device__ inline double from_float<double>(float v) {
  return static_cast<double>(v);
}
// Integer linear interpolates the stored codes. A convex blend stays inside
// the input range, and the store truncates toward zero.
template <> __device__ inline float to_float<uint8_t>(uint8_t v) {
  return static_cast<float>(v);
}
template <> __device__ inline float to_float<int8_t>(int8_t v) {
  return static_cast<float>(v);
}
template <> __device__ inline uint8_t from_float<uint8_t>(float v) {
  return static_cast<uint8_t>(v);
}
template <> __device__ inline int8_t from_float<int8_t>(float v) {
  return static_cast<int8_t>(v);
}

// Map output coord -> continuous input coord per the chosen mode.
// `onnx_scale` is the supplied ONNX scale (output / input), or 0 to use
// out_dim / in_dim.
__device__ inline float coord_to_input(int o, int out_dim, int in_dim,
                                       int coord_transform, float onnx_scale) {
  float scale = onnx_scale > 0.0f
                    ? onnx_scale
                    : static_cast<float>(out_dim) / static_cast<float>(in_dim);
  switch (coord_transform) {
    case 0: { // half_pixel
      // x_original = (x_resized + 0.5) / scale - 0.5
      return (static_cast<float>(o) + 0.5f) / scale - 0.5f;
    }
    case 1: { // asymmetric
      return static_cast<float>(o) / scale;
    }
    case 2: { // align_corners
      if (out_dim == 1) return 0.0f;
      return static_cast<float>(o) *
             static_cast<float>(in_dim - 1) /
             static_cast<float>(out_dim - 1);
    }
    case 3: { // pytorch_half_pixel
      // Same map as half_pixel, except an output axis of length 1 samples
      // input coordinate 0.
      if (out_dim == 1)
        return 0.0f;
      return (static_cast<float>(o) + 0.5f) / scale - 0.5f;
    }
    default:
      return static_cast<float>(o);
  }
}

__device__ inline int nearest_round(float x, int nearest_mode) {
  // ONNX nearest_mode:
  //   0 = round_prefer_floor: ties go DOWN
  //   1 = round_prefer_ceil:  ties go UP
  //   2 = floor
  //   3 = ceil
  switch (nearest_mode) {
    case 1: return static_cast<int>(floorf(x + 0.5f));
    case 2: return static_cast<int>(floorf(x));
    case 3: return static_cast<int>(ceilf(x));
    case 0:
    default: return static_cast<int>(ceilf(x - 0.5f));
  }
}

template <typename T>
__global__ void resize_nearest_kernel(const T* __restrict__ input,
                                      T* __restrict__ output,
                                      int spatial_rank,
                                      int64_t N, int64_t C,
                                      int64_t in0, int64_t in1, int64_t in2,
                                      int64_t out0, int64_t out1, int64_t out2,
                                      int coord_transform,
                                      int nearest_mode,
                                      float s0, float s1, float s2,
                                      int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  int64_t plane = out0 * out1 * out2;
  int64_t channel_stride = plane;
  int64_t batch_stride = C * channel_stride;
  int64_t n = tid / batch_stride;
  int64_t rem = tid - n * batch_stride;
  int64_t c = rem / channel_stride;
  rem -= c * channel_stride;
  int64_t o0 = rem / (out1 * out2);
  int64_t r1 = rem - o0 * (out1 * out2);
  int64_t o1 = r1 / out2;
  int64_t o2 = r1 - o1 * out2;

  // Map each axis (skipping unused) — for spatial_rank<3 the unused axis
  // has out_dim=in_dim=1 and o_i=0, so the coord trivially resolves to 0.
  float x0 = coord_to_input(static_cast<int>(o0), static_cast<int>(out0),
                            static_cast<int>(in0), coord_transform, s0);
  float x1 = coord_to_input(static_cast<int>(o1), static_cast<int>(out1),
                            static_cast<int>(in1), coord_transform, s1);
  float x2 = coord_to_input(static_cast<int>(o2), static_cast<int>(out2),
                            static_cast<int>(in2), coord_transform, s2);

  int i0 = nearest_round(x0, nearest_mode);
  int i1 = nearest_round(x1, nearest_mode);
  int i2 = nearest_round(x2, nearest_mode);
  if (i0 < 0) i0 = 0;
  if (i0 >= in0) i0 = static_cast<int>(in0 - 1);
  if (i1 < 0) i1 = 0;
  if (i1 >= in1) i1 = static_cast<int>(in1 - 1);
  if (i2 < 0) i2 = 0;
  if (i2 >= in2) i2 = static_cast<int>(in2 - 1);

  int64_t base_in = (n * C + c) * (in0 * in1 * in2);
  int64_t in_off = static_cast<int64_t>(i0) * in1 * in2 +
                   static_cast<int64_t>(i1) * in2 + static_cast<int64_t>(i2);
  output[tid] = input[base_in + in_off];
}

template <typename T>
__device__ inline float fetch(const T* base, int64_t i0, int64_t i1, int64_t i2,
                              int64_t in0, int64_t in1, int64_t in2) {
  if (i0 < 0) i0 = 0;
  if (i0 >= in0) i0 = in0 - 1;
  if (i1 < 0) i1 = 0;
  if (i1 >= in1) i1 = in1 - 1;
  if (i2 < 0) i2 = 0;
  if (i2 >= in2) i2 = in2 - 1;
  return to_float<T>(base[i0 * in1 * in2 + i1 * in2 + i2]);
}

template <typename T>
__global__ void resize_linear_kernel(const T* __restrict__ input,
                                     T* __restrict__ output,
                                     int spatial_rank,
                                     int64_t N, int64_t C,
                                     int64_t in0, int64_t in1, int64_t in2,
                                     int64_t out0, int64_t out1, int64_t out2,
                                     int coord_transform,
                                     float s0, float s1, float s2,
                                     int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  int64_t plane = out0 * out1 * out2;
  int64_t channel_stride = plane;
  int64_t batch_stride = C * channel_stride;
  int64_t n = tid / batch_stride;
  int64_t rem = tid - n * batch_stride;
  int64_t c = rem / channel_stride;
  rem -= c * channel_stride;
  int64_t o0 = rem / (out1 * out2);
  int64_t r1 = rem - o0 * (out1 * out2);
  int64_t o1 = r1 / out2;
  int64_t o2 = r1 - o1 * out2;

  float x0 = coord_to_input(static_cast<int>(o0), static_cast<int>(out0),
                            static_cast<int>(in0), coord_transform, s0);
  float x1 = coord_to_input(static_cast<int>(o1), static_cast<int>(out1),
                            static_cast<int>(in1), coord_transform, s1);
  float x2 = coord_to_input(static_cast<int>(o2), static_cast<int>(out2),
                            static_cast<int>(in2), coord_transform, s2);

  int64_t lo0 = static_cast<int64_t>(floorf(x0));
  int64_t lo1 = static_cast<int64_t>(floorf(x1));
  int64_t lo2 = static_cast<int64_t>(floorf(x2));
  float t0 = x0 - static_cast<float>(lo0);
  float t1 = x1 - static_cast<float>(lo1);
  float t2 = x2 - static_cast<float>(lo2);
  // For axes that aren't actually being resized (in_dim == 1 or
  // spatial_rank < 3), the contribution collapses to a single sample.

  const T* base = input + (n * C + c) * (in0 * in1 * in2);

  // 8-corner trilinear (degenerates correctly when any axis is size 1).
  float c000 = fetch(base, lo0,     lo1,     lo2,     in0, in1, in2);
  float c001 = fetch(base, lo0,     lo1,     lo2 + 1, in0, in1, in2);
  float c010 = fetch(base, lo0,     lo1 + 1, lo2,     in0, in1, in2);
  float c011 = fetch(base, lo0,     lo1 + 1, lo2 + 1, in0, in1, in2);
  float c100 = fetch(base, lo0 + 1, lo1,     lo2,     in0, in1, in2);
  float c101 = fetch(base, lo0 + 1, lo1,     lo2 + 1, in0, in1, in2);
  float c110 = fetch(base, lo0 + 1, lo1 + 1, lo2,     in0, in1, in2);
  float c111 = fetch(base, lo0 + 1, lo1 + 1, lo2 + 1, in0, in1, in2);

  // Reduce along axis 2 (innermost).
  float c00 = c000 * (1.0f - t2) + c001 * t2;
  float c01 = c010 * (1.0f - t2) + c011 * t2;
  float c10 = c100 * (1.0f - t2) + c101 * t2;
  float c11 = c110 * (1.0f - t2) + c111 * t2;
  // Reduce along axis 1.
  float c0 = c00 * (1.0f - t1) + c01 * t1;
  float c1 = c10 * (1.0f - t1) + c11 * t1;
  // Reduce along axis 0.
  float v = c0 * (1.0f - t0) + c1 * t0;

  output[tid] = from_float<T>(v);
}

// Antialiased linear (mode 1 with HIP_RESIZE_FLAG_ANTIALIAS and
// HIP_RESIZE_FLAG_EXCLUDE_OUTSIDE): ONNX Resize with antialias = 1 and
// exclude_outside = 1, which is PyTorch's interpolate(..., antialias=True).
// Along an axis that shrinks, the triangle filter widens to 1 / scale input
// pixels; each output is the weighted sum of the input pixels inside the
// filter's support, over the sum of their weights (pixels outside the input
// do not count). An axis that does not shrink is plain linear. Only the
// half_pixel and pytorch_half_pixel transforms (coord_transform 0 and 3).
struct AntialiasAxis {
  float center;     // sample position, in input pixels, plus 0.5
  float support;    // half-width of the filter, in input pixels
  float invSupport; // 1 / support
  int64_t lo, hi;   // input pixels inside the support, [lo, hi)
};

__device__ inline AntialiasAxis antialias_axis(int64_t o, int64_t out_dim,
                                               int64_t in_dim,
                                               int coord_transform,
                                               float onnx_scale) {
  const float scale = onnx_scale > 0.0f ? onnx_scale
                                        : static_cast<float>(out_dim) /
                                              static_cast<float>(in_dim);
  AntialiasAxis a;
  // pytorch_half_pixel samples input coordinate 0 on an output axis of 1.
  a.center = coord_transform == 3 && out_dim == 1
                 ? 0.5f
                 : (static_cast<float>(o) + 0.5f) / scale;
  a.support = fmaxf(1.0f / scale, 1.0f);
  a.invSupport = 1.0f / a.support;
  a.lo = max(int64_t(0), int64_t(a.center - a.support + 0.5f));
  a.hi = min(in_dim, int64_t(a.center + a.support + 0.5f));
  return a;
}

__device__ inline float antialias_weight(const AntialiasAxis &a, int64_t i) {
  return fmaxf(
      1.0f - fabsf((static_cast<float>(i) - a.center + 0.5f) * a.invSupport),
      0.0f);
}

template <typename T>
__global__ void
resize_antialias_kernel(const T *__restrict__ input, T *__restrict__ output,
                        int64_t N, int64_t C, int64_t in0, int64_t in1,
                        int64_t in2, int64_t out0, int64_t out1, int64_t out2,
                        int coord_transform, float s0, float s1, float s2,
                        int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs)
      return;

  int64_t plane = out0 * out1 * out2;
  int64_t n = tid / (C * plane);
  int64_t rem = tid - n * C * plane;
  int64_t c = rem / plane;
  rem -= c * plane;
  int64_t o0 = rem / (out1 * out2);
  int64_t r1 = rem - o0 * (out1 * out2);
  int64_t o1 = r1 / out2;
  int64_t o2 = r1 - o1 * out2;

  // An unused axis (in = out = 1) has the one pixel 0 at weight exactly 1.
  const AntialiasAxis a0 = antialias_axis(o0, out0, in0, coord_transform, s0);
  const AntialiasAxis a1 = antialias_axis(o1, out1, in1, coord_transform, s1);
  const AntialiasAxis a2 = antialias_axis(o2, out2, in2, coord_transform, s2);

  const T *base = input + (n * C + c) * (in0 * in1 * in2);
  float value = 0.0f;
  float total = 0.0f;
  for (int64_t i0 = a0.lo; i0 < a0.hi; ++i0) {
      const float w0 = antialias_weight(a0, i0);
      for (int64_t i1 = a1.lo; i1 < a1.hi; ++i1) {
        const float w1 = antialias_weight(a1, i1);
        for (int64_t i2 = a2.lo; i2 < a2.hi; ++i2) {
          const float weight = antialias_weight(a2, i2) * w1 * w0;
          if (weight <= 0.0f)
            continue;
          value += to_float<T>(base[(i0 * in1 + i1) * in2 + i2]) * weight;
          total += weight;
        }
      }
  }
  if (total > 0.0f)
      value /= total;
  output[tid] = from_float<T>(value);
}

// Cubic (mode 2): Keys' kernel with a = -0.75 over the 4 samples around
// each coordinate, a tap past the edge taking the edge sample (ONNX cubic
// with exclude_outside = 0, PyTorch's and GGML's bicubic). Written as GGML's
// CPU bicubic is, weights and sums in the same order, the innermost axis
// first. An axis the resize leaves at length 1 is one tap of weight 1.
struct CubicAxis {
  int64_t index[4];
  float weight[4];
  int taps;
};

__device__ inline CubicAxis cubic_axis(int64_t o, int64_t out_dim,
                                       int64_t in_dim, int coord_transform,
                                       float onnx_scale) {
  CubicAxis c;
  if (in_dim == 1 && out_dim == 1) {
      c.index[0] = 0;
      c.weight[0] = 1.0f;
      c.taps = 1;
      return c;
  }
  constexpr float a = -0.75f;
  const auto near = [a](float x) {
    return ((a + 2) * x - (a + 3)) * x * x + 1;
  };
  const auto far = [a](float x) {
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  };
  const float x =
      coord_to_input(static_cast<int>(o), static_cast<int>(out_dim),
                     static_cast<int>(in_dim), coord_transform, onnx_scale);
  const int64_t x0 = static_cast<int64_t>(floorf(x));
  const float d = x - static_cast<float>(x0);
  c.weight[0] = far(d + 1);
  c.weight[1] = near(d);
  c.weight[2] = near(1 - d);
  c.weight[3] = far(2 - d);
  for (int k = 0; k < 4; ++k)
      c.index[k] = min(max(x0 - 1 + k, int64_t(0)), in_dim - 1);
  c.taps = 4;
  return c;
}

template <typename T>
__global__ void resize_cubic_kernel(const T *__restrict__ input,
                                    T *__restrict__ output, int64_t N,
                                    int64_t C, int64_t in0, int64_t in1,
                                    int64_t in2, int64_t out0, int64_t out1,
                                    int64_t out2, int coord_transform, float s0,
                                    float s1, float s2, int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs)
      return;

  int64_t plane = out0 * out1 * out2;
  int64_t n = tid / (C * plane);
  int64_t rem = tid - n * C * plane;
  int64_t c = rem / plane;
  rem -= c * plane;
  int64_t o0 = rem / (out1 * out2);
  int64_t r1 = rem - o0 * (out1 * out2);
  int64_t o1 = r1 / out2;
  int64_t o2 = r1 - o1 * out2;

  const CubicAxis a0 = cubic_axis(o0, out0, in0, coord_transform, s0);
  const CubicAxis a1 = cubic_axis(o1, out1, in1, coord_transform, s1);
  const CubicAxis a2 = cubic_axis(o2, out2, in2, coord_transform, s2);

  const T *base = input + (n * C + c) * (in0 * in1 * in2);
  float value = 0.0f;
  for (int k0 = 0; k0 < a0.taps; ++k0) {
      float row = 0.0f;
      for (int k1 = 0; k1 < a1.taps; ++k1) {
        float column = 0.0f;
        for (int k2 = 0; k2 < a2.taps; ++k2)
          column += to_float<T>(base[(a0.index[k0] * in1 + a1.index[k1]) * in2 +
                                     a2.index[k2]]) *
                    a2.weight[k2];
        row += column * a1.weight[k1];
      }
      value += row * a0.weight[k0];
  }
  output[tid] = from_float<T>(value);
}

// The filtered resizes (antialias, cubic) take float element types only;
// resize_dispatch rejects them for integer codes before launching.
template <typename T>
static void launch_filtered(hipStream_t stream, const T *input, T *output,
                            int64_t grid, int block, bool antialias, int64_t N,
                            int64_t C, int64_t in0, int64_t in1, int64_t in2,
                            int64_t out0, int64_t out1, int64_t out2,
                            int coord_transform, float s0, float s1, float s2,
                            int64_t total) {
  if constexpr (!std::is_integral<T>::value) {
      if (antialias)
        hipLaunchKernelGGL(resize_antialias_kernel<T>,
                           dim3(static_cast<unsigned>(grid)), dim3(block), 0,
                           stream, input, output, N, C, in0, in1, in2, out0,
                           out1, out2, coord_transform, s0, s1, s2, total);
      else
        hipLaunchKernelGGL(resize_cubic_kernel<T>,
                           dim3(static_cast<unsigned>(grid)), dim3(block), 0,
                           stream, input, output, N, C, in0, in1, in2, out0,
                           out1, out2, coord_transform, s0, s1, s2, total);
  }
}

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  int spatial_rank, int64_t N, int64_t C,
                  int64_t in0, int64_t in1, int64_t in2,
                  int64_t out0, int64_t out1, int64_t out2,
                  int mode, int coord_transform, int nearest_mode,
                  float s0, float s1, float s2) {
  int64_t total = N * C * out0 * out1 * out2;
  if (total <= 0) return 0;
  int block = 256;
  int64_t grid = (total + block - 1) / block;
  const int base_mode = mode & HIP_RESIZE_MODE_MASK;
  const bool antialias = (mode & HIP_RESIZE_FLAG_ANTIALIAS) != 0;

  (void)hipGetLastError();
  if (base_mode == 0) {
      hipLaunchKernelGGL(
          resize_nearest_kernel<T>, dim3(static_cast<unsigned>(grid)),
          dim3(block), 0, stream, static_cast<const T *>(input),
          static_cast<T *>(output), spatial_rank, N, C, in0, in1, in2, out0,
          out1, out2, coord_transform, nearest_mode, s0, s1, s2, total);
  } else if (base_mode == 1 && !antialias) {
      hipLaunchKernelGGL(
          resize_linear_kernel<T>, dim3(static_cast<unsigned>(grid)),
          dim3(block), 0, stream, static_cast<const T *>(input),
          static_cast<T *>(output), spatial_rank, N, C, in0, in1, in2, out0,
          out1, out2, coord_transform, s0, s1, s2, total);
  } else {
      launch_filtered<T>(stream, static_cast<const T *>(input),
                         static_cast<T *>(output), grid, block, antialias, N, C,
                         in0, in1, in2, out0, out1, out2, coord_transform, s0,
                         s1, s2, total);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

static int resize_dispatch(void* stream, const void* input, void* output,
                           int hip_dtype, int spatial_rank, int64_t N,
                           int64_t C, int64_t in_d0, int64_t in_d1,
                           int64_t in_d2, int64_t out_d0, int64_t out_d1,
                           int64_t out_d2, int mode, int coord_transform,
                           int nearest_mode, float s0, float s1, float s2) {
  if (spatial_rank < 1 || spatial_rank > 3) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: unsupported spatial_rank=%d\n",
            spatial_rank);
    return -1;
  }
  const int base_mode = mode & HIP_RESIZE_MODE_MASK;
  const bool antialias = (mode & HIP_RESIZE_FLAG_ANTIALIAS) != 0;
  const bool exclude_outside = (mode & HIP_RESIZE_FLAG_EXCLUDE_OUTSIDE) != 0;
  const int known = HIP_RESIZE_MODE_MASK | HIP_RESIZE_FLAG_ANTIALIAS |
                    HIP_RESIZE_FLAG_EXCLUDE_OUTSIDE;
  if (base_mode > 2 || (mode & ~known) != 0 ||
      (antialias && (base_mode != 1 || !exclude_outside ||
                     (coord_transform != 0 && coord_transform != 3))) ||
      (base_mode == 2 && exclude_outside)) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: unsupported mode=0x%x "
            "coord_transform=%d\n",
            mode, coord_transform);
    return -1;
  }
  if ((antialias || base_mode == 2) &&
      (hip_dtype == HIP_DTYPE_INT8 || hip_dtype == HIP_DTYPE_UINT8)) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: antialias and cubic take "
            "float types, not dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_resize: dtype=%d spatial_rank=%d N=%lld C=%lld "
      "in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] mode=%d ct=%d nearest=%d "
      "scales=[%g,%g,%g]\n",
      hip_dtype, spatial_rank, (long long)N, (long long)C,
      (long long)in_d0, (long long)in_d1, (long long)in_d2,
      (long long)out_d0, (long long)out_d1, (long long)out_d2,
      mode, coord_transform, nearest_mode, s0, s1, s2);

  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      return launch<float>(hip_stream, input, output, spatial_rank, N, C,
                           in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                           mode, coord_transform, nearest_mode, s0, s1, s2);
    case HIP_DTYPE_FLOAT16:
      return launch<__half>(hip_stream, input, output, spatial_rank, N, C,
                            in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                            mode, coord_transform, nearest_mode, s0, s1, s2);
    case HIP_DTYPE_BFLOAT16:
      return launch<hip_bfloat16>(hip_stream, input, output, spatial_rank,
                                  N, C, in_d0, in_d1, in_d2,
                                  out_d0, out_d1, out_d2,
                                  mode, coord_transform, nearest_mode,
                                  s0, s1, s2);
    case HIP_DTYPE_FLOAT64:
      return launch<double>(hip_stream, input, output, spatial_rank, N, C,
                            in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                            mode, coord_transform, nearest_mode, s0, s1, s2);
    case HIP_DTYPE_INT8:
      return launch<int8_t>(hip_stream, input, output, spatial_rank, N, C,
                            in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                            mode, coord_transform, nearest_mode, s0, s1, s2);
    case HIP_DTYPE_UINT8:
      return launch<uint8_t>(hip_stream, input, output, spatial_rank, N, C,
                             in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                             mode, coord_transform, nearest_mode, s0, s1, s2);
    default:
      fprintf(stderr,
              "[custom_kernels] hip_resize: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}

} // namespace

extern "C" int hip_resize(void* stream,
                          const void* input,
                          void* output,
                          int hip_dtype,
                          int spatial_rank,
                          int64_t N, int64_t C,
                          int64_t in_d0, int64_t in_d1, int64_t in_d2,
                          int64_t out_d0, int64_t out_d1, int64_t out_d2,
                          int mode,
                          int coord_transform,
                          int nearest_mode) {
  return resize_dispatch(stream, input, output, hip_dtype, spatial_rank, N, C,
                         in_d0, in_d1, in_d2, out_d0, out_d1, out_d2, mode,
                         coord_transform, nearest_mode, 0.0f, 0.0f, 0.0f);
}

extern "C" int hip_resize_scaled(void* stream,
                                 const void* input,
                                 void* output,
                                 int hip_dtype,
                                 int spatial_rank,
                                 int64_t N, int64_t C,
                                 int64_t in_d0, int64_t in_d1, int64_t in_d2,
                                 int64_t out_d0, int64_t out_d1,
                                 int64_t out_d2,
                                 int mode,
                                 int coord_transform,
                                 int nearest_mode,
                                 float scale0, float scale1, float scale2) {
  return resize_dispatch(stream, input, output, hip_dtype, spatial_rank, N, C,
                         in_d0, in_d1, in_d2, out_d0, out_d1, out_d2, mode,
                         coord_transform, nearest_mode, scale0, scale1,
                         scale2);
}
