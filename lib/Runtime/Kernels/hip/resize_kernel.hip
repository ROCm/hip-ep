/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// =============================================================================
// Resize (1D / 2D / 3D spatial)
// =============================================================================
//
// One thread per output element.  Output is laid out row-major over
// (N, C, O_1, O_2, O_3) where unused trailing axes are size 1.  Each
// thread maps its (o_i) coords back to the input domain via the configured
// coordinate_transformation_mode, then either picks the nearest input
// sample (mode=0) or N-linearly interpolates over the surrounding 2^k
// samples (mode=1).  (N, C) prefix is pass-through, scale is recovered as
// `in_dim / out_dim` per axis.

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

// Map output coord -> continuous input coord per the chosen mode.
__device__ inline float coord_to_input(int o, int out_dim, int in_dim,
                                       int coord_transform) {
  // scale_along_axis = in_dim / out_dim  (= length_original / length_resized)
  // We compute x_original directly from `o`, `in_dim`, `out_dim`.
  switch (coord_transform) {
    case 0: { // half_pixel
      // x_original = (x_resized + 0.5) / scale_resized - 0.5
      // where scale_resized = out_dim / in_dim
      float scale = static_cast<float>(out_dim) / static_cast<float>(in_dim);
      return (static_cast<float>(o) + 0.5f) / scale - 0.5f;
    }
    case 1: { // asymmetric
      float scale = static_cast<float>(out_dim) / static_cast<float>(in_dim);
      return static_cast<float>(o) / scale;
    }
    case 2: { // align_corners
      if (out_dim == 1) return 0.0f;
      return static_cast<float>(o) *
             static_cast<float>(in_dim - 1) /
             static_cast<float>(out_dim - 1);
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
                            static_cast<int>(in0), coord_transform);
  float x1 = coord_to_input(static_cast<int>(o1), static_cast<int>(out1),
                            static_cast<int>(in1), coord_transform);
  float x2 = coord_to_input(static_cast<int>(o2), static_cast<int>(out2),
                            static_cast<int>(in2), coord_transform);

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
                            static_cast<int>(in0), coord_transform);
  float x1 = coord_to_input(static_cast<int>(o1), static_cast<int>(out1),
                            static_cast<int>(in1), coord_transform);
  float x2 = coord_to_input(static_cast<int>(o2), static_cast<int>(out2),
                            static_cast<int>(in2), coord_transform);

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

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  int spatial_rank, int64_t N, int64_t C,
                  int64_t in0, int64_t in1, int64_t in2,
                  int64_t out0, int64_t out1, int64_t out2,
                  int mode, int coord_transform, int nearest_mode) {
  int64_t total = N * C * out0 * out1 * out2;
  if (total <= 0) return 0;
  int block = 256;
  int64_t grid = (total + block - 1) / block;

  if (mode == 0) {
    hipLaunchKernelGGL(resize_nearest_kernel<T>,
                       dim3(static_cast<unsigned>(grid)), dim3(block),
                       0, stream,
                       static_cast<const T*>(input), static_cast<T*>(output),
                       spatial_rank, N, C, in0, in1, in2,
                       out0, out1, out2,
                       coord_transform, nearest_mode, total);
  } else {
    hipLaunchKernelGGL(resize_linear_kernel<T>,
                       dim3(static_cast<unsigned>(grid)), dim3(block),
                       0, stream,
                       static_cast<const T*>(input), static_cast<T*>(output),
                       spatial_rank, N, C, in0, in1, in2,
                       out0, out1, out2,
                       coord_transform, total);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
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
  if (spatial_rank < 1 || spatial_rank > 3) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: unsupported spatial_rank=%d\n",
            spatial_rank);
    return -1;
  }
  if (mode != 0 && mode != 1) {
    fprintf(stderr, "[custom_kernels] hip_resize: bad mode=%d\n", mode);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_resize: dtype=%d spatial_rank=%d N=%lld C=%lld "
      "in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] mode=%d ct=%d nearest=%d\n",
      hip_dtype, spatial_rank, (long long)N, (long long)C,
      (long long)in_d0, (long long)in_d1, (long long)in_d2,
      (long long)out_d0, (long long)out_d1, (long long)out_d2,
      mode, coord_transform, nearest_mode);

  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      return launch<float>(hip_stream, input, output, spatial_rank, N, C,
                           in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                           mode, coord_transform, nearest_mode);
    case HIP_DTYPE_FLOAT16:
      return launch<__half>(hip_stream, input, output, spatial_rank, N, C,
                            in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                            mode, coord_transform, nearest_mode);
    case HIP_DTYPE_BFLOAT16:
      return launch<hip_bfloat16>(hip_stream, input, output, spatial_rank,
                                  N, C, in_d0, in_d1, in_d2,
                                  out_d0, out_d1, out_d2,
                                  mode, coord_transform, nearest_mode);
    case HIP_DTYPE_FLOAT64:
      return launch<double>(hip_stream, input, output, spatial_rank, N, C,
                            in_d0, in_d1, in_d2, out_d0, out_d1, out_d2,
                            mode, coord_transform, nearest_mode);
    default:
      fprintf(stderr,
              "[custom_kernels] hip_resize: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}
