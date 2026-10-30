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
// The caller packs a rank-r window as (O_1, O_2, O_3) with the unused
// trailing axes size 1. The host moves those unused axes to the front, so the
// output is row-major over (N, C, D, H, W) with W the innermost resized axis.
// Each thread writes VEC consecutive outputs of one W row: it maps the row's
// D and H coordinates once, and only the W coordinate per output. The host
// picks the widest store of 16, 8, 4 or 2 bytes whose VEC divides the row
// length and whose width divides the output alignment, else VEC = 1. Nearest
// (mode=0) copies the selected sample; linear (mode=1) blends W, then H, then
// D, the same f32 order for every rank. (N, C) is pass-through. An axis
// samples with its ONNX scale when hip_resize_scaled supplies one (> 0), else
// with `out_dim / in_dim`.
//
// The source coordinate divides by the scale, as ONNX defines it. A multiply
// by a reciprocal moves exact boundaries: with scale 3, 3 * (1 / 3.0f) floors
// to 0, not 1.

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

__device__ inline int clamp_index(int i, int dim) {
  return i < 0 ? 0 : (i >= dim ? dim - 1 : i);
}

__device__ inline int nearest_index(int o, int out_dim, int in_dim,
                                    int coord_transform, int nearest_mode,
                                    float scale) {
  return clamp_index(
      nearest_round(coord_to_input(o, out_dim, in_dim, coord_transform, scale),
                    nearest_mode),
      in_dim);
}

struct LinearTap {
  int lo;
  int hi;
  float t;
};

__device__ inline LinearTap linear_tap(int o, int out_dim, int in_dim,
                                       int coord_transform, float scale) {
  float x = coord_to_input(o, out_dim, in_dim, coord_transform, scale);
  int lo = static_cast<int>(floorf(x));
  return {clamp_index(lo, in_dim), clamp_index(lo + 1, in_dim),
          x - static_cast<float>(lo)};
}

// One launch after the unused axes moved to the front. Every extent fits in
// int, and rows (N * C * out_d * out_h) fits in int.
struct RowShape {
  int in_d, in_h, in_w;
  int out_d, out_h, out_w;
  int rows;
  int lanes; // out_w / VEC
  float s_d, s_h, s_w;
};

template <typename T, int VEC> struct alignas(sizeof(T) * VEC) VecOf {
  T v[VEC];
};

// grid.x * blockDim.x and grid.y * blockDim.y must each fit in 32 bits, so
// the host caps both grid dimensions and the kernels loop over rows and over
// the lanes of a long row.
__device__ inline void decode_row(const RowShape& s, int row, int& nc, int& od,
                                  int& oh) {
  oh = row % s.out_h;
  int t = row / s.out_h;
  od = t % s.out_d;
  nc = t / s.out_d;
}

__device__ inline int first_row() {
  return static_cast<int>(blockIdx.x * blockDim.y + threadIdx.y);
}

__device__ inline int row_stride() {
  return static_cast<int>(gridDim.x * blockDim.y);
}

__device__ inline int first_lane() {
  return static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
}

__device__ inline int lane_stride() {
  return static_cast<int>(gridDim.y * blockDim.x);
}

template <typename T, int VEC>
__global__ void resize_nearest_rows_kernel(const T* __restrict__ input,
                                           T* __restrict__ output, RowShape s,
                                           int coord_transform,
                                           int nearest_mode) {
  for (int row = first_row(); row < s.rows; row += row_stride()) {
    int nc, od, oh;
    decode_row(s, row, nc, od, oh);
    int id = nearest_index(od, s.out_d, s.in_d, coord_transform, nearest_mode,
                           s.s_d);
    int ih = nearest_index(oh, s.out_h, s.in_h, coord_transform, nearest_mode,
                           s.s_h);
    const T* src =
        input +
        ((static_cast<int64_t>(nc) * s.in_d + id) * s.in_h + ih) * s.in_w;
    T* dst = output + static_cast<int64_t>(row) * s.out_w;

    for (int lane = first_lane(); lane < s.lanes; lane += lane_stride()) {
      int w0 = lane * VEC;
      VecOf<T, VEC> out;
#pragma unroll
      for (int k = 0; k < VEC; ++k)
        out.v[k] = src[nearest_index(w0 + k, s.out_w, s.in_w,
                                     coord_transform, nearest_mode, s.s_w)];
      *reinterpret_cast<VecOf<T, VEC>*>(dst + w0) = out;
    }
  }
}

// Every product and sum rounds to f32 on its own: with contraction, which
// product the compiler fuses into an FMA changes the last bit.
template <typename T>
__device__ inline float lerp_w(const T* row, const LinearTap& w) {
#pragma clang fp contract(off)
  return to_float<T>(row[w.lo]) * (1.0f - w.t) + to_float<T>(row[w.hi]) * w.t;
}

// The unused front axes (D below rank 3, H below rank 2) map to index 0 with
// weight 0, so the kernel skips them instead of blending a sample with itself.
template <typename T, int VEC>
__global__ void resize_linear_rows_kernel(const T* __restrict__ input,
                                          T* __restrict__ output, RowShape s,
                                          int spatial_rank,
                                          int coord_transform) {
  const int64_t plane = static_cast<int64_t>(s.in_h) * s.in_w;
  for (int row = first_row(); row < s.rows; row += row_stride()) {
    int nc, od, oh;
    decode_row(s, row, nc, od, oh);
    LinearTap td = {0, 0, 0.0f};
    LinearTap th = {0, 0, 0.0f};
    if (spatial_rank >= 3)
      td = linear_tap(od, s.out_d, s.in_d, coord_transform, s.s_d);
    if (spatial_rank >= 2)
      th = linear_tap(oh, s.out_h, s.in_h, coord_transform, s.s_h);

    const T* base = input + static_cast<int64_t>(nc) * s.in_d * plane;
    const T* r00 = base + td.lo * plane + static_cast<int64_t>(th.lo) * s.in_w;
    const T* r01 = base + td.lo * plane + static_cast<int64_t>(th.hi) * s.in_w;
    const T* r10 = base + td.hi * plane + static_cast<int64_t>(th.lo) * s.in_w;
    const T* r11 = base + td.hi * plane + static_cast<int64_t>(th.hi) * s.in_w;

    T* dst = output + static_cast<int64_t>(row) * s.out_w;

    for (int lane = first_lane(); lane < s.lanes; lane += lane_stride()) {
      int w0 = lane * VEC;
      VecOf<T, VEC> out;
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
#pragma clang fp contract(off)
        LinearTap tw =
            linear_tap(w0 + k, s.out_w, s.in_w, coord_transform, s.s_w);
        float v = lerp_w(r00, tw);
        if (spatial_rank >= 2) {
          v = v * (1.0f - th.t) + lerp_w(r01, tw) * th.t;
          if (spatial_rank >= 3) {
            float v1 =
                lerp_w(r10, tw) * (1.0f - th.t) + lerp_w(r11, tw) * th.t;
            v = v * (1.0f - td.t) + v1 * td.t;
          }
        }
        out.v[k] = from_float<T>(v);
      }
      *reinterpret_cast<VecOf<T, VEC>*>(dst + w0) = out;
    }
  }
}

// Moves the unused trailing axes of a rank-r window to the front, so
// (d0, .., d_{r-1}, 1, ..) becomes (1, .., d0, .., d_{r-1}) with W last.
static bool make_row_shape(int spatial_rank, int64_t N, int64_t C,
                           const int64_t in[3], const int64_t out[3],
                           const float scales[3], int vec, RowShape& s) {
  int shift = 3 - spatial_rank;
  int64_t din[3], dout[3];
  float sc[3];
  for (int i = 0; i < 3; ++i) {
    din[i] = i < shift ? 1 : in[i - shift];
    dout[i] = i < shift ? 1 : out[i - shift];
    sc[i] = i < shift ? 0.0f : scales[i - shift];
    if (din[i] > INT32_MAX || dout[i] > INT32_MAX)
      return false;
  }
  int64_t rows = N * C * dout[0] * dout[1];
  if (rows > INT32_MAX)
    return false;
  s = {static_cast<int>(din[0]),  static_cast<int>(din[1]),
       static_cast<int>(din[2]),  static_cast<int>(dout[0]),
       static_cast<int>(dout[1]), static_cast<int>(dout[2]),
       static_cast<int>(rows),    static_cast<int>(dout[2] / vec),
       sc[0],                     sc[1],
       sc[2]};
  return true;
}

// A smaller cap lets a test cover the row and lane loops with small tensors.
#ifndef HIP_RESIZE_MAX_GRID
#define HIP_RESIZE_MAX_GRID 65535u
#endif

// The block spans the row's lanes (up to 256 threads) and packs several rows
// when a row needs fewer lanes.
static void row_launch_config(const RowShape& s, dim3& grid, dim3& block) {
  unsigned bx = 32;
  while (bx < static_cast<unsigned>(s.lanes) && bx < 256)
    bx *= 2;
  unsigned by = 256 / bx;
  unsigned row_blocks = (static_cast<unsigned>(s.rows) + by - 1) / by;
  unsigned lane_blocks = (static_cast<unsigned>(s.lanes) + bx - 1) / bx;
  block = dim3(bx, by);
  grid = dim3(row_blocks < HIP_RESIZE_MAX_GRID ? row_blocks
                                               : HIP_RESIZE_MAX_GRID,
              lane_blocks < HIP_RESIZE_MAX_GRID ? lane_blocks
                                                : HIP_RESIZE_MAX_GRID);
}

// The widest store of 16, 8, 4 or 2 bytes that tiles the row and keeps every
// row start aligned; else one element per store.
template <typename T>
static int pick_vec(int64_t out_w, const void* output) {
  for (int bytes = 16; bytes >= 2; bytes /= 2) {
    if (bytes < static_cast<int>(sizeof(T)))
      break;
    int vec = bytes / static_cast<int>(sizeof(T));
    if (vec > 1 && out_w % vec == 0 &&
        reinterpret_cast<uintptr_t>(output) % bytes == 0)
      return vec;
  }
  return 1;
}

// Launches the <T, VEC> instantiation that matches `vec`, a power of two up
// to 16 bytes.
template <typename T, bool kLinear, int VEC>
static void launch_vec(int vec, dim3 grid, dim3 block, hipStream_t stream,
                       const T* x, T* y, const RowShape& s, int spatial_rank,
                       int coord_transform, int nearest_mode) {
  if constexpr (VEC > 1) {
    if (vec != VEC) {
      launch_vec<T, kLinear, VEC / 2>(vec, grid, block, stream, x, y, s,
                                      spatial_rank, coord_transform,
                                      nearest_mode);
      return;
    }
  }
  if constexpr (kLinear)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(resize_linear_rows_kernel<T, VEC>),
                       grid, block, 0, stream, x, y, s, spatial_rank,
                       coord_transform);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(resize_nearest_rows_kernel<T, VEC>),
                       grid, block, 0, stream, x, y, s, coord_transform,
                       nearest_mode);
}

// Integer element types instantiate only the nearest kernel.
template <typename T, bool kLinear>
static int launch_rows(hipStream_t stream, const void* input, void* output,
                       int spatial_rank, int64_t N, int64_t C,
                       int64_t in0, int64_t in1, int64_t in2,
                       int64_t out0, int64_t out1, int64_t out2,
                       int coord_transform, int nearest_mode,
                       float s0, float s1, float s2) {
  if (N * C * out0 * out1 * out2 <= 0) return 0;
  const int64_t in[3] = {in0, in1, in2};
  const int64_t out[3] = {out0, out1, out2};
  const float scales[3] = {s0, s1, s2};
  int vec = pick_vec<T>(out[spatial_rank - 1], output);
  RowShape s;
  if (!make_row_shape(spatial_rank, N, C, in, out, scales, vec, s)) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: extents exceed the int range\n");
    return -1;
  }
  dim3 grid, block;
  row_launch_config(s, grid, block);

  (void)hipGetLastError();
  launch_vec<T, kLinear, 16 / sizeof(T)>(
      vec, grid, block, stream, static_cast<const T*>(input),
      static_cast<T*>(output), s, spatial_rank, coord_transform, nearest_mode);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

template <typename T>
static int launch_nearest(hipStream_t stream, const void* input, void* output,
                          int spatial_rank, int64_t N, int64_t C,
                          int64_t in0, int64_t in1, int64_t in2,
                          int64_t out0, int64_t out1, int64_t out2,
                          int coord_transform, int nearest_mode,
                          float s0, float s1, float s2) {
  return launch_rows<T, false>(stream, input, output, spatial_rank, N, C, in0,
                               in1, in2, out0, out1, out2, coord_transform,
                               nearest_mode, s0, s1, s2);
}

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  int spatial_rank, int64_t N, int64_t C,
                  int64_t in0, int64_t in1, int64_t in2,
                  int64_t out0, int64_t out1, int64_t out2,
                  int mode, int coord_transform, int nearest_mode,
                  float s0, float s1, float s2) {
  if (mode == 0)
    return launch_nearest<T>(stream, input, output, spatial_rank, N, C, in0,
                             in1, in2, out0, out1, out2, coord_transform,
                             nearest_mode, s0, s1, s2);
  return launch_rows<T, true>(stream, input, output, spatial_rank, N, C, in0,
                              in1, in2, out0, out1, out2, coord_transform,
                              nearest_mode, s0, s1, s2);
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
  if (mode != 0 && mode != 1) {
    fprintf(stderr, "[custom_kernels] hip_resize: bad mode=%d\n", mode);
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
    case HIP_DTYPE_UINT8:
      if (mode != 0) {
        fprintf(stderr,
                "[custom_kernels] hip_resize: integer dtype=%d supports "
                "nearest only\n",
                hip_dtype);
        return -1;
      }
      if (hip_dtype == HIP_DTYPE_UINT8) {
        return launch_nearest<uint8_t>(
            hip_stream, input, output, spatial_rank, N, C, in_d0, in_d1,
            in_d2, out_d0, out_d1, out_d2, coord_transform, nearest_mode,
            s0, s1, s2);
      }
      return launch_nearest<int8_t>(
          hip_stream, input, output, spatial_rank, N, C, in_d0, in_d1, in_d2,
          out_d0, out_d1, out_d2, coord_transform, nearest_mode, s0, s1, s2);
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
