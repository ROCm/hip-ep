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
// Nearest Resize with asymmetric coordinates and floor rounding maps an
// integer upsample directly to input[o / scale]. This avoids the floating
// coordinate transform, rounding, and clamping needed by the generic kernel.
template <typename T>
__global__ void resize_nearest_integer_2d_kernel(
    const T* __restrict__ input, T* __restrict__ output, int64_t N,
    int64_t C, int64_t in0, int64_t in1, int64_t out0, int64_t out1,
    int64_t scale0, int64_t scale1, int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  int64_t plane = out0 * out1;
  int64_t n = tid / (C * plane);
  int64_t rem = tid - n * C * plane;
  int64_t c = rem / plane;
  rem -= c * plane;
  int64_t o0 = rem / out1;
  int64_t o1 = rem - o0 * out1;

  int64_t i0 = o0 / scale0;
  int64_t i1 = o1 / scale1;
  int64_t in_off = ((n * C + c) * in0 + i0) * in1 + i1;
  output[tid] = input[in_off];
}

template <typename T>
__device__ inline float fetch_2d(const T* base, int64_t i0, int64_t i1,
                                 int64_t in0, int64_t in1) {
  if (i0 < 0) i0 = 0;
  if (i0 >= in0) i0 = in0 - 1;
  if (i1 < 0) i1 = 0;
  if (i1 >= in1) i1 = in1 - 1;
  return to_float<T>(base[i0 * in1 + i1]);
}

// Common image upsampling case: half-pixel coordinates with a uniform 2x or
// 4x scale. The output-to-input mapping uses 32-bit integer quotient/remainder
// operations, which compile to shifts and masks for Scale. The interior path
// skips the four bounds checks needed only on the image border.
template <typename T, uint32_t Scale>
__global__ void resize_linear_2d_pow2_kernel(
    const T* __restrict__ input, T* __restrict__ output, uint32_t N,
    uint32_t C, uint32_t in0, uint32_t in1, uint32_t out0, uint32_t out1,
    uint32_t total_outputs) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  uint32_t plane = out0 * out1;
  uint32_t n = tid / (C * plane);
  uint32_t rem = tid - n * C * plane;
  uint32_t c = rem / plane;
  rem -= c * plane;
  uint32_t o0 = rem / out1;
  uint32_t o1 = rem - o0 * out1;

  uint32_t r0 = o0 % Scale;
  uint32_t r1 = o1 % Scale;
  int32_t lo0 = static_cast<int32_t>(o0 / Scale) +
                ((2 * r0 + 1 < Scale) ? -1 : 0);
  int32_t lo1 = static_cast<int32_t>(o1 / Scale) +
                ((2 * r1 + 1 < Scale) ? -1 : 0);
  float t0 = static_cast<float>((2 * r0 + 1 + Scale) % (2 * Scale)) /
             static_cast<float>(2 * Scale);
  float t1 = static_cast<float>((2 * r1 + 1 + Scale) % (2 * Scale)) /
             static_cast<float>(2 * Scale);

  const T* base = input + (n * C + c) * (in0 * in1);
  float c00;
  float c01;
  float c10;
  float c11;
  if (lo0 >= 0 && lo0 + 1 < static_cast<int32_t>(in0) && lo1 >= 0 &&
      lo1 + 1 < static_cast<int32_t>(in1)) {
    const T* row0 = base + static_cast<uint32_t>(lo0) * in1;
    const T* row1 = row0 + in1;
    c00 = to_float<T>(row0[lo1]);
    c01 = to_float<T>(row0[lo1 + 1]);
    c10 = to_float<T>(row1[lo1]);
    c11 = to_float<T>(row1[lo1 + 1]);
  } else {
    c00 = fetch_2d(base, lo0, lo1, in0, in1);
    c01 = fetch_2d(base, lo0, lo1 + 1, in0, in1);
    c10 = fetch_2d(base, lo0 + 1, lo1, in0, in1);
    c11 = fetch_2d(base, lo0 + 1, lo1 + 1, in0, in1);
  }
  float c0 = c00 * (1.0f - t1) + c01 * t1;
  float c1 = c10 * (1.0f - t1) + c11 * t1;
  output[tid] = from_float<T>(c0 * (1.0f - t0) + c1 * t0);
}


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

static bool get_integer_upsample_scale(int64_t in_dim, int64_t out_dim,
                                       float supplied_scale,
                                       int64_t &integer_scale) {
  if (in_dim <= 0 || out_dim <= 0)
    return false;
  if (supplied_scale > 0.0f) {
    if (!std::isfinite(supplied_scale))
      return false;
    integer_scale = static_cast<int64_t>(supplied_scale);
    return integer_scale > 0 &&
           supplied_scale == static_cast<float>(integer_scale) &&
           out_dim / in_dim == integer_scale && out_dim % in_dim == 0;
  }
  integer_scale = out_dim / in_dim;
  return integer_scale > 0 && out_dim % in_dim == 0;
}

static bool can_launch_linear_pow2_2d(int64_t N, int64_t C, int64_t in0,
                                      int64_t in1, int64_t out0, int64_t out1,
                                      int coord_transform, float s0, float s1,
                                      int64_t &scale) {
  if ((coord_transform != 0 && coord_transform != 3) || out0 <= 1 ||
      out1 <= 1 || N <= 0 || C <= 0 || in0 <= 0 || in1 <= 0 ||
      out0 > UINT32_MAX || out1 > UINT32_MAX || in0 > UINT32_MAX ||
      in1 > UINT32_MAX || N > UINT32_MAX || C > UINT32_MAX)
    return false;
  if (static_cast<uint64_t>(N) * static_cast<uint64_t>(C) *
          static_cast<uint64_t>(out0) * static_cast<uint64_t>(out1) >
      UINT32_MAX)
    return false;
  int64_t scale0 = 0;
  int64_t scale1 = 0;
  if (!get_integer_upsample_scale(in0, out0, s0, scale0) ||
      !get_integer_upsample_scale(in1, out1, s1, scale1) || scale0 != scale1)
    return false;
  scale = scale0;
  return scale == 2 || scale == 4;
}

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  int spatial_rank, int64_t N, int64_t C,
                  int64_t in0, int64_t in1, int64_t in2,
                  int64_t out0, int64_t out1, int64_t out2,
                  int mode, int coord_transform, int nearest_mode,
                  float s0, float s1, float s2) {
  int64_t total = N * C * out0 * out1 * out2;
  if (mode == 0) {
    int64_t scale0 = 0;
    int64_t scale1 = 0;
    bool integerNearest2d =
        spatial_rank == 2 && coord_transform == 1 && nearest_mode == 2 &&
        total > 0 && total <= UINT32_MAX &&
        get_integer_upsample_scale(in0, out0, s0, scale0) &&
        get_integer_upsample_scale(in1, out1, s1, scale1);
    if (integerNearest2d) {
      int block = 256;
      int64_t grid = (total + block - 1) / block;
      (void)hipGetLastError();
      hipLaunchKernelGGL(resize_nearest_integer_2d_kernel<T>,
                         dim3(static_cast<unsigned>(grid)), dim3(block),
                         0, stream,
                         static_cast<const T*>(input),
                         static_cast<T*>(output), N, C, in0, in1, out0, out1,
                         scale0, scale1, total);
      hipError_t err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
                hipGetErrorString(err));
      }
      return static_cast<int>(err);
    }
    return launch_nearest<T>(stream, input, output, spatial_rank, N, C, in0,
                             in1, in2, out0, out1, out2, coord_transform,
                             nearest_mode, s0, s1, s2);
  }
  if (spatial_rank == 2 && total > 0) {
    int64_t scale = 0;
    if (can_launch_linear_pow2_2d(N, C, in0, in1, out0, out1,
                                  coord_transform, s0, s1, scale)) {
      constexpr int pow2Block = 256;
      int64_t pow2Grid = (total + pow2Block - 1) / pow2Block;
      (void)hipGetLastError();
      if (scale == 2) {
        hipLaunchKernelGGL(
            (resize_linear_2d_pow2_kernel<T, 2>),
            dim3(static_cast<unsigned>(pow2Grid)), dim3(pow2Block),
            0, stream, static_cast<const T*>(input),
            static_cast<T*>(output), static_cast<uint32_t>(N),
            static_cast<uint32_t>(C), static_cast<uint32_t>(in0),
            static_cast<uint32_t>(in1), static_cast<uint32_t>(out0),
            static_cast<uint32_t>(out1), static_cast<uint32_t>(total));
      } else {
        hipLaunchKernelGGL(
            (resize_linear_2d_pow2_kernel<T, 4>),
            dim3(static_cast<unsigned>(pow2Grid)), dim3(pow2Block),
            0, stream, static_cast<const T*>(input),
            static_cast<T*>(output), static_cast<uint32_t>(N),
            static_cast<uint32_t>(C), static_cast<uint32_t>(in0),
            static_cast<uint32_t>(in1), static_cast<uint32_t>(out0),
            static_cast<uint32_t>(out1), static_cast<uint32_t>(total));
      }
      hipError_t err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
                hipGetErrorString(err));
      }
      return static_cast<int>(err);
    }
  }
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
