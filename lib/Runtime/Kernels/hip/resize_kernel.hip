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
// Resize, one extent pair per axis
// =============================================================================
//
// One thread per output element.  The buffer is row-major over `rank` axes
// (at most kMaxResizeRank).  An axis whose input and output extents match is
// copied through.  An axis whose extents differ is mapped back into the input
// with coordinate_transformation_mode, then either nearest-sampled (mode=0)
// or included in the N-linear corner product (mode=1).  Scale on a resized
// axis is `out_dim / in_dim`.

constexpr int kMaxResizeRank = 5;

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

__device__ inline int64_t clamp_axis(int64_t index, int64_t extent) {
  if (extent <= 0)
    return 0;
  if (index < 0)
    return 0;
  if (index >= extent)
    return extent - 1;
  return index;
}

// Row-major coordinate of `tid` over the first `rank` output extents.
__device__ inline void decode_output_index(int64_t tid, int rank,
                                           const int64_t* out_lens,
                                           int64_t* coord) {
  for (int axis = 0; axis < rank; ++axis) {
    int64_t stride = 1;
    for (int inner = axis + 1; inner < rank; ++inner)
      stride *= out_lens[inner];
    int64_t index = stride > 0 ? tid / stride : 0;
    coord[axis] = index;
    tid -= index * stride;
  }
}

__device__ inline int64_t flat_offset(int rank, const int64_t* lens,
                                      const int64_t* index) {
  int64_t offset = 0;
  int64_t stride = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    offset += index[axis] * stride;
    stride *= lens[axis];
  }
  return offset;
}

template <typename T>
__global__ void resize_nearest_kernel(const T* __restrict__ input,
                                      T* __restrict__ output,
                                      int rank,
                                      int64_t in0, int64_t in1, int64_t in2,
                                      int64_t in3, int64_t in4,
                                      int64_t out0, int64_t out1, int64_t out2,
                                      int64_t out3, int64_t out4,
                                      int coord_transform,
                                      int nearest_mode,
                                      int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  const int64_t in_lens[kMaxResizeRank] = {in0, in1, in2, in3, in4};
  const int64_t out_lens[kMaxResizeRank] = {out0, out1, out2, out3, out4};
  int64_t coord[kMaxResizeRank];
  int64_t index[kMaxResizeRank];
  decode_output_index(tid, rank, out_lens, coord);

  for (int axis = 0; axis < rank; ++axis) {
    if (in_lens[axis] == out_lens[axis] || in_lens[axis] <= 0) {
      index[axis] = clamp_axis(coord[axis], in_lens[axis]);
      continue;
    }
    float x = coord_to_input(static_cast<int>(coord[axis]),
                             static_cast<int>(out_lens[axis]),
                             static_cast<int>(in_lens[axis]),
                             coord_transform);
    index[axis] = clamp_axis(nearest_round(x, nearest_mode), in_lens[axis]);
  }
  output[tid] = input[flat_offset(rank, in_lens, index)];
}

template <typename T>
__global__ void resize_linear_kernel(const T* __restrict__ input,
                                     T* __restrict__ output,
                                     int rank,
                                     int64_t in0, int64_t in1, int64_t in2,
                                     int64_t in3, int64_t in4,
                                     int64_t out0, int64_t out1, int64_t out2,
                                     int64_t out3, int64_t out4,
                                     int coord_transform,
                                     int64_t total_outputs) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_outputs) return;

  const int64_t in_lens[kMaxResizeRank] = {in0, in1, in2, in3, in4};
  const int64_t out_lens[kMaxResizeRank] = {out0, out1, out2, out3, out4};
  int64_t coord[kMaxResizeRank];
  decode_output_index(tid, rank, out_lens, coord);

  // Pass-through axes stay at the output coordinate.  Only axes whose
  // extents differ enter the 2^k corner product.
  int64_t lo[kMaxResizeRank];
  float frac[kMaxResizeRank];
  int active[kMaxResizeRank];
  int n_active = 0;
  for (int axis = 0; axis < rank; ++axis) {
    if (in_lens[axis] == out_lens[axis] || in_lens[axis] <= 0) {
      lo[axis] = clamp_axis(coord[axis], in_lens[axis]);
      frac[axis] = 0.0f;
      continue;
    }
    float x = coord_to_input(static_cast<int>(coord[axis]),
                             static_cast<int>(out_lens[axis]),
                             static_cast<int>(in_lens[axis]),
                             coord_transform);
    float flo = floorf(x);
    lo[axis] = static_cast<int64_t>(flo);
    frac[axis] = x - flo;
    active[n_active++] = axis;
  }

  float acc = 0.0f;
  const int corners = 1 << n_active;
  for (int mask = 0; mask < corners; ++mask) {
    float weight = 1.0f;
    int64_t index[kMaxResizeRank];
    for (int axis = 0; axis < rank; ++axis)
      index[axis] = lo[axis];
    for (int bit = 0; bit < n_active; ++bit) {
      int axis = active[bit];
      if (mask & (1 << bit)) {
        weight *= frac[axis];
        index[axis] = lo[axis] + 1;
      } else {
        weight *= 1.0f - frac[axis];
      }
    }
    if (weight == 0.0f)
      continue;
    for (int axis = 0; axis < rank; ++axis)
      index[axis] = clamp_axis(index[axis], in_lens[axis]);
    acc += weight * to_float<T>(input[flat_offset(rank, in_lens, index)]);
  }
  output[tid] = from_float<T>(acc);
}

template <typename T>
static int launch(hipStream_t stream, const void* input, void* output,
                  int rank,
                  int64_t in0, int64_t in1, int64_t in2, int64_t in3, int64_t in4,
                  int64_t out0, int64_t out1, int64_t out2, int64_t out3,
                  int64_t out4,
                  int mode, int coord_transform, int nearest_mode) {
  const int64_t out_lens[kMaxResizeRank] = {out0, out1, out2, out3, out4};
  int64_t total = 1;
  for (int axis = 0; axis < rank; ++axis)
    total *= out_lens[axis];
  if (total <= 0) return 0;
  int block = 256;
  int64_t grid = (total + block - 1) / block;

  (void)hipGetLastError();
  if (mode == 0) {
    hipLaunchKernelGGL(resize_nearest_kernel<T>,
                       dim3(static_cast<unsigned>(grid)), dim3(block),
                       0, stream,
                       static_cast<const T*>(input), static_cast<T*>(output),
                       rank, in0, in1, in2, in3, in4,
                       out0, out1, out2, out3, out4,
                       coord_transform, nearest_mode, total);
  } else {
    hipLaunchKernelGGL(resize_linear_kernel<T>,
                       dim3(static_cast<unsigned>(grid)), dim3(block),
                       0, stream,
                       static_cast<const T*>(input), static_cast<T*>(output),
                       rank, in0, in1, in2, in3, in4,
                       out0, out1, out2, out3, out4,
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
                          int rank,
                          int64_t in_d0, int64_t in_d1, int64_t in_d2,
                          int64_t in_d3, int64_t in_d4,
                          int64_t out_d0, int64_t out_d1, int64_t out_d2,
                          int64_t out_d3, int64_t out_d4,
                          int mode,
                          int coord_transform,
                          int nearest_mode) {
  if (rank < 1 || rank > kMaxResizeRank) {
    fprintf(stderr,
            "[custom_kernels] hip_resize: unsupported rank=%d\n", rank);
    return -1;
  }
  if (mode != 0 && mode != 1) {
    fprintf(stderr, "[custom_kernels] hip_resize: bad mode=%d\n", mode);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_resize: dtype=%d rank=%d "
      "in=[%lld,%lld,%lld,%lld,%lld] out=[%lld,%lld,%lld,%lld,%lld] "
      "mode=%d ct=%d nearest=%d\n",
      hip_dtype, rank,
      (long long)in_d0, (long long)in_d1, (long long)in_d2,
      (long long)in_d3, (long long)in_d4,
      (long long)out_d0, (long long)out_d1, (long long)out_d2,
      (long long)out_d3, (long long)out_d4,
      mode, coord_transform, nearest_mode);

  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      return launch<float>(hip_stream, input, output, rank,
                           in_d0, in_d1, in_d2, in_d3, in_d4,
                           out_d0, out_d1, out_d2, out_d3, out_d4,
                           mode, coord_transform, nearest_mode);
    case HIP_DTYPE_FLOAT16:
      return launch<__half>(hip_stream, input, output, rank,
                            in_d0, in_d1, in_d2, in_d3, in_d4,
                            out_d0, out_d1, out_d2, out_d3, out_d4,
                            mode, coord_transform, nearest_mode);
    case HIP_DTYPE_BFLOAT16:
      return launch<hip_bfloat16>(hip_stream, input, output, rank,
                                  in_d0, in_d1, in_d2, in_d3, in_d4,
                                  out_d0, out_d1, out_d2, out_d3, out_d4,
                                  mode, coord_transform, nearest_mode);
    case HIP_DTYPE_FLOAT64:
      return launch<double>(hip_stream, input, output, rank,
                            in_d0, in_d1, in_d2, in_d3, in_d4,
                            out_d0, out_d1, out_d2, out_d3, out_d4,
                            mode, coord_transform, nearest_mode);
    default:
      fprintf(stderr,
              "[custom_kernels] hip_resize: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}
