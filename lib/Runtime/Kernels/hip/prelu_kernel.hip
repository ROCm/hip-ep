/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_PRELU_MAX_RANK 8

// Y = X >= 0 ? X : slope * X. slope is NumPy-broadcast onto X. A stride of 0
// is a broadcast axis (padded dim 1, or a shorter-rank operand).
struct PreluDimsND {
  int64_t v[HIP_PRELU_MAX_RANK];
};

template <typename Storage> struct PreluTraits;

template <> struct PreluTraits<float> {
  using Compute = float;
  __device__ static Compute load(float value) { return value; }
  __device__ static float store(Compute value) { return value; }
};

template <> struct PreluTraits<__half> {
  using Compute = float;
  __device__ static Compute load(__half value) { return __half2float(value); }
  __device__ static __half store(Compute value) { return __float2half(value); }
};

template <> struct PreluTraits<__hip_bfloat16> {
  using Compute = float;
  __device__ static Compute load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __hip_bfloat16 store(Compute value) {
    return __float2bfloat16(value);
  }
};

template <> struct PreluTraits<double> {
  using Compute = double;
  __device__ static Compute load(double value) { return value; }
  __device__ static double store(Compute value) { return value; }
};

template <typename Storage>
__global__ void prelu_nd_kernel(const Storage *__restrict__ x,
                                const Storage *__restrict__ slope,
                                Storage *__restrict__ output,
                                int64_t total_elements, int rank,
                                PreluDimsND out_shape, PreluDimsND x_stride,
                                PreluDimsND slope_stride) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  int64_t x_idx = 0;
  int64_t slope_idx = 0;
  int64_t tmp = idx;
  for (int k = rank - 1; k >= 0; --k) {
    int64_t dim = out_shape.v[k];
    int64_t coord = tmp % dim;
    tmp /= dim;
    x_idx += coord * x_stride.v[k];
    slope_idx += coord * slope_stride.v[k];
  }

  using Traits = PreluTraits<Storage>;
  typename Traits::Compute xv = Traits::load(x[x_idx]);
  typename Traits::Compute sv = Traits::load(slope[slope_idx]);
  typename Traits::Compute y =
      xv >= static_cast<typename Traits::Compute>(0) ? xv : sv * xv;
  output[idx] = Traits::store(y);
}

// Left-pad src_shape to out_rank with 1s, then row-major strides. A padded
// dim of 1 broadcasts (stride 0). Any other dim must equal the output dim.
static bool prelu_broadcast_layout(const int64_t *src_shape, int src_rank,
                                   int out_rank, const int64_t *out_shape,
                                   PreluDimsND *stride, const char *name) {
  int pad = out_rank - src_rank;
  int64_t padded[HIP_PRELU_MAX_RANK];
  for (int i = 0; i < pad; ++i)
    padded[i] = 1;
  for (int i = 0; i < src_rank; ++i)
    padded[pad + i] = src_shape[i];

  int64_t running = 1;
  for (int k = out_rank - 1; k >= 0; --k) {
    int64_t dim = padded[k];
    if (dim == 1) {
      stride->v[k] = 0;
    } else if (dim == out_shape[k]) {
      stride->v[k] = running;
    } else {
      fprintf(stderr,
              "[custom_kernels] hip_prelu: %s dim %lld is "
              "not broadcast-compatible with out dim %lld\n",
              name, (long long)dim, (long long)out_shape[k]);
      return false;
    }
    // Advance by the source extent. A broadcast axis (dim 1) must not
    // inflate the stride of the next non-broadcast axis.
    running *= dim;
  }
  return true;
}

template <typename Storage>
static int launch_prelu(hipStream_t stream, const void *x, const void *slope,
                        void *output, int64_t total, int out_rank,
                        const PreluDimsND &out_shape,
                        const PreluDimsND &x_stride,
                        const PreluDimsND &slope_stride) {
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  (void)hipGetLastError();
  hipLaunchKernelGGL(prelu_nd_kernel<Storage>, dim3(grid_size),
                     dim3(kBlockSize), 0, stream,
                     static_cast<const Storage *>(x),
                     static_cast<const Storage *>(slope),
                     static_cast<Storage *>(output), total, out_rank,
                     out_shape, x_stride, slope_stride);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_prelu launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_prelu(void *stream, const void *x, const void *slope,
                         void *output, const int64_t *x_shape, int64_t x_rank,
                         const int64_t *slope_shape, int64_t slope_rank,
                         const int64_t *out_shape, int64_t out_rank,
                         int hip_dtype) {
  // Rank 0 is a scalar and broadcasts. The shape pointer is unused then.
  if (out_rank < 0 || out_rank > HIP_PRELU_MAX_RANK || x_rank < 0 ||
      x_rank > out_rank || slope_rank < 0 || slope_rank > out_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_prelu: invalid ranks (max=%d): "
            "x=%lld, slope=%lld, out=%lld\n",
            HIP_PRELU_MAX_RANK, (long long)x_rank, (long long)slope_rank,
            (long long)out_rank);
    return -1;
  }

  int64_t total = 1;
  for (int i = 0; i < out_rank; ++i)
    total *= out_shape[i];
  if (total <= 0)
    return 0;

  PreluDimsND out_shape_nd = {};
  PreluDimsND x_stride = {};
  PreluDimsND slope_stride = {};
  for (int i = 0; i < out_rank; ++i)
    out_shape_nd.v[i] = out_shape[i];

  if (!prelu_broadcast_layout(x_shape, (int)x_rank, (int)out_rank, out_shape,
                              &x_stride, "x"))
    return -1;
  if (!prelu_broadcast_layout(slope_shape, (int)slope_rank, (int)out_rank,
                              out_shape, &slope_stride, "slope"))
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return launch_prelu<float>(hip_stream, x, slope, output, total,
                               (int)out_rank, out_shape_nd, x_stride,
                               slope_stride);
  case HIP_DTYPE_FLOAT16:
    return launch_prelu<__half>(hip_stream, x, slope, output, total,
                                (int)out_rank, out_shape_nd, x_stride,
                                slope_stride);
  case HIP_DTYPE_BFLOAT16:
    return launch_prelu<__hip_bfloat16>(hip_stream, x, slope, output, total,
                                        (int)out_rank, out_shape_nd, x_stride,
                                        slope_stride);
  case HIP_DTYPE_FLOAT64:
    return launch_prelu<double>(hip_stream, x, slope, output, total,
                                (int)out_rank, out_shape_nd, x_stride,
                                slope_stride);
  default:
    fprintf(stderr, "[custom_kernels] hip_prelu: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
