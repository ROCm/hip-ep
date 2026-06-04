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
#include <math.h>

// =============================================================================
// Global Pool (average / max / lp)
// =============================================================================
//
// Treats the input as a 2D matrix of shape [outer, reduce_size] where
//   outer       = N * C
//   reduce_size = D_1 * D_2 * ... * D_k   (product of all spatial dims)
//
// One block per output element (= per (n, c) slice). Each thread accumulates
// a partial reduction over a strided portion of its slice into a float
// accumulator, then a tree-reduce in shared memory gives the per-block
// total. The final thread applies the mode-specific finalize step (divide
// by reduce_size for AVG, identity for MAX, pow(., 1/p) for LP) and narrows
// to the storage type.
//
// Float accumulation is used for all element types (including fp32) to keep
// the kernel uniform and to avoid silent precision loss on long spatial
// reductions of fp16 / bf16 inputs.

namespace {

// Mode tags. Values mirror HIPDNN_EP_GLOBAL_POOL_* (lib/Runtime/hipdnn_ep_runtime.h)
// so the runtime can pass the same enum through unchanged.
enum GlobalPoolMode {
  kModeAverage = 0,
  kModeMax = 1,
  kModeLp = 2,
};

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

// Per-mode reduction primitives.

template <int MODE> __device__ inline float pool_init() {
  // -INFINITY is the HIP/CUDA-portable idiom; std::numeric_limits<float>::
  // infinity() is __host__-only and would fail to compile in __device__ code.
  if constexpr (MODE == kModeMax)
    return -INFINITY;
  return 0.0f;
}

template <int MODE>
__device__ inline float pool_lift(float v, int p) {
  if constexpr (MODE == kModeLp) {
    float a = fabsf(v);
    if (p == 1) return a;
    if (p == 2) return a * a;
    return __powf(a, static_cast<float>(p));
  } else {
    return v;
  }
}

template <int MODE>
__device__ inline float pool_combine(float a, float b) {
  if constexpr (MODE == kModeMax) {
    return fmaxf(a, b);
  } else {
    // AVG and LP both accumulate by addition (LP sums |x|^p).
    return a + b;
  }
}

template <int MODE>
__device__ inline float pool_finalize(float acc, int64_t reduce_size, int p) {
  if constexpr (MODE == kModeAverage) {
    return acc / static_cast<float>(reduce_size);
  } else if constexpr (MODE == kModeLp) {
    if (p == 1) return acc;
    if (p == 2) return sqrtf(acc);
    return __powf(acc, 1.0f / static_cast<float>(p));
  } else {
    // MAX: identity.
    return acc;
  }
}

template <typename T, int MODE>
__global__ void global_pool_kernel(const T* __restrict__ data,
                                   T* __restrict__ output,
                                   int64_t reduce_size, int64_t outer, int p) {
  extern __shared__ float sdata[];

  int64_t out_idx = blockIdx.x;
  if (out_idx >= outer) return;

  const T* slice = data + out_idx * reduce_size;
  int tid = threadIdx.x;

  float acc = pool_init<MODE>();
  for (int64_t i = tid; i < reduce_size; i += blockDim.x) {
    float v = pool_lift<MODE>(to_float<T>(slice[i]), p);
    acc = pool_combine<MODE>(acc, v);
  }
  sdata[tid] = acc;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (static_cast<unsigned int>(tid) < s) {
      sdata[tid] = pool_combine<MODE>(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    float v = pool_finalize<MODE>(sdata[0], reduce_size, p);
    output[out_idx] = from_float<T>(v);
  }
}

template <typename T>
int launch_for_dtype(hipStream_t stream, const void* data, void* output,
                     int64_t outer, int64_t reduce_size, int mode, int p,
                     dim3 grid, dim3 block, size_t shared_mem) {
  switch (mode) {
    case kModeAverage:
      hipLaunchKernelGGL((global_pool_kernel<T, kModeAverage>), grid, block,
                         shared_mem, stream, static_cast<const T*>(data),
                         static_cast<T*>(output), reduce_size, outer, p);
      return 0;
    case kModeMax:
      hipLaunchKernelGGL((global_pool_kernel<T, kModeMax>), grid, block,
                         shared_mem, stream, static_cast<const T*>(data),
                         static_cast<T*>(output), reduce_size, outer, p);
      return 0;
    case kModeLp:
      hipLaunchKernelGGL((global_pool_kernel<T, kModeLp>), grid, block,
                         shared_mem, stream, static_cast<const T*>(data),
                         static_cast<T*>(output), reduce_size, outer, p);
      return 0;
    default:
      fprintf(stderr, "[custom_kernels] hip_global_pool: unsupported mode=%d\n",
              mode);
      return -1;
  }
}

} // namespace

extern "C" int hip_global_pool(void* stream, const void* data, void* output,
                               int64_t outer, int64_t reduce_size,
                               int hip_dtype, int mode, int p) {
  if (outer <= 0 || reduce_size <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Cap block size at 256, round down to the largest power of two <= reduce_size
  // for small reductions so the in-kernel tree reduction has a power-of-two
  // range (matches the reduce_sum block-size derivation).
  int block_size = 256;
  if (reduce_size < 256) {
    block_size = 1;
    while (block_size < reduce_size) block_size <<= 1;
    if (block_size < 1) block_size = 1;
  }
  size_t shared_mem = block_size * sizeof(float);
  dim3 grid(static_cast<unsigned>(outer));
  dim3 block(block_size);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_global_pool: outer=%lld, reduce_size=%lld, "
      "dtype=%d, mode=%d, p=%d, grid=%lld, block=%d, smem=%zu\n",
      (long long)outer, (long long)reduce_size, hip_dtype, mode, p,
      (long long)outer, block_size, shared_mem);

  int rc = -1;
  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      rc = launch_for_dtype<float>(hip_stream, data, output, outer,
                                   reduce_size, mode, p, grid, block,
                                   shared_mem);
      break;
    case HIP_DTYPE_FLOAT16:
      rc = launch_for_dtype<__half>(hip_stream, data, output, outer,
                                    reduce_size, mode, p, grid, block,
                                    shared_mem);
      break;
    case HIP_DTYPE_BFLOAT16:
      rc = launch_for_dtype<hip_bfloat16>(hip_stream, data, output, outer,
                                          reduce_size, mode, p, grid, block,
                                          shared_mem);
      break;
    case HIP_DTYPE_FLOAT64:
      rc = launch_for_dtype<double>(hip_stream, data, output, outer,
                                    reduce_size, mode, p, grid, block,
                                    shared_mem);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_global_pool: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }

  if (rc != 0) return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_global_pool launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
