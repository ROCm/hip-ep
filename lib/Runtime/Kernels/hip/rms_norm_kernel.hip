/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * RMS normalization (ONNX RMSNormalization / SimplifiedLayerNormalization):
 * per-row root-mean-square normalization followed by an elementwise scale.
 *
 *   y = x * rsqrt(mean(x^2) + epsilon) * scale
 *
 * Layout: input is reshaped as `[outer, norm_size]` on the host
 *         (outer = input_num_elements / scale_num_elements,
 *          norm_size = scale_num_elements).
 *
 * Algorithm: one block per row, two passes with a shared-memory block
 * reduction.
 *   Pass A: each block sums x^2 over `norm_size`, in FP32.
 *   Pass B: each thread writes y in the input dtype, using FP32 math.
 *
 * FP32 accumulation is not optional: a 4096-wide sum of squares in FP16
 * saturates well before the row ends.
 *
 * The VEC2 template parameter selects a packed __half2 body. RMS norm is
 * memory-bound, so 2-wide loads/stores are the main lever; the launcher picks
 * it only when norm_size is even and every pointer is 4-byte aligned.
 *
 * Structurally mirrors layer_norm_kernel.hip (same block size, same reduction
 * shape) minus the mean subtraction and the bias term.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kRmsNormBlockSize = 256;

namespace {

__device__ inline float rms_load(const float *p) { return *p; }
__device__ inline float rms_load(const __half *p) { return __half2float(*p); }

__device__ inline void rms_store(float *p, float v) { *p = v; }
__device__ inline void rms_store(__half *p, float v) { *p = __float2half(v); }

// Block-wide sum. Every thread supplies a partial; every thread gets the total.
__device__ inline float rms_block_sum(float partial, float *s_reduce) {
  s_reduce[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  return s_reduce[0];
}

// One block per row. blockDim.x == kRmsNormBlockSize.
// VEC2 requires norm_size % 2 == 0 and 4-byte aligned rows; the launcher
// guarantees both before selecting this instantiation.
template <typename T, bool VEC2>
__global__ void hip_rms_norm_kernel(const T *__restrict__ input,
                                    const T *__restrict__ scale,
                                    T *__restrict__ output, int64_t outer,
                                    int64_t norm_size, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T *x_row = input + row * norm_size;
  T *y_row = output + row * norm_size;

  __shared__ float s_reduce[kRmsNormBlockSize];
  __shared__ float s_inv_rms;

  float local_sumsq = 0.0f;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 v = __half22float2(x2[i]);
      local_sumsq += v.x * v.x + v.y * v.y;
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float v = rms_load(x_row + i);
      local_sumsq += v * v;
    }
  }

  float sumsq = rms_block_sum(local_sumsq, s_reduce);
  if (threadIdx.x == 0)
    s_inv_rms = rsqrtf(sumsq / static_cast<float>(norm_size) + epsilon);
  __syncthreads();

  const float inv_rms = s_inv_rms;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    const __half2 *g2 = reinterpret_cast<const __half2 *>(scale);
    __half2 *y2 = reinterpret_cast<__half2 *>(y_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 v = __half22float2(x2[i]);
      float2 g = __half22float2(g2[i]);
      float2 r;
      r.x = v.x * inv_rms * g.x;
      r.y = v.y * inv_rms * g.y;
      y2[i] = __float22half2_rn(r);
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float v = rms_load(x_row + i);
      float g = rms_load(scale + i);
      rms_store(y_row + i, v * inv_rms * g);
    }
  }
}

// Packed __half2 requires an even row and 4-byte aligned bases. Row starts stay
// aligned across rows when norm_size is even, so checking the bases is enough.
inline bool rms_can_vectorize(const void *input, const void *scale,
                              const void *output, int64_t norm_size) {
  if (norm_size % 2 != 0)
    return false;
  auto aligned = [](const void *p) {
    return (reinterpret_cast<uintptr_t>(p) % 4) == 0;
  };
  return aligned(input) && aligned(scale) && aligned(output);
}

} // namespace

extern "C" int hip_rms_norm(void *stream, const void *input, const void *scale,
                            void *output, int64_t outer, int64_t norm_size,
                            float epsilon, int hip_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kRmsNormBlockSize);

  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to THIS
  // launch only (otherwise an upstream error is misreported as
  // "hip_rms_norm launch failed").
  (void)hipGetLastError();

  const bool vec2 =
      (hip_dtype == HIP_DTYPE_FLOAT16) &&
      rms_can_vectorize(input, scale, output, norm_size);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_rms_norm: dtype=%d, outer=%lld, norm_size=%lld, "
      "epsilon=%f, vec2=%s\n",
      hip_dtype, (long long)outer, (long long)norm_size, epsilon,
      vec2 ? "yes" : "no");

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    if (vec2) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<__half, true>),
                         grid, block, 0, hip_stream,
                         static_cast<const __half *>(input),
                         static_cast<const __half *>(scale),
                         static_cast<__half *>(output), outer, norm_size,
                         epsilon);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<__half, false>),
                         grid, block, 0, hip_stream,
                         static_cast<const __half *>(input),
                         static_cast<const __half *>(scale),
                         static_cast<__half *>(output), outer, norm_size,
                         epsilon);
    }
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<float, false>), grid,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<float *>(output), outer, norm_size, epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_rms_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
