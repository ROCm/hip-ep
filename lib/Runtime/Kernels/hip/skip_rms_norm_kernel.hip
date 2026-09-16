/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * SkipSimplifiedLayerNormalization (com.microsoft): a residual add fused with
 * RMS normalization.
 *
 *   sum = input + skip [+ bias]
 *   y   = sum * rsqrt(mean(sum^2) + epsilon) * gamma
 *
 * Layout: tensors are reshaped as `[outer, norm_size]` on the host
 *         (outer = input_num_elements / gamma_num_elements,
 *          norm_size = gamma_num_elements). `bias` and `gamma` are per-column,
 *         broadcast across rows.
 *
 * Algorithm: one block per row, two passes with a shared-memory block
 * reduction, FP32 accumulators throughout.
 *   Pass A: form sum, accumulate sum^2, and write `input_skip_bias_sum`
 *           when the caller asked for it.
 *   Pass B: re-read sum from `input_skip_bias_sum` if it was written,
 *           otherwise recompute it, then write y.
 *
 * Being able to recompute is what lets this kernel run with no workspace at
 * all. The MIOpen path it replaces needed a scratch buffer the size of the
 * whole tensor whenever input_skip_bias_sum was absent; two extra reads of
 * already-cached data beat an allocation plus two more full passes over
 * memory. When the buffer does exist, reading it back is both cheaper than
 * recomputing and matches MIOpen, which likewise normalizes the rounded
 * intermediate rather than a full-precision one.
 *
 * The VEC2 template parameter selects a packed __half2 body; this kernel
 * touches up to six tensors per row, so the 2-wide path matters more here
 * than in rms_norm_kernel.hip.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kSkipRmsNormBlockSize = 256;

namespace {

__device__ inline float skip_load(const float *p) { return *p; }
__device__ inline float skip_load(const __half *p) { return __half2float(*p); }

__device__ inline void skip_store(float *p, float v) { *p = v; }
__device__ inline void skip_store(__half *p, float v) { *p = __float2half(v); }

__device__ inline float skip_block_sum(float partial, float *s_reduce) {
  s_reduce[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  return s_reduce[0];
}

// One block per row. blockDim.x == kSkipRmsNormBlockSize.
// `bias` and `sum_out` may be null. VEC2 requires an even norm_size and
// 4-byte aligned bases on every non-null pointer; the launcher checks both.
template <typename T, bool VEC2>
__global__ void
hip_skip_rms_norm_kernel(const T *__restrict__ input,
                         const T *__restrict__ skip,
                         const T *__restrict__ gamma,
                         const T *__restrict__ bias, T *__restrict__ output,
                         T *__restrict__ sum_out, int64_t outer,
                         int64_t norm_size, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T *x_row = input + row * norm_size;
  const T *s_row = skip + row * norm_size;
  T *y_row = output + row * norm_size;
  T *sum_row = sum_out ? sum_out + row * norm_size : nullptr;

  __shared__ float s_reduce[kSkipRmsNormBlockSize];
  __shared__ float s_inv_rms;

  float local_sumsq = 0.0f;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    const __half2 *s2 = reinterpret_cast<const __half2 *>(s_row);
    const __half2 *b2 = reinterpret_cast<const __half2 *>(bias);
    __half2 *sum2 = reinterpret_cast<__half2 *>(sum_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 x = __half22float2(x2[i]);
      float2 s = __half22float2(s2[i]);
      float2 t;
      t.x = x.x + s.x;
      t.y = x.y + s.y;
      if (b2) {
        float2 b = __half22float2(b2[i]);
        t.x += b.x;
        t.y += b.y;
      }
      local_sumsq += t.x * t.x + t.y * t.y;
      if (sum2)
        sum2[i] = __float22half2_rn(t);
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float t = skip_load(x_row + i) + skip_load(s_row + i);
      if (bias)
        t += skip_load(bias + i);
      local_sumsq += t * t;
      if (sum_row)
        skip_store(sum_row + i, t);
    }
  }

  float sumsq = skip_block_sum(local_sumsq, s_reduce);
  if (threadIdx.x == 0)
    s_inv_rms = rsqrtf(sumsq / static_cast<float>(norm_size) + epsilon);
  __syncthreads();

  const float inv_rms = s_inv_rms;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    const __half2 *s2 = reinterpret_cast<const __half2 *>(s_row);
    const __half2 *b2 = reinterpret_cast<const __half2 *>(bias);
    const __half2 *g2 = reinterpret_cast<const __half2 *>(gamma);
    const __half2 *sum2 = reinterpret_cast<const __half2 *>(sum_row);
    __half2 *y2 = reinterpret_cast<__half2 *>(y_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 t;
      if (sum2) {
        t = __half22float2(sum2[i]);
      } else {
        float2 x = __half22float2(x2[i]);
        float2 s = __half22float2(s2[i]);
        t.x = x.x + s.x;
        t.y = x.y + s.y;
        if (b2) {
          float2 b = __half22float2(b2[i]);
          t.x += b.x;
          t.y += b.y;
        }
      }
      float2 g = __half22float2(g2[i]);
      float2 r;
      r.x = t.x * inv_rms * g.x;
      r.y = t.y * inv_rms * g.y;
      y2[i] = __float22half2_rn(r);
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float t;
      if (sum_row) {
        t = skip_load(sum_row + i);
      } else {
        t = skip_load(x_row + i) + skip_load(s_row + i);
        if (bias)
          t += skip_load(bias + i);
      }
      skip_store(y_row + i, t * inv_rms * skip_load(gamma + i));
    }
  }
}

// If any participating pointer fails the alignment test the whole launch
// falls back to scalar; mixing packed and scalar bodies per-tensor is not
// worth the instantiation blowup.
inline bool skip_can_vectorize(const void *input, const void *skip,
                               const void *gamma, const void *bias,
                               const void *output, const void *sum_out,
                               int64_t norm_size) {
  if (norm_size % 2 != 0)
    return false;
  auto aligned = [](const void *p) {
    return p == nullptr || (reinterpret_cast<uintptr_t>(p) % 4) == 0;
  };
  return aligned(input) && aligned(skip) && aligned(gamma) && aligned(bias) &&
         aligned(output) && aligned(sum_out);
}

} // namespace

extern "C" int hip_skip_rms_norm(void *stream, const void *input,
                                 const void *skip, const void *gamma,
                                 const void *bias, void *output,
                                 void *input_skip_bias_sum, int64_t outer,
                                 int64_t norm_size, float epsilon,
                                 int hip_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kSkipRmsNormBlockSize);

  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to THIS
  // launch only.
  (void)hipGetLastError();

  const bool vec2 = (hip_dtype == HIP_DTYPE_FLOAT16) &&
                    skip_can_vectorize(input, skip, gamma, bias, output,
                                       input_skip_bias_sum, norm_size);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_skip_rms_norm: dtype=%d, outer=%lld, "
      "norm_size=%lld, epsilon=%f, bias=%s, sum_out=%s, vec2=%s\n",
      hip_dtype, (long long)outer, (long long)norm_size, epsilon,
      bias ? "yes" : "null", input_skip_bias_sum ? "yes" : "null",
      vec2 ? "yes" : "no");

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    if (vec2) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_skip_rms_norm_kernel<__half, true>), grid, block,
          0, hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(skip),
          static_cast<const __half *>(gamma),
          static_cast<const __half *>(bias), static_cast<__half *>(output),
          static_cast<__half *>(input_skip_bias_sum), outer, norm_size,
          epsilon);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_skip_rms_norm_kernel<__half, false>), grid, block,
          0, hip_stream, static_cast<const __half *>(input),
          static_cast<const __half *>(skip),
          static_cast<const __half *>(gamma),
          static_cast<const __half *>(bias), static_cast<__half *>(output),
          static_cast<__half *>(input_skip_bias_sum), outer, norm_size,
          epsilon);
    }
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_skip_rms_norm_kernel<float, false>),
                       grid, block, 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<const float *>(skip),
                       static_cast<const float *>(gamma),
                       static_cast<const float *>(bias),
                       static_cast<float *>(output),
                       static_cast<float *>(input_skip_bias_sum), outer,
                       norm_size, epsilon);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_skip_rms_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_skip_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
