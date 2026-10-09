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
#include <type_traits>

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
// Each tensor has its own element type (bias shares skip's) so the fp32
// residual stream of an fp16 model can be normalized without separate Casts.
template <typename TIn, typename TSkip, typename TG, typename TOut,
          typename TSum, bool VEC2>
__global__ void
hip_skip_rms_norm_kernel(const TIn *__restrict__ input,
                         const TSkip *__restrict__ skip,
                         const TG *__restrict__ gamma,
                         const TSkip *__restrict__ bias,
                         TOut *__restrict__ output, TSum *__restrict__ sum_out,
                         int64_t outer, int64_t norm_size, float epsilon) {
  static_assert(!VEC2 || (std::is_same<TIn, __half>::value &&
                          std::is_same<TSkip, __half>::value &&
                          std::is_same<TG, __half>::value &&
                          std::is_same<TOut, __half>::value &&
                          std::is_same<TSum, __half>::value),
                "the packed body is fp16-only");
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const TIn *x_row = input + row * norm_size;
  const TSkip *s_row = skip + row * norm_size;
  TOut *y_row = output + row * norm_size;
  TSum *sum_row = sum_out ? sum_out + row * norm_size : nullptr;

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

//===----------------------------------------------------------------------===//
// Register-resident row kernel
//===----------------------------------------------------------------------===//
//
// For rows of up to kRowMaxVec * 4 * kSkipRmsNormBlockSize elements each
// thread holds its MAXV 4-wide chunks of sum (and gamma) in registers, so the
// row is read once and reduced once (wave shuffle, then one LDS step across
// waves). The two-pass kernel above re-reads the row and runs a log2(block)
// shared-memory tree with a barrier per level, which on a single decode row
// (gpt-oss: one 2880-wide row per call) costs ~3x the launch floor.
//
// Numerics follow the two-pass kernel exactly: sum^2 accumulates the fp32
// sum, and when `sum_out` is written the normalization reads the sum rounded
// to its storage type.

static constexpr int kRowMaxVec = 8;

__device__ inline float4 row_ld4(const float *p, int64_t i) {
  return reinterpret_cast<const float4 *>(p)[i];
}
__device__ inline float4 row_ld4(const __half *p, int64_t i) {
  const uint2 u = reinterpret_cast<const uint2 *>(p)[i];
  const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ inline void row_st4(float *p, int64_t i, float4 v) {
  reinterpret_cast<float4 *>(p)[i] = v;
}
__device__ inline void row_st4(__half *p, int64_t i, float4 v) {
  const __half2 a = __floats2half2_rn(v.x, v.y);
  const __half2 b = __floats2half2_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const unsigned *>(&a);
  u.y = *reinterpret_cast<const unsigned *>(&b);
  reinterpret_cast<uint2 *>(p)[i] = u;
}
__device__ inline float4 row_round(const float *, float4 v) { return v; }
__device__ inline float4 row_round(const __half *, float4 v) {
  return make_float4(__half2float(__float2half(v.x)),
                     __half2float(__float2half(v.y)),
                     __half2float(__float2half(v.z)),
                     __half2float(__float2half(v.w)));
}

// One block per row. blockDim.x == kSkipRmsNormBlockSize, norm_size % 4 == 0,
// norm_size / 4 <= MAXV * blockDim.x. `bias` and `sum_out` may be null.
// Per-tensor element types as in hip_skip_rms_norm_kernel.
template <typename TIn, typename TSkip, typename TG, typename TOut,
          typename TSum, int MAXV>
__global__ __launch_bounds__(kSkipRmsNormBlockSize) void
hip_skip_rms_norm_row_kernel(const TIn *__restrict__ input,
                             const TSkip *__restrict__ skip,
                             const TG *__restrict__ gamma,
                             const TSkip *__restrict__ bias,
                             TOut *__restrict__ output,
                             TSum *__restrict__ sum_out, int64_t norm_size,
                             float epsilon) {
  constexpr int kWaves = kSkipRmsNormBlockSize / 32;
  __shared__ float s_wave[kWaves];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * norm_size;
  const int64_t norm4 = norm_size / 4;
  const TIn *x = input + base;
  const TSkip *s = skip + base;

  float4 t[MAXV];
  float ss = 0.f;
#pragma unroll
  for (int v = 0; v < MAXV; v++) {
    const int64_t i = threadIdx.x + static_cast<int64_t>(v) * blockDim.x;
    if (i < norm4) {
      const float4 a = row_ld4(x, i);
      const float4 b = row_ld4(s, i);
      t[v] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
      if (bias) {
        const float4 c = row_ld4(bias, i);
        t[v].x += c.x;
        t[v].y += c.y;
        t[v].z += c.z;
        t[v].w += c.w;
      }
      ss += t[v].x * t[v].x + t[v].y * t[v].y + t[v].z * t[v].z +
            t[v].w * t[v].w;
      if (sum_out) {
        row_st4(sum_out + base, i, t[v]);
        t[v] = row_round(sum_out, t[v]);
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    ss += __shfl_xor(ss, off, 32);
  if ((threadIdx.x & 31) == 0)
    s_wave[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < kWaves; w++)
    tot += s_wave[w];
  const float inv_rms =
      rsqrtf(tot / static_cast<float>(norm_size) + epsilon);

  TOut *y = output + base;
#pragma unroll
  for (int v = 0; v < MAXV; v++) {
    const int64_t i = threadIdx.x + static_cast<int64_t>(v) * blockDim.x;
    if (i < norm4) {
      const float4 g = row_ld4(gamma, i);
      row_st4(y, i,
              make_float4(t[v].x * inv_rms * g.x, t[v].y * inv_rms * g.y,
                          t[v].z * inv_rms * g.z, t[v].w * inv_rms * g.w));
    }
  }
}

template <typename TIn, typename TSkip, typename TG, typename TOut,
          typename TSum>
inline bool skip_row_eligible(const void *input, const void *skip,
                              const void *gamma, const void *bias,
                              const void *output, const void *sum_out,
                              int64_t norm_size) {
  if (norm_size % 4 != 0 ||
      norm_size / 4 > static_cast<int64_t>(kRowMaxVec) * kSkipRmsNormBlockSize)
    return false;
  auto aligned = [](const void *p, size_t elem) {
    return p == nullptr || (reinterpret_cast<uintptr_t>(p) % (4 * elem)) == 0;
  };
  return aligned(input, sizeof(TIn)) && aligned(skip, sizeof(TSkip)) &&
         aligned(gamma, sizeof(TG)) && aligned(bias, sizeof(TSkip)) &&
         aligned(output, sizeof(TOut)) && aligned(sum_out, sizeof(TSum));
}

template <typename TIn, typename TSkip, typename TG, typename TOut,
          typename TSum>
inline void launch_skip_row(hipStream_t stream, const void *input,
                            const void *skip, const void *gamma,
                            const void *bias, void *output, void *sum_out,
                            int64_t outer, int64_t norm_size, float epsilon) {
  const int64_t norm4 = norm_size / 4;
  const int maxv = static_cast<int>(
      (norm4 + kSkipRmsNormBlockSize - 1) / kSkipRmsNormBlockSize);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kSkipRmsNormBlockSize);
#define SKIP_ROW_CASE(V)                                                       \
  case V:                                                                      \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(                                                       \
            hip_skip_rms_norm_row_kernel<TIn, TSkip, TG, TOut, TSum, V>),      \
        grid, block, 0, stream, static_cast<const TIn *>(input),               \
        static_cast<const TSkip *>(skip), static_cast<const TG *>(gamma),      \
        static_cast<const TSkip *>(bias), static_cast<TOut *>(output),         \
        static_cast<TSum *>(sum_out), norm_size, epsilon);                     \
    break;
  switch (maxv) {
    SKIP_ROW_CASE(1)
    SKIP_ROW_CASE(2)
    SKIP_ROW_CASE(3)
    SKIP_ROW_CASE(4)
    SKIP_ROW_CASE(5)
    SKIP_ROW_CASE(6)
    SKIP_ROW_CASE(7)
    SKIP_ROW_CASE(8)
  default:
    break;
  }
#undef SKIP_ROW_CASE
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

// Kernel choice for one dtype combination: the register-resident row kernel
// when the row fits and every tensor is 4-element aligned, else the two-pass
// kernel (packed fp16 body when everything is fp16 and 2-element aligned).
template <typename TIn, typename TSkip, typename TG, typename TOut,
          typename TSum>
int launch_skip_rms_norm(hipStream_t stream, const void *input,
                         const void *skip, const void *gamma, const void *bias,
                         void *output, void *sum_out, int64_t outer,
                         int64_t norm_size, float epsilon) {
  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to THIS
  // launch only.
  (void)hipGetLastError();

  constexpr bool kAllHalf = std::is_same<TIn, __half>::value &&
                            std::is_same<TSkip, __half>::value &&
                            std::is_same<TG, __half>::value &&
                            std::is_same<TOut, __half>::value &&
                            std::is_same<TSum, __half>::value;
  const bool row = skip_row_eligible<TIn, TSkip, TG, TOut, TSum>(
      input, skip, gamma, bias, output, sum_out, norm_size);
  const bool vec2 = kAllHalf && !row &&
                    skip_can_vectorize(input, skip, gamma, bias, output,
                                       sum_out, norm_size);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_skip_rms_norm: elem bytes in/skip/gamma/out/sum="
      "%d/%d/%d/%d/%d, outer=%lld, norm_size=%lld, bias=%s, sum_out=%s, "
      "kernel=%s\n",
      (int)sizeof(TIn), (int)sizeof(TSkip), (int)sizeof(TG), (int)sizeof(TOut),
      (int)sizeof(TSum), (long long)outer, (long long)norm_size,
      bias ? "yes" : "null", sum_out ? "yes" : "null",
      row ? "row" : (vec2 ? "two-pass vec2" : "two-pass"));

  if (row) {
    launch_skip_row<TIn, TSkip, TG, TOut, TSum>(stream, input, skip, gamma,
                                                bias, output, sum_out, outer,
                                                norm_size, epsilon);
  } else {
    dim3 grid(static_cast<unsigned int>(outer));
    dim3 block(kSkipRmsNormBlockSize);
    auto args = [&](auto kernel) {
      hipLaunchKernelGGL(kernel, grid, block, 0, stream,
                         static_cast<const TIn *>(input),
                         static_cast<const TSkip *>(skip),
                         static_cast<const TG *>(gamma),
                         static_cast<const TSkip *>(bias),
                         static_cast<TOut *>(output),
                         static_cast<TSum *>(sum_out), outer, norm_size,
                         epsilon);
    };
    if constexpr (kAllHalf) {
      if (vec2)
        args(hip_skip_rms_norm_kernel<TIn, TSkip, TG, TOut, TSum, true>);
      else
        args(hip_skip_rms_norm_kernel<TIn, TSkip, TG, TOut, TSum, false>);
    } else {
      args(hip_skip_rms_norm_kernel<TIn, TSkip, TG, TOut, TSum, false>);
    }
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_skip_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
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
  if (hip_dtype == HIP_DTYPE_FLOAT16)
    return launch_skip_rms_norm<__half, __half, __half, __half, __half>(
        hip_stream, input, skip, gamma, bias, output, input_skip_bias_sum,
        outer, norm_size, epsilon);
  if (hip_dtype == HIP_DTYPE_FLOAT32)
    return launch_skip_rms_norm<float, float, float, float, float>(
        hip_stream, input, skip, gamma, bias, output, input_skip_bias_sum,
        outer, norm_size, epsilon);
  fprintf(stderr, "[custom_kernels] hip_skip_rms_norm: unsupported dtype=%d\n",
          hip_dtype);
  return -1;
}

extern "C" int hip_skip_rms_norm_mixed(
    void *stream, const void *input, const void *skip, const void *gamma,
    const void *bias, void *output, void *input_skip_bias_sum, int64_t outer,
    int64_t norm_size, float epsilon, int input_dtype, int skip_dtype,
    int gamma_dtype, int output_dtype, int sum_dtype) {
  if (input_dtype == skip_dtype && input_dtype == gamma_dtype &&
      input_dtype == output_dtype && input_dtype == sum_dtype)
    return hip_skip_rms_norm(stream, input, skip, gamma, bias, output,
                             input_skip_bias_sum, outer, norm_size, epsilon,
                             input_dtype);
  if (outer <= 0 || norm_size <= 0)
    return 0;
  if (input_dtype == HIP_DTYPE_FLOAT32 && skip_dtype == HIP_DTYPE_FLOAT16 &&
      gamma_dtype == HIP_DTYPE_FLOAT16 && output_dtype == HIP_DTYPE_FLOAT16 &&
      sum_dtype == HIP_DTYPE_FLOAT32)
    return launch_skip_rms_norm<float, __half, __half, __half, float>(
        static_cast<hipStream_t>(stream), input, skip, gamma, bias, output,
        input_skip_bias_sum, outer, norm_size, epsilon);
  fprintf(stderr,
          "[custom_kernels] hip_skip_rms_norm_mixed: unsupported dtypes "
          "in/skip/gamma/out/sum=%d/%d/%d/%d/%d\n",
          input_dtype, skip_dtype, gamma_dtype, output_dtype, sum_dtype);
  return -1;
}
