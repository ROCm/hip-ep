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
 *
 * hip_skip_rms_norm_quant additionally writes the per-group int8 copy of y
 * that hip.matmul_nbits reads on its dp4a path, saving the separate
 * hip.quantize_act launch and its full re-read of y from memory. Rows of up
 * to 8192 elements with 16-byte aligned tensors take a single-pass kernel
 * that keeps the row in registers and quantizes every group in parallel;
 * other rows fall back to the two-pass kernel followed by a per-group
 * re-read of y.
 */

#include "hip_arch_compat.h"
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

// Quantizes one fp16 row, already written by this block, into the int8
// activation layout of matmul_nbits_quant_act_i8_kernel: one wave per
// `block_size` group, scale = max|y| / 127, q = rint(y * 127 / max|y|).
// The max is order-independent and every q uses the same expression, so the
// bytes match the standalone kernel exactly.
__device__ inline void skip_quantize_row(const __half *y_row, int8_t *q_row,
                                         float *s_row, int64_t norm_size,
                                         int64_t block_size) {
  constexpr int kWave = HIPDNN_WAVE_SIZE;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int wave = static_cast<int>(threadIdx.x) / kWave;
  const int num_waves = static_cast<int>(blockDim.x) / kWave;
  const int64_t k_blocks = norm_size / block_size;
  for (int64_t grp = wave; grp < k_blocks; grp += num_waves) {
    const int64_t g0 = grp * block_size;
    const int64_t g1 = g0 + block_size;
    float gmax = 0.0f;
    for (int64_t k = g0 + lane; k < g1; k += kWave)
      gmax = fmaxf(gmax, fabsf(__half2float(y_row[k])));
    for (int off = kWave >> 1; off > 0; off >>= 1)
      gmax = fmaxf(gmax, __shfl_xor(gmax, off));
    const float inv_scale = (gmax > 0.0f) ? (127.0f / gmax) : 0.0f;
    if (lane == 0)
      s_row[grp] = (gmax > 0.0f) ? (gmax * (1.0f / 127.0f)) : 1.0f;
    for (int64_t k = g0 + lane; k < g1; k += kWave) {
      int q = __float2int_rn(__half2float(y_row[k]) * inv_scale);
      q = max(-127, min(127, q));
      const int64_t c = k >> 3;
      const int p = static_cast<int>(k & 7);
      const int64_t byte_off =
          (p & 1) ? (c * 8 + 4 + (p >> 1)) : (c * 8 + (p >> 1));
      q_row[byte_off] = static_cast<int8_t>(q);
    }
  }
}

// One block per row. blockDim.x == kSkipRmsNormBlockSize.
// `bias` and `sum_out` may be null. VEC2 requires an even norm_size and
// 4-byte aligned bases on every non-null pointer; the launcher checks both.
// QUANT (T = __half only) also writes y quantized per `quant_block_size`
// group into a_quant / a_scale; the launcher checks the block size divides
// norm_size and is a power of two >= 32.
template <typename T, bool VEC2, bool QUANT = false>
__global__ void
hip_skip_rms_norm_kernel(const T *__restrict__ input,
                         const T *__restrict__ skip,
                         const T *__restrict__ gamma,
                         const T *__restrict__ bias, T *__restrict__ output,
                         T *__restrict__ sum_out, int64_t outer,
                         int64_t norm_size, float epsilon,
                         int8_t *__restrict__ a_quant = nullptr,
                         float *__restrict__ a_scale = nullptr,
                         int64_t quant_block_size = 0) {
  static_assert(!QUANT || std::is_same_v<T, __half>, "QUANT reads fp16 y");
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

  if constexpr (QUANT) {
    // Groups span lanes of different waves, so the row must be complete and
    // visible block-wide before any wave reads it back.
    __syncthreads();
    skip_quantize_row(y_row, a_quant + row * norm_size,
                      a_scale + row * (norm_size / quant_block_size),
                      norm_size, quant_block_size);
  }
}

// Register-resident QUANT path. Thread t owns the 8 contiguous elements
// k = (it * 256 + t) * 8 .. k + 7 of iteration `it`, so the row stays in
// registers between the sum, the norm and the quantize. Those 8 elements are
// exactly one 8-byte chunk of the quant_act layout (even elements in bytes
// 0-3, odd in 4-7), and a quant group is `quant_block_size / 8` aligned lanes
// of one wave in one iteration, so each group max is a few lane shuffles and
// every group quantizes in parallel. The launcher guarantees norm_size <=
// ITERS * kSkipQuantSpan, quant_block_size <= 8 * 32, and 16-byte aligned
// fp16 rows (8-byte for a_quant).
static constexpr int kSkipQuantVec = 8;
static constexpr int kSkipQuantSpan = kSkipRmsNormBlockSize * kSkipQuantVec;
static constexpr int kSkipQuantMaxIters = 4;
static constexpr int kSkipQuantMaxBlock = kSkipQuantVec * 32;

__device__ inline void skip_unpack8(const uint4 &v, float f[8]) {
  const __half2 *h = reinterpret_cast<const __half2 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float2 p = __half22float2(h[j]);
    f[2 * j] = p.x;
    f[2 * j + 1] = p.y;
  }
}

__device__ inline uint4 skip_pack8(const float f[8]) {
  uint4 v;
  __half2 *h = reinterpret_cast<__half2 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++)
    h[j] = __float22half2_rn(make_float2(f[2 * j], f[2 * j + 1]));
  return v;
}

__device__ inline uint32_t skip_quant_byte(float y, float inv_scale) {
  int q = __float2int_rn(y * inv_scale);
  q = max(-127, min(127, q));
  return static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int8_t>(q)));
}

template <int ITERS>
__global__ void __launch_bounds__(kSkipRmsNormBlockSize)
    hip_skip_rms_norm_quant_reg_kernel(
        const __half *__restrict__ input, const __half *__restrict__ skip,
        const __half *__restrict__ gamma, const __half *__restrict__ bias,
        __half *__restrict__ output, __half *__restrict__ sum_out,
        int64_t norm_size, float epsilon, int8_t *__restrict__ a_quant,
        float *__restrict__ a_scale, int quant_block_size) {
  constexpr int kWave = HIPDNN_WAVE_SIZE;
  constexpr int kWaves = kSkipRmsNormBlockSize / kWave;
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * norm_size;

  float t[ITERS][kSkipQuantVec];
  uint4 g_raw[ITERS];
  float local_sumsq = 0.0f;
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int64_t k =
        (static_cast<int64_t>(it) * kSkipRmsNormBlockSize + tid) * kSkipQuantVec;
    if (k < norm_size) {
      float x[kSkipQuantVec], s[kSkipQuantVec];
      skip_unpack8(*reinterpret_cast<const uint4 *>(input + base + k), x);
      skip_unpack8(*reinterpret_cast<const uint4 *>(skip + base + k), s);
      g_raw[it] = *reinterpret_cast<const uint4 *>(gamma + k);
#pragma unroll
      for (int j = 0; j < kSkipQuantVec; j++)
        t[it][j] = x[j] + s[j];
      if (bias) {
        float b[kSkipQuantVec];
        skip_unpack8(*reinterpret_cast<const uint4 *>(bias + k), b);
#pragma unroll
        for (int j = 0; j < kSkipQuantVec; j++)
          t[it][j] += b[j];
      }
#pragma unroll
      for (int j = 0; j < kSkipQuantVec; j += 2)
        local_sumsq += t[it][j] * t[it][j] + t[it][j + 1] * t[it][j + 1];
      // As in the two-pass kernel, a written sum is normalized in its rounded
      // form, while the sum of squares uses the unrounded one.
      if (sum_out) {
        const uint4 p = skip_pack8(t[it]);
        *reinterpret_cast<uint4 *>(sum_out + base + k) = p;
        skip_unpack8(p, t[it]);
      }
    }
  }

  for (int off = kWave >> 1; off > 0; off >>= 1)
    local_sumsq += __shfl_xor(local_sumsq, off);
  __shared__ float s_part[kWaves];
  if (tid % kWave == 0)
    s_part[tid / kWave] = local_sumsq;
  __syncthreads();
  float sumsq = 0.0f;
#pragma unroll
  for (int w = 0; w < kWaves; w++)
    sumsq += s_part[w];
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(norm_size) + epsilon);

  const int group_lanes = quant_block_size / kSkipQuantVec;
  int8_t *q_row = a_quant + base;
  float *scale_row = a_scale + blockIdx.x * (norm_size / quant_block_size);
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int64_t k =
        (static_cast<int64_t>(it) * kSkipRmsNormBlockSize + tid) * kSkipQuantVec;
    const bool active = k < norm_size;
    float y[kSkipQuantVec];
    float gmax = 0.0f;
    uint4 y_raw;
    if (active) {
      float g[kSkipQuantVec];
      skip_unpack8(g_raw[it], g);
#pragma unroll
      for (int j = 0; j < kSkipQuantVec; j++)
        y[j] = t[it][j] * inv_rms * g[j];
      y_raw = skip_pack8(y);
      // Quantize the fp16 values actually stored, as quant_act would read them.
      skip_unpack8(y_raw, y);
#pragma unroll
      for (int j = 0; j < kSkipQuantVec; j++)
        gmax = fmaxf(gmax, fabsf(y[j]));
    }
    // Groups are aligned to group_lanes and never straddle the active edge,
    // so every lane can shuffle unconditionally.
    for (int off = group_lanes >> 1; off > 0; off >>= 1)
      gmax = fmaxf(gmax, __shfl_xor(gmax, off));
    if (active) {
      *reinterpret_cast<uint4 *>(output + base + k) = y_raw;
      const float inv_scale = (gmax > 0.0f) ? (127.0f / gmax) : 0.0f;
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        lo |= skip_quant_byte(y[2 * j], inv_scale) << (8 * j);
        hi |= skip_quant_byte(y[2 * j + 1], inv_scale) << (8 * j);
      }
      *reinterpret_cast<uint2 *>(q_row + k) = make_uint2(lo, hi);
      if ((tid & (group_lanes - 1)) == 0)
        scale_row[k / quant_block_size] =
            (gmax > 0.0f) ? (gmax * (1.0f / 127.0f)) : 1.0f;
    }
  }
}

template <int ITERS>
void skip_launch_quant_reg(hipStream_t stream, int64_t outer,
                           const void *input, const void *skip,
                           const void *gamma, const void *bias, void *output,
                           void *sum_out, int64_t norm_size, float epsilon,
                           void *a_quant, void *a_scale,
                           int64_t quant_block_size) {
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(hip_skip_rms_norm_quant_reg_kernel<ITERS>),
      dim3(static_cast<unsigned int>(outer)), dim3(kSkipRmsNormBlockSize), 0,
      stream, static_cast<const __half *>(input),
      static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
      static_cast<const __half *>(bias), static_cast<__half *>(output),
      static_cast<__half *>(sum_out), norm_size, epsilon,
      static_cast<int8_t *>(a_quant), static_cast<float *>(a_scale),
      static_cast<int>(quant_block_size));
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

extern "C" int hip_skip_rms_norm_quant(
    void *stream, const void *input, const void *skip, const void *gamma,
    const void *bias, void *output, void *input_skip_bias_sum, int64_t outer,
    int64_t norm_size, float epsilon, void *a_quant, void *a_scale,
    int64_t quant_block_size) {
  if (!a_quant || !a_scale || quant_block_size < 32 ||
      (quant_block_size & (quant_block_size - 1)) != 0 ||
      norm_size % quant_block_size != 0)
    return static_cast<int>(hipErrorInvalidValue);
  if (outer <= 0 || norm_size <= 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kSkipRmsNormBlockSize);
  const bool vec2 = skip_can_vectorize(input, skip, gamma, bias, output,
                                       input_skip_bias_sum, norm_size);
  auto aligned = [](const void *p, uintptr_t a) {
    return p == nullptr || (reinterpret_cast<uintptr_t>(p) % a) == 0;
  };
  const int64_t iters = (norm_size + kSkipQuantSpan - 1) / kSkipQuantSpan;
  const bool reg = iters <= kSkipQuantMaxIters &&
                   quant_block_size <= kSkipQuantMaxBlock &&
                   aligned(input, 16) && aligned(skip, 16) &&
                   aligned(gamma, 16) && aligned(bias, 16) &&
                   aligned(output, 16) && aligned(input_skip_bias_sum, 16) &&
                   aligned(a_quant, 8);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_skip_rms_norm_quant: outer=%lld, norm_size=%lld, "
      "quant_block_size=%lld, path=%s\n",
      (long long)outer, (long long)norm_size, (long long)quant_block_size,
      reg ? "reg" : (vec2 ? "vec2" : "scalar"));

  (void)hipGetLastError();
  if (reg) {
    auto launch = skip_launch_quant_reg<kSkipQuantMaxIters>;
    switch (iters) {
    case 1: launch = skip_launch_quant_reg<1>; break;
    case 2: launch = skip_launch_quant_reg<2>; break;
    case 3: launch = skip_launch_quant_reg<3>; break;
    default: break;
    }
    launch(hip_stream, outer, input, skip, gamma, bias, output,
           input_skip_bias_sum, norm_size, epsilon, a_quant, a_scale,
           quant_block_size);
  } else if (vec2) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_skip_rms_norm_kernel<__half, true, true>), grid,
        block, 0, hip_stream, static_cast<const __half *>(input),
        static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
        static_cast<const __half *>(bias), static_cast<__half *>(output),
        static_cast<__half *>(input_skip_bias_sum), outer, norm_size, epsilon,
        static_cast<int8_t *>(a_quant), static_cast<float *>(a_scale),
        quant_block_size);
  } else {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_skip_rms_norm_kernel<__half, false, true>), grid,
        block, 0, hip_stream, static_cast<const __half *>(input),
        static_cast<const __half *>(skip), static_cast<const __half *>(gamma),
        static_cast<const __half *>(bias), static_cast<__half *>(output),
        static_cast<__half *>(input_skip_bias_sum), outer, norm_size, epsilon,
        static_cast<int8_t *>(a_quant), static_cast<float *>(a_scale),
        quant_block_size);
  }
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_skip_rms_norm_quant launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
