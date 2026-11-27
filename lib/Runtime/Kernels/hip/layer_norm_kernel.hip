/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * LayerNormalization (ONNX-17): per-row mean/variance normalization
 * followed by elementwise (scale, bias). Bias and the optional `mean` /
 * `inv_std` outputs may be null.
 *
 *   y = (x - mean) * rsqrt(var + epsilon) * scale + bias
 *
 * The same kernels serve the residual form (hip_skip_layer_norm), where the
 * normalized row is t = input + skip [+ bias], summed in fp32 and optionally
 * written to `input_skip_bias_sum`. With round_sum, t is rounded to the
 * element type before it is normalized, as a separate Add would; without it
 * the fp32 sum is normalized, as ORT's SkipLayerNormalization does. Input row
 * r reads skip row r mod skip_rows, so a skip of fewer rows repeats over the
 * input.
 *
 * Layout: input is reshaped as `[outer, norm_size]` on the host
 *         (outer = input_num_elements / scale_num_elements,
 *          norm_size = scale_num_elements).
 *
 * Numerics (both kernels): corrected two-pass statistics in FP32.
 *   m0:    t[0] + sum((t - t[0]) / norm_size)
 *   d:     t - m0
 *   c:     sum(d / norm_size)            (the rounding error left in m0)
 *   var:   sum(d^2) / norm_size - c^2
 *   y:     (d - c) * inv_std * scale + bias;  mean = m0 + c
 * Centering the second pass avoids the E[t^2] - E[t]^2 cancellation when
 * |mean| >> stddev (e.g. t = [8192, 8193]) and cannot overflow for rows
 * whose plain sum of squares is finite (e.g. t = [1e19, -1e19]). The
 * correction c recovers the mean when the first pass rounds, as when one
 * large sample dominates the row (e.g. [1e8, 1, 1, ...]). Scaling each term
 * before it is summed keeps the partial sums of the means within the row's
 * largest magnitude, so a row of large finite values (e.g. [1e38] * 4) does
 * not overflow to an infinite mean. Measuring m0 from t[0] makes it exact for
 * a constant row whatever norm_size is, so its deviations stay 0 instead of
 * an ulp of t whose square overflows (e.g. [1e38] * 5).
 *
 * Kernels:
 *   Row kernel (default): a group of G threads (one wave for hidden sizes up
 *     to 8 * 32 16-byte chunks, e.g. 2048 fp16 / 1024 fp32) normalizes one
 *     row, and a 256-thread block covers 256 / G rows. Each thread loads its
 *     chunks with 16-byte vector loads and keeps them in registers, so the
 *     row is read from memory once and both reductions run on registers
 *     (wave shuffles, plus one LDS step when G spans several waves). It
 *     needs norm_size to be a multiple of the chunk width and 16-byte
 *     aligned tensors.
 *   Block kernel (fallback): one 256-thread block per row with scalar loads
 *     and shared-memory tree reductions; it re-reads the row for each pass.
 *     Used for unaligned tensors, ragged hidden sizes and rows too wide for
 *     the register budget.
 *
 * Source: onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu @ v1.22.2
 *         (structurally, for the block kernel).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kLayerNormBlockSize = 256;

namespace {

__device__ inline float ln_load(const float *p) { return *p; }
__device__ inline float ln_load(const __half *p) { return __half2float(*p); }

__device__ inline void ln_store(float *p, float v) { *p = v; }
__device__ inline void ln_store(__half *p, float v) { *p = __float2half(v); }

__device__ inline float ln_round(const float *, float v) { return v; }
__device__ inline float ln_round(const __half *, float v) {
  return __half2float(__float2half(v));
}

// Stats outputs are fp16 or fp32 per stash_type, chosen at runtime.
__device__ inline void ln_store_stat(void *p, int64_t i, bool half, float v) {
  if (half)
    static_cast<__half *>(p)[i] = __float2half(v);
  else
    static_cast<float *>(p)[i] = v;
}

// Row element i: x[i], or x[i] + skip[i] [+ bias[i]] for the residual form:
// fp32 adds in ORT's order, rounded once when round_sum.
template <typename T>
__device__ inline float ln_elem(const T *x, const T *skip, const T *bias,
                                bool round_sum, int64_t i) {
  float v = ln_load(x + i);
  if (skip) {
    v += ln_load(skip + i);
    if (bias)
      v += ln_load(bias + i);
    if (round_sum)
      v = ln_round(x, v);
  }
  return v;
}

// mean(d^2) - c^2, which rounding can push just below zero. NaN stays NaN.
__device__ inline float ln_var(float mean_dsq, float c) {
  float var = mean_dsq - c * c;
  return var < 0.0f ? 0.0f : var;
}

__device__ inline float ln_block_sum(float partial, float *s_reduce) {
  s_reduce[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  float total = s_reduce[0];
  __syncthreads();
  return total;
}

//===----------------------------------------------------------------------===//
// Block kernel: one block per row, scalar loads, any norm_size / alignment.
//===----------------------------------------------------------------------===//

template <typename T>
__global__ void hip_layer_norm_block_kernel(
    const T *__restrict__ input, const T *__restrict__ skip,
    const T *__restrict__ skip_bias, const T *__restrict__ scale,
    const T *__restrict__ bias, T *__restrict__ output, T *__restrict__ sum_out,
    void *__restrict__ mean_out, void *__restrict__ inv_std_out,
    bool stats_half, bool round_sum, int64_t outer, int64_t skip_rows,
    int64_t norm_size, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const int64_t base = row * norm_size;
  const T *x_row = input + base;
  const T *s_row = skip ? skip + (row % skip_rows) * norm_size : nullptr;
  T *y_row = output + base;
  T *sum_row = sum_out ? sum_out + base : nullptr;
  const float n = static_cast<float>(norm_size);
  const float inv_n = 1.0f / n;

  __shared__ float s_reduce[kLayerNormBlockSize];

  const float t0 = ln_elem(x_row, s_row, skip_bias, round_sum, 0);
  float local_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float t = ln_elem(x_row, s_row, skip_bias, round_sum, i);
    if (sum_row)
      ln_store(sum_row + i, t);
    local_sum += (t - t0) * inv_n;
  }
  const float m0 = t0 + ln_block_sum(local_sum, s_reduce);

  float local_d = 0.0f;
  float local_dsq = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float d = ln_elem(x_row, s_row, skip_bias, round_sum, i) - m0;
    local_d += d * inv_n;
    local_dsq += d * d;
  }
  const float c = ln_block_sum(local_d, s_reduce);
  const float inv_std =
      rsqrtf(ln_var(ln_block_sum(local_dsq, s_reduce) / n, c) + epsilon);
  if (threadIdx.x == 0) {
    if (mean_out)
      ln_store_stat(mean_out, row, stats_half, m0 + c);
    if (inv_std_out)
      ln_store_stat(inv_std_out, row, stats_half, inv_std);
  }

  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float d = ln_elem(x_row, s_row, skip_bias, round_sum, i) - m0;
    float b = bias ? ln_load(bias + i) : 0.0f;
    ln_store(y_row + i, (d - c) * inv_std * ln_load(scale + i) + b);
  }
}

//===----------------------------------------------------------------------===//
// Row kernel: G threads per row, 16-byte chunks held in registers.
//===----------------------------------------------------------------------===//

static constexpr int kRowWave = 32;
static constexpr int kRowMaxChunks = 8;
static constexpr int kRowMaxGroup = 512;

template <typename T> struct Chunk {
  static constexpr int N = 16 / sizeof(T);
};

__device__ inline void ln_ld16(const float *p, int64_t c, float (&o)[4]) {
  const float4 v = reinterpret_cast<const float4 *>(p)[c];
  o[0] = v.x;
  o[1] = v.y;
  o[2] = v.z;
  o[3] = v.w;
}
__device__ inline void ln_ld16(const __half *p, int64_t c, float (&o)[8]) {
  const uint4 u = reinterpret_cast<const uint4 *>(p)[c];
  const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&w[k]));
    o[2 * k] = f.x;
    o[2 * k + 1] = f.y;
  }
}
__device__ inline void ln_st16(float *p, int64_t c, const float (&v)[4]) {
  reinterpret_cast<float4 *>(p)[c] = make_float4(v[0], v[1], v[2], v[3]);
}
__device__ inline void ln_st16(__half *p, int64_t c, const float (&v)[8]) {
  unsigned w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const __half2 h = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    w[k] = *reinterpret_cast<const unsigned *>(&h);
  }
  reinterpret_cast<uint4 *>(p)[c] = make_uint4(w[0], w[1], w[2], w[3]);
}

// Sum over the G threads of this thread's group. Every thread of the block
// must call it (it may contain block barriers).
template <int G>
__device__ inline float ln_group_sum(float v, float *s_wave) {
#pragma unroll
  for (int off = kRowWave / 2; off > 0; off >>= 1)
    v += __shfl_xor(v, off, kRowWave);
  if constexpr (G == kRowWave) {
    return v;
  } else {
    constexpr int kWavesPerGroup = G / kRowWave;
    if ((threadIdx.x % kRowWave) == 0)
      s_wave[threadIdx.x / kRowWave] = v;
    __syncthreads();
    const int first = (threadIdx.x / G) * kWavesPerGroup;
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < kWavesPerGroup; ++w)
      total += s_wave[first + w];
    __syncthreads();
    return total;
  }
}

template <int G> struct RowBlock {
  static constexpr int BS = G < kLayerNormBlockSize ? kLayerNormBlockSize : G;
  static constexpr int Rows = BS / G;
};

// C = chunks per thread (compile-time upper bound; the tail is predicated).
template <typename T, int G, int C>
__global__ __launch_bounds__(RowBlock<G>::BS) void hip_layer_norm_row_kernel(
    const T *__restrict__ input, const T *__restrict__ skip,
    const T *__restrict__ skip_bias, const T *__restrict__ scale,
    const T *__restrict__ bias, T *__restrict__ output, T *__restrict__ sum_out,
    void *__restrict__ mean_out, void *__restrict__ inv_std_out,
    bool stats_half, bool round_sum, int64_t outer, int64_t skip_rows,
    int64_t norm_size, float epsilon) {
  constexpr int N = Chunk<T>::N;
  constexpr int BS = RowBlock<G>::BS;
  __shared__ float s_wave[BS / kRowWave];

  const int lane = threadIdx.x % G;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * RowBlock<G>::Rows + threadIdx.x / G;
  // Rows past the end still take part in the group reductions' barriers.
  const bool active = row < outer;
  const int64_t nchunks = norm_size / N;
  const int64_t base = (active ? row : 0) * norm_size;
  const T *x = input + base;
  const T *s =
      skip ? skip + ((active ? row : 0) % skip_rows) * norm_size : nullptr;
  const float n = static_cast<float>(norm_size);
  const float inv_n = 1.0f / n;
  const float t0 = ln_elem(x, s, skip_bias, round_sum, 0);

  float v[C][N];
  float sum = 0.0f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int64_t idx = lane + static_cast<int64_t>(c) * G;
    if (active && idx < nchunks) {
      ln_ld16(x, idx, v[c]);
      if (s) {
        float o[N];
        ln_ld16(s, idx, o);
#pragma unroll
        for (int k = 0; k < N; ++k)
          v[c][k] += o[k];
        if (skip_bias) {
          ln_ld16(skip_bias, idx, o);
#pragma unroll
          for (int k = 0; k < N; ++k)
            v[c][k] += o[k];
        }
        if (round_sum) {
#pragma unroll
          for (int k = 0; k < N; ++k)
            v[c][k] = ln_round(x, v[c][k]);
        }
        if (sum_out)
          ln_st16(sum_out + base, idx, v[c]);
      }
#pragma unroll
      for (int k = 0; k < N; ++k)
        sum += (v[c][k] - t0) * inv_n;
    }
  }
  const float m0 = t0 + ln_group_sum<G>(sum, s_wave);

  float dsum = 0.0f;
  float dsq = 0.0f;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int64_t idx = lane + static_cast<int64_t>(c) * G;
    if (active && idx < nchunks) {
#pragma unroll
      for (int k = 0; k < N; ++k) {
        v[c][k] -= m0;
        dsum += v[c][k] * inv_n;
        dsq += v[c][k] * v[c][k];
      }
    }
  }
  const float corr = ln_group_sum<G>(dsum, s_wave);
  const float mean_dsq = ln_group_sum<G>(dsq, s_wave) / n;
  if (!active)
    return;
  const float inv_std = rsqrtf(ln_var(mean_dsq, corr) + epsilon);
  if (lane == 0) {
    if (mean_out)
      ln_store_stat(mean_out, row, stats_half, m0 + corr);
    if (inv_std_out)
      ln_store_stat(inv_std_out, row, stats_half, inv_std);
  }

  T *y = output + base;
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int64_t idx = lane + static_cast<int64_t>(c) * G;
    if (idx < nchunks) {
      float g[N], b[N];
      ln_ld16(scale, idx, g);
      if (bias) {
        ln_ld16(bias, idx, b);
      } else {
#pragma unroll
        for (int k = 0; k < N; ++k)
          b[k] = 0.0f;
      }
#pragma unroll
      for (int k = 0; k < N; ++k)
        v[c][k] = (v[c][k] - corr) * inv_std * g[k] + b[k];
      ln_st16(y, idx, v[c]);
    }
  }
}

struct LayerNormArgs {
  const void *input;
  const void *skip;
  const void *skip_bias;
  const void *scale;
  const void *bias;
  void *output;
  void *sum_out;
  void *mean_out;
  void *inv_std_out;
  bool stats_half;
  bool round_sum;
  int64_t outer;
  int64_t skip_rows;
  int64_t norm_size;
  float epsilon;
};

// Threads per row for the row kernel, or 0 when the row kernel does not
// apply. The smallest group that fits the row in kRowMaxChunks chunks per
// thread: one wave up to 256 chunks, wider groups for wider rows.
template <typename T> int row_group_size(const LayerNormArgs &a) {
  constexpr int N = Chunk<T>::N;
  if (a.norm_size % N != 0)
    return 0;
  auto aligned = [](const void *p) {
    return p == nullptr || (reinterpret_cast<uintptr_t>(p) % 16) == 0;
  };
  if (!aligned(a.input) || !aligned(a.skip) || !aligned(a.skip_bias) ||
      !aligned(a.scale) || !aligned(a.bias) || !aligned(a.output) ||
      !aligned(a.sum_out))
    return 0;
  const int64_t nchunks = a.norm_size / N;
  for (int g = kRowWave; g <= kRowMaxGroup; g *= 2)
    if (nchunks <= static_cast<int64_t>(g) * kRowMaxChunks)
      return g;
  return 0;
}

template <typename T, int G, int C>
void launch_row(hipStream_t stream, const LayerNormArgs &a) {
  const int64_t blocks =
      (a.outer + RowBlock<G>::Rows - 1) / RowBlock<G>::Rows;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(hip_layer_norm_row_kernel<T, G, C>),
      dim3(static_cast<unsigned int>(blocks)), dim3(RowBlock<G>::BS), 0,
      stream, static_cast<const T *>(a.input), static_cast<const T *>(a.skip),
      static_cast<const T *>(a.skip_bias), static_cast<const T *>(a.scale),
      static_cast<const T *>(a.bias), static_cast<T *>(a.output),
      static_cast<T *>(a.sum_out), a.mean_out, a.inv_std_out, a.stats_half,
      a.round_sum, a.outer, a.skip_rows, a.norm_size, a.epsilon);
}

// Chunks per thread are rounded up to {1, 2, 4, 6, 8} to bound the number of
// instantiations; a group wider than one wave always needs more than 4.
template <typename T, int G>
void launch_row_group(hipStream_t stream, const LayerNormArgs &a) {
  const int64_t nchunks = a.norm_size / Chunk<T>::N;
  const int64_t c = (nchunks + G - 1) / G;
  if constexpr (G == kRowWave) {
    if (c <= 1)
      return launch_row<T, G, 1>(stream, a);
    if (c <= 2)
      return launch_row<T, G, 2>(stream, a);
    if (c <= 4)
      return launch_row<T, G, 4>(stream, a);
  }
  if (c <= 6)
    return launch_row<T, G, 6>(stream, a);
  return launch_row<T, G, 8>(stream, a);
}

template <typename T>
int launch_layer_norm(hipStream_t stream, const LayerNormArgs &a,
                      const char *name) {
  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to
  // THIS launch only.
  (void)hipGetLastError();

  const int group = row_group_size<T>(a);
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] %s: elem_bytes=%d, outer=%lld, norm_size=%lld, "
      "epsilon=%f, skip=%s, sum=%s, stats=%s, kernel=%s(%d)\n",
      name, (int)sizeof(T), (long long)a.outer, (long long)a.norm_size,
      a.epsilon, a.skip ? "yes" : "null", a.sum_out ? "yes" : "null",
      (a.mean_out || a.inv_std_out) ? (a.stats_half ? "f16" : "f32") : "null",
      group ? "row" : "block", group);

  switch (group) {
  case 32:
    launch_row_group<T, 32>(stream, a);
    break;
  case 64:
    launch_row_group<T, 64>(stream, a);
    break;
  case 128:
    launch_row_group<T, 128>(stream, a);
    break;
  case 256:
    launch_row_group<T, 256>(stream, a);
    break;
  case 512:
    launch_row_group<T, 512>(stream, a);
    break;
  default:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_layer_norm_block_kernel<T>),
        dim3(static_cast<unsigned int>(a.outer)), dim3(kLayerNormBlockSize), 0,
        stream, static_cast<const T *>(a.input),
        static_cast<const T *>(a.skip), static_cast<const T *>(a.skip_bias),
        static_cast<const T *>(a.scale), static_cast<const T *>(a.bias),
        static_cast<T *>(a.output), static_cast<T *>(a.sum_out), a.mean_out,
        a.inv_std_out, a.stats_half, a.round_sum, a.outer, a.skip_rows,
        a.norm_size, a.epsilon);
    break;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] %s launch failed: %s\n", name,
            hipGetErrorString(err));
  return static_cast<int>(err);
}

int dispatch_layer_norm(void *stream, const LayerNormArgs &a, int hip_dtype,
                        const char *name) {
  if (a.outer <= 0 || a.norm_size <= 0)
    return 0;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  if (hip_dtype == HIP_DTYPE_FLOAT16)
    return launch_layer_norm<__half>(hip_stream, a, name);
  if (hip_dtype == HIP_DTYPE_FLOAT32)
    return launch_layer_norm<float>(hip_stream, a, name);
  fprintf(stderr, "[custom_kernels] %s: unsupported dtype=%d\n", name,
          hip_dtype);
  return -1;
}

} // namespace

extern "C" int hip_layer_norm(void *stream, const void *input,
                              const void *scale, const void *bias,
                              void *output, void *mean_out, void *inv_std_out,
                              int64_t outer, int64_t norm_size, float epsilon,
                              int hip_dtype, int mean_dtype) {
  // fp32 input always writes fp32 stats.
  const bool stats_half =
      hip_dtype == HIP_DTYPE_FLOAT16 && mean_dtype == HIP_DTYPE_FLOAT16;
  LayerNormArgs a{input,
                  /*skip=*/nullptr,
                  /*skip_bias=*/nullptr,
                  scale,
                  bias,
                  output,
                  /*sum_out=*/nullptr,
                  mean_out,
                  inv_std_out,
                  stats_half,
                  /*round_sum=*/false,
                  outer,
                  /*skip_rows=*/1,
                  norm_size,
                  epsilon};
  return dispatch_layer_norm(stream, a, hip_dtype, "hip_layer_norm");
}

extern "C" int hip_skip_layer_norm(void *stream, const void *input,
                                   const void *skip, const void *gamma,
                                   const void *beta, const void *bias,
                                   void *output, void *input_skip_bias_sum,
                                   int64_t outer, int64_t skip_rows,
                                   int64_t norm_size, float epsilon,
                                   int round_sum, int hip_dtype) {
  if (!skip || skip_rows <= 0) {
    fprintf(stderr, "[custom_kernels] hip_skip_layer_norm: null skip or "
                    "skip_rows=%lld\n",
            (long long)skip_rows);
    return -1;
  }
  LayerNormArgs a{input,
                  skip,
                  bias,
                  gamma,
                  beta,
                  output,
                  input_skip_bias_sum,
                  /*mean_out=*/nullptr,
                  /*inv_std_out=*/nullptr,
                  /*stats_half=*/false,
                  round_sum != 0,
                  outer,
                  skip_rows,
                  norm_size,
                  epsilon};
  return dispatch_layer_norm(stream, a, hip_dtype, "hip_skip_layer_norm");
}
