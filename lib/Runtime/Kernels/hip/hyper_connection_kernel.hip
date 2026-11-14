/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft HyperConnectionPreMix / HyperConnectionPostMix.
 *
 * Streams are (row, branch, hidden). A flattened (..., C * H) tensor is
 * that layout with no extra transpose.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

__device__ inline float hc_load(const float *p) { return *p; }
__device__ inline float hc_load(const __half *p) { return __half2float(*p); }
__device__ inline void hc_store(float *p, float v) { *p = v; }
__device__ inline void hc_store(__half *p, float v) { *p = __float2half(v); }

// Match a separate fp16 elementwise node: compute in FP32, then round to the
// element type before the value is consumed again. f32 is unchanged.
template <typename T>
__device__ float hc_round(float v) {
  if constexpr (sizeof(T) == sizeof(__half))
    return __half2float(__float2half(v));
  return v;
}

template <typename T>
__device__ float hc_sigmoid(float x) {
  return hc_round<T>(1.f / (1.f + expf(-x)));
}

// Optional int8 copy of an output row in the layout the integer-dot
// MatMulNBits GEMV reads (hip_matmul_nbits_dp4a_prequantized), so the
// consumer skips its own quantize launch. Bit-exact with
// matmul_nbits_quant_act_i8_kernel: per-group scale max|a| / 127, round to
// nearest, clamp to +-127, nibble-deinterleaved bytes. `a` must be the value
// as stored in the fp16 output.
struct HcQuantOut {
  int8_t *qb = nullptr;  // [rows, K] bytes
  float *scale = nullptr; // [rows, K / block]
  int64_t k = 0;          // row length of the quantized activation
  int block = 0;          // power of two in [32, kHcBlock]
  int shift = 0;          // log2(block)
};

constexpr int kHcBlock = 256;
constexpr int kHcSegments = kHcBlock / 32;

// Writes column `col` of the int8 row given its group's max |a|.
__device__ inline void hc_quant_write(const HcQuantOut &q, int64_t row,
                                      int64_t col, float a, float m) {
  const float scale = (m > 0.0f) ? (m * (1.0f / 127.0f)) : 1.0f;
  const float inv_scale = (m > 0.0f) ? (127.0f / m) : 0.0f;
  if ((col & (q.block - 1)) == 0)
    q.scale[row * (q.k >> q.shift) + (col >> q.shift)] = scale;
  int v = __float2int_rn(a * inv_scale);
  v = max(-127, min(127, v));
  const int64_t c = col >> 3;
  const int p = static_cast<int>(col & 7);
  const int64_t byte = (p & 1) ? (c * 8 + 4 + (p >> 1)) : (c * 8 + (p >> 1));
  q.qb[row * q.k + byte] = static_cast<int8_t>(v);
}

__device__ inline float hc_segment_absmax(float a, bool valid) {
  float m = valid ? fabsf(a) : 0.f;
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
  return m;
}

// Every thread of the block calls this the same number of times. Each run of
// `q.block` consecutive threads must hold one quantization group: the caller
// maps thread t to column col0 + t with col0 a multiple of q.block, so groups
// line up with 32-lane segments on wave32 and wave64 alike.
__device__ void hc_quant_store(const HcQuantOut &q, int64_t row, int64_t col,
                               float a, bool valid, float *s_seg) {
  float m = hc_segment_absmax(a, valid);
  if (q.block > 32) {
    const int seg = static_cast<int>(threadIdx.x) / 32;
    const int segs = q.block / 32;
    if ((threadIdx.x & 31) == 0)
      s_seg[seg] = m;
    __syncthreads();
    const int first = seg / segs * segs;
    m = 0.f;
    for (int i = 0; i < segs; ++i)
      m = fmaxf(m, s_seg[first + i]);
    __syncthreads();
  }
  if (valid)
    hc_quant_write(q, row, col, a, m);
}

template <typename T>
__global__ void hip_hc_pre_mix_kernel(const T *__restrict__ streams,
                                      const T *__restrict__ gate,
                                      T *__restrict__ output, int64_t rows,
                                      int64_t branches, int64_t hidden,
                                      int feature_gate, int gate_sigmoid,
                                      float reduction_scale, HcQuantOut q) {
  __shared__ float s_seg[kHcSegments];
  int64_t n = rows * hidden;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool valid = i < n;
  if (!valid && !q.qb)
    return;
  int64_t row = valid ? i / hidden : 0;
  int64_t h = i - row * hidden;
  float sum = 0.f;
  for (int64_t c = 0; valid && c < branches; ++c) {
    int64_t x_index = (row * branches + c) * hidden + h;
    int64_t g_index = feature_gate ? x_index : row * branches + c;
    float g = hc_load(gate + g_index);
    if (gate_sigmoid)
      g = hc_sigmoid<T>(g);
    sum += hc_load(streams + x_index) * g;
  }
  if (valid)
    hc_store(output + i, sum * reduction_scale);
  if (q.qb)
    hc_quant_store(q, row, h, hc_round<T>(sum * reduction_scale), valid,
                   s_seg);
}

template <typename T> struct alignas(16) HcVec {
  static constexpr int kN = 16 / sizeof(T);
  T v[kN];
};

// hip_hc_pre_mix_kernel over one 16-byte chunk of hidden per thread. Every
// output element is computed exactly as the scalar kernel computes it; only
// the loads are wider and the row/column split is done once per chunk.
template <typename T>
__global__ void hip_hc_pre_mix_vec_kernel(const T *__restrict__ streams,
                                          const T *__restrict__ gate,
                                          T *__restrict__ output, int64_t rows,
                                          int64_t branches, int64_t hidden,
                                          int feature_gate, int gate_sigmoid,
                                          float reduction_scale) {
  constexpr int kN = HcVec<T>::kN;
  int64_t chunks = hidden / kN;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * chunks)
    return;
  int64_t row = i / chunks;
  int64_t h0 = (i - row * chunks) * kN;
  float sum[kN];
#pragma unroll
  for (int k = 0; k < kN; ++k)
    sum[k] = 0.f;
  for (int64_t c = 0; c < branches; ++c) {
    int64_t x_index = (row * branches + c) * hidden + h0;
    HcVec<T> x = *reinterpret_cast<const HcVec<T> *>(streams + x_index);
    if (feature_gate) {
      HcVec<T> gv = *reinterpret_cast<const HcVec<T> *>(gate + x_index);
#pragma unroll
      for (int k = 0; k < kN; ++k) {
        float g = hc_load(&gv.v[k]);
        if (gate_sigmoid)
          g = hc_sigmoid<T>(g);
        sum[k] += hc_load(&x.v[k]) * g;
      }
    } else {
      float g = hc_load(gate + row * branches + c);
      if (gate_sigmoid)
        g = hc_sigmoid<T>(g);
#pragma unroll
      for (int k = 0; k < kN; ++k)
        sum[k] += hc_load(&x.v[k]) * g;
    }
  }
  HcVec<T> out;
#pragma unroll
  for (int k = 0; k < kN; ++k)
    hc_store(&out.v[k], sum[k] * reduction_scale);
  *reinterpret_cast<HcVec<T> *>(output + row * hidden + h0) = out;
}

inline bool hc_aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <typename T>
__global__ void hip_hc_post_mix_kernel(const T *__restrict__ streams,
                                       const T *__restrict__ block_output,
                                       const T *__restrict__ post_mix,
                                       const T *__restrict__ stream_mix,
                                       T *__restrict__ output, int64_t rows,
                                       int64_t branches, int64_t hidden,
                                       int feature_gate) {
  int64_t n = rows * branches * hidden;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t h = i % hidden;
  int64_t tmp = i / hidden;
  int64_t c = tmp % branches;
  int64_t row = tmp / branches;
  float value = 0.f;
  if (stream_mix == nullptr) {
    value = hc_load(streams + i);
  } else {
    for (int64_t source = 0; source < branches; ++source) {
      int64_t mix_index = (row * branches + source) * branches + c;
      int64_t x_index = (row * branches + source) * hidden + h;
      value += hc_load(stream_mix + mix_index) * hc_load(streams + x_index);
    }
  }
  int64_t g_index =
      feature_gate ? (row * branches + c) * hidden + h : row * branches + c;
  value += hc_load(post_mix + g_index) * hc_load(block_output + row * hidden + h);
  hc_store(output + i, value);
}

int launch_count(int64_t n, dim3 &grid) {
  constexpr int kBlock = 256;
  if (n <= 0)
    return 0;
  int64_t blocks = (n + kBlock - 1) / kBlock;
  if (blocks > 0x7fffffff)
    return -1;
  grid = dim3(static_cast<unsigned>(blocks));
  return kBlock;
}

template <typename T>
int launch_pre(hipStream_t stream, const void *streams, const void *gate,
               void *output, int64_t rows, int64_t branches, int64_t hidden,
               int feature_gate, int gate_sigmoid, float reduction_scale,
               const HcQuantOut &q) {
  dim3 grid;
  // The chunked kernel launches 1/kN as many threads, which leaves a few-row
  // (decode) call on too few workgroups to hide its load latency; it pays off
  // once the scalar grid would be large. Only the scalar kernel writes the
  // int8 copy, which is a few-row (decode) output anyway.
  constexpr int kN = HcVec<T>::kN;
  constexpr int64_t kMinVecThreads = 4096;
  if (!q.qb && hidden % kN == 0 && rows * (hidden / kN) >= kMinVecThreads &&
      hc_aligned16(streams) && hc_aligned16(output) &&
      (!feature_gate || hc_aligned16(gate))) {
    int threads = launch_count(rows * (hidden / kN), grid);
    if (threads < 0)
      return -1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_pre_mix_vec_kernel<T>), grid,
                       dim3(threads), 0, stream,
                       static_cast<const T *>(streams),
                       static_cast<const T *>(gate), static_cast<T *>(output),
                       rows, branches, hidden, feature_gate, gate_sigmoid,
                       reduction_scale);
    return 0;
  }
  int threads = launch_count(rows * hidden, grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_pre_mix_kernel<T>), grid,
                     dim3(threads), 0, stream, static_cast<const T *>(streams),
                     static_cast<const T *>(gate), static_cast<T *>(output),
                     rows, branches, hidden, feature_gate, gate_sigmoid,
                     reduction_scale, q);
  return 0;
}

template <typename T>
int launch_post(hipStream_t stream, const void *streams,
                const void *block_output, const void *post_mix,
                const void *stream_mix, void *output, int64_t rows,
                int64_t branches, int64_t hidden, int feature_gate) {
  dim3 grid;
  int threads = launch_count(rows * branches * hidden, grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_kernel<T>), grid,
                     dim3(threads), 0, stream, static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(post_mix),
                     static_cast<const T *>(stream_mix),
                     static_cast<T *>(output), rows, branches, hidden,
                     feature_gate);
  return 0;
}

template <typename T>
__device__ float hc_gate(float raw, int gate_activation) {
  if (!gate_activation)
    return raw;
  return hc_round<T>(hc_sigmoid<T>(hc_round<T>(raw / 4.f)) * 2.f);
}

// One block per (row, branch).
template <typename T>
__global__ void hip_hc_post_mix_norm_kernel(
    const T *__restrict__ streams, const T *__restrict__ block_output,
    const T *__restrict__ gate, const T *__restrict__ stream_mix,
    const T *__restrict__ scale, T *__restrict__ residual, T *__restrict__ normed,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_activation, float epsilon, HcQuantOut q) {
  int64_t branch_row = static_cast<int64_t>(blockIdx.x);
  if (branch_row >= rows * branches)
    return;
  int64_t row = branch_row / branches;
  int64_t c = branch_row - row * branches;
  float branch_gate = 0.f;
  if (!feature_gate)
    branch_gate = hc_gate<T>(hc_load(gate + row * branches + c), gate_activation);

  const T *block_row = block_output + row * hidden;
  const T *scale_row = scale + c * hidden;
  T *residual_row = residual + branch_row * hidden;
  T *normed_row = normed + branch_row * hidden;

  __shared__ float s_reduce[256];
  __shared__ float s_inv_rms;
  float local_sumsq = 0.f;
  for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float base = 0.f;
    if (stream_mix == nullptr) {
      base = hc_load(streams + branch_row * hidden + h);
    } else {
      for (int64_t source = 0; source < branches; ++source) {
        int64_t mix_index = (row * branches + source) * branches + c;
        int64_t x_index = (row * branches + source) * hidden + h;
        base += hc_load(stream_mix + mix_index) * hc_load(streams + x_index);
      }
    }
    float g = branch_gate;
    if (feature_gate) {
      g = hc_gate<T>(hc_load(gate + branch_row * hidden + h), gate_activation);
    }
    float y = hc_round<T>(base + g * hc_load(block_row + h));
    hc_store(residual_row + h, y);
    local_sumsq += y * y;
  }
  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    s_inv_rms = rsqrtf(s_reduce[0] / static_cast<float>(hidden) + epsilon);
  __syncthreads();
  float inv_rms = s_inv_rms;
  if (!q.qb) {
    for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
      float y = hc_load(residual_row + h);
      hc_store(normed_row + h, y * inv_rms * hc_load(scale_row + h));
    }
    return;
  }
  // The int8 row is the flattened (branch, hidden) row the consumer reads;
  // the caller guarantees hidden % q.block == 0, so no group spans branches.
  __shared__ float s_seg[kHcSegments];
  for (int64_t h0 = 0; h0 < hidden; h0 += blockDim.x) {
    const int64_t h = h0 + threadIdx.x;
    const bool valid = h < hidden;
    float v = 0.f;
    if (valid) {
      float y = hc_load(residual_row + h);
      v = hc_round<T>(y * inv_rms * hc_load(scale_row + h));
      hc_store(normed_row + h, v);
    }
    hc_quant_store(q, row, c * hidden + h, v, valid, s_seg);
  }
}

// hip_hc_post_mix_norm_kernel for hidden <= 256 * J: the same thread-to-column
// map (h = threadIdx.x + 256 * j), accumulation order and rounding, so the
// outputs are identical, but each thread issues all of its loads before the
// first use and keeps y in registers instead of re-reading the residual row.
// A few-row call runs one block per (row, branch), so the serialized
// load-use chains of the strided loops are what it waits on.
constexpr int kHcNormThreads = 256;

template <typename T, int J, bool kMix>
__global__ void __launch_bounds__(kHcNormThreads) hip_hc_post_mix_norm_reg_kernel(
    const T *__restrict__ streams, const T *__restrict__ block_output,
    const T *__restrict__ gate, const T *__restrict__ stream_mix,
    const T *__restrict__ scale, T *__restrict__ residual, T *__restrict__ normed,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_activation, float epsilon, HcQuantOut q) {
  const int64_t branch_row = static_cast<int64_t>(blockIdx.x);
  const int64_t row = branch_row / branches;
  const int64_t c = branch_row - row * branches;
  const int tid = static_cast<int>(threadIdx.x);
  const int width = static_cast<int>(hidden);
  const T *block_row = block_output + row * hidden;
  const T *scale_row = scale + c * hidden;
  T *residual_row = residual + branch_row * hidden;
  T *normed_row = normed + branch_row * hidden;

  // Loads first, outside any data-dependent branch, so every load of a thread
  // is in flight before the first one is used.
  float base[J], g[J], b[J];
  const float gate_raw = feature_gate ? 0.f : hc_load(gate + row * branches + c);
  if constexpr (!kMix) {
    const T *stream_row = streams + branch_row * hidden;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const int h = tid + kHcNormThreads * j;
      base[j] = h < width ? hc_load(stream_row + h) : 0.f;
    }
  } else {
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const int h = tid + kHcNormThreads * j;
      base[j] = 0.f;
      if (h >= width)
        continue;
      for (int64_t source = 0; source < branches; ++source) {
        int64_t mix_index = (row * branches + source) * branches + c;
        int64_t x_index = (row * branches + source) * hidden + h;
        base[j] += hc_load(stream_mix + mix_index) * hc_load(streams + x_index);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int h = tid + kHcNormThreads * j;
    b[j] = h < width ? hc_load(block_row + h) : 0.f;
  }
  if (feature_gate) {
    const T *gate_row = gate + branch_row * hidden;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const int h = tid + kHcNormThreads * j;
      g[j] = h < width ? hc_gate<T>(hc_load(gate_row + h), gate_activation)
                       : 0.f;
    }
  } else {
    const float branch_gate = hc_gate<T>(gate_raw, gate_activation);
#pragma unroll
    for (int j = 0; j < J; ++j)
      g[j] = branch_gate;
  }

  float y[J];
  float local_sumsq = 0.f;
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int h = tid + kHcNormThreads * j;
    y[j] = 0.f;
    if (h < width) {
      y[j] = hc_round<T>(base[j] + g[j] * b[j]);
      hc_store(residual_row + h, y[j]);
      local_sumsq += y[j] * y[j];
    }
  }

  __shared__ float s_reduce[kHcNormThreads];
  __shared__ float s_inv_rms;
  s_reduce[tid] = local_sumsq;
  __syncthreads();
  for (int stride = kHcNormThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      s_reduce[tid] += s_reduce[tid + stride];
    __syncthreads();
  }
  if (tid == 0)
    s_inv_rms = rsqrtf(s_reduce[0] / static_cast<float>(hidden) + epsilon);
  __syncthreads();
  const float inv_rms = s_inv_rms;

  float sv[J];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int h = tid + kHcNormThreads * j;
    sv[j] = h < width ? hc_load(scale_row + h) : 0.f;
  }
  if (!q.qb) {
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const int h = tid + kHcNormThreads * j;
      if (h < width)
        hc_store(normed_row + h, y[j] * inv_rms * sv[j]);
    }
    return;
  }
  // A 32-wide group is one 32-lane segment, so it needs no barrier; keeping
  // the barrier path out of this loop lets the stores of all J columns
  // overlap.
  if (q.block == 32) {
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const int h = tid + kHcNormThreads * j;
      const bool valid = h < width;
      float v = 0.f;
      if (valid) {
        v = hc_round<T>(y[j] * inv_rms * sv[j]);
        hc_store(normed_row + h, v);
      }
      const float m = hc_segment_absmax(v, valid);
      if (valid)
        hc_quant_write(q, row, c * hidden + h, v, m);
    }
    return;
  }
  __shared__ float s_seg[kHcSegments];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int h = tid + kHcNormThreads * j;
    const bool valid = h < width;
    float v = 0.f;
    if (valid) {
      v = hc_round<T>(y[j] * inv_rms * sv[j]);
      hc_store(normed_row + h, v);
    }
    hc_quant_store(q, row, c * hidden + h, v, valid, s_seg);
  }
}

template <typename T, int J>
void launch_post_norm_reg(hipStream_t stream, dim3 grid, const void *streams,
                          const void *block_output, const void *gate,
                          const void *stream_mix, const void *scale,
                          void *residual, void *normed, int64_t rows,
                          int64_t branches, int64_t hidden, int feature_gate,
                          int gate_activation, float epsilon,
                          const HcQuantOut &q) {
  auto kernel = stream_mix ? hip_hc_post_mix_norm_reg_kernel<T, J, true>
                           : hip_hc_post_mix_norm_reg_kernel<T, J, false>;
  hipLaunchKernelGGL(kernel, grid, dim3(kHcNormThreads), 0, stream,
                     static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(gate),
                     static_cast<const T *>(stream_mix),
                     static_cast<const T *>(scale), static_cast<T *>(residual),
                     static_cast<T *>(normed), rows, branches, hidden,
                     feature_gate, gate_activation, epsilon, q);
}

template <typename T>
int launch_post_norm(hipStream_t stream, const void *streams,
                     const void *block_output, const void *gate,
                     const void *stream_mix, const void *scale, void *residual,
                     void *normed, int64_t rows, int64_t branches,
                     int64_t hidden, int feature_gate, int gate_activation,
                     float epsilon, const HcQuantOut &q) {
  int64_t blocks = rows * branches;
  if (blocks <= 0)
    return 0;
  if (blocks > 0x7fffffff)
    return -1;
  const dim3 grid(static_cast<unsigned>(blocks));
  const int64_t per_thread = (hidden + kHcNormThreads - 1) / kHcNormThreads;
  auto reg = per_thread <= 4    ? launch_post_norm_reg<T, 4>
             : per_thread <= 8  ? launch_post_norm_reg<T, 8>
             : per_thread <= 10 ? launch_post_norm_reg<T, 10>
             : per_thread <= 12 ? launch_post_norm_reg<T, 12>
             : per_thread <= 16 ? launch_post_norm_reg<T, 16>
                                : nullptr;
  if (reg) {
    reg(stream, grid, streams, block_output, gate, stream_mix, scale, residual,
        normed, rows, branches, hidden, feature_gate, gate_activation, epsilon,
        q);
    return 0;
  }
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_norm_kernel<T>),
                     grid, dim3(kHcNormThreads), 0, stream,
                     static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(gate),
                     static_cast<const T *>(stream_mix),
                     static_cast<const T *>(scale), static_cast<T *>(residual),
                     static_cast<T *>(normed), rows, branches, hidden,
                     feature_gate, gate_activation, epsilon, q);
  return 0;
}

// The epilogue's group mapping needs blocks of whole groups (see
// hc_quant_store) that never span a hidden row.
bool hc_quant_ok(const void *qb, const void *qscale, int64_t block,
                 int64_t hidden) {
  return qb && qscale && block >= 32 && block <= kHcBlock &&
         (block & (block - 1)) == 0 && hidden % block == 0;
}

} // namespace

extern "C" int hip_hyper_connection_pre_mix(
    void *stream, const void *streams, const void *pre_mix, void *output,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_sigmoid, float reduction_scale, void *q_act, void *q_scale,
    int64_t q_block, int hip_dtype) {
  if (!streams || !pre_mix || !output || branches <= 0 || hidden <= 0 ||
      rows < 0)
    return -1;
  HcQuantOut q;
  if (q_act) {
    if (!hc_quant_ok(q_act, q_scale, q_block, hidden))
      return hipErrorInvalidValue;
    q = HcQuantOut{static_cast<int8_t *>(q_act), static_cast<float *>(q_scale),
                   hidden, static_cast<int>(q_block),
                   __builtin_ctzll(static_cast<unsigned long long>(q_block))};
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_pre<__half>(hip_stream, streams, pre_mix, output, rows, branches,
                            hidden, feature_gate, gate_sigmoid, reduction_scale,
                            q);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_pre<float>(hip_stream, streams, pre_mix, output, rows, branches,
                           hidden, feature_gate, gate_sigmoid, reduction_scale,
                           q);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_pre_mix: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_pre_mix launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_hyper_connection_post_mix(
    void *stream, const void *streams, const void *block_output,
    const void *post_mix, const void *stream_mix, void *output, int64_t rows,
    int64_t branches, int64_t hidden, int feature_gate, int hip_dtype) {
  if (!streams || !block_output || !post_mix || !output || branches <= 0 ||
      hidden <= 0 || rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_post<__half>(hip_stream, streams, block_output, post_mix,
                             stream_mix, output, rows, branches, hidden,
                             feature_gate);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_post<float>(hip_stream, streams, block_output, post_mix,
                            stream_mix, output, rows, branches, hidden,
                            feature_gate);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix launch failed: "
            "%s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_hyper_connection_post_mix_norm(
    void *stream, const void *streams, const void *block_output,
    const void *gate, const void *stream_mix, const void *scale,
    void *residual, void *normed, int64_t rows, int64_t branches,
    int64_t hidden, int feature_gate, int gate_activation, float epsilon,
    void *q_act, void *q_scale, int64_t q_block, int hip_dtype) {
  if (!streams || !block_output || !gate || !scale || !residual || !normed ||
      branches <= 0 || hidden <= 0 || rows < 0)
    return -1;
  HcQuantOut q;
  if (q_act) {
    if (!hc_quant_ok(q_act, q_scale, q_block, hidden))
      return hipErrorInvalidValue;
    q = HcQuantOut{static_cast<int8_t *>(q_act), static_cast<float *>(q_scale),
                   branches * hidden, static_cast<int>(q_block),
                   __builtin_ctzll(static_cast<unsigned long long>(q_block))};
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_post_norm<__half>(hip_stream, streams, block_output, gate,
                                  stream_mix, scale, residual, normed, rows,
                                  branches, hidden, feature_gate,
                                  gate_activation, epsilon, q);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_post_norm<float>(hip_stream, streams, block_output, gate,
                                 stream_mix, scale, residual, normed, rows,
                                 branches, hidden, feature_gate, gate_activation,
                                 epsilon, q);
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix_norm: unsupported "
            "dtype=%d\n",
            hip_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix_norm launch "
            "failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
