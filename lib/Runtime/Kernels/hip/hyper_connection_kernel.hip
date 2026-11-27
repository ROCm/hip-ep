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

// Writes element k of a row quantized per 32-element group, in
// matmul_nbits_quant_act_i8_kernel's layout and with its rounding. `v` is the
// fp16-rounded value the kernel stored. The 32 lanes of an aligned lane group
// call this together for the 32 elements of one group, lane l holding element
// 32 * (k / 32) + l, so the group maximum is a 32-lane reduction.
__device__ inline void hc_quant32(float v, int64_t k, int8_t *__restrict__ a_quant,
                                  float *__restrict__ a_scale) {
  float gmax = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    gmax = fmaxf(gmax, __shfl_xor(gmax, off));
  const float inv_scale = (gmax > 0.0f) ? (127.0f / gmax) : 0.0f;
  const int q = max(-127, min(127, __float2int_rn(v * inv_scale)));
  const int64_t c = k >> 3;
  const int p = static_cast<int>(k & 7);
  a_quant[(p & 1) ? (c * 8 + 4 + (p >> 1)) : (c * 8 + (p >> 1))] =
      static_cast<int8_t>(q);
  if ((k & 31) == 0)
    a_scale[k >> 5] = (gmax > 0.0f) ? (gmax * (1.0f / 127.0f)) : 1.0f;
}

// QUANT (fp16, hidden % 32 == 0) also writes the output quantized with
// hc_quant32: thread i owns element i, so 32 aligned lanes own one group.
template <typename T, bool QUANT = false>
__global__ void hip_hc_pre_mix_kernel(const T *__restrict__ streams,
                                      const T *__restrict__ gate,
                                      T *__restrict__ output, int64_t rows,
                                      int64_t branches, int64_t hidden,
                                      int feature_gate, int gate_sigmoid,
                                      float reduction_scale,
                                      int8_t *__restrict__ a_quant = nullptr,
                                      float *__restrict__ a_scale = nullptr) {
  int64_t n = rows * hidden;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t row = i / hidden;
  int64_t h = i - row * hidden;
  float sum = 0.f;
  for (int64_t c = 0; c < branches; ++c) {
    int64_t x_index = (row * branches + c) * hidden + h;
    int64_t g_index = feature_gate ? x_index : row * branches + c;
    float g = hc_load(gate + g_index);
    if (gate_sigmoid)
      g = hc_sigmoid<T>(g);
    sum += hc_load(streams + x_index) * g;
  }
  hc_store(output + i, sum * reduction_scale);
  if constexpr (QUANT)
    hc_quant32(hc_round<T>(sum * reduction_scale), i, a_quant, a_scale);
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
               int feature_gate, int gate_sigmoid, float reduction_scale) {
  dim3 grid;
  // The chunked kernel launches 1/kN as many threads, which leaves a few-row
  // (decode) call on too few workgroups to hide its load latency; it pays off
  // once the scalar grid would be large.
  constexpr int kN = HcVec<T>::kN;
  constexpr int64_t kMinVecThreads = 4096;
  if (hidden % kN == 0 && rows * (hidden / kN) >= kMinVecThreads &&
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
                     reduction_scale);
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

// One block per (row, branch). QUANT (fp16, hidden % 32 == 0) also writes
// `normed` quantized with hc_quant32 over the flattened (branches * hidden)
// row: the store loop strides by the 256-thread block, so each aligned
// 32-lane group stores one 32-element group per iteration.
template <typename T, bool QUANT = false>
__global__ void hip_hc_post_mix_norm_kernel(
    const T *__restrict__ streams, const T *__restrict__ block_output,
    const T *__restrict__ gate, const T *__restrict__ stream_mix,
    const T *__restrict__ scale, T *__restrict__ residual, T *__restrict__ normed,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_activation, float epsilon, int8_t *__restrict__ a_quant = nullptr,
    float *__restrict__ a_scale = nullptr) {
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
  for (int64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float y = hc_load(residual_row + h);
    float v = y * inv_rms * hc_load(scale_row + h);
    hc_store(normed_row + h, v);
    if constexpr (QUANT)
      hc_quant32(hc_round<T>(v), branch_row * hidden + h, a_quant, a_scale);
  }
}

// hip_hc_post_mix_norm_kernel for a few (row, branch) pairs, as at decode: a
// 1024-thread block per pair, thread t owning h = t + 1024 j (j < IT) in
// registers through the reduction, the scale loaded with the inputs. With
// 256 threads, decode's 4 blocks run 10 dependent load rounds per thread;
// at 1 row, 4 branches, hidden 2560 with the inputs in L2 this is 2.2 vs
// 5.5 us (3.3 vs 7.7 with QUANT). The sum of squares is reduced in another
// order, so normed may differ from the 256-thread kernel by an fp16 ulp.
constexpr int kPostNormWideThreads = 1024;
constexpr int64_t kPostNormWideMaxBlocks = 64;

template <typename T, int IT, bool QUANT = false>
__global__ void __launch_bounds__(kPostNormWideThreads)
    hip_hc_post_mix_norm_wide_kernel(
        const T *__restrict__ streams, const T *__restrict__ block_output,
        const T *__restrict__ gate, const T *__restrict__ stream_mix,
        const T *__restrict__ scale, T *__restrict__ residual,
        T *__restrict__ normed, int64_t rows, int64_t branches, int64_t hidden,
        int feature_gate, int gate_activation, float epsilon,
        int8_t *__restrict__ a_quant = nullptr,
        float *__restrict__ a_scale = nullptr) {
  constexpr int NT = kPostNormWideThreads;
  const int64_t branch_row = static_cast<int64_t>(blockIdx.x);
  const int64_t row = branch_row / branches;
  const int64_t c = branch_row - row * branches;
  float branch_gate = 0.f;
  if (!feature_gate)
    branch_gate = hc_gate<T>(hc_load(gate + row * branches + c), gate_activation);

  const T *block_row = block_output + row * hidden;
  const T *scale_row = scale + c * hidden;
  T *residual_row = residual + branch_row * hidden;
  T *normed_row = normed + branch_row * hidden;

  float y[IT], sc[IT];
  float sumsq = 0.f;
#pragma unroll
  for (int j = 0; j < IT; ++j) {
    const int64_t h = threadIdx.x + int64_t(NT) * j;
    y[j] = 0.f;
    sc[j] = 0.f;
    if (h < hidden) {
      sc[j] = hc_load(scale_row + h);
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
      if (feature_gate)
        g = hc_gate<T>(hc_load(gate + branch_row * hidden + h), gate_activation);
      y[j] = hc_round<T>(base + g * hc_load(block_row + h));
      hc_store(residual_row + h, y[j]);
      sumsq += y[j] * y[j];
    }
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    sumsq += __shfl_xor(sumsq, off);
  __shared__ float s_part[NT / 32];
  __shared__ float s_inv_rms;
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves = NT / warpSize;
  if (lane == 0)
    s_part[wave] = sumsq;
  __syncthreads();
  if (wave == 0) {
    float v = lane < waves ? s_part[lane] : 0.f;
    for (int off = warpSize / 2; off > 0; off >>= 1)
      v += __shfl_xor(v, off);
    if (lane == 0)
      s_inv_rms = rsqrtf(v / static_cast<float>(hidden) + epsilon);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;
#pragma unroll
  for (int j = 0; j < IT; ++j) {
    const int64_t h = threadIdx.x + int64_t(NT) * j;
    if (h < hidden) {
      const float v = y[j] * inv_rms * sc[j];
      hc_store(normed_row + h, v);
      if constexpr (QUANT)
        hc_quant32(hc_round<T>(v), branch_row * hidden + h, a_quant, a_scale);
    }
  }
}

// Launches the wide kernel when the call has few enough (row, branch) pairs
// and hidden fits 4 elements per thread; false leaves it to the 256-thread
// kernel. QUANT needs hidden % 32 == 0 so 32 aligned lanes own one group.
template <typename T, bool QUANT>
bool launch_post_norm_wide(hipStream_t stream, const void *streams,
                           const void *block_output, const void *gate,
                           const void *stream_mix, const void *scale,
                           void *residual, void *normed, int64_t rows,
                           int64_t branches, int64_t hidden, int feature_gate,
                           int gate_activation, float epsilon,
                           void *a_quant = nullptr, void *a_scale = nullptr) {
  const int64_t blocks = rows * branches;
  if (blocks <= 0 || blocks > kPostNormWideMaxBlocks)
    return false;
  auto go = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(kPostNormWideThreads), 0, stream,
                       static_cast<const T *>(streams),
                       static_cast<const T *>(block_output),
                       static_cast<const T *>(gate),
                       static_cast<const T *>(stream_mix),
                       static_cast<const T *>(scale), static_cast<T *>(residual),
                       static_cast<T *>(normed), rows, branches, hidden,
                       feature_gate, gate_activation, epsilon,
                       static_cast<int8_t *>(a_quant),
                       static_cast<float *>(a_scale));
    return true;
  };
  constexpr int64_t NT = kPostNormWideThreads;
  if (hidden <= NT)
    return go(hip_hc_post_mix_norm_wide_kernel<T, 1, QUANT>);
  if (hidden <= 2 * NT)
    return go(hip_hc_post_mix_norm_wide_kernel<T, 2, QUANT>);
  if (hidden <= 3 * NT)
    return go(hip_hc_post_mix_norm_wide_kernel<T, 3, QUANT>);
  if (hidden <= 4 * NT)
    return go(hip_hc_post_mix_norm_wide_kernel<T, 4, QUANT>);
  return false;
}

template <typename T>
int launch_post_norm(hipStream_t stream, const void *streams,
                     const void *block_output, const void *gate,
                     const void *stream_mix, const void *scale, void *residual,
                     void *normed, int64_t rows, int64_t branches,
                     int64_t hidden, int feature_gate, int gate_activation,
                     float epsilon) {
  int64_t blocks = rows * branches;
  if (blocks <= 0)
    return 0;
  if (blocks > 0x7fffffff)
    return -1;
  if (launch_post_norm_wide<T, false>(stream, streams, block_output, gate,
                                      stream_mix, scale, residual, normed, rows,
                                      branches, hidden, feature_gate,
                                      gate_activation, epsilon))
    return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_norm_kernel<T>),
                     dim3(static_cast<unsigned>(blocks)), dim3(256), 0, stream,
                     static_cast<const T *>(streams),
                     static_cast<const T *>(block_output),
                     static_cast<const T *>(gate),
                     static_cast<const T *>(stream_mix),
                     static_cast<const T *>(scale), static_cast<T *>(residual),
                     static_cast<T *>(normed), rows, branches, hidden,
                     feature_gate, gate_activation, epsilon);
  return 0;
}

} // namespace

extern "C" int hip_hyper_connection_pre_mix(
    void *stream, const void *streams, const void *pre_mix, void *output,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_sigmoid, float reduction_scale, int hip_dtype) {
  if (!streams || !pre_mix || !output || branches <= 0 || hidden <= 0 ||
      rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_pre<__half>(hip_stream, streams, pre_mix, output, rows, branches,
                            hidden, feature_gate, gate_sigmoid, reduction_scale);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_pre<float>(hip_stream, streams, pre_mix, output, rows, branches,
                           hidden, feature_gate, gate_sigmoid, reduction_scale);
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
    int hip_dtype) {
  if (!streams || !block_output || !gate || !scale || !residual || !normed ||
      branches <= 0 || hidden <= 0 || rows < 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  int rc = 0;
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    rc = launch_post_norm<__half>(hip_stream, streams, block_output, gate,
                                  stream_mix, scale, residual, normed, rows,
                                  branches, hidden, feature_gate,
                                  gate_activation, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    rc = launch_post_norm<float>(hip_stream, streams, block_output, gate,
                                 stream_mix, scale, residual, normed, rows,
                                 branches, hidden, feature_gate, gate_activation,
                                 epsilon);
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

extern "C" int hip_hyper_connection_pre_mix_quant(
    void *stream, const void *streams, const void *pre_mix, void *output,
    int64_t rows, int64_t branches, int64_t hidden, int feature_gate,
    int gate_sigmoid, float reduction_scale, void *a_quant, void *a_scale) {
  if (!streams || !pre_mix || !output || !a_quant || !a_scale ||
      branches <= 0 || !hip_norm_quant_supported(rows, hidden, 32))
    return static_cast<int>(hipErrorInvalidValue);
  dim3 grid;
  int threads = launch_count(rows * hidden, grid);
  if (threads <= 0)
    return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_pre_mix_kernel<__half, true>), grid,
                     dim3(threads), 0, hip_stream,
                     static_cast<const __half *>(streams),
                     static_cast<const __half *>(pre_mix),
                     static_cast<__half *>(output), rows, branches, hidden,
                     feature_gate, gate_sigmoid, reduction_scale,
                     static_cast<int8_t *>(a_quant),
                     static_cast<float *>(a_scale));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_pre_mix_quant launch "
            "failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_hyper_connection_post_mix_norm_quant(
    void *stream, const void *streams, const void *block_output,
    const void *gate, const void *stream_mix, const void *scale,
    void *residual, void *normed, int64_t rows, int64_t branches,
    int64_t hidden, int feature_gate, int gate_activation, float epsilon,
    void *a_quant, void *a_scale) {
  if (!streams || !block_output || !gate || !scale || !residual || !normed ||
      !a_quant || !a_scale || branches <= 0 || hidden <= 0 ||
      !hip_norm_quant_supported(rows, branches * hidden, 32))
    return static_cast<int>(hipErrorInvalidValue);
  // A branch that does not start on a group boundary takes the plain kernel
  // and the standalone quantizer, which write the same bytes.
  if (hidden % 32 != 0) {
    int rc = hip_hyper_connection_post_mix_norm(
        stream, streams, block_output, gate, stream_mix, scale, residual,
        normed, rows, branches, hidden, feature_gate, gate_activation, epsilon,
        HIP_DTYPE_FLOAT16);
    if (rc != 0)
      return rc;
    return hip_matmul_nbits_quant_act_i8(stream, normed, rows,
                                         branches * hidden, 32, a_quant,
                                         a_scale);
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  if (!launch_post_norm_wide<__half, true>(
          hip_stream, streams, block_output, gate, stream_mix, scale, residual,
          normed, rows, branches, hidden, feature_gate, gate_activation,
          epsilon, a_quant, a_scale))
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_hc_post_mix_norm_kernel<__half, true>),
        dim3(static_cast<unsigned>(rows * branches)), dim3(256), 0, hip_stream,
        static_cast<const __half *>(streams),
        static_cast<const __half *>(block_output),
        static_cast<const __half *>(gate),
        static_cast<const __half *>(stream_mix),
        static_cast<const __half *>(scale), static_cast<__half *>(residual),
        static_cast<__half *>(normed), rows, branches, hidden, feature_gate,
        gate_activation, epsilon, static_cast<int8_t *>(a_quant),
        static_cast<float *>(a_scale));
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_hyper_connection_post_mix_norm_quant launch "
            "failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
