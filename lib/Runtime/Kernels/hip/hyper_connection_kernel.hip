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
};

constexpr int kHcBlock = 256;
constexpr int kHcSegments = kHcBlock / 32;

// Every thread of the block calls this the same number of times. Each run of
// `q.block` consecutive threads must hold one quantization group: the caller
// maps thread t to column col0 + t with col0 a multiple of q.block, so groups
// line up with 32-lane segments on wave32 and wave64 alike.
__device__ void hc_quant_store(const HcQuantOut &q, int64_t row, int64_t col,
                               float a, bool valid, float *s_seg) {
  float m = valid ? fabsf(a) : 0.f;
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor(m, off, 32));
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
  if (!valid)
    return;
  const float scale = (m > 0.0f) ? (m * (1.0f / 127.0f)) : 1.0f;
  const float inv_scale = (m > 0.0f) ? (127.0f / m) : 0.0f;
  if (col % q.block == 0)
    q.scale[row * (q.k / q.block) + col / q.block] = scale;
  int v = __float2int_rn(a * inv_scale);
  v = max(-127, min(127, v));
  const int64_t c = col >> 3;
  const int p = static_cast<int>(col & 7);
  const int64_t byte = (p & 1) ? (c * 8 + 4 + (p >> 1)) : (c * 8 + (p >> 1));
  q.qb[row * q.k + byte] = static_cast<int8_t>(v);
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
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_hc_post_mix_norm_kernel<T>),
                     dim3(static_cast<unsigned>(blocks)), dim3(256), 0, stream,
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

// --- hyper_connection_mixer ------------------------------------------------

// round(swish(round(a * act_scale), swish_alpha)), each step rounded to fp16
// exactly as the separate hip.mul and hip.swish kernels round it.
__device__ float hc_mixer_act(float a, float act_scale, float swish_alpha) {
  const float x = hc_round<__half>(a * act_scale);
  const float alpha = hc_round<__half>(swish_alpha);
  const float scaled = hc_round<__half>(alpha * x);
  const float sig = hc_round<__half>(1.f / (1.f + expf(-scaled)));
  return hc_round<__half>(x * sig);
}

// Separate-kernel path, first step: act = the scaled Swish of proj's leading
// act_width columns and gate = its trailing columns, both contiguous.
__global__ void hip_hc_mixer_act_kernel(const __half *__restrict__ proj,
                                        __half *__restrict__ act,
                                        __half *__restrict__ gate,
                                        int64_t rows, int64_t act_width,
                                        int64_t branches, float act_scale,
                                        float swish_alpha) {
  const int64_t width = act_width + branches;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * width)
    return;
  const int64_t row = i / width;
  const int64_t col = i - row * width;
  const float a = __half2float(proj[i]);
  if (col < act_width)
    act[row * act_width + col] =
        __float2half(hc_mixer_act(a, act_scale, swish_alpha));
  else
    gate[row * branches + col - act_width] = proj[i];
}

// Fused few-row path. A workgroup owns kMixTile hidden columns h0.. of every
// row: it builds the activation in LDS, projects the branches * kMixTile gate
// columns {c * hidden + h} it needs (kMixTeam lanes per column over K, in a
// fixed order), then reduces the streams through the sigmoid gates exactly as
// hip_hc_pre_mix_kernel does. The int8 copy needs one 32-wide quantization
// group per row segment, which is why kMixTile is 32.
constexpr int kMixTile = 32;
constexpr int kMixTeam = 8;
constexpr int kMixMaxRows = 8;
constexpr int kMixThreads = kMixMaxRows * kMixTile;

__global__ __launch_bounds__(kMixThreads) void hip_hc_mixer_decode_kernel(
    const __half *__restrict__ streams, const __half *__restrict__ proj,
    const uint32_t *__restrict__ weight, const __half *__restrict__ scales,
    __half *__restrict__ output, __half *__restrict__ gate, int rows,
    int branches, int64_t hidden, int act_width, int block_size,
    float act_scale, float swish_alpha, float reduction_scale, HcQuantOut q) {
  extern __shared__ float s_mem[];
  float *s_act = s_mem;                           // [rows][act_width]
  float *s_mix = s_mem + rows * act_width;        // [rows][branches][kMixTile]
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t h0 = static_cast<int64_t>(blockIdx.x) * kMixTile;
  const int proj_width = act_width + branches;

  if (blockIdx.x == 0)
    for (int i = tid; i < rows * branches; i += blockDim.x)
      gate[i] = proj[(i / branches) * proj_width + act_width + i % branches];
  for (int i = tid; i < rows * act_width; i += blockDim.x) {
    const int r = i / act_width;
    s_act[i] = hc_mixer_act(__half2float(proj[r * proj_width + i % act_width]),
                            act_scale, swish_alpha);
  }
  __syncthreads();

  const int words = act_width / 8;
  const int teams = blockDim.x / kMixTeam;
  const int team = tid / kMixTeam;
  const int lane = tid % kMixTeam;
  const int k_blocks = act_width / block_size;
  for (int j = team; j < branches * kMixTile; j += teams) {
    const int c = j / kMixTile;
    const int hh = j - c * kMixTile;
    const int64_t n = c * hidden + h0 + hh;
    const uint32_t *w_row = weight + n * words;
    const __half *s_row = scales + n * k_blocks;
    float acc[kMixMaxRows] = {};
    for (int w = lane; w < words; w += kMixTeam) {
      // 8 nibbles k = 8w..8w+7, low nibble first; block_size >= 8 keeps
      // them in one quantization block.
      const uint32_t bits = w_row[w];
      const float scale = __half2float(s_row[(w * 8) / block_size]);
      float part[kMixMaxRows] = {};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float wv = static_cast<float>(
            static_cast<int>((bits >> (4 * i)) & 0xFu) - 8);
        const float *a = s_act + w * 8 + i;
#pragma unroll
        for (int r = 0; r < kMixMaxRows; ++r)
          if (r < rows)
            part[r] += a[r * act_width] * wv;
      }
#pragma unroll
      for (int r = 0; r < kMixMaxRows; ++r)
        acc[r] += part[r] * scale;
    }
#pragma unroll
    for (int r = 0; r < kMixMaxRows; ++r)
      for (int off = kMixTeam / 2; off > 0; off >>= 1)
        acc[r] += __shfl_xor(acc[r], off, kMixTeam);
    if (lane == 0)
      for (int r = 0; r < rows; ++r)
        s_mix[(r * branches + c) * kMixTile + hh] = hc_round<__half>(acc[r]);
  }
  __syncthreads();

  const bool valid = tid < rows * kMixTile;
  const int r = valid ? tid / kMixTile : 0;
  const int64_t h = h0 + tid % kMixTile;
  float sum = 0.f;
  for (int c = 0; valid && c < branches; ++c)
    sum += hc_load(streams + (static_cast<int64_t>(r) * branches + c) * hidden +
                   h) *
           hc_sigmoid<__half>(s_mix[(r * branches + c) * kMixTile +
                                    tid % kMixTile]);
  if (valid)
    hc_store(output + r * hidden + h, sum * reduction_scale);
  if (q.qb)
    hc_quant_store(q, r, h, hc_round<__half>(sum * reduction_scale), valid,
                   nullptr);
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
                   hidden, static_cast<int>(q_block)};
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
                   branches * hidden, static_cast<int>(q_block)};
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

extern "C" int hip_hyper_connection_mixer_act(void *stream, const void *proj,
                                              void *act, void *gate,
                                              int64_t rows, int64_t act_width,
                                              int64_t branches,
                                              float act_scale,
                                              float swish_alpha) {
  if (!proj || !act || !gate || rows < 0 || act_width <= 0 || branches <= 0)
    return -1;
  dim3 grid;
  int threads = launch_count(rows * (act_width + branches), grid);
  if (threads < 0)
    return -1;
  if (threads == 0)
    return 0;
  (void)hipGetLastError();
  hipLaunchKernelGGL(hip_hc_mixer_act_kernel, grid, dim3(threads), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(proj),
                     static_cast<__half *>(act), static_cast<__half *>(gate),
                     rows, act_width, branches, act_scale, swish_alpha);
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_hyper_connection_mixer_decode_ok(int64_t rows,
                                                   int64_t branches,
                                                   int64_t hidden,
                                                   int64_t act_width,
                                                   int64_t block_size) {
  const int64_t lds =
      rows * (act_width + branches * kMixTile) * static_cast<int64_t>(sizeof(float));
  return rows >= 1 && rows <= kMixMaxRows && branches >= 1 &&
         hidden % kMixTile == 0 && act_width % 8 == 0 && block_size >= 8 &&
         (block_size & (block_size - 1)) == 0 && act_width % block_size == 0 &&
         lds <= 48 * 1024;
}

extern "C" int hip_hyper_connection_mixer_decode(
    void *stream, const void *streams, const void *proj, const void *weight,
    const void *scales, void *output, void *gate, int64_t rows,
    int64_t branches, int64_t hidden, int64_t act_width, int64_t block_size,
    float act_scale, float swish_alpha, float reduction_scale, void *q_act,
    void *q_scale, int64_t q_block) {
  if (!streams || !proj || !weight || !scales || !output || !gate ||
      !hip_hyper_connection_mixer_decode_ok(rows, branches, hidden, act_width,
                                            block_size))
    return hipErrorInvalidValue;
  HcQuantOut q;
  if (q_act) {
    if (!q_scale || q_block != kMixTile)
      return hipErrorInvalidValue;
    q = HcQuantOut{static_cast<int8_t *>(q_act), static_cast<float *>(q_scale),
                   hidden, static_cast<int>(q_block)};
  }
  const size_t lds = static_cast<size_t>(rows) *
                     static_cast<size_t>(act_width + branches * kMixTile) *
                     sizeof(float);
  (void)hipGetLastError();
  hipLaunchKernelGGL(hip_hc_mixer_decode_kernel,
                     dim3(static_cast<unsigned>(hidden / kMixTile)),
                     dim3(kMixThreads), lds, static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(streams),
                     static_cast<const __half *>(proj),
                     static_cast<const uint32_t *>(weight),
                     static_cast<const __half *>(scales),
                     static_cast<__half *>(output), static_cast<__half *>(gate),
                     static_cast<int>(rows), static_cast<int>(branches), hidden,
                     static_cast<int>(act_width), static_cast<int>(block_size),
                     act_scale, swish_alpha, reduction_scale, q);
  return static_cast<int>(hipGetLastError());
}
