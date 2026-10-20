/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// MSE-only QR KV codec, based on the standalone TurboQuantMSE reference.
// v1: B1/Hq24/Hkv4/D256, FP16 activations, inline FP32 norm + 128 code bytes.
// The scan uses unnormalized (maximum, denominator, numerator) summaries.

#include "hip_custom_kernels.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int D = 256, G = 4, H = 24, WORDS = 33, STATE = 258;

__global__ void encode_append(const half *key, const half *value,
                             uint32_t *pk, uint32_t *pv, int count, int past,
                             int capacity, const float *pi,
                             const float *cbk, const float *cbv) {
  const int g = blockIdx.x, token = blockIdx.y, i = threadIdx.x;
  const bool is_value = blockIdx.z != 0;
  const half *src = is_value ? value : key;
  const float *cb = is_value ? cbv : cbk;
  uint32_t *dst = (is_value ? pv : pk) +
                 (int64_t(g) * capacity + past + token) * WORDS;
  __shared__ float x[D], reduction[D];
  __shared__ uint32_t index[D];
  float v = __half2float(src[(int64_t(token) * G + g) * D + i]);
  x[i] = v;
  reduction[i] = v * v;
  __syncthreads();
  for (int step = D / 2; step; step >>= 1) {
    if (i < step)
      reduction[i] += reduction[i + step];
    __syncthreads();
  }
  const float norm = sqrtf(reduction[0]);
  x[i] = norm > 0 ? x[i] / norm : 0;
  __syncthreads();
  float rotated = 0;
  for (int j = 0; j < D; ++j)
    rotated = fmaf(x[j], pi[i * D + j], rotated);
  float distance = fabsf(rotated - cb[0]);
  uint32_t best = 0;
  for (uint32_t c = 1; c < 16; ++c) {
    float candidate = fabsf(rotated - cb[c]);
    if (candidate < distance) {
      distance = candidate;
      best = c;
    }
  }
  index[i] = norm > 0 ? best : 0;
  __syncthreads();
  if (i == 0)
    dst[0] = __float_as_uint(norm);
  if (i < 32) {
    uint32_t packed = 0;
    for (int j = 0; j < 8; ++j)
      packed |= index[i * 8 + j] << (4 * j);
    dst[i + 1] = packed;
  }
}

__global__ void rotate_query(const half *q, float *qr, const float *pi) {
  const int h = blockIdx.x, i = threadIdx.x;
  __shared__ float x[D];
  x[i] = __half2float(q[h * D + i]);
  __syncthreads();
  float sum = 0;
  for (int j = 0; j < D; ++j)
    sum = fmaf(x[j], pi[i * D + j], sum);
  qr[h * D + i] = sum;
}

// Corrected scalar reference can be compiled for numerical/performance controls.
#ifdef TQ4_REFERENCE_DECODE
__global__ void scan(const float *qr, const uint32_t *pk, const uint32_t *pv,
                     int length, int capacity, int parts, float scale,
                     const float *cbk, const float *cbv, float *partial) {
  const int lane = threadIdx.x, h = blockIdx.x, part = blockIdx.y;
  __shared__ float books[32];
  if (lane < 16) { books[lane] = cbk[lane]; books[16 + lane] = cbv[lane]; }
  __syncthreads();
  const int chunk = (length + parts - 1) / parts;
  const int begin = part * chunk, end = min(length, begin + chunk);
  const int64_t head_row = int64_t(h / 6) * capacity;
  float q[8], acc[8] = {};
  for (int i = 0; i < 8; ++i)
    q[i] = qr[h * D + lane * 8 + i];
  float maximum = -INFINITY, denominator = 0;
  for (int t = begin; t < end; ++t) {
    const uint32_t *k = pk + (head_row + t) * WORDS;
    const uint32_t *v = pv + (head_row + t) * WORDS;
    uint32_t ki = k[lane + 1], vi = v[lane + 1];
    float score = 0;
    for (int i = 0; i < 8; ++i)
      score = fmaf(q[i], books[(ki >> (4 * i)) & 15], score);
    for (int delta = 16; delta; delta >>= 1)
      score += __shfl_down(score, delta, 32);
    score = __shfl(score, 0, 32) * __uint_as_float(k[0]) * scale;
    float next = fmaxf(maximum, score);
    float alpha = denominator == 0 ? 0 : expf(maximum - next);
    float beta = expf(score - next);
    denominator = alpha * denominator + beta;
    float weight = beta * __uint_as_float(v[0]);
    for (int i = 0; i < 8; ++i)
      acc[i] = alpha * acc[i] + weight * books[16 + ((vi >> (4 * i)) & 15)];
    maximum = next;
  }
  float *dst = partial + (h * parts + part) * STATE;
  if (lane == 0) {
    dst[0] = maximum;
    dst[1] = denominator;
  }
  for (int i = 0; i < 8; ++i)
    dst[2 + lane * 8 + i] = acc[i];
}

#else
// GQA/Ring-inspired 8KB centroid tile shared by all six query heads.
// FP16 centroids and paired LDS loads; FP32 norms/probabilities/accumulators.
// Scores and partition maxima remain in natural-exponential units.
__global__ __launch_bounds__(192) void scan(
    const float *__restrict__ qr, const uint32_t *__restrict__ pk,
    const uint32_t *__restrict__ pv, int length, int capacity, int parts,
    float scale, const float *__restrict__ cbk, const float *__restrict__ cbv,
    float *__restrict__ partial) {
  constexpr int TILE = 8, THREADS = 192;
  const int tid = threadIdx.x, lane = tid % 32;
  const int g = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int h = g * 6 + __builtin_amdgcn_readfirstlane(tid / 32);
  const int part = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int chunk = ((length + parts - 1) / parts + TILE - 1) / TILE * TILE;
  const int begin = part * chunk, end = min(length, begin + chunk);
  __shared__ half keys[TILE][D], values[TILE][D];
  __shared__ float books[32], knorms[TILE], vnorms[TILE];
  if (tid < 32)
    books[tid] = tid < 16 ? cbk[tid] : cbv[tid - 16];
  __syncthreads();
  float q[8], acc[8] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    q[i] = qr[h * D + (i / 2) * 64 + lane * 2 + i % 2] * scale;
  float maximum = -INFINITY, denominator = 0;
  for (int base = begin; base < end; base += TILE) {
    for (int z = tid; z < TILE * 32; z += THREADS) {
      const int row = __builtin_amdgcn_readfirstlane(z / 32);
      const int d = (z % 32) * 8;
      const int64_t offset = (int64_t(g) * capacity + base + row) * WORDS;
      const bool valid = base + row < end;
      uint32_t ki = valid ? pk[offset + 1 + z % 32] : 0;
      uint32_t vi = valid ? pv[offset + 1 + z % 32] : 0;
      float kn = valid ? __uint_as_float(pk[offset]) : 0;
      float vn = valid ? __uint_as_float(pv[offset]) : 0;
      if (z % 32 == 0) { knorms[row] = kn; vnorms[row] = vn; }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        keys[row][d + i] = __float2half(books[(ki >> (4 * i)) & 15]);
        values[row][d + i] = __float2half(books[16 + ((vi >> (4 * i)) & 15)]);
      }
    }
    __syncthreads();
    for (int t = 0; t < min(TILE, end - base); ++t) {
      float score = 0;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        float2 k = __half22float2(*reinterpret_cast<const half2 *>(&keys[t][(i / 2) * 64 + lane * 2]));
        score = fmaf(q[i], k.x, score);
        score = fmaf(q[i + 1], k.y, score);
      }
#pragma unroll
      for (int delta = 16; delta; delta >>= 1)
        score += __shfl_xor(score, delta, 32);
      score *= knorms[t];
      float next = fmaxf(maximum, score);
      float alpha = expf(maximum - next), beta = expf(score - next);
      denominator = alpha * denominator + beta;
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        float2 v = __half22float2(*reinterpret_cast<const half2 *>(&values[t][(i / 2) * 64 + lane * 2]));
        float weight = beta * vnorms[t];
        acc[i] = alpha * acc[i] + weight * v.x;
        acc[i + 1] = alpha * acc[i + 1] + weight * v.y;
      }
      maximum = next;
    }
    __syncthreads();
  }
  float *dst = partial + (h * parts + part) * STATE;
  if (lane == 0) { dst[0] = maximum; dst[1] = denominator; }
#pragma unroll
  for (int i = 0; i < 8; ++i)
    dst[2 + (i / 2) * 64 + lane * 2 + i % 2] = acc[i];
}

#endif

__global__ void merge(const float *partial, int parts, const float *pi,
                      half *output) {
  const int h = blockIdx.x, i = threadIdx.x;
  const float *base = partial + h * parts * STATE;
  // Experimental long-context profile: ceil(262144 / 128) partitions.
  __shared__ float reduction[D], weights[2048], rotated[D];
  float maximum = -INFINITY;
  for (int p = i; p < parts; p += D)
    if (base[p * STATE + 1] > 0)
      maximum = fmaxf(maximum, base[p * STATE]);
  reduction[i] = maximum;
  __syncthreads();
  for (int delta = D / 2; delta; delta >>= 1) {
    if (i < delta)
      reduction[i] = fmaxf(reduction[i], reduction[i + delta]);
    __syncthreads();
  }
  maximum = reduction[0];
  float denominator = 0;
  for (int p = i; p < parts; p += D) {
    float factor = base[p * STATE + 1] > 0 ? expf(base[p * STATE] - maximum) : 0;
    weights[p] = factor;
    denominator += factor * base[p * STATE + 1];
  }
  __syncthreads();
  reduction[i] = denominator;
  __syncthreads();
  for (int delta = D / 2; delta; delta >>= 1) {
    if (i < delta)
      reduction[i] += reduction[i + delta];
    __syncthreads();
  }
  denominator = reduction[0];
  float numerator = 0;
  for (int p = 0; p < parts; ++p)
    numerator += weights[p] * base[p * STATE + 2 + i];
  rotated[i] = numerator / denominator;
  __syncthreads();
  float result = 0;
  for (int j = 0; j < D; ++j)
    result = fmaf(rotated[j], pi[j * D + i], result);
  output[h * D + i] = __float2half(result);
}

// Bounded query chunks; history is reconstructed only in Ring's LDS tiles.
constexpr int APPEND_CHUNK = 128;
__global__ void rotate_append_query(const half *q, half *qr, const float *pi) {
  const int row = blockIdx.x, i = threadIdx.x;
  __shared__ float x[D];
  x[i] = __half2float(q[row * D + i]);
  __syncthreads();
  float sum = 0;
  for (int j = 0; j < D; ++j)
    sum = fmaf(x[j], pi[i * D + j], sum);
  // A rotation can exceed the FP16 range even for finite FP16 inputs.
  // Power-of-two scaling protects staging; Ring restores QK scaling in FP32.
  qr[row * D + i] = __float2half(sum * 0.0625f);
}

__global__ void join_append(const float *prefix, const float *suffix,
                            const float *plse, const float *slse,
                            const float *pi, half *output) {
  const int row = blockIdx.x, i = threadIdx.x;
  __shared__ float rotated[D];
  rotated[i] = prefix[row * D + i] * 16.f;
  __syncthreads();
  float original = 0;
  for (int j = 0; j < D; ++j)
    original = fmaf(rotated[j], pi[j * D + i], original);
  float m = fmaxf(plse[row], slse[row]);
  float a = expf(plse[row] - m), b = expf(slse[row] - m);
  output[row * D + i] = __float2half(
      (a * original + b * suffix[row * D + i]) / (a + b));
}

int partitions(int length) {
  // Offline gfx1151 policy. Bound both the merge scratch and short-context
  // launch overhead; no event timing or host autotuning in steady decode.
  return length <= 32 ? 1 : length <= 128 ? 4 : length <= 1024 ? 8
                                                           : (length + 127) / 128;
}
} // namespace

extern "C" HIP_KERNEL_API size_t hip_tq4_decode_workspace(int length) {
  if (length < 1 || length > 262144)
    return 0;
  return size_t(H * D + H * partitions(length) * STATE) * sizeof(float);
}

extern "C" HIP_KERNEL_API int hip_tq4_encode_append(
    void *stream, const void *key, const void *value, void *pk, void *pv,
    int count, int past, int capacity, const float *pi,
    const float *cbk, const float *cbv) {
  if (!key || !value || !pk || !pv || !pi || !cbk || !cbv || count < 1 ||
      past < 0 || capacity < 1 || capacity > 262144 ||
      int64_t(past) + count > capacity || pk == pv)
    return -1;
  hipLaunchKernelGGL(encode_append, dim3(G, count, 2), dim3(D), 0,
                    static_cast<hipStream_t>(stream), static_cast<const half *>(key),
                    static_cast<const half *>(value), static_cast<uint32_t *>(pk),
                    static_cast<uint32_t *>(pv), count, past, capacity, pi, cbk, cbv);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" HIP_KERNEL_API int hip_tq4_decode(
    void *stream, const void *query, const void *pk, const void *pv,
    void *output, int length, int capacity, float scale, const float *pi,
    const float *cbk, const float *cbv, void *workspace, size_t bytes) {
  const size_t required = hip_tq4_decode_workspace(length);
  if (!required || !query || !pk || !pv || !output || !pi || !cbk || !cbv ||
      !workspace || bytes < required || capacity < length || capacity > 262144 ||
      !std::isfinite(scale))
    return -1;
  hipStream_t s = static_cast<hipStream_t>(stream);
  float *qr = static_cast<float *>(workspace), *partial = qr + H * D;
  int parts = partitions(length);
  hipLaunchKernelGGL(rotate_query, dim3(H), dim3(D), 0, s,
                    static_cast<const half *>(query), qr, pi);
  if (hipGetLastError() != hipSuccess)
    return -1;
#ifdef TQ4_REFERENCE_DECODE
  hipLaunchKernelGGL(scan, dim3(H, parts), dim3(32), 0, s, qr,
#else
  hipLaunchKernelGGL(scan, dim3(G, parts), dim3(192), 0, s, qr,
#endif
                    static_cast<const uint32_t *>(pk), static_cast<const uint32_t *>(pv),
                    length, capacity, parts, scale, cbk, cbv, partial);
  if (hipGetLastError() != hipSuccess)
    return -1;
  hipLaunchKernelGGL(merge, dim3(H), dim3(D), 0, s, partial, parts, pi,
                    static_cast<half *>(output));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" HIP_KERNEL_API size_t hip_tq4_prefill_workspace(int count) {
#if HIP_EP_GQA_IMPL_ID != 0 && HIP_EP_RING_DEVICE_ENABLED
  if (count < 1 || count > 131072)
    return 0;
  const size_t rows = size_t(count < APPEND_CHUNK ? count : APPEND_CHUNK) * H;
  return rows * (D * sizeof(half) + (2 * D + 2) * sizeof(float));
#else
  return 0;
#endif
}

// Does not modify the cache. The caller encodes new K/V after attention.
extern "C" HIP_KERNEL_API int hip_tq4_prefill(
    void *stream, const void *query, const void *key, const void *value,
    const void *pk, const void *pv, void *output, int count, int past,
    int capacity, const float *pi, const float *cbk, const float *cbv,
    void *workspace, size_t bytes) {
  // The default GQA build omits ring_gqa.hip entirely. Do not reference Ring
  // exports there, or in an architecture slice that cannot execute Ring.
#if HIP_EP_GQA_IMPL_ID != 0 && HIP_EP_RING_DEVICE_ENABLED
  const size_t required = hip_tq4_prefill_workspace(count);
  if (!required || !query || !key || !value || !pk || !pv || !output ||
      !pi || !cbk || !cbv || !workspace || bytes < required || past < 1 ||
      capacity < 1 || capacity > 262144 || past > capacity - count ||
      !hip_gqa_ring_device_supported())
    return -1;
  auto s = static_cast<hipStream_t>(stream);
  int chunk = count < APPEND_CHUNK ? count : APPEND_CHUNK;
  half *qr = static_cast<half *>(workspace);
  float *prefix = reinterpret_cast<float *>(qr + chunk * H * D);
  float *suffix = prefix + chunk * H * D;
  float *plse = suffix + chunk * H * D, *slse = plse + chunk * H;
  (void)hipGetLastError();
  for (int start = 0; start < count; start += chunk) {
    int n = count - start < chunk ? count - start : chunk;
    const half *q = static_cast<const half *>(query) + int64_t(start) * H * D;
    half *out = static_cast<half *>(output) + int64_t(start) * H * D;
    hipLaunchKernelGGL(rotate_append_query, dim3(n * H), dim3(D), 0, s, q, qr, pi);
    if (hipGetLastError() != hipSuccess ||
        hip_gqa_ring_tq4_prefill(stream, qr, q, pk, pv, key, value, prefix,
                                suffix, plse, slse, n, start, past, capacity,
                                cbk, cbv) != 0)
      return -1;
    hipLaunchKernelGGL(join_append, dim3(n * H), dim3(D), 0, s, prefix, suffix,
                      plse, slse, pi, out);
    if (hipGetLastError() != hipSuccess)
      return -1;
  }
  return 0;
#else
  return int(hipErrorNotSupported);
#endif
}
