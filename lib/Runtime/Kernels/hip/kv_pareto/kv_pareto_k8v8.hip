/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// KV-Pareto, per-token codec `k8v8_g32_planar_v1`: INT8 K and V with one BF16
// scale and INT8 zero point per token, KV head and group of 32 channels.
// (KV-Pareto's per-block variant is not part of this file.)
//
// Compiled by default (HIP_EP_ENABLE_KV_PARETO=ON); inference is opt-in with
// HIPDNN_EP_USE_KV_PARETO=1. The codec lives inside ORT's own GQA cache
// tensors (selected by the provider options kvCacheQuantizationBits=8 +
// kv_cache_codec=k8v8_g32_planar_v1; the graph keeps the stock
// com.microsoft GroupQueryAttention node, no QDQ). This file owns every step
// behind one picker entry, hip_kv_pareto_k8v8_gqa(), which selects the phase:
//   decode (1 new token, history > 0): fused scan + merge (2 launches)
//     - scan reads packed history rows [0, past) once per KV group, attends to
//       the current token from its direct FP16 K/V, and (partition 0) writes
//       the token's packed row `past`;
//     - merge combines the FP32 partition summaries.
//   prefill (fresh or append): fused stage + native WMMA prefill
//     - stage dequantizes history rows into an FP16 BNSD workspace, copies the
//       new K/V rows next to them and encodes the new rows into the cache in
//       the same pass;
//     - the attention itself reuses hip-ep's tuned v8x flash prefill kernel.
// The runtime only decides between dense GQA and KV-Pareto and supplies the
// workspace and the prefill launch config (resolved with the same policy as
// dense GQA).
//
// Geometry: batch 1, 4 KV heads, head dimension 256, 4 or 6 query heads per KV
// head (Qwen3.5-4B H16, Qwen3.8-27B H24), FP16 activations, causal, external
// RoPE, capacity <= 262272, up to 262144 new tokens per call.
//
// Cache tensor [1, 4, C, 140] FP16 carriers = 4*C*280 bytes, one planar block:
//   INT8 payload [4][C][256] | BF16 scales [4][C][8] | INT8 zero points [4][C][8]
// Asymmetric INT8 per token, KV head and 32-channel group, with BF16-rounded
// scale intermediates (the kernel unit test checks the encoder bit for bit
// against a host implementation of these formulas):
//   scale = bf16(max(bf16(max - min), bf16(1e-6)) / 255)
//   zero  = clamp(-128 - rint(bf16(min / scale)), -128, 127)
//   code  = clamp(rint(bf16(x / scale)) + zero, -128, 127)
//   x'    = (code - zero) * scale          (FP32, no further rounding)

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
constexpr int D = 256, G = 4, GROUPS = D / 32, STATE = D + 2;
constexpr int MAX_CAPACITY = HIP_EP_KV_PARETO_MAX_CAPACITY;
constexpr int MAX_PREFILL = HIP_EP_KV_PARETO_MAX_NEW_TOKENS;
constexpr int MAX_PARTS = 2048;

bool heads_ok(int num_heads) { return num_heads == 16 || num_heads == 24; }

#if HIP_EP_KV_PARETO_DEVICE_ENABLED
bool is_decode(int seq_q, int past) { return seq_q == 1 && past > 0; }

struct Planar {
  int8_t *payload;
  uint16_t *scales; // BF16 bits
  int8_t *zeros;
};

__host__ __device__ inline Planar planar(const void *base, int capacity) {
  auto *bytes = static_cast<uint8_t *>(const_cast<void *>(base));
  const size_t payload = size_t(G) * capacity * D;
  const size_t scales = size_t(G) * capacity * GROUPS * sizeof(uint16_t);
  return {reinterpret_cast<int8_t *>(bytes),
          reinterpret_cast<uint16_t *>(bytes + payload),
          reinterpret_cast<int8_t *>(bytes + payload + scales)};
}

// Round-to-nearest-even FP32 -> BF16 (matches __float2bfloat16 / ml_dtypes).
__device__ __forceinline__ uint16_t to_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return uint16_t((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
__device__ __forceinline__ float from_bf16(uint16_t h) {
  return __uint_as_float(uint32_t(h) << 16);
}
__device__ __forceinline__ float bf(float x) { return from_bf16(to_bf16(x)); }

// Encode one 32-channel group of one row. Must be called by all 32 lanes of a
// width-32 lane group (lane = channel within the group).
__device__ __forceinline__ void encode_group(float x, int lane, size_t row,
                                             int group, const Planar &c) {
  float lo = x, hi = x;
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    lo = fminf(lo, __shfl_xor(lo, off, 32));
    hi = fmaxf(hi, __shfl_xor(hi, off, 32));
  }
  const float range = fmaxf(bf(hi - lo), bf(1e-6f));
  const uint16_t scale_bits = to_bf16(range / 255.0f);
  const float scale = from_bf16(scale_bits);
  const float zero =
      fminf(fmaxf(-128.0f - rintf(bf(lo / scale)), -128.0f), 127.0f);
  const float code =
      fminf(fmaxf(rintf(bf(x / scale)) + zero, -128.0f), 127.0f);
  c.payload[row * D + group * 32 + lane] = int8_t(code);
  if (lane == 0) {
    c.scales[row * GROUPS + group] = scale_bits;
    c.zeros[row * GROUPS + group] = int8_t(zero);
  }
}

__device__ __forceinline__ float dequant(const Planar &c, size_t row, int d) {
  const size_t m = row * GROUPS + d / 32;
  return (float(c.payload[row * D + d]) - float(c.zeros[m])) *
         from_bf16(c.scales[m]);
}

//===----------------------------------------------------------------------===//
// Encode rows [start, start+count) from BSHD FP16 K/V (also used to seed caches).
//===----------------------------------------------------------------------===//
__global__ __launch_bounds__(D) void encode(const __half *__restrict__ key,
                                           const __half *__restrict__ value,
                                           void *k_cache, void *v_cache,
                                           int start, int capacity) {
  const int token = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const bool is_value = blockIdx.z != 0;
  const __half *src = is_value ? value : key;
  const float x = __half2float(src[(size_t(token) * G + g) * D + d]);
  encode_group(x, d & 31, size_t(g) * capacity + start + token, d / 32,
               planar(is_value ? v_cache : k_cache, capacity));
}

//===----------------------------------------------------------------------===//
// Prefill stage, one 256-thread block per (row, KV head, K|V):
//   row <  past: dequantize the packed row into dense row `row`;
//   row >= past: copy the new FP16 row into dense row `row` and encode it into
//                packed row `row` (old packed rows are never rewritten).
// Dense layout BNSD [4][total][256], what the v8x prefill reads.
//===----------------------------------------------------------------------===//
__global__ __launch_bounds__(D) void stage_prefill(
    const __half *__restrict__ key, const __half *__restrict__ value,
    void *k_cache, void *v_cache, __half *__restrict__ dense_k,
    __half *__restrict__ dense_v, int past, int capacity, int total) {
  const int row = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const bool is_value = blockIdx.z != 0;
  const Planar c = planar(is_value ? v_cache : k_cache, capacity);
  __half *dst = (is_value ? dense_v : dense_k) + (size_t(g) * total + row) * D;
  const size_t packed_row = size_t(g) * capacity + row;
  if (row < past) {
    dst[d] = __float2half(dequant(c, packed_row, d));
  } else {
    const __half x =
        (is_value ? value : key)[(size_t(row - past) * G + g) * D + d];
    dst[d] = x;
    encode_group(__half2float(x), d & 31, packed_row, d / 32, c);
  }
}

//===----------------------------------------------------------------------===//
// Decode scan: grid (KV head, partition), 256 threads, 32-token tiles. The
// K/V tile is read from memory once and shared by all HPG query heads.
//   load:    thread (row = tid / 8, group = tid % 8) fetches the group's 32
//            code bytes with two 16-byte loads plus its BF16 scale and INT8
//            zero point (next tile prefetched into registers), and writes the
//            32 dequantized FP16 values to LDS (unused rows are zero-filled).
//   scores:  S[16 x 32] = Q[16 x 256] . K^T on the matrix units (WMMA, FP32
//            accumulation; Q rows >= HPG are zero). 8 waves = 2 token halves
//            x 4 channel quarters, 4 WMMAs each.
//   softmax: one max/sum reduction and one rescale per tile per head, natural
//            exponential units; P is stored as FP16 and the denominator sums
//            the same FP16 values.
//   values:  O[16 x 256] += P[16 x 32] . V[32 x 256] (WMMA, FP32); wave w owns
//            channels [32w, 32w + 32).
// The last partition also attends to the current token from its direct FP16
// K/V; partition 0 also encodes that token into packed row `past`. Requires
// the RDNA matrix units (gfx11/gfx12 encodings via hip_arch_compat.h); the
// runtime rejects the codec on wave64 devices.
//===----------------------------------------------------------------------===//
constexpr int TILE = 32, DECODE_THREADS = 256, ROW = D + 8; // 16-byte row pad

// 32 INT8 codes (two 16-byte words) -> 32 dequantized FP16 values at dst.
__device__ __forceinline__ void store_dequant(__half *dst, uint4 lo, uint4 hi,
                                              float scale, float zero) {
  const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  __half vals[32];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int b = 0; b < 4; ++b)
      vals[i * 4 + b] = __float2half(
          (float(int(int8_t(w[i] >> (8 * b)))) - zero) * scale);
  const uint4 *src = reinterpret_cast<const uint4 *>(vals);
  uint4 *out = reinterpret_cast<uint4 *>(dst);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    out[i] = src[i];
}

#if !HIPDNN_HAS_WMMA
typedef _Float16 k8v8_wfrag __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float k8v8_wacc __attribute__((ext_vector_type(8)));
static __device__ __forceinline__ k8v8_wacc k8v8_wmma(k8v8_wfrag, k8v8_wfrag, k8v8_wacc c) {
  __builtin_trap();
  return c;
}
#else
typedef _Float16 k8v8_wfrag __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float k8v8_wacc __attribute__((ext_vector_type(8)));
static __device__ __forceinline__ k8v8_wacc k8v8_wmma(k8v8_wfrag a, k8v8_wfrag b, k8v8_wacc c) {
  return HIPDNN_WMMA_F32_16X16X16_F16(a, b, c);
}
#endif

template <int HPG>
__global__ __launch_bounds__(DECODE_THREADS) void decode_scan(
    const __half *__restrict__ query, const __half *__restrict__ key,
    const __half *__restrict__ value, void *k_cache, void *v_cache, int past,
    int capacity, int parts, int chunk, float scale,
    float *__restrict__ partial) {
  constexpr int PSTR = TILE + 8, KQ = 4;
  const int tid = threadIdx.x, lane = tid % 32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
  const int g = __builtin_amdgcn_readfirstlane(blockIdx.x);
  const int part = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int begin = min(past, part * chunk), end = min(past, begin + chunk);
  const Planar K = planar(k_cache, capacity), V = planar(v_cache, capacity);
  __shared__ __align__(16) _Float16 ks[TILE][ROW], vs[TILE][ROW], qs[HPG][ROW];
  __shared__ __align__(16) _Float16 pl[HPG][PSTR];
  __shared__ float sp[KQ][HPG][TILE + 1], alphas[16];
  for (int i = tid; i < HPG * D; i += DECODE_THREADS)
    qs[i / D][i % D] = _Float16(__half2float(query[size_t(g) * HPG * D + i]) * scale);
  if (tid < 16)
    alphas[tid] = 1.0f; // rows >= HPG: O rows stay zero
  const int lr = tid / 8, lg = tid % 8;
  const int wl = lane % 16, pair = lane / 16;
  const int koff = hipdnn_wmma_k_off(pair);
  float m = -INFINITY, l = 0;
  k8v8_wacc o0 = {}, o1 = {};

  auto tile = [&](int n) {
    __syncthreads();
    { // scores
      const int nt = wave & 1, kq = wave >> 1;
      k8v8_wacc s = {};
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        const int c0 = kq * 64 + step * 16 + koff;
        k8v8_wfrag a = {};
        if (wl < HPG)
          a = *reinterpret_cast<const k8v8_wfrag *>(&qs[wl][c0]);
        const k8v8_wfrag b = *reinterpret_cast<const k8v8_wfrag *>(&ks[nt * 16 + wl][c0]);
        s = k8v8_wmma(a, b, s);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = hipdnn_wmma_acc_row(pair, e);
        if (r < HPG)
          sp[kq][r][nt * 16 + wl] = s[e];
      }
    }
    __syncthreads();
    if (wave < HPG) { // softmax, one rescale per tile
      float s = -INFINITY;
      if (lane < n)
        s = (sp[0][wave][lane] + sp[1][wave][lane]) + (sp[2][wave][lane] + sp[3][wave][lane]);
      float tile_max = s;
#pragma unroll
      for (int off = 16; off; off >>= 1)
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 32));
      const float next = fmaxf(m, tile_max);
      const float p = lane < n ? expf(s - next) : 0.0f;
      const float alpha = expf(m - next);
      const _Float16 ph = _Float16(p);
      float sum = float(ph); // denominator matches the FP16 weights used in P.V
#pragma unroll
      for (int off = 16; off; off >>= 1)
        sum += __shfl_xor(sum, off, 32);
      l = fmaf(alpha, l, sum);
      m = next;
      pl[wave][lane] = ph;
      if (lane == 0)
        alphas[wave] = alpha;
    }
    __syncthreads();
    { // values
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float a = alphas[hipdnn_wmma_acc_row(pair, e)];
        o0[e] *= a;
        o1[e] *= a;
      }
#pragma unroll
      for (int step = 0; step < 2; ++step) {
        k8v8_wfrag a = {};
        if (wl < HPG)
          a = *reinterpret_cast<const k8v8_wfrag *>(&pl[wl][step * 16 + koff]);
        k8v8_wfrag b0, b1;
#pragma unroll
        for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e) {
          b0[e] = vs[step * 16 + koff + e][wave * 32 + wl];
          b1[e] = vs[step * 16 + koff + e][wave * 32 + 16 + wl];
        }
        o0 = k8v8_wmma(a, b0, o0);
        o1 = k8v8_wmma(a, b1, o1);
      }
    }
  };

  uint4 k0{}, k1{}, v0{}, v1{};
  uint16_t kscale = 0, vscale = 0;
  int8_t kzero = 0, vzero = 0;
  bool valid = false;
  auto fetch = [&](int base) {
    const int row = base + lr;
    valid = row < end;
    if (valid) {
      const size_t r = size_t(g) * capacity + row;
      const uint4 *kp = reinterpret_cast<const uint4 *>(K.payload + r * D + lg * 32);
      const uint4 *vp = reinterpret_cast<const uint4 *>(V.payload + r * D + lg * 32);
      k0 = kp[0];
      k1 = kp[1];
      v0 = vp[0];
      v1 = vp[1];
      kscale = K.scales[r * GROUPS + lg];
      vscale = V.scales[r * GROUPS + lg];
      kzero = K.zeros[r * GROUPS + lg];
      vzero = V.zeros[r * GROUPS + lg];
    }
  };
  auto store_zero = [&](int row) {
    uint4 *kd = reinterpret_cast<uint4 *>(&ks[row][lg * 32]);
    uint4 *vd = reinterpret_cast<uint4 *>(&vs[row][lg * 32]);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      kd[i] = vd[i] = uint4{0, 0, 0, 0};
  };
  if (begin < end)
    fetch(begin);
  for (int base = begin; base < end; base += TILE) {
    const int n = min(TILE, end - base);
    __syncthreads();
    if (valid) {
      store_dequant(reinterpret_cast<__half *>(&ks[lr][lg * 32]), k0, k1, from_bf16(kscale), float(kzero));
      store_dequant(reinterpret_cast<__half *>(&vs[lr][lg * 32]), v0, v1, from_bf16(vscale), float(vzero));
    } else {
      store_zero(lr); // P is 0 there, but 0 x stale/NaN would poison O
    }
    if (base + TILE < end)
      fetch(base + TILE);
    tile(n);
  }
  if (part == parts - 1) {
    __syncthreads();
    if (lr > 0)
      store_zero(lr);
    if (tid < 2 * 8 * 4) {
      const bool is_value = tid >= 32;
      const int chunk16 = tid % 32;
      const uint4 *src = reinterpret_cast<const uint4 *>(
          (is_value ? value : key) + size_t(g) * D);
      reinterpret_cast<uint4 *>(is_value ? vs[0] : ks[0])[chunk16] = src[chunk16];
    }
    tile(1);
  }
  if (part == 0) {
    for (int task = wave; task < 2 * GROUPS; task += DECODE_THREADS / 32) {
      const int group = task % GROUPS;
      const bool is_value = task >= GROUPS;
      const float x =
          __half2float((is_value ? value : key)[g * D + group * 32 + lane]);
      encode_group(x, lane, size_t(g) * capacity + past, group,
                   is_value ? V : K);
    }
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int r = hipdnn_wmma_acc_row(pair, e);
    if (r < HPG) {
      float *dst = partial + (size_t(g * HPG + r) * parts + part) * STATE;
      dst[2 + wave * 32 + wl] = o0[e];
      dst[2 + wave * 32 + 16 + wl] = o1[e];
    }
  }
  if (wave < HPG && lane == 0) {
    float *dst = partial + (size_t(g * HPG + wave) * parts + part) * STATE;
    dst[0] = m;
    dst[1] = l;
  }
}

// Partitions: one full wave of resident scan blocks. With G x parts blocks
// every work-group processor (WGP) runs its resident blocks exactly once
// (gfx1151: 3 blocks per WGP x 20 WGPs = 60 blocks, 15 partitions per KV head),
// each streaming one contiguous run of whole tiles. On gfx1151 this was 5-22%
// faster from 2K to 128K (H24 and H16) than up to 512 blocks of at least 128
// tokens each, and the merge reads ~9x fewer FP32 partials. Counts just above
// one wave (72-80 blocks) leave a mostly idle second wave and were 25-35%
// slower.
//
// Residency is computed here, not with
// hipOccupancyMaxActiveBlocksPerMultiprocessor: HIP counts RDNA WGPs as
// multiprocessors but reports one CU's 64 KB of LDS for each, and its occupancy
// query then allows 1 block. A WGP's two CUs share 128 KB in WGP mode, which
// fits 3 scan blocks (40.7 KB at H24, 38.4 KB at H16), as measured.
template <int HPG> int wave_parts() {
  static std::atomic<int> cached[16];
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    device = 0;
  const bool cacheable = device >= 0 && device < 16;
  if (cacheable)
    if (const int known = cached[device].load(std::memory_order_relaxed))
      return known;
  int wgps = 0, lds_per_cu = 0, threads_per_wgp = 0,
      parts = 15; // fallback: gfx1151
  hipFuncAttributes attr{};
  if (hipDeviceGetAttribute(&wgps, hipDeviceAttributeMultiprocessorCount,
                            device) == hipSuccess &&
      hipDeviceGetAttribute(&lds_per_cu,
                            hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                            device) == hipSuccess &&
      hipDeviceGetAttribute(&threads_per_wgp,
                            hipDeviceAttributeMaxThreadsPerMultiProcessor,
                            device) == hipSuccess &&
      hipFuncGetAttributes(&attr, reinterpret_cast<const void *>(
                                      decode_scan<HPG>)) == hipSuccess &&
      wgps > 0 && lds_per_cu > 0 && attr.sharedSizeBytes > 0) {
    int resident = int(2 * size_t(lds_per_cu) / attr.sharedSizeBytes);
    const int by_threads = threads_per_wgp / DECODE_THREADS;
    if (by_threads > 0 && by_threads < resident)
      resident = by_threads;
    parts = wgps * (resident > 0 ? resident : 1) / G;
  }
  parts = parts < 1 ? 1 : parts > MAX_PARTS ? MAX_PARTS : parts;
  if (cacheable)
    cached[device].store(parts, std::memory_order_relaxed);
  return parts;
}
// Tiles per partition for `history` rows (history > 0).
int decode_chunk_tiles(int history, int hpg) {
  const int tiles = (history + TILE - 1) / TILE;
  const int wave = hpg == 6 ? wave_parts<6>() : wave_parts<4>();
  const int parts = tiles < wave ? tiles : wave;
  return (tiles + parts - 1) / parts;
}
int decode_chunk(int history, int hpg) {
  return decode_chunk_tiles(history, hpg) * TILE;
}
int partitions(int history, int hpg) {
  const int tiles = (history + TILE - 1) / TILE;
  const int per = decode_chunk_tiles(history, hpg);
  return (tiles + per - 1) / per; // no empty partitions
}

// Merge partition summaries per query head; empty partitions (denominator 0)
// are skipped.
__global__ __launch_bounds__(D) void merge(const float *__restrict__ partial,
                                          int parts, __half *__restrict__ out) {
  const int h = blockIdx.x, i = threadIdx.x;
  const float *base = partial + size_t(h) * parts * STATE;
  __shared__ float reduction[D], weights[MAX_PARTS];
  float maximum = -INFINITY;
  for (int p = i; p < parts; p += D)
    if (base[size_t(p) * STATE + 1] > 0)
      maximum = fmaxf(maximum, base[size_t(p) * STATE]);
  reduction[i] = maximum;
  __syncthreads();
  for (int delta = D / 2; delta; delta >>= 1) {
    if (i < delta)
      reduction[i] = fmaxf(reduction[i], reduction[i + delta]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float denominator = 0;
  for (int p = i; p < parts; p += D) {
    const float l = base[size_t(p) * STATE + 1];
    const float w = l > 0 ? expf(base[size_t(p) * STATE] - maximum) : 0;
    weights[p] = w;
    denominator += w * l;
  }
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
    numerator = fmaf(weights[p], base[size_t(p) * STATE + 2 + i], numerator);
  out[h * D + i] = __float2half(numerator / denominator);
}

#endif // HIP_EP_KV_PARETO_DEVICE_ENABLED
} // namespace

// Validated on Strix Halo (gfx1151) only. The gfx11-generic slice also loads
// on other RDNA3 parts; they report unsupported and keep existing GQA.
extern "C" HIP_KERNEL_API int hip_kv_pareto_device_supported(void) {
#if HIP_EP_KV_PARETO_DEVICE_ENABLED
  static std::atomic<int> cached[16]; // 0 unknown, 1 supported, -1 not
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return 0;
  const bool cacheable = device >= 0 && device < 16;
  if (cacheable)
    if (const int known = cached[device].load(std::memory_order_relaxed))
      return known > 0;
  hipDeviceProp_t props{};
  int supported = 0;
  if (hipGetDeviceProperties(&props, device) == hipSuccess)
    supported = std::strncmp(props.gcnArchName, "gfx1151", 7) == 0 &&
                (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':');
  if (cacheable)
    cached[device].store(supported ? 1 : -1, std::memory_order_relaxed);
  return supported;
#else
  return 0;
#endif
}

extern "C" HIP_KERNEL_API int hip_kv_pareto_k8v8_supported(int num_heads,
                                                 int kv_num_heads,
                                                 int head_dim) {
  return heads_ok(num_heads) && kv_num_heads == G && head_dim == D;
}

#if HIP_EP_KV_PARETO_DEVICE_ENABLED
extern "C" HIP_KERNEL_API size_t hip_kv_pareto_k8v8_workspace(int num_heads,
                                                       int seq_q, int past) {
  if (!heads_ok(num_heads) || seq_q < 1 || seq_q > MAX_PREFILL || past < 0 ||
      int64_t(past) + seq_q > MAX_CAPACITY)
    return 0;
  if (is_decode(seq_q, past))
    return size_t(num_heads) * partitions(past, num_heads / G) * STATE *
           sizeof(float);
  return size_t(2) * (size_t(past) + seq_q) * G * D * sizeof(__half);
}

extern "C" HIP_KERNEL_API int hip_kv_pareto_k8v8_encode(void *stream, const void *key,
                                              const void *value, void *k_cache,
                                              void *v_cache, int start,
                                              int count, int capacity) {
  if (!key || !value || !k_cache || !v_cache || k_cache == v_cache ||
      count < 1 || start < 0 || capacity < 1 || capacity > MAX_CAPACITY ||
      int64_t(start) + count > capacity || !hip_kv_pareto_device_supported())
    return -1;
  hipLaunchKernelGGL(encode, dim3(count, G, 2), dim3(D), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(key),
                     static_cast<const __half *>(value), k_cache, v_cache,
                     start, capacity);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" HIP_KERNEL_API int hip_kv_pareto_k8v8_gqa(
    void *stream, const void *query, const void *key, const void *value,
    void *k_cache, void *v_cache, void *output, int num_heads, int seq_q,
    int past, int capacity, float scale, void *workspace, size_t bytes,
    const int *prefill_config) {
  const size_t required = hip_kv_pareto_k8v8_workspace(num_heads, seq_q, past);
  if (!required || !query || !key || !value || !k_cache || !v_cache ||
      k_cache == v_cache || !output || !workspace || bytes < required ||
      capacity < 1 || capacity > MAX_CAPACITY ||
      int64_t(past) + seq_q > capacity || !std::isfinite(scale) ||
      scale <= 0.0f || !hip_kv_pareto_device_supported())
    return -1;
  auto s = static_cast<hipStream_t>(stream);
  const auto *q = static_cast<const __half *>(query);
  const auto *k = static_cast<const __half *>(key);
  const auto *v = static_cast<const __half *>(value);
  auto *out = static_cast<__half *>(output);
  (void)hipGetLastError();
  if (is_decode(seq_q, past)) {
    const int parts = partitions(past, num_heads / G);
    float *partial = static_cast<float *>(workspace);
    const int chunk = decode_chunk(past, num_heads / G);
    if (num_heads == 24)
      hipLaunchKernelGGL(decode_scan<6>, dim3(G, parts), dim3(DECODE_THREADS),
                         0, s, q, k, v, k_cache, v_cache, past, capacity, parts,
                         chunk, scale, partial);
    else
      hipLaunchKernelGGL(decode_scan<4>, dim3(G, parts), dim3(DECODE_THREADS),
                         0, s, q, k, v, k_cache, v_cache, past, capacity, parts,
                         chunk, scale, partial);
    if (hipGetLastError() != hipSuccess)
      return -1;
    hipLaunchKernelGGL(merge, dim3(num_heads), dim3(D), 0, s, partial, parts,
                       out);
    return hipGetLastError() == hipSuccess ? 0 : -1;
  }
  const int total = past + seq_q;
  auto *dense_k = static_cast<__half *>(workspace);
  auto *dense_v = dense_k + size_t(total) * G * D;
  hipLaunchKernelGGL(stage_prefill, dim3(total, G, 2), dim3(D), 0, s, k, v,
                     k_cache, v_cache, dense_k, dense_v, past, capacity, total);
  if (hipGetLastError() != hipSuccess)
    return -1;
  // Attention over [dequantized history | direct FP16 new rows] with hip-ep's
  // tuned fused WMMA prefill (v8x at head dim 256), BNSD max_seq = total.
  if (prefill_config)
    return hip_gqa_flash_prefill_v3_configured(
        stream, q, dense_k, dense_v, out, 1, num_heads, G, seq_q, total, D,
        total, past, scale, -1, nullptr, num_heads, 0, prefill_config[0],
        prefill_config[1], prefill_config[2], prefill_config[3],
        prefill_config[4]);
  return hip_gqa_flash_prefill_v2(stream, q, dense_k, dense_v, out, 1,
                                  num_heads, G, seq_q, total, D, total, past,
                                  scale);
}
#else
// Kernel slices without KV-Pareto device code export the same symbols, so the
// runtime links unchanged and reports the codec as unsupported.
extern "C" HIP_KERNEL_API size_t hip_kv_pareto_k8v8_workspace(int, int, int) {
  return 0;
}
extern "C" HIP_KERNEL_API int hip_kv_pareto_k8v8_encode(void *, const void *,
                                                        const void *, void *,
                                                        void *, int, int, int) {
  return -1;
}
extern "C" HIP_KERNEL_API int
hip_kv_pareto_k8v8_gqa(void *, const void *, const void *, const void *, void *,
                       void *, void *, int, int, int, int, float, void *,
                       size_t, const int *) {
  return -1;
}
#endif // HIP_EP_KV_PARETO_DEVICE_ENABLED
