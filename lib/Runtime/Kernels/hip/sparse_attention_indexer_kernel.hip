/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft SparseAttentionIndexer, policy_mode = qsa. Numerics follow the
 * ORT CUDA kernel: fp32 pooled keys, RMSNorm, RoPE and scores; blocks ranked
 * by score descending, then block index ascending.
 */

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace {

constexpr int kD = 128;
constexpr int kMaxHeads = 8;
constexpr int kMaxTopK = 512;
constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
constexpr int kScanItems = 4;
constexpr int kScanChunk = kThreads * kScanItems;
constexpr int kKChunk = 16;
constexpr int kSelectUnroll = 8;
constexpr int kPoolGroup = 4;
constexpr int kUniformSections = 4;
constexpr int kWideSelectThreads = 1024;
constexpr int kWideSelectItems = 32;
constexpr int kWideSelectRows = 16;
constexpr int kWideSelectMinBlocks = 8192;
static_assert(kWideSelectMinBlocks > kMaxTopK, "wide rows run the radix passes");
constexpr int kFusedRows = 3;
constexpr int kTileRows = 16;
constexpr int kTileBlocks = 128;
constexpr size_t kChunkBudgetBytes = size_t{128} << 20;

// Lane exchanges on the VALU: rowXor<M> reads lane ^ M within a 16-lane row
// (DPP row_xmask) and swapHalves reads lane ^ 16 (v_permlanex16). __shfl_xor
// lowers to ds_bpermute, an LDS round trip. Archs without these (the kernels
// only run on wave32 but must compile everywhere) keep __shfl_xor.
template <int M> __device__ __forceinline__ float rowXor(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
  static_assert(M > 0 && M < 16, "row_xmask stays within a 16-lane row");
  return __builtin_bit_cast(
      float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                         0x160 | M, 0xF, 0xF, false));
#else
  return __shfl_xor(v, M);
#endif
}

__device__ __forceinline__ float swapHalves(float v) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
  // The inline-asm copy into the tied "old" operand keeps it out of a VOPD
  // pair (see v6_swap_halves in gqa_kernel.hip).
  const int x = __builtin_bit_cast(int, v);
  int old;
  asm("v_mov_b32 %0, %1" : "=v"(old) : "v"(x));
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlanex16(old, x, 0x76543210u, 0xfedcba98u,
                                          false, false));
#else
  return __shfl_xor(v, 16);
#endif
}

__device__ __forceinline__ float waveSum(float x) {
  x += swapHalves(x);
  x += rowXor<8>(x);
  x += rowXor<4>(x);
  x += rowXor<2>(x);
  x += rowXor<1>(x);
  return x;
}

// Orders one wave's LDS accesses across its lanes without a block barrier.
__device__ __forceinline__ void waveLdsSync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Without a mask every position is visible: visible[k] = k and a row at
// past_len + s sees past_len + s + 1 positions. The kernels then get null
// visible / row_count arrays.
__device__ __forceinline__ int rowCount(const int *row_count, int b,
                                        int seq_len, int total, int s) {
  return row_count ? row_count[b * seq_len + s] : total - seq_len + s + 1;
}

__device__ __forceinline__ int visibleAt(const int *vis, int k) {
  return vis ? vis[k] : k;
}

__device__ __forceinline__ const int *visibleRow(const int *visible, int b,
                                                 int total) {
  return visible ? visible + static_cast<size_t>(b) * total : nullptr;
}

// Inclusive scan over the block; every thread must call it.
__device__ __forceinline__ int blockInclusiveScan(int v, int *wave_sums) {
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up(v, off);
    if (lane >= off) {
      v += y;
    }
  }
  if (lane == 31) {
    wave_sums[wave] = v;
  }
  __syncthreads();
  int add = 0;
  for (int w = 0; w < wave; ++w) {
    add += wave_sums[w];
  }
  __syncthreads();
  return v + add;
}

__device__ __forceinline__ int clampPos(int pos, int rows) {
  return pos < 0 ? 0 : (pos < rows ? pos : rows - 1);
}

typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef float float4_t __attribute__((ext_vector_type(4)));

// Channels 4 * lane .. 4 * lane + 3 of a 128-wide row; rows are 8-byte
// (fp16) or 16-byte (fp32) aligned.
__device__ __forceinline__ void load4(const _Float16 *p, float (&x)[4]) {
  const half4_t v = *reinterpret_cast<const half4_t *>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = static_cast<float>(v[i]);
  }
}

__device__ __forceinline__ void load4(const float *p, float (&x)[4]) {
  const float4_t v = *reinterpret_cast<const float4_t *>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = v[i];
  }
}

__device__ __forceinline__ void store4(float *p, const float (&x)[4]) {
  *reinterpret_cast<float4_t *>(p) = float4_t{x[0], x[1], x[2], x[3]};
}

// Split-half RoPE angles.
// - Without ids, cos / sin are [rows, width] tables, batch_stride apart per
//   batch (0 when shared).
// - With ids, they are [table_rows, width / 2] base tables. Frequency f of row
//   t in batch b is in MRoPE section section[f] and reads base row
//   ids[section[f] * section_stride + b * batch_stride + t].
// - With at most kUniformSections sections, section_bits also holds
//   section[f], in bits 2 * (f % 16) and 2 * (f % 16) + 1 of word f / 16. The
//   kernels read it from scalar registers.
template <typename T> struct Rope {
  const T *cos;
  const T *sin;
  const int64_t *ids;
  int64_t batch_stride;
  int64_t section_stride;
  int rows;
  int width;
  int table_rows;
  int sections;
  uint32_t section_bits[kD / 32];
  uint8_t section[kD / 2];
};

template <int M>
__device__ __forceinline__ void rowXor4(const float (&x)[4], float (&y)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    y[i] = rowXor<M>(x[i]);
  }
}

// Sets y[i] to the RoPE partner d ^ half of channel d = 4l + i < 2 * half that
// lane l holds. half is a power of two from 4 to 64, so the partner sits in
// lane l ^ (half / 4).
__device__ __forceinline__ void ropePartners(const float (&x)[4], int half,
                                             float (&y)[4]) {
  switch (half) {
  case 4:
    rowXor4<1>(x, y);
    break;
  case 8:
    rowXor4<2>(x, y);
    break;
  case 16:
    rowXor4<4>(x, y);
    break;
  case 32:
    rowXor4<8>(x, y);
    break;
  default:
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i] = swapHalves(x[i]);
    }
  }
}

// Base-table row of gathered position p: a negative position wraps once, as
// ONNX Gather indices do.
__device__ __forceinline__ size_t ropeRow(int64_t p, int table_rows) {
  p = p < 0 ? p + table_rows : p;
  return static_cast<size_t>(p < 0 ? 0
                                   : (p < table_rows ? p : table_rows - 1));
}

// Angles of the channels 4l .. 4l + 3 that lane l holds, at row
// clamp(pos, 0, rows - 1); every lane of the wave passes the same pos.
// Channels past width read channel 0's angle and are left unrotated. With up
// to kUniformSections sections, the row's position in each section is a
// scalar load and lanes pick theirs, so the table loads wait on one round
// trip instead of a section-map / id / table chain per channel.
template <typename T>
__device__ __forceinline__ void ropeFetch(const Rope<T> &rope, int b, int pos,
                                          float (&c)[4], float (&s)[4]) {
  const int lane = threadIdx.x % 32;
  const int t = __builtin_amdgcn_readfirstlane(clampPos(pos, rope.rows));
  const int half = rope.width / 2;
  int d[4], f[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    d[i] = 4 * lane + i < rope.width ? 4 * lane + i : 0;
    f[i] = d[i] < half ? d[i] : d[i] - half;
  }
  size_t at[4];
  if (rope.ids == nullptr) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      at[i] = b * rope.batch_stride + static_cast<size_t>(t) * rope.width + d[i];
    }
  } else {
    const int64_t *ids = rope.ids + b * rope.batch_stride + t;
    if (rope.sections <= kUniformSections) {
      int64_t p[kUniformSections];
#pragma unroll
      for (int m = 0; m < kUniformSections; ++m) {
        p[m] = ids[(m < rope.sections ? m : 0) * rope.section_stride];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int w = f[i] / 16;
        uint32_t bits = rope.section_bits[0];
#pragma unroll
        for (int k = 1; k < kD / 32; ++k) {
          bits = w == k ? rope.section_bits[k] : bits;
        }
        const int m = (bits >> (2 * (f[i] % 16))) & 3;
        int64_t q = p[0];
#pragma unroll
        for (int k = 1; k < kUniformSections; ++k) {
          q = m == k ? p[k] : q;
        }
        at[i] = ropeRow(q, rope.table_rows) * half + f[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        at[i] = ropeRow(ids[rope.section[f[i]] * rope.section_stride],
                        rope.table_rows) *
                    half +
                f[i];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c[i] = static_cast<float>(rope.cos[at[i]]);
    s[i] = static_cast<float>(rope.sin[at[i]]);
  }
}

// The norm weights of the channels lane l holds. Loaded ahead of the vector
// they scale, since loads complete in order.
template <typename T>
__device__ __forceinline__ void normWeight(const T *weight, float (&w)[4]) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = static_cast<float>(weight[4 * lane + i]);
  }
}

// RMSNorm with weights w, then split-half RoPE with ropeFetch's angles on the
// first width channels of one 128-wide vector held by a wave, lane l owning
// channels 4l .. 4l + 3. Every lane of the wave must call it; lds is the
// wave's own kD floats.
__device__ __forceinline__ void normRope(float (&x)[4], const float (&w)[4],
                                         const float (&c)[4],
                                         const float (&s)[4], int width,
                                         float eps, float *lds) {
  const int lane = threadIdx.x % 32;
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ss += x[i] * x[i];
  }
  ss = waveSum(ss);
  const float inv = rsqrtf(ss / static_cast<float>(kD) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = x[i] * inv * w[i];
  }
  const int half = width / 2;
  float paired[4];
  if (half >= 4 && (half & (half - 1)) == 0) {
    ropePartners(x, half, paired);
  } else {
    store4(lds + 4 * lane, x);
    waveLdsSync();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = 4 * lane + i;
      paired[i] = d < half ? lds[d + half] : lds[d - half];
    }
    waveLdsSync();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = 4 * lane + i;
    const float r = x[i] * c[i] + (d < half ? -paired[i] : paired[i]) * s[i];
    x[i] = d < width ? r : x[i];
  }
}

// Pooled, normalized and rotated key of the block whose positions are
// vis[first .. first + ratio - 1], or first .. first + ratio - 1 without vis
// (one wave). Every lane of the wave must call it.
template <bool kVis, typename T>
__device__ __forceinline__ void
pooledKeyOf(const T *keys, const int *vis, int first, int ratio,
            const T *key_norm_weight, const Rope<T> &rope, int b, float eps,
            float *lds, float (&x)[4]) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = 0.0f;
  }
  // kPoolGroup independent row loads at a time (a runtime-bounded loop waits
  // for each row before issuing the next). Adding +0 for the clamped rows
  // past ratio leaves the sum unchanged, since it starts at +0.
  float v[kPoolGroup][4];
  auto load = [&](int k0) {
#pragma unroll
    for (int u = 0; u < kPoolGroup; ++u) {
      int row = first + (k0 + u < ratio ? k0 + u : ratio - 1);
      if (kVis) {
        row = vis[row];
      }
      load4(keys + static_cast<size_t>(row) * kD + 4 * lane, v[u]);
    }
  };
  auto add = [&](int k0) {
#pragma unroll
    for (int u = 0; u < kPoolGroup; ++u) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[i] += k0 + u < ratio ? v[u][i] : 0.0f;
      }
    }
  };
  // The angles are loaded after the first rows, which the norm needs first
  // (loads complete in order).
  float w[4];
  normWeight(key_norm_weight, w);
  load(0);
  float c[4], s[4];
  ropeFetch(rope, b, kVis ? vis[first] : first, c, s);
  add(0);
  for (int k0 = kPoolGroup; k0 < ratio; k0 += kPoolGroup) {
    load(k0);
    add(k0);
  }
  // Scaling by an exact power-of-two reciprocal rounds like the division.
  if ((ratio & (ratio - 1)) == 0) {
    const float inv = 1.0f / static_cast<float>(ratio);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[i] *= inv;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[i] /= static_cast<float>(ratio);
    }
  }
  normRope(x, w, c, s, rope.width, eps, lds);
}

template <typename T>
__device__ __forceinline__ void pooledKey(const T *keys, const int *vis,
                                          int first, int ratio,
                                          const T *key_norm_weight,
                                          const Rope<T> &rope, int b,
                                          float eps, float *lds,
                                          float (&x)[4]) {
  if (vis) {
    pooledKeyOf<true>(keys, vis, first, ratio, key_norm_weight, rope, b, eps,
                      lds, x);
  } else {
    pooledKeyOf<false>(keys, vis, first, ratio, key_norm_weight, rope, b,
                       eps, lds, x);
  }
}

__device__ __forceinline__ int visibleFlags(const int64_t *mask, int b,
                                            int total, int t0,
                                            bool (&flag)[kScanItems]) {
  int count = 0;
#pragma unroll
  for (int i = 0; i < kScanItems; ++i) {
    const int t = t0 + i;
    flag[i] = t < total &&
              (mask == nullptr || mask[static_cast<size_t>(b) * total + t] != 0);
    count += flag[i] ? 1 : 0;
  }
  return count;
}

__device__ __forceinline__ void countChunk(const int64_t *mask, int *chunk_count,
                                           int b, int c, int total,
                                           int num_chunks) {
  __shared__ int wave_sums[kWaves];
  bool flag[kScanItems];
  const int n = visibleFlags(mask, b, total,
                             c * kScanChunk + threadIdx.x * kScanItems, flag);
  const int incl = blockInclusiveScan(n, wave_sums);
  if (threadIdx.x == kThreads - 1) {
    chunk_count[b * num_chunks + c] = incl;
  }
}

// visible[b][k] = k-th visible position; row_count[b][s] = number of visible
// positions <= past_len + s.
__global__ void __launch_bounds__(kThreads)
sai_scatter_kernel(const int64_t *__restrict__ mask,
                   const int *__restrict__ chunk_count,
                   int *__restrict__ visible, int *__restrict__ row_count,
                   int total, int past_len, int seq_len, int num_chunks) {
  __shared__ int wave_sums[kWaves];
  __shared__ int base_lds;
  const int b = blockIdx.y;
  const int c = blockIdx.x;
  int before = 0;
  for (int k = threadIdx.x; k < c; k += kThreads) {
    before += chunk_count[b * num_chunks + k];
  }
  before = blockInclusiveScan(before, wave_sums);
  if (threadIdx.x == kThreads - 1) {
    base_lds = before;
  }
  __syncthreads();
  const int base = base_lds;

  const int t0 = c * kScanChunk + threadIdx.x * kScanItems;
  bool flag[kScanItems];
  const int n = visibleFlags(mask, b, total, t0, flag);
  int pos = base + blockInclusiveScan(n, wave_sums) - n;
  int *vis = visible + static_cast<size_t>(b) * total;
#pragma unroll
  for (int i = 0; i < kScanItems; ++i) {
    const int t = t0 + i;
    if (flag[i]) {
      vis[pos++] = t;
    }
    if (t >= past_len && t < total) {
      row_count[b * seq_len + (t - past_len)] = pos;
    }
  }
}

// One wave per block j; blocks at or past the last row's block count are
// never scored and stay unwritten.
template <typename T>
__global__ void __launch_bounds__(kThreads)
sai_pooled_kernel(const T *__restrict__ present, const int *__restrict__ visible,
                  const int *__restrict__ row_count,
                  const T *__restrict__ key_norm_weight, const Rope<T> rope,
                  float *__restrict__ pooled, int seq_len, int total,
                  int present_stride, int max_blocks, int ratio, float eps) {
  __shared__ __align__(16) float lds[kWaves][kD];
  const int b = blockIdx.y;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int blocks =
      rowCount(row_count, b, seq_len, total, seq_len - 1) / ratio;
  const int j = blockIdx.x * kWaves + wave;
  if (j >= blocks) {
    return;
  }
  float x[4];
  pooledKey<T>(present + static_cast<size_t>(b) * present_stride * kD,
               visibleRow(visible, b, total), j * ratio, ratio,
               key_norm_weight, rope, b, eps, lds[wave], x);
  store4(pooled + (static_cast<size_t>(b) * max_blocks + j) * kD + 4 * lane,
         x);
}

// One wave per (row, head) of the chunk rows [s0, s0 + rows).
template <typename T>
__device__ __forceinline__ void
queryItems(const T *query, const T *query_norm_weight, const Rope<T> &rope,
           float *qrot, int b, int item_block, int seq_len, int s0, int rows,
           int chunk_stride, int num_heads, int past_len, float eps) {
  __shared__ __align__(16) float lds[kWaves][kD];
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int item = item_block * kWaves + wave;
  if (item >= rows * num_heads) {
    return;
  }
  const int sl = item / num_heads;
  const int h = item % num_heads;
  const int s = s0 + sl;

  float x[4], w[4], c[4], sn[4];
  normWeight(query_norm_weight, w);
  load4(query +
            (static_cast<size_t>(b) * seq_len + s) * (num_heads + 1) * kD +
            h * kD + 4 * lane,
        x);
  ropeFetch(rope, b, past_len + s, c, sn);
  normRope(x, w, c, sn, rope.width, eps, lds[wave]);
  store4(qrot +
             ((static_cast<size_t>(b) * chunk_stride + sl) * num_heads + h) *
                 kD +
             4 * lane,
         x);
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
sai_query_kernel(const T *__restrict__ query,
                 const T *__restrict__ query_norm_weight, const Rope<T> rope,
                 float *__restrict__ qrot, int seq_len, int s0, int rows,
                 int chunk_stride, int num_heads, int past_len, float eps) {
  queryItems<T>(query, query_norm_weight, rope, qrot, blockIdx.y, blockIdx.x,
                seq_len, s0, rows, chunk_stride, num_heads, past_len, eps);
}

template <int Bytes> struct RawVec;
template <> struct RawVec<8> { typedef uint2 type; };
template <> struct RawVec<16> { typedef uint4 type; };

// Grid x holds count_blocks mask-count blocks, then query_blocks blocks for
// the chunk at s0 = 0, then the blocks writing present_key rows
// [copy_start, total) (past_key rows followed by the new key rows, kCopy
// elements per thread). past_key and present may alias.
template <typename T, int kCopy>
__global__ void __launch_bounds__(kThreads)
sai_prologue_kernel(const T *__restrict__ query,
                    const T *__restrict__ query_norm_weight,
                    const Rope<T> rope, const int64_t *__restrict__ mask,
                    const T *past_key, T *present,
                    int *__restrict__ chunk_count, float *__restrict__ qrot,
                    int count_blocks, int query_blocks, int seq_len,
                    int past_len, int total, int copy_start, int past_stride,
                    int present_stride, int chunk_stride, int num_heads,
                    float eps) {
  const int b = blockIdx.y;
  const int x = blockIdx.x;
  if (x < count_blocks) {
    countChunk(mask, chunk_count, b, x, total, count_blocks);
    return;
  }
  if (x < count_blocks + query_blocks) {
    queryItems<T>(query, query_norm_weight, rope, qrot, b, x - count_blocks,
                  seq_len, 0, seq_len, chunk_stride, num_heads, past_len, eps);
    return;
  }
  typedef typename RawVec<kCopy * sizeof(T)>::type V;
  constexpr int kVecPerRow = kD / kCopy;
  const int64_t v =
      static_cast<int64_t>(x - count_blocks - query_blocks) * kThreads +
      threadIdx.x;
  if (v >= static_cast<int64_t>(total - copy_start) * kVecPerRow) {
    return;
  }
  const int t = copy_start + static_cast<int>(v / kVecPerRow);
  const int c = static_cast<int>(v % kVecPerRow) * kCopy;
  const T *src =
      t < past_len
          ? past_key + (static_cast<size_t>(b) * past_stride + t) * kD + c
          : query +
                (static_cast<size_t>(b) * seq_len + (t - past_len)) *
                    (num_heads + 1) * kD +
                num_heads * kD + c;
  *reinterpret_cast<V *>(present +
                         (static_cast<size_t>(b) * present_stride + t) * kD +
                         c) = *reinterpret_cast<const V *>(src);
}

__device__ __forceinline__ float finishScore(float score, float scale) {
  score *= scale;
  return score == 0.0f ? 0.0f : score;
}

// Few rows: each wave builds the key of one block and scores it against every
// row of the chunk, so pooled keys never go through memory.
template <typename T>
__global__ void __launch_bounds__(kThreads)
sai_pooled_score_kernel(const T *__restrict__ present,
                        const int *__restrict__ visible,
                        const int *__restrict__ row_count,
                        const T *__restrict__ key_norm_weight,
                        const Rope<T> rope, const float *__restrict__ qrot,
                        float *__restrict__ scores, int seq_len, int s0,
                        int rows, int chunk_stride, int num_heads, int total,
                        int present_stride, int max_blocks, int ratio,
                        float eps, float scale) {
  __shared__ __align__(16) float q_lds[kFusedRows * kMaxHeads * kD];
  __shared__ __align__(16) float lds[kWaves][kD];
  const int b = blockIdx.y;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int blocks =
      rowCount(row_count, b, seq_len, total, s0 + rows - 1) / ratio;
  const int j0 = blockIdx.x * kWaves;
  if (j0 >= blocks) {
    return;
  }
  // The query rows are loaded first and stored to LDS once the key is built,
  // so their loads overlap the key's.
  constexpr int kQVecs = kFusedRows * kMaxHeads * kD / 4 / kThreads;
  static_assert(kQVecs * kThreads * 4 == kFusedRows * kMaxHeads * kD,
                "query rows must tile the block");
  const float *q_src =
      qrot + static_cast<size_t>(b) * chunk_stride * num_heads * kD;
  const int q_vecs = rows * num_heads * kD / 4;
  float4_t qv[kQVecs];
#pragma unroll
  for (int u = 0; u < kQVecs; ++u) {
    const int idx = threadIdx.x + u * kThreads;
    qv[u] = *reinterpret_cast<const float4_t *>(q_src +
                                                4 * (idx < q_vecs ? idx : 0));
  }
  const int j = j0 + wave;
  const bool valid = j < blocks;
  float x[4];
  if (valid) {
    pooledKey<T>(present + static_cast<size_t>(b) * present_stride * kD,
                 visibleRow(visible, b, total), j * ratio, ratio,
                 key_norm_weight, rope, b, eps, lds[wave], x);
  }
#pragma unroll
  for (int u = 0; u < kQVecs; ++u) {
    const int idx = threadIdx.x + u * kThreads;
    if (idx < q_vecs) {
      *reinterpret_cast<float4_t *>(q_lds + 4 * idx) = qv[u];
    }
  }
  __syncthreads();
  if (!valid) {
    return;
  }
  for (int r = 0; r < rows; ++r) {
    if (j >= rowCount(row_count, b, seq_len, total, s0 + r) / ratio) {
      continue;
    }
    float score = 0.0f;
    for (int h = 0; h < num_heads; ++h) {
      float q[4];
      load4(q_lds + (r * num_heads + h) * kD + 4 * lane, q);
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dot = fmaf(q[i], x[i], dot);
      }
      score += fmaxf(waveSum(dot), 0.0f);
    }
    if (lane == 0) {
      scores[(static_cast<size_t>(b) * chunk_stride + r) * max_blocks + j] =
          finishScore(score, scale);
    }
  }
}

// Tile of kTileRows rows x kTileBlocks blocks.
// - Wave w owns rows 2w and 2w + 1; lane l owns blocks l + 32 * jj.
// - The pooled keys go through LDS kKChunk channels at a time. q is
//   wave-uniform and read with scalar loads, which keeps the inner loop off
//   the LDS bandwidth limit.
// - It computes kH >= num_heads heads, so the head loops carry no branches.
//   Heads past num_heads repeat the last one and are dropped.
template <int kH>
__global__ void __launch_bounds__(kThreads)
sai_score_kernel(const float *__restrict__ qrot,
                 const float *__restrict__ pooled,
                 const int *__restrict__ row_count, float *__restrict__ scores,
                 int seq_len, int s0, int rows, int chunk_stride, int num_heads,
                 int total, int max_blocks, int ratio, float scale) {
  constexpr int RPT = kTileRows / kWaves;
  constexpr int JPT = kTileBlocks / 32;
  constexpr int KS = kKChunk + 4;
  static_assert(RPT * kWaves == kTileRows, "rows must tile the waves");
  __shared__ __align__(16) float k_lds[kTileBlocks * KS];

  const int b = blockIdx.z;
  const int sl0 = blockIdx.y * kTileRows;
  const int tile_rows = rows - sl0 < kTileRows ? rows - sl0 : kTileRows;
  const int blocks =
      rowCount(row_count, b, seq_len, total, s0 + sl0 + tile_rows - 1) /
      ratio;
  const int j0 = blockIdx.x * kTileBlocks;
  if (j0 >= blocks) {
    return;
  }

  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lane = tid % 32;

  float acc[RPT][kH][JPT];
#pragma unroll
  for (int rr = 0; rr < RPT; ++rr) {
#pragma unroll
    for (int h = 0; h < kH; ++h) {
#pragma unroll
      for (int jj = 0; jj < JPT; ++jj) {
        acc[rr][h][jj] = 0.0f;
      }
    }
  }

  const float *q_src =
      qrot + (static_cast<size_t>(b) * chunk_stride + sl0) * num_heads * kD;
  const float *q_at[RPT][kH];
#pragma unroll
  for (int rr = 0; rr < RPT; ++rr) {
    const int r = wave * RPT + rr < tile_rows ? wave * RPT + rr : tile_rows - 1;
#pragma unroll
    for (int h = 0; h < kH; ++h) {
      q_at[rr][h] = q_src + (static_cast<size_t>(r) * num_heads +
                             (h < num_heads ? h : num_heads - 1)) *
                                kD;
    }
  }
  const float *k_src = pooled + static_cast<size_t>(b) * max_blocks * kD;
  constexpr int kVecPerRow = kKChunk / 4;
  for (int dc = 0; dc < kD; dc += kKChunk) {
    __syncthreads();
    for (int idx = tid; idx < kTileBlocks * kVecPerRow; idx += kThreads) {
      const int c = idx % kVecPerRow;
      const int r = idx / kVecPerRow;
      const int j = j0 + r;
      float4_t v = float4_t{0.0f, 0.0f, 0.0f, 0.0f};
      if (j < blocks) {
        v = *reinterpret_cast<const float4_t *>(
            k_src + static_cast<size_t>(j) * kD + dc + 4 * c);
      }
      *reinterpret_cast<float4_t *>(k_lds + r * KS + 4 * c) = v;
    }
    __syncthreads();
#pragma unroll 2
    for (int dd = 0; dd < kKChunk; dd += 4) {
      float4_t kv[JPT];
#pragma unroll
      for (int jj = 0; jj < JPT; ++jj) {
        kv[jj] = *reinterpret_cast<const float4_t *>(
            k_lds + (lane + 32 * jj) * KS + dd);
      }
#pragma unroll
      for (int rr = 0; rr < RPT; ++rr) {
#pragma unroll
        for (int h = 0; h < kH; ++h) {
          const float4_t qv =
              *reinterpret_cast<const float4_t *>(q_at[rr][h] + dc + dd);
#pragma unroll
          for (int jj = 0; jj < JPT; ++jj) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              acc[rr][h][jj] = fmaf(qv[e], kv[jj][e], acc[rr][h][jj]);
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (int rr = 0; rr < RPT; ++rr) {
    const int row = wave * RPT + rr;
    if (row >= tile_rows) {
      continue;
    }
    float *out =
        scores +
        (static_cast<size_t>(b) * chunk_stride + sl0 + row) * max_blocks;
#pragma unroll
    for (int jj = 0; jj < JPT; ++jj) {
      const int j = j0 + lane + 32 * jj;
      if (j < blocks) {
        float score = 0.0f;
#pragma unroll
        for (int h = 0; h < kH; ++h) {
          if (h < num_heads) {
            score += fmaxf(acc[rr][h][jj], 0.0f);
          }
        }
        out[j] = finishScore(score, scale);
      }
    }
  }
}

// Larger key = better block: orderable score bits, then the complemented
// block index so equal scores prefer the lower index. NaN ranks last.
__device__ __forceinline__ unsigned long long blockKey(float score, int j) {
  const unsigned u = __float_as_uint(score);
  const unsigned ord =
      score != score ? 0u : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
  return (static_cast<unsigned long long>(ord) << 32) |
         (0xFFFFFFFFu - static_cast<unsigned>(j));
}

template <int M> __device__ __forceinline__ float laneXor(float v) {
  return rowXor<M>(v);
}

template <> __device__ __forceinline__ float laneXor<16>(float v) {
  return swapHalves(v);
}

template <int M>
__device__ __forceinline__ unsigned long long laneXorKey(unsigned long long v) {
  auto half = [](unsigned w) {
    return __builtin_bit_cast(unsigned,
                              laneXor<M>(__builtin_bit_cast(float, w)));
  };
  return static_cast<unsigned long long>(half(static_cast<unsigned>(v >> 32)))
             << 32 |
         half(static_cast<unsigned>(v));
}

// Bitonic stage of stride M <= 16 for key index i held by lane i % 32: the
// lower index of a pair keeps the larger key in a descending size block.
template <int M>
__device__ __forceinline__ unsigned long long
bitonicLane(unsigned long long v, int i, int size) {
  const unsigned long long o = laneXorKey<M>(v);
  const bool keep_max = ((i & M) == 0) == ((i & size) == 0);
  return (v < o) == keep_max ? o : v;
}

// The stages of stride <= min(size / 2, 32) on one wave's 64 keys, lane l
// holding keys i and i + 32 (i = 64 * wave + l) in a and c.
__device__ __forceinline__ void bitonicWave(unsigned long long &a,
                                            unsigned long long &c, int i,
                                            int size) {
  if (size >= 64) {
    const bool desc = (i & size) == 0;
    if ((a < c) == desc) {
      const unsigned long long t = a;
      a = c;
      c = t;
    }
  }
  auto stage = [&](auto m) {
    constexpr int M = decltype(m)::value;
    if (size > M) {
      a = bitonicLane<M>(a, i, size);
      c = bitonicLane<M>(c, i + 32, size);
    }
  };
  stage(std::integral_constant<int, 16>{});
  stage(std::integral_constant<int, 8>{});
  stage(std::integral_constant<int, 4>{});
  stage(std::integral_constant<int, 2>{});
  stage(std::integral_constant<int, 1>{});
}

// One block of kT threads per row.
// - An MSB-first 8-bit radix select finds the key of the topk-th best block.
//   The winners are gathered and bitonic-sorted.
// - With kItems > 0, the row's scores (at most kT * kItems) are read once into
//   registers, item u of thread t holding block u * kT + t. The radix passes
//   then read no memory, so a few long rows (decode) run on one wide block.
// - counts, when given, receives the row's number of valid entries.
template <int kT, int kItems>
__global__ void __launch_bounds__(kT)
sai_select_kernel(const float *__restrict__ scores,
                  const int *__restrict__ visible,
                  const int *__restrict__ row_count,
                  int *__restrict__ selected, int *__restrict__ counts,
                  int seq_len, int s0, int chunk_stride, int total,
                  int max_blocks, int ratio, int block_topk, int capacity) {
  static_assert(kT >= 256 && kT % 32 == 0, "one thread per histogram bin");
  constexpr int kPairs = kMaxTopK / 2;
  __shared__ unsigned hist[256];
  __shared__ unsigned long long keys[kMaxTopK];
  __shared__ int wave_sums[kT / 32];
  __shared__ int bucket_lds;
  __shared__ int above_lds;
  __shared__ int in_bucket_lds;
  __shared__ int count_lds;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int b = blockIdx.y;
  const int sl = blockIdx.x;
  const int s = s0 + sl;
  const int count = rowCount(row_count, b, seq_len, total, s);
  const int blocks = count / ratio;
  const int topk = blocks < block_topk ? blocks : block_topk;
  const float *row_scores =
      scores + (static_cast<size_t>(b) * chunk_stride + sl) * max_blocks;
  const int *vis = visibleRow(visible, b, total);
  int *out = selected + (static_cast<size_t>(b) * seq_len + s) * capacity;

  constexpr int kCache = kItems > 0 ? kItems : 1;
  float cache[kCache];
  if (kItems > 0) {
#pragma unroll
    for (int u = 0; u < kCache; ++u) {
      const int j = u * kT + tid;
      cache[u] = j < blocks ? row_scores[j] : 0.0f;
    }
  }
  // fn(j, score) for every block j < blocks this thread owns.
  auto forEach = [&](auto &&fn) {
    if (kItems > 0) {
#pragma unroll
      for (int u = 0; u < kCache; ++u) {
        const int j = u * kT + tid;
        if (j < blocks) {
          fn(j, cache[u]);
        }
      }
      return;
    }
    for (int base = 0; base < blocks; base += kT * kSelectUnroll) {
      float v[kSelectUnroll];
#pragma unroll
      for (int u = 0; u < kSelectUnroll; ++u) {
        const int j = base + u * kT + tid;
        v[u] = j < blocks ? row_scores[j] : 0.0f;
      }
#pragma unroll
      for (int u = 0; u < kSelectUnroll; ++u) {
        const int j = base + u * kT + tid;
        if (j < blocks) {
          fn(j, v[u]);
        }
      }
    }
  };

  unsigned long long threshold = 0;
  if (blocks > topk) {
    unsigned long long prefix = 0;
    unsigned long long prefix_mask = 0;
    int need = topk;
    for (int shift = 56; shift >= 0; shift -= 8) {
      if (tid < 256) {
        hist[tid] = 0;
      }
      __syncthreads();
      forEach([&](int j, float score) {
        const unsigned long long key = blockKey(score, j);
        if ((key & prefix_mask) != prefix) {
          return;
        }
        const unsigned bucket = static_cast<unsigned>(key >> shift) & 255u;
        const unsigned first = __builtin_amdgcn_readfirstlane(bucket);
        const unsigned long long same = __ballot(bucket == first);
        if (bucket != first) {
          atomicAdd(&hist[bucket], 1u);
        } else if (lane == __ffsll(static_cast<long long>(same)) - 1) {
          atomicAdd(&hist[bucket], static_cast<unsigned>(__popcll(same)));
        }
      });
      __syncthreads();
      const int c = tid < 256 ? static_cast<int>(hist[255 - tid]) : 0;
      const int at_least = blockInclusiveScan(c, wave_sums);
      if (at_least - c < need && at_least >= need) {
        bucket_lds = 255 - tid;
        above_lds = at_least - c;
        in_bucket_lds = c;
      }
      __syncthreads();
      prefix |= static_cast<unsigned long long>(bucket_lds) << shift;
      prefix_mask |= 0xFFull << shift;
      need -= above_lds;
      const bool done = in_bucket_lds == need;
      __syncthreads();
      if (done) {
        break;
      }
    }
    threshold = prefix;
  }

  if (tid == 0) {
    count_lds = 0;
  }
  __syncthreads();
  forEach([&](int j, float score) {
    const unsigned long long key = blockKey(score, j);
    if (key >= threshold) {
      const int pos = atomicAdd(&count_lds, 1);
      if (pos < kMaxTopK) {
        keys[pos] = key;
      }
    }
  });
  __syncthreads();
  for (int i = tid; i < kMaxTopK; i += kT) {
    if (i >= topk) {
      keys[i] = 0;
    }
  }
  __syncthreads();
  // Bitonic sort, descending. The first kPairs threads' waves each hold 64
  // keys in registers and run the stages of stride <= 32 there; only the
  // strides reaching across waves go through LDS.
  if (topk > 1) {
    const bool sorter = tid < kPairs;
    const int i = 64 * (tid / 32) + lane;
    unsigned long long a = 0;
    unsigned long long c = 0;
    if (sorter) {
      a = keys[i];
      c = keys[i + 32];
    }
    for (int size = 2; size <= kMaxTopK; size <<= 1) {
      if (size > 64) {
        if (sorter) {
          keys[i] = a;
          keys[i + 32] = c;
        }
        __syncthreads();
        for (int stride = size / 2; stride >= 64; stride >>= 1) {
          if (sorter) {
            const int lo = 2 * tid - (tid & (stride - 1));
            const int hi = lo + stride;
            const bool desc = (lo & size) == 0;
            const unsigned long long x = keys[lo];
            const unsigned long long y = keys[hi];
            if ((x < y) == desc) {
              keys[lo] = y;
              keys[hi] = x;
            }
          }
          __syncthreads();
        }
        if (sorter) {
          a = keys[i];
          c = keys[i + 32];
        }
      }
      if (sorter) {
        bitonicWave(a, c, i, size);
      }
    }
    if (sorter) {
      keys[i] = a;
      keys[i + 32] = c;
    }
    __syncthreads();
  }

  for (int idx = tid; idx < topk * ratio; idx += kT) {
    const int rank = idx / ratio;
    const int j = static_cast<int>(0xFFFFFFFFu -
                                   static_cast<unsigned>(keys[rank]));
    out[idx] = visibleAt(vis, j * ratio + idx % ratio);
  }
  const int tail = count - blocks * ratio;
  for (int idx = tid; idx < tail; idx += kT) {
    out[topk * ratio + idx] = visibleAt(vis, blocks * ratio + idx);
  }
  for (int idx = topk * ratio + tail + tid; idx < capacity; idx += kT) {
    out[idx] = -1;
  }
  if (counts != nullptr && tid == 0) {
    counts[static_cast<size_t>(b) * seq_len + s] = topk * ratio + tail;
  }
}

struct Layout {
  int64_t total;
  int64_t max_blocks;
  int64_t num_chunks;
  int64_t chunk_rows;
  size_t visible;
  size_t row_count;
  size_t chunk_count;
  size_t pooled;
  size_t qrot;
  size_t scores;
  size_t bytes;
};

size_t alignUp(size_t v) { return (v + 255) & ~size_t{255}; }

Layout makeLayout(int64_t batch, int64_t seq_len, int64_t past_len,
                  int64_t num_heads, int64_t ratio) {
  Layout l{};
  l.total = past_len + seq_len;
  l.max_blocks = ratio > 0 ? l.total / ratio : 0;
  l.num_chunks = (l.total + kScanChunk - 1) / kScanChunk;
  const size_t per_row = static_cast<size_t>(batch) *
                         static_cast<size_t>(l.max_blocks + num_heads * kD) *
                         sizeof(float);
  l.chunk_rows = seq_len;
  if (per_row > 0 && seq_len > 0) {
    const int64_t fit = static_cast<int64_t>(kChunkBudgetBytes / per_row);
    l.chunk_rows = fit < 1 ? 1 : (fit < seq_len ? fit : seq_len);
  }
  size_t off = 0;
  auto take = [&](size_t bytes) {
    const size_t at = off;
    off += alignUp(bytes);
    return at;
  };
  l.visible = take(static_cast<size_t>(batch * l.total) * sizeof(int));
  l.row_count = take(static_cast<size_t>(batch * seq_len) * sizeof(int));
  l.chunk_count = take(static_cast<size_t>(batch * l.num_chunks) * sizeof(int));
  l.pooled = take(seq_len > kFusedRows ? static_cast<size_t>(batch *
                                                             l.max_blocks) *
                                             kD * sizeof(float)
                                       : 0);
  l.qrot = take(static_cast<size_t>(batch * l.chunk_rows * num_heads) * kD *
                sizeof(float));
  l.scores = take(static_cast<size_t>(batch * l.chunk_rows * l.max_blocks) *
                  sizeof(float));
  l.bytes = off;
  return l;
}

unsigned gridFor(int64_t n, int64_t per) {
  return static_cast<unsigned>((n + per - 1) / per);
}

template <typename T>
hipError_t launchAll(hipStream_t stream, const Layout &l, const T *query,
                     const T *query_norm_weight, const T *key_norm_weight,
                     const Rope<T> &rope, const int64_t *mask,
                     const T *past_key, int32_t *selected, int32_t *counts,
                     T *present, char *ws, int batch, int seq_len, int past_len,
                     int past_stride, int present_stride, int num_heads,
                     int ratio, int block_topk, int capacity, float eps,
                     float scale) {
  const int total = static_cast<int>(l.total);
  if (total == 0) {
    return hipSuccess;
  }
  // Without a mask the positions are the identity and no scatter runs.
  const bool masked = mask != nullptr;
  int *visible = masked ? reinterpret_cast<int *>(ws + l.visible) : nullptr;
  int *row_count =
      masked ? reinterpret_cast<int *>(ws + l.row_count) : nullptr;
  int *chunk_count = reinterpret_cast<int *>(ws + l.chunk_count);
  float *pooled = reinterpret_cast<float *>(ws + l.pooled);
  float *qrot = reinterpret_cast<float *>(ws + l.qrot);
  float *scores = reinterpret_cast<float *>(ws + l.scores);
  const int num_chunks = static_cast<int>(l.num_chunks);
  const int max_blocks = static_cast<int>(l.max_blocks);
  const int chunk_stride = static_cast<int>(l.chunk_rows);

  const bool prologue_query =
      seq_len > 0 && max_blocks > 0 && chunk_stride >= seq_len;
  const int count_blocks = seq_len > 0 && masked ? num_chunks : 0;
  const int query_blocks =
      prologue_query
          ? static_cast<int>(
                gridFor(static_cast<int64_t>(seq_len) * num_heads, kWaves))
          : 0;
  constexpr int kWide = 16 / sizeof(T);
  const bool wide = ((reinterpret_cast<uintptr_t>(query) |
                      (past_len > 0 ? reinterpret_cast<uintptr_t>(past_key)
                                    : 0) |
                      reinterpret_cast<uintptr_t>(present)) %
                     16) == 0;
  const int copy_start = past_key == present ? past_len : 0;
  const unsigned present_blocks =
      gridFor(static_cast<int64_t>(total - copy_start) *
                  (kD / (wide ? kWide : 4)),
              kThreads);
  const dim3 grid(count_blocks + query_blocks + present_blocks, batch);
  if (wide) {
    sai_prologue_kernel<T, kWide><<<grid, kThreads, 0, stream>>>(
        query, query_norm_weight, rope, mask, past_key, present, chunk_count,
        qrot, count_blocks, query_blocks, seq_len, past_len, total,
        copy_start, past_stride, present_stride, chunk_stride, num_heads, eps);
  } else {
    sai_prologue_kernel<T, 4><<<grid, kThreads, 0, stream>>>(
        query, query_norm_weight, rope, mask, past_key, present, chunk_count,
        qrot, count_blocks, query_blocks, seq_len, past_len, total,
        copy_start, past_stride, present_stride, chunk_stride, num_heads, eps);
  }
  if (seq_len == 0) {
    return hipGetLastError();
  }

  if (masked) {
    sai_scatter_kernel<<<dim3(num_chunks, batch), kThreads, 0, stream>>>(
        mask, chunk_count, visible, row_count, total, past_len, seq_len,
        num_chunks);
  }
  const bool fused = seq_len <= kFusedRows;
  if (max_blocks > 0 && !fused) {
    sai_pooled_kernel<T>
        <<<dim3(gridFor(max_blocks, kWaves), batch), kThreads, 0, stream>>>(
            present, visible, row_count, key_norm_weight, rope, pooled,
            seq_len, total, present_stride, max_blocks, ratio, eps);
  }
  for (int s0 = 0; s0 < seq_len; s0 += chunk_stride) {
    const int rows =
        seq_len - s0 < chunk_stride ? seq_len - s0 : chunk_stride;
    if (max_blocks > 0) {
      if (!prologue_query) {
        sai_query_kernel<T>
            <<<dim3(gridFor(static_cast<int64_t>(rows) * num_heads, kWaves),
                    batch),
               kThreads, 0, stream>>>(query, query_norm_weight, rope, qrot,
                                      seq_len, s0, rows, chunk_stride,
                                      num_heads, past_len, eps);
      }
      if (fused) {
        sai_pooled_score_kernel<T>
            <<<dim3(gridFor(max_blocks, kWaves), batch), kThreads, 0,
               stream>>>(present, visible, row_count, key_norm_weight, rope,
                         qrot, scores, seq_len, s0, rows, chunk_stride,
                         num_heads, total, present_stride, max_blocks, ratio,
                         eps, scale);
      } else {
        const dim3 score_grid(gridFor(max_blocks, kTileBlocks),
                              gridFor(rows, kTileRows), batch);
        if (num_heads <= 4) {
          sai_score_kernel<4><<<score_grid, kThreads, 0, stream>>>(
              qrot, pooled, row_count, scores, seq_len, s0, rows,
              chunk_stride, num_heads, total, max_blocks, ratio, scale);
        } else {
          sai_score_kernel<kMaxHeads><<<score_grid, kThreads, 0, stream>>>(
              qrot, pooled, row_count, scores, seq_len, s0, rows,
              chunk_stride, num_heads, total, max_blocks, ratio, scale);
        }
      }
    }
    // The wide block's fixed cost only pays off on long rows.
    if (static_cast<int64_t>(rows) * batch <= kWideSelectRows &&
        max_blocks >= kWideSelectMinBlocks &&
        max_blocks <= kWideSelectThreads * kWideSelectItems) {
      sai_select_kernel<kWideSelectThreads, kWideSelectItems>
          <<<dim3(rows, batch), kWideSelectThreads, 0, stream>>>(
              scores, visible, row_count, selected, counts, seq_len, s0,
              chunk_stride, total, max_blocks, ratio, block_topk, capacity);
    } else {
      sai_select_kernel<kThreads, 0>
          <<<dim3(rows, batch), kThreads, 0, stream>>>(
              scores, visible, row_count, selected, counts, seq_len, s0,
              chunk_stride, total, max_blocks, ratio, block_topk, capacity);
    }
  }
  return hipGetLastError();
}

template <typename T>
Rope<T> makeRope(const void *cos, const void *sin, const int64_t *ids,
                 const uint8_t *sections, int64_t batch, int64_t width,
                 int64_t rows, int64_t batched, int64_t id_sections,
                 int64_t id_rows) {
  Rope<T> rope{};
  rope.cos = static_cast<const T *>(cos);
  rope.sin = static_cast<const T *>(sin);
  rope.ids = ids;
  rope.width = static_cast<int>(width);
  if (ids == nullptr) {
    rope.rows = static_cast<int>(rows);
    rope.batch_stride = batched ? rows * width : 0;
    return rope;
  }
  rope.rows = static_cast<int>(id_rows);
  rope.table_rows = static_cast<int>(rows);
  rope.sections = static_cast<int>(id_sections < INT32_MAX ? id_sections
                                                           : INT32_MAX);
  rope.batch_stride = id_rows;
  rope.section_stride = batch * id_rows;
  for (int64_t f = 0; f < width / 2; ++f) {
    rope.section[f] = sections[f];
    if (id_sections <= kUniformSections) {
      rope.section_bits[f / 16] |= static_cast<uint32_t>(sections[f])
                                   << (2 * (f % 16));
    }
  }
  return rope;
}

} // namespace

extern "C" size_t hip_sparse_attention_indexer_workspace_size(
    int64_t batch, int64_t seq_len, int64_t past_len, int64_t num_heads,
    int64_t head_size, int64_t compress_ratio) {
  (void)head_size;
  return makeLayout(batch, seq_len, past_len, num_heads, compress_ratio).bytes;
}

extern "C" int hip_sparse_attention_indexer(
    void *stream, const void *query, const void *query_norm_weight,
    const void *key_norm_weight, const void *cos_cache, const void *sin_cache,
    const int64_t *mask, const void *past_key, int32_t *selected_indices,
    void *present_key, void *workspace, size_t workspace_bytes, int64_t batch,
    int64_t seq_len, int64_t past_len, int64_t past_rows, int64_t present_rows,
    int64_t num_heads, int64_t head_size,
    int64_t rotary_width, int64_t rotary_rows, int64_t rotary_batched,
    int64_t compress_ratio, int64_t token_budget, int element_size_bytes,
    float epsilon, float scale, const int64_t *rotary_position_ids,
    const uint8_t *rotary_sections, int64_t id_sections, int64_t id_rows,
    int32_t *selected_counts) {
  if (!query || !query_norm_weight || !key_norm_weight || !cos_cache ||
      !sin_cache || (past_len > 0 && !past_key) || !selected_indices ||
      !present_key || (rotary_position_ids && !rotary_sections)) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: null pointer\n");
    return -1;
  }
  if (batch < 0 || seq_len < 0 || past_len < 0 || head_size != kD ||
      num_heads < 1 || num_heads > kMaxHeads || rotary_width <= 0 ||
      rotary_width % 2 != 0 || rotary_width > kD || rotary_rows <= 0 ||
      compress_ratio <= 0 || token_budget <= 0 ||
      token_budget % compress_ratio != 0 ||
      token_budget / compress_ratio > kMaxTopK ||
      past_rows < past_len || present_rows < past_len + seq_len ||
      past_rows > INT32_MAX || present_rows > INT32_MAX ||
      (element_size_bytes != 2 && element_size_bytes != 4)) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: unsupported "
            "config B=%lld S=%lld P=%lld past_rows=%lld present_rows=%lld "
            "N=%lld D=%lld R=%lld X=%lld r=%lld budget=%lld elem=%d\n",
            (long long)batch, (long long)seq_len, (long long)past_len,
            (long long)past_rows, (long long)present_rows,
            (long long)num_heads, (long long)head_size,
            (long long)rotary_width, (long long)rotary_rows,
            (long long)compress_ratio, (long long)token_budget,
            element_size_bytes);
    return -1;
  }
  if (rotary_position_ids) {
    bool sections_ok = !rotary_batched && id_sections >= 1 && id_rows >= 1 &&
                       id_rows <= INT32_MAX && rotary_rows <= INT32_MAX;
    for (int64_t f = 0; sections_ok && f < rotary_width / 2; ++f) {
      sections_ok = rotary_sections[f] < id_sections;
    }
    if (!sections_ok) {
      fprintf(stderr,
              "[custom_kernels] hip_sparse_attention_indexer: gathered rotary "
              "needs rank 2 base tables, [M >= 1, B, X >= 1] ids "
              "(M=%lld X=%lld) and every section below M\n",
              (long long)id_sections, (long long)id_rows);
      return -1;
    }
  }
  if (past_key == present_key && past_rows != present_rows) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: aliased past_key "
            "(%lld rows) and present_key (%lld rows) must have the same "
            "rows\n",
            (long long)past_rows, (long long)present_rows);
    return -1;
  }
  const uintptr_t vec_align = 4u * static_cast<unsigned>(element_size_bytes);
  if (reinterpret_cast<uintptr_t>(query) % vec_align != 0 ||
      reinterpret_cast<uintptr_t>(present_key) % vec_align != 0 ||
      (past_len > 0 &&
       reinterpret_cast<uintptr_t>(past_key) % vec_align != 0)) {
    fprintf(stderr, "[custom_kernels] hip_sparse_attention_indexer: query, "
                    "past_key and present_key must be aligned to 4 "
                    "elements\n");
    return -1;
  }
  if (batch == 0) {
    return 0;
  }
  if (hipdnn_device_wave_size() != 32) {
    fprintf(stderr, "[custom_kernels] hip_sparse_attention_indexer: requires "
                    "a wave32 device\n");
    return -1;
  }
  const Layout l =
      makeLayout(batch, seq_len, past_len, num_heads, compress_ratio);
  if (l.bytes > workspace_bytes || (l.bytes > 0 && !workspace)) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: workspace %zu "
            "bytes, need %zu\n",
            workspace_bytes, l.bytes);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const int capacity = static_cast<int>(token_budget + compress_ratio - 1);
  const int block_topk = static_cast<int>(token_budget / compress_ratio);
  hipError_t err;
  if (element_size_bytes == 2) {
    err = launchAll<_Float16>(
        s, l, static_cast<const _Float16 *>(query),
        static_cast<const _Float16 *>(query_norm_weight),
        static_cast<const _Float16 *>(key_norm_weight),
        makeRope<_Float16>(cos_cache, sin_cache, rotary_position_ids,
                           rotary_sections, batch, rotary_width, rotary_rows,
                           rotary_batched, id_sections, id_rows),
        mask, static_cast<const _Float16 *>(past_key), selected_indices,
        selected_counts, static_cast<_Float16 *>(present_key),
        static_cast<char *>(workspace),
        static_cast<int>(batch), static_cast<int>(seq_len),
        static_cast<int>(past_len), static_cast<int>(past_rows),
        static_cast<int>(present_rows), static_cast<int>(num_heads),
        static_cast<int>(compress_ratio), block_topk, capacity, epsilon,
        scale);
  } else {
    err = launchAll<float>(
        s, l, static_cast<const float *>(query),
        static_cast<const float *>(query_norm_weight),
        static_cast<const float *>(key_norm_weight),
        makeRope<float>(cos_cache, sin_cache, rotary_position_ids,
                        rotary_sections, batch, rotary_width, rotary_rows,
                        rotary_batched, id_sections, id_rows),
        mask, static_cast<const float *>(past_key), selected_indices,
        selected_counts, static_cast<float *>(present_key),
        static_cast<char *>(workspace),
        static_cast<int>(batch), static_cast<int>(seq_len),
        static_cast<int>(past_len), static_cast<int>(past_rows),
        static_cast<int>(present_rows), static_cast<int>(num_heads),
        static_cast<int>(compress_ratio), block_topk, capacity, epsilon,
        scale);
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: launch failed "
            "(%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
