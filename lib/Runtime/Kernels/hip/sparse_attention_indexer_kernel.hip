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
constexpr int kFusedRows = 3;
constexpr int kTileRows = 16;
constexpr int kTileBlocks = 128;
constexpr size_t kChunkBudgetBytes = size_t{128} << 20;

__device__ __forceinline__ float waveSum(float x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    x += __shfl_xor(x, off);
  }
  return x;
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

// RMSNorm then split-half RoPE on the first R channels of one 128-wide
// vector held by a wave, lane l owning channels 4l .. 4l + 3. Every thread of
// the block must call it (block barriers).
template <typename T>
__device__ __forceinline__ void normRope(float (&x)[4], const T *weight,
                                         const T *cos_row, const T *sin_row,
                                         int rotary, float eps, float *lds) {
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
    x[i] = x[i] * inv * static_cast<float>(weight[4 * lane + i]);
  }
  store4(lds + 4 * lane, x);
  __syncthreads();
  const int half = rotary / 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = 4 * lane + i;
    if (d < rotary) {
      const float paired = d < half ? -lds[d + half] : lds[d - half];
      x[i] = x[i] * static_cast<float>(cos_row[d]) +
             paired * static_cast<float>(sin_row[d]);
    }
  }
  __syncthreads();
}

// Pooled, normalized and rotated key of block j (one wave). Every thread of
// the block must call it.
template <typename T>
__device__ __forceinline__ void pooledKey(
    const T *keys, const int *group, int ratio, const T *key_norm_weight,
    const T *cos_cache, const T *sin_cache, int rotary, int rotary_rows,
    size_t cos_batch_offset, float eps, float *lds, float (&x)[4]) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] = 0.0f;
  }
  for (int k = 0; k < ratio; ++k) {
    float v[4];
    load4(keys + static_cast<size_t>(group[k]) * kD + 4 * lane, v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[i] += v[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    x[i] /= static_cast<float>(ratio);
  }
  const size_t rope_row =
      cos_batch_offset +
      static_cast<size_t>(clampPos(group[0], rotary_rows)) * rotary;
  normRope<T>(x, key_norm_weight, cos_cache + rope_row, sin_cache + rope_row,
              rotary, eps, lds);
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
sai_append_kernel(const T *__restrict__ query, T *__restrict__ present,
                  int seq_len, int past_len, int total, int width,
                  int key_offset) {
  const int b = blockIdx.y;
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= seq_len * kD) {
    return;
  }
  const int s = i / kD;
  const int d = i % kD;
  present[(static_cast<size_t>(b) * total + past_len + s) * kD + d] =
      query[(static_cast<size_t>(b) * seq_len + s) * width + key_offset + d];
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

__global__ void __launch_bounds__(kThreads)
sai_count_kernel(const int64_t *__restrict__ mask, int *__restrict__ chunk_count,
                 int total, int num_chunks) {
  __shared__ int wave_sums[kWaves];
  const int b = blockIdx.y;
  bool flag[kScanItems];
  const int n = visibleFlags(mask, b, total,
                             blockIdx.x * kScanChunk + threadIdx.x * kScanItems,
                             flag);
  const int incl = blockInclusiveScan(n, wave_sums);
  if (threadIdx.x == kThreads - 1) {
    chunk_count[b * num_chunks + blockIdx.x] = incl;
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
                  const T *__restrict__ key_norm_weight,
                  const T *__restrict__ cos_cache,
                  const T *__restrict__ sin_cache, float *__restrict__ pooled,
                  int seq_len, int total, int max_blocks, int ratio,
                  int rotary, int rotary_rows, int64_t cos_batch_stride,
                  float eps) {
  __shared__ __align__(16) float lds[kWaves][kD];
  const int b = blockIdx.y;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int blocks = row_count[b * seq_len + seq_len - 1] / ratio;
  const int j0 = blockIdx.x * kWaves;
  if (j0 >= blocks) {
    return;
  }
  const int j = j0 + wave;
  const bool valid = j < blocks;
  float x[4];
  pooledKey<T>(present + static_cast<size_t>(b) * total * kD,
               visible + static_cast<size_t>(b) * total +
                   (valid ? j : j0) * ratio,
               ratio, key_norm_weight, cos_cache, sin_cache, rotary,
               rotary_rows, b * cos_batch_stride, eps, lds[wave], x);
  if (valid) {
    store4(pooled + (static_cast<size_t>(b) * max_blocks + j) * kD + 4 * lane,
           x);
  }
}

// One wave per (row, head) of the chunk rows [s0, s0 + rows).
template <typename T>
__global__ void __launch_bounds__(kThreads)
sai_query_kernel(const T *__restrict__ query,
                 const T *__restrict__ query_norm_weight,
                 const T *__restrict__ cos_cache,
                 const T *__restrict__ sin_cache, float *__restrict__ qrot,
                 int seq_len, int s0, int rows, int chunk_stride,
                 int num_heads, int past_len, int rotary, int rotary_rows,
                 int64_t cos_batch_stride, float eps) {
  __shared__ __align__(16) float lds[kWaves][kD];
  const int b = blockIdx.y;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int item = blockIdx.x * kWaves + wave;
  const bool valid = item < rows * num_heads;
  const int it = valid ? item : 0;
  const int sl = it / num_heads;
  const int h = it % num_heads;
  const int s = s0 + sl;

  float x[4];
  load4(query +
            (static_cast<size_t>(b) * seq_len + s) * (num_heads + 1) * kD +
            h * kD + 4 * lane,
        x);
  const size_t rope_row =
      b * cos_batch_stride +
      static_cast<size_t>(clampPos(past_len + s, rotary_rows)) * rotary;
  normRope<T>(x, query_norm_weight, cos_cache + rope_row, sin_cache + rope_row,
              rotary, eps, lds[wave]);
  if (valid) {
    store4(qrot +
               ((static_cast<size_t>(b) * chunk_stride + sl) * num_heads + h) *
                   kD +
               4 * lane,
           x);
  }
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
                        const T *__restrict__ cos_cache,
                        const T *__restrict__ sin_cache,
                        const float *__restrict__ qrot,
                        float *__restrict__ scores, int seq_len, int s0,
                        int rows, int chunk_stride, int num_heads, int total,
                        int max_blocks, int ratio, int rotary, int rotary_rows,
                        int64_t cos_batch_stride, float eps, float scale) {
  __shared__ __align__(16) float q_lds[kFusedRows * kMaxHeads * kD];
  __shared__ __align__(16) float lds[kWaves][kD];
  const int b = blockIdx.y;
  const int wave = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int blocks = row_count[b * seq_len + s0 + rows - 1] / ratio;
  const int j0 = blockIdx.x * kWaves;
  if (j0 >= blocks) {
    return;
  }
  const float *q_src =
      qrot + static_cast<size_t>(b) * chunk_stride * num_heads * kD;
  for (int idx = threadIdx.x; idx < rows * num_heads * kD; idx += kThreads) {
    q_lds[idx] = q_src[idx];
  }
  const int j = j0 + wave;
  const bool valid = j < blocks;
  float x[4];
  pooledKey<T>(present + static_cast<size_t>(b) * total * kD,
               visible + static_cast<size_t>(b) * total +
                   (valid ? j : j0) * ratio,
               ratio, key_norm_weight, cos_cache, sin_cache, rotary,
               rotary_rows, b * cos_batch_stride, eps, lds[wave], x);
  if (!valid) {
    return;
  }
  for (int r = 0; r < rows; ++r) {
    if (j >= row_count[b * seq_len + s0 + r] / ratio) {
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

// Tile of kTileRows rows x kTileBlocks blocks. Wave w owns rows 2w, 2w + 1
// and lane l blocks l + 32 * jj; q and the pooled keys are staged through LDS
// kKChunk channels at a time.
__global__ void __launch_bounds__(kThreads)
sai_score_kernel(const float *__restrict__ qrot,
                 const float *__restrict__ pooled,
                 const int *__restrict__ row_count, float *__restrict__ scores,
                 int seq_len, int s0, int rows, int chunk_stride, int num_heads,
                 int max_blocks, int ratio, float scale) {
  constexpr int RPT = kTileRows / kWaves;
  constexpr int JPT = kTileBlocks / 32;
  constexpr int KS = kKChunk + 4;
  static_assert(RPT * kWaves == kTileRows, "rows must tile the waves");
  __shared__ __align__(16) float q_lds[kTileRows * kMaxHeads * kKChunk];
  __shared__ __align__(16) float k_lds[kTileBlocks * KS];

  const int b = blockIdx.z;
  const int sl0 = blockIdx.y * kTileRows;
  const int tile_rows = rows - sl0 < kTileRows ? rows - sl0 : kTileRows;
  const int blocks =
      row_count[b * seq_len + s0 + sl0 + tile_rows - 1] / ratio;
  const int j0 = blockIdx.x * kTileBlocks;
  if (j0 >= blocks) {
    return;
  }

  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane = tid % 32;

  float acc[RPT][kMaxHeads][JPT];
#pragma unroll
  for (int rr = 0; rr < RPT; ++rr) {
#pragma unroll
    for (int h = 0; h < kMaxHeads; ++h) {
#pragma unroll
      for (int jj = 0; jj < JPT; ++jj) {
        acc[rr][h][jj] = 0.0f;
      }
    }
  }

  const float *q_src =
      qrot + (static_cast<size_t>(b) * chunk_stride + sl0) * num_heads * kD;
  const float *k_src = pooled + static_cast<size_t>(b) * max_blocks * kD;
  constexpr int kVecPerRow = kKChunk / 4;
  for (int dc = 0; dc < kD; dc += kKChunk) {
    __syncthreads();
    for (int idx = tid; idx < kTileRows * num_heads * kVecPerRow;
         idx += kThreads) {
      const int c = idx % kVecPerRow;
      const int rh = idx / kVecPerRow;
      const int r = rh / num_heads;
      const int h = rh % num_heads;
      float4_t v = float4_t{0.0f, 0.0f, 0.0f, 0.0f};
      if (r < tile_rows) {
        v = *reinterpret_cast<const float4_t *>(
            q_src + static_cast<size_t>(rh) * kD + dc + 4 * c);
      }
      *reinterpret_cast<float4_t *>(
          q_lds + (r * kMaxHeads + h) * kKChunk + 4 * c) = v;
    }
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
        for (int h = 0; h < kMaxHeads; ++h) {
          if (h < num_heads) {
            const float4_t qv = *reinterpret_cast<const float4_t *>(
                q_lds + ((wave * RPT + rr) * kMaxHeads + h) * kKChunk + dd);
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
        for (int h = 0; h < kMaxHeads; ++h) {
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

// One block per row: an MSB-first 8-bit radix select finds the key of the
// topk-th best block, the winners are gathered and bitonic-sorted in LDS.
__global__ void __launch_bounds__(kThreads)
sai_select_kernel(const float *__restrict__ scores,
                  const int *__restrict__ visible,
                  const int *__restrict__ row_count,
                  int *__restrict__ selected, int seq_len, int s0,
                  int chunk_stride, int total, int max_blocks, int ratio,
                  int block_topk, int capacity) {
  __shared__ unsigned hist[256];
  __shared__ unsigned long long keys[kMaxTopK];
  __shared__ int wave_sums[kWaves];
  __shared__ int bucket_lds;
  __shared__ int above_lds;
  __shared__ int in_bucket_lds;
  __shared__ int count_lds;

  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int b = blockIdx.y;
  const int sl = blockIdx.x;
  const int s = s0 + sl;
  const int count = row_count[b * seq_len + s];
  const int blocks = count / ratio;
  const int topk = blocks < block_topk ? blocks : block_topk;
  const float *row_scores =
      scores + (static_cast<size_t>(b) * chunk_stride + sl) * max_blocks;
  const int *vis = visible + static_cast<size_t>(b) * total;
  int *out = selected + (static_cast<size_t>(b) * seq_len + s) * capacity;

  unsigned long long threshold = 0;
  if (blocks > topk) {
    unsigned long long prefix = 0;
    unsigned long long prefix_mask = 0;
    int need = topk;
    for (int shift = 56; shift >= 0; shift -= 8) {
      hist[tid] = 0;
      __syncthreads();
      for (int base = 0; base < blocks; base += kThreads * kSelectUnroll) {
        float v[kSelectUnroll];
#pragma unroll
        for (int u = 0; u < kSelectUnroll; ++u) {
          const int j = base + u * kThreads + tid;
          v[u] = j < blocks ? row_scores[j] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < kSelectUnroll; ++u) {
          const int j = base + u * kThreads + tid;
          if (j >= blocks) {
            continue;
          }
          const unsigned long long key = blockKey(v[u], j);
          if ((key & prefix_mask) != prefix) {
            continue;
          }
          const unsigned bucket = static_cast<unsigned>(key >> shift) & 255u;
          const unsigned first = __builtin_amdgcn_readfirstlane(bucket);
          const unsigned long long same = __ballot(bucket == first);
          if (bucket != first) {
            atomicAdd(&hist[bucket], 1u);
          } else if (lane == __ffsll(static_cast<long long>(same)) - 1) {
            atomicAdd(&hist[bucket], static_cast<unsigned>(__popcll(same)));
          }
        }
      }
      __syncthreads();
      const int c = static_cast<int>(hist[255 - tid]);
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
  for (int j = tid; j < blocks; j += kThreads) {
    const unsigned long long key = blockKey(row_scores[j], j);
    if (key >= threshold) {
      const int pos = atomicAdd(&count_lds, 1);
      if (pos < kMaxTopK) {
        keys[pos] = key;
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < kMaxTopK; i += kThreads) {
    if (i >= topk) {
      keys[i] = 0;
    }
  }
  __syncthreads();
  if (topk > 1) {
    for (int size = 2; size <= kMaxTopK; size <<= 1) {
      for (int stride = size / 2; stride > 0; stride >>= 1) {
        const int lo = 2 * tid - (tid & (stride - 1));
        const int hi = lo + stride;
        const bool desc = (lo & size) == 0;
        const unsigned long long a = keys[lo];
        const unsigned long long c = keys[hi];
        if ((a < c) == desc) {
          keys[lo] = c;
          keys[hi] = a;
        }
        __syncthreads();
      }
    }
  }

  for (int idx = tid; idx < topk * ratio; idx += kThreads) {
    const int rank = idx / ratio;
    const int j = static_cast<int>(0xFFFFFFFFu -
                                   static_cast<unsigned>(keys[rank]));
    out[idx] = vis[j * ratio + idx % ratio];
  }
  const int tail = count - blocks * ratio;
  for (int idx = tid; idx < tail; idx += kThreads) {
    out[topk * ratio + idx] = vis[blocks * ratio + idx];
  }
  for (int idx = topk * ratio + tail + tid; idx < capacity; idx += kThreads) {
    out[idx] = -1;
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
                     const T *cos_cache, const T *sin_cache,
                     const int64_t *mask, const T *past_key,
                     int32_t *selected, T *present, char *ws, int batch,
                     int seq_len, int past_len, int num_heads, int rotary,
                     int rotary_rows, int64_t cos_batch_stride, int ratio,
                     int block_topk, int capacity, float eps, float scale) {
  const int total = static_cast<int>(l.total);
  hipError_t err = hipSuccess;
  if (past_len > 0) {
    err = hipMemcpy2DAsync(present, static_cast<size_t>(total) * kD * sizeof(T),
                           past_key,
                           static_cast<size_t>(past_len) * kD * sizeof(T),
                           static_cast<size_t>(past_len) * kD * sizeof(T),
                           batch, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
      return err;
    }
  }
  if (seq_len == 0) {
    return hipSuccess;
  }
  sai_append_kernel<T>
      <<<dim3(gridFor(static_cast<int64_t>(seq_len) * kD, kThreads), batch),
         kThreads, 0, stream>>>(query, present, seq_len, past_len, total,
                                (num_heads + 1) * kD, num_heads * kD);

  int *visible = reinterpret_cast<int *>(ws + l.visible);
  int *row_count = reinterpret_cast<int *>(ws + l.row_count);
  int *chunk_count = reinterpret_cast<int *>(ws + l.chunk_count);
  float *pooled = reinterpret_cast<float *>(ws + l.pooled);
  float *qrot = reinterpret_cast<float *>(ws + l.qrot);
  float *scores = reinterpret_cast<float *>(ws + l.scores);
  const int num_chunks = static_cast<int>(l.num_chunks);
  const int max_blocks = static_cast<int>(l.max_blocks);
  const int chunk_stride = static_cast<int>(l.chunk_rows);

  sai_count_kernel<<<dim3(num_chunks, batch), kThreads, 0, stream>>>(
      mask, chunk_count, total, num_chunks);
  sai_scatter_kernel<<<dim3(num_chunks, batch), kThreads, 0, stream>>>(
      mask, chunk_count, visible, row_count, total, past_len, seq_len,
      num_chunks);
  const bool fused = seq_len <= kFusedRows;
  if (max_blocks > 0 && !fused) {
    sai_pooled_kernel<T>
        <<<dim3(gridFor(max_blocks, kWaves), batch), kThreads, 0, stream>>>(
            present, visible, row_count, key_norm_weight, cos_cache,
            sin_cache, pooled, seq_len, total, max_blocks, ratio, rotary,
            rotary_rows, cos_batch_stride, eps);
  }
  for (int s0 = 0; s0 < seq_len; s0 += chunk_stride) {
    const int rows =
        seq_len - s0 < chunk_stride ? seq_len - s0 : chunk_stride;
    if (max_blocks > 0) {
      sai_query_kernel<T>
          <<<dim3(gridFor(static_cast<int64_t>(rows) * num_heads, kWaves),
                  batch),
             kThreads, 0, stream>>>(query, query_norm_weight, cos_cache,
                                    sin_cache, qrot, seq_len, s0, rows,
                                    chunk_stride, num_heads, past_len, rotary,
                                    rotary_rows, cos_batch_stride, eps);
      if (fused) {
        sai_pooled_score_kernel<T>
            <<<dim3(gridFor(max_blocks, kWaves), batch), kThreads, 0,
               stream>>>(present, visible, row_count, key_norm_weight,
                         cos_cache, sin_cache, qrot, scores, seq_len, s0,
                         rows, chunk_stride, num_heads, total, max_blocks,
                         ratio, rotary, rotary_rows, cos_batch_stride, eps,
                         scale);
      } else {
        sai_score_kernel<<<dim3(gridFor(max_blocks, kTileBlocks),
                                gridFor(rows, kTileRows), batch),
                           kThreads, 0, stream>>>(
            qrot, pooled, row_count, scores, seq_len, s0, rows, chunk_stride,
            num_heads, max_blocks, ratio, scale);
      }
    }
    sai_select_kernel<<<dim3(rows, batch), kThreads, 0, stream>>>(
        scores, visible, row_count, selected, seq_len, s0, chunk_stride, total,
        max_blocks, ratio, block_topk, capacity);
  }
  return hipGetLastError();
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
    int64_t seq_len, int64_t past_len, int64_t num_heads, int64_t head_size,
    int64_t rotary_width, int64_t rotary_rows, int64_t rotary_batched,
    int64_t compress_ratio, int64_t token_budget, int element_size_bytes,
    float epsilon, float scale) {
  if (!query || !query_norm_weight || !key_norm_weight || !cos_cache ||
      !sin_cache || (past_len > 0 && !past_key) || !selected_indices ||
      !present_key) {
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
      past_len + seq_len > INT32_MAX ||
      (element_size_bytes != 2 && element_size_bytes != 4)) {
    fprintf(stderr,
            "[custom_kernels] hip_sparse_attention_indexer: unsupported "
            "config B=%lld S=%lld P=%lld N=%lld D=%lld R=%lld X=%lld r=%lld "
            "budget=%lld elem=%d\n",
            (long long)batch, (long long)seq_len, (long long)past_len,
            (long long)num_heads, (long long)head_size,
            (long long)rotary_width, (long long)rotary_rows,
            (long long)compress_ratio, (long long)token_budget,
            element_size_bytes);
    return -1;
  }
  const uintptr_t vec_align = 4u * static_cast<unsigned>(element_size_bytes);
  if (reinterpret_cast<uintptr_t>(query) % vec_align != 0 ||
      reinterpret_cast<uintptr_t>(present_key) % vec_align != 0) {
    fprintf(stderr, "[custom_kernels] hip_sparse_attention_indexer: query and "
                    "present_key must be aligned to 4 elements\n");
    return -1;
  }
  if (batch == 0) {
    return 0;
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
  const int64_t cos_batch_stride = rotary_batched ? rotary_rows * rotary_width : 0;
  const int capacity = static_cast<int>(token_budget + compress_ratio - 1);
  const int block_topk = static_cast<int>(token_budget / compress_ratio);
  hipError_t err;
  if (element_size_bytes == 2) {
    err = launchAll<_Float16>(
        s, l, static_cast<const _Float16 *>(query),
        static_cast<const _Float16 *>(query_norm_weight),
        static_cast<const _Float16 *>(key_norm_weight),
        static_cast<const _Float16 *>(cos_cache),
        static_cast<const _Float16 *>(sin_cache), mask,
        static_cast<const _Float16 *>(past_key), selected_indices,
        static_cast<_Float16 *>(present_key), static_cast<char *>(workspace),
        static_cast<int>(batch), static_cast<int>(seq_len),
        static_cast<int>(past_len), static_cast<int>(num_heads),
        static_cast<int>(rotary_width), static_cast<int>(rotary_rows),
        cos_batch_stride, static_cast<int>(compress_ratio), block_topk,
        capacity, epsilon, scale);
  } else {
    err = launchAll<float>(
        s, l, static_cast<const float *>(query),
        static_cast<const float *>(query_norm_weight),
        static_cast<const float *>(key_norm_weight),
        static_cast<const float *>(cos_cache),
        static_cast<const float *>(sin_cache), mask,
        static_cast<const float *>(past_key), selected_indices,
        static_cast<float *>(present_key), static_cast<char *>(workspace),
        static_cast<int>(batch), static_cast<int>(seq_len),
        static_cast<int>(past_len), static_cast<int>(num_heads),
        static_cast<int>(rotary_width), static_cast<int>(rotary_rows),
        cos_batch_stride, static_cast<int>(compress_ratio), block_topk,
        capacity, epsilon, scale);
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
