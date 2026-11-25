/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

struct DeviceMeta {
  const int32_t *cumulative;
  const int32_t *past;
  const int32_t *block_table;
};

__device__ int find_batch(const int32_t *cumulative, int batch, int token) {
  for (int b = 0; b < batch; ++b) {
    if (token >= cumulative[b] && token < cumulative[b + 1])
      return b;
  }
  return -1;
}

__device__ int physical_slot(const int32_t *block_table, int max_blocks,
                             int block_size, int batch_index, int position) {
  if (position < 0)
    return -1;
  const int column = position / block_size;
  if (column >= max_blocks)
    return -1;
  const int block_id = block_table[batch_index * max_blocks + column];
  if (block_id < 0)
    return -1;
  return block_id * block_size + (position % block_size);
}

__global__ void write_paged_kv(const __half *key, const __half *value,
                               __half *key_cache, __half *value_cache,
                               DeviceMeta meta, int num_tokens, int batch,
                               int kv_heads, int head_size, int block_size,
                               int max_blocks, int cache_slots) {
  const int token = blockIdx.x;
  if (token >= num_tokens)
    return;
  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0)
    return;
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  const int slot =
      physical_slot(meta.block_table, max_blocks, block_size, batch_index, position);
  if (slot < 0 || slot >= cache_slots)
    return;

  const int width = kv_heads * head_size;
  const __half *key_row = key + static_cast<int64_t>(token) * width;
  const __half *value_row = value + static_cast<int64_t>(token) * width;
  __half *key_dst = key_cache + static_cast<int64_t>(slot) * width;
  __half *value_dst = value_cache + static_cast<int64_t>(slot) * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    key_dst[i] = key_row[i];
    value_dst[i] = value_row[i];
  }
}

constexpr int kAttentionThreads = 256;
constexpr int kMaxHeadSize = 512;
// Elements of a head a single lane owns when a wave is at least 32 lanes wide.
constexpr int kMaxDimsPerLane = kMaxHeadSize / 32;

__device__ float wave_sum(float value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    value += __shfl_xor(value, offset, warpSize);
  return value;
}

// Floats of one split partial: running max, denominator, then head_size
// unnormalized accumulators.
__host__ __device__ inline int64_t partial_stride(int head_size) {
  return head_size + 2;
}

// One block per (query token, head). Each wave walks a strided subset of the
// causal context with an online softmax; the per-wave partials are merged
// through shared memory. A lane owns dims lane, lane + warpSize, ...
// With `partials` set, blockIdx.z selects a `chunk`-position slice of the
// context and the block writes its unnormalized result for
// paged_attention_reduce instead of the output row.
__global__ void __launch_bounds__(kAttentionThreads)
    paged_attention_fp16(const __half *query, const __half *key_cache,
                         const __half *value_cache, __half *output,
                         float *partials, DeviceMeta meta, int num_tokens,
                         int batch, int num_heads, int kv_heads, int head_size,
                         int block_size, int max_blocks, int cache_slots,
                         float scale, int chunk, int splits) {
  extern __shared__ float shared[];
  const int num_waves = blockDim.x / warpSize;
  float *wave_max = shared;
  float *wave_denom = shared + num_waves;
  float *wave_acc = shared + 2 * num_waves;

  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize;
  const int wave = tid / warpSize;
  __half *out =
      output + (static_cast<int64_t>(token) * num_heads + head) * head_size;

  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0) {
    for (int d = tid; d < head_size; d += blockDim.x)
      out[d] = __float2half(0.f);
    return;
  }
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  int first = 0;
  int last = position;
  if (partials) {
    first = static_cast<int>(blockIdx.z) * chunk;
    if (first > position)
      return;
    last = min(position, first + chunk - 1);
  }
  const int kv_head = head / (num_heads / kv_heads);
  const int kv_width = kv_heads * head_size;
  const __half *q =
      query + (static_cast<int64_t>(token) * num_heads + head) * head_size;

  float q_reg[kMaxDimsPerLane];
  float acc[kMaxDimsPerLane];
  for (int i = 0; i < kMaxDimsPerLane; ++i) {
    const int d = lane + i * warpSize;
    q_reg[i] = d < head_size ? __half2float(q[d]) : 0.f;
    acc[i] = 0.f;
  }

  float running_max = -INFINITY;
  float running_denom = 0.f;
  for (int past_pos = first + wave; past_pos <= last; past_pos += num_waves) {
    const int slot = physical_slot(meta.block_table, max_blocks, block_size,
                                   batch_index, past_pos);
    if (slot < 0 || slot >= cache_slots)
      continue;
    const int64_t row = static_cast<int64_t>(slot) * kv_width + kv_head * head_size;
    const __half *k = key_cache + row;
    const __half *v = value_cache + row;

    float partial = 0.f;
    for (int i = 0; i < kMaxDimsPerLane; ++i) {
      const int d = lane + i * warpSize;
      if (d < head_size)
        partial += q_reg[i] * __half2float(k[d]);
    }
    const float score = wave_sum(partial) * scale;

    const float new_max = fmaxf(running_max, score);
    const float correction =
        running_max == -INFINITY ? 0.f : expf(running_max - new_max);
    const float weight = expf(score - new_max);
    running_denom = running_denom * correction + weight;
    for (int i = 0; i < kMaxDimsPerLane; ++i) {
      const int d = lane + i * warpSize;
      if (d < head_size)
        acc[i] = acc[i] * correction + weight * __half2float(v[d]);
    }
    running_max = new_max;
  }

  if (lane == 0) {
    wave_max[wave] = running_max;
    wave_denom[wave] = running_denom;
  }
  for (int i = 0; i < kMaxDimsPerLane; ++i) {
    const int d = lane + i * warpSize;
    if (d < head_size)
      wave_acc[wave * head_size + d] = acc[i];
  }
  __syncthreads();

  float global_max = -INFINITY;
  for (int w = 0; w < num_waves; ++w)
    global_max = fmaxf(global_max, wave_max[w]);
  float denom = 0.f;
  if (global_max != -INFINITY) {
    for (int w = 0; w < num_waves; ++w) {
      if (wave_max[w] != -INFINITY)
        denom += wave_denom[w] * expf(wave_max[w] - global_max);
    }
  }
  float *partial = nullptr;
  if (partials) {
    partial = partials +
              ((static_cast<int64_t>(token) * num_heads + head) * splits +
               blockIdx.z) * partial_stride(head_size);
    if (tid == 0) {
      partial[0] = global_max;
      partial[1] = denom;
    }
  }
  for (int d = tid; d < head_size; d += blockDim.x) {
    float sum = 0.f;
    if (denom > 0.f) {
      for (int w = 0; w < num_waves; ++w) {
        if (wave_max[w] != -INFINITY)
          sum += wave_acc[w * head_size + d] * expf(wave_max[w] - global_max);
      }
      if (!partials)
        sum /= denom;
    }
    if (partials)
      partial[2 + d] = sum;
    else
      out[d] = __float2half(sum);
  }
}

// Merges the split partials of one (token, head) into the output row. Splits
// past the end of the context were never written, so only the first
// ceil((position + 1) / chunk) are read.
__global__ void __launch_bounds__(kAttentionThreads)
    paged_attention_reduce(const float *partials, __half *output,
                           DeviceMeta meta, int batch, int num_heads,
                           int head_size, int chunk, int splits) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  __half *out =
      output + (static_cast<int64_t>(token) * num_heads + head) * head_size;
  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0) {
    for (int d = threadIdx.x; d < head_size; d += blockDim.x)
      out[d] = __float2half(0.f);
    return;
  }
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  const int used = min(splits, position / chunk + 1);
  const int64_t stride = partial_stride(head_size);
  const float *base =
      partials + (static_cast<int64_t>(token) * num_heads + head) * splits *
                     stride;
  float global_max = -INFINITY;
  for (int s = 0; s < used; ++s)
    global_max = fmaxf(global_max, base[s * stride]);
  float denom = 0.f;
  if (global_max != -INFINITY) {
    for (int s = 0; s < used; ++s) {
      const float *p = base + s * stride;
      if (p[0] != -INFINITY)
        denom += p[1] * expf(p[0] - global_max);
    }
  }
  for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
    float sum = 0.f;
    if (denom > 0.f) {
      for (int s = 0; s < used; ++s) {
        const float *p = base + s * stride;
        if (p[0] != -INFINITY)
          sum += p[2 + d] * expf(p[0] - global_max);
      }
      sum /= denom;
    }
    out[d] = __float2half(sum);
  }
}

// ---------------------------------------------------------------------------
// Tiled WMMA prefill for head_size 256 (FlashAttention-2 style).
//
// A block owns 16 consecutive query rows of one sequence and HB query heads of
// one KV group. ND waves per head split the head dimension, sum their partial
// scores through shared memory and each keep the online softmax state of the
// same 16 queries. Every K/V tile is staged once in shared memory for the
// whole block, so history is read once per 16 * HB query rows instead of once
// per (query, head). Scores are computed transposed (S^T = K Q^T) so that a
// lane owns one query and P^T feeds the value GEMM straight from registers.
// A K/V tile never crosses a cache block (block_size is a multiple of BKV), so
// each tile is a run of rows at a fixed stride of kv_heads * head_size.
// ---------------------------------------------------------------------------
typedef _Float16 half16_t __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef _Float16 half16_raw_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));

constexpr int kFlashHeadSize = 256;
constexpr int kFlashQueryRows = 16;

#define PA_HALF16_LOAD(ptr, pair)                                              \
  (*reinterpret_cast<const half16_t *>((ptr) + hipdnn_wmma_k_off(pair)))
#define PA_HALF16_RAW_LOAD(ptr) (*reinterpret_cast<const half16_raw_t *>(ptr))

// Exchanges a value with lane ^ 16 on the VALU. The inline-asm copy keeps the
// compiler from dual-issuing the move with an unrelated one, which can read a
// stale source on gfx11.
__device__ __forceinline__ int swap_halves(int x) {
#if HIPDNN_HAS_WMMA && HIPDNN_WMMA_FRAG_ELEMS != 8
  int old;
  asm("v_mov_b32 %0, %1" : "=v"(old) : "v"(x));
  return __builtin_amdgcn_permlanex16(old, x, 0x76543210u, 0xfedcba98u, false,
                                      false);
#elif HIPDNN_HAS_WMMA
  return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false,
                                      false);
#else
  return __shfl_xor(x, 16);
#endif
}
__device__ __forceinline__ float swap_halves(float x) {
  return __builtin_bit_cast(float, swap_halves(__builtin_bit_cast(int, x)));
}
__device__ __forceinline__ bool wave_any(bool p) {
#if HIPDNN_HAS_WMMA
  return __builtin_amdgcn_ballot_w32(p) != 0;
#else
  return __any(p);
#endif
}

// With 16-half fragments the K and V tiles are prefetched one after the other
// to stay inside the register budget; with 8-half fragments both fit.
#if HIPDNN_WMMA_FRAG_ELEMS == 8
constexpr bool kSplitKv = false;
constexpr int kFragRing = 3;
#else
constexpr bool kSplitKv = true;
constexpr int kFragRing = 2;
#endif

// Feeds NF operand fragments to use(f, frag) in order with at most RING reads
// issued ahead of the WMMA that consumes them.
template <int NF, typename Frag, int RING, typename Load, typename Use>
__device__ __forceinline__ void frag_stream(Load &&load, Use &&use) {
  constexpr int R = RING < NF ? RING : NF;
  typedef int dw_t __attribute__((ext_vector_type(sizeof(Frag) / 4)));
  dw_t ring[R];
#pragma unroll
  for (int f = 0; f < R; ++f)
    ring[f] = __builtin_bit_cast(dw_t, load(f));
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    dw_t x = ring[f % R];
    if (f + R < NF)
      ring[f % R] = __builtin_bit_cast(dw_t, load(f + R));
    asm("" : "+v"(x));
    use(f, __builtin_bit_cast(Frag, x));
#if HIPDNN_HAS_WMMA
    __builtin_amdgcn_sched_barrier(0);
#endif
  }
}

// Stages a BKV-key K tile (row-major in LDS) and V tile (transposed in LDS)
// with a row stride in global memory. Each tile is BKV * D / 16 chunks of 16
// halves spread over the NT threads of the block; K chunk i is column i % 16
// of row i / 16, V chunk j is key j % BKV of head-dim chunk j / BKV.
template <int BKV, int D, int NT, int VT_STR, int K_STR>
struct PagedKvStage {
  static constexpr int D_TILES = D / 16;
  static constexpr int kChunks = BKV * D_TILES;
  static constexpr int kPerThr = (kChunks + NT - 1) / NT;
  static constexpr int kKRowsPerStep = NT / D_TILES;
  static_assert(kChunks % 32 == 0 && NT % 32 == 0 && NT % BKV == 0 &&
                    NT % D_TILES == 0,
                "chunk ownership must be wave-uniform");
  static_assert(NT < 2 * kChunks, "every thread stages both K and V");

  int tid;
  int wave_lo;
  int k_row0;
  int k_col;
  int v_key;
  int v_col;
  int vt0, vt1;

  __device__ PagedKvStage(int tid_, int wave_id)
      : tid(tid_), wave_lo(32 * wave_id) {
    k_row0 = tid / D_TILES;
    k_col = (tid % D_TILES) * 16;
    v_key = tid % BKV;
    v_col = (tid / BKV) * 16;
    const int row0 = (tid / BKV) * 16 * VT_STR;
#if HIPDNN_WMMA_FRAG_ELEMS != 8
    // On the 16-half encoding a 16-key block of a V^T row stores the 8 keys of
    // parity (row & 1) first, which is the K order the P^T operand uses.
    const int kb = (v_key & ~15) | ((v_key & 15) >> 1);
    vt0 = row0 + kb + ((v_key & 1) << 3);
    vt1 = row0 + kb + (((v_key & 1) ^ 1) << 3);
#else
    vt0 = vt1 = row0 + v_key;
#endif
  }

  __device__ bool has(int r) const {
    if ((r + 1) * NT <= kChunks)
      return true;
    return r * NT + wave_lo < kChunks;
  }

  // A tail tile (one reaching past the sequence end) reads its rows past the
  // end from the last valid key: K is masked afterwards, V is zeroed.
  template <bool TAIL>
  __device__ half16_raw_t loadK(const _Float16 *tile, int stride, int r,
                                int last_row) const {
    if (!has(r))
      return half16_raw_t{};
    int row = k_row0 + r * kKRowsPerStep;
    if constexpr (TAIL)
      row = min(row, last_row);
    return PA_HALF16_RAW_LOAD(tile + row * stride + k_col);
  }
  template <bool TAIL>
  __device__ half16_raw_t loadV(const _Float16 *tile, int stride, int r,
                                int last_row) const {
    if (!has(r))
      return half16_raw_t{};
    const int key = TAIL ? min(v_key, last_row) : v_key;
    const half16_raw_t v = PA_HALF16_RAW_LOAD(
        tile + key * stride + r * (NT / BKV) * 16 + v_col);
    if constexpr (TAIL)
      return v_key <= last_row ? v : half16_raw_t{};
    return v;
  }

  __device__ void storeK(_Float16 *K_lds, int r, const half16_raw_t &v) const {
    if (!has(r))
      return;
    const int off = tid * 16 + (tid / D_TILES) * 8 + r * NT * 16 +
                    r * (NT / D_TILES) * 8;
    *reinterpret_cast<half16_raw_t *>(&K_lds[off]) = v;
  }
  __device__ void storeV(_Float16 *V_lds, int r, const half16_raw_t &v) const {
    if (!has(r))
      return;
    const int o = r * (NT / BKV) * 16 * VT_STR;
#pragma unroll
    for (int c = 0; c < 16; ++c)
      V_lds[((c & 1) ? vt1 : vt0) + o + c * VT_STR] = v[c];
  }
};

constexpr size_t flash_smem_bytes(int bkv, int nd, int hb) {
  return static_cast<size_t>(kFlashHeadSize) * (bkv + 8) * sizeof(_Float16) +
         static_cast<size_t>(bkv) * (kFlashHeadSize + 8) * sizeof(_Float16) +
         (nd > 1 ? static_cast<size_t>(hb) * nd * bkv * 16 * sizeof(float) : 0);
}

template <int BKV, int ND, int HB>
__global__ void __launch_bounds__(32 * ND * HB)
    paged_flash_prefill(const _Float16 *__restrict__ Q,
                        const _Float16 *__restrict__ key_cache,
                        const _Float16 *__restrict__ value_cache,
                        _Float16 *__restrict__ O, DeviceMeta meta, int batch,
                        int Hq, int G, int max_blocks, int block_size,
                        int cache_slots, int nqb, float scale,
                        float *__restrict__ partials, int split_tiles) {
#if HIPDNN_HAS_WMMA
  constexpr int D = kFlashHeadSize;
  constexpr int NW = ND * HB;
  constexpr int kWmmaTile = 16;
  constexpr int S_TILES_J = BKV / kWmmaTile;
  constexpr int D_TILES = D / kWmmaTile;
  constexpr int DT = D_TILES / ND;
  static_assert(DT * ND == D_TILES, "ND must divide the head dim tiles");
  constexpr int VT_STR = BKV + 8;
  constexpr int K_STR = D + 8;
  constexpr int kAcc = 8;
  constexpr float kLog2e = 1.4426950408889634f;

  // Query heads of one KV group are adjacent so they stream the same K/V
  // through L2 together; query tiles run last-first because under the causal
  // mask they walk the most keys.
  const int cpg = (Hq / G) / HB;
  // With `partials` set the grid is replicated once per slice of the key
  // range; the slice index is the slowest-varying part of the block id.
  const int base_grid = (Hq / HB) * nqb;
  const int split = static_cast<int>(blockIdx.x) / base_grid;
  const int bid = static_cast<int>(blockIdx.x) - split * base_grid;
  const int qb = (bid / cpg) % nqb;
  const int chunk = (bid / (cpg * nqb)) * cpg + bid % cpg;

  int total_tiles = 0;
  for (int b = 0; b < batch; ++b)
    total_tiles +=
        (meta.cumulative[b + 1] - meta.cumulative[b] + kFlashQueryRows - 1) /
        kFlashQueryRows;
  if (qb >= total_tiles)
    return;
  int tile = total_tiles - 1 - qb;
  int batch_index = 0;
  for (int b = 0; b < batch; ++b) {
    const int n = (meta.cumulative[b + 1] - meta.cumulative[b] +
                   kFlashQueryRows - 1) / kFlashQueryRows;
    if (tile < n) {
      batch_index = b;
      break;
    }
    tile -= n;
  }
  const int seq_begin = meta.cumulative[batch_index];
  const int sq = meta.cumulative[batch_index + 1] - seq_begin;
  const int past_len = meta.past[batch_index];
  const int skv = past_len + sq;

  const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int g = wave_id % ND;
  const int hb = wave_id / ND;
  const int head_q = __builtin_amdgcn_readfirstlane(chunk * HB + hb);
  const int head_kv = __builtin_amdgcn_readfirstlane(chunk / cpg);
  const int q_start = tile * kFlashQueryRows;
  const int dt0 = g * DT;
  const int kv_width = G * D;

  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int wmma_lane = lane_id % kWmmaTile;
  const int pair = lane_id / kWmmaTile;

  extern __shared__ char smem_flash[];
  _Float16 *V_lds = reinterpret_cast<_Float16 *>(smem_flash);
  _Float16 *K_lds = V_lds + D * VT_STR;
  float *Sp = reinterpret_cast<float *>(K_lds + BKV * K_STR);

  half16_t Q_reg[DT];
  {
    const int q_pos = q_start + wmma_lane;
    const bool valid = q_pos < sq;
    const size_t q_row = (static_cast<size_t>(seq_begin + (valid ? q_pos : 0)) *
                              Hq + head_q) * D;
#pragma unroll
    for (int tk = 0; tk < DT; ++tk)
      Q_reg[tk] = valid ? PA_HALF16_LOAD(&Q[q_row + (dt0 + tk) * 16], pair)
                        : half16_t{};
  }

  float8_t O_frag[DT] = {};
  float m_reg = -INFINITY, l_reg = 0.0f;
  const int abs_q = past_len + q_start + wmma_lane;

  const int kv_max = (skv - 1 < past_len + q_start + kFlashQueryRows - 1)
                         ? (skv - 1)
                         : (past_len + q_start + kFlashQueryRows - 1);
  const int num_tiles = kv_max / BKV + 1;
  const int t_begin = split * split_tiles;
  if (t_begin >= num_tiles)
    return;
  const int t_end = min(num_tiles, t_begin + split_tiles);

  using Stage = PagedKvStage<BKV, D, 32 * NW, VT_STR, K_STR>;
  constexpr int kPerThr = Stage::kPerThr;
  constexpr int kPf = kPerThr > 4 ? 4 : kPerThr;
  const Stage st(tid, wave_id);
  typedef half16_raw_t Chunks[kPf];
  auto tile_ptr = [&](const _Float16 *cache, int kv0) {
    int slot = physical_slot(meta.block_table, max_blocks, block_size,
                             batch_index, kv0);
    if (slot < 0 || slot + BKV > cache_slots)
      slot = 0;
    return cache + static_cast<size_t>(slot) * kv_width + head_kv * D;
  };
  auto load_k = [&](Chunks &dst, int kv0) {
    const _Float16 *t = tile_ptr(key_cache, kv0);
    if (kv0 + BKV <= skv) {
#pragma unroll
      for (int r = 0; r < kPf; ++r)
        dst[r] = st.template loadK<false>(t, kv_width, r, 0);
    } else {
#pragma unroll
      for (int r = 0; r < kPf; ++r)
        dst[r] = st.template loadK<true>(t, kv_width, r, skv - kv0 - 1);
    }
  };
  auto load_v = [&](Chunks &dst, int kv0) {
    const _Float16 *t = tile_ptr(value_cache, kv0);
    if (kv0 + BKV <= skv) {
#pragma unroll
      for (int r = 0; r < kPf; ++r)
        dst[r] = st.template loadV<false>(t, kv_width, r, 0);
    } else {
#pragma unroll
      for (int r = 0; r < kPf; ++r)
        dst[r] = st.template loadV<true>(t, kv_width, r, skv - kv0 - 1);
    }
  };
  // At most kPf chunks per thread are prefetched; the rest load at commit.
  auto commit_k = [&](const Chunks &src, int kv0) {
#pragma unroll
    for (int r = 0; r < kPf; ++r)
      st.storeK(K_lds, r, src[r]);
    if constexpr (kPerThr > kPf) {
      const bool tail = kv0 + BKV > skv;
      const _Float16 *t = tile_ptr(key_cache, kv0);
#pragma unroll
      for (int r = kPf; r < kPerThr; ++r)
        st.storeK(K_lds, r,
                  tail ? st.template loadK<true>(t, kv_width, r, skv - kv0 - 1)
                       : st.template loadK<false>(t, kv_width, r, 0));
    }
  };
  auto commit_v = [&](const Chunks &src, int kv0) {
#pragma unroll
    for (int r = 0; r < kPf; ++r)
      st.storeV(V_lds, r, src[r]);
    if constexpr (kPerThr > kPf) {
      const bool tail = kv0 + BKV > skv;
      const _Float16 *t = tile_ptr(value_cache, kv0);
#pragma unroll
      for (int r = kPf; r < kPerThr; ++r)
        st.storeV(V_lds, r,
                  tail ? st.template loadV<true>(t, kv_width, r, skv - kv0 - 1)
                       : st.template loadV<false>(t, kv_width, r, 0));
    }
  };
  const _Float16 *K_frag_base = &K_lds[wmma_lane * K_STR + dt0 * 16];
  const _Float16 *V_frag_base = &V_lds[(dt0 * 16 + wmma_lane) * VT_STR];
  Chunks pf, pv;
  const int kv_begin = t_begin * BKV;
  load_k(pf, kv_begin);
  load_v(pv, kv_begin);
  commit_k(pf, kv_begin);
  commit_v(pv, kv_begin);
  __syncthreads();
  if (t_end - t_begin > 1) {
    load_k(pf, kv_begin + BKV);
    if constexpr (!kSplitKv)
      load_v(pv, kv_begin + BKV);
  }

  for (int t = t_begin; t < t_end; ++t) {
    const int kv0 = t * BKV;

    float8_t S_reg[S_TILES_J];
#pragma unroll
    for (int sj = 0; sj < S_TILES_J; ++sj)
      S_reg[sj] = float8_t{};
    frag_stream<S_TILES_J * DT, half16_t, kFragRing>(
        [&](int f) {
          return PA_HALF16_LOAD(
              &K_frag_base[(f % S_TILES_J) * 16 * K_STR + (f / S_TILES_J) * 16],
              pair);
        },
        [&](int f, const half16_t &k_frag) {
          S_reg[f % S_TILES_J] = HIPDNN_WMMA_F32_16X16X16_F16(
              k_frag, Q_reg[f / S_TILES_J], S_reg[f % S_TILES_J]);
        });

    // The barrier that publishes the partial scores also means every wave is
    // past this tile's K, so the next K goes in and the next V is fetched
    // while this tile finishes.
    if constexpr (ND > 1) {
#pragma unroll
      for (int sj = 0; sj < S_TILES_J; ++sj)
        *reinterpret_cast<float8_t *>(
            &Sp[(((hb * ND + g) * S_TILES_J + sj) * 32 + lane_id) * kAcc]) =
            S_reg[sj];
    }
    __syncthreads();
    if constexpr (kSplitKv) {
      if (t + 1 < t_end) {
        commit_k(pf, kv0 + BKV);
        load_v(pf, kv0 + BKV);
      }
    }
    if constexpr (ND > 1) {
#pragma unroll
      for (int g2 = 0; g2 < ND; ++g2) {
        if (g2 == g)
          continue;
#pragma unroll
        for (int sj = 0; sj < S_TILES_J; ++sj) {
          const float8_t o = *reinterpret_cast<const float8_t *>(
              &Sp[(((hb * ND + g2) * S_TILES_J + sj) * 32 + lane_id) * kAcc]);
          S_reg[sj] += o;
        }
      }
    }

    // Only tiles crossing the causal diagonal or the sequence end need a mask.
    const bool need_mask =
        (kv0 + BKV > skv) || (kv0 + BKV - 1 > past_len + q_start);
    if (need_mask) {
#pragma unroll
      for (int sj = 0; sj < S_TILES_J; ++sj)
#pragma unroll
        for (int e = 0; e < kAcc; ++e) {
          const int kv_pos = kv0 + sj * 16 + hipdnn_wmma_acc_row(pair, e);
          if (kv_pos >= skv || kv_pos > abs_q)
            S_reg[sj][e] = -INFINITY;
        }
    }

    // Scores are unscaled; m_reg is kept in scaled log2 units.
    const float c = scale * kLog2e;
    half16_t Pb[S_TILES_J];
    float rmax = -INFINITY;
#pragma unroll
    for (int sj = 0; sj < S_TILES_J; ++sj)
#pragma unroll
      for (int e = 0; e < kAcc; ++e)
        rmax = fmaxf(rmax, S_reg[sj][e]);
    rmax = fmaxf(rmax, swap_halves(rmax));
    const float m_new = fmaxf(m_reg, rmax * c);
    // A slice that starts past a row's last visible key sees only masked
    // scores for that row; keep the subtraction finite so it contributes zero.
    const float m_sub = (m_new == -INFINITY) ? 0.0f : m_new;
    const float corr = __builtin_amdgcn_exp2f(m_reg - m_sub);
    float rsum = 0.0f;
#pragma unroll
    for (int sj = 0; sj < S_TILES_J; ++sj) {
      _Float16 own[kAcc];
#pragma unroll
      for (int e = 0; e < kAcc; ++e) {
        const float p = __builtin_amdgcn_exp2f(fmaf(S_reg[sj][e], c, -m_sub));
        rsum += p;
        own[e] = static_cast<_Float16>(p);
      }
#if HIPDNN_WMMA_GFX12
#pragma unroll
      for (int e = 0; e < kAcc; ++e)
        Pb[sj][e] = own[e];
#else
      // A lane holds keys 2e + pair; the B operand needs all 16 keys of the
      // block, so the other parity comes from lane ^ 16.
      typedef _Float16 half2v_t __attribute__((ext_vector_type(2)));
      typedef int int8v_t __attribute__((ext_vector_type(8)));
      int8v_t pk;
#pragma unroll
      for (int j = 0; j < kAcc / 2; ++j)
        pk[j] = __builtin_bit_cast(int, half2v_t{own[2 * j], own[2 * j + 1]});
#pragma unroll
      for (int j = 0; j < kAcc / 2; ++j)
        pk[kAcc / 2 + j] = swap_halves(pk[j]);
      Pb[sj] = __builtin_bit_cast(half16_t, pk);
#endif
    }
    rsum += swap_halves(rsum);
    m_reg = m_new;
    l_reg = corr * l_reg + rsum;

    if (wave_any(corr != 1.0f)) {
#pragma unroll
      for (int tj = 0; tj < DT; ++tj)
#pragma unroll
        for (int e = 0; e < kAcc; ++e)
          O_frag[tj][e] *= corr;
    }

    frag_stream<DT * S_TILES_J, half16_t, kFragRing>(
        [&](int f) {
          return PA_HALF16_LOAD(
              &V_frag_base[(f % DT) * 16 * VT_STR + (f / DT) * 16], pair);
        },
        [&](int f, const half16_t &v_frag) {
          O_frag[f % DT] = HIPDNN_WMMA_F32_16X16X16_F16(v_frag, Pb[f / DT],
                                                        O_frag[f % DT]);
        });
    __syncthreads();  // V_lds and Sp are reused by the next tile
    if constexpr (kSplitKv) {
      if (t + 1 < t_end) {
        commit_v(pf, kv0 + BKV);
        if (t + 2 < t_end)
          load_k(pf, kv0 + 2 * BKV);
      }
    } else {
      if (t + 1 < t_end) {
        commit_k(pf, kv0 + BKV);
        commit_v(pv, kv0 + BKV);
        __syncthreads();
        if (t + 2 < t_end) {
          load_k(pf, kv0 + 2 * BKV);
          load_v(pv, kv0 + 2 * BKV);
        }
      }
    }
  }

  const int q_pos = q_start + wmma_lane;
  if (partials) {
    if (q_pos < sq) {
      const size_t splits = gridDim.x / base_grid;
      float *part =
          partials +
          ((static_cast<size_t>(seq_begin + q_pos) * Hq + head_q) * splits +
           split) *
              partial_stride(D);
#pragma unroll
      for (int tj = 0; tj < DT; ++tj)
#pragma unroll
        for (int e = 0; e < kAcc; ++e)
          part[2 + (dt0 + tj) * 16 + hipdnn_wmma_acc_row(pair, e)] =
              O_frag[tj][e];
      if (g == 0 && pair == 0) {
        part[0] = m_reg;
        part[1] = l_reg;
      }
    }
    return;
  }
  const float inv_l = 1.0f / fmaxf(l_reg, 1e-6f);
  if (q_pos < sq) {
    _Float16 *o_row =
        &O[(static_cast<size_t>(seq_begin + q_pos) * Hq + head_q) * D];
#pragma unroll
    for (int tj = 0; tj < DT; ++tj)
#pragma unroll
      for (int e = 0; e < kAcc; ++e)
        o_row[(dt0 + tj) * 16 + hipdnn_wmma_acc_row(pair, e)] =
            static_cast<_Float16>(O_frag[tj][e] * inv_l);
  }
#else
  __builtin_trap();
#endif
}

// Merges the key-range slices written by paged_flash_prefill. Partial maxima
// are in scaled log2 units. A slice is only read when it can hold a visible
// key of the row.
__global__ void __launch_bounds__(kAttentionThreads)
    paged_flash_reduce(const float *partials, __half *output, DeviceMeta meta,
                       int batch, int num_heads, int splits, int split_keys) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  __half *out = output + (static_cast<int64_t>(token) * num_heads + head) *
                             kFlashHeadSize;
  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0) {
    for (int d = threadIdx.x; d < kFlashHeadSize; d += blockDim.x)
      out[d] = __float2half(0.f);
    return;
  }
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  const int used = min(splits, position / split_keys + 1);
  const int64_t stride = partial_stride(kFlashHeadSize);
  const float *base =
      partials + (static_cast<int64_t>(token) * num_heads + head) * splits *
                     stride;
  float global_max = -INFINITY;
  for (int s = 0; s < used; ++s)
    global_max = fmaxf(global_max, base[s * stride]);
  float denom = 0.f;
  if (global_max != -INFINITY) {
    for (int s = 0; s < used; ++s) {
      const float *p = base + s * stride;
      if (p[0] != -INFINITY)
        denom += p[1] * __builtin_amdgcn_exp2f(p[0] - global_max);
    }
  }
  for (int d = threadIdx.x; d < kFlashHeadSize; d += blockDim.x) {
    float sum = 0.f;
    if (denom > 0.f) {
      for (int s = 0; s < used; ++s) {
        const float *p = base + s * stride;
        if (p[0] != -INFINITY)
          sum += p[2 + d] * __builtin_amdgcn_exp2f(p[0] - global_max);
      }
      sum /= denom;
    }
    out[d] = __float2half(sum);
  }
}

struct FlashConfig {
  int nd;
  int hb;
  int bkv;
};

// The history a block walks is cut into slices of at most this many keys so
// that the K/V working set live at once stays within the last-level cache; the
// slices are merged by paged_flash_reduce. The slice count follows the cache
// capacity, so a history that fits one slice takes the unsplit path.
constexpr int kFlashSplitKeys = 8192;
constexpr size_t kMaxFlashPartialBytes = 512u << 20;
constexpr int kNoSplitTiles = 1 << 28;

struct FlashSplitPlan {
  int splits;
  int split_tiles;
  size_t partial_bytes;
};

FlashSplitPlan plan_flash_splits(int64_t num_tokens, int num_heads, int bkv,
                                 int block_size, int max_blocks) {
  int keys = kFlashSplitKeys;
  const int64_t capacity = static_cast<int64_t>(max_blocks) * block_size;
  const size_t per_split = static_cast<size_t>(num_tokens) * num_heads *
                           partial_stride(kFlashHeadSize) * sizeof(float);
  int64_t splits = (capacity + keys - 1) / keys;
  const int64_t budget =
      static_cast<int64_t>(kMaxFlashPartialBytes / per_split);
  if (splits > budget && budget >= 1) {
    splits = budget;
    keys = static_cast<int>((capacity + splits - 1) / splits);
  }
  keys = (keys + bkv - 1) / bkv * bkv;
  splits = (capacity + keys - 1) / keys;
  if (splits <= 1 || splits > budget || keys <= 0)
    return {1, kNoSplitTiles, 0};
  return {static_cast<int>(splits), keys / bkv, per_split * splits};
}

// Picks the head grouping and the number of waves per head. Three heads with
// four waves per head keeps the accumulators and query fragments of each wave
// small enough to stay in registers; wider groupings spill. Returns false when
// the shape has no tiled path (the caller falls back to the per-token kernel).
bool pick_flash_config(int num_heads, int kv_heads, int head_size,
                       int block_size, FlashConfig *cfg) {
  if (head_size != kFlashHeadSize)
    return false;
  const int group = num_heads / kv_heads;
  const int bkv = 32;
  if (block_size % bkv != 0)
    return false;
  FlashConfig c{2, 1, bkv};
  if (group % 3 == 0)
    c = FlashConfig{4, 3, bkv};
  else if (group % 2 == 0)
    c = FlashConfig{2, 2, bkv};
  *cfg = c;
  return true;
}

// The tiled kernels use WMMA, which gfx11 and gfx12 provide.
bool device_has_wmma() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return false;
  static std::mutex mutex;
  static std::unordered_map<int, bool> cache;
  std::lock_guard<std::mutex> lock(mutex);
  const auto it = cache.find(device);
  if (it != cache.end())
    return it->second;
  hipDeviceProp_t prop;
  bool ok = false;
  if (hipGetDeviceProperties(&prop, device) == hipSuccess)
    ok = std::strncmp(prop.gcnArchName, "gfx11", 5) == 0 ||
         std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
  cache[device] = ok;
  return ok;
}

template <int BKV, int ND, int HB>
hipError_t launch_flash(hipStream_t stream, const FlashConfig &cfg,
                        const void *query, const void *key_cache,
                        const void *value_cache, void *output,
                        const DeviceMeta &meta, int64_t num_tokens, int batch,
                        int num_heads, int kv_heads, int block_size,
                        int max_blocks, int cache_slots, float scale,
                        float *partials_scratch,
                        size_t partials_scratch_bytes) {
  const int nqb = static_cast<int>(
      (num_tokens + kFlashQueryRows - 1) / kFlashQueryRows + batch);
  FlashSplitPlan plan =
      plan_flash_splits(num_tokens, num_heads, BKV, block_size, max_blocks);
  float *partials = nullptr;
  if (plan.splits > 1 && plan.partial_bytes <= partials_scratch_bytes)
    partials = partials_scratch;
  else
    plan = {1, kNoSplitTiles, 0};
  const int64_t grid =
      static_cast<int64_t>(num_heads / HB) * nqb * plan.splits;
  if (grid > 2147483647)
    return hipErrorInvalidValue;
  paged_flash_prefill<BKV, ND, HB>
      <<<dim3(static_cast<unsigned>(grid)), dim3(32 * ND * HB),
         flash_smem_bytes(BKV, ND, HB), stream>>>(
          static_cast<const _Float16 *>(query),
          static_cast<const _Float16 *>(key_cache),
          static_cast<const _Float16 *>(value_cache),
          static_cast<_Float16 *>(output), meta, batch, num_heads, kv_heads,
          max_blocks, block_size, cache_slots, nqb, scale, partials,
          plan.split_tiles);
  hipError_t err = hipGetLastError();
  if (partials && err == hipSuccess) {
    paged_flash_reduce<<<dim3(static_cast<unsigned>(num_tokens),
                              static_cast<unsigned>(num_heads)),
                         dim3(kAttentionThreads), 0, stream>>>(
        partials, static_cast<__half *>(output), meta, batch, num_heads,
        plan.splits, plan.split_tiles * BKV);
    err = hipGetLastError();
  }
  return err;
}

hipError_t dispatch_flash(hipStream_t stream, const FlashConfig &cfg,
                          const void *query, const void *key_cache,
                          const void *value_cache, void *output,
                          const DeviceMeta &meta, int64_t num_tokens,
                          int batch, int num_heads, int kv_heads,
                          int block_size, int max_blocks, int cache_slots,
                          float scale, float *partials_scratch,
                          size_t partials_scratch_bytes) {
#define PA_LAUNCH(ND_, HB_)                                                    \
  launch_flash<32, ND_, HB_>(stream, cfg, query, key_cache, value_cache,      \
                             output, meta, num_tokens, batch, num_heads,       \
                             kv_heads, block_size, max_blocks, cache_slots,    \
                             scale, partials_scratch, partials_scratch_bytes)
  if (cfg.nd == 4 && cfg.hb == 3)
    return PA_LAUNCH(4, 3);
  if (cfg.hb == 2)
    return PA_LAUNCH(2, 2);
  return PA_LAUNCH(2, 1);
#undef PA_LAUNCH
}

// Decode-sized attention for head_size 256 on wave32 devices. A block owns HB
// query heads of one KV group, so each K/V row is read once for all of them.
// 16 lanes cover one position (16 contiguous dims each, two 16-byte loads per
// row), which makes a wave work on two positions at once and keeps the score
// reduction to four shuffle steps. Every 16-lane group runs its own online
// softmax over the positions it visits and the groups are merged through
// shared memory. As in paged_attention_fp16, `partials` selects a slice of
// the context by blockIdx.z and defers the normalization to the reduce kernel.
constexpr int kGroupedLanes = 16;
constexpr int kGroupedDims = kFlashHeadSize / kGroupedLanes;
constexpr int kGroupedThreads = 256;
constexpr int kGroupedGroups = kGroupedThreads / kGroupedLanes;
constexpr int kMaxGroupedHeads = 3;

constexpr size_t grouped_smem_bytes(int hb) {
  return (static_cast<size_t>(hb) * kGroupedGroups * kFlashHeadSize +
          2 * static_cast<size_t>(hb) * kGroupedGroups) *
         sizeof(float);
}

__device__ __forceinline__ void load_dims(const __half *src,
                                          float (&out)[kGroupedDims]) {
  const uint4 lo = reinterpret_cast<const uint4 *>(src)[0];
  const uint4 hi = reinterpret_cast<const uint4 *>(src)[1];
  const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 f = __half22float2(__builtin_bit_cast(__half2, words[i]));
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

template <int HB>
__global__ void __launch_bounds__(kGroupedThreads)
    paged_decode_grouped(const __half *query, const __half *key_cache,
                         const __half *value_cache, __half *output,
                         float *partials, DeviceMeta meta, int batch,
                         int num_heads, int kv_heads, int block_size,
                         int max_blocks, int cache_slots, float scale,
                         int chunk, int splits) {
  constexpr int D = kFlashHeadSize;
  extern __shared__ float grouped_smem[];
  float *s_acc = grouped_smem;
  float *s_max = s_acc + HB * kGroupedGroups * D;
  float *s_denom = s_max + HB * kGroupedGroups;

  const int token = blockIdx.x;
  const int heads_per_kv = num_heads / kv_heads;
  const int blocks_per_kv = heads_per_kv / HB;
  const int kv_head = blockIdx.y / blocks_per_kv;
  const int head0 = kv_head * heads_per_kv + (blockIdx.y % blocks_per_kv) * HB;
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int sub = lane / kGroupedLanes;
  const int dim0 = (lane % kGroupedLanes) * kGroupedDims;
  const int group = (tid / 32) * 2 + sub;

  const int batch_index = find_batch(meta.cumulative, batch, token);
  if (batch_index < 0) {
    if (!partials) {
#pragma unroll
      for (int j = 0; j < HB; ++j)
        output[(static_cast<int64_t>(token) * num_heads + head0 + j) * D + tid] =
            __float2half(0.f);
    }
    return;
  }
  const int position =
      meta.past[batch_index] + (token - meta.cumulative[batch_index]);
  int first = 0;
  int last = position;
  if (partials) {
    first = static_cast<int>(blockIdx.z) * chunk;
    if (first > position)
      return;
    last = min(position, first + chunk - 1);
  }

  const int kv_width = kv_heads * D;
  float q[HB][kGroupedDims];
  float acc[HB][kGroupedDims];
  float running_max[HB];
  float running_denom[HB];
#pragma unroll
  for (int j = 0; j < HB; ++j) {
    load_dims(query + (static_cast<int64_t>(token) * num_heads + head0 + j) * D +
                  dim0,
              q[j]);
#pragma unroll
    for (int i = 0; i < kGroupedDims; ++i) {
      q[j][i] *= scale;
      acc[j][i] = 0.f;
    }
    running_max[j] = -INFINITY;
    running_denom[j] = 0.f;
  }

  for (int pos = first + group; pos <= last; pos += kGroupedGroups) {
    const int slot = physical_slot(meta.block_table, max_blocks, block_size,
                                   batch_index, pos);
    if (slot < 0 || slot >= cache_slots)
      continue;
    const int64_t row = static_cast<int64_t>(slot) * kv_width + kv_head * D + dim0;
    float k[kGroupedDims];
    float v[kGroupedDims];
    load_dims(key_cache + row, k);
    load_dims(value_cache + row, v);
#pragma unroll
    for (int j = 0; j < HB; ++j) {
      float score = 0.f;
#pragma unroll
      for (int i = 0; i < kGroupedDims; ++i)
        score += q[j][i] * k[i];
#pragma unroll
      for (int offset = kGroupedLanes / 2; offset > 0; offset /= 2)
        score += __shfl_xor(score, offset, kGroupedLanes);
      const float new_max = fmaxf(running_max[j], score);
      const float correction =
          running_max[j] == -INFINITY ? 0.f : expf(running_max[j] - new_max);
      const float weight = expf(score - new_max);
      running_denom[j] = running_denom[j] * correction + weight;
#pragma unroll
      for (int i = 0; i < kGroupedDims; ++i)
        acc[j][i] = acc[j][i] * correction + weight * v[i];
      running_max[j] = new_max;
    }
  }

#pragma unroll
  for (int j = 0; j < HB; ++j) {
    float *dst = s_acc + (j * kGroupedGroups + group) * D + dim0;
#pragma unroll
    for (int i = 0; i < kGroupedDims; ++i)
      dst[i] = acc[j][i];
    if (lane % kGroupedLanes == 0) {
      s_max[j * kGroupedGroups + group] = running_max[j];
      s_denom[j * kGroupedGroups + group] = running_denom[j];
    }
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < HB; ++j) {
    float global_max = -INFINITY;
    for (int g = 0; g < kGroupedGroups; ++g)
      global_max = fmaxf(global_max, s_max[j * kGroupedGroups + g]);
    float denom = 0.f;
    float sum = 0.f;
    if (global_max != -INFINITY) {
      for (int g = 0; g < kGroupedGroups; ++g) {
        const float m = s_max[j * kGroupedGroups + g];
        if (m == -INFINITY)
          continue;
        const float f = expf(m - global_max);
        denom += s_denom[j * kGroupedGroups + g] * f;
        sum += s_acc[(j * kGroupedGroups + g) * D + tid] * f;
      }
    }
    const int head = head0 + j;
    if (partials) {
      float *partial = partials +
                       ((static_cast<int64_t>(token) * num_heads + head) * splits +
                        blockIdx.z) * partial_stride(D);
      if (tid == 0) {
        partial[0] = global_max;
        partial[1] = denom;
      }
      partial[2 + tid] = sum;
    } else {
      output[(static_cast<int64_t>(token) * num_heads + head) * D + tid] =
          __float2half(denom > 0.f ? sum / denom : 0.f);
    }
  }
}

// A context is split across blocks in slices of at least this many positions.
constexpr int kMinSplitChunk = 512;
constexpr int kMaxSplits = 256;
constexpr size_t kMaxPartialBytes = 256u << 20;
constexpr size_t kScratchAlign = 256;

struct SplitPlan {
  int chunk;
  int splits;
  size_t partial_bytes;
};

SplitPlan plan_splits(int num_tokens, int num_heads, int head_size,
                      int block_size, int max_blocks) {
  const int capacity = max_blocks * block_size;
  int chunk = kMinSplitChunk;
  while ((capacity + chunk - 1) / chunk > kMaxSplits)
    chunk *= 2;
  const int splits = (capacity + chunk - 1) / chunk;
  const size_t partial_bytes = static_cast<size_t>(num_tokens) * num_heads *
                               splits * partial_stride(head_size) *
                               sizeof(float);
  return {chunk, splits, partial_bytes};
}

// Bytes of the int32 metadata staging area at the start of the caller scratch.
size_t meta_stage_bytes(int64_t batch, int64_t max_blocks) {
  const size_t count = static_cast<size_t>(batch + 1) +
                       static_cast<size_t>(batch) +
                       static_cast<size_t>(batch) *
                           static_cast<size_t>(max_blocks);
  const size_t bytes = count * sizeof(int32_t);
  return (bytes + kScratchAlign - 1) / kScratchAlign * kScratchAlign;
}

// Per-token attention. A decode-sized batch has only tokens * heads blocks,
// each walking the whole context serially, so a long context is cut into
// slices that run in parallel and are merged by paged_attention_reduce. The
// slice partials live in caller scratch; when it is too small the context is
// not split.
hipError_t launch_per_token(hipStream_t stream, const void *query,
                            void *key_cache, void *value_cache, void *output,
                            const DeviceMeta &meta, int num_tokens, int batch,
                            int num_heads, int kv_heads, int head_size,
                            int block_size, int max_blocks, int cache_slots,
                            float scale, float *partials_scratch,
                            size_t partials_scratch_bytes) {
  int wave_size = 32;
  (void)hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, 0);
  const int num_waves = kAttentionThreads / wave_size;
  const size_t shared_bytes =
      (2 * static_cast<size_t>(num_waves) +
       static_cast<size_t>(num_waves) * static_cast<size_t>(head_size)) *
      sizeof(float);
  const SplitPlan plan =
      plan_splits(num_tokens, num_heads, head_size, block_size, max_blocks);
  const int chunk = plan.chunk;
  const int splits = plan.splits;
  float *partials = nullptr;
  if (splits > 1 && plan.partial_bytes <= kMaxPartialBytes &&
      plan.partial_bytes <= partials_scratch_bytes)
    partials = partials_scratch;
  const unsigned z = partials ? static_cast<unsigned>(splits) : 1u;
  const int heads_per_kv = num_heads / kv_heads;
  int grouped_heads = 0;
  if (head_size == kFlashHeadSize && device_has_wmma()) {
    for (int hb = kMaxGroupedHeads; hb >= 1 && grouped_heads == 0; --hb)
      if (heads_per_kv % hb == 0)
        grouped_heads = hb;
  }
  if (grouped_heads > 0) {
    const dim3 grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(kv_heads * (heads_per_kv / grouped_heads)),
                    z);
#define PA_LAUNCH_GROUPED(HB_)                                                 \
  hipLaunchKernelGGL(                                                          \
      paged_decode_grouped<HB_>, grid, dim3(kGroupedThreads),                  \
      grouped_smem_bytes(HB_), stream, static_cast<const __half *>(query),     \
      static_cast<const __half *>(key_cache),                                  \
      static_cast<const __half *>(value_cache), static_cast<__half *>(output), \
      partials, meta, batch, num_heads, kv_heads, block_size, max_blocks,      \
      cache_slots, scale, chunk, splits)
    if (grouped_heads == 3)
      PA_LAUNCH_GROUPED(3);
    else if (grouped_heads == 2)
      PA_LAUNCH_GROUPED(2);
    else
      PA_LAUNCH_GROUPED(1);
#undef PA_LAUNCH_GROUPED
  } else {
    const dim3 grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(num_heads), z);
    hipLaunchKernelGGL(
        paged_attention_fp16, grid, dim3(kAttentionThreads), shared_bytes,
        stream, static_cast<const __half *>(query),
        static_cast<const __half *>(key_cache),
        static_cast<const __half *>(value_cache), static_cast<__half *>(output),
        partials, meta, num_tokens, batch, num_heads, kv_heads, head_size,
        block_size, max_blocks, cache_slots, scale, chunk, splits);
  }
  hipError_t err = hipGetLastError();
  if (partials) {
    if (err == hipSuccess) {
      hipLaunchKernelGGL(
          paged_attention_reduce,
          dim3(static_cast<unsigned>(num_tokens),
               static_cast<unsigned>(num_heads)),
          dim3(kAttentionThreads), 0, stream, partials,
          static_cast<__half *>(output), meta, batch, num_heads, head_size,
          chunk, splits);
      err = hipGetLastError();
    }
  }
  return err;
}

int copy_i32(hipStream_t stream, const void *src, int count, int32_t *dst) {
  const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
  return static_cast<int>(
      hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream));
}

// Fewer than two query tiles of tokens is decode-sized work, where the
// per-token kernel is the better fit.
bool select_flash(int64_t num_tokens, int64_t num_heads, int64_t kv_num_heads,
                  int64_t head_size, int64_t block_size, FlashConfig *cfg) {
  return num_tokens >= kFlashQueryRows * 2 &&
         pick_flash_config(static_cast<int>(num_heads),
                           static_cast<int>(kv_num_heads),
                           static_cast<int>(head_size),
                           static_cast<int>(block_size), cfg) &&
         device_has_wmma();
}

}  // namespace

// Scratch layout: int32 metadata staging first, then the key-range slice
// partials of whichever path (tiled or per-token) the shape selects.
extern "C" size_t hip_paged_attention_scratch_bytes(
    int64_t num_heads, int64_t kv_num_heads, int64_t num_tokens,
    int64_t head_size, int64_t batch, int64_t block_size, int64_t max_blocks) {
  const int64_t kIntMax = 2147483647;
  if (num_tokens <= 0 || num_heads <= 0 || kv_num_heads <= 0 ||
      head_size <= 0 || batch <= 0 || block_size < 16 || max_blocks <= 0 ||
      num_tokens > kIntMax || num_heads > kIntMax || head_size > kIntMax ||
      block_size > kIntMax || max_blocks > kIntMax ||
      batch > kIntMax / max_blocks || max_blocks > kIntMax / block_size)
    return 0;
  size_t bytes = meta_stage_bytes(batch, max_blocks);
  FlashConfig flash_cfg{};
  if (select_flash(num_tokens, num_heads, kv_num_heads, head_size, block_size,
                   &flash_cfg)) {
    bytes += plan_flash_splits(num_tokens, static_cast<int>(num_heads),
                               flash_cfg.bkv, static_cast<int>(block_size),
                               static_cast<int>(max_blocks))
                 .partial_bytes;
  } else {
    const SplitPlan plan =
        plan_splits(static_cast<int>(num_tokens), static_cast<int>(num_heads),
                    static_cast<int>(head_size), static_cast<int>(block_size),
                    static_cast<int>(max_blocks));
    if (plan.splits > 1 && plan.partial_bytes <= kMaxPartialBytes)
      bytes += plan.partial_bytes;
  }
  return bytes;
}

extern "C" int hip_paged_attention(
    void *stream_ptr, const void *query, const void *key, const void *value,
    void *key_cache, void *value_cache, const void *cumulative_sequence_length,
    const void *past_seqlens, const void *block_table, void *output,
    int64_t num_heads, int64_t kv_num_heads, int64_t num_tokens,
    int64_t head_size, int64_t batch, int64_t num_blocks, int64_t block_size,
    int64_t max_blocks, float scale, void *scratch, size_t scratch_bytes) {
  if (num_tokens == 0)
    return 0;
  if (!query || !key || !value || !key_cache || !value_cache || !output ||
      !cumulative_sequence_length || !past_seqlens || !block_table)
    return -1;
  if (num_heads <= 0 || kv_num_heads <= 0 || head_size <= 0 || batch <= 0 ||
      num_blocks <= 0 || max_blocks <= 0 || block_size < 16 ||
      (block_size & (block_size - 1)) != 0 || num_heads % kv_num_heads != 0)
    return -1;
  const int64_t kIntMax = 2147483647;
  if (num_tokens > kIntMax || num_heads > kIntMax || kv_num_heads > kIntMax ||
      head_size > kIntMax || batch > kIntMax || num_blocks > kIntMax ||
      block_size > kIntMax || max_blocks > kIntMax ||
      num_blocks > kIntMax / block_size || batch > kIntMax / max_blocks)
    return -1;
  if (head_size > kMaxHeadSize || num_heads > 65535)
    return -1;

  const size_t meta_bytes = meta_stage_bytes(batch, max_blocks);
  if (!scratch || scratch_bytes < meta_bytes)
    return -1;

  auto stream = static_cast<hipStream_t>(stream_ptr);
  int32_t *cumulative = static_cast<int32_t *>(scratch);
  int32_t *past = cumulative + (batch + 1);
  int32_t *table = past + batch;
  int status = copy_i32(stream, cumulative_sequence_length,
                        static_cast<int>(batch + 1), cumulative);
  if (status == 0)
    status = copy_i32(stream, past_seqlens, static_cast<int>(batch), past);
  if (status == 0)
    status = copy_i32(stream, block_table,
                      static_cast<int>(batch * max_blocks), table);
  if (status != 0)
    return status;

  DeviceMeta meta{cumulative, past, table};
  const int cache_slots = static_cast<int>(num_blocks * block_size);
  (void)hipGetLastError();
  hipLaunchKernelGGL(write_paged_kv, dim3(static_cast<unsigned>(num_tokens)),
                     dim3(128), 0, stream, static_cast<const __half *>(key),
                     static_cast<const __half *>(value),
                     static_cast<__half *>(key_cache),
                     static_cast<__half *>(value_cache), meta,
                     static_cast<int>(num_tokens), static_cast<int>(batch),
                     static_cast<int>(kv_num_heads), static_cast<int>(head_size),
                     static_cast<int>(block_size), static_cast<int>(max_blocks),
                     cache_slots);
  hipError_t err = hipGetLastError();
  FlashConfig flash_cfg{};
  const bool use_flash =
      err == hipSuccess &&
      select_flash(num_tokens, num_heads, kv_num_heads, head_size, block_size,
                   &flash_cfg);
  if (use_flash) {
    err = dispatch_flash(stream, flash_cfg, query, key_cache, value_cache,
                         output, meta, num_tokens, static_cast<int>(batch),
                         static_cast<int>(num_heads),
                         static_cast<int>(kv_num_heads),
                         static_cast<int>(block_size),
                         static_cast<int>(max_blocks), cache_slots, scale,
                         reinterpret_cast<float *>(
                             static_cast<char *>(scratch) + meta_bytes),
                         scratch_bytes - meta_bytes);
  } else if (err == hipSuccess) {
    err = launch_per_token(stream, query, key_cache, value_cache, output, meta,
                           static_cast<int>(num_tokens),
                           static_cast<int>(batch), static_cast<int>(num_heads),
                           static_cast<int>(kv_num_heads),
                           static_cast<int>(head_size),
                           static_cast<int>(block_size),
                           static_cast<int>(max_blocks), cache_slots, scale,
                           reinterpret_cast<float *>(
                               static_cast<char *>(scratch) + meta_bytes),
                           scratch_bytes - meta_bytes);
  }
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_paged_attention launch failed: %s\n",
            hipGetErrorString(err));
  }
  return err == hipSuccess ? 0 : static_cast<int>(err);
}
