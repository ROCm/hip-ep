/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- paged_attention_kernel.hip - Paged KV Cache Attention Kernels ------===//
//
// Implements three GPU kernels for com.microsoft.PagedAttention (Phase 4b):
//
//   1. hip_paged_kv_write      — scatter new tokens into the paged KV slab
//   2. hip_paged_flash_decode  — FA-2 split-K decode over paged KV blocks
//   3. hip_paged_flash_decode_reduce — merge split-K partials
//
// KV cache layout: NHD (ORT-compatible)
//   key_cache / value_cache: [num_blocks, block_size, kv_num_heads, head_dim]
//
// Physical address for slot (block, token_in_block, head_kv, d_i):
//   ptr[block * block_size * G * D + token_in_block * G * D + head_kv * D + d_i]
//
// Block size is fixed at compile time via the BLOCK_SIZE template parameter
// (default 16 — matches AMD mfma16x16x16 block granularity and vLLM default).
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"
#include "debug_log.h"

static constexpr int PAGED_WAVE_SIZE = 32; // RDNA wave32
static constexpr float PAGED_LOG2E = 1.4426950408889634f;

// =========================================================================
// Kernel 1: paged_kv_write_kernel
// =========================================================================
// Scatters new KV tokens into the paged slab using slot_mapping.
//
// Thread layout: one thread per (token_idx, head_kv, d_i) element.
//   gridX = ceil(num_tokens * kv_num_heads * head_dim / BLOCK_DIM)
//   blockX = BLOCK_DIM = 256
//
// slot = slot_mapping[token_idx]  →  block_num = slot / BLOCK_SIZE
//                                    block_off  = slot % BLOCK_SIZE
// dst_idx = block_num * BLOCK_SIZE * G * D
//         + block_off * G * D
//         + head_kv * D + d_i
// src_idx = token_idx * G * D + head_kv * D + d_i  (BSHD-flattened)
// =========================================================================

template <typename T, int BLOCK_SIZE>
__global__ void paged_kv_write_kernel(
    const T *__restrict__ new_key,
    const T *__restrict__ new_value,
    T *__restrict__ key_cache,
    T *__restrict__ value_cache,
    const int *__restrict__ slot_mapping,
    int num_tokens, int G, int D) {
  // Total elements: num_tokens * G * D
  const int total = num_tokens * G * D;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  // Decompose flat index
  const int d_i = idx % D;
  const int head_kv = (idx / D) % G;
  const int token_idx = idx / (D * G);

  const int slot = __builtin_amdgcn_readfirstlane(slot_mapping[token_idx]);
  if (slot < 0)
    return; // masked / padding token

  const int block_num = slot / BLOCK_SIZE;
  const int block_off = slot % BLOCK_SIZE;

  // NHD layout: [block, token_in_block, head, d]
  const size_t dst = static_cast<size_t>(block_num) * BLOCK_SIZE * G * D +
                     block_off * G * D + head_kv * D + d_i;
  const size_t src = static_cast<size_t>(token_idx) * G * D + head_kv * D + d_i;

  key_cache[dst] = new_key[src];
  value_cache[dst] = new_value[src];
}

// =========================================================================
// Kernel 2: paged_flash_decode_kernel
// =========================================================================
// FA-2 split-K decode attention over a paged KV cache (block_size=16, NHD).
//
// Adapted from gqa_flash_decode_kernel in gqa_kernel.hip. The only structural
// difference: the KV tile lookup uses block_table indirection instead of
// a contiguous base offset.
//
// Grid:  (B * G, K_SPLITS)  — one block per (batch × kv_head, partition)
// Block: HPG * 32 threads   — HPG waves, each wave handles one query head
//
// Inner tile loop:
//   tile_start iterates over token positions 0..seq_len in steps of BKV=16
//   For each tile:
//     block_idx = tile_start / BLOCK_SIZE
//     phys_block = block_table[seq * max_blocks + block_idx]
//     K_ptr = key_cache + phys_block * BLOCK_SIZE * G * D
//                       + block_off * G * D + head_kv * D
//   (compared to contiguous: K_ptr = Kcache + (batch*G+head_kv)*max_seq*D
//                                           + tile_start * D)
//
// Partial layout: [B, H, K_SPLITS, D+2]  (m:float, l:float, O[D]:float)
// =========================================================================

__device__ __forceinline__ size_t paged_partial_offset(
    int b, int h_q, int k_split, int H, int K_SPLITS, int D) {
  return ((size_t)((b * H + h_q) * K_SPLITS + k_split)) * (D + 2);
}

template <int D, int HPG, int BLOCK_SIZE>
__global__
    __launch_bounds__(HPG *PAGED_WAVE_SIZE) void paged_flash_decode_kernel(
        const __half *__restrict__ Q_roped, // [B, 1, H, D]
        const __half *__restrict__ key_cache,   // [num_blocks, BS, G, D] NHD
        const __half *__restrict__ value_cache, // [num_blocks, BS, G, D] NHD
        const int *__restrict__ block_table, // [B, max_blocks_per_seq]
        const int *__restrict__ seq_lens,    // [B]
        float *__restrict__ partials,        // [B, H, K_SPLITS, D+2]
        int H, int G, int max_blocks_per_seq, float scale, int K_SPLITS) {
  constexpr int EPT = D / PAGED_WAVE_SIZE; // elements per thread along d
  constexpr int THREADS = HPG * PAGED_WAVE_SIZE;

  static_assert(D % PAGED_WAVE_SIZE == 0, "D must be multiple of WAVE_SIZE");
  static_assert(HPG >= 1, "HPG must be >= 1");

  const int batch = __builtin_amdgcn_readfirstlane(blockIdx.x / G);
  const int head_kv = __builtin_amdgcn_readfirstlane(blockIdx.x % G);
  const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
  const int tid = threadIdx.x;
  const int wave_id = tid / PAGED_WAVE_SIZE; // [0, HPG)
  const int lane_id = tid % PAGED_WAVE_SIZE;
  const int head_q = head_kv * HPG + wave_id;

  // Effective KV sequence length for this batch.
  const int seq_len = seq_lens ? seq_lens[batch] : 0;
  const int eff_skv = (seq_len > 0) ? seq_len : 0;

  // Partition this split's KV range.
  const int chunk = (eff_skv + K_SPLITS - 1) / K_SPLITS;
  const int kv_start = chunk * split_idx;
  const int kv_end_raw = kv_start + chunk;
  const int kv_end = (kv_end_raw < eff_skv) ? kv_end_raw : eff_skv;

  float *partial_dst =
      partials + paged_partial_offset(batch, head_q, split_idx, H, K_SPLITS, D);

  // Empty slice: write identity.
  if (kv_start >= kv_end) {
    if (lane_id == 0) {
      partial_dst[0] = -INFINITY;
      partial_dst[1] = 0.0f;
    }
#pragma unroll
    for (int e = 0; e < EPT; ++e)
      partial_dst[2 + lane_id * EPT + e] = 0.0f;
    return;
  }

  // LDS tile for K (shared across HPG waves for GQA dedup).
  constexpr bool kStage = (HPG > 1);
  constexpr int kShElems = kStage ? (BLOCK_SIZE * D) : 1;
  __shared__ __align__(16) __half K_lds[kShElems];
  __shared__ __align__(16) __half V_lds[kShElems];

  // Load Q into registers (fp32).
  float Q_f[EPT];
  {
    const __half *Q_ptr = Q_roped + (size_t)(batch * H + head_q) * D;
#pragma unroll
    for (int e = 0; e < EPT; ++e)
      Q_f[e] = __half2float(Q_ptr[lane_id * EPT + e]);
  }

  const float scale_log2e = scale * PAGED_LOG2E;

  float m_acc = -INFINITY;
  float l_acc = 0.0f;
  float O_acc[EPT];
#pragma unroll
  for (int e = 0; e < EPT; ++e)
    O_acc[e] = 0.0f;

  // Iterate over KV blocks in this partition.
  // Each tile = one physical block of BLOCK_SIZE tokens.
  for (int tile_start = kv_start; tile_start < kv_end;
       tile_start += BLOCK_SIZE) {
    const int tile_n =
        ((tile_start + BLOCK_SIZE) <= kv_end) ? BLOCK_SIZE : (kv_end - tile_start);

    // Block table lookup: logical block → physical block.
    const int block_idx = tile_start / BLOCK_SIZE;
    const int phys_block =
        block_table[batch * max_blocks_per_seq + block_idx];

    // NHD layout: key_cache[phys_block, :, head_kv, :]
    // base: phys_block * BLOCK_SIZE * G * D + head_kv * D
    const size_t kv_block_base =
        (size_t)phys_block * BLOCK_SIZE * G * D + (size_t)head_kv * D;

    // Stage K tile into LDS (GQA: HPG heads share the load).
    if constexpr (kStage) {
      // Cooperative load: THREADS threads load tile_n * D fp16 elements.
      const int total_elems = tile_n * D;
      for (int ei = tid; ei < total_elems; ei += THREADS) {
        const int tok = ei / D;
        const int di = ei % D;
        // NHD: token stride is G*D, d stride is 1
        K_lds[tok * D + di] =
            key_cache[kv_block_base + (size_t)tok * G * D + di];
        V_lds[tok * D + di] =
            value_cache[kv_block_base + (size_t)tok * G * D + di];
      }
      __syncthreads();
    }

    // QK dot products for this tile.
    for (int t = 0; t < tile_n; ++t) {
      // Load K vector for token t.
      float k_f[EPT];
      if constexpr (kStage) {
        const __half *Kt = K_lds + t * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e)
          k_f[e] = __half2float(Kt[lane_id * EPT + e]);
      } else {
        const __half *Kt = key_cache + kv_block_base + (size_t)t * G * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e)
          k_f[e] = __half2float(Kt[lane_id * EPT + e]);
      }

      // Dot product Q·K (sum across lanes via shuffle).
      float qk = 0.0f;
#pragma unroll
      for (int e = 0; e < EPT; ++e)
        qk += Q_f[e] * k_f[e];
#pragma unroll
      for (int offset = PAGED_WAVE_SIZE / 2; offset > 0; offset >>= 1)
        qk += __shfl_xor(qk, offset);

      const float qk_scaled = qk * scale_log2e;

      // Online softmax update (log2e space).
      const float m_prev = m_acc;
      m_acc = (m_prev > qk_scaled) ? m_prev : qk_scaled;
      const float rescale = __builtin_exp2f(m_prev - m_acc);
      l_acc = l_acc * rescale + __builtin_exp2f(qk_scaled - m_acc);

      // V accumulation.
      const float p = __builtin_exp2f(qk_scaled - m_acc);
      if constexpr (kStage) {
        const __half *Vt = V_lds + t * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
          O_acc[e] = O_acc[e] * rescale +
                     p * __half2float(Vt[lane_id * EPT + e]);
        }
      } else {
        const __half *Vt = value_cache + kv_block_base + (size_t)t * G * D;
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
          O_acc[e] = O_acc[e] * rescale +
                     p * __half2float(Vt[lane_id * EPT + e]);
        }
      }
    }

    if constexpr (kStage)
      __syncthreads();
  }

  // Write partials.
  if (lane_id == 0) {
    partial_dst[0] = m_acc;
    partial_dst[1] = l_acc;
  }
#pragma unroll
  for (int e = 0; e < EPT; ++e)
    partial_dst[2 + lane_id * EPT + e] = O_acc[e];
}

// =========================================================================
// Kernel 3: paged_flash_decode_reduce_kernel
// =========================================================================
// Merges K_SPLITS partial results into the final output.
// Identical algorithm to gqa_flash_decode_reduce_kernel in gqa_kernel.hip.
// One block per (batch, query_head); threads cover the D dimension.
// =========================================================================

template <int D>
__global__ void paged_flash_decode_reduce_kernel(
    const float *__restrict__ partials, // [B, H, K_SPLITS, D+2]
    __half *__restrict__ output,        // [B, H, D] BNHD
    int H, int K_SPLITS) {
  const int batch = blockIdx.x;
  const int head_q = blockIdx.y;
  const int tid = threadIdx.x;

  constexpr int EPT = D / PAGED_WAVE_SIZE;
  const int lane = tid % PAGED_WAVE_SIZE;

  float m_global = -INFINITY;
  float l_global = 0.0f;
  float O_out[EPT];
#pragma unroll
  for (int e = 0; e < EPT; ++e)
    O_out[e] = 0.0f;

  for (int s = 0; s < K_SPLITS; ++s) {
    const float *p =
        partials + paged_partial_offset(batch, head_q, s, H, K_SPLITS, D);
    const float m_s = p[0];
    const float l_s = p[1];
    if (m_s == -INFINITY)
      continue;

    const float m_new = (m_global > m_s) ? m_global : m_s;
    const float rg = __builtin_exp2f(m_global - m_new);
    const float rs = __builtin_exp2f(m_s - m_new);
    l_global = l_global * rg + l_s * rs;
    m_global = m_new;
#pragma unroll
    for (int e = 0; e < EPT; ++e)
      O_out[e] = O_out[e] * rg + p[2 + lane * EPT + e] * rs;
  }

  // Normalize and write output.
  const float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
  __half *out_ptr = output + (size_t)(batch * H + head_q) * D;
#pragma unroll
  for (int e = 0; e < EPT; ++e)
    out_ptr[lane * EPT + e] = __float2half(O_out[e] * inv_l);
}

// =========================================================================
// Launcher helpers
// =========================================================================

template <typename T>
static hipError_t launch_paged_kv_write(
    void *stream, const T *new_key, const T *new_value, T *key_cache,
    T *value_cache, const int *slot_mapping, int num_tokens, int G, int D,
    int block_size) {
  (void)hipGetLastError();
  const int total = num_tokens * G * D;
  const int block_dim = 256;
  const int grid_dim = (total + block_dim - 1) / block_dim;
  if (block_size == 16) {
    paged_kv_write_kernel<T, 16><<<grid_dim, block_dim, 0,
                                   static_cast<hipStream_t>(stream)>>>(
        new_key, new_value, key_cache, value_cache, slot_mapping, num_tokens, G,
        D);
  } else {
    fprintf(stderr, "hip_paged_kv_write: unsupported block_size=%d\n",
            block_size);
    return hipErrorInvalidValue;
  }
  return hipGetLastError();
}

// Dispatch table for paged_flash_decode (D × HPG).
template <int D, int HPG>
static hipError_t dispatch_paged_decode(
    void *stream, const __half *Q, const __half *Kc, const __half *Vc,
    const int *block_table, const int *seq_lens, float *partials, int B, int H,
    int G, int max_blocks_per_seq, float scale, int K_SPLITS) {
  (void)hipGetLastError();
  dim3 grid(B * G, K_SPLITS);
  const int block_threads = HPG * PAGED_WAVE_SIZE;
  paged_flash_decode_kernel<D, HPG, 16><<<grid, block_threads, 0,
                                          static_cast<hipStream_t>(stream)>>>(
      Q, Kc, Vc, block_table, seq_lens, partials, H, G, max_blocks_per_seq,
      scale, K_SPLITS);
  return hipGetLastError();
}

// =========================================================================
// Extern C entry points
// =========================================================================

extern "C" int hip_paged_kv_write(
    void *stream, const void *new_key, const void *new_value, void *key_cache,
    void *value_cache, const int *slot_mapping, int num_tokens, int kv_num_heads,
    int head_dim, int block_size, int element_size_bytes,
    const void * /*cos_cache*/, const void * /*sin_cache*/,
    const int * /*past_lens*/, const int * /*token_to_seq*/) {
  // RoPE integration deferred to Milestone 2 (requires seqlens + position math)
  if (element_size_bytes == 2) {
    return launch_paged_kv_write<__half>(
        stream, static_cast<const __half *>(new_key),
        static_cast<const __half *>(new_value),
        static_cast<__half *>(key_cache), static_cast<__half *>(value_cache),
        slot_mapping, num_tokens, kv_num_heads, head_dim, block_size);
  }
  fprintf(stderr, "hip_paged_kv_write: unsupported element_size_bytes=%d\n",
          element_size_bytes);
  return -1;
}

extern "C" int hip_paged_flash_decode(
    void *stream, const void *Q_roped, const void *key_cache,
    const void *value_cache, const int *block_table, const int *sequence_lengths,
    float *partials, int B, int H, int G, int head_dim, int max_blocks_per_seq,
    int /*block_size*/, int K_SPLITS, float scale, int element_size_bytes) {
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "hip_paged_flash_decode: unsupported element_size_bytes=%d\n",
            element_size_bytes);
    return -1;
  }
  const auto *Q = static_cast<const __half *>(Q_roped);
  const auto *Kc = static_cast<const __half *>(key_cache);
  const auto *Vc = static_cast<const __half *>(value_cache);
  const int HPG = H / G;

  if (head_dim == 64) {
    if (HPG == 4)
      return dispatch_paged_decode<64, 4>(stream, Q, Kc, Vc, block_table,
                                          sequence_lengths, partials, B, H, G,
                                          max_blocks_per_seq, scale, K_SPLITS);
    if (HPG == 8)
      return dispatch_paged_decode<64, 8>(stream, Q, Kc, Vc, block_table,
                                          sequence_lengths, partials, B, H, G,
                                          max_blocks_per_seq, scale, K_SPLITS);
  } else if (head_dim == 128) {
    if (HPG == 4)
      return dispatch_paged_decode<128, 4>(stream, Q, Kc, Vc, block_table,
                                           sequence_lengths, partials, B, H, G,
                                           max_blocks_per_seq, scale, K_SPLITS);
    if (HPG == 8)
      return dispatch_paged_decode<128, 8>(stream, Q, Kc, Vc, block_table,
                                           sequence_lengths, partials, B, H, G,
                                           max_blocks_per_seq, scale, K_SPLITS);
  }
  fprintf(stderr,
          "hip_paged_flash_decode: unsupported head_dim=%d HPG=%d\n",
          head_dim, HPG);
  return -1;
}

extern "C" int hip_paged_flash_decode_reduce(
    void *stream, const float *partials, void *output, int B, int H,
    int head_dim, int K_SPLITS, int element_size_bytes) {
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "hip_paged_flash_decode_reduce: unsupported element_size_bytes=%d\n",
            element_size_bytes);
    return -1;
  }
  (void)hipGetLastError();
  dim3 grid(B, H);
  if (head_dim == 64) {
    paged_flash_decode_reduce_kernel<64><<<grid, 64, 0,
                                          static_cast<hipStream_t>(stream)>>>(
        partials, static_cast<__half *>(output), H, K_SPLITS);
  } else if (head_dim == 128) {
    paged_flash_decode_reduce_kernel<128><<<grid, 128, 0,
                                           static_cast<hipStream_t>(stream)>>>(
        partials, static_cast<__half *>(output), H, K_SPLITS);
  } else {
    fprintf(stderr,
            "hip_paged_flash_decode_reduce: unsupported head_dim=%d\n",
            head_dim);
    return -1;
  }
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_paged_flash_decode_reduce failed: %s\n",
            hipGetErrorString(err));
  return err;
}
