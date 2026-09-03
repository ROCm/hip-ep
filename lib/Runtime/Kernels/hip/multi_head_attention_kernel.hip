/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Fused FA-2 WMMA prefill kernel for com.microsoft.MultiHeadAttention.
//
// Replaces the decomposed hipBLASLt pipeline (Q/K/V transpose -> score GEMM ->
// softmax -> value GEMM -> O transpose) that materializes the full fp32 score
// matrix S[B,N,Sq,Skv] in global memory. For the Qwen VLM vision encoder
// (B=1, N=16, Sq=Skv=7296, H=72) that matrix is ~3.4 GB and the op is entirely
// HBM-bandwidth bound (~2.9 s / prefill). Flash-attention keeps the running
// (m, l, O) softmax state in registers and never writes S to DRAM.
//
// This is the non-causal (bidirectional), self-attention (N_q == N_kv, HpG==1)
// specialization modeled on the GQA flash-prefill v7 kernel (gqa_kernel.hip):
//   * Q resident in registers as 16x16 WMMA fragments (D-tiled).
//   * K streamed from global straight into WMMA fragments.
//   * V staged into LDS; P (probabilities) transposed through LDS.
//   * Score / value GEMMs run on the RDNA3.5 WMMA matrix unit (wave32).
//   * Intra-wave online softmax in log2e space.
//
// Head dim H need not be a multiple of 16: the kernel is templated on the
// padded D (next multiple of 16) and the tail D-tile is masked-loaded (valid
// columns real, padded columns zero) so the padded dims contribute 0 to the
// QK dot and produce discarded (unwritten) output columns.
//
// Layout (matches the runtime call site after the K/V transpose):
//   Q      : [B, Sq,  N, H]   fp16   (BSND == BSHD, no transpose needed)
//   Kcache : [B, N, max_seq, H]  fp16   (BNSD, transposed from BSND)
//   Vcache : [B, N, max_seq, H]  fp16   (BNSD)
//   O      : [B, Sq,  N, H]   fp16   (BSND, no transpose needed)
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

namespace {

constexpr int kWmmaTile = 16;  // RDNA3 WMMA 16x16x16 tile
constexpr float kLog2e = 1.4426950408889634f;

// half16_t holds one WMMA A/B fragment; its width tracks HIPDNN_WMMA_FRAG_ELEMS
// (16 on gfx11, 8 on the gfx12-style encoding gfx1170/gfx12xx use). half16_raw_t
// stays a fixed 16 for the plain D-tile copies that stage V into LDS -- those
// are not fragments and must move a whole tile regardless of encoding.
typedef _Float16 half16_t __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef _Float16 half16_raw_t __attribute__((ext_vector_type(16)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));

#define MHA_FRAG_LOAD(ptr) (*reinterpret_cast<const half16_t *>(ptr))
#define MHA_HALF16_LOAD(ptr) (*reinterpret_cast<const half16_raw_t *>(ptr))

// Trap only where NEITHER encoding exists (e.g. CDNA/wave64). Keying on the
// gfx11 spelling alone made every WMMA site here a __builtin_trap() on gfx1170.
#if !HIPDNN_HAS_WMMA
static __device__ __forceinline__ float8_t
mha_wmma_unavailable(half16_t, half16_t, float8_t c) {
  __builtin_trap();
  return c;
}
#undef HIPDNN_WMMA_F32_16X16X16_F16
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c) mha_wmma_unavailable((a), (b), (c))
#endif

// Load a whole 16-wide D-tile of a contiguous head-dim row, zero-padding
// columns >= d. Used to stage V into LDS, where a full tile is always needed.
// The common case is a single vectorized 128-bit load; only the tail tile of a
// non-multiple-of-16 head dim takes the masked path.
__device__ __forceinline__ half16_raw_t mha_load_dtile(const _Float16 *row,
                                                       int dt, int d) {
  if ((dt + 1) * kWmmaTile <= d)
    return MHA_HALF16_LOAD(row + dt * kWmmaTile);
  half16_raw_t r{};
#pragma unroll
  for (int e = 0; e < kWmmaTile; ++e) {
    const int c = dt * kWmmaTile + e;
    r[e] = (c < d) ? row[c] : (_Float16)0.0f;
  }
  return r;
}

// Load this lane's WMMA A/B fragment from a D-tile: HIPDNN_WMMA_FRAG_ELEMS
// halves starting at dt*16 + k_off, where k_off is 0 on gfx11 (each lane takes
// the full 16-wide K range) and pair*8 on the gfx12-style encoding (each lane
// takes only its half of K).
__device__ __forceinline__ half16_t mha_load_frag(const _Float16 *row, int dt,
                                                  int k_off, int d) {
  const int base = dt * kWmmaTile + k_off;
  if (base + HIPDNN_WMMA_FRAG_ELEMS <= d)
    return MHA_FRAG_LOAD(row + base);
  half16_t r{};
#pragma unroll
  for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
    r[e] = (base + e < d) ? row[base + e] : (_Float16)0.0f;
  return r;
}

// ---------------------------------------------------------------------------
// Non-causal FA-2 WMMA prefill (HpG == 1). Grid: (num_q_tiles * B, N).
// Block: NW*32 threads. Each wave owns MT 16-row query sub-tiles; the K/V tile
// is shared block-wide (V in LDS, K streamed from global), and each K/V
// fragment feeds all MT M-tiles so LDS/global K/V traffic is amortized MT x.
// ---------------------------------------------------------------------------
// The softmax state (Q fragments, O accumulators, per-row m/l) must stay
// register-resident; an explicit occupancy target stops the backend from buying
// extra waves with scratch traffic. The bound tracks register demand, which
// grows with the D_PAD / kWmmaTile tile count.
template <int NW, int BKV, int D_PAD, int MT>
__global__ void __launch_bounds__(NW * 32)
__attribute__((amdgpu_waves_per_eu(D_PAD <= 32 ? 9 : (D_PAD <= 48 ? 7 : 6))))
mha_flash_prefill_kernel(
    const _Float16 *__restrict__ Q,       // [B, sq, N, d]
    const _Float16 *__restrict__ Kcache,  // [B, N, max_seq, d]
    const _Float16 *__restrict__ Vcache,  // [B, N, max_seq, d]
    _Float16 *__restrict__ O,             // [B, sq, N, d]
    int N, int sq, int skv, int d, int max_seq, float scale) {
  constexpr int ROWS = NW * MT * kWmmaTile;   // query rows per block
  constexpr int S_TILES_J = BKV / kWmmaTile;  // key-tiles per KV block
  constexpr int D_TILES = D_PAD / kWmmaTile;  // padded head-dim tiles
  constexpr int P_STR = BKV + 2;              // P_lds row stride (bank pad)
  constexpr int KV_STR = D_PAD + 2;           // V_lds row stride (bank pad)

  const int num_q_tiles = (sq + ROWS - 1) / ROWS;
  const int q_tile_idx = __builtin_amdgcn_readfirstlane(blockIdx.x % num_q_tiles);
  const int batch = __builtin_amdgcn_readfirstlane(blockIdx.x / num_q_tiles);
  const int head = __builtin_amdgcn_readfirstlane(blockIdx.y);  // HpG==1

  const int tid = threadIdx.x;
  const int wave_id = tid / 32;
  const int lane_id = tid % 32;
  const int wmma_lane = lane_id % kWmmaTile;
  const int pair = lane_id / kWmmaTile;  // selects this lane's accumulator rows
  // K-axis base this lane's pair-half contributes to every A/B fragment: 0 on
  // gfx11 (lanes 0-15 and 16-31 hold identical data), pair*8 on the gfx12-style
  // encoding (K split across the lane halves, no replication).
  const int k_off = hipdnn_wmma_k_off(pair);

  const int q_start = q_tile_idx * ROWS;
  const int row_base = q_start + wave_id * MT * kWmmaTile;  // wave's first row
  const int pbase = wave_id * MT * kWmmaTile;               // wave's P_lds base

  extern __shared__ char smem_mha[];
  _Float16 *V_lds = reinterpret_cast<_Float16 *>(smem_mha);
  _Float16 *P_lds =
      reinterpret_cast<_Float16 *>(smem_mha + (size_t)BKV * KV_STR * sizeof(_Float16));

  // KV base for this (batch, head): Kcache/Vcache are BNSD.
  const size_t kv_base =
      ((size_t)batch * N + head) * (size_t)max_seq * (size_t)d;

  // ---- Q resident in registers: Q_reg[mt][tk], row = row_base+mt*16+wmma_lane
  half16_t Q_reg[MT][D_TILES];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int q_pos = row_base + mt * kWmmaTile + wmma_lane;
    const bool valid = q_pos < sq;
    const size_t q_row =
        ((size_t)(batch * sq + (valid ? q_pos : 0)) * N + head) * (size_t)d;
#pragma unroll
    for (int tk = 0; tk < D_TILES; ++tk)
      Q_reg[mt][tk] =
          valid ? mha_load_frag(&Q[q_row], tk, k_off, d) : half16_t{};
  }

  // ---- running softmax state; O carried compactly as fp16 ----
  half8_t O_h[MT][D_TILES] = {};
  float m_reg[MT][8];
  float l_reg[MT][8];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      m_reg[mt][e] = -INFINITY;
      l_reg[mt][e] = 0.0f;
    }

  // Non-causal: every query attends to all keys [0, skv).
  const int num_tiles = (skv + BKV - 1) / BKV;

  for (int t = 0; t < num_tiles; ++t) {
    const int kv0 = t * BKV;

    // ---- cooperative V staging into LDS (K streams from global) ----
    for (int i = tid; i < BKV * D_TILES; i += NW * 32) {
      const int kv_local = i / D_TILES;
      const int dt = i % D_TILES;
      const int kv_pos = kv0 + kv_local;
      const bool kvalid = kv_pos < skv;
      half16_raw_t vv =
          kvalid ? mha_load_dtile(&Vcache[kv_base + (size_t)kv_pos * d], dt, d)
                 : half16_raw_t{};
      *reinterpret_cast<half16_raw_t *>(
          &V_lds[kv_local * KV_STR + dt * kWmmaTile]) = vv;
    }

    // ---- score GEMM: K streamed from global, reused across MT M-tiles ----
    float8_t S_reg[MT][S_TILES_J];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int sj = 0; sj < S_TILES_J; ++sj) S_reg[mt][sj] = float8_t{};

#pragma unroll
    for (int sj = 0; sj < S_TILES_J; ++sj) {
      const int key = kv0 + sj * kWmmaTile + wmma_lane;
      const bool kvalid = key < skv;
      const _Float16 *k_row = &Kcache[kv_base + (size_t)(kvalid ? key : 0) * d];
#pragma unroll  // static tk keeps Q_reg[][] register-resident
      for (int tk = 0; tk < D_TILES; ++tk) {
        half16_t k_frag =
            kvalid ? mha_load_frag(k_row, tk, k_off, d) : half16_t{};
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          S_reg[mt][sj] = HIPDNN_WMMA_F32_16X16X16_F16(
              Q_reg[mt][tk], k_frag, S_reg[mt][sj]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int sj = 0; sj < S_TILES_J; ++sj)
#pragma unroll
        for (int e = 0; e < 8; ++e) S_reg[mt][sj][e] *= scale * kLog2e;

    // ---- online softmax (per M-tile, intra-wave); fold corr into O_h ----
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float corr_e[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float rmax = -INFINITY;
#pragma unroll
        for (int sj = 0; sj < S_TILES_J; ++sj) {
          const int kv_pos = kv0 + sj * kWmmaTile + wmma_lane;
          if (kv_pos >= skv) S_reg[mt][sj][e] = -INFINITY;
          rmax = fmaxf(rmax, S_reg[mt][sj][e]);
        }
        rmax = fmaxf(rmax, __shfl_xor(rmax, 1));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 2));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 4));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 8));

        const float m_old = m_reg[mt][e];
        const float m_new = fmaxf(m_old, rmax);
        const float corr = exp2f(m_old - m_new);
        corr_e[e] = corr;

        const int row = hipdnn_wmma_acc_row(pair, e);
        float rsum = 0.0f;
#pragma unroll
        for (int sj = 0; sj < S_TILES_J; ++sj) {
          const float p = exp2f(S_reg[mt][sj][e] - m_new);
          rsum += p;
          P_lds[(pbase + mt * kWmmaTile + row) * P_STR + sj * kWmmaTile +
                wmma_lane] = static_cast<_Float16>(p);
        }
        rsum += __shfl_xor(rsum, 1);
        rsum += __shfl_xor(rsum, 2);
        rsum += __shfl_xor(rsum, 4);
        rsum += __shfl_xor(rsum, 8);

        m_reg[mt][e] = m_new;
        l_reg[mt][e] = corr * l_reg[mt][e] + rsum;
      }
#pragma unroll
      for (int tj = 0; tj < D_TILES; ++tj)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          O_h[mt][tj][e] =
              static_cast<_Float16>((float)O_h[mt][tj][e] * corr_e[e]);
    }
    __syncthreads();  // P_lds produced; V_lds staged -> both read below

    // ---- value GEMM: V gathered once per (tj,tk), reused across MT ----
#pragma unroll  // static tj keeps O_h[][] register-resident
    for (int tj = 0; tj < D_TILES; ++tj) {
      float8_t Of[MT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 8; ++e) Of[mt][e] = (float)O_h[mt][tj][e];
#pragma unroll
      for (int tk = 0; tk < S_TILES_J; ++tk) {
        half16_t v_frag;
#pragma unroll
        for (int e = 0; e < HIPDNN_WMMA_FRAG_ELEMS; ++e)
          v_frag[e] = V_lds[(tk * kWmmaTile + k_off + e) * KV_STR +
                            tj * kWmmaTile + wmma_lane];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          half16_t p_frag = MHA_FRAG_LOAD(
              &P_lds[(pbase + mt * kWmmaTile + wmma_lane) * P_STR +
                     tk * kWmmaTile + k_off]);
          Of[mt] = HIPDNN_WMMA_F32_16X16X16_F16(p_frag, v_frag, Of[mt]);
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          O_h[mt][tj][e] = static_cast<_Float16>(Of[mt][e]);
    }
    __syncthreads();  // WAR: V_lds reused next tile
  }

  // ---- Epilogue: O / l, write only valid head-dim columns ----
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int tj = 0; tj < D_TILES; ++tj)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int row = hipdnn_wmma_acc_row(pair, e);
        const int c = tj * kWmmaTile + wmma_lane;
        const int q_pos = row_base + mt * kWmmaTile + row;
        if (q_pos < sq && c < d) {
          const float l_val = fmaxf(l_reg[mt][e], 1e-6f);
          const float out_val = (float)O_h[mt][tj][e] / l_val;
          O[((size_t)(batch * sq + q_pos) * N + head) * d + c] =
              static_cast<_Float16>(out_val);
        }
      }
}

static inline size_t mha_prefill_smem_bytes(int nw, int bkv, int d_pad, int mt) {
  const size_t v_lds = (size_t)bkv * (d_pad + 2) * sizeof(_Float16);
  const size_t p_lds = (size_t)(nw * mt * 16) * (bkv + 2) * sizeof(_Float16);
  return v_lds + p_lds;
}

// Default launch config (correctness-first; autotune can be layered later).
// rows/block = NW*MT*16 = 64; V+P LDS at d_pad=80,BKV=64 ~= 19 KB.
constexpr int kMhaNW = 4;
constexpr int kMhaBKV = 64;
constexpr int kMhaMT = 1;

template <int D_PAD>
static inline int launch_mha_prefill(hipStream_t stream, const _Float16 *Q,
                                     const _Float16 *K, const _Float16 *V,
                                     _Float16 *O, int B, int N, int sq, int skv,
                                     int d, int max_seq, float scale) {
  const int rows = kMhaNW * kMhaMT * 16;
  const int num_q_tiles = (sq + rows - 1) / rows;
  dim3 grid(num_q_tiles * B, N);
  const size_t smem = mha_prefill_smem_bytes(kMhaNW, kMhaBKV, D_PAD, kMhaMT);
  (void)hipGetLastError();
  mha_flash_prefill_kernel<kMhaNW, kMhaBKV, D_PAD, kMhaMT>
      <<<grid, kMhaNW * 32, smem, stream>>>(Q, K, V, O, N, sq, skv, d, max_seq,
                                            scale);
  return (hipGetLastError() == hipSuccess) ? 0 : -1;
}

}  // namespace

// hip_mha_flash_prefill: fused non-causal FA-2 prefill for MultiHeadAttention.
// Q [B,sq,N,d], K/V [B,N,max_seq,d] (BNSD), O [B,sq,N,d]; fp16. Returns 0 on
// success, -1 if the head dim is unsupported (d_pad > 256).
extern "C" int hip_mha_flash_prefill(void *stream, const void *Q,
                                     const void *Kcache, const void *Vcache,
                                     void *O, int B, int N, int sq, int skv,
                                     int d, int max_seq, float scale) {
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  const _Float16 *q = reinterpret_cast<const _Float16 *>(Q);
  const _Float16 *k = reinterpret_cast<const _Float16 *>(Kcache);
  const _Float16 *v = reinterpret_cast<const _Float16 *>(Vcache);
  _Float16 *o = reinterpret_cast<_Float16 *>(O);
  const int d_pad = ((d + 15) / 16) * 16;

  switch (d_pad) {
  case 16:  return launch_mha_prefill<16>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 32:  return launch_mha_prefill<32>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 48:  return launch_mha_prefill<48>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 64:  return launch_mha_prefill<64>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 80:  return launch_mha_prefill<80>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 96:  return launch_mha_prefill<96>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 112: return launch_mha_prefill<112>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 128: return launch_mha_prefill<128>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  case 256: return launch_mha_prefill<256>(s, q, k, v, o, B, N, sq, skv, d, max_seq, scale);
  default:
    fprintf(stderr, "[hip_mha_flash_prefill] unsupported head dim d=%d (d_pad=%d)\n",
            d, d_pad);
    return -1;
  }
}
