/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Device half of the MultiHeadAttention flash-prefill kernel, split out so the
// same text can be compiled two ways: included by multi_head_attention_kernel.hip
// for the AOT build, and fed verbatim to hipRTC as an embedded string at
// runtime. Both paths must see identical source or the two code objects are not
// comparable.
//
// hipRTC constraints this file must respect:
//   * no <hip/*.h> includes -- hipRTC injects its own preamble and those paths
//     do not resolve
//   * no host-only headers (hip_custom_kernels.h)
//   * <cstdint> is required: without it int64_t resolves only inside
//     __hip_internal and the translation unit fails to compile

#ifndef HIPDNN_EP_RTC_MULTI_HEAD_ATTENTION_DEVICE_H
#define HIPDNN_EP_RTC_MULTI_HEAD_ATTENTION_DEVICE_H

#if !defined(__HIPCC_RTC__)
  #include <hip/hip_fp16.h>
  #include <hip/hip_runtime.h>
#endif

#include <cstdint>

#if defined(__HIPCC_RTC__)
  // hipRTC's preamble supplies the math functions but not the <math.h> macros,
  // and <cmath> is not an option here: it conflicts with clang's HIP mode (the
  // same conflict that made this kernel use INFINITY over numeric_limits).
  #define INFINITY __builtin_inff()
#endif

constexpr int kWmmaTile = 16;  // RDNA3 WMMA 16x16x16 tile
constexpr float kLog2e = 1.4426950408889634f;

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));

#define MHA_HALF16_LOAD(ptr) (*reinterpret_cast<const half16_t *>(ptr))

// WMMA builtin is RDNA3-family (wave32) only; trap on unsupported arch so the
// file still compiles and a bad launch aborts loudly instead of silent garbage.
#if !defined(__has_builtin) ||                                                  \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ float8_t
mha_wmma_unavailable(half16_t, half16_t, float8_t c) {
  __builtin_trap();
  return c;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)                     \
  mha_wmma_unavailable((a), (b), (c))
#endif

// Load one 16-wide D-tile of a contiguous head-dim row, zero-padding columns
// >= d. The common case (full tile) is a single vectorized 128-bit load; only
// the tail tile of a non-multiple-of-16 head dim takes the masked path.
__device__ __forceinline__ half16_t mha_load_dtile(const _Float16 *row, int dt,
                                                   int d) {
  if ((dt + 1) * kWmmaTile <= d)
    return MHA_HALF16_LOAD(row + dt * kWmmaTile);
  half16_t r{};
#pragma unroll
  for (int e = 0; e < kWmmaTile; ++e) {
    const int c = dt * kWmmaTile + e;
    r[e] = (c < d) ? row[c] : (_Float16)0.0f;
  }
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
  const int pair = lane_id / kWmmaTile;  // 0 -> even rows, 1 -> odd rows

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
      Q_reg[mt][tk] = valid ? mha_load_dtile(&Q[q_row], tk, d) : half16_t{};
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
      half16_t vv = kvalid ? mha_load_dtile(&Vcache[kv_base + (size_t)kv_pos * d],
                                            dt, d)
                           : half16_t{};
      *reinterpret_cast<half16_t *>(&V_lds[kv_local * KV_STR + dt * kWmmaTile]) = vv;
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
        half16_t k_frag = kvalid ? mha_load_dtile(k_row, tk, d) : half16_t{};
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          S_reg[mt][sj] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
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

        const int row = e * 2 + pair;
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
        for (int e = 0; e < kWmmaTile; ++e)
          v_frag[e] = V_lds[(tk * kWmmaTile + e) * KV_STR + tj * kWmmaTile +
                            wmma_lane];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          half16_t p_frag = MHA_HALF16_LOAD(
              &P_lds[(pbase + mt * kWmmaTile + wmma_lane) * P_STR +
                     tk * kWmmaTile]);
          Of[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_frag, v_frag,
                                                              Of[mt]);
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
        const int row = e * 2 + pair;
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

#endif  // HIPDNN_EP_RTC_MULTI_HEAD_ATTENTION_DEVICE_H
