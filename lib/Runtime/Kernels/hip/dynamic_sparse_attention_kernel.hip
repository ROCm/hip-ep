/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * com.microsoft DynamicSparseAttention, attention_mode = selected_only over the
 * main KV cache: every query row attends to its own list of selected cache
 * positions. Numerics follow the ORT CUDA kernel: fp32 logits and online
 * softmax, invalid candidates skipped, empty candidate set -> zero output.
 */

#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kTile = 16;
constexpr int kBkv = 32;
constexpr int kMaxSplits = 128;
constexpr int kTargetBlocks = 256;
constexpr int kAppendThreads = 256;
constexpr float kLog2e = 1.4426950408889634f;

typedef _Float16 frag_t
    __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float acc_t __attribute__((ext_vector_type(8)));
typedef unsigned int vec16_t __attribute__((ext_vector_type(4)));

#if HIPDNN_HAS_WMMA
#define DSA_WMMA(a, b, c) HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)
#else
__device__ __forceinline__ acc_t dsa_wmma_unavailable(frag_t, frag_t, acc_t c) {
  __builtin_trap();
  return c;
}
#define DSA_WMMA(a, b, c) dsa_wmma_unavailable((a), (b), (c))
#endif

__device__ __forceinline__ frag_t load_frag(const _Float16 *p, int pair) {
  return *reinterpret_cast<const frag_t *>(p + hipdnn_wmma_k_off(pair));
}

// Writes the new tokens' K/V (BSHD) into the BNSH present cache at
// seqlens_k[b] + 1 - S + s, dropping only the tokens that land outside the
// cache. One thread moves one 16-byte chunk of K and the same chunk of V.
__global__ void __launch_bounds__(kAppendThreads)
dsa_kv_append_kernel(const vec16_t *__restrict__ key,
                     const vec16_t *__restrict__ value,
                     vec16_t *__restrict__ present_key,
                     vec16_t *__restrict__ present_value,
                     const int *__restrict__ seqlens_k, int seq_len,
                     int kv_heads, int chunks_per_head, int cap) {
  const int b = blockIdx.y;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= seq_len * kv_heads * chunks_per_head) {
    return;
  }
  const int c = i % chunks_per_head;
  const int h = (i / chunks_per_head) % kv_heads;
  const int s = i / (chunks_per_head * kv_heads);
  const int dest = seqlens_k[b] + 1 - seq_len + s;
  if (dest < 0 || dest >= cap) {
    return;
  }
  const size_t src =
      (static_cast<size_t>(b * seq_len + s) * kv_heads + h) * chunks_per_head +
      c;
  const size_t dst =
      ((static_cast<size_t>(b) * kv_heads + h) * cap + dest) * chunks_per_head +
      c;
  present_key[dst] = key[src];
  present_value[dst] = value[src];
}

// Exchanges a value with lane ^ 16 on the VALU. The inline-asm copy keeps the
// compiler from dual-issuing the copy of x into the tied "old" operand, which
// can lose its interlock on gfx11 (see v6_swap_halves in gqa_kernel.hip).
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

// Orders one wave's LDS accesses across its lanes without a block barrier.
__device__ __forceinline__ void wave_lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

typedef int dw8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half2v_t __attribute__((ext_vector_type(2)));

// Feeds NF operand fragments to use(f, frag) with at most RING reads issued
// ahead of their WMMA. Left alone, the scheduler issues every LDS read of a
// GEMM step up front and the tile spills (see v6_frag_stream in
// gqa_kernel.hip).
template <int NF, int RING, typename Load, typename Use>
__device__ __forceinline__ void frag_stream(Load &&load, Use &&use) {
  constexpr int R = RING < NF ? RING : NF;
  typedef int dw_t __attribute__((ext_vector_type(sizeof(frag_t) / 4)));
  dw_t ring[R];
#pragma unroll
  for (int f = 0; f < R; ++f) {
    ring[f] = __builtin_bit_cast(dw_t, load(f));
  }
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    dw_t x = ring[f % R];
    if (f + R < NF) {
      ring[f % R] = __builtin_bit_cast(dw_t, load(f + R));
    }
    asm("" : "+v"(x));
    use(f, __builtin_bit_cast(frag_t, x));
#if HIPDNN_HAS_WMMA
    __builtin_amdgcn_sched_barrier(0);
#endif
  }
}

// One block per (query row, KV head, split), transposed-score core: S^T =
// K Q^T with the candidates as WMMA rows and the KV group's heads as columns
// (heads >= hpg are zero columns and are never written), so a lane owns one
// head. The row max and sum reduce in registers plus one half-wave exchange,
// and P^T is the B operand of O^T = V^T P^T straight from registers. The ND
// waves each own D/ND of the head dim: K fragments come straight from the
// cache (prefetched one tile ahead) and each wave stages its own V^T slice,
// so the only block barriers per tile publish the partial scores, summed in
// the same order by every wave. Lane i of a tile holds candidate i's index.
// splits == 1 writes the normalized output, otherwise (m, l, O) partials in
// log2 units for dsa_combine_kernel.
template <int D, int ND>
__global__ void __launch_bounds__(32 * ND)
__attribute__((amdgpu_waves_per_eu(HIPDNN_WAVES_FOR_VGPRS(256))))
dsa_attention_kernel(const _Float16 *__restrict__ query,
                     const _Float16 *__restrict__ present_key,
                     const _Float16 *__restrict__ present_value,
                     const int *__restrict__ selected_indices,
                     const int *__restrict__ selected_counts,
                     const int *__restrict__ seqlens_k,
                     _Float16 *__restrict__ output,
                     float *__restrict__ partials, int seq_len, int num_heads,
                     int kv_heads, int cap, int max_sel, int splits,
                     float scale_log2) {
  constexpr int DT = D / kTile / ND;
  constexpr int DS = DT * kTile;
  constexpr int N_TILES = kBkv / kTile;
  constexpr int VT_STR = kBkv + 8;
  constexpr int V_CHUNKS = DS / kTile;
  // The running max only moves once a tile's max exceeds it by this much
  // (log2 units), so P stays <= 2^8 and most tiles skip the O rescale.
  constexpr float kRescaleSlack = 8.0f;
  static_assert(DT * kTile * ND == D, "ND must divide the 16-wide d tiles");
  static_assert(kBkv == 32, "one candidate per lane");

  __shared__ __align__(16) _Float16 vt_lds[ND][DS * VT_STR];
  __shared__ __align__(16) float sp_lds[ND][N_TILES][32 * 8];

  const int row = blockIdx.x;
  const int g = blockIdx.y;
  const int split = blockIdx.z;
  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lane = tid % 32;
  const int wl = lane % kTile;
  const int pair = lane / kTile;
  const int hpg = num_heads / kv_heads;
  const int b = row / seq_len;
  const int s = row % seq_len;

  const int total = __builtin_amdgcn_readfirstlane(seqlens_k[b] + 1);
  const bool valid_len = seq_len <= total && total <= cap;
  const int qpos = total - seq_len + s;
  int count = 0;
  if (valid_len) {
    count = __builtin_amdgcn_readfirstlane(selected_counts[row]);
    count = count < 0 ? 0 : (count > max_sel ? max_sel : count);
  }
  int chunk = (count + splits - 1) / splits;
  chunk = (chunk + kBkv - 1) / kBkv * kBkv;
  const int c_lo = split * chunk;
  const int c_hi = c_lo + chunk < count ? c_lo + chunk : count;
  const int num_tiles = c_hi > c_lo ? (c_hi - c_lo + kBkv - 1) / kBkv : 0;

  // The tile max is taken on the raw scores and scaled afterwards, which
  // needs sc > 0 (the converter rejects a negative scale); a zero scale is a
  // zero Q (all logits 0) with sc = 1.
  const float sc = scale_log2 > 0.0f ? scale_log2 : 1.0f;
  frag_t q_reg[DT];
  {
    const bool ok = wl < hpg && scale_log2 > 0.0f;
    const _Float16 *q_row =
        query +
        (static_cast<size_t>(row) * num_heads + g * hpg + (ok ? wl : 0)) * D +
        wave * DS;
#pragma unroll
    for (int tk = 0; tk < DT; ++tk) {
      q_reg[tk] = ok ? load_frag(q_row + tk * kTile, pair) : frag_t{};
    }
  }

  const size_t kv_base = (static_cast<size_t>(b) * kv_heads + g) * cap * D;
  const _Float16 *k_base = present_key + kv_base + wave * DS;
  const _Float16 *v_base = present_value + kv_base + wave * DS;
  const int *sel_row = selected_indices + static_cast<size_t>(row) * max_sel;
  _Float16 *vt = vt_lds[wave];

  auto load_idx = [&](int t) {
    const int c = c_lo + t * kBkv + lane;
    int idx = -1;
    if (t < num_tiles && c < c_hi) {
      idx = sel_row[c];
      // valid_len implies qpos < total <= cap, so this is the whole ORT
      // filter 0 <= idx < total, idx <= qpos, idx < cap.
      if (idx < 0 || idx > qpos) {
        idx = -1;
      }
    }
    return idx;
  };
  // Skipped candidates read cache row 0, which is in bounds whenever a tile
  // exists (count > 0 needs valid_len, so total >= 1): their scores are
  // masked to -inf and their V rows zeroed before staging. Row offsets are
  // 32-bit byte offsets (a [cap, D] head is far below 4 GiB) on the
  // wave-uniform head base, so each address is one VGPR.
  auto row_at = [&](const _Float16 *base, int idx) {
    return reinterpret_cast<const _Float16 *>(
        reinterpret_cast<const char *>(base) +
        static_cast<unsigned>(idx < 0 ? 0 : idx) *
            static_cast<unsigned>(D * sizeof(_Float16)));
  };
  // Row wl of A-operand tile sj is candidate sj * 16 + wl.
  auto load_k = [&](frag_t(&kf)[N_TILES][DT], int idx) {
    const int other = swap_halves(idx);
#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      const _Float16 *k_row = row_at(k_base, sj == pair ? idx : other);
#pragma unroll
      for (int tk = 0; tk < DT; ++tk) {
        kf[sj][tk] = load_frag(k_row + tk * kTile, pair);
      }
    }
  };
  // Lane i loads candidate i's slice of V.
  auto load_v = [&](dw8_t(&vc)[V_CHUNKS], int idx) {
    const _Float16 *v_row = row_at(v_base, idx);
#pragma unroll
    for (int j = 0; j < V_CHUNKS; ++j) {
      vc[j] = *reinterpret_cast<const dw8_t *>(v_row + j * kTile);
    }
  };
  // Candidate (lane) position in a V^T row: the K order of the P^T operand.
  // On gfx11 each 16-candidate block of a row holds the 8 candidates of the
  // row's parity first (see KvStage in gqa_kernel.hip).
#if HIPDNN_WMMA_FRAG_ELEMS != 8
  const int kk = lane & 15;
  const int pos_even = (lane & 16) | ((kk & 1) << 3) | (kk >> 1);
  const int pos_odd = (lane & 16) | (((kk & 1) ^ 1) << 3) | (kk >> 1);
#else
  const int pos_even = lane;
  const int pos_odd = lane;
#endif

  float m_run = -FLT_MAX;
  float l_run = 0.0f;
  acc_t o_acc[DT];
#pragma unroll
  for (int tj = 0; tj < DT; ++tj) {
    o_acc[tj] = acc_t{};
  }

  frag_t kf[N_TILES][DT];
  // Dwords rather than half16, like pb below.
  dw8_t vc[V_CHUNKS];
  int idx_cur = load_idx(0);
  int idx_nxt = load_idx(1);
  load_k(kf, idx_cur);
  load_v(vc, idx_cur);

  for (int t = 0; t < num_tiles; ++t) {
    acc_t s_acc[N_TILES];
#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      s_acc[sj] = acc_t{};
    }
#pragma unroll
    for (int f = 0; f < N_TILES * DT; ++f) {
      const int sj = f % N_TILES;
      const int tk = f / N_TILES;
      s_acc[sj] = DSA_WMMA(kf[sj][tk], q_reg[tk], s_acc[sj]);
#if HIPDNN_HAS_WMMA
      __builtin_amdgcn_sched_barrier(0);
#endif
    }

    if (ND > 1) {
#pragma unroll
      for (int sj = 0; sj < N_TILES; ++sj) {
        *reinterpret_cast<acc_t *>(&sp_lds[wave][sj][lane * 8]) = s_acc[sj];
      }
      __syncthreads();
    }
    if (t + 1 < num_tiles) {
      load_k(kf, idx_nxt);
    }
    const int idx_after = load_idx(t + 2);
    if (ND > 1) {
      // Every wave adds the partials in wave order, so all of them see the
      // same scores. Streamed two at a time: hoisted, the ND * N_TILES reads
      // would hold 8 * ND * N_TILES registers.
      acc_t sum[N_TILES];
      constexpr int R = 2;
      acc_t ring[R];
      auto read = [&](int f) {
        return *reinterpret_cast<const acc_t *>(
            &sp_lds[f / N_TILES][f % N_TILES][lane * 8]);
      };
#pragma unroll
      for (int f = 0; f < R; ++f) {
        ring[f] = read(f);
      }
#pragma unroll
      for (int f = 0; f < ND * N_TILES; ++f) {
        const acc_t x = ring[f % R];
        if (f + R < ND * N_TILES) {
          ring[f % R] = read(f + R);
        }
        const int sj = f % N_TILES;
        sum[sj] = f < N_TILES ? x : sum[sj] + x;
#if HIPDNN_HAS_WMMA
        __builtin_amdgcn_sched_barrier(0);
#endif
      }
#pragma unroll
      for (int sj = 0; sj < N_TILES; ++sj) {
        s_acc[sj] = sum[sj];
      }
      __syncthreads();
    }

    const unsigned vmask = __builtin_amdgcn_ballot_w32(idx_cur >= 0);
    float tmax = -INFINITY;
#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int key = sj * kTile + hipdnn_wmma_acc_row(pair, e);
        const float x = (vmask >> key) & 1u ? s_acc[sj][e] : -INFINITY;
        s_acc[sj][e] = x;
        tmax = fmaxf(tmax, x);
      }
    }
    tmax = fmaxf(tmax, swap_halves(tmax)) * sc;
    // Every lane of all ND waves sees the same scores, so the decision is
    // uniform across the block.
    if (__builtin_amdgcn_ballot_w32(tmax > m_run + kRescaleSlack) != 0) {
      const float m_new = fmaxf(m_run, tmax);
      // Exponent <= 0: the raw v_exp_f32 needs none of exp2f's denormal
      // fixup (see the GQA core).
      const float corr = __builtin_amdgcn_exp2f(m_run - m_new);
      m_run = m_new;
      l_run *= corr;
#pragma unroll
      for (int tj = 0; tj < DT; ++tj) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          o_acc[tj][e] *= corr;
        }
      }
    }
    float tsum = 0.0f;
#if HIPDNN_WMMA_FRAG_ELEMS == 8
    frag_t pb[N_TILES];
#else
    // Kept as dwords up to the WMMA: a half16 P^T carried across the V
    // masking branch below gets every odd element rebuilt at the join (see
    // Pb in the GQA core).
    dw8_t pb[N_TILES];
#endif
#pragma unroll
    for (int sj = 0; sj < N_TILES; ++sj) {
      _Float16 own[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        // Exponent <= kRescaleSlack.
        const float p = __builtin_amdgcn_exp2f(fmaf(s_acc[sj][e], sc, -m_run));
        tsum += p;
        own[e] = static_cast<_Float16>(p);
      }
#if HIPDNN_WMMA_FRAG_ELEMS == 8
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        pb[sj][e] = own[e];
      }
#else
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        pb[sj][j] =
            __builtin_bit_cast(int, half2v_t{own[2 * j], own[2 * j + 1]});
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        pb[sj][4 + j] = swap_halves(pb[sj][j]);
      }
#endif
    }
    tsum += swap_halves(tsum);
    l_run += tsum;

    // A masked p of 0 must not meet a non-finite V value of the row-0
    // stand-in (0 * Inf).
    if (vmask != 0xffffffffu) {
#pragma unroll
      for (int j = 0; j < V_CHUNKS; ++j) {
        vc[j] = idx_cur >= 0 ? vc[j] : dw8_t{};
      }
    }
#pragma unroll
    for (int j = 0; j < V_CHUNKS; ++j) {
#pragma unroll
      for (int e = 0; e < kTile; ++e) {
        // Shift and mask: ROCm 7.16 compiles a __builtin_bit_cast of the
        // vector element vc[j][e / 2] as a read of vc[0][0].
        const unsigned w = static_cast<unsigned>(vc[j][e / 2]);
        vt[(j * kTile + e) * VT_STR + ((e & 1) ? pos_odd : pos_even)] =
            __builtin_bit_cast(_Float16, static_cast<unsigned short>(
                                             (e & 1) ? w >> 16 : w & 0xffffu));
      }
    }
    wave_lds_sync();

    const _Float16 *vf_base = vt + wl * VT_STR;
    frag_stream<N_TILES * DT, 2>(
        [&](int f) {
          return load_frag(
              vf_base + (f % DT) * kTile * VT_STR + (f / DT) * kTile, pair);
        },
        [&](int f, const frag_t &vf) {
          o_acc[f % DT] = DSA_WMMA(vf, __builtin_bit_cast(frag_t, pb[f / DT]),
                                   o_acc[f % DT]);
        });
    wave_lds_sync();
#if HIPDNN_HAS_WMMA
    // Keeps the next tile's V loads out of the value GEMM, where their
    // registers would overlap the P^T and V^T fragments and spill.
    __builtin_amdgcn_sched_barrier(0);
#endif
    if (t + 1 < num_tiles) {
      load_v(vc, idx_nxt);
    }
    idx_cur = idx_nxt;
    idx_nxt = idx_after;
  }

  if (wl >= hpg) {
    return;
  }
  const int d0 = wave * DS;
  if (splits == 1) {
    const float inv = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    _Float16 *o_row =
        output + (static_cast<size_t>(row) * num_heads + g * hpg + wl) * D + d0;
#pragma unroll
    for (int tj = 0; tj < DT; ++tj) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        o_row[tj * kTile + hipdnn_wmma_acc_row(pair, e)] =
            static_cast<_Float16>(o_acc[tj][e] * inv);
      }
    }
    return;
  }
  float *pd =
      partials +
      ((static_cast<size_t>(row) * num_heads + g * hpg + wl) * splits + split) *
          (D + 2);
#pragma unroll
  for (int tj = 0; tj < DT; ++tj) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      pd[2 + d0 + tj * kTile + hipdnn_wmma_acc_row(pair, e)] = o_acc[tj][e];
    }
  }
  if (wave == 0 && pair == 0) {
    pd[0] = m_run;
    pd[1] = l_run;
  }
}


template <bool IS_MAX>
__device__ __forceinline__ float blockReduce(float x, float *red) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float y = __shfl_xor(x, off);
    x = IS_MAX ? fmaxf(x, y) : x + y;
  }
  const int wave = threadIdx.x / 32;
  const int waves = blockDim.x / 32;
  __syncthreads();
  if (threadIdx.x % 32 == 0) {
    red[wave] = x;
  }
  __syncthreads();
  x = red[0];
  for (int w = 1; w < waves; ++w) {
    x = IS_MAX ? fmaxf(x, red[w]) : x + red[w];
  }
  return x;
}

// One block per (query row, head). Thread k < splits owns split k's (m, l);
// thread d then merges column d of every split.
template <int D>
__global__ void __launch_bounds__(D)
dsa_combine_kernel(const float *__restrict__ partials,
                   _Float16 *__restrict__ output, int splits) {
  static_assert(D >= kMaxSplits, "one thread per split");
  __shared__ float w_lds[kMaxSplits];
  __shared__ float red[D / 32];
  const float *base =
      partials + static_cast<size_t>(blockIdx.x) * splits * (D + 2);
  const int t = threadIdx.x;
  const bool owner = t < splits;
  const float m_k = owner ? base[static_cast<size_t>(t) * (D + 2)] : -FLT_MAX;
  const float l_k = owner ? base[static_cast<size_t>(t) * (D + 2) + 1] : 0.0f;
  const float gm = blockReduce<true>(m_k, red);
  const float w = owner ? exp2f(m_k - gm) : 0.0f;
  if (owner) {
    w_lds[t] = w;
  }
  const float l = blockReduce<false>(w * l_k, red);
  float o = 0.0f;
#pragma unroll 4
  for (int k = 0; k < splits; ++k) {
    o += w_lds[k] * base[static_cast<size_t>(k) * (D + 2) + 2 + t];
  }
  output[static_cast<size_t>(blockIdx.x) * D + t] =
      static_cast<_Float16>(l > 0.0f ? o / l : 0.0f);
}

int numSplits(int64_t rows, int64_t kv_heads, int64_t max_selected) {
  const int64_t blocks = rows * kv_heads;
  if (blocks >= kTargetBlocks || max_selected <= kBkv) {
    return 1;
  }
  int64_t splits = (kTargetBlocks + blocks - 1) / blocks;
  const int64_t useful = (max_selected + kBkv - 1) / kBkv;
  if (splits > useful) {
    splits = useful;
  }
  if (splits > kMaxSplits) {
    splits = kMaxSplits;
  }
  return static_cast<int>(splits);
}

bool aligned16(const void *p) {
  return (reinterpret_cast<uintptr_t>(p) & 15u) == 0;
}

template <int D, int ND>
hipError_t launchAttention(hipStream_t stream, const void *query,
                           const void *present_key, const void *present_value,
                           const int32_t *selected_indices,
                           const int32_t *selected_counts,
                           const int32_t *seqlens_k, void *output,
                           float *partials, int rows, int seq_len,
                           int num_heads, int kv_heads, int cap, int max_sel,
                           int splits, float scale) {
  const dim3 grid(rows, kv_heads, splits);
  dsa_attention_kernel<D, ND><<<grid, 32 * ND, 0, stream>>>(
      static_cast<const _Float16 *>(query),
      static_cast<const _Float16 *>(present_key),
      static_cast<const _Float16 *>(present_value), selected_indices,
      selected_counts, seqlens_k, static_cast<_Float16 *>(output), partials,
      seq_len, num_heads, kv_heads, cap, max_sel, splits, scale * kLog2e);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess || splits == 1) {
    return err;
  }
  dsa_combine_kernel<D><<<rows * num_heads, D, 0, stream>>>(
      partials, static_cast<_Float16 *>(output), splits);
  return hipGetLastError();
}

} // namespace

extern "C" size_t hip_dynamic_sparse_attention_workspace_size(
    int64_t batch, int64_t seq_len, int64_t num_heads, int64_t kv_num_heads,
    int64_t head_size, int64_t max_selected) {
  const int splits = numSplits(batch * seq_len, kv_num_heads, max_selected);
  if (splits == 1) {
    return 0;
  }
  return static_cast<size_t>(batch * seq_len * num_heads) * splits *
         (head_size + 2) * sizeof(float);
}

extern "C" int hip_dynamic_sparse_attention(
    void *stream, const void *query, const void *key, const void *value,
    const void *past_key, const void *past_value,
    const int32_t *selected_indices, const int32_t *selected_counts,
    const int32_t *seqlens_k, void *output, void *present_key,
    void *present_value, void *workspace, size_t workspace_bytes,
    int64_t batch, int64_t seq_len, int64_t num_heads, int64_t kv_num_heads,
    int64_t head_size, int64_t capacity, int64_t max_selected, float scale) {
  if (!query || !key || !value || !past_key || !past_value ||
      !selected_counts || !seqlens_k || !output || !present_key ||
      !present_value || (max_selected > 0 && !selected_indices)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: null pointer\n");
    return -1;
  }
  if (batch <= 0 || seq_len <= 0 || kv_num_heads <= 0 || capacity <= 0 ||
      max_selected < 0 || num_heads % kv_num_heads != 0 ||
      num_heads / kv_num_heads < 1 || num_heads / kv_num_heads > kTile ||
      (head_size != 128 && head_size != 256)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: unsupported shape "
            "B=%lld S=%lld N=%lld kvN=%lld D=%lld cap=%lld max_selected=%lld\n",
            (long long)batch, (long long)seq_len, (long long)num_heads,
            (long long)kv_num_heads, (long long)head_size, (long long)capacity,
            (long long)max_selected);
    return -1;
  }
  if (!aligned16(query) || !aligned16(key) || !aligned16(value) ||
      !aligned16(past_key) || !aligned16(past_value) ||
      !aligned16(present_key) || !aligned16(present_value)) {
    fprintf(stderr, "[custom_kernels] hip_dynamic_sparse_attention: Q/K/V and "
                    "caches must be 16-byte aligned\n");
    return -1;
  }
  if (hipdnn_device_wave_size() != 32) {
    fprintf(stderr, "[custom_kernels] hip_dynamic_sparse_attention: requires "
                    "a wave32 device with WMMA\n");
    return -1;
  }
  const int64_t rows = batch * seq_len;
  const int splits = numSplits(rows, kv_num_heads, max_selected);
  const size_t needed = hip_dynamic_sparse_attention_workspace_size(
      batch, seq_len, num_heads, kv_num_heads, head_size, max_selected);
  if (needed > workspace_bytes || (needed > 0 && !workspace)) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: workspace %zu "
            "bytes, need %zu\n",
            workspace_bytes, needed);
    return -1;
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const size_t cache_bytes = static_cast<size_t>(batch * kv_num_heads) *
                             capacity * head_size * sizeof(_Float16);
  hipError_t err = hipSuccess;
  // Full-cache copy only when past and present are distinct buffers;
  // shared-buffer decode skips it and appends in place.
  if (past_key != present_key) {
    err = hipMemcpyAsync(present_key, past_key, cache_bytes,
                         hipMemcpyDeviceToDevice, s);
  }
  if (err == hipSuccess && past_value != present_value) {
    err = hipMemcpyAsync(present_value, past_value, cache_bytes,
                         hipMemcpyDeviceToDevice, s);
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: past copy failed "
            "(%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  const int chunks_per_head = static_cast<int>(head_size / 8);
  const int64_t per_batch = seq_len * kv_num_heads * chunks_per_head;
  dsa_kv_append_kernel<<<dim3(static_cast<unsigned>(
                                  (per_batch + kAppendThreads - 1) /
                                  kAppendThreads),
                              static_cast<unsigned>(batch)),
                         kAppendThreads, 0, s>>>(
      static_cast<const vec16_t *>(key), static_cast<const vec16_t *>(value),
      static_cast<vec16_t *>(present_key), static_cast<vec16_t *>(present_value),
      seqlens_k, static_cast<int>(seq_len), static_cast<int>(kv_num_heads),
      chunks_per_head, static_cast<int>(capacity));
  err = hipGetLastError();

  if (err == hipSuccess) {
    float *partials = static_cast<float *>(workspace);
    if (head_size == 256) {
      err = launchAttention<256, 4>(
          s, query, present_key, present_value, selected_indices,
          selected_counts, seqlens_k, output, partials, static_cast<int>(rows),
          static_cast<int>(seq_len), static_cast<int>(num_heads),
          static_cast<int>(kv_num_heads), static_cast<int>(capacity),
          static_cast<int>(max_selected), splits, scale);
    } else {
      err = launchAttention<128, 2>(
          s, query, present_key, present_value, selected_indices,
          selected_counts, seqlens_k, output, partials, static_cast<int>(rows),
          static_cast<int>(seq_len), static_cast<int>(num_heads),
          static_cast<int>(kv_num_heads), static_cast<int>(capacity),
          static_cast<int>(max_selected), splits, scale);
    }
  }
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_dynamic_sparse_attention: launch failed "
            "(%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
