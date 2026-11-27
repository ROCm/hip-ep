/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
// Included by gemm_kernel.hip after gemm_batched_impl.h.
//
// General dense GEMM family selected only by exact offline-LUT points
// (config family >= 1). Each kernel covers Y[b] = alpha * A[b] * op(B[b])
// (+ a row-broadcast C when beta == 1), row-major A[M][K], B[K][N] (NN) or
// B[N][K] (NT), tight leading dimensions and element batch strides.
//
//  wmma   F16 WMMA tiles, four-wave or one-row-of-waves workgroups. A single
//         LDS buffer holds tile t while tile t+1 waits in registers (two
//         barriers per K tile). Buffer loads read zero past the last row, the
//         final K tile is masked, the epilogue is staged in LDS for 16-byte
//         stores. NN keeps B in its natural [k][n] layout. Optional K split
//         into F32 partials (deterministic reduction kernel) and K-start
//         staggering for row strides that alias DRAM channels.
//  gemv   F16 NN, M == 1: whole K per workgroup, A staged in LDS, B rows read
//         in 16-byte segments, cross-wave LDS reduction; one launch.
//  f32*   FP32 FMA kernels: per-lane K slices reduced by shuffles (dot), B in
//         LDS with interleaved A rows (rows), whole K in LDS (lds) and a
//         register-blocked LDS tile with optional K split / staggering (tile).
//
// Every launch, grid and workspace offset carries the batch and split
// explicitly. No device allocation; split partials use the caller workspace.

#include <climits>
#include <cstdint>

namespace gemm_gen {

typedef _Float16 h16v __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef float f4v __attribute__((ext_vector_type(4)));
typedef float f2v __attribute__((ext_vector_type(2)));

struct Args {
  const void *a, *b, *c;
  void *y;
  int m, n, k;
  int lda, ldb, ldy;
  int64_t sa, sb, sc, sy;  // batch strides in elements
  float alpha;
  int bias;                // 1: Y += C[col], C a row-broadcast vector
  int group_m;             // M tiles per raster band
  int stagger_mod;         // > 1: rotate each workgroup's first K tile
  int stagger_tiles;       // K tiles between rotated starts
  int splits;              // > 1: K split over blockIdx.z, F32 partials in ws
  float *ws;               // [batch][split][m][n]
};

#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
#define GEMM_GEN_BUFFER 1
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void *p, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), 0, bytes, 0x31004000);
}
#else
#define GEMM_GEN_BUFFER 0
#endif

// Zero the halves of a 16-byte chunk whose K index (kc + e) >= rem.
__device__ __forceinline__ u4v maskTail(u4v v, int kc, int rem) {
  if (kc + 8 <= rem) return v;
  if (kc >= rem) return u4v{0, 0, 0, 0};
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const int e = kc + 2 * d;
    uint32_t w = v[d];
    if (e >= rem) w = 0;
    else if (e + 1 >= rem) w &= 0xffffu;
    v[d] = w;
  }
  return v;
}

template <int BM, int BN, int BK, int WGM, int WGN, bool TB>
struct WmmaGeom {
  static constexpr int THR = WGM * WGN * 32;
  static constexpr int TM = BM / (WGM * 16);
  static constexpr int TN = BN / (WGN * 16);
  static constexpr int FE = HIPDNN_WMMA_FRAG_ELEMS;
  static constexpr int KP = BK + 8;                     // padded K row (halves)
  static constexpr int A_HALVES = BM * KP;
  static constexpr int B_HALVES = TB ? BN * KP : BK * BN + (BK / 16) * 8;
  static constexpr int A_TOT = BM * BK / 8, B_TOT = BN * BK / 8;  // 16 B chunks
  static constexpr int A_CH = (A_TOT + THR - 1) / THR;
  static constexpr int B_CH = (B_TOT + THR - 1) / THR;
  static constexpr int STAGE = THR / 32 * (TM * 16) * (TN * 16 + 8);
  static constexpr int LDS_HALVES =
      A_HALVES + B_HALVES > STAGE ? A_HALVES + B_HALVES : STAGE;
  static_assert(BM % (WGM * 16) == 0 && BN % (WGN * 16) == 0, "wave tile");
  static_assert(BK % 16 == 0, "BK");
  static_assert(LDS_HALVES * 2 <= 65536, "LDS");
};

template <int BM, int BN, int BK, int WGM, int WGN, bool TB>
__global__ void __launch_bounds__(WGM * WGN * 32)
gemm_gen_wmma_kernel(Args p) {
#if GEMM_GEN_BUFFER
  using T = _Float16;
  using C = WmmaGeom<BM, BN, BK, WGM, WGN, TB>;
  constexpr int THR = C::THR, TM = C::TM, TN = C::TN, FE = C::FE, KP = C::KP;
  __shared__ __align__(16) T lds[C::LDS_HALVES];
  T *sA = lds;
  T *sB = lds + C::A_HALVES;

  const int tid = threadIdx.x;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int lane = tid & 15, sub = (tid >> 4) & 1;
  const int wy = wave / WGN, wx = wave % WGN;

  const int tiles_m = (p.m + BM - 1) / BM, tiles_n = (p.n + BN - 1) / BN;
  const int pid = blockIdx.x;
  const int band = p.group_m * tiles_n;
  const int first_m = (pid / band) * p.group_m;
  const int gsz = min(tiles_m - first_m, p.group_m);
  const int tile_m = first_m + (pid % band) % gsz;
  const int tile_n = (pid % band) / gsz;
  const int row0 = tile_m * BM, col0 = tile_n * BN;
  const int splits = p.splits > 1 ? p.splits : 1;
  const int batch = blockIdx.z / splits, split = blockIdx.z % splits;

  const T *A = static_cast<const T *>(p.a) + batch * p.sa + (int64_t)row0 * p.lda;
  const T *B = static_cast<const T *>(p.b) + batch * p.sb +
               (TB ? (int64_t)col0 * p.ldb : (int64_t)col0);
  const uint32_t a_bytes = (uint32_t)((int64_t)(p.m - row0) * p.lda * 2);
  const uint32_t b_bytes = TB ? (uint32_t)((int64_t)(p.n - col0) * p.ldb * 2)
                              : (uint32_t)(((int64_t)p.k * p.ldb - col0) * 2);
  const auto ra = rsrc(A, a_bytes);
  const auto rb = rsrc(B, b_bytes);

  constexpr int ACPR = BK / 8;
  int a_vo[C::A_CH], a_lds[C::A_CH], a_kc[C::A_CH];
#pragma unroll
  for (int i = 0; i < C::A_CH; ++i) {
    const int q = tid + i * THR;
    const int r = q / ACPR, kc = (q % ACPR) * 8;
    a_vo[i] = (r * p.lda + kc) * 2;
    a_lds[i] = r * KP + kc;
    a_kc[i] = kc;
  }
  int b_vo[C::B_CH], b_lds[C::B_CH], b_kc[C::B_CH];
#pragma unroll
  for (int i = 0; i < C::B_CH; ++i) {
    const int q = tid + i * THR;
    if constexpr (TB) {
      const int r = q / ACPR, kc = (q % ACPR) * 8;
      b_vo[i] = (r * p.ldb + kc) * 2;
      b_lds[i] = r * KP + kc;
      b_kc[i] = kc;
    } else {
      constexpr int BCPR = BN / 8;
      const int kk = q / BCPR, nc = (q % BCPR) * 8;
      b_vo[i] = (kk * p.ldb + nc) * 2;
      b_lds[i] = kk * BN + (kk / 16) * 8 + nc;
      b_kc[i] = kk;
    }
  }

  u4v ra_reg[C::A_CH], rb_reg[C::B_CH];
  auto aon = [&](int i) __attribute__((always_inline)) {
    return C::A_TOT % THR == 0 || i < C::A_TOT / THR || tid + i * THR < C::A_TOT;
  };
  auto bon = [&](int i) __attribute__((always_inline)) {
    return C::B_TOT % THR == 0 || i < C::B_TOT / THR || tid + i * THR < C::B_TOT;
  };
  const int nt_all = (p.k + BK - 1) / BK;
  const int kt0 = (int)((int64_t)split * nt_all / splits);
  const int nt = (int)((int64_t)(split + 1) * nt_all / splits) - kt0;
  const int a_step = BK * 2;
  const int b_step = TB ? BK * 2 : BK * p.ldb * 2;

  auto gload = [&](int t) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < C::A_CH; ++i)
      if (aon(i))
        ra_reg[i] = __builtin_bit_cast(
            u4v, __builtin_amdgcn_raw_buffer_load_b128(ra, a_vo[i], t * a_step, 0));
#pragma unroll
    for (int i = 0; i < C::B_CH; ++i)
      if (bon(i))
        rb_reg[i] = __builtin_bit_cast(
            u4v, __builtin_amdgcn_raw_buffer_load_b128(rb, b_vo[i], t * b_step, 0));
  };
  auto mask = [&](int t) __attribute__((always_inline)) {
    const int rem = p.k - t * BK;
    if (rem >= BK) return;
#pragma unroll
    for (int i = 0; i < C::A_CH; ++i) ra_reg[i] = maskTail(ra_reg[i], a_kc[i], rem);
    if constexpr (TB) {
#pragma unroll
      for (int i = 0; i < C::B_CH; ++i) rb_reg[i] = maskTail(rb_reg[i], b_kc[i], rem);
    }
    // NN: rows k >= K are outside the B buffer range and read zero.
  };
  auto lstore = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < C::A_CH; ++i)
      if (aon(i)) *reinterpret_cast<u4v *>(sA + a_lds[i]) = ra_reg[i];
#pragma unroll
    for (int i = 0; i < C::B_CH; ++i)
      if (bon(i)) *reinterpret_cast<u4v *>(sB + b_lds[i]) = rb_reg[i];
  };

  f8v acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = f8v{};

  const int koff = hipdnn_wmma_k_off(sub);
  auto compute = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      h16v bf[TN], af[TM];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = (j * WGN + wx) * 16 + lane;
        if constexpr (TB) {
          const T *src = sB + n * KP + ks * 16 + koff;
#pragma unroll
          for (int v = 0; v < FE / 8; ++v)
            reinterpret_cast<u4v *>(&bf[j])[v] = reinterpret_cast<const u4v *>(src)[v];
        } else {
          const T *src = sB + (ks * 16 + koff) * BN + ks * 8 + n;
#pragma unroll
          for (int e = 0; e < FE; ++e) bf[j][e] = src[e * BN];
        }
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int r = (i * WGM + wy) * 16 + lane;
        const T *src = sA + r * KP + ks * 16 + koff;
#pragma unroll
        for (int v = 0; v < FE / 8; ++v)
          reinterpret_cast<u4v *>(&af[i])[v] = reinterpret_cast<const u4v *>(src)[v];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = HIPDNN_WMMA_F32_16X16X16_F16(af[i], bf[j], acc[i][j]);
    }
  };

  if (nt > 0) {
    // Visit K tiles from a rotated start; the masked tail keeps its index.
    int st = 0;
    if (p.stagger_mod > 1) st = ((pid % p.stagger_mod) * p.stagger_tiles) % nt;
    auto tidx = [&](int u) __attribute__((always_inline)) {
      const int t = u + st;
      return kt0 + (t >= nt ? t - nt : t);
    };
    auto maskIf = [&](int t) __attribute__((always_inline)) {
      if (t == nt_all - 1) mask(t);
    };
    gload(tidx(0));
    maskIf(tidx(0));
    lstore();
    if (nt > 1) gload(tidx(1));
    HIPDNN_LDS_BARRIER();
    for (int u = 0; u < nt; ++u) {
      compute();
      HIPDNN_LDS_BARRIER();
      if (u + 1 < nt) {
        maskIf(tidx(u + 1));
        lstore();
        if (u + 2 < nt) gload(tidx(u + 2));
        HIPDNN_LDS_BARRIER();
      }
    }
  }

  if (splits > 1) {
    // Raw F32 partial; alpha, bias and conversion happen in the reduction.
    float *W = p.ws + ((int64_t)batch * splits + split) * p.m * p.n;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gc = col0 + (j * WGN + wx) * 16 + lane;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int gr = row0 + (i * WGM + wy) * 16 + hipdnn_wmma_acc_row(sub, e);
          if (gr < p.m && gc < p.n) W[(int64_t)gr * p.n + gc] = acc[i][j][e];
        }
    }
    return;
  }

  // Epilogue: stage each wave's converted tile in LDS, then 16-byte rows.
  constexpr int WR = TM * 16, WC = TN * 16, WCP = WC + 8;
  T *stage = lds + wave * WR * WCP;
  const T *bias = p.bias ? static_cast<const T *>(p.c) + batch * p.sc : nullptr;
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int lc = j * 16 + lane;
    const int gc = col0 + (j * WGN + wx) * 16 + lane;
    const float bv = bias && gc < p.n ? (float)bias[gc] : 0.f;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int lr = i * 16 + hipdnn_wmma_acc_row(sub, e);
        stage[lr * WCP + lc] = (T)(acc[i][j][e] * p.alpha + bv);
      }
  }
  HIPDNN_LDS_BARRIER();
  T *Y = static_cast<T *>(p.y) + batch * p.sy;
  const int lid = tid & 31;
  constexpr int CPR = 2;  // 16-byte chunks per 16-column group
  constexpr int CH = WR * TN * CPR;
  const bool vec = (p.ldy % 8 == 0) && ((reinterpret_cast<uintptr_t>(Y) & 15) == 0);
#pragma unroll
  for (int q0 = 0; q0 < CH; q0 += 32) {
    const int q = q0 + lid;
    const int lr = q / (TN * CPR);
    const int rem = q % (TN * CPR);
    const int j = rem / CPR, h = rem % CPR;
    const int gr = row0 + ((lr / 16) * WGM + wy) * 16 + (lr % 16);
    const int gc = col0 + (j * WGN + wx) * 16 + h * 8;
    const u4v v = *reinterpret_cast<const u4v *>(stage + lr * WCP + j * 16 + h * 8);
    if (gr < p.m) {
      T *dst = Y + (int64_t)gr * p.ldy + gc;
      if (vec && gc + 8 <= p.n) {
        *reinterpret_cast<u4v *>(dst) = v;
      } else {
        const T *vals = reinterpret_cast<const T *>(&v);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          if (gc + e < p.n) dst[e] = vals[e];
      }
    }
  }
#else
  (void)p;
  __builtin_trap();
#endif
}

// Sum split partials in fixed split order, then alpha, bias and conversion.
template <class T>
__global__ void __launch_bounds__(256) gemm_gen_reduce_kernel(Args p) {
  const int batch = blockIdx.z;
  const int64_t count = (int64_t)p.m * p.n;
  const int64_t q = ((int64_t)blockIdx.x * 256 + threadIdx.x) * 4;
  if (q >= count) return;
  const float *W = p.ws + (int64_t)batch * p.splits * count;
  T *Y = static_cast<T *>(p.y) + batch * p.sy;
  const T *bias = p.bias ? static_cast<const T *>(p.c) + batch * p.sc : nullptr;
  const bool full = q + 4 <= count && (count & 3) == 0;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
  if (full) {
    for (int s = 0; s < p.splits; ++s) {
      const f4v x = *reinterpret_cast<const f4v *>(W + s * count + q);
      v[0] += x[0]; v[1] += x[1]; v[2] += x[2]; v[3] += x[3];
    }
  } else {
    for (int s = 0; s < p.splits; ++s)
      for (int e = 0; e < 4; ++e)
        if (q + e < count) v[e] += W[s * count + q + e];
  }
  for (int e = 0; e < 4; ++e) {
    if (q + e >= count) break;
    const int64_t idx = q + e;
    const int r = (int)(idx / p.n), c = (int)(idx % p.n);
    float x = v[e] * p.alpha;
    if (bias) x += GemmElem<T>::tof(bias[c]);
    Y[(int64_t)r * p.ldy + c] = GemmElem<T>::fromf(x);
  }
}

// F16 NN GEMV, M == 1: COLS columns per workgroup, whole K.
template <int COLS, int THREADS, int U>
__global__ void __launch_bounds__(THREADS) gemm_gen_gemv_kernel(Args p) {
#if GEMM_GEN_BUFFER
  using T = _Float16;
  constexpr int LPR = COLS / 8;         // lanes per 16-byte B row segment
  constexpr int RPI = THREADS / LPR;    // rows per load instruction
  static_assert(COLS % 8 == 0 && LPR <= 32 && 32 % LPR == 0, "geometry");
  extern __shared__ __align__(16) unsigned char smem[];
  float *sA = reinterpret_cast<float *>(smem);
  const int tid = threadIdx.x, batch = blockIdx.z, K = p.k;
  const T *A = static_cast<const T *>(p.a) + batch * p.sa;
  const T *B = static_cast<const T *>(p.b) + batch * p.sb;
  for (int i = tid; i < K; i += THREADS) sA[i] = (float)A[i];
  __syncthreads();
  const int c0 = blockIdx.x * COLS + (tid % LPR) * 8;
  const int r0 = tid / LPR;
  const bool full = c0 + 8 <= p.n;
  const int cb = c0 < p.n ? c0 : 0;
  const auto rb = rsrc(B + cb, (uint32_t)(((int64_t)K * p.ldb - cb) * 2));
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  auto fma8 = [&](const u4v &v, int k) __attribute__((always_inline)) {
    const T *h = reinterpret_cast<const T *>(&v);
    const float a = sA[k];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = __builtin_fmaf(a, (float)h[e], acc[e]);
  };
  int k = r0;
  if (full) {
    for (; k + (U - 1) * RPI < K; k += U * RPI) {
      u4v v[U];
#pragma unroll
      for (int u = 0; u < U; ++u)
        v[u] = __builtin_bit_cast(
            u4v, __builtin_amdgcn_raw_buffer_load_b128(rb, (k + u * RPI) * p.ldb * 2, 0, 0));
#pragma unroll
      for (int u = 0; u < U; ++u) fma8(v[u], k + u * RPI);
    }
    for (; k < K; k += RPI)
      fma8(__builtin_bit_cast(u4v, __builtin_amdgcn_raw_buffer_load_b128(rb, k * p.ldb * 2, 0, 0)), k);
  } else if (c0 < p.n) {
    for (; k < K; k += RPI) {
      T h[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) h[e] = c0 + e < p.n ? B[(int64_t)k * p.ldb + c0 + e] : T{};
      u4v v;
      __builtin_memcpy(&v, h, 16);
      fma8(v, k);
    }
  }
  // Lanes sharing a column segment are LPR apart: shuffle, then one row per
  // wave in LDS and a final sum over waves.
  __syncthreads();
  float *red = reinterpret_cast<float *>(smem);
  constexpr int NW = THREADS / 32;
  const int lane = tid & 31, wave = tid >> 5;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    float x = acc[e];
#pragma unroll
    for (int off = 16; off >= LPR; off >>= 1) x += __shfl_xor(x, off, 32);
    acc[e] = x;
  }
  if (lane < LPR) {
#pragma unroll
    for (int e = 0; e < 8; ++e) red[wave * COLS + lane * 8 + e] = acc[e];
  }
  __syncthreads();
  const T *bias = p.bias ? static_cast<const T *>(p.c) + batch * p.sc : nullptr;
  T *Y = static_cast<T *>(p.y) + batch * p.sy;
  for (int c = tid; c < COLS; c += THREADS) {
    const int gc = blockIdx.x * COLS + c;
    if (gc >= p.n) continue;
    float s = 0.f;
    for (int w = 0; w < NW; ++w) s += red[w * COLS + c];
    s *= p.alpha;
    if (bias) s += (float)bias[gc];
    Y[gc] = (T)s;
  }
#else
  (void)p;
  __builtin_trap();
#endif
}

// FP32: KP consecutive lanes split K (KS per lane) for TN outputs of one row;
// every load is issued before the FMAs, partials reduced by shuffles.
template <int TN, int KP, int KS, int THREADS>
__global__ void __launch_bounds__(THREADS) gemm_gen_f32dot_kernel(Args p) {
  constexpr int GROUPS = THREADS / KP;
  const int tid = threadIdx.x, batch = blockIdx.z;
  const int nb = (p.n + TN - 1) / TN;
  const int64_t g = (int64_t)blockIdx.x * GROUPS + tid / KP;
  const int kp = tid % KP;
  const bool live = g < (int64_t)p.m * nb;
  const int m = live ? (int)(g / nb) : 0;
  const int n0 = live ? (int)(g % nb) * TN : 0;
  const float *A = static_cast<const float *>(p.a) + batch * p.sa + (int64_t)m * p.lda;
  const float *B = static_cast<const float *>(p.b) + batch * p.sb;
  const int k0 = kp * KS;
  float av[KS], bv[KS][TN];
  const bool full = live && k0 + KS <= p.k;
  const bool vecA = full && KS % 4 == 0 && p.lda % 4 == 0 && p.sa % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(p.a) & 15) == 0;
  constexpr int BV = TN % 4 == 0 ? 4 : TN % 2 == 0 ? 2 : 1;
  const bool vecB = BV > 1 && full && n0 + TN <= p.n && p.ldb % BV == 0 && p.sb % BV == 0 &&
                    (reinterpret_cast<uintptr_t>(p.b) & (BV * 4 - 1)) == 0;
  if (vecA) {
#pragma unroll
    for (int i = 0; i < KS / 4; ++i) {
      const f4v v = *reinterpret_cast<const f4v *>(A + k0 + 4 * i);
      av[4 * i] = v[0]; av[4 * i + 1] = v[1]; av[4 * i + 2] = v[2]; av[4 * i + 3] = v[3];
    }
  } else {
#pragma unroll
    for (int i = 0; i < KS; ++i) av[i] = live && k0 + i < p.k ? A[k0 + i] : 0.f;
  }
  if (vecB) {
#pragma unroll
    for (int i = 0; i < KS; ++i) {
      const float *src = B + (int64_t)(k0 + i) * p.ldb + n0;
#pragma unroll
      for (int j = 0; j < TN; j += BV) {
        if constexpr (BV == 4) {
          const f4v v = *reinterpret_cast<const f4v *>(src + j);
          bv[i][j] = v[0]; bv[i][j + 1] = v[1]; bv[i][j + 2] = v[2]; bv[i][j + 3] = v[3];
        } else if constexpr (BV == 2) {
          const f2v v = *reinterpret_cast<const f2v *>(src + j);
          bv[i][j] = v[0]; bv[i][j + 1] = v[1];
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < KS; ++i) {
      const int k = k0 + i;
      const bool ok = live && k < p.k;
#pragma unroll
      for (int j = 0; j < TN; ++j)
        bv[i][j] = ok && n0 + j < p.n ? B[(int64_t)k * p.ldb + n0 + j] : 0.f;
    }
  }
  float acc[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    float x = 0.f;
#pragma unroll
    for (int i = 0; i < KS; ++i) x = __builtin_fmaf(av[i], bv[i][j], x);
#pragma unroll
    for (int off = KP / 2; off >= 1; off >>= 1) x += __shfl_xor(x, off, 32);
    acc[j] = x;
  }
  if (!live || kp) return;
  float *Y = static_cast<float *>(p.y) + batch * p.sy + (int64_t)m * p.ldy;
  const float *bias = p.bias ? static_cast<const float *>(p.c) + batch * p.sc : nullptr;
#pragma unroll
  for (int j = 0; j < TN; ++j)
    if (n0 + j < p.n) Y[n0 + j] = acc[j] * p.alpha + (bias ? bias[n0 + j] : 0.f);
}

// FP32, N <= NMAX and K == KMAX: B in LDS as [n][k]; KP lanes own one output
// row, read its A row in interleaved float4s and reduce each column.
template <int NMAX, int KMAX, int KP, int THREADS>
__global__ void __launch_bounds__(THREADS) gemm_gen_f32rows_kernel(Args p) {
  constexpr int KS = KMAX / KP, BP = KMAX + 4;
  static_assert(KS % 4 == 0, "float4 slices");
  __shared__ __align__(16) float Bs[NMAX * BP];
  const int tid = threadIdx.x, batch = blockIdx.z;
  const float *B = static_cast<const float *>(p.b) + batch * p.sb;
  for (int i = tid; i < NMAX * BP; i += THREADS) {
    const int n = i / BP, k = i - n * BP;
    Bs[i] = n < p.n && k < p.k ? B[(int64_t)k * p.ldb + n] : 0.f;
  }
  const int64_t m = (int64_t)blockIdx.x * (THREADS / KP) + tid / KP;
  const int kp = tid % KP;
  const bool live = m < p.m;
  const float *A = static_cast<const float *>(p.a) + batch * p.sa + (live ? m : 0) * p.lda;
  auto kidx = [&](int i) __attribute__((always_inline)) {
    return 4 * ((i / 4) * KP + kp) + (i & 3);
  };
  float a[KS];
  const bool vec = live && p.lda % 4 == 0 && p.sa % 4 == 0 &&
                   (reinterpret_cast<uintptr_t>(p.a) & 15) == 0;
  if (vec) {
#pragma unroll
    for (int i = 0; i < KS / 4; ++i) {
      const f4v v = *reinterpret_cast<const f4v *>(A + kidx(4 * i));
      a[4 * i] = v[0]; a[4 * i + 1] = v[1]; a[4 * i + 2] = v[2]; a[4 * i + 3] = v[3];
    }
  } else {
#pragma unroll
    for (int i = 0; i < KS; ++i) a[i] = live ? A[kidx(i)] : 0.f;
  }
  __syncthreads();
  float *Y = static_cast<float *>(p.y) + batch * p.sy + (live ? m : 0) * p.ldy;
  const float *bias = p.bias ? static_cast<const float *>(p.c) + batch * p.sc : nullptr;
#pragma unroll
  for (int n = 0; n < NMAX; ++n) {
    if (n >= p.n) break;
    float x = 0.f;
#pragma unroll
    for (int i = 0; i < KS; ++i) x = __builtin_fmaf(a[i], Bs[n * BP + kidx(i)], x);
#pragma unroll
    for (int off = KP / 2; off >= 1; off >>= 1) x += __shfl_xor(x, off, 32);
    if (live && kp == 0) Y[n] = x * p.alpha + (bias ? bias[n] : 0.f);
  }
}

// FP32, K <= KMAX: one load round of an A row panel and B column panel into
// LDS, one barrier, TM x TN outputs per thread.
template <int BM, int BN, int KMAX, int TM, int TN, int THREADS>
__global__ void __launch_bounds__(THREADS) gemm_gen_f32lds_kernel(Args p) {
  constexpr int KP = KMAX + 1, BP = BN + 1;
  constexpr int TX = BN / TN, TY = BM / TM;
  static_assert(TX * TY <= THREADS, "thread tile");
  __shared__ float As[BM * KP];
  __shared__ float Bs[KMAX * BP];
  const int tid = threadIdx.x, batch = blockIdx.z;
  const int tiles_n = (p.n + BN - 1) / BN;
  const int row0 = (blockIdx.x / tiles_n) * BM, col0 = (blockIdx.x % tiles_n) * BN;
  const float *A = static_cast<const float *>(p.a) + batch * p.sa;
  const float *B = static_cast<const float *>(p.b) + batch * p.sb;
  const int K = p.k;
  for (int i = tid; i < BM * K; i += THREADS) {
    const int r = i / K, k = i - r * K;
    As[r * KP + k] = row0 + r < p.m ? A[(int64_t)(row0 + r) * p.lda + k] : 0.f;
  }
  for (int i = tid; i < BN * K; i += THREADS) {
    const int k = i / BN, c = i - k * BN;
    Bs[k * BP + c] = col0 + c < p.n ? B[(int64_t)k * p.ldb + col0 + c] : 0.f;
  }
  __syncthreads();
  if (tid >= TX * TY) return;
  const int tx = tid % TX, ty = tid / TX;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  for (int k = 0; k < K; ++k) {
    float a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = As[(ty + i * TY) * KP + k];
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = Bs[k * BP + tx + j * TX];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
  }
  float *Y = static_cast<float *>(p.y) + batch * p.sy;
  const float *bias = p.bias ? static_cast<const float *>(p.c) + batch * p.sc : nullptr;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = row0 + ty + i * TY;
    if (r >= p.m) continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = col0 + tx + j * TX;
      if (c < p.n) Y[(int64_t)r * p.ldy + c] = acc[i][j] * p.alpha + (bias ? bias[c] : 0.f);
    }
  }
}

// FP32 register-blocked LDS tile; next K tile in registers, two barriers per
// tile, optional K split into F32 partials and K-start staggering.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_gen_f32tile_kernel(Args p) {
  constexpr int TX = BN / TN, TY = BM / TM, THR = TX * TY;
  constexpr int LDA = BM + 4, LDB = BN + 4;
  constexpr int A4 = BM * BK / 4, B4 = BN * BK / 4;
  constexpr int AC = (A4 + THR - 1) / THR, BC = (B4 + THR - 1) / THR;
  static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0, "vector tiles");
  __shared__ __align__(16) float As[BK * LDA];
  __shared__ __align__(16) float Bs[BK * LDB];
  const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;
  const int splits = p.splits > 1 ? p.splits : 1;
  const int batch = blockIdx.z / splits, split = blockIdx.z % splits;
  const int tiles_n = (p.n + BN - 1) / BN;
  const int row0 = (blockIdx.x / tiles_n) * BM, col0 = (blockIdx.x % tiles_n) * BN;
  const float *A = static_cast<const float *>(p.a) + batch * p.sa;
  const float *B = static_cast<const float *>(p.b) + batch * p.sb;
  const int nt_all = (p.k + BK - 1) / BK;
  const int kt0 = (int)((int64_t)split * nt_all / splits);
  const int kt1 = (int)((int64_t)(split + 1) * nt_all / splits);
  const bool avec = p.lda % 4 == 0 && p.k % 4 == 0 && p.sa % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(p.a) & 15) == 0;
  const bool bvec = p.ldb % 4 == 0 && p.n % 4 == 0 && p.sb % 4 == 0 &&
                    (reinterpret_cast<uintptr_t>(p.b) & 15) == 0;
  f4v ra[AC], rb[BC];
  auto loadA = [&](int t) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < AC; ++i) {
      const int q = tid + i * THR;
      f4v v = {0.f, 0.f, 0.f, 0.f};
      if (A4 % THR == 0 || q < A4) {
        const int r = row0 + q / (BK / 4), k = t * BK + (q % (BK / 4)) * 4;
        if (r < p.m) {
          const float *src = A + (int64_t)r * p.lda + k;
          if (avec && k + 4 <= p.k) v = *reinterpret_cast<const f4v *>(src);
          else
            for (int e = 0; e < 4; ++e) if (k + e < p.k) v[e] = src[e];
        }
      }
      ra[i] = v;
    }
  };
  auto loadB = [&](int t) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < BC; ++i) {
      const int q = tid + i * THR;
      f4v v = {0.f, 0.f, 0.f, 0.f};
      if (B4 % THR == 0 || q < B4) {
        const int k = t * BK + q / (BN / 4), n = col0 + (q % (BN / 4)) * 4;
        if (k < p.k) {
          const float *src = B + (int64_t)k * p.ldb + n;
          if (bvec && n + 4 <= p.n) v = *reinterpret_cast<const f4v *>(src);
          else
            for (int e = 0; e < 4; ++e) if (n + e < p.n) v[e] = src[e];
        }
      }
      rb[i] = v;
    }
  };
  auto store = [&]() __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < AC; ++i) {
      const int q = tid + i * THR;
      if (A4 % THR == 0 || q < A4) {
        const int r = q / (BK / 4), k = (q % (BK / 4)) * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) As[(k + e) * LDA + r] = ra[i][e];
      }
    }
#pragma unroll
    for (int i = 0; i < BC; ++i) {
      const int q = tid + i * THR;
      if (B4 % THR == 0 || q < B4) {
        const int k = q / (BN / 4), n = (q % (BN / 4)) * 4;
        *reinterpret_cast<f4v *>(&Bs[k * LDB + n]) = rb[i];
      }
    }
  };
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
  // Each thread owns 4-wide row/column groups strided by TY*4 / TX*4.
  auto compute = [&]() __attribute__((always_inline)) {
#pragma unroll 4
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM / 4; ++i) {
        const f4v v = *reinterpret_cast<const f4v *>(&As[kk * LDA + (i * TY + ty) * 4]);
        a[i * 4] = v[0]; a[i * 4 + 1] = v[1]; a[i * 4 + 2] = v[2]; a[i * 4 + 3] = v[3];
      }
#pragma unroll
      for (int j = 0; j < TN / 4; ++j) {
        const f4v v = *reinterpret_cast<const f4v *>(&Bs[kk * LDB + (j * TX + tx) * 4]);
        b[j * 4] = v[0]; b[j * 4 + 1] = v[1]; b[j * 4 + 2] = v[2]; b[j * 4 + 3] = v[3];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
    }
  };
  if (kt1 > kt0) {
    const int nt = kt1 - kt0;
    int st = 0;
    if (p.stagger_mod > 1) st = ((blockIdx.x % p.stagger_mod) * p.stagger_tiles) % nt;
    auto tix = [&](int u) __attribute__((always_inline)) {
      const int t = u + st;
      return kt0 + (t >= nt ? t - nt : t);
    };
    loadA(tix(0)); loadB(tix(0));
    store();
    if (nt > 1) { loadA(tix(1)); loadB(tix(1)); }
    __syncthreads();
    for (int u = 0; u < nt; ++u) {
      compute();
      __syncthreads();
      if (u + 1 < nt) {
        store();
        if (u + 2 < nt) { loadA(tix(u + 2)); loadB(tix(u + 2)); }
        __syncthreads();
      }
    }
  }
  if (splits > 1) {
    float *W = p.ws + ((int64_t)batch * splits + split) * p.m * p.n;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int r = row0 + ((i / 4) * TY + ty) * 4 + i % 4;
      if (r >= p.m) continue;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int c = col0 + ((j / 4) * TX + tx) * 4 + j % 4;
        if (c < p.n) W[(int64_t)r * p.n + c] = acc[i][j];
      }
    }
    return;
  }
  float *Y = static_cast<float *>(p.y) + batch * p.sy;
  const float *bias = p.bias ? static_cast<const float *>(p.c) + batch * p.sc : nullptr;
  const bool yvec = p.ldy % 4 == 0 && p.sy % 4 == 0 && (reinterpret_cast<uintptr_t>(p.y) & 15) == 0;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = row0 + ((i / 4) * TY + ty) * 4 + i % 4;
    if (r >= p.m) continue;
#pragma unroll
    for (int j4 = 0; j4 < TN / 4; ++j4) {
      const int c = col0 + (j4 * TX + tx) * 4;
      f4v v;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        v[e] = acc[i][j4 * 4 + e] * p.alpha + (bias && c + e < p.n ? bias[c + e] : 0.f);
      float *dst = Y + (int64_t)r * p.ldy + c;
      if (yvec && c + 4 <= p.n) *reinterpret_cast<f4v *>(dst) = v;
      else
        for (int e = 0; e < 4; ++e) if (c + e < p.n) dst[e] = v[e];
    }
  }
}

// ---------------------------------------------------------------------------
// Config catalog. The LUT stores the geometry (GemmTuneConfig family >= 1);
// ids are positions in this table and never enter the LUT.
// ---------------------------------------------------------------------------
enum Family : int {
  kWmma = 1, kGemv = 2, kF32Tile = 3, kF32Dot = 4, kF32Rows = 5, kF32Lds = 6
};
// LUT field mapping (all families): bm16/bn16 (bm, bn), bk, wt_m, wt_n,
// threads, split_k, swizzle = stagger_mod, tile_n = stagger_tiles, except
// gemv: tile_n = columns per workgroup, bk = load unroll; f32dot: wt_m = lanes
// per output (KP), wt_n = outputs per lane group, bk = K per lane; f32rows:
// bn = NMAX, bk = KMAX, wt_m = KP.
struct Cfg {
  int family, bm, bn, bk, wt_m, wt_n, threads, splits, smod, stiles, tile_n;
};
#define GEMM_GEN_CFGS(X)                                                   \
  /* id fam    bm  bn  bk tm tn thr  sp smod st tn  */                     \
  X(0, kWmma, 128, 96, 64, 4, 3, 128, 1, 0, 0, 0)                          \
  X(1, kWmma, 96, 128, 64, 3, 4, 128, 1, 0, 0, 0)                          \
  X(2, kWmma, 128, 64, 64, 4, 2, 128, 1, 0, 0, 0)                          \
  X(3, kWmma, 96, 64, 64, 3, 2, 128, 1, 0, 0, 0)                           \
  X(4, kWmma, 64, 64, 64, 2, 2, 128, 1, 0, 0, 0)                           \
  X(5, kWmma, 64, 64, 64, 2, 2, 128, 4, 0, 0, 0)                           \
  X(6, kWmma, 64, 64, 64, 2, 2, 128, 8, 0, 0, 0)                           \
  X(7, kWmma, 32, 128, 32, 2, 2, 128, 1, 0, 0, 0)                          \
  X(8, kWmma, 16, 64, 64, 1, 1, 128, 1, 0, 0, 0)                           \
  X(9, kWmma, 16, 64, 64, 1, 1, 128, 1, 64, 2, 0)                          \
  X(10, kWmma, 16, 80, 64, 1, 1, 160, 1, 0, 0, 0)                          \
  X(11, kWmma, 16, 80, 64, 1, 1, 160, 1, 32, 2, 0)                         \
  X(12, kGemv, 0, 0, 4, 0, 0, 512, 1, 0, 0, 8)                             \
  X(13, kF32Dot, 0, 0, 8, 16, 1, 128, 1, 0, 0, 0)                          \
  X(14, kF32Dot, 0, 0, 8, 16, 2, 256, 1, 0, 0, 0)                          \
  X(15, kF32Dot, 0, 0, 16, 8, 4, 256, 1, 0, 0, 0)                          \
  X(16, kF32Dot, 0, 0, 8, 4, 4, 256, 1, 0, 0, 0)                           \
  X(17, kF32Lds, 32, 32, 128, 2, 2, 256, 1, 0, 0, 0)                       \
  X(18, kF32Rows, 0, 16, 128, 4, 0, 512, 1, 0, 0, 0)                       \
  X(19, kF32Rows, 0, 16, 128, 4, 0, 256, 1, 0, 0, 0)                       \
  X(20, kF32Tile, 128, 128, 16, 8, 8, 256, 1, 0, 0, 0)                     \
  X(21, kF32Tile, 128, 128, 16, 8, 8, 256, 4, 32, 2, 0)                    \
  X(22, kF32Tile, 64, 128, 32, 8, 8, 128, 1, 32, 2, 0)                     \
  X(23, kF32Tile, 64, 128, 16, 8, 8, 128, 4, 0, 0, 0)                      \
  X(24, kF32Tile, 64, 128, 32, 8, 8, 128, 1, 0, 0, 0)                     \
  X(25, kWmma, 128, 128, 64, 2, 4, 256, 1, 0, 0, 0)
#define GEMM_GEN_ENTRY(ID, F, BM, BN, BK, TM, TN, TH, SP, SM, ST, TL) \
  {F, BM, BN, BK, TM, TN, TH, SP, SM, ST, TL},
static constexpr Cfg kCfg[] = {GEMM_GEN_CFGS(GEMM_GEN_ENTRY)};
#undef GEMM_GEN_ENTRY
static constexpr int kNumCfg = static_cast<int>(sizeof(kCfg) / sizeof(kCfg[0]));

inline bool isF16Family(int f) { return f == kWmma || f == kGemv; }

// Legality for a tight row-major request (strides are element batch strides).
inline bool legal(const Cfg &c, int m, int n, int k, int ta, int tb, int in,
                  int out, int batches) {
  if (ta || in != out || m < 1 || n < 1 || k < 1 || batches < 1) return false;
  if (isF16Family(c.family) ? in != 0 : in != 1) return false;
  switch (c.family) {
  case kWmma: {
    if (k % 8 || (tb ? k : n) % 8) return false;
    const int64_t tiles = int64_t((m + c.bm - 1) / c.bm) * ((n + c.bn - 1) / c.bn);
    if (tiles > INT_MAX || int64_t(batches) * c.splits > 65535) return false;
    // A and B byte ranges are 32-bit buffer offsets.
    if (int64_t(m) * k * 2 > UINT32_MAX || int64_t(n) * k * 2 > UINT32_MAX) return false;
    if (c.splits > 1 &&
        (k / c.splits < 2 * c.bk || int64_t(m) * n * batches * c.splits * 4 > (int64_t(16) << 20)))
      return false;
    return true;
  }
  case kGemv:
    return !tb && m == 1 && n % 8 == 0 && k <= 16384 && int64_t(n) * k * 2 <= UINT32_MAX;
  case kF32Dot:
    return !tb && k <= c.wt_m * c.bk;
  case kF32Rows:
    return !tb && n <= c.bn && k == c.bk;
  case kF32Lds:
    return !tb && k <= c.bk;
  case kF32Tile:
    if (tb || int64_t(batches) * c.splits > 65535) return false;
    if (c.splits > 1 &&
        (k / c.splits < 4 * c.bk || int64_t(m) * n * batches * c.splits * 4 > (int64_t(16) << 20)))
      return false;
    return true;
  default:
    return false;
  }
}

inline size_t workspaceBytes(const Cfg &c, int m, int n, int batches) {
  return c.splits > 1 ? size_t(m) * n * batches * c.splits * sizeof(float) : 0;
}

template <class T>
static int launchReduce(const Args &p, int batches, hipStream_t s) {
  const int64_t count = int64_t(p.m) * p.n;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_reduce_kernel<T>),
                     dim3(unsigned((count + 1023) / 1024), 1, batches), dim3(256), 0, s, p);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

template <int BM, int BN, int BK, int WGM, int WGN, bool TB>
static void launchWmmaGen(const Args &p, int batches, hipStream_t s) {
  const int tiles = ((p.m + BM - 1) / BM) * ((p.n + BN - 1) / BN);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_wmma_kernel<BM, BN, BK, WGM, WGN, TB>),
                     dim3(tiles, 1, batches * (p.splits > 1 ? p.splits : 1)),
                     dim3(WGM * WGN * 32), 0, s, p);
}

template <bool TB>
static bool launchWmmaCfg(int id, const Args &p, int batches, hipStream_t s) {
  switch (id) {
#define GEN_W(ID, BM, BN, BK, WGM, WGN) \
  case ID: launchWmmaGen<BM, BN, BK, WGM, WGN, TB>(p, batches, s); return true;
    GEN_W(0, 128, 96, 64, 2, 2)
    GEN_W(1, 96, 128, 64, 2, 2)
    GEN_W(2, 128, 64, 64, 2, 2)
    GEN_W(3, 96, 64, 64, 2, 2)
    case 4: case 5:
    GEN_W(6, 64, 64, 64, 2, 2)
    GEN_W(7, 32, 128, 32, 1, 4)
    case 8:
    GEN_W(9, 16, 64, 64, 1, 4)
    case 10:
    GEN_W(11, 16, 80, 64, 1, 5)
    GEN_W(25, 128, 128, 64, 4, 2)
#undef GEN_W
  default:
    return false;
  }
}

// Launch config `id`. The caller verified legal() and workspace size.
static int launch(int id, hipStream_t s, const void *a, const void *b, const void *c,
                  void *y, int m, int n, int k, int tb, float alpha, int bias,
                  int batches, int64_t sa, int64_t sb, int64_t sc, int64_t sy,
                  void *workspace) {
  const Cfg &g = kCfg[id];
  Args p{};
  p.a = a; p.b = b; p.c = c; p.y = y;
  p.m = m; p.n = n; p.k = k;
  p.lda = k; p.ldb = tb ? k : n; p.ldy = n;
  p.sa = sa; p.sb = sb; p.sc = sc; p.sy = sy;
  p.alpha = alpha; p.bias = bias;
  p.group_m = 8;
  p.stagger_mod = g.smod; p.stagger_tiles = g.stiles;
  p.splits = g.splits;
  p.ws = static_cast<float *>(workspace);
  (void)hipGetLastError();
  switch (g.family) {
  case kWmma:
    if (!(tb ? launchWmmaCfg<true>(id, p, batches, s) : launchWmmaCfg<false>(id, p, batches, s)))
      return -1;
    if (g.splits > 1 && launchReduce<_Float16>(p, batches, s)) return 1;
    break;
  case kGemv: {
    const size_t lds = std::max(size_t(k), size_t(g.threads / 32) * g.tile_n) * sizeof(float);
    if (lds > 65536) return -1;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_gemv_kernel<8, 512, 4>),
                       dim3((n + 7) / 8, 1, batches), dim3(512), lds, s, p);
    break;
  }
  case kF32Dot: {
#define GEN_D(ID, TN, KP, KS, TH)                                                     \
  if (id == ID) {                                                                     \
    const int64_t groups = int64_t(m) * ((n + TN - 1) / TN);                         \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_f32dot_kernel<TN, KP, KS, TH>),      \
                       dim3(unsigned((groups + TH / KP - 1) / (TH / KP)), 1, batches), \
                       dim3(TH), 0, s, p);                                            \
  }
    GEN_D(13, 1, 16, 8, 128)
    GEN_D(14, 2, 16, 8, 256)
    GEN_D(15, 4, 8, 16, 256)
    GEN_D(16, 4, 4, 8, 256)
#undef GEN_D
    break;
  }
  case kF32Rows:
    if (id == 18)
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_f32rows_kernel<16, 128, 4, 512>),
                         dim3(unsigned((m + 127) / 128), 1, batches), dim3(512), 0, s, p);
    else
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_f32rows_kernel<16, 128, 4, 256>),
                         dim3(unsigned((m + 63) / 64), 1, batches), dim3(256), 0, s, p);
    break;
  case kF32Lds:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_f32lds_kernel<32, 32, 128, 2, 2, 256>),
                       dim3(unsigned(((m + 31) / 32) * ((n + 31) / 32)), 1, batches), dim3(256), 0, s, p);
    break;
  case kF32Tile: {
    const dim3 z(1, 1, batches * (g.splits > 1 ? g.splits : 1));
#define GEN_T(BM, BN, BK, TM, TN)                                                     \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gen_f32tile_kernel<BM, BN, BK, TM, TN>),   \
                     dim3(unsigned(((m + BM - 1) / BM) * ((n + BN - 1) / BN)), 1, z.z), \
                     dim3((BM / TM) * (BN / TN)), 0, s, p)
    if (id == 20 || id == 21) GEN_T(128, 128, 16, 8, 8);
    else if (id == 23) GEN_T(64, 128, 16, 8, 8);
    else GEN_T(64, 128, 32, 8, 8);
#undef GEN_T
    if (g.splits > 1 && launchReduce<float>(p, batches, s)) return 1;
    break;
  }
  default:
    return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

}  // namespace gemm_gen
