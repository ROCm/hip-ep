/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include <array>
#include <climits>
#include <cstddef>
#include "gemm_general_impl.h"

// Included by gemm_kernel.hip after the existing uniform implementation.
// Batch and K ownership remain explicit in every launch and workspace offset.
template <typename T, typename Out, int TM, int THREADS, int TN>
__global__ void __launch_bounds__(THREADS)
gemm_nn_parallel_kernel(const T *a, const T *b, const Out *c, Out *y,
    int m, int n, int k, float alpha, float beta, int crows, int ccols,
    int64_t sa, int64_t sb, int64_t sc, int64_t sy, int ag, int bg,
    int splits, float *partial) {
  constexpr int NW = THREADS / 32, VEC = TN / 32;
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int batch = blockIdx.z / splits, split = blockIdx.z % splits;
  const int row0 = blockIdx.y * TM, col0 = blockIdx.x * TN + lane * VEC;
  a += int64_t(batch / ag) * sa;
  b += int64_t(batch / bg) * sb;
  if (c) c += int64_t(batch) * sc;
  y += int64_t(batch) * sy;
  const int kb = int((int64_t(split) * k) / splits);
  const int ke = int((int64_t(split + 1) * k) / splits);
  float acc[TM][VEC] = {};
  T av[2][TM], bv[2][VEC];
  auto load = [&](int64_t kk, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if (col0 + VEC <= n) {
      __builtin_memcpy(bv[S], b + kk * n + col0, sizeof(bv[S]));
    } else {
#pragma unroll
      for (int v = 0; v < VEC; ++v)
        bv[S][v] = col0 + v < n ? b[kk * n + col0 + v] : T{};
    }
#pragma unroll
    for (int r = 0; r < TM; ++r)
      av[S][r] = row0 + r < m ? a[int64_t(row0 + r) * k + kk] : T{};
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
#pragma unroll
    for (int r = 0; r < TM; ++r) {
      const float value = GemmElem<T>::tof(av[S][r]);
#pragma unroll
      for (int v = 0; v < VEC; ++v)
        acc[r][v] = __fmaf_rn(value, GemmElem<T>::tof(bv[S][v]), acc[r][v]);
    }
  };
  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, 1>;
  int64_t kk = int64_t(kb) + wave;
  if (kk < ke) {
    load(kk, S0{});
    for (; kk + 2 * NW < ke; kk += 2 * NW) {
      load(kk + NW, S1{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      load(kk + 2 * NW, S0{});
      GEMM_SCHED_FENCE();
      compute(S1{});
    }
    if (kk + NW < ke) {
      load(kk + NW, S1{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      compute(S1{});
    } else compute(S0{});
  }
  __shared__ __align__(16) float sums[NW][TM][TN];
#pragma unroll
  for (int r = 0; r < TM; ++r)
#pragma unroll
    for (int v = 0; v < VEC; ++v) sums[wave][r][lane * VEC + v] = acc[r][v];
  HIPDNN_LDS_BARRIER();
  if (wave != 0) return;
#pragma unroll
  for (int r = 0; r < TM; ++r)
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) sum += sums[w][r][lane * VEC + v];
      if (row0 + r < m && col0 + v < n) {
        if (partial) {
          const size_t offset = (size_t(batch) * splits + split) * size_t(m) * n;
          partial[offset + size_t(row0 + r) * n + col0 + v] = sum;
        } else gemmStoreY(y, c, row0 + r, col0 + v, n, crows, ccols, sum, alpha, beta);
      }
    }
}

template <typename Out>
__global__ void __launch_bounds__(256)
gemm_batch_reduce_kernel(const float *partial, const Out *c,
    Out *y, int m, int n, int batches, int splits, int64_t sc, int64_t sy,
    int crows, int ccols, float alpha, float beta) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t matrix = size_t(m) * n, count = matrix * batches;
  if (index >= count) return;
  const size_t batch = index / matrix, local = index % matrix;
  float sum = 0.f;
  for (int split = 0; split < splits; ++split)
    sum += partial[(batch * splits + split) * matrix + local];
  gemmStoreY(y + batch * sy, c ? c + batch * sc : nullptr,
      int(local / n), int(local % n), n, crows, ccols, sum, alpha, beta);
}

template <typename T, typename Out>
static int launchParallelGemvNn(int threads, int tile_n, hipStream_t stream,
    const T *a, const T *b, const Out *c, Out *y, int m, int n, int k,
    float alpha, float beta, int crows, int ccols, int batches,
    int64_t sa, int64_t sb, int64_t sc, int64_t sy, int ag, int bg,
    int splits, float *partial) {
  const int tm = m == 1 ? 1 : m <= 2 ? 2 : m <= 4 || tile_n == 256 ? 4 : 8;
  const dim3 grid(unsigned((int64_t(n) + tile_n - 1) / tile_n),
                  unsigned((int64_t(m) + tm - 1) / tm), batches * splits);
#define NN_CASE(TM, TH, TN) \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_nn_parallel_kernel<T, Out, TM, TH, TN>), \
      grid, dim3(TH), 0, stream, a, b, c, y, m, n, k, alpha, beta, crows, ccols, \
      sa, sb, sc, sy, ag, bg, splits, partial)
#define NN_TILE(TM, TH) \
  if (tile_n == 32) { NN_CASE(TM, TH, 32); } \
  else if (tile_n == 64) { NN_CASE(TM, TH, 64); } \
  else { NN_CASE(TM, TH, 128); }
#define NN_TILE_WIDE(TM, TH) \
  if (tile_n == 256) { NN_CASE(TM, TH, 256); } \
  else { NN_TILE(TM, TH); }
#define NN_THREADS(TM) \
  if (threads == 128) { NN_TILE_WIDE(TM, 128); } \
  else if (threads == 256) { NN_TILE_WIDE(TM, 256); } \
  else { NN_TILE_WIDE(TM, 512); }
  if (tm == 1) { NN_THREADS(1); }
  else if (tm == 2) { NN_THREADS(2); }
  else if (tm == 4) { NN_THREADS(4); }
  else {
    if (threads == 128) { NN_TILE(8, 128); }
    else if (threads == 256) { NN_TILE(8, 256); }
    else { NN_TILE(8, 512); }
  }
#undef NN_THREADS
#undef NN_TILE_WIDE
#undef NN_TILE
#undef NN_CASE
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

// Small NT rows use register fragments, avoiding a padded large-M LDS panel.
template <typename T, typename Out, int NTL, int KU>
__global__ void __launch_bounds__(32)
gemm_small_nt_batched_kernel(const T *a, const T *b, const Out *c, Out *y,
    int m, int n, int k, float alpha, float beta, int crows, int ccols,
    int64_t sa, int64_t sb, int64_t sc, int64_t sy, int ag, int bg) {
  using Frag = typename std::conditional<std::is_same<T, _Float16>::value,
                                        half16, bf16xN>::type;
  constexpr int FE = HIPDNN_WMMA_FRAG_ELEMS, KC = 16 * KU;
  const int lane = threadIdx.x & 15, sub = threadIdx.x >> 4;
  const int row0 = blockIdx.y * 16, col0 = blockIdx.x * (16 * NTL);
  const int batch = blockIdx.z, koff = hipdnn_wmma_k_off(sub);
  a += int64_t(batch / ag) * sa;
  b += int64_t(batch / bg) * sb;
  if (c) c += int64_t(batch) * sc;
  y += int64_t(batch) * sy;
  const T *ap = a + int64_t(row0 + lane < m ? row0 + lane : 0) * k;
  const T *bp[NTL];
#pragma unroll
  for (int j = 0; j < NTL; ++j)
    bp[j] = b + int64_t(col0 + j * 16 + lane < n ? col0 + j * 16 + lane : 0) * k;
  Frag fa[2][KU], fb[2][NTL][KU];
  float8 acc[NTL];
#pragma unroll
  for (int j = 0; j < NTL; ++j) acc[j] = float8{};
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
#pragma unroll
    for (int q = 0; q < KU; ++q) {
      const int offset = k0 + q * 16 + koff;
      if (!TL || offset + FE <= k) {
        __builtin_memcpy(&fa[S][q], ap + offset, sizeof(Frag));
#pragma unroll
        for (int j = 0; j < NTL; ++j)
          __builtin_memcpy(&fb[S][j][q], bp[j] + offset, sizeof(Frag));
      } else {
        T tail_a[FE]{}, tail_b[NTL][FE]{};
#pragma unroll
        for (int v = 0; v < FE; ++v) {
          if (offset + v < k) {
            tail_a[v] = ap[offset + v];
#pragma unroll
            for (int j = 0; j < NTL; ++j)
              tail_b[j][v] = bp[j][offset + v];
          }
        }
        __builtin_memcpy(&fa[S][q], tail_a, sizeof(Frag));
#pragma unroll
        for (int j = 0; j < NTL; ++j)
          __builtin_memcpy(&fb[S][j][q], tail_b[j], sizeof(Frag));
      }
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
#pragma unroll
    for (int q = 0; q < KU; ++q)
#pragma unroll
      for (int j = 0; j < NTL; ++j) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[j] = HIPDNN_WMMA_F32_16X16X16_F16(fa[S][q], fb[S][j][q], acc[j]);
        else acc[j] = HIPDNN_WMMA_F32_16X16X16_BF16(fa[S][q], fb[S][j][q], acc[j]);
      }
  };
  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, 1>;
  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int chunks = k / KC;
  if (chunks) {
    load(0, S0{}, Full{});
    int q = 0;
    for (; q + 2 < chunks; q += 2) {
      load((q + 1) * KC, S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      load((q + 2) * KC, S0{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S1{});
    }
    if (chunks - q == 2) {
      load((q + 1) * KC, S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      compute(S1{});
    } else compute(S0{});
  }
  if (chunks * KC < k) {
    load(chunks * KC, S0{}, Tail{});
    compute(S0{});
  }
#pragma unroll
  for (int j = 0; j < NTL; ++j)
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int row = row0 + hipdnn_wmma_acc_row(sub, v), col = col0 + j * 16 + lane;
      if (row < m && col < n)
        gemmStoreY(y, c, row, col, n, crows, ccols, acc[j][v], alpha, beta);
    }
}

// Each wave owns a short B panel. Contiguous NT rows are read by several
// lanes; NN rows reuse the existing packed transpose instead of scalar gathers.
template <typename T, typename Out, bool TB>
__global__ void __launch_bounds__(32)
gemm_small_cooperative_batched_kernel(const T *a, const T *b, const Out *c,
    Out *y, int m, int n, int k, float alpha, float beta, int crows, int ccols,
    int64_t sa, int64_t sb, int64_t sc, int64_t sy, int ag, int bg,
    int splits, float *partial) {
  constexpr int NTL = 4, KU = 2, KC = 32, FE = HIPDNN_WMMA_FRAG_ELEMS;
  using Frag = typename std::conditional<std::is_same<T, _Float16>::value,
                                        half16, bf16xN>::type;
  using PB = SkinnyT<T, NTL, KC, true>;
  constexpr int SEG = KC / 8, RPI = 32 / SEG, NCP = 16 * NTL / RPI;
  const int lid = threadIdx.x, lane = lid & 15, sub = lid >> 4;
  const int row0 = blockIdx.y * 16, col0 = blockIdx.x * 64;
  const int batch = blockIdx.z / splits, split = blockIdx.z % splits;
  a += int64_t(batch / ag) * sa;
  b += int64_t(batch / bg) * sb;
  if (c) c += int64_t(batch) * sc;
  y += int64_t(batch) * sy;
  const int chunks = int((int64_t(k) + KC - 1) / KC);
  const int first = int(int64_t(split) * chunks / splits) * KC;
  const int last = std::min(k, int(int64_t(split + 1) * chunks / splits) * KC);
  const T *ap = a + int64_t(row0 + lane < m ? row0 + lane : 0) * k;
  const int koff = hipdnn_wmma_k_off(sub);
  const int cp_seg = (lid % SEG) * 8;
  const bool vecB = n >= PB::CW && n % PB::CW == 0 &&
                    (reinterpret_cast<uintptr_t>(b) & 15) == 0;
  const bool vecNT = k % 8 == 0 && (reinterpret_cast<uintptr_t>(b) & 15) == 0;
  const T *cp[NCP];
#pragma unroll
  for (int q = 0; q < NCP; ++q) {
    const int column = col0 + q * RPI + lid / SEG;
    cp[q] = b + int64_t(column < n ? column : 0) * k;
  }
  __shared__ __align__(16) T tile[16 * NTL][PB::STR];
  Frag fa[2][KU];
  SkinnyReg<T, NTL, KC, true> rb[2];
  u32x4 rc[2][NCP];
  float8 acc[NTL]{};
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
#pragma unroll
    for (int q = 0; q < KU; ++q) {
      const int offset = k0 + q * 16 + koff;
      if (!TL || offset + FE <= last) {
        __builtin_memcpy(&fa[S][q], ap + offset, sizeof(Frag));
      } else {
        T values[FE]{};
#pragma unroll
        for (int v = 0; v < FE; ++v)
          if (offset + v < last) values[v] = ap[offset + v];
        __builtin_memcpy(&fa[S][q], values, sizeof(Frag));
      }
    }
    if constexpr (TB) {
#pragma unroll
      for (int q = 0; q < NCP; ++q) {
        const int offset = k0 + cp_seg;
        if (vecNT && (!TL || offset + 8 <= last)) {
          rc[S][q] = *reinterpret_cast<const u32x4 *>(cp[q] + offset);
        } else {
          T values[8]{};
#pragma unroll
          for (int v = 0; v < 8; ++v)
            if (!TL || offset + v < last) values[v] = cp[q][offset + v];
          __builtin_memcpy(&rc[S][q], values, sizeof(values));
        }
      }
    } else {
      skinnyLoadT<T, NTL, KC, true, TL>(b, n, n, last, k0, col0, vecB, rb[S]);
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TB) {
#pragma unroll
      for (int q = 0; q < NCP; ++q) {
        const int column = q * RPI + lid / SEG;
        *reinterpret_cast<u32x4 *>(&tile[column][cp_seg]) = rc[S][q];
      }
    } else {
      skinnyStoreT<T, NTL, KC, true>(&tile[0][0], rb[S]);
    }
#pragma unroll
    for (int q = 0; q < KU; ++q)
#pragma unroll
      for (int j = 0; j < NTL; ++j) {
        Frag fb;
        __builtin_memcpy(&fb, &tile[j * 16 + lane][q * 16 + koff], sizeof(Frag));
        acc[j] = gemmWmmaAccumulate<true>(fa[S][q], fb, acc[j]);
      }
  };
  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, 1>;
  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int whole = (last - first) / KC;
  if (whole > 0) {
    load(first, S0{}, Full{});
    int q = 0;
    for (; q + 2 < whole; q += 2) {
      load(first + (q + 1) * KC, S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      load(first + (q + 2) * KC, S0{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S1{});
    }
    if (whole - q == 2) {
      load(first + (q + 1) * KC, S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      compute(S1{});
    } else compute(S0{});
  }
  if (first + whole * KC < last) {
    load(first + whole * KC, S0{}, Tail{});
    compute(S0{});
  }
#pragma unroll
  for (int j = 0; j < NTL; ++j)
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      const int row = row0 + hipdnn_wmma_acc_row(sub, v);
      const int column = col0 + j * 16 + lane;
      if (row < m && column < n) {
        if (partial) {
          const size_t offset = (size_t(batch) * splits + split) * size_t(m) * n;
          partial[offset + size_t(row) * n + column] = acc[j][v];
        } else gemmStoreY(y, c, row, column, n, crows, ccols, acc[j][v], alpha, beta);
      }
    }
}

struct BatchedWmmaCfg { int bm, bn, wm, wn, bk, splits; int direct_nt = 0; };
#define GEMM_BATCH_WMMA_SET(SK) \
    {16,64,1,2,32,SK}, {16,128,1,2,64,SK}, {32,64,2,2,32,SK}, \
    {64,64,2,2,32,SK}, {64,128,2,4,32,SK}, {128,64,4,2,32,SK}, \
    {128,128,4,2,32,SK}, {128,128,4,2,16,SK}
static constexpr BatchedWmmaCfg kBatchedWmma[] = {
    GEMM_BATCH_WMMA_SET(1), GEMM_BATCH_WMMA_SET(2), GEMM_BATCH_WMMA_SET(4),
    {16,32,1,2,64,1,1}, {16,64,1,4,64,1,1},
    {16,64,1,4,32,1,2}, {16,64,1,4,32,8,2},
    {16,64,1,4,32,16,2}, {16,64,1,4,32,32,2}, {16,64,1,4,32,64,2},
    {32,32,2,2,32,1}, {32,32,2,2,32,2}, {32,32,2,2,32,4},
    {64,32,2,2,32,1}, {64,32,2,2,32,2}, {64,32,2,2,32,4},
};
#undef GEMM_BATCH_WMMA_SET
static constexpr int kNumBatchedWmma = sizeof(kBatchedWmma) / sizeof(kBatchedWmma[0]);

template <int BM, int BN, int WM, int WN, int BK, bool TA, bool TB,
          typename T, typename Out>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM,BN,WM,WN), GEMM_WG_SZ(BM,BN,WM,WN))))
__attribute__((amdgpu_waves_per_eu(2,8)))
void gemm_wmma_batched_kernel(const T *a, const T *b, const Out *c, Out *y,
    int m, int n, int k, float alpha, float beta, int crows, int ccols,
    int64_t sa, int64_t sb, int64_t sc, int64_t sy, int ag, int bg,
    int splits, float *partial) {
  const int batch = blockIdx.z / splits, split = blockIdx.z % splits;
  a += int64_t(batch / ag) * sa;
  b += int64_t(batch / bg) * sb;
  if (c) c += int64_t(batch) * sc;
  y += int64_t(batch) * sy;
  if (partial) partial += size_t(batch) * splits * size_t(m) * n;
  GemmWmmaImpl<BM,BN,WM,WN,TB,T,BK,true,true,1,TA,Out,true,true>(
      a,b,c,y,m,n,k,alpha,beta,crows,ccols,8,partial,TB?k:n,0,TA?m:k,0,splits,split);
}

template <typename T, typename Out, bool TA, bool TB>
static void launchBatchedWmma(int id, hipStream_t stream, const T *a, const T *b,
    const Out *c, Out *y, int m, int n, int k, float alpha, float beta,
    int crows, int ccols, int batches, int64_t sa, int64_t sb, int64_t sc,
    int64_t sy, int ag, int bg, int splits, float *partial) {
  const auto& cfg = kBatchedWmma[id];
  dim3 grid(unsigned((int64_t(n) + cfg.bn - 1) / cfg.bn),
            unsigned((int64_t(m) + cfg.bm - 1) / cfg.bm), batches * splits);
  if (cfg.direct_nt == 2) {
    if constexpr (!TA)
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_small_cooperative_batched_kernel<T,Out,TB>),
          grid,dim3(32),0,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols,
          sa,sb,sc,sy,ag,bg,splits,partial);
    return;
  }
  if (cfg.direct_nt == 1) {
    if constexpr (!TA && TB) {
      if (cfg.bn == 32)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_small_nt_batched_kernel<T,Out,2,4>),
            grid,dim3(32),0,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols,sa,sb,sc,sy,ag,bg);
      else
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_small_nt_batched_kernel<T,Out,4,4>),
            grid,dim3(32),0,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols,sa,sb,sc,sy,ag,bg);
    }
    return;
  }
#define BW(ID,BM,BN,WM,WN,BK) \
  case ID: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_batched_kernel<BM,BN,WM,WN,BK,TA,TB,T,Out>), \
      grid,dim3(GEMM_WG_SZ(BM,BN,WM,WN)),0,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols, \
      sa,sb,sc,sy,ag,bg,splits,partial);break
  if (id >= 31) {
    switch (id) {
      BW(31,32,32,2,2,32); BW(32,32,32,2,2,32); BW(33,32,32,2,2,32);
      BW(34,64,32,2,2,32); BW(35,64,32,2,2,32); BW(36,64,32,2,2,32);
    }
  } else switch (id % 8) {
    BW(0,16,64,1,2,32); BW(1,16,128,1,2,64); BW(2,32,64,2,2,32);
    BW(3,64,64,2,2,32); BW(4,64,128,2,4,32); BW(5,128,64,4,2,32);
    BW(6,128,128,4,2,32); BW(7,128,128,4,2,16);
  }
#undef BW
}

// General FMA batching retains represented inputs and one final conversion.
// The uniform single-matrix route continues to use the existing resolver.
template <typename T, typename Out, bool TA, bool TB,
          int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ void __launch_bounds__(THREADS)
gemm_fma_batched_kernel(const T *a,const T *b,const Out *c,Out *y,
    int m,int n,int k,float alpha,float beta,int crows,int ccols,
    int64_t sa,int64_t sb,int64_t sc,int64_t sy,int ag,int bg) {
  constexpr int NC = BN / TN;
  static_assert(THREADS == (BM / TM) * NC, "FMA thread geometry");
  const int batch=blockIdx.z,tid=threadIdx.x;
  const int r0=blockIdx.y*BM+(tid/NC)*TM, c0=blockIdx.x*BN+(tid%NC)*TN;
  a+=int64_t(batch/ag)*sa;b+=int64_t(batch/bg)*sb;
  if(c)c+=int64_t(batch)*sc;y+=int64_t(batch)*sy;
  __shared__ __align__(16) float at[BM][BK],bt[BK][BN];
  float acc[TM][TN]={};
  for(int64_t kb=0;kb<k;kb+=BK) {
    for(int i=tid;i<BM*BK;i+=THREADS) {
      int r=blockIdx.y*BM+i/BK; int64_t kk=kb+i%BK;
      at[i/BK][i%BK]=r<m&&kk<k?GemmElem<T>::tof(a[TA?int64_t(kk)*m+r:int64_t(r)*k+kk]):0.f;
    }
    for(int i=tid;i<BK*BN;i+=THREADS) {
      int64_t kk=kb+i/BN; int col=blockIdx.x*BN+i%BN;
      bt[i/BN][i%BN]=col<n&&kk<k?GemmElem<T>::tof(b[TB?int64_t(col)*k+kk:int64_t(kk)*n+col]):0.f;
    }
    HIPDNN_LDS_BARRIER();
#pragma unroll
    for(int kk=0;kk<BK;++kk)
#pragma unroll
      for(int r=0;r<TM;++r)
#pragma unroll
        for(int col=0;col<TN;++col)
          acc[r][col]=__fmaf_rn(at[(tid/NC)*TM+r][kk],bt[kk][(tid%NC)*TN+col],acc[r][col]);
    HIPDNN_LDS_BARRIER();
  }
#pragma unroll
  for(int r=0;r<TM;++r)
#pragma unroll
    for(int col=0;col<TN;++col)
      if(r0+r<m&&c0+col<n)gemmStoreY(y,c,r0+r,c0+col,n,crows,ccols,acc[r][col],alpha,beta);
}


#define GEMM_BATCH_FMA_CFGS(X) \
  GEMM_F32_TILE_CFGS(X) \
  X(7,64,64,16,4,4,256) \
  X(8,128,64,16,4,4,512) \
  X(9,64,128,16,4,4,512) \
  X(10,32,32,16,2,2,256)
#define GEMM_BATCH_FMA_ENTRY(ID,BM,BN,BK,TM,TN,TH) {BM,BN,BK,TM,TN,TH},
static constexpr TiledFmaCfg kBatchedFma[] = {
    GEMM_BATCH_FMA_CFGS(GEMM_BATCH_FMA_ENTRY)};
#undef GEMM_BATCH_FMA_ENTRY
static constexpr int kNumBatchedFma = sizeof(kBatchedFma) / sizeof(kBatchedFma[0]);
static int gemmBatchedFmaFromGeom(const gemm_lut::TiledFmaAnswer& g) {
  for(int id=0;id<kNumBatchedFma;++id) {
    const auto& c=kBatchedFma[id];
    if(c.bm==g.bm && c.bn==g.bn && c.bk==g.bk && c.tm==g.tm &&
       c.tn==g.tn && c.threads==g.threads) return id;
  }
  return -1;
}
template <typename T, typename Out, bool TA, bool TB>
static void launchBatchedFma(int id,hipStream_t stream,const T*a,const T*b,
    const Out*c,Out*y,int m,int n,int k,float alpha,float beta,int crows,int ccols,
    int batches,int64_t sa,int64_t sb,int64_t sc,int64_t sy,int ag,int bg) {
  const auto& cfg=kBatchedFma[id];
  const dim3 grid(unsigned((int64_t(n)+cfg.bn-1)/cfg.bn),
                  unsigned((int64_t(m)+cfg.bm-1)/cfg.bm),batches);
#define GEMM_BATCH_FMA_LAUNCH(ID,BM,BN,BK,TM,TN,TH) \
  case ID: hipLaunchKernelGGL( \
      HIP_KERNEL_NAME(gemm_fma_batched_kernel<T,Out,TA,TB,BM,BN,BK,TM,TN,TH>), \
      grid,dim3(TH),0,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols, \
      sa,sb,sc,sy,ag,bg); break;
  switch(id) { GEMM_BATCH_FMA_CFGS(GEMM_BATCH_FMA_LAUNCH) }
#undef GEMM_BATCH_FMA_LAUNCH
}

static int gemmBatchCores() {
  thread_local int cached_device=-1,cores=0;
  int device=0;if(hipGetDevice(&device)!=hipSuccess)return 32;
  if(device!=cached_device) {
    if(hipDeviceGetAttribute(&cores,hipDeviceAttributeMultiprocessorCount,device)!=hipSuccess||cores<1)cores=32;
    cached_device=device;
  }
  return cores;
}
static bool gemmBatchTypes(int in,int out) {
  return in>=0&&in<=3&&(out==in||((in==0||in==3)&&out==1));
}
static bool gemmParallelNnShape(int m,int n,int k,int ta,int tb,int in,int batches) {
  return in!=2&&!ta&&!tb&&m<=16&&(m==1||n<=512)&&(batches>1||k>=512);
}
struct GemmBatchChoice {
  int family=0,id=-1,splits=1;
  gemm_lut::Source source=gemm_lut::Source::None;
  bool online=false;
};
using GemmBatchKey = std::array<int64_t,20>;
struct GemmBatchKeyHash {
  size_t operator()(const GemmBatchKey& key) const {
    size_t hash=0;
    for(int64_t value:key) hash^=std::hash<int64_t>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
    return hash;
  }
};
static std::unordered_map<GemmBatchKey,GemmBatchChoice,GemmBatchKeyHash> g_gemm_batch_cache;
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
static thread_local int gemmBatchRankConfig=-2;
#endif
static int gemmNnSplits(int m,int n,int k,int batches,int tile_n) {
  const int tm=m==1?1:m<=2?2:m<=4||tile_n==256?4:8;
  const int64_t groups=((int64_t(m)+tm-1)/tm)*((int64_t(n)+tile_n-1)/tile_n)*batches;
  const int target=gemmBatchCores()*4;
  int sk=int(std::max<int64_t>(1,(target+groups-1)/groups));
  return std::min({64,sk,std::max(1,k/128),65535/batches});
}
static bool gemmBatchedWmmaLegal(const BatchedWmmaCfg& c, int m, int n,
                                int k, int batches, int ta, int tb) {
  if (c.direct_nt == 1) return !ta && tb && m <= 16;
  if (c.direct_nt == 2 && (ta || m > 16)) return false;
  if (c.splits == 1) return true;
  const int64_t grid = ((int64_t(m) + c.bm - 1) / c.bm) *
                       ((int64_t(n) + c.bn - 1) / c.bn) * batches;
  return k >= 512 && k / c.splits >= 128 &&
         grid < gemmBatchCores() * 4 && c.splits <= 65535 / batches;
}
static int gemmBatchedWmmaFromGeom(const gemm_lut::WmmaAnswer& g) {
  for (int id = 0; id < kNumBatchedWmma; ++id) {
    const auto& c = kBatchedWmma[id];
    if (c.bm == g.bm && c.bn == g.bn && c.wm == g.wt_m &&
        c.wn == g.wt_n && c.bk == g.bk && g.swizzle_n == 8 &&
        c.splits == g.split_k) return id;
  }
  return -1;
}
static bool gemmUniformGridSplit(int m,int n,int k,int ta,int tb,int in,int out,
                                 int batches,int ag,int bg) {
  // A narrow output grid leaves the K loop serial on too few workgroups.
  // The caller-owned typed workspace lets the same WMMA core split this work.
  if (in!=out || (in!=0&&in!=3) || batches!=1 || ag!=1 || bg!=1 || ta || tb ||
      m<16 || k<512 || hipdnn_device_is_wave64()) return false;
  const int64_t grid=((int64_t(m)+127)/128)*((int64_t(n)+127)/128);
  return grid < gemmBatchCores()*2;
}
static bool gemmSmallGridWmma(int m,int n,int k,int ta,int tb,int in,int batches) {
  if ((in!=0&&in!=3) || ta || tb || m>16 || k<512 ||
      hipdnn_device_is_wave64()) return false;
  const int64_t grid=((int64_t(n)+63)/64)*batches;
  return grid < gemmBatchCores()*2;
}
static int gemmBatchFamily(int m,int n,int k,int ta,int tb,int in,int out,int batches,
                           int ag,int bg) {
  if(!k)return 3;
  if(in==2)return 4;
  if(gemmForcedRoute())return in==out?4:-1;
  if(gemmSmallGridWmma(m,n,k,ta,tb,in,batches))return 2;
  if(gemmParallelNnShape(m,n,k,ta,tb,in,batches))return 1;
  if(gemmUniformGridSplit(m,n,k,ta,tb,in,out,batches,ag,bg))return 2;
  if(in==out&&batches==1&&ag==1&&bg==1)return 0;
  return (in==0||in==3)&&!hipdnn_device_is_wave64()?2:3;
}
static GemmBatchChoice gemmBatchChoice(int m,int n,int k,int ta,int tb,int in,int out,
    int batches,int ag,int bg) {
  GemmBatchChoice choice;
  if (k == 0) return {3,-1,1};
  if (in == 2) return {4,-1,1};
  if (gemmForcedRoute()) return in == out ? GemmBatchChoice{4,-1,1} : GemmBatchChoice{-1,-1,1};
  const bool small_grid=gemmSmallGridWmma(m,n,k,ta,tb,in,batches);
  if(!small_grid&&gemmParallelNnShape(m,n,k,ta,tb,in,batches)) {
    choice.family=1;choice.id=n>512?10:6;
    gemm_lut::Request req;
    req.kind=gemm_lut::Kind::GemvNn;
    req.act_dtype=req.wts_dtype=in==0?gemm_lut::HipdnnDType::F16:in==3?gemm_lut::HipdnnDType::BF16:gemm_lut::HipdnnDType::F32;
    req.out_dtype=out==0?gemm_lut::HipdnnDType::F16:out==3?gemm_lut::HipdnnDType::BF16:gemm_lut::HipdnnDType::F32;
    req.m=m;req.n=n;req.k=k;req.trans_b=0;
    req.batch_count=batches;req.a_group=ag;req.b_group=bg;
    auto valid=[](void* opaque,const gemm_lut::GemvAnswer& g) {
      return (*static_cast<const bool*>(opaque) || g.tile_n>0) && gemvNnConfigFromGeom(g)>=0;
    };
    const bool uniform = in == out;
    if(gemmLutLookupEnabled() && !gemmForcedRoute()) {
      auto hit=gemm_lut::resolve(req,nullptr,valid,nullptr,const_cast<bool*>(&uniform));
      if(hit.source!=gemm_lut::Source::None) {choice.id=gemvNnConfigFromGeom(hit.gemv);choice.source=hit.source;}
    }
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
    if(gemmBatchRankConfig>=0)choice.id=gemmBatchRankConfig;
#endif
    if (choice.id < 0 || choice.id >= kNumGemvNn ||
        (in != out && !kGemvNn[choice.id].tile_n)) return {-1,-1,1};
    choice.splits=kGemvNn[choice.id].tile_n?gemmNnSplits(m,n,k,batches,kGemvNn[choice.id].tile_n):1;
    return choice;
  }
  const bool grid_split=gemmUniformGridSplit(m,n,k,ta,tb,in,out,batches,ag,bg);
  if(in==out&&batches==1&&ag==1&&bg==1&&!grid_split&&!small_grid)return choice;
  if((in==0||in==3)&&!hipdnn_device_is_wave64()) {
    choice.family=2;
    choice.id=grid_split?3:m<=16?26:m<=32?2:m<=64?3:(k%32==16?7:6);
    if (small_grid) {
      const int64_t grid=((int64_t(n)+63)/64)*batches;
      const int wanted=int((gemmBatchCores()*4+grid-1)/grid);
      for (int id=27;id<=30;++id)
        if (kBatchedWmma[id].splits <= wanted &&
            gemmBatchedWmmaLegal(kBatchedWmma[id],m,n,k,batches,ta,tb)) choice.id=id;
    }
    const auto& c=kBatchedWmma[choice.id];
    const int64_t grid=((int64_t(m)+c.bm-1)/c.bm)*((int64_t(n)+c.bn-1)/c.bn)*batches;
    if (choice.id < 8 && k >= 512 && grid < gemmBatchCores() * 2) {
      if (gemmBatchedWmmaLegal(kBatchedWmma[choice.id + 16], m, n, k, batches,ta,tb))
        choice.id += 16;
      else if (gemmBatchedWmmaLegal(kBatchedWmma[choice.id + 8], m, n, k, batches,ta,tb))
        choice.id += 8;
    }
    gemm_lut::Request req;
    req.kind = gemm_lut::Kind::Wmma;
    req.act_dtype = req.wts_dtype = in == 0 ? gemm_lut::HipdnnDType::F16 : gemm_lut::HipdnnDType::BF16;
    req.out_dtype = out == 1 ? gemm_lut::HipdnnDType::F32 : req.act_dtype;
    req.m=m;req.n=n;req.k=k;req.trans_b=tb;
    req.batch_count=batches;req.a_group=ag;req.b_group=bg;
    struct Context { int m,n,k,batches,ta,tb; } context{m,n,k,batches,ta,tb};
    auto valid=[](void* opaque,const gemm_lut::WmmaAnswer& g) {
      const auto& s=*static_cast<Context*>(opaque);
      const int id=gemmBatchedWmmaFromGeom(g);
      return id>=0&&gemmBatchedWmmaLegal(kBatchedWmma[id],s.m,s.n,s.k,s.batches,s.ta,s.tb);
    };
    if (!ta && gemmLutLookupEnabled() && !gemmForcedRoute()) {
      auto hit=gemm_lut::resolve(req,valid,nullptr,nullptr,&context);
      if(hit.source!=gemm_lut::Source::None) {choice.id=gemmBatchedWmmaFromGeom(hit.wmma);choice.source=hit.source;}
    }
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
    if(gemmBatchRankConfig>=0)choice.id=gemmBatchRankConfig;
#endif
    if (choice.id < 0 || choice.id >= kNumBatchedWmma ||
        !gemmBatchedWmmaLegal(kBatchedWmma[choice.id],m,n,k,batches,ta,tb))
      return {-1,-1,1};
    choice.splits=kBatchedWmma[choice.id].splits;
  } else {
    choice.family=3; choice.id=m<32?5:m<64?4:m<128?1:0;
    gemm_lut::Request req;
    req.kind=gemm_lut::Kind::TiledFma;
    req.act_dtype=req.wts_dtype=in==0?gemm_lut::HipdnnDType::F16:
        in==3?gemm_lut::HipdnnDType::BF16:gemm_lut::HipdnnDType::F32;
    req.out_dtype=out==0?gemm_lut::HipdnnDType::F16:
        out==3?gemm_lut::HipdnnDType::BF16:gemm_lut::HipdnnDType::F32;
    req.m=m;req.n=n;req.k=k;req.trans_b=tb;
    req.batch_count=batches;req.a_group=ag;req.b_group=bg;
    auto valid=[](void*,const gemm_lut::TiledFmaAnswer& g) {
      return gemmBatchedFmaFromGeom(g)>=0;
    };
    if(!ta && gemmLutLookupEnabled()) {
      auto hit=gemm_lut::resolve(req,nullptr,nullptr,valid,nullptr);
      if(hit.source!=gemm_lut::Source::None) {choice.id=gemmBatchedFmaFromGeom(hit.tiled_fma);choice.source=hit.source;}
    }
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
    if(gemmBatchRankConfig>=0)choice.id=gemmBatchRankConfig;
#endif
    if(choice.id<0||choice.id>=kNumBatchedFma)return {-1,-1,1};
  }
  return choice;
}

// General family: only an exact offline-LUT point selects it. Legacy LUT
// lookups never see these points (Request::family), so every other request
// keeps its existing route. HIPDNN_EP_GEMM_GEN=0 disables the family;
// HIPDNN_EP_GEMM_GEN_CFG=<catalog id> forces a legal entry.
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
static thread_local int gemmGenRankConfig=-1;
#endif
struct GemmGenContext { int m,n,k,ta,tb,in,out,batches,id; };
static int gemmGenLookup(int m,int n,int k,int ta,int tb,int in,int out,int batches) {
  using namespace gemm_lut;
  GemmGenContext ctx{m,n,k,ta,tb,in,out,batches,-1};
  auto wv=[](void* o,const WmmaAnswer& g) {
    auto& x=*static_cast<GemmGenContext*>(o);
    for(int id=0;id<gemm_gen::kNumCfg;++id){const auto& c=gemm_gen::kCfg[id];
      if(c.family==g.family&&c.bm==g.bm&&c.bn==g.bn&&c.bk==g.bk&&c.wt_m==g.wt_m&&c.wt_n==g.wt_n&&
         c.threads==g.threads&&c.splits==g.split_k&&c.smod==g.swizzle_n&&c.stiles==g.tile_n&&
         gemm_gen::legal(c,x.m,x.n,x.k,x.ta,x.tb,x.in,x.out,x.batches)){x.id=id;return true;}}
    return false;
  };
  auto gv=[](void* o,const GemvAnswer& g) {
    auto& x=*static_cast<GemmGenContext*>(o);
    for(int id=0;id<gemm_gen::kNumCfg;++id){const auto& c=gemm_gen::kCfg[id];
      if(c.family==g.family&&c.threads==g.threads&&c.tile_n==g.tile_n&&c.bk==g.bk&&
         gemm_gen::legal(c,x.m,x.n,x.k,x.ta,x.tb,x.in,x.out,x.batches)){x.id=id;return true;}}
    return false;
  };
  auto tv=[](void* o,const TiledFmaAnswer& g) {
    auto& x=*static_cast<GemmGenContext*>(o);
    for(int id=0;id<gemm_gen::kNumCfg;++id){const auto& c=gemm_gen::kCfg[id];
      if(c.family==g.family&&c.bm==g.bm&&c.bn==g.bn&&c.bk==g.bk&&c.wt_m==g.tm&&c.wt_n==g.tn&&
         c.threads==g.threads&&c.splits==g.split_k&&c.smod==g.swizzle&&c.stiles==g.tile_n&&
         gemm_gen::legal(c,x.m,x.n,x.k,x.ta,x.tb,x.in,x.out,x.batches)){x.id=id;return true;}}
    return false;
  };
  Request req;
  req.act_dtype=req.wts_dtype=req.out_dtype=in==0?HipdnnDType::F16:HipdnnDType::F32;
  req.m=m;req.n=n;req.k=k;req.trans_b=tb;req.batch_count=batches;req.a_group=req.b_group=1;
  req.family=1;
  const Kind kinds[2]={in==0?Kind::Wmma:Kind::TiledFma,Kind::GemvNn};
  for(int i=0;i<(in==0&&!tb?2:1);++i) {
    req.kind=kinds[i];ctx.id=-1;
    auto hit=resolve(req,wv,gv,tv,&ctx);
    if(hit.source==Source::Exact&&ctx.id>=0)return ctx.id;
  }
  return -1;
}
static int gemmGenSelect(int m,int n,int k,int ta,int tb,int in,int out,int batches,int ag,int bg,
                         const void* c,int crows,int ccols,float beta) {
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_GEN",1)!=0;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_GEN_CFG",-1);
  if(!on||ta||in!=out||(in!=0&&in!=1)||ag!=1||bg!=1||batches<1||!m||!n||!k||
     hipdnn_device_is_wave64())return -1;
  if(beta!=0.f&&!(beta==1.f&&c&&crows==1&&ccols==n))return -1;
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
  if(gemmGenRankConfig<0)return -1;
  return gemmGenRankConfig<gemm_gen::kNumCfg&&
         gemm_gen::legal(gemm_gen::kCfg[gemmGenRankConfig],m,n,k,ta,tb,in,out,batches)?gemmGenRankConfig:-1;
#else
  if(forced>=0)return forced<gemm_gen::kNumCfg&&
         gemm_gen::legal(gemm_gen::kCfg[forced],m,n,k,ta,tb,in,out,batches)?forced:-1;
  if(!gemmLutLookupEnabled()||gemmForcedRoute())return -1;
  return gemmGenLookup(m,n,k,ta,tb,in,out,batches);
#endif
}
// Largest legal general-family workspace for this request, independent of the
// LUT choice, so a later selection never exceeds a live allocation.
static size_t gemmGenWorkspace(int m,int n,int k,int ta,int tb,int in,int out,int batches,int ag,int bg) {
  if(ag!=1||bg!=1)return 0;
  size_t bytes=0;
  for(int id=0;id<gemm_gen::kNumCfg;++id)
    if(gemm_gen::legal(gemm_gen::kCfg[id],m,n,k,ta,tb,in,out,batches))
      bytes=std::max(bytes,gemm_gen::workspaceBytes(gemm_gen::kCfg[id],m,n,batches));
  return bytes;
}
static std::unordered_map<GemmBatchKey,int,GemmBatchKeyHash> g_gemm_gen_cache;

extern "C" int hip_gemm_workspace_size(int m,int n,int k,int ta,int tb,int in,int out,
    int batches,int ag,int bg,size_t *bytes) {
  if(!bytes||m<0||n<0||k<0||m>INT_MAX-256||n>INT_MAX-256||k>INT_MAX-256||ta<0||ta>1||tb<0||tb>1||!gemmBatchTypes(in,out)||
     batches<0||batches>65535||ag<1||bg<1||batches%ag||batches%bg)return -1;
  *bytes=0;if(!m||!n||!batches)return 0;
  const uint64_t matrix=uint64_t(m)*n;
  const size_t ibytes=in==2?8:in==1?4:2, obytes=out==2?8:out==1?4:2;
  if (uint64_t(m)*k>uint64_t(PTRDIFF_MAX)/ibytes ||
      uint64_t(n)*k>uint64_t(PTRDIFF_MAX)/ibytes ||
      matrix>uint64_t(PTRDIFF_MAX)/obytes) return -1;
  if (!k) return 0;
  const int family=gemmBatchFamily(m,n,k,ta,tb,in,out,batches,ag,bg);
  if (family < 0) return -1;
  if ((family == 1 && (int64_t(m)+7)/8 > 65535) ||
      (family == 2 && (int64_t(m)+15)/16 > 65535) ||
      (family == 3 && (int64_t(m)+15)/16 > 65535)) return -1;
  // Allocate the legal catalog's maximum independently of the current LUT or
  // cache choice. Selection cannot replace an already-live workspace pointer.
  int splits=1;
  if(family==1)
    for (int tile_n : {32,64,128,256})
      splits=std::max(splits,gemmNnSplits(m,n,k,batches,tile_n));
  if(family==2) {
    for (const auto& cfg : kBatchedWmma)
      if (gemmBatchedWmmaLegal(cfg,m,n,k,batches,ta,tb))
        splits=std::max(splits,cfg.splits);
  }
  const size_t general=gemmGenWorkspace(m,n,k,ta,tb,in,out,batches,ag,bg);
  if(splits<=1){*bytes=general;return 0;}
  const uint64_t count=uint64_t(m)*n;
  if(count>uint64_t(PTRDIFF_MAX)/sizeof(float)/batches/splits ||
     count>uint64_t(INT_MAX)*256/batches)return -1;
  *bytes=std::max(general,size_t(count)*batches*splits*sizeof(float));return 0;
}

template <typename T,typename Out,bool TA,bool TB>
static int gemmBatchTyped(GemmBatchChoice choice,hipStream_t stream,const T*a,const T*b,
    const Out*c,Out*y,int m,int n,int k,float alpha,float beta,int crows,int ccols,
    int batches,int64_t sa,int64_t sb,int64_t sc,int64_t sy,int ag,int bg,float*partial) {
  if (!k) {
    for (int batch=0;batch<batches;++batch) {
      int rc=launchKZero<Out>(stream,c?c+int64_t(batch)*sc:nullptr,
                             y+int64_t(batch)*sy,m,n,beta,crows,ccols);
      if(rc)return rc;
    }
    return 0;
  }
  if(choice.family==1) {
    const auto& cfg=kGemvNn[choice.id];
    if(cfg.tile_n) {
      int rc=launchParallelGemvNn(cfg.threads,cfg.tile_n,stream,a,b,c,y,m,n,k,alpha,beta,
          crows,ccols,batches,sa,sb,sc,sy,ag,bg,choice.splits,partial);
      if(rc)return rc;
    } else {
      if constexpr(std::is_same<T,Out>::value) {
        for(int batch=0;batch<batches;++batch)
          {
            launchGemvNn(choice.id,stream,a+int64_t(batch/ag)*sa,b+int64_t(batch/bg)*sb,
                c?c+int64_t(batch)*sc:nullptr,y+int64_t(batch)*sy,m,n,k,alpha,beta,crows,ccols);
            if (hipGetLastError()!=hipSuccess) return 1;
          }
      } else return -1;
    }
  } else if(choice.family==2) {
    if constexpr(sizeof(T)==2)
      launchBatchedWmma<T,Out,TA,TB>(choice.id,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols,
          batches,sa,sb,sc,sy,ag,bg,choice.splits,partial);
    else return -1;
  } else {
    launchBatchedFma<T,Out,TA,TB>(choice.id,stream,a,b,c,y,m,n,k,alpha,beta,crows,ccols,
        batches,sa,sb,sc,sy,ag,bg);
  }
  if (hipGetLastError()!=hipSuccess)return 1;
  if(partial) {
    size_t count=size_t(m)*n*batches;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_batch_reduce_kernel<Out>),
        dim3(unsigned((count+255)/256)),dim3(256),0,stream,partial,c,y,m,n,batches,
        choice.splits,sc,sy,crows,ccols,alpha,beta);
  }
  return hipGetLastError()==hipSuccess?0:1;
}


static GemmBatchKey gemmBatchCacheKey(int m,int n,int k,int ta,int tb,int in,int out,
    int batches,int ag,int bg,int crows,int ccols,int64_t sa,int64_t sb,int64_t sc,
    int64_t sy,float alpha,float beta,bool safe) {
  int device=0;
  if(hipGetDevice(&device)!=hipSuccess)device=-1;
  uint32_t alpha_bits,beta_bits;
  std::memcpy(&alpha_bits,&alpha,sizeof(alpha));
  std::memcpy(&beta_bits,&beta,sizeof(beta));
  return {device,m,n,k,ta,tb,in,out,batches,ag,bg,crows,ccols,sa,sb,sc,sy,
          alpha_bits,beta_bits,safe};
}


static void gemmBatchLog(const GemmBatchChoice& choice,int m,int n,int k,
                         int in,int out,int batches,int ag,int bg) {
  const char* source=choice.online?"online":
      choice.source==gemm_lut::Source::Exact?"exact":
      choice.source==gemm_lut::Source::Nearest?"nearest":
      choice.source==gemm_lut::Source::Fallback?"fallback":
      choice.source==gemm_lut::Source::Alias?"alias":"default";
  if(choice.family==1) {
    const auto& c=kGemvNn[choice.id];
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] batch final M=%d N=%d K=%d in=%d out=%d B=%d ag=%d bg=%d -> source=%s family=gemv_nn cfg=%d th=%d tn=%d splits=%d\n",
        m,n,k,in,out,batches,ag,bg,source,choice.id,c.threads,c.tile_n,choice.splits);
  } else if(choice.family==2) {
    const auto& c=kBatchedWmma[choice.id];
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] batch final M=%d N=%d K=%d in=%d out=%d B=%d ag=%d bg=%d -> source=%s family=wmma cfg=%d bm=%d bn=%d wm=%d wn=%d bk=%d sw=8 splits=%d\n",
        m,n,k,in,out,batches,ag,bg,source,choice.id,c.bm,c.bn,c.wm,c.wn,c.bk,choice.splits);
  } else if(choice.family==3 && k) {
    const auto& c=kBatchedFma[choice.id];
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] batch final M=%d N=%d K=%d in=%d out=%d B=%d ag=%d bg=%d -> source=%s family=tiled_fma cfg=%d bm=%d bn=%d tm=%d tn=%d bk=%d th=%d\n",
        m,n,k,in,out,batches,ag,bg,source,choice.id,c.bm,c.bn,c.tm,c.tn,c.bk,c.threads);
  } else if(!k) {
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] batch final M=%d N=%d K=0 in=%d out=%d B=%d -> source=epilogue\n",m,n,in,out,batches);
  }
}

template <typename Launch>
static int gemmBatchTune(GemmBatchChoice& choice,hipStream_t stream,int m,int n,int k,
    int in,int out,int batches,int ta,int tb,Launch&& launch) {
  hipEvent_t begin=nullptr,end=nullptr;
  if(hipEventCreate(&begin)!=hipSuccess)return 1;
  if(hipEventCreate(&end)!=hipSuccess){hipEventDestroy(begin);return 1;}
  int error=0;
  auto checked=[&](const GemmBatchChoice& candidate){
    if(!error)error=launch(candidate);
  };
  gemmSettleClock(stream,begin,end,[&]{checked(choice);},40);
  float best_ms=1e30f;
  const int count=choice.family==1?kNumGemvNn:
                  choice.family==2?kNumBatchedWmma:kNumBatchedFma;
  GemmBatchChoice best=choice;
  for(int id=0;id<count&&!error;++id) {
    GemmBatchChoice candidate=choice;candidate.id=id;
    if(choice.family==1) {
      if(in!=out&&!kGemvNn[id].tile_n)continue;
      candidate.splits=kGemvNn[id].tile_n?
          gemmNnSplits(m,n,k,batches,kGemvNn[id].tile_n):1;
    } else if(choice.family==2) {
      if(!gemmBatchedWmmaLegal(kBatchedWmma[id],m,n,k,batches,ta,tb))continue;
      candidate.splits=kBatchedWmma[id].splits;
    }
    const float ms=gemmTimeConfigPeak(stream,begin,end,[&]{checked(candidate);},3,5);
    if(ms<best_ms){best_ms=ms;best=candidate;}
  }
  if(hipStreamSynchronize(stream)!=hipSuccess)error=1;
  if(hipEventDestroy(begin)!=hipSuccess)error=1;
  if(hipEventDestroy(end)!=hipSuccess)error=1;
  if(error)return error;
  if(!(best_ms<1e30f))return 1;
  choice=best;choice.online=true;
  return 0;
}

extern "C" int hip_gemm_strided_batched(void*stream,const void*a,const void*b,const void*c,
    void*y,int m,int n,int k,float alpha,float beta,int ta,int tb,int in,int out,
    int crows,int ccols,int batches,int64_t sa,int64_t sb,int64_t sc,int64_t sy,
    int ag,int bg,void*workspace,size_t workspace_bytes) {
  size_t required=0;
  if(hip_gemm_workspace_size(m,n,k,ta,tb,in,out,batches,ag,bg,&required)||
     sa<0||sb<0||sc<0||sy<0||(c&&((crows!=1&&crows!=m)||(ccols!=1&&ccols!=n))))return -1;
  if(!m||!n||!batches)return 0;
  if(!stream||(k&&(!a||!b))||!y||(required&&(!workspace||workspace_bytes<required||
      reinterpret_cast<uintptr_t>(workspace)%alignof(float))))return -1;
  const size_t ibytes=in==2?8:in==1?4:2, obytes=out==2?8:out==1?4:2;
  auto extent=[](int count,int group,int64_t stride,uint64_t matrix,size_t element,size_t& bytes) {
    const uint64_t last=uint64_t(count/group-1);
    const uint64_t limit=uint64_t(PTRDIFF_MAX)/element;
    if(matrix>limit || (last&&uint64_t(stride)>(limit-matrix)/last))return false;
    bytes=size_t(last*uint64_t(stride)+matrix)*element;return true;
  };
  size_t abytes,bbytes,cbytes=0,ybytes;
  if ((batches>1&&uint64_t(sy)<uint64_t(m)*n) ||
      !extent(batches,ag,sa,uint64_t(m)*k,ibytes,abytes) ||
      !extent(batches,bg,sb,uint64_t(n)*k,ibytes,bbytes) ||
      !extent(batches,1,sy,uint64_t(m)*n,obytes,ybytes) ||
      (c&&!extent(batches,1,sc,uint64_t(crows)*ccols,obytes,cbytes)))return -1;
  auto overlaps=[](const void* p,size_t bytes,const void* q,size_t other) {
    const uintptr_t pp=reinterpret_cast<uintptr_t>(p),qq=reinterpret_cast<uintptr_t>(q);
    return p&&q&&bytes&&other&&(pp<qq?qq-pp<bytes:pp-qq<other);
  };
  if(overlaps(y,ybytes,a,abytes)||overlaps(y,ybytes,b,bbytes)||
     (c&&beta!=0.f&&overlaps(y,ybytes,c,cbytes)&&
      (c!=y||crows!=m||ccols!=n||sc!=sy)) ||
     (required&&(overlaps(workspace,required,a,abytes)||overlaps(workspace,required,b,bbytes)||
      overlaps(workspace,required,c,cbytes)||overlaps(workspace,required,y,ybytes))))return -1;
  if(k&&in==out) {
#ifndef HIPDNN_UT_RANK_GEMM_ADAPTER
    const GemmBatchKey gkey=gemmBatchCacheKey(m,n,k,ta,tb,in,out,batches,ag,bg,crows,ccols,
        sa,sb,sc,sy,alpha,beta,c!=nullptr);
    int gid=-2;
    {
      std::lock_guard<std::mutex> lock(g_tune_mu);
      auto it=g_gemm_gen_cache.find(gkey);
      if(it!=g_gemm_gen_cache.end())gid=it->second;
    }
    if(gid==-2) {
      gid=gemmGenSelect(m,n,k,ta,tb,in,out,batches,ag,bg,c,crows,ccols,beta);
      std::lock_guard<std::mutex> lock(g_tune_mu);
      g_gemm_gen_cache.emplace(gkey,gid);
      if(gid>=0) {
        const auto& g=gemm_gen::kCfg[gid];
        CUSTOM_KERNELS_DEBUG_LOG("[gemm] general final M=%d N=%d K=%d in=%d B=%d tb=%d -> cfg=%d family=%d bm=%d bn=%d bk=%d tm=%d tn=%d th=%d splits=%d stagger=%d/%d tile_n=%d\n",
            m,n,k,in,batches,tb,gid,g.family,g.bm,g.bn,g.bk,g.wt_m,g.wt_n,g.threads,g.splits,g.smod,g.stiles,g.tile_n);
      }
    }
#else
    const int gid=gemmGenSelect(m,n,k,ta,tb,in,out,batches,ag,bg,c,crows,ccols,beta);
#endif
    // F16 general kernels read 16-byte buffer chunks from A and B: their bases
    // and batch strides must be 16-byte aligned; otherwise keep the existing route.
    const bool aligned=in!=0||(((reinterpret_cast<uintptr_t>(a)|reinterpret_cast<uintptr_t>(b))&15)==0&&
        (batches==1||((sa|sb)&7)==0));
    if(gid>=0&&aligned) {
      if(gemm_gen::workspaceBytes(gemm_gen::kCfg[gid],m,n,batches)>required)return -1;
      return gemm_gen::launch(gid,static_cast<hipStream_t>(stream),a,b,c,y,m,n,k,tb,alpha,
          beta!=0.f?1:0,batches,sa,sb,sc,sy,workspace);
    }
  }
  GemmBatchChoice choice;
  choice.family=gemmBatchFamily(m,n,k,ta,tb,in,out,batches,ag,bg);
#ifndef HIPDNN_UT_RANK_GEMM_ADAPTER
  const GemmBatchKey key=gemmBatchCacheKey(m,n,k,ta,tb,in,out,batches,ag,bg,crows,ccols,
      sa,sb,sc,sy,alpha,beta,gemmSweepSafe(c,y,beta));
  bool cached=false;
  if(k && choice.family>0 && choice.family<4) {
    std::lock_guard<std::mutex> lock(g_tune_mu);
    auto it=g_gemm_batch_cache.find(key);
    if(it!=g_gemm_batch_cache.end()){choice=it->second;cached=true;}
  }
  if(!cached && choice.family>0 && choice.family<4)
    choice=gemmBatchChoice(m,n,k,ta,tb,in,out,batches,ag,bg);
#else
  choice=gemmBatchChoice(m,n,k,ta,tb,in,out,batches,ag,bg);
#endif
  if(choice.family<0)return -1;
  if(choice.family==0)
    return hip_gemm(stream,a,b,c,y,m,n,k,alpha,beta,ta,tb,in,crows,ccols);
  if(in==2||choice.family==4) {
    // Retain FP64 accumulation and the existing ABI's correct fallback.
    for(int batch=0;batch<batches;++batch) {
      int rc=hip_gemm(stream,a?static_cast<const char*>(a)+size_t(batch/ag)*sa*ibytes:nullptr,
          b?static_cast<const char*>(b)+size_t(batch/bg)*sb*ibytes:nullptr,
          c?static_cast<const char*>(c)+size_t(batch)*sc*obytes:nullptr,
          static_cast<char*>(y)+size_t(batch)*sy*obytes,m,n,k,alpha,beta,ta,tb,in,crows,ccols);
      if(rc)return rc;
    }
    return 0;
  }
  auto run=[&](auto input,auto output) {
    using T=decltype(input);using Out=decltype(output);
    auto aa=static_cast<const T*>(a),bb=static_cast<const T*>(b);
    auto cc=static_cast<const Out*>(c);auto yy=static_cast<Out*>(y);
    auto launch=[&](const GemmBatchChoice& selected) {
      float*partial=selected.splits>1?static_cast<float*>(workspace):nullptr;
#define GB(TA,TB) gemmBatchTyped<T,Out,TA,TB>(selected,static_cast<hipStream_t>(stream),aa,bb,cc,yy, \
    m,n,k,alpha,beta,crows,ccols,batches,sa,sb,sc,sy,ag,bg,partial)
      if(ta)return tb?GB(true,true):GB(true,false);
      return tb?GB(false,true):GB(false,false);
#undef GB
    };
#ifndef HIPDNN_UT_RANK_GEMM_ADAPTER
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_BATCH_CFG",-1);
    const int count=choice.family==1?kNumGemvNn:choice.family==2?kNumBatchedWmma:kNumBatchedFma;
    if(k&&!cached&&forced>=0) {
      if(forced>=count)return -1;
      if(choice.family==1) {
        if(in!=out&&!kGemvNn[forced].tile_n)return -1;
        choice.splits=kGemvNn[forced].tile_n?
            gemmNnSplits(m,n,k,batches,kGemvNn[forced].tile_n):1;
      } else if(choice.family==2) {
        if(!gemmBatchedWmmaLegal(kBatchedWmma[forced],m,n,k,batches,ta,tb))return -1;
        choice.splits=kBatchedWmma[forced].splits;
      }
      choice.id=forced;
    } else if(k&&!cached&&!gemmLutLookupEnabled()&&gemmAutotuneOn()&&
              gemmSweepSafe(c,y,beta)) {
      int rc=gemmBatchTune(choice,static_cast<hipStream_t>(stream),m,n,k,in,out,batches,ta,tb,launch);
      if(rc)return rc;
    }
    if(k&&!cached) {
      std::lock_guard<std::mutex> lock(g_tune_mu);
      g_gemm_batch_cache.emplace(key,choice);
    }
#endif
#ifndef HIPDNN_UT_RANK_GEMM_ADAPTER
    if(!cached)gemmBatchLog(choice,m,n,k,in,out,batches,ag,bg);
#else
    static thread_local std::string logged;
    const std::string selected=std::to_string(m)+":"+std::to_string(n)+":"+
        std::to_string(k)+":"+std::to_string(in)+":"+std::to_string(out)+":"+
        std::to_string(batches)+":"+std::to_string(ag)+":"+std::to_string(bg)+":"+
        std::to_string(choice.family)+":"+std::to_string(choice.id);
    if(logged!=selected){gemmBatchLog(choice,m,n,k,in,out,batches,ag,bg);logged=selected;}
#endif
    return launch(choice);
  };
  if(in==0)return out==1?run(_Float16{},float{}):run(_Float16{},_Float16{});
  if(in==3)return out==1?run(__hip_bfloat16{},float{}):run(__hip_bfloat16{},__hip_bfloat16{});
  return run(float{},float{});
}
