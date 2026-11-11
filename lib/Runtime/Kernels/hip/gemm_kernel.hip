/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX Gemm: Y = alpha * op(A) * op(B) + beta * C
 *
 * Independent of gemm_wmma_kernel.hip. The inner loop is the same LDS + WMMA
 * pipeline as matmul_nbits_kernel.hip (double-buffered tiles, ds_read_b128
 * padding, LDS-only barrier, register prefetch, grid swizzle). Decode
 * (small M) follows llama.cpp's VEC path: do not run WMMA when M < 16;
 * K-parallel GEMV with A loaded once and reused across TILE_N B rows.
 * Split-K (grid.z) plus occupancy-aware tiles keep workgroups on gfx1151's
 * ~40 CUs when ceil(M/BM)*ceil(N/BN) would otherwise underfill the GPU.
 *
 * Layout (ONNX row-major):
 *   transA=0: A[M,K] lda=K     transA=1: A[K,M] lda=M
 *   transB=0: B[K,N] ldb=N     transB=1: B[N,K] ldb=K   (NT, Linear weights)
 *   Y[M,N] ldc=N
 *
 * Accumulator is always fp32 (matches hipBLASLt HIPBLAS_COMPUTE_32F).
 */

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"
#include "debug_log.h"
#include "autotune/common/hipdnn_dtype.h"
#include "autotune/gemm/gemm_autotune.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef unsigned short bf16xN
    __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

#if !HIPDNN_HAS_WMMA
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#undef HIPDNN_WMMA_F32_16X16X16_F16
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                                 \
  hipdnn_wmma_f32_16x16x16_f16_unavailable((a), (b), (c))
#endif

#if !HIPDNN_HAS_WMMA_BF16
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_bf16_unavailable(bf16xN, bf16xN, float8 c) {
  __builtin_trap();
  return c;
}
#define HIPDNN_WMMA_F32_16X16X16_BF16(a, b, c)                                \
  hipdnn_wmma_f32_16x16x16_bf16_unavailable((a), (b), (c))
#endif

#define WMMA_TILE 16

/* LDS-only workgroup barrier. Same rationale as matmul_nbits_kernel.hip:
 * __syncthreads() also invalidates L0; the K-loop only syncs shared memory. */
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) &&                 \
    !defined(__GFX12__)
#define HIPDNN_LDS_BARRIER()                                                    \
  __asm__ __volatile__("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
#else
#define HIPDNN_LDS_BARRIER() __syncthreads()
#endif

/* Compiler scheduling fence (no instruction moves across it). */
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define GEMM_SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)
#else
#define GEMM_SCHED_FENCE() ((void)0)
#endif

static constexpr int kTypeFloat16 = 0;
static constexpr int kTypeFloat32 = 1;
static constexpr int kTypeFloat64 = 2;
static constexpr int kTypeBFloat16 = 3;

/* ---- element traits ---- */

template <typename T> struct GemmElem;

template <> struct GemmElem<_Float16> {
  __device__ static inline float tof(_Float16 x) { return (float)x; }
  __device__ static inline _Float16 fromf(float x) { return (_Float16)x; }
};

template <> struct GemmElem<__hip_bfloat16> {
  __device__ static inline float tof(__hip_bfloat16 x) {
    return __bfloat162float(x);
  }
  __device__ static inline __hip_bfloat16 fromf(float x) {
    return __float2bfloat16(x);
  }
};

template <> struct GemmElem<float> {
  __device__ static inline float tof(float x) { return x; }
  __device__ static inline float fromf(float x) { return x; }
};

template <> struct GemmElem<double> {
  __device__ static inline float tof(double x) { return (float)x; }
  __device__ static inline double fromf(float x) { return (double)x; }
};

/* Maps each element type this kernel dispatches on to the shared HipdnnDType
 * vocabulary the LUT keys on (gemm_autotune.fbs / hipdnn_dtype.h). This
 * replaces the old `sizeof(T)` element-size key, which could not tell fp16
 * from bf16 (both 2 bytes) -- see plan.md §1 D2. gemm has no activation/weight
 * split yet (hip_gemm takes a single `type_code`), so act == wts == out for
 * every call site today; the three-field Request already has room for that to
 * change without another schema break (plan.md §2.2). */
template <typename T> struct GemmDTypeOf;
template <> struct GemmDTypeOf<_Float16> {
  static constexpr hipdnn_ep::common::HipdnnDType value =
      hipdnn_ep::common::HipdnnDType::F16;
};
template <> struct GemmDTypeOf<__hip_bfloat16> {
  static constexpr hipdnn_ep::common::HipdnnDType value =
      hipdnn_ep::common::HipdnnDType::BF16;
};
template <> struct GemmDTypeOf<float> {
  static constexpr hipdnn_ep::common::HipdnnDType value =
      hipdnn_ep::common::HipdnnDType::F32;
};
template <> struct GemmDTypeOf<double> {
  static constexpr hipdnn_ep::common::HipdnnDType value =
      hipdnn_ep::common::HipdnnDType::F64;
};
template <typename T> constexpr hipdnn_ep::common::HipdnnDType gemmDType() {
  return GemmDTypeOf<T>::value;
}

template <typename T>
__device__ __forceinline__ float gemmLoadC(const T *C, int r, int c,
                                            int cDim0, int cDim1) {
  if (!C)
    return 0.0f;
  const int cr = (cDim0 == 1) ? 0 : r;
  const int cc = (cDim1 == 1) ? 0 : c;
  return GemmElem<T>::tof(C[static_cast<size_t>(cr) * cDim1 + cc]);
}

template <typename T>
__device__ __forceinline__ void gemmStoreY(T *Y, const T *C, int r, int c,
                                           int N, int cDim0, int cDim1,
                                           float acc, float alpha, float beta) {
  float v = acc * alpha;
  if (beta != 0.0f && C)
    v = fmaf(beta, gemmLoadC(C, r, c, cDim0, cDim1), v);
  Y[static_cast<size_t>(r) * N + c] = GemmElem<T>::fromf(v);
}

__device__ __forceinline__ void gemmSwizzleBlock(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

/* =========================================================================
 * GEMV — llama.cpp VEC / matmul_nbits decode path
 *
 * WMMA 16x16 tiles waste most of the M dimension when M < 16 (llama.cpp
 * fattn: decode uses VEC, not WMMA). Two layouts:
 *
 *   NT (transB=1): B rows are K-contiguous. K-parallel reduction, TILE_N
 *   output columns per block, A loaded once per K-step (matmul_nbits).
 *
 *   NN (transB=0): B[k,n] is coalesced across consecutive n at a fixed k.
 *   N-parallel: one output column per thread, walk K. llama.cpp mmf-style.
 * ========================================================================= */

template <typename T, int BLOCK_SIZE, int TILE_N>
__global__ void __launch_bounds__(BLOCK_SIZE)
    gemm_gemv_nt_kernel(const T *__restrict__ A, const T *__restrict__ B,
                       const T *__restrict__ C, T *__restrict__ Y, int M,
                       int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  constexpr int WARP = HIPDNN_WAVE_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP;

  const int n_base = blockIdx.x * TILE_N;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  if (m >= M)
    return;

  const T *a_base = A + static_cast<size_t>(m) * K;
  const T *b_row[TILE_N];
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn) {
    const int n = n_base + tn;
    const int sn = (n < N) ? n : 0;
    b_row[tn] = B + static_cast<size_t>(sn) * K;
  }

  float partial[TILE_N];
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn)
    partial[tn] = 0.0f;

  const int num_vec = K / 8;
  for (int idx = tid; idx < num_vec; idx += BLOCK_SIZE) {
    const int k = idx * 8;
    float af[8];
    if constexpr (std::is_same<T, _Float16>::value) {
      u32x4 a_pk = *reinterpret_cast<const u32x4 *>(a_base + k);
      const _Float16 *ah = reinterpret_cast<const _Float16 *>(&a_pk);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        af[i] = (float)ah[i];
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        af[i] = GemmElem<T>::tof(a_base[k + i]);
    }

#pragma unroll
    for (int tn = 0; tn < TILE_N; ++tn) {
      if constexpr (std::is_same<T, _Float16>::value) {
        u32x4 b_pk = *reinterpret_cast<const u32x4 *>(b_row[tn] + k);
        const _Float16 *bh = reinterpret_cast<const _Float16 *>(&b_pk);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          partial[tn] = __fmaf_rn(af[i], (float)bh[i], partial[tn]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          partial[tn] = __fmaf_rn(af[i], GemmElem<T>::tof(b_row[tn][k + i]),
                                  partial[tn]);
      }
    }
  }

  const int k8 = num_vec * 8;
  for (int k = k8 + tid; k < K; k += BLOCK_SIZE) {
    const float av = GemmElem<T>::tof(a_base[k]);
#pragma unroll
    for (int tn = 0; tn < TILE_N; ++tn)
      partial[tn] =
          __fmaf_rn(av, GemmElem<T>::tof(b_row[tn][k]), partial[tn]);
  }

  const int lane = tid % WARP;
  const int warp = tid / WARP;
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn) {
    for (int off = WARP >> 1; off > 0; off >>= 1)
      partial[tn] += __shfl_down(partial[tn], off);
  }

  if constexpr (BLOCK_SIZE <= WARP) {
    if (lane == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn) {
        const int n = n_base + tn;
        if (n < N)
          gemmStoreY(Y, C, m, n, N, cDim0, cDim1, partial[tn], alpha, beta);
      }
    }
  } else {
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    if (lane == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn)
        warp_sums[tn * NUM_WARPS + warp] = partial[tn];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn) {
        float val = (lane < NUM_WARPS) ? warp_sums[tn * NUM_WARPS + lane] : 0.f;
        for (int off = WARP >> 1; off > 0; off >>= 1)
          val += __shfl_down(val, off);
        if (lane == 0) {
          const int n = n_base + tn;
          if (n < N)
            gemmStoreY(Y, C, m, n, N, cDim0, cDim1, val, alpha, beta);
        }
      }
    }
  }
}

/* NN GEMV: coalesced B[k, n] across the warp. A is reused from registers
 * along K; one column per thread. */
template <typename T, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
    gemm_gemv_nn_kernel(const T *__restrict__ A, const T *__restrict__ B,
                       const T *__restrict__ C, T *__restrict__ Y, int M,
                       int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  const int n = blockIdx.x * BLOCK + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  const T *a = A + static_cast<size_t>(m) * K;
  const T *b = B + n; /* B[k,n] at k*N + n */
  const size_t ldb = static_cast<size_t>(N);
  float acc = 0.0f;
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    acc = __fmaf_rn(GemmElem<T>::tof(a[k]), GemmElem<T>::tof(b[k * ldb]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 1]),
                    GemmElem<T>::tof(b[(k + 1) * ldb]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 2]),
                    GemmElem<T>::tof(b[(k + 2) * ldb]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 3]),
                    GemmElem<T>::tof(b[(k + 3) * ldb]), acc);
  }
  for (; k < K; ++k)
    acc = __fmaf_rn(GemmElem<T>::tof(a[k]), GemmElem<T>::tof(b[k * ldb]), acc);

  gemmStoreY(Y, C, m, n, N, cDim0, cDim1, acc, alpha, beta);
}

/* transA=1 GEMV: A is [K,M], column m is strided by M. Rare; keep correct. */
template <typename T>
__global__ void gemm_gemv_ta_kernel(const T *__restrict__ A, const T *__restrict__ B,
                                   const T *__restrict__ C, T *__restrict__ Y,
                                   int M, int N, int K, int transB, float alpha,
                                   float beta, int cDim0, int cDim1) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    const float av = GemmElem<T>::tof(A[static_cast<size_t>(k) * M + m]);
    const float bv = transB ? GemmElem<T>::tof(B[static_cast<size_t>(n) * K + k])
                            : GemmElem<T>::tof(B[static_cast<size_t>(k) * N + n]);
    acc = __fmaf_rn(av, bv, acc);
  }
  gemmStoreY(Y, C, m, n, N, cDim0, cDim1, acc, alpha, beta);
}

/* Calls f(std::integral_constant<int, i>) for i = I .. N-1: unrolled loops
 * whose register-array indices must be compile-time. */
template <int I, int N, typename F>
__device__ __forceinline__ void gemmStaticFor(F &&f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    gemmStaticFor<I + 1, N>(f);
  }
}

/* 8 packed 16-bit elements (one 16-byte load) -> float. bf16 is the top
 * half of an fp32, so it widens with a shift instead of a conversion. */
template <typename T>
__device__ __forceinline__ void gemmUnpack8(const u32x4 &v, float (&f)[8]) {
  if constexpr (std::is_same<T, _Float16>::value) {
    const _Float16 *h = reinterpret_cast<const _Float16 *>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      f[i] = (float)h[i];
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[2 * i] = __uint_as_float(v[i] << 16);
      f[2 * i + 1] = __uint_as_float(v[i] & 0xffff0000u);
    }
  }
}

/* 8 consecutive elements at p -> float: one 16-byte load for the 16-bit types,
 * two for fp32. p must be 16-byte aligned. */
template <typename T>
__device__ __forceinline__ void gemmLoad8(const T *p, float (&f)[8]) {
  if constexpr (std::is_same<T, float>::value) {
    const float4 lo = reinterpret_cast<const float4 *>(p)[0];
    const float4 hi = reinterpret_cast<const float4 *>(p)[1];
    f[0] = lo.x;
    f[1] = lo.y;
    f[2] = lo.z;
    f[3] = lo.w;
    f[4] = hi.x;
    f[5] = hi.y;
    f[6] = hi.z;
    f[7] = hi.w;
  } else {
    gemmUnpack8<T>(*reinterpret_cast<const u32x4 *>(p), f);
  }
}

/* Multi-row NT GEMV (fp16/bf16/fp32, 2 <= M < 16, B = [N, K]): one block owns
 * TN output columns for ALL M rows, so B -- the large operand -- is streamed
 * once (16-byte loads, 8 k per thread per chunk) instead of once per row as
 * in gemm_gemv_nt_kernel. TA=false: A is [M, K], each row's 8-k run is one
 * 16-byte load (L2-resident, A is tiny). TA=true (16-bit only): A is [K, M];
 * for a chunk of KC k-rows the slice A[kc .. kc+KC)[0 .. M) is one contiguous
 * run, copied to LDS with 16-byte loads and read back per (k, m). Requires
 * M <= MR. */
template <typename T, bool TA, int MR, int TN>
__global__ void __launch_bounds__(128)
    gemm_gemv_mr_nt_kernel(const T *__restrict__ A, const T *__restrict__ B,
                           const T *__restrict__ C, T *__restrict__ Y, int M,
                           int N, int K, float alpha, float beta, int cDim0,
                           int cDim1) {
  constexpr int WARP = HIPDNN_WAVE_SIZE;
  constexpr int BLOCK = 128;
  constexpr int KC = BLOCK * 8;
  constexpr int NW = BLOCK / WARP;
  __shared__ __align__(16) T smA[TA ? KC * MR : 8];
  __shared__ float red[NW][MR * TN];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * TN;

  const T *brow[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn)
    brow[tn] = B + static_cast<size_t>(n0 + tn < N ? n0 + tn : 0) * K;
  const bool kvec = (K % 8) == 0;

  float acc[MR][TN];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      acc[m][tn] = 0.0f;

  for (int kc = 0; kc < K; kc += KC) {
    const int klen = (K - kc < KC) ? K - kc : KC;
    if constexpr (TA) {
      /* kc*M*2 bytes = 2048*M*(kc/KC): every chunk start is 16-byte
       * aligned. */
      const T *src = A + static_cast<size_t>(kc) * M;
      const int cnt = klen * M;
      const int cnt8 = cnt / 8;
      for (int i = tid; i < cnt8; i += BLOCK)
        reinterpret_cast<u32x4 *>(smA)[i] =
            reinterpret_cast<const u32x4 *>(src)[i];
      for (int i = cnt8 * 8 + tid; i < cnt; i += BLOCK)
        smA[i] = src[i];
      __syncthreads();
    }

    const int k = tid * 8;
    if (k < klen) {
      const bool full = kvec && k + 8 <= klen;
      float bf[TN][8];
      if (full) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
          gemmLoad8<T>(brow[tn] + kc + k, bf[tn]);
      } else {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            bf[tn][i] = (k + i < klen) ? GemmElem<T>::tof(brow[tn][kc + k + i])
                                       : 0.0f;
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        if (m < M) {
          float af[8];
          if constexpr (TA) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
              af[i] = (k + i < klen) ? GemmElem<T>::tof(smA[(k + i) * M + m])
                                     : 0.0f;
          } else {
            const T *arow = A + static_cast<size_t>(m) * K + kc + k;
            if (full) {
              gemmLoad8<T>(arow, af);
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i)
                af[i] = (k + i < klen) ? GemmElem<T>::tof(arow[i]) : 0.0f;
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
              acc[m][tn] = __fmaf_rn(af[i], bf[tn][i], acc[m][tn]);
        }
      }
    }
    if constexpr (TA)
      __syncthreads();
  }

  const int lane = tid % WARP;
  const int warp = tid / WARP;
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      float v = acc[m][tn];
      for (int off = WARP >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
      if (lane == 0)
        red[warp][m * TN + tn] = v;
    }
  __syncthreads();
  if (tid < MR * TN) {
    const int m = tid / TN;
    const int n = n0 + tid % TN;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w)
        v += red[w][tid];
      gemmStoreY(Y, C, m, n, N, cDim0, cDim1, v, alpha, beta);
    }
  }
}

/* B = [K, N] GEMV (fp16/bf16/fp32, small M): each thread owns 8 consecutive
 * output columns (one 16-byte B load per k-row, two for fp32) and a 1/KS share
 * of K; the KS
 * partial sums meet in LDS. A block covers MR rows, so B is re-read only
 * ceil(M/MR) times. TA selects the A layout ([K, M] if set, else [M, K]). */
template <typename T, bool TA, int MR, int BLOCK, int KS>
__global__ void __launch_bounds__(BLOCK)
    gemm_gemv_kn_kernel(const T *__restrict__ A, const T *__restrict__ B,
                        const T *__restrict__ C, T *__restrict__ Y, int M,
                        int N, int K, float alpha, float beta, int cDim0,
                        int cDim1) {
  constexpr int CG = BLOCK / KS;
  __shared__ float red[KS][CG * 8];
  const int tid = threadIdx.x;
  const int cg = tid % CG;
  const int ks = tid / CG;
  const int nblk = blockIdx.x * CG * 8;
  const int n0 = nblk + cg * 8;
  const int m0 = blockIdx.y * MR;
  const bool vec = (N % 8) == 0 && n0 + 8 <= N;

  float acc[MR][8];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc[m][e] = 0.0f;

#pragma unroll 2
  for (int k = ks; k < K; k += KS) {
    float b[8];
    const T *brow = B + static_cast<size_t>(k) * N + n0;
    if (vec) {
      gemmLoad8<T>(brow, b);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        b[e] = (n0 + e < N) ? GemmElem<T>::tof(brow[e]) : 0.0f;
    }
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      if (m0 + m < M) {
        const float a =
            GemmElem<T>::tof(TA ? A[static_cast<size_t>(k) * M + m0 + m]
                                : A[static_cast<size_t>(m0 + m) * K + k]);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[m][e] = __fmaf_rn(a, b[e], acc[m][e]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MR; ++m) {
    if (m0 + m >= M)
      break;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      red[ks][cg * 8 + e] = acc[m][e];
    __syncthreads();
    for (int i = tid; i < CG * 8; i += BLOCK) {
      const int n = nblk + i;
      if (n < N) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < KS; ++j)
          v += red[j][i];
        gemmStoreY(Y, C, m0 + m, n, N, cDim0, cDim1, v, alpha, beta);
      }
    }
    __syncthreads();
  }
}

/* B = [K, N] GEMV with a register ring (fp16/bf16/fp32, M <= MR, NN only).
 * A block is KW waves over NB = LPR * VE output columns (VE = elements per
 * 16-byte load). In a wave, LPR adjacent lanes read one k-row's NB columns
 * as one contiguous run (LPR = 8: a whole 128-byte line at 16 bits; LPR = 32
 * at fp32: 512 bytes) and the R = 32 / LPR lane rows read different k-rows.
 * A lane owns U consecutive k-rows of a group of GR = R * U rows, so its A
 * values for one row are one U-element vector load. Groups go round-robin
 * to the KW waves (adjacent waves stream adjacent rows). Each wave keeps NS
 * groups in flight in a statically indexed register ring: the loop is
 * unrolled by NS and refills the slot it has just consumed, and the
 * prologue and the last refill load unconditionally (clamped to the last
 * group, never consumed) -- a load that may or may not have been issued
 * makes the compiler wait for every outstanding load at the loop head.
 * gemm_gemv_kn_kernel waits for each 16-byte load right after issuing it
 * (vmcnt(0) per load, one load in flight per wave). The R lane rows meet by
 * xor shuffles, the KW waves in LDS once at the end. One lane computes MR
 * rows from the same B registers; rows past M re-read row 0 and are not
 * stored. Requires K % GR == 0, N % VE == 0, N >= VE and 16-byte aligned A
 * and B (launchGemvKnRing checks). */
template <typename T, int MR, int LPR, int KW, int U, int NS>
__global__ void __launch_bounds__(KW * 32)
    gemm_gemv_kn_ring_kernel(const T *__restrict__ A, const T *__restrict__ B,
                             const T *__restrict__ C, T *__restrict__ Y, int M,
                             int N, int K, float alpha, float beta, int cDim0,
                             int cDim1) {
  constexpr int VE = 16 / static_cast<int>(sizeof(T));
  constexpr int R = 32 / LPR;
  constexpr int NB = LPR * VE;
  constexpr int GR = R * U;
  constexpr int AW = U * static_cast<int>(sizeof(T)) / 4; /* A dwords */
  static_assert(R * LPR == 32 && NS >= 2, "kn ring shape");
  static_assert(AW >= 1 && (AW <= 2 || AW % 4 == 0), "kn ring A width");
  __shared__ float red[KW > 1 ? KW : 1][MR][NB];

  const int lid = threadIdx.x % 32;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int cl = lid % LPR;
  const int rl = lid / LPR;
  const int nblk = blockIdx.x * NB;
  const int m0 = blockIdx.y * MR;
  const int n0 = nblk + cl * VE;
  const int nc = (n0 + VE <= N) ? n0 : N - VE;
  const size_t ldn = static_cast<size_t>(N);
  const T *bp = B + static_cast<size_t>(rl * U) * ldn + nc;
  const T *ap[MR];
#pragma unroll
  for (int m = 0; m < MR; ++m)
    ap[m] = A + static_cast<size_t>(m0 + m < M ? m0 + m : 0) * K + rl * U;

  const int ng = K / GR;
  const int nw = ng > w ? (ng - w + KW - 1) / KW : 0;

  u32x4 rb[NS][U];
  uint32_t ra[NS][MR][AW];
  float acc[MR][VE];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int e = 0; e < VE; ++e)
      acc[m][e] = 0.0f;

  /* the wave's j-th group is group w + j * KW */
  auto load = [&](int j, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    const int g = (w + j * KW < ng) ? w + j * KW : ng - 1;
    const T *b = bp + static_cast<size_t>(g) * GR * ldn;
#pragma unroll
    for (int u = 0; u < U; ++u)
      rb[S][u] = *reinterpret_cast<const u32x4 *>(b + u * ldn);
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      const T *src = ap[m] + g * GR;
      if constexpr (AW % 4 == 0) {
#pragma unroll
        for (int v = 0; v < AW / 4; ++v)
          *reinterpret_cast<u32x4 *>(&ra[S][m][4 * v]) =
              reinterpret_cast<const u32x4 *>(src)[v];
      } else if constexpr (AW == 2) {
        *reinterpret_cast<u32x2 *>(&ra[S][m][0]) =
            *reinterpret_cast<const u32x2 *>(src);
      } else {
        ra[S][m][0] = *reinterpret_cast<const uint32_t *>(src);
      }
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float b[VE];
      if constexpr (std::is_same<T, float>::value) {
#pragma unroll
        for (int e = 0; e < VE; ++e)
          b[e] = __uint_as_float(rb[S][u][e]);
      } else {
        gemmUnpack8<T>(rb[S][u], b);
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const T *av = reinterpret_cast<const T *>(&ra[S][m][0]);
        const float a = GemmElem<T>::tof(av[u]);
#pragma unroll
        for (int e = 0; e < VE; ++e)
          acc[m][e] = __fmaf_rn(a, b[e], acc[m][e]);
      }
    }
  };
  gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
    load(decltype(s)::value, s);
  });
  int j = 0;
  for (; j + 2 * NS <= nw; j += NS) {
    gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
      compute(s);
      load(j + NS + decltype(s)::value, s);
      GEMM_SCHED_FENCE();
    });
  }
  if (j + NS <= nw) {
    gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
      compute(s);
      load(j + NS + decltype(s)::value, s);
      GEMM_SCHED_FENCE();
    });
    j += NS;
  }
  gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
    if (j + decltype(s)::value < nw)
      compute(s);
  });

  /* the R lane rows of a column -> lane row 0 */
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int e = 0; e < VE; ++e) {
      float v = acc[m][e];
#pragma unroll
      for (int off = LPR; off < 32; off <<= 1)
        v += __shfl_xor(v, off);
      acc[m][e] = v;
    }
  if (rl == 0) {
#pragma unroll
    for (int m = 0; m < MR; ++m)
#pragma unroll
      for (int e = 0; e < VE; ++e)
        red[w][m][cl * VE + e] = acc[m][e];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < MR * NB; i += KW * 32) {
    const int m = i / NB, c = i % NB;
    const int n = nblk + c;
    if (m0 + m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < KW; ++q)
        v += red[q][m][c];
      gemmStoreY(Y, C, m0 + m, n, N, cDim0, cDim1, v, alpha, beta);
    }
  }
}

/* =========================================================================
 * Skinny-M WMMA (fp16/bf16, small M; NT, NN (B = [K, N]) and transA).
 *
 * On gfx11 a WMMA operand lane holds one row's 16 consecutive k; for a
 * row-major A[M, K] and an NT weight B[N, K] that is one contiguous 32-byte
 * run, so the A lanes load their fragments straight from global memory. A
 * workgroup is KW waves that share one (16*MT) x (16*NTL) output tile and
 * split K between them (wave w takes the KC-wide k-chunks w, w+KW, ...:
 * together the KW waves stream adjacent bytes of each B row), with no
 * workgroup barrier in the K loop. One register double buffer keeps the next
 * chunk's loads in flight during the WMMAs; the KW partial tiles meet in LDS
 * at the end. Rows past M (and columns past N) read row 0: their outputs are
 * never stored, and WMMA output rows / columns do not mix, so below 16 rows
 * the padded MACs are free on this bandwidth-bound path. Workgroups of the
 * same N-tile (M-tiles) are adjacent in dispatch order and share the B panel
 * in L2. StaggerU (stag, as in GemmWmmaImpl, keyed by N-tile) rotates each
 * N-tile's chunk order.
 *
 * B either loads straight into fragments like A (SB = false: per instruction
 * 16 rows x 16 bytes, with the two half-waves loading the same data) or, with
 * SB, as a plain row copy through a wave-private LDS tile: SEG lanes read one
 * row's KC halves as consecutive 16-byte pieces, so every load instruction
 * covers whole 64/128-byte row runs and no register holds duplicate data. The
 * tile is private to the wave and the LDS operations of one wave complete in
 * order, so this needs no barrier either.
 *
 * An operand whose k is not contiguous (NN B = [K, N], transA A = [K, M]) is
 * read as k-row runs -- each lane 4 k-rows x CW columns -- and transposed
 * through the same wave-private LDS tile with packed 8-byte stores (the
 * packing of GemmWmmaImpl's TransA staging). Requires 16-byte aligned A/B and
 * K % 8 == 0 for a k-contiguous operand (checked by the launcher).
 * ========================================================================= */
template <typename T, int F, int KC, bool L128 = false>
struct SkinnyT {
  /* One wave stages a KC x (16*F) block of a [k][c] matrix in R passes: in
   * pass p, lane slot p*32+lid = (kg, cg) loads rows kg*4 .. +4, columns
   * cg*CW .. +CW. Default: one pass, CW = F*KC/8. L128 caps CW at 8 (16
   * bytes) so the CG lanes of one k-row read it as one contiguous run -- a
   * whole 128-byte line at 64 columns -- at the cost of a second pass. */
  static constexpr int CW = (!L128 || F * KC / 8 < 8) ? F * KC / 8 : 8;
  static constexpr int CG = CW > 0 ? 16 * F / CW : 1;
  static constexpr int R = CG * (KC / 4) / 32;
  static constexpr int STR = KC + 8; /* LDS row stride (halfs), 16 B aligned */
  static constexpr bool ok = (CW == 4 || CW == 8 || CW == 16) && R >= 1 &&
                             R * 32 == CG * (KC / 4);
};

template <typename T, int F, int KC, bool L128>
using SkinnyReg = uint32_t[SkinnyT<T, F, KC, L128>::R][4]
                          [SkinnyT<T, F, KC, L128>::CW / 2];

template <typename T, int F, int KC, bool L128, bool TAIL>
__device__ __forceinline__ void
skinnyLoadT(const T *__restrict__ src, int ld, int nc, int K, int k0, int c0,
            bool vec, SkinnyReg<T, F, KC, L128> &r) {
  /* Main-loop chunks (TAIL=false) are whole, so every k-row is in range and
   * the loads are unconditional: a column run past nc is clamped back inside
   * the matrix (those columns feed only output rows/columns that are never
   * stored). The tail chunk zero-fills k >= K instead. */
  using P = SkinnyT<T, F, KC, L128>;
  static_assert(P::ok, "staged operand: unsupported F x KC");
  constexpr int CW = P::CW;
  const int lid = threadIdx.x % 32;
#pragma unroll
  for (int pass = 0; pass < P::R; ++pass) {
    const int slot = pass * 32 + lid;
    const int c = c0 + (slot % P::CG) * CW;
    const int kb = k0 + (slot / P::CG) * 4;
    const int cv = (c + CW <= nc) ? c : nc - CW;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = kb + q;
      const bool kin = !TAIL || k < K;
      const T *row = src + static_cast<size_t>(kin ? k : 0) * ld;
      if (vec) {
        if constexpr (CW == 4) {
          *reinterpret_cast<u32x2 *>(r[pass][q]) =
              *reinterpret_cast<const u32x2 *>(row + cv);
        } else {
#pragma unroll
          for (int v = 0; v < CW / 8; ++v)
            *reinterpret_cast<u32x4 *>(&r[pass][q][v * 4]) =
                reinterpret_cast<const u32x4 *>(row + cv)[v];
        }
      } else {
#pragma unroll
        for (int e = 0; e < CW; ++e)
          reinterpret_cast<T *>(r[pass][q])[e] =
              row[c + e < nc ? c + e : nc - 1];
      }
      if constexpr (TAIL) {
        if (!kin) {
#pragma unroll
          for (int v = 0; v < CW / 2; ++v)
            r[pass][q][v] = 0;
        }
      }
    }
  }
}

template <typename T, int F, int KC, bool L128>
__device__ __forceinline__ void
skinnyStoreT(T *__restrict__ tile, const SkinnyReg<T, F, KC, L128> &r) {
  using P = SkinnyT<T, F, KC, L128>;
  constexpr int CW = P::CW;
  const int lid = threadIdx.x % 32;
#pragma unroll
  for (int pass = 0; pass < P::R; ++pass) {
    const int slot = pass * 32 + lid;
    const int cb = (slot % P::CG) * CW;
    const int kb = (slot / P::CG) * 4;
#pragma unroll
    for (int j = 0; j < CW; ++j) {
      const uint32_t w0 = r[pass][0][j / 2], w1 = r[pass][1][j / 2];
      const uint32_t w2 = r[pass][2][j / 2], w3 = r[pass][3][j / 2];
      u32x2 v;
      if (j & 1) {
        v[0] = (w0 >> 16) | (w1 & 0xffff0000u);
        v[1] = (w2 >> 16) | (w3 & 0xffff0000u);
      } else {
        v[0] = (w0 & 0xffffu) | (w1 << 16);
        v[1] = (w2 & 0xffffu) | (w3 << 16);
      }
      *reinterpret_cast<u32x2 *>(&tile[(cb + j) * P::STR + kb]) = v;
    }
  }
}

template <typename T, int MT, int NTL, int KW, int KU, bool TA, bool TB,
          bool SB>
__global__ void __launch_bounds__(KW * 32)
    gemm_wmma_skinny_kernel(const T *__restrict__ A, const T *__restrict__ B,
                            const T *__restrict__ C, T *__restrict__ Y, int M,
                            int N, int K, float alpha, float beta, int cDim0,
                            int cDim1, int mtiles, int stag) {
  constexpr int FE = HIPDNN_WMMA_FRAG_ELEMS;
  constexpr int FV = FE / 8; /* u32x4 per fragment */
  constexpr int KC = 16 * KU;
  using Frag = typename std::conditional<std::is_same<T, _Float16>::value,
                                         half16, bf16xN>::type;
  /* SB: NT -> B through the wave's LDS tile as a plain row copy; NN -> the
   * 128-byte-line form of B's transposed staging (SkinnyT L128). */
  constexpr bool LB128 = !TB && SB;
  using PA = SkinnyT<T, MT, KC>;
  using PB = SkinnyT<T, NTL, KC, LB128>;
  constexpr bool BCP = TB && SB;
  constexpr int SEG = KC / 8;
  constexpr int RPI = 32 / SEG;
  constexpr int NCP = BCP ? 16 * NTL / RPI : 1;
  static_assert(!BCP || (32 % SEG == 0 && (16 * NTL) % RPI == 0), "SB map");
  constexpr int LA = TA ? 16 * MT * PA::STR : 0;
  constexpr int LB = (TB && !SB) ? 0 : 16 * NTL * PB::STR;
  constexpr int SLOTS = MT * NTL * 8;
  /* The staging tiles (K loop) and the split-K reduction (after it) share
   * one LDS block. */
  constexpr size_t SM_T = static_cast<size_t>(KW) * (LA + LB) * sizeof(T);
  constexpr size_t SM_R = KW > 1 ? static_cast<size_t>(KW) * SLOTS * 32 * 4 : 0;
  __shared__ __align__(16) char sm_raw[(SM_T > SM_R ? SM_T : SM_R) + 16];

  const int tid = threadIdx.x;
  /* Wave index as a scalar: the K loop below is driven by it, and a value the
   * compiler treats as per-lane turns that loop into exec-masked branches
   * with a full vmcnt(0) wait before every chunk's WMMAs. */
  const int w = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int mt = static_cast<int>(blockIdx.x) % mtiles;
  const int nt = static_cast<int>(blockIdx.x) / mtiles;
  const int row0 = mt * 16 * MT;
  const int col0 = nt * 16 * NTL;
  const int koff = hipdnn_wmma_k_off(sub);
  T *tA = reinterpret_cast<T *>(sm_raw) + w * (LA + LB);
  T *tB = tA + LA;
  /* Packed vector loads of a [k][c] operand need every row start aligned to
   * the per-lane column run. */
  const bool vecA = (M % PA::CW) == 0 && M >= PA::CW;
  const bool vecB = (N % PB::CW) == 0 && N >= PB::CW;

  const T *ap[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int r = row0 + i * 16 + lane;
    ap[i] = A + static_cast<size_t>(r < M ? r : 0) * K + koff;
  }
  const T *bp[NTL];
#pragma unroll
  for (int j = 0; j < NTL; ++j) {
    const int c = col0 + j * 16 + lane;
    bp[j] = B + static_cast<size_t>(c < N ? c : 0) * K + koff;
  }
  const T *cp[NCP];
  int cp_lds[NCP];
  const int cp_seg = (lid % SEG) * 8;
#pragma unroll
  for (int q = 0; q < NCP; ++q) {
    const int r = q * RPI + lid / SEG;
    const int c = col0 + r;
    cp[q] = B + static_cast<size_t>(c < N ? c : 0) * K + cp_seg;
    cp_lds[q] = r * PB::STR + cp_seg;
  }

  const int nfull = K / KC;
  int shift = 0;
  if (stag != 0 && nfull > 1)
    shift = ((nt & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % nfull;

  u32x4 fa[2][TA ? 1 : MT][KU][FV];
  u32x4 fb[2][TB ? NTL : 1][KU][FV];
  SkinnyReg<T, TA ? MT : 1, TA ? KC : 32, false> ra[2];
  SkinnyReg<T, TB ? 1 : NTL, TB ? 32 : KC, LB128> rb[2];
  u32x4 rc[2][NCP];
  float8 acc[MT][NTL];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  /* Loads chunk k0 .. k0+KC into slot S. TAIL zero-fills k >= K. */
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
    if constexpr (TA)
      skinnyLoadT<T, MT, KC, false, TL>(A, M, M, K, k0, row0, vecA, ra[S]);
    if constexpr (!TB)
      skinnyLoadT<T, NTL, KC, LB128, TL>(B, N, N, K, k0, col0, vecB,
                                         rb[S]);
    if constexpr (BCP) {
      const bool kin = !TL || (k0 + cp_seg + 8 <= K);
#pragma unroll
      for (int q = 0; q < NCP; ++q) {
        rc[S][q] = *reinterpret_cast<const u32x4 *>(cp[q] + (kin ? k0 : 0));
        if (TL && !kin)
          rc[S][q] = u32x4{0, 0, 0, 0};
      }
    }
#pragma unroll
    for (int kk = 0; kk < KU; ++kk) {
#pragma unroll
      for (int v = 0; v < FV; ++v) {
        const int k = k0 + kk * 16 + v * 8;
        const bool kin = !TL || (k + koff + 8 <= K);
        const int ke = kin ? k : 0;
        if constexpr (!TA) {
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            fa[S][i][kk][v] = *reinterpret_cast<const u32x4 *>(ap[i] + ke);
            if (TL && !kin)
              fa[S][i][kk][v] = u32x4{0, 0, 0, 0};
          }
        }
        if constexpr (TB && !SB) {
#pragma unroll
          for (int j = 0; j < NTL; ++j) {
            fb[S][j][kk][v] = *reinterpret_cast<const u32x4 *>(bp[j] + ke);
            if (TL && !kin)
              fb[S][j][kk][v] = u32x4{0, 0, 0, 0};
          }
        }
      }
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TA)
      skinnyStoreT<T, MT, KC, false>(tA, ra[S]);
    if constexpr (!TB)
      skinnyStoreT<T, NTL, KC, LB128>(tB, rb[S]);
    if constexpr (BCP) {
#pragma unroll
      for (int q = 0; q < NCP; ++q)
        *reinterpret_cast<u32x4 *>(&tB[cp_lds[q]]) = rc[S][q];
    }
#pragma unroll
    for (int kk = 0; kk < KU; ++kk) {
#pragma unroll
      for (int j = 0; j < NTL; ++j) {
        Frag b;
        if constexpr (TB && !SB) {
          __builtin_memcpy(&b, fb[S][j][kk], sizeof(Frag));
        } else {
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &tB[(j * 16 + lane) * PB::STR + kk * 16 + koff]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b);
#pragma unroll
          for (int v = 0; v < FV; ++v)
            dst[v] = src[v];
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          Frag a;
          if constexpr (TA) {
            const u32x4 *src = reinterpret_cast<const u32x4 *>(
                &tA[(i * 16 + lane) * PA::STR + kk * 16 + koff]);
            u32x4 *dst = reinterpret_cast<u32x4 *>(&a);
#pragma unroll
            for (int v = 0; v < FV; ++v)
              dst[v] = src[v];
          } else {
            __builtin_memcpy(&a, fa[S][i][kk], sizeof(Frag));
          }
          if constexpr (std::is_same<T, _Float16>::value)
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_F16(a, b, acc[i][j]);
          else
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_BF16(a, b, acc[i][j]);
        }
      }
    }
  };

  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, 1>;
  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  auto kofs = [&](int c) __attribute__((always_inline)) {
    int ch = c + shift;
    if (ch >= nfull)
      ch -= nfull;
    return ch * KC;
  };
  /* Whole chunks w, w+KW, ... (n of them) with one register double buffer.
   * The steady-state loop issues its loads unconditionally and the last one
   * or two chunks are peeled: a load that may or may not be issued before a
   * compute makes the compiler's vmcnt bookkeeping wait for it either way. */
  const int n = nfull > w ? (nfull - w + KW - 1) / KW : 0;
  if (n > 0) {
    load(kofs(w), S0{}, Full{});
    int i = 0;
    for (; i + 2 < n; i += 2) {
      load(kofs(w + (i + 1) * KW), S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      load(kofs(w + (i + 2) * KW), S0{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S1{});
    }
    if (n - i == 2) {
      load(kofs(w + (i + 1) * KW), S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      compute(S1{});
    } else {
      compute(S0{});
    }
  }
  /* Partial last chunk (K % KC != 0): one wave, zero-filled past K. */
  if (nfull * KC < K && w == nfull % KW) {
    load(nfull * KC, S0{}, Tail{});
    compute(S0{});
  }

  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f;
  auto store = [&](int i, int j, int e, int l, float v)
      __attribute__((always_inline)) {
    const int r = row0 + i * 16 + hipdnn_wmma_acc_row(l / 16, e);
    const int cc = col0 + j * 16 + l % 16;
    if (r < M && cc < N) {
      if (fast)
        Y[static_cast<size_t>(r) * N + cc] = GemmElem<T>::fromf(v);
      else
        gemmStoreY(Y, C, r, cc, N, cDim0, cDim1, v, alpha, beta);
    }
  };
  if constexpr (KW == 1) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NTL; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          store(i, j, e, lid, acc[i][j][e]);
  } else {
    float(*red)[SLOTS][32] = reinterpret_cast<float(*)[SLOTS][32]>(sm_raw);
    if constexpr (SM_T > 0)
      __syncthreads(); /* other waves may still read their staging tiles */
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NTL; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          red[w][(i * NTL + j) * 8 + e][lid] = acc[i][j][e];
    __syncthreads();
#pragma unroll
    for (int idx = tid; idx < SLOTS * 32; idx += KW * 32) {
      const int slot = idx / 32;
      const int l = idx % 32;
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < KW; ++q)
        v += red[q][slot][l];
      store(slot / (NTL * 8), (slot / 8) % NTL, slot % 8, l, v);
    }
  }
}

/* =========================================================================
 * NN (B = [K, N]) mid-M WMMA, fp16/bf16 (M about 32..128).
 *
 * One workgroup owns ALL M rows -- NWM waves of MW 16-row tiles each -- and
 * BN = 16 * NTL columns, so B is read exactly once and transposed into LDS
 * once per workgroup, where all NWM waves share it. LDS is used as the
 * transpose, not as a cache: A is k-contiguous, so each wave loads its own
 * rows' fragments straight from global memory (L2) into registers, with no
 * LDS round trip. (The skinny kernel transposes B per wave through a private
 * tile and re-reads it per M-tile workgroup; the LDS tiles stage A as well
 * and run a bounds-checked K loop on a ragged M.)
 *
 * Per KC-row chunk the block's threads load B as 16-byte row pieces (BN / 8
 * lanes per k-row: whole BN * 2-byte runs) and write each 4-k-row x
 * 8-column piece transposed with packed 8-byte stores into one of two LDS
 * buffers; one barrier per chunk. PF = 1: chunk t+1's B and A loads are
 * issued before chunk t's WMMAs; PF = 2: B runs two chunks ahead in two
 * register slots (A one chunk ahead). The loop is unrolled by two so the LDS
 * buffer and register slots are compile-time indices, and the loads past the
 * last chunk are clamped to it (issued, never used) so every step issues the
 * same loads. Rows past M re-read row M-1 and are not stored; waves whose
 * rows are all past M stage B but skip the WMMAs. Requires K % KC == 0,
 * N % 8 == 0 and 16-byte aligned A and B (launchNnMid checks).
 * ========================================================================= */
template <typename T, int MW, int NTL, int NWM, int KC, int PF>
__global__ void __launch_bounds__(NWM * 32)
    gemm_wmma_nnmid_kernel(const T *__restrict__ A, const T *__restrict__ B,
                           const T *__restrict__ C, T *__restrict__ Y, int M,
                           int N, int K, float alpha, float beta, int cDim0,
                           int cDim1) {
  constexpr int FE = HIPDNN_WMMA_FRAG_ELEMS;
  constexpr int FV = FE / 8; /* u32x4 per fragment */
  constexpr int THR = NWM * 32;
  constexpr int BN = 16 * NTL;
  constexpr int KS = KC / 16;
  constexpr int STR = KC + 8; /* LDS row stride (halfs), 16 B aligned */
  constexpr int CGN = BN / 8;         /* 8-column groups per k-row */
  constexpr int NSL = (KC / 4) * CGN; /* 4-row x 8-column B pieces per chunk */
  constexpr int SPT = (NSL + THR - 1) / THR;
  static_assert(PF == 1 || PF == 2, "nnmid PF");
  using Frag = typename std::conditional<std::is_same<T, _Float16>::value,
                                         half16, bf16xN>::type;
  __shared__ __align__(16) T tB[2][BN * STR];

  const int tid = threadIdx.x;
  /* scalar wave index: the per-wave row range and the skip test stay
   * uniform */
  const int wv = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int koff = hipdnn_wmma_k_off(sub);
  const int col0 = blockIdx.x * BN;
  const int wrow0 = wv * MW * 16;
  const bool wact = wrow0 < M;

  /* this thread's B pieces: offset in chunk 0 and the LDS destination */
  size_t bq[SPT];
  int sq[SPT];
#pragma unroll
  for (int p = 0; p < SPT; ++p) {
    const int q = p * THR + tid;
    const int cg = q % CGN, kq = (q / CGN) % (KC / 4);
    const int c = col0 + cg * 8;
    const int cv = (c + 8 <= N) ? c : N - 8;
    bq[p] = static_cast<size_t>(kq * 4) * N + cv;
    sq[p] = cg * 8 * STR + kq * 4;
  }
  /* the last pass is partial when THR does not divide NSL */
  auto qon = [&](int p) __attribute__((always_inline)) {
    return (NSL % THR) == 0 || p * THR + tid < NSL;
  };
  const T *ap[MW];
#pragma unroll
  for (int i = 0; i < MW; ++i) {
    const int r = wrow0 + i * 16 + lane;
    ap[i] = A + static_cast<size_t>(r < M ? r : M - 1) * K + koff;
  }
  const size_t ldn = static_cast<size_t>(N);
  const int nch = K / KC;

  u32x4 rb[PF][SPT][4];
  u32x4 fa[2][MW][KS][FV];
  float8 acc[MW][NTL];
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  auto loadB = [&](int c, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    const T *bc = B + static_cast<size_t>(c < nch ? c : nch - 1) * KC * ldn;
#pragma unroll
    for (int p = 0; p < SPT; ++p)
      if (qon(p)) {
#pragma unroll
        for (int r = 0; r < 4; ++r)
          rb[S][p][r] =
              *reinterpret_cast<const u32x4 *>(bc + bq[p] + r * ldn);
      }
  };
  auto loadA = [&](int c, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    const int k0 = (c < nch ? c : nch - 1) * KC;
#pragma unroll
    for (int i = 0; i < MW; ++i)
#pragma unroll
      for (int kk = 0; kk < KS; ++kk)
#pragma unroll
        for (int v = 0; v < FV; ++v)
          fa[S][i][kk][v] = *reinterpret_cast<const u32x4 *>(
              ap[i] + k0 + kk * 16 + v * 8);
  };
  /* 4 k-rows x 8 columns -> 8 columns of 4 k (the packing of skinnyStoreT) */
  auto storeB = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int BF = decltype(buf)::value;
#pragma unroll
    for (int p = 0; p < SPT; ++p) {
      if (qon(p)) {
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
          const uint32_t w0 = rb[S][p][0][jj / 2], w1 = rb[S][p][1][jj / 2];
          const uint32_t w2 = rb[S][p][2][jj / 2], w3 = rb[S][p][3][jj / 2];
          u32x2 v;
          if (jj & 1) {
            v[0] = (w0 >> 16) | (w1 & 0xffff0000u);
            v[1] = (w2 >> 16) | (w3 & 0xffff0000u);
          } else {
            v[0] = (w0 & 0xffffu) | (w1 << 16);
            v[1] = (w2 & 0xffffu) | (w3 << 16);
          }
          *reinterpret_cast<u32x2 *>(&tB[BF][sq[p] + jj * STR]) = v;
        }
      }
    }
  };
  auto compute = [&](auto aslot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(aslot)::value;
    constexpr int BF = decltype(buf)::value;
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
#pragma unroll
      for (int j = 0; j < NTL; ++j) {
        Frag b;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &tB[BF][(j * 16 + lane) * STR + kk * 16 + koff]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b);
#pragma unroll
        for (int v = 0; v < FV; ++v)
          dst[v] = src[v];
#pragma unroll
        for (int i = 0; i < MW; ++i) {
          Frag a;
          __builtin_memcpy(&a, fa[S][i][kk], sizeof(Frag));
          if constexpr (std::is_same<T, _Float16>::value)
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_F16(a, b, acc[i][j]);
          else
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_BF16(a, b, acc[i][j]);
        }
      }
    }
  };
  using C0 = std::integral_constant<int, 0>;
  using C1 = std::integral_constant<int, 1>;
  /* step t: issue B(t + PF) and A(t + 1); WMMAs on chunk t; stage B(t + 1)
   * into the other LDS buffer; barrier */
  auto step = [&](int t, auto par) __attribute__((always_inline)) {
    constexpr int P = decltype(par)::value;
    using Pc = std::integral_constant<int, P>;
    using Pn = std::integral_constant<int, 1 - P>;
    if constexpr (PF == 2)
      loadB(t + 2, Pc{});
    else
      loadB(t + 1, C0{});
    loadA(t + 1, Pn{});
    GEMM_SCHED_FENCE();
    if (wact)
      compute(Pc{}, Pc{});
    GEMM_SCHED_FENCE();
    if constexpr (PF == 2)
      storeB(Pn{}, Pn{});
    else
      storeB(C0{}, Pn{});
    HIPDNN_LDS_BARRIER();
  };
  /* prologue: B(0) staged, B(1) in flight (PF 2), A(0) in slot 0 */
  loadB(0, C0{});
  loadA(0, C0{});
  storeB(C0{}, C0{});
  if constexpr (PF == 2)
    loadB(1, C1{});
  HIPDNN_LDS_BARRIER();
  int t = 0;
  for (; t + 2 <= nch; t += 2) {
    step(t, C0{});
    step(t + 1, C1{});
  }
  if (t < nch)
    step(t, C0{});

  if (!wact)
    return;
  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f;
#pragma unroll
  for (int i = 0; i < MW; ++i)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = wrow0 + i * 16 + hipdnn_wmma_acc_row(sub, e);
        const int cc = col0 + j * 16 + lane;
        if (r < M && cc < N) {
          if (fast)
            Y[static_cast<size_t>(r) * N + cc] =
                GemmElem<T>::fromf(acc[i][j][e]);
          else
            gemmStoreY(Y, C, r, cc, N, cDim0, cDim1, acc[i][j][e], alpha,
                       beta);
        }
      }
}

struct GemvCfg {
  int threads;
  int tile_n;
};

static constexpr GemvCfg kGemvNt[] = {
    {32, 1},  {32, 4},  {32, 8},  {64, 4},  {64, 8}, {128, 4},
    {128, 8}, {256, 4}, {256, 8}, {512, 4}, {512, 8},
};
static constexpr int kNumGemvNt =
    static_cast<int>(sizeof(kGemvNt) / sizeof(kGemvNt[0]));

template <typename T>
static void launchGemvNt(int id, hipStream_t s, const T *A, const T *B,
                           const T *C, T *Y, int M, int N, int K, float alpha,
                           float beta, int cDim0, int cDim1) {
  const GemvCfg c = kGemvNt[id];
  dim3 grid((N + c.tile_n - 1) / c.tile_n, M);
  dim3 block(c.threads);
#define GEMM_GEMV_NT(ID, BS, TN)                                               \
  case ID:                                                                     \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_nt_kernel<T, BS, TN>), grid,  \
                       block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,  \
                       cDim1);                                                 \
    break
  switch (id) {
    GEMM_GEMV_NT(0, 32, 1);
    GEMM_GEMV_NT(1, 32, 4);
    GEMM_GEMV_NT(2, 32, 8);
    GEMM_GEMV_NT(3, 64, 4);
    GEMM_GEMV_NT(4, 64, 8);
    GEMM_GEMV_NT(5, 128, 4);
    GEMM_GEMV_NT(6, 128, 8);
    GEMM_GEMV_NT(7, 256, 4);
    GEMM_GEMV_NT(8, 256, 8);
    GEMM_GEMV_NT(9, 512, 4);
    GEMM_GEMV_NT(10, 512, 8);
  default:
    break;
  }
#undef GEMM_GEMV_NT
}

/* NN decode config table (Stage A7): one axis, the block/thread count -- the
 * NN kernel is one output column per thread, so "tile_n" has no meaning here
 * (the GemvCfg.tile_n field is unused/0 for every NN entry, matching the fbs
 * comment). Previously launchGemvNn hardcoded BLOCK=256 with no config table
 * and no tuning at all; fp32 NN decode shapes therefore had zero LUT coverage
 * (plan.md §0.1). */
static constexpr GemvCfg kGemvNn[] = {
    {32, 0}, {64, 0}, {128, 0}, {256, 0}, {512, 0},
};
static constexpr int kNumGemvNn =
    static_cast<int>(sizeof(kGemvNn) / sizeof(kGemvNn[0]));

template <typename T>
static void launchGemvNn(int id, hipStream_t s, const T *A, const T *B,
                          const T *C, T *Y, int M, int N, int K, float alpha,
                          float beta, int cDim0, int cDim1) {
  const int threads = kGemvNn[id].threads;
  dim3 grid((N + threads - 1) / threads, M);
#define GEMM_GEMV_NN(ID, BS)                                                 \
  case ID:                                                                   \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_nn_kernel<T, BS>), grid,    \
                       dim3(BS), 0, s, A, B, C, Y, M, N, K, alpha, beta,     \
                       cDim0, cDim1);                                        \
    break
  switch (id) {
    GEMM_GEMV_NN(0, 32);
    GEMM_GEMV_NN(1, 64);
    GEMM_GEMV_NN(2, 128);
    GEMM_GEMV_NN(3, 256);
    GEMM_GEMV_NN(4, 512);
  default:
    break;
  }
#undef GEMM_GEMV_NN
}

static int gemmEnvInt(const char *name, int defv) {
#ifdef _WIN32
  char *dup = nullptr;
  size_t len = 0;
  if (_dupenv_s(&dup, &len, name) != 0 || !dup)
    return defv;
  int v = atoi(dup);
  free(dup);
  return v;
#else
  const char *v = getenv(name);
  return (v && v[0]) ? atoi(v) : defv;
#endif
}

static bool gemmAutotuneOn() {
  static const bool on = [] {
#ifdef _WIN32
    char *dup = nullptr;
    size_t len = 0;
    if (_dupenv_s(&dup, &len, "HIPDNN_EP_GEMM_AUTOTUNE") != 0 || !dup)
      return true;
    bool yes = dup[0] != '0';
    free(dup);
    return yes;
#else
    const char *v = getenv("HIPDNN_EP_GEMM_AUTOTUNE");
    return !v || v[0] != '0';
#endif
  }();
  return on;
}

/* Cross-platform env string read (mirrors matmulNbitsLutLookupEnabled). */
static std::string gemmEnvStr(const char *name) {
  std::string raw;
#ifdef _WIN32
  char *v = nullptr;
  size_t len = 0;
  if (_dupenv_s(&v, &len, name) == 0 && v) {
    raw = v;
    free(v);
  }
#else
  if (const char *v = getenv(name))
    raw = v;
#endif
  return raw;
}

/* Config source, mirroring matmul_nbits' HIPDNN_MATMUL_AUTOTUNE_MODE:
 *   lookup (default) - resolve the config from the offline LUT, falling back to
 *                      the existing static resolver default on a miss.
 *   online           - bypass the LUT and always run the sweep (A/B baseline).
 * Both paths are compiled in; this only selects which one a process takes.
 * Read once and latched: the tuner sits on the dispatch path and a per-call
 * getenv would serialize through the CRT environment lock. */
static bool gemmLutLookupEnabled() {
  static const bool lookup = [] {
    std::string s;
    for (char c : gemmEnvStr("HIPDNN_GEMM_AUTOTUNE_MODE")) {
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        continue;
      s.push_back(c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c);
    }
    if (s == "online")
      return false;
    if (!s.empty() && s != "lookup")
      fprintf(stderr,
              "[gemm] unrecognised HIPDNN_GEMM_AUTOTUNE_MODE=\"%s\" (expected "
              "\"lookup\" or \"online\"); using lookup\n",
              s.c_str());
    return true; /* default, and explicit "lookup" */
  }();
  return lookup;
}

/* ---- Offline LUT bridge --------------------------------------------------
 *
 * The autotune winner for a shape is durable knowledge that belongs in an
 * offline table, exactly like matmul_nbits' hipdnn_ep::matmul_nbits_autotune.
 * `gemm_autotune::resolve` (autotune/gemm/gemm_autotune.cpp, embedded per-arch
 * into custom_kernels_<arch>) does the nearest-neighbour lookup; the table
 * stores tile *geometry* (not a config index) and the validator maps it back
 * onto the live kWmma/kGemvNt table, so a stale LUT is never worse than no LUT.
 * With no table embedded (the size-0 stub) resolve() reports a miss and the
 * caller sweeps / uses the heuristic exactly as before.
 *
 * gemm_lut is an alias so the existing call sites (map -> LUT -> sweep) and the
 * validator/geometry helpers are unchanged. */
namespace gemm_lut = hipdnn_ep::gemm_autotune;

static std::mutex g_tune_mu;
static std::unordered_map<std::string, int> g_gemv_nt_cache;

/* gemm allocates no device memory of its own, so the online sweep times every
 * candidate on the caller's Y. That is only harmless when Y is not also the C
 * the epilogue reads. */
static bool gemmSweepSafe(const void *C, const void *Y, float beta) {
  return C != Y || beta == 0.0f;
}

/* Clock settle loop (hip-kernel-perf-measurement skill): drive a representative
 * config until per-iter latency plateaus (>=60 ms total, then 3 consecutive
 * reps within 2%), BEFORE the candidate loop. This is the piece the old
 * tuneGemvNt was missing entirely (WARMUP=1, no settle, averaged) -- the exact
 * shape of the gfx1152 decode-tuner incident the skill documents. `launch1`
 * is a zero-arg thunk that issues one representative-config launch. */
template <typename Launch1>
static void gemmSettleClock(hipStream_t s, hipEvent_t e0, hipEvent_t e1,
                            Launch1 &&launch1, int init_warm) {
  float prev = 1e30f, total = 0.0f;
  int stable = 0;
  for (int i = 0; i < init_warm && stable < 3 && total < 200.0f; ++i) {
    hipEventRecord(e0, s);
    launch1();
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    float cur = 0.f;
    hipEventElapsedTime(&cur, e0, e1);
    if (total >= 60.0f && cur <= prev * 1.02f)
      ++stable;
    else
      stable = 0;
    prev = cur;
    total += cur;
  }
}

/* Time one candidate: WARMUP launches (untimed, re-equilibrate), then ITERS
 * timed launches, keeping the MIN (peak, least-throttled) with the mandatory
 * negative-value guard -- hipEventElapsedTime occasionally returns a small
 * negative on sub-0.1 ms kernels, and an ungated min always "wins" with it,
 * picking a garbage config (hip-kernel-perf-measurement skill, rule 3). */
template <typename LaunchCid>
static float gemmTimeConfigPeak(hipStream_t s, hipEvent_t e0, hipEvent_t e1,
                                LaunchCid &&launch_cid, int warmup, int iters) {
  for (int w = 0; w < warmup; ++w)
    launch_cid();
  float best = 1e30f;
  for (int i = 0; i < iters; ++i) {
    hipEventRecord(e0, s);
    launch_cid();
    hipEventRecord(e1, s);
    if (hipEventSynchronize(e1) != hipSuccess)
      continue;
    float ms = 0.f;
    hipEventElapsedTime(&ms, e0, e1);
    if (ms > 0.0f && ms < best)
      best = ms;
  }
  return best;
}

/* Geometry <-> config-index bridge for the GEMV NT LUT (see gemm_lut). */
static int gemvNtConfigFromGeom(const gemm_lut::GemvAnswer &g) {
  for (int cid = 0; cid < kNumGemvNt; ++cid)
    if (kGemvNt[cid].threads == g.threads && kGemvNt[cid].tile_n == g.tile_n)
      return cid;
  return -1;
}
static bool gemvNtLutValidator(void *, const gemm_lut::GemvAnswer &g) {
  return gemvNtConfigFromGeom(g) >= 0;
}
template <typename T>
static int gemmLutGemvNtConfig(int M, int N, int K, int transB) {
  if (!gemmLutLookupEnabled())
    return -1; /* online mode: skip the LUT and sweep */
  gemm_lut::Request req;
  req.kind = gemm_lut::Kind::GemvNt;
  req.act_dtype = req.wts_dtype = req.out_dtype = gemmDType<T>();
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  const gemm_lut::Result r =
      gemm_lut::resolve(req, nullptr, gemvNtLutValidator, nullptr, nullptr);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return gemvNtConfigFromGeom(r.gemv);
}

template <typename T>
static int tuneGemvNt(hipStream_t s, const T *A, const T *B, const T *C, T *Y,
                       int M, int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  std::string key = std::string("nt_") +
                      hipdnn_ep::common::toString(gemmDType<T>()) + "_" +
                      std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K);
  {
    std::lock_guard<std::mutex> lk(g_tune_mu);
    auto it = g_gemv_nt_cache.find(key);
    if (it != g_gemv_nt_cache.end())
      return it->second;
  }

  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG", -1);
  int best = 6; /* 128x8 */
  if (forced >= 0 && forced < kNumGemvNt) {
    best = forced;
  } else if (int lut = gemmLutGemvNtConfig<T>(M, N, K, /*transB=*/1);
             lut >= 0) {
    best = lut;
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nt LUT hit M=%d N=%d K=%d -> cfg[%d] th=%d tn=%d\n", M, N,
        K, best, kGemvNt[best].threads, kGemvNt[best].tile_n);
  } else if (gemmLutLookupEnabled()) {
    /* Default (lookup) mode, LUT miss: keep the static default (best=6). The
     * on-device sweep runs only in `online` mode, for LUT generation. */
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nt default M=%d N=%d K=%d -> cfg[%d] th=%d tn=%d\n", M, N,
        K, best, kGemvNt[best].threads, kGemvNt[best].tile_n);
  } else if (gemmAutotuneOn() && gemmSweepSafe(C, Y, beta)) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *out = Y;

    /* Sub-0.1 ms decode GEMV sits near the hipEventElapsedTime floor, so this
     * phase gets more iters than tuneWmma's compute-bound tiles (skill rule 4:
     * raise iters only in this regime). Env-tunable per plan.md §4.1. */
    const int kInitWarm =
        gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_INITWARM", 60);
    const int kWarm = gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_WARM", 3);
    const int kIters = gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_ITERS", 7);

    gemmSettleClock(
        s, e0, e1,
        [&] {
          launchGemvNt<T>(best, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                          cDim1);
        },
        kInitWarm);

    float best_ms = 1e30f;
    for (int cid = 0; cid < kNumGemvNt; ++cid) {
      const float ms = gemmTimeConfigPeak(
          s, e0, e1,
          [&] {
            launchGemvNt<T>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                            cDim1);
          },
          kWarm, kIters);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] gemv-nt cfg[%d] th=%d tn=%d : %.4f ms (peak)\n", cid,
          kGemvNt[cid].threads, kGemvNt[cid].tile_n, ms);
      if (ms < best_ms) {
        best_ms = ms;
        best = cid;
      }
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nt autotune M=%d N=%d K=%d -> cfg[%d] th=%d tn=%d\n", M, N,
        K, best, kGemvNt[best].threads, kGemvNt[best].tile_n);
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_gemv_nt_cache[key] = best;
  return best;
}

/* Geometry <-> config-index bridge for the GEMV NN LUT (Stage A7; mirrors the
 * NT bridge above). */
static int gemvNnConfigFromGeom(const gemm_lut::GemvAnswer &g) {
  for (int cid = 0; cid < kNumGemvNn; ++cid)
    if (kGemvNn[cid].threads == g.threads)
      return cid;
  return -1;
}
static bool gemvNnLutValidator(void *, const gemm_lut::GemvAnswer &g) {
  return gemvNnConfigFromGeom(g) >= 0;
}
template <typename T>
static int gemmLutGemvNnConfig(int M, int N, int K) {
  if (!gemmLutLookupEnabled())
    return -1;
  gemm_lut::Request req;
  req.kind = gemm_lut::Kind::GemvNn;
  req.act_dtype = req.wts_dtype = req.out_dtype = gemmDType<T>();
  req.trans_b = 0;
  req.m = M;
  req.n = N;
  req.k = K;
  const gemm_lut::Result r =
      gemm_lut::resolve(req, nullptr, gemvNnLutValidator, nullptr, nullptr);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return gemvNnConfigFromGeom(r.gemv);
}

static std::unordered_map<std::string, int> g_gemv_nn_cache;

/* Same settle-once & peak(min) protocol as tuneGemvNt (Stage A7: previously
 * launchGemvNn had a single hardcoded BLOCK=256, no config table and no
 * tuning at all, so fp32 NN decode had zero LUT coverage -- plan.md §0.1). */
template <typename T>
static int tuneGemvNn(hipStream_t s, const T *A, const T *B, const T *C, T *Y,
                       int M, int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  std::string key = std::string("nn_") +
                      hipdnn_ep::common::toString(gemmDType<T>()) + "_" +
                      std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K);
  {
    std::lock_guard<std::mutex> lk(g_tune_mu);
    auto it = g_gemv_nn_cache.find(key);
    if (it != g_gemv_nn_cache.end())
      return it->second;
  }

  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG", -1);
  int best = 3; /* 256 threads, the old hardcoded default */
  if (forced >= 0 && forced < kNumGemvNn) {
    best = forced;
  } else if (int lut = gemmLutGemvNnConfig<T>(M, N, K); lut >= 0) {
    best = lut;
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nn LUT hit M=%d N=%d K=%d -> cfg[%d] th=%d\n", M, N, K,
        best, kGemvNn[best].threads);
  } else if (gemmLutLookupEnabled()) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nn default M=%d N=%d K=%d -> cfg[%d] th=%d\n", M, N, K,
        best, kGemvNn[best].threads);
  } else if (gemmAutotuneOn() && gemmSweepSafe(C, Y, beta)) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *out = Y;

    const int kInitWarm =
        gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_INITWARM", 60);
    const int kWarm = gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_WARM", 3);
    const int kIters = gemmEnvInt("HIPDNN_GEMM_GEMV_TUNE_ITERS", 7);

    gemmSettleClock(
        s, e0, e1,
        [&] {
          launchGemvNn<T>(best, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                          cDim1);
        },
        kInitWarm);

    float best_ms = 1e30f;
    for (int cid = 0; cid < kNumGemvNn; ++cid) {
      const float ms = gemmTimeConfigPeak(
          s, e0, e1,
          [&] {
            launchGemvNn<T>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                            cDim1);
          },
          kWarm, kIters);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] gemv-nn cfg[%d] th=%d : %.4f ms (peak)\n",
                               cid, kGemvNn[cid].threads, ms);
      if (ms < best_ms) {
        best_ms = ms;
        best = cid;
      }
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nn autotune M=%d N=%d K=%d -> cfg[%d] th=%d\n", M, N, K,
        best, kGemvNn[best].threads);
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_gemv_nn_cache[key] = best;
  return best;
}

/* gemm_gemv_kn_ring_kernel configs {MR, LPR, KW, U, NS} (X-macro: table and
 * dispatch cannot drift apart). 0-2 fp16 / bf16 (one 128-byte line per k-row
 * and load), 3-5 fp32 (3: a 512-byte k-row run per load; 4-5: 256 bytes). */
#define GEMM_KN_RING_CFGS(X)                                                   \
  X(0, 1, 8, 8, 2, 2)                                                          \
  X(1, 2, 8, 8, 2, 2)                                                          \
  X(2, 4, 8, 8, 2, 2)                                                          \
  X(3, 1, 32, 8, 1, 2)                                                         \
  X(4, 4, 16, 8, 1, 2)                                                         \
  X(5, 8, 16, 8, 1, 2)

struct KnRingCfg {
  int mr, lpr, kw, u, ns;
};
#define GEMM_KN_RING_ENTRY(ID, MR, LPR, KW, U, NS) {MR, LPR, KW, U, NS},
static constexpr KnRingCfg kKnRing[] = {GEMM_KN_RING_CFGS(GEMM_KN_RING_ENTRY)};
#undef GEMM_KN_RING_ENTRY
static constexpr int kNumKnRing =
    static_cast<int>(sizeof(kKnRing) / sizeof(kKnRing[0]));

/* NN small-M route to gemm_gemv_kn_ring_kernel (-1 = keep the existing
 * path). Screening harness (product kernels and the ring in one process,
 * 2 rounds ABBA, slower round; bench3 in the commit), ms fp16 / bf16, old
 * route -> ring:
 *   - fp16 / bf16, M = 1, B <= 32 MB (Infinity-Cache resident) -> cfg 0:
 *     3072^2 0.0319 / 0.0308 -> 0.0138 / 0.0149, 2048^2 0.0114 / 0.0080 ->
 *     0.0067 / 0.0066. A larger B keeps gemm_gemv_kn_kernel (its 16-wave
 *     entry streams DRAM at 216-226 GB/s; the ring reached 193-204).
 *   - fp16 / bf16, M = 2..4 (old: the skinny kernel on B <= 32 MB, the
 *     [K, N] GEMV on a larger B), B <= 32 MB or N < 2K -> cfg 1 / 2:
 *     4x3072^2 0.0237 / 0.0236 -> 0.0184 / 0.0155, 4x4096x14336 0.941 /
 *     0.950 -> 0.562 / 0.611, 2x8192^2 0.780 / 0.790 -> 0.625 / 0.667;
 *     N >= 2K on a larger B (14336x4096) keeps the GEMV (ring 1.0-1.1x).
 *   - fp32, M = 1, B > 32 MB and N <= K -> cfg 3: 4096^2 0.392 -> 0.290,
 *     4096x14336 1.359 -> 0.998, 8192^2 1.478 -> 1.12; N > K keeps the GEMV
 *     (14336x4096 0.94x, 11008x4096 0.97x).
 *   - fp32, M = 2..8, B <= 32 MB (old: gemm_f32_skinny_nn_kernel) -> cfg 4
 *     / 5: 4x2048^2 0.0300 -> 0.0125, 8x2048^2 0.0303 -> 0.0138. */
static int pickKnRing(int M, int N, int K, int esize) {
  const bool big = static_cast<size_t>(N) * K * esize > (32ull << 20);
  if (esize == 4) {
    if (M == 1)
      return (big && N <= K) ? 3 : -1;
    return (M <= 8 && !big) ? (M <= 4 ? 4 : 5) : -1;
  }
  if (M == 1)
    return big ? -1 : 0;
  if (M <= 4 && (!big || N < 2 * K))
    return M <= 2 ? 1 : 2;
  return -1;
}

static bool gemmKnRingOn() {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_KN_RING", 1) != 0;
  return on;
}

/* The ring config for an NN shape, or -1 when the shape does not meet the
 * kernel's preconditions (K % GR, N % VE, 16-byte aligned A and B; N * K
 * >= 64K like the other multi-row GEMVs) or the route keeps it elsewhere.
 * HIPDNN_EP_GEMM_KN_RING_CFG=<id> forces a config for every eligible NN
 * shape with M <= its MR. */
template <typename T>
static int gemmKnRingConfig(const T *A, const T *B, int M, int N, int K) {
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG", -1);
  constexpr int esize = static_cast<int>(sizeof(T));
  constexpr int VE = 16 / esize;
  if (!gemmKnRingOn() || M < 1 || N < VE || (N % VE) != 0 ||
      static_cast<size_t>(N) * K < (size_t{1} << 16) ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) != 0)
    return -1;
  int id = pickKnRing(M, N, K, esize);
  if (forced >= 0 && forced < kNumKnRing && M <= kKnRing[forced].mr)
    id = forced;
  if (id < 0 || (esize == 4) != (id >= 3))
    return -1;
  const KnRingCfg &c = kKnRing[id];
  if ((K % ((32 / c.lpr) * c.u)) != 0)
    return -1;
  return id;
}

template <typename T>
static void launchGemvKnRing(int id, hipStream_t s, const T *A, const T *B,
                             const T *C, T *Y, int M, int N, int K,
                             float alpha, float beta, int cDim0, int cDim1) {
  constexpr int VE = 16 / static_cast<int>(sizeof(T));
  const KnRingCfg &c = kKnRing[id];
  const int nb = c.lpr * VE;
  const dim3 grid((N + nb - 1) / nb, (M + c.mr - 1) / c.mr);
#define GEMM_KN_RING_CASE(ID, MR, LPR, KW, U, NS)                              \
  case ID:                                                                     \
    if constexpr ((ID >= 3) == std::is_same<T, float>::value) {                \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(gemm_gemv_kn_ring_kernel<T, MR, LPR, KW, U, NS>),    \
          grid, dim3(KW * 32), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,  \
          cDim1);                                                              \
    }                                                                          \
    break;
  switch (id) {
    GEMM_KN_RING_CFGS(GEMM_KN_RING_CASE)
  default:
    break;
  }
#undef GEMM_KN_RING_CASE
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] gemv-kn-ring M=%d N=%d K=%d -> ring-cfg[%d] mr=%d lpr=%d kw=%d "
      "u=%d ns=%d\n",
      M, N, K, id, c.mr, c.lpr, c.kw, c.u, c.ns);
}

/* B = [K, N] GEMV launch (gemm_gemv_kn_kernel). {BLOCK, KS} ladder from
 * wide (16 column groups x 8 columns per block) to narrow; the first entry
 * that puts >= 80 blocks on the 40-CU part wins (decode N=4096 needs the
 * narrow ones to fill the GPU). HIPDNN_EP_GEMM_GEMV_KN_CFG=<idx> forces one
 * (A/B only). */
template <typename T, bool TA, int MR>
static void launchGemvKnMr(hipStream_t s, const T *A, const T *B, const T *C,
                           T *Y, int M, int N, int K, float alpha, float beta,
                           int cDim0, int cDim1) {
  static constexpr int kCols[] = {128, 64, 32};
  static const int kn_forced = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG", -1);
  int id = kn_forced;
  if (id < 0 || id > 2) {
    id = 2;
    for (int i = 0; i < 2; ++i)
      if (((N + kCols[i] - 1) / kCols[i]) * ((M + MR - 1) / MR) >= 80) {
        id = i;
        break;
      }
    /* NN M = 1 on a B past the Infinity Cache: the 16-wave, 128-column-group
     * entry streams DRAM best whatever the block count (the ladder's 32 or
     * 64-column blocks on N = 4096 / 8192 read 64 or 128 bytes of a k-row per
     * block). Screening harness, M = 1, ms fp16 / fp32, ladder pick -> 0:
     * 4096x14336 0.977 / 1.372 -> 0.543 / 0.996, 8192^2 0.777 / 1.467 ->
     * 0.607 / 1.197, 4096^2 fp32 0.390 -> 0.292 (bench3 in the commit).
     * HIPDNN_EP_GEMM_KN_RING=0 keeps the ladder. */
    if (!TA && MR == 1 && gemmKnRingOn() &&
        static_cast<size_t>(N) * K * sizeof(T) > (32ull << 20))
      id = 0;
  }
  const dim3 grid((N + kCols[id] - 1) / kCols[id], (M + MR - 1) / MR);
#define GEMM_GEMV_KN(BLOCK, KS)                                                \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_kernel<T, TA, MR, BLOCK, KS>), \
                     grid, dim3(BLOCK), 0, s, A, B, C, Y, M, N, K, alpha, beta, \
                     cDim0, cDim1)
  if (id == 0)
    GEMM_GEMV_KN(256, 16);
  else if (id == 1)
    GEMM_GEMV_KN(512, 64);
  else
    GEMM_GEMV_KN(512, 128);
#undef GEMM_GEMV_KN
  CUSTOM_KERNELS_DEBUG_LOG("[gemm] gemv-kn M=%d N=%d K=%d ta=%d -> kn-cfg[%d]\n",
                           M, N, K, TA ? 1 : 0, id);
}

template <typename T, bool TA>
static void launchGemvKn(hipStream_t s, const T *A, const T *B, const T *C,
                         T *Y, int M, int N, int K, float alpha, float beta,
                         int cDim0, int cDim1) {
  if constexpr (!TA) {
    const int ring = gemmKnRingConfig<T>(A, B, M, N, K);
    if (ring >= 0) {
      launchGemvKnRing<T>(ring, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                          cDim1);
      return;
    }
  }
  if (M == 1)
    launchGemvKnMr<T, TA, 1>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
  else if (M <= 4)
    launchGemvKnMr<T, TA, 4>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
  else
    launchGemvKnMr<T, TA, 8>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
}

template <typename T>
static int launchSmallM(hipStream_t s, const T *A, const T *B, const T *C,
                         T *Y, int M, int N, int K, int transA, int transB,
                         float alpha, float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
  constexpr bool k16 = std::is_same<T, _Float16>::value ||
                       std::is_same<T, __hip_bfloat16>::value;
  if constexpr (k16) {
    /* Multi-row GEMV: B streamed once for all M rows (transA any M < 16;
     * NT 2 <= M < 16 unless HIPDNN_EP_GEMM_GEMV_MR=0). wave64 reaches here
     * with transA and M >= 16 too: that keeps the kn / scalar kernels. */
    static const bool mr_on = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_MR", 1) != 0;
    /* Tiny problems (N*K < 64K) stay on the one-row kernels: launch-bound,
     * where the multi-row kernels' fixed cost loses. */
    const bool mr_big = static_cast<size_t>(N) * K >= (size_t{1} << 16);
    if (transB && M < 16 && mr_big && (transA || (M >= 2 && mr_on))) {
#define GEMM_GEMV_MR(TA, MR, TN)                                               \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_mr_nt_kernel<T, TA, MR, TN>),   \
                     dim3((N + TN - 1) / TN), dim3(128), 0, s, A, B, C, Y, M, \
                     N, K, alpha, beta, cDim0, cDim1)
#define GEMM_GEMV_MR_ALL(TA)                                                   \
  do {                                                                         \
    if (M <= 4)                                                                \
      GEMM_GEMV_MR(TA, 4, 8);                                                  \
    else if (M <= 8)                                                           \
      GEMM_GEMV_MR(TA, 8, 8);                                                  \
    else                                                                       \
      GEMM_GEMV_MR(TA, 16, 4);                                                 \
  } while (0)
      if (transA)
        GEMM_GEMV_MR_ALL(true);
      else
        GEMM_GEMV_MR_ALL(false);
#undef GEMM_GEMV_MR_ALL
#undef GEMM_GEMV_MR
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] gemv-mr M=%d N=%d K=%d ta=%d\n", M, N,
                               K, transA);
      return hipGetLastError() == hipSuccess ? 0 : 1;
    }
    if (!transB && mr_big && (transA || mr_on)) {
      /* B = [K, N]: 8 columns per thread, 16-byte B loads, split-K in the
       * block (transA any M; NN M < 16 unless HIPDNN_EP_GEMM_GEMV_MR=0). */
      if (transA)
        launchGemvKn<T, true>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                              cDim1);
      else
        launchGemvKn<T, false>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                               cDim1);
      return hipGetLastError() == hipSuccess ? 0 : 1;
    }
  }
  if constexpr (std::is_same<T, float>::value) {
    /* fp32 takes the same multi-row NT (2 <= M < 16) and B = [K, N] kernels,
     * reading 8-float pieces as two 16-byte loads: no transA staging, and the
     * vector loads need 16-byte aligned A and B. */
    static const bool mr_on = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_MR", 1) != 0;
    const bool mr_big = static_cast<size_t>(N) * K >= (size_t{1} << 16);
    const bool aligned = ((reinterpret_cast<uintptr_t>(A) |
                           reinterpret_cast<uintptr_t>(B)) &
                          15) == 0;
    if (!transA && mr_on && mr_big && aligned) {
      if (transB && M >= 2) {
        if (M <= 4)
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_gemv_mr_nt_kernel<T, false, 4, 8>),
              dim3((N + 7) / 8), dim3(128), 0, s, A, B, C, Y, M, N, K, alpha,
              beta, cDim0, cDim1);
        else if (M <= 8)
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_gemv_mr_nt_kernel<T, false, 8, 8>),
              dim3((N + 7) / 8), dim3(128), 0, s, A, B, C, Y, M, N, K, alpha,
              beta, cDim0, cDim1);
        else
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_gemv_mr_nt_kernel<T, false, 16, 4>),
              dim3((N + 3) / 4), dim3(128), 0, s, A, B, C, Y, M, N, K, alpha,
              beta, cDim0, cDim1);
        CUSTOM_KERNELS_DEBUG_LOG("[gemm] gemv-mr M=%d N=%d K=%d ta=0\n", M, N,
                                 K);
        return hipGetLastError() == hipSuccess ? 0 : 1;
      }
      if (!transB) {
        launchGemvKn<T, false>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                               cDim1);
        return hipGetLastError() == hipSuccess ? 0 : 1;
      }
    }
  }
  if (transA) {
    dim3 grid((N + 255) / 256, M);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_ta_kernel<T>), grid, dim3(256),
                       0, s, A, B, C, Y, M, N, K, transB, alpha, beta, cDim0,
                       cDim1);
    return hipGetLastError() == hipSuccess ? 0 : 1;
  }
  if (transB) {
    int cfg = tuneGemvNt<T>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
    launchGemvNt<T>(cfg, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
  } else {
    int cfg = tuneGemvNn<T>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
    launchGemvNn<T>(cfg, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
  }
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* =========================================================================
 * WMMA GEMM — matmul_nbits GemmFp16U4Impl with FUSED_DQ=false, plus NT/NN
 * B staging and an alpha/beta/C epilogue.
 *
 * BK is 32 for every tile so K%32==0 skips remainder logic in the common
 * LLM/vision shapes, and double-buffered (BM+BN)*(BK+8) halfs stay under
 * 64 KB LDS. K remainder is still handled (vision K=4304).
 * ========================================================================= */


template <int BM, int BN, int WT_M, int WT_N, bool TransB, typename T, int BK,
          bool BC = true, bool DB = true, int PGR = 1, bool TransA = false>
__device__ __forceinline__ void GemmWmmaImpl(const T *__restrict__ A,
                                              const T *__restrict__ B,
                                              const T *__restrict__ C,
                                              T *__restrict__ Y, int M, int N,
                                              int K, float alpha, float beta,
                                              int cDim0, int cDim1,
                                              int swizzle_n,
                                              float *__restrict__ sk_acc,
                                              int ldb, int n_base, int lda,
                                              int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int PAD_K = 8;
  constexpr int K_STR = BK + PAD_K;
  constexpr int K_STEPS = BK / WMMA_TILE;

  constexpr int A_HPT = (BM * BK) / THR;
  /* 16-byte A chunks when they tile a thread's share; 8-byte ones otherwise
   * (BM = 96 at BK = 16 / 32 gives 12 halfs per thread). */
  constexpr int A_CH = (A_HPT >= 8 && A_HPT % 8 == 0) ? 8 : 4;
  constexpr int A_NCH = A_HPT / A_CH;
  constexpr int A_ROW_SPLIT = BK / A_CH;

  constexpr int B_HPT = (BN * BK) / THR;
  constexpr int B_U32 = B_HPT / 8; /* TransB: 8 halfs per u32x4 */
  constexpr int B_TPR = BK / B_HPT;
  /* Non-power-of-two BN (96 / 80, hipBLASLt's MT96 / MT80 N-tiles): one
   * B_HPT-wide run per thread no longer tiles the BN x BK panel, so B is
   * staged like A -- B_NCH chunks of B_CH halfs per thread, row = chunk /
   * (BK / B_CH). PGR bit 4 asks for the same staging on a power-of-two tile
   * (BK / 8 lanes per B row: one load instruction covers whole 128 B lines
   * of a few rows instead of 16-32 B pieces of many rows). Every other tile
   * keeps the one-run mapping. NT only. */
  constexpr bool B_GEN =
      (PGR & 4) != 0 || (B_TPR * BN != THR) || (BK % B_HPT) != 0;
  constexpr int B_CH = ((BN * BK) % (8 * THR)) == 0 ? 8 : 4;
  constexpr int B_NCH = B_GEN ? (BN * BK) / (B_CH * THR) : 1;
  constexpr int B_ROW_SPLIT = BK / B_CH;
  constexpr int PF_B = B_NCH > 8 ? B_NCH : 8;
  static_assert(!B_GEN || (TransB && B_NCH * B_CH * THR == BN * BK),
                "B_GEN chunking");

  static_assert(A_NCH >= 1, "A_NCH");
  static_assert(A_HPT % A_CH == 0, "A_HPT");
  static_assert(BK % A_CH == 0, "BK/A_CH");
  static_assert(BK % WMMA_TILE == 0, "BK/WMMA");
  static_assert((K_STR * 2) % 16 == 0, "LDS stride 16B");
  static_assert(B_GEN || (B_HPT >= 4 && B_HPT % 4 == 0), "B_HPT");
  static_assert(B_GEN || B_HPT <= 64, "pf_b cap");

  /* TransA: A is stored [K, M] (lda = row stride of that layout, normally M).
   * A thread loads TA_KR consecutive k-rows of one TA_MC-wide m vector (one
   * 4/8/16-byte load per row) and writes the TA_MC transposed k-runs as packed
   * TA_KR-half (8-byte) LDS stores, so the LDS tile keeps the [m][k] layout the
   * WMMA fragment reads expect -- no 2-byte scalar ds_write. Consecutive
   * threads take consecutive m vectors, so each row load is coalesced. */
  constexpr int TA_KR = 4;
  constexpr int TA_MC = (A_HPT / TA_KR >= 8) ? 8 : A_HPT / TA_KR;
  constexpr int TA_NCH = TransA ? A_HPT / (TA_KR * TA_MC) : 1;
  constexpr int TA_MG = BM / TA_MC;
  static_assert(!TransA || (TA_MC == 2 || TA_MC == 4 || TA_MC == 8), "TA_MC");
  static_assert(!TransA || (BK % TA_KR == 0 && BM % TA_MC == 0 &&
                            TA_NCH * TA_KR * TA_MC == A_HPT),
                "TransA chunking");
  /* NBUF LDS buffers: 2 = double-buffered (default, hides global-load latency
   * behind compute); 1 = single-buffered, half the LDS so two workgroups fit
   * per CU and hide the K-loop barriers across each other -- a win for large
   * LDS-bound tiles (256x128) where double-buffer caps occupancy at 1 WG/CU. */
  /* Double-buffer only when it actually fits in the 64 KB LDS. launchWmma picks
   * db=false for any tile whose double-buffered LDS is too big, so it is always
   * launched single-buffered -- but the dispatch macro still instantiates that
   * tile's dead DB=true variant. Without this cap, a large tile (e.g. a 256-wide
   * BK=64 tile at 108 KB double-buffered) fails the static_assert at compile
   * time even though it is never launched double-buffered. Falling NBUF back to
   * 1 keeps it compilable and never changes a launched tile (the runtime db flag
   * matches this predicate). */
  constexpr int NBUF = (DB && (BM + BN) * K_STR * 2 * 2 <= 65536) ? 2 : 1;
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  /* n_base shifts this launch's column origin. The N-edge grid-split launches
   * the aligned interior with n_base=0 (BC=false) and the ragged edge column
   * tile with n_base=floor(N/BN)*BN (BC=true), so the interior keeps the fast
   * path even when N%BN!=0. */
  const int col0 = n_base + bx * BN;

  constexpr int B_COL_SHIFT = __builtin_ctz(B_TPR > 0 ? B_TPR : 1);
  const int b_col = tid >> B_COL_SHIFT;
  const int b_k8 = (tid & (B_TPR - 1)) * B_HPT;
  const int b_n_g_raw = col0 + b_col;
  const bool b_valid = b_n_g_raw < N;
  const int b_n_g = b_valid ? b_n_g_raw : 0;

  /* NN (TransB=0): one thread owns an N-contiguous chunk of one K-row. */
  constexpr int N_CH = B_HPT;
  constexpr int N_CHUNKS = BN / N_CH;
  const int nn_k = tid / N_CHUNKS;
  const int nn_n = (tid % N_CHUNKS) * N_CH;

  int k_begin = 0;
  int k_end = K;
  if (gridDim.z > 1) {
    const int nkt = (K + BK - 1) / BK;
    const int z = static_cast<int>(blockIdx.z);
    const int sk = static_cast<int>(gridDim.z);
    const int kt0 = static_cast<int>((static_cast<long long>(z) * nkt) / sk);
    const int kt1 =
        static_cast<int>((static_cast<long long>(z + 1) * nkt) / sk);
    k_begin = kt0 * BK;
    k_end = kt1 * BK;
    if (k_end > K)
      k_end = K;
  }

  /* StaggerU: each workgroup starts its K loop `shift` tiles into [k_begin,
   * k_end) and wraps, so workgroups that run in lockstep do not hit the same
   * power-of-two-strided B/A rows in the same cache sets at the same time.
   * stag packs (SU-1) | step_tiles << 8 | map << 20; 0 = off. Only when the K
   * range is a whole number of BK tiles (the wrap needs an exact tile edge). */
  const int span = k_end - k_begin;
  const int niter = (span + BK - 1) / BK;
  int shift = 0;
  if (stag != 0 && (span % BK) == 0 && niter > 1) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter);
  }
  const int k_first = k_begin + shift * BK;

  int a_row[A_NCH];
  int a_koff[A_NCH];
  const T *a_src[A_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int idx = tid + c * THR;
    a_row[c] = idx / A_ROW_SPLIT;
    a_koff[c] = (idx % A_ROW_SPLIT) * A_CH;
    /* BC=false with a ragged M (NT, launchWmma): rows past M re-read the last
     * row (their outputs are masked in the epilogue). */
    const int ar = row0 + a_row[c];
    a_src[c] = A + static_cast<size_t>(BC || ar < M ? ar : M - 1) * lda +
               k_first + a_koff[c];
  }

  /* ldb is the B row stride in elements (K for a dense [N, K] B). Only the
   * row base scales by ldb -- the within-row k range is still [0, K). */
  const T *b_src_nt = B + static_cast<size_t>(b_n_g) * ldb + k_first +
                      b_k8; /* TransB=1 [N,K] */
  int b_row[B_NCH];
  int b_koff[B_NCH];
  bool b_ok[B_NCH];
  const T *b_src_g[B_NCH];
  if constexpr (B_GEN) {
#pragma unroll
    for (int c = 0; c < B_NCH; ++c) {
      const int idx = tid + c * THR;
      b_row[c] = idx / B_ROW_SPLIT;
      b_koff[c] = (idx % B_ROW_SPLIT) * B_CH;
      const int n = col0 + b_row[c];
      b_ok[c] = n < N;
      b_src_g[c] = B + static_cast<size_t>(b_ok[c] ? n : 0) * ldb + k_first +
                   b_koff[c];
    }
  }

  int ta_m[TA_NCH];
  int ta_k[TA_NCH];
  const T *ta_src[TA_NCH];
  if constexpr (TransA) {
#pragma unroll
    for (int c = 0; c < TA_NCH; ++c) {
      const int q = tid + c * THR;
      ta_m[c] = (q % TA_MG) * TA_MC;
      ta_k[c] = (q / TA_MG) * TA_KR;
      ta_src[c] = A + static_cast<size_t>(k_first + ta_k[c]) * lda + row0 +
                  ta_m[c];
    }
  }
  /* Vector A loads need every row start TA_MC-aligned (BC path only; the
   * launcher keeps lda%8==0 for BC=false). */
  const bool ta_vec = (lda % TA_MC) == 0;

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  /* PGR (prefetch global read depth): 1 = the next tile's global loads are in
   * flight during one compute step; 2 = two tiles in flight (register slot
   * per tile), hipBLASLt's PGR2. Only instantiated for the BK=32 NT tiles the
   * aliasing-K launch rule uses (launchWmma); BK=64 tiles lose with it. */
  constexpr int PGRV = PGR & 3; /* bit 4: B staging, bit 8: bias epilogue */
  static_assert(PGRV == 1 || PGRV == 2, "PGR");
  constexpr int NSLOT = PGRV;
  uint32_t pf_a[NSLOT][A_NCH][A_CH / 2];
  uint32_t pf_ta[NSLOT][TA_NCH][TA_KR][TA_MC / 2];
  u32x4 pf_b[NSLOT][PF_B];
  alignas(16) T pf_nn[N_CH];

  auto issueLoads = [&](int k0, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    /* After the last tile of the range the source pointers wrap back to
     * k_begin (a no-op without stagger: nothing is loaded after the last tile). */
    const int adv = (k0 + BK >= k_end) ? (BK - span) : BK;
    if constexpr (TransA) {
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
        const int m = row0 + ta_m[c];
#pragma unroll
        for (int r = 0; r < TA_KR; ++r) {
          const T *src = ta_src[c] + static_cast<size_t>(r) * lda;
          const int k = k0 + ta_k[c] + r;
          if (!BC || (ta_vec && k < K && m + TA_MC <= M)) {
            if constexpr (TA_MC == 8)
              *reinterpret_cast<u32x4 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x4 *>(src);
            else if constexpr (TA_MC == 4)
              *reinterpret_cast<u32x2 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x2 *>(src);
            else
              pf_ta[S][c][r][0] = *reinterpret_cast<const uint32_t *>(src);
          } else {
#pragma unroll
            for (int e = 0; e < TA_MC; ++e)
              reinterpret_cast<T *>(pf_ta[S][c][r])[e] =
                  (k < K && m + e < M) ? src[e] : (T)0;
          }
        }
        ta_src[c] += static_cast<ptrdiff_t>(adv) * lda;
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        if constexpr (!BC) {
          /* Tile fully covers M and K%BK==0: no row/K bounds check. */
          if constexpr (A_CH == 8)
            *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x4 *>(a_src[c]);
          else
            *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x2 *>(a_src[c]);
        } else {
          const int row = row0 + a_row[c];
          const int k = k0 + a_koff[c];
          const bool ok = (row < M) && (k + A_CH <= K);
          if (ok) {
            if constexpr (A_CH == 8)
              *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x4 *>(a_src[c]);
            else
              *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x2 *>(a_src[c]);
          } else {
#pragma unroll
            for (int e = 0; e < A_CH / 2; ++e)
              pf_a[S][c][e] = 0;
            if (row < M) {
#pragma unroll
              for (int e = 0; e < A_CH; ++e) {
                const int kk = k + e;
                const T v =
                    (kk < K) ? A[static_cast<size_t>(row) * lda + kk] : (T)0;
                reinterpret_cast<T *>(pf_a[S][c])[e] = v;
              }
            }
          }
        }
        a_src[c] += adv;
      }
    }

    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        if (!BC || (b_ok[c] && k0 + b_koff[c] + B_CH <= K)) {
          if constexpr (B_CH == 8)
            pf_b[S][c] = *reinterpret_cast<const u32x4 *>(b_src_g[c]);
          else
            *reinterpret_cast<u32x2 *>(&pf_b[S][c]) =
                *reinterpret_cast<const u32x2 *>(b_src_g[c]);
        } else {
          pf_b[S][c] = u32x4{0, 0, 0, 0};
          if (b_ok[c]) {
#pragma unroll
            for (int e = 0; e < B_CH; ++e)
              if (k0 + b_koff[c] + e < K)
                reinterpret_cast<T *>(&pf_b[S][c])[e] = b_src_g[c][e];
          }
        }
        b_src_g[c] += adv;
      }
    } else if constexpr (TransB) {
      if (!BC || (b_valid && k0 + b_k8 + B_HPT <= K)) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(pf_b[S]) =
              *reinterpret_cast<const u32x2 *>(b_src_nt);
        } else if constexpr (B_U32 == 1) {
          pf_b[S][0] = *reinterpret_cast<const u32x4 *>(b_src_nt);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            pf_b[S][w] = reinterpret_cast<const u32x4 *>(b_src_nt)[w];
        }
      } else {
#pragma unroll
        for (int w = 0; w < 8; ++w)
          pf_b[S][w] = u32x4{0, 0, 0, 0};
        if (b_valid) {
#pragma unroll
          for (int e = 0; e < B_HPT; ++e) {
            const int kk = k0 + b_k8 + e;
            const T v = (kk < K) ? B[static_cast<size_t>(b_n_g) * ldb + kk]
                                  : (T)0;
            reinterpret_cast<T *>(pf_b[S])[e] = v;
          }
        }
      }
      b_src_nt += adv;
    } else {
      const int gk = k0 + nn_k;
      const int gn = col0 + nn_n;
      if (gn + N_CH <= N && (!BC || gk < K)) {
        if constexpr (N_CH >= 8) {
          constexpr int NV = N_CH / 8;
#pragma unroll
          for (int w = 0; w < NV; ++w)
            reinterpret_cast<u32x4 *>(pf_nn)[w] =
                reinterpret_cast<const u32x4 *>(
                    &B[static_cast<size_t>(gk) * N + gn])[w];
          if constexpr ((N_CH % 8) == 4)
            reinterpret_cast<u32x2 *>(pf_nn)[NV * 2] =
                *reinterpret_cast<const u32x2 *>(
                    &B[static_cast<size_t>(gk) * N + gn + NV * 8]);
        } else {
          *reinterpret_cast<u32x2 *>(pf_nn) =
              *reinterpret_cast<const u32x2 *>(
                  &B[static_cast<size_t>(gk) * N + gn]);
        }
      } else {
#pragma unroll
        for (int e = 0; e < N_CH; ++e)
          pf_nn[e] = (T)0;
        if (gk < K) {
#pragma unroll
          for (int e = 0; e < N_CH; ++e) {
            const int n = gn + e;
            if (n < N)
              pf_nn[e] = B[static_cast<size_t>(gk) * N + n];
          }
        }
      }
    }
  };

  auto storeToShared = [&](int buf, auto slot)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        T *dst = &smB[buf][b_row[c]][b_koff[c]];
        if (!BC || b_ok[c]) {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = pf_b[S][c];
          else
            *reinterpret_cast<u32x2 *>(dst) =
                *reinterpret_cast<const u32x2 *>(&pf_b[S][c]);
        } else {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = u32x4{0, 0, 0, 0};
          else
            *reinterpret_cast<u32x2 *>(dst) = u32x2{0, 0};
        }
      }
    } else if constexpr (TransB) {
      if (!BC || b_valid) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) =
              *reinterpret_cast<const u32x2 *>(pf_b[S]);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = pf_b[S][w];
        }
      } else {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) = u32x2{0, 0};
        } else {
          const u32x4 z{0, 0, 0, 0};
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = z;
        }
      }
    } else {
#pragma unroll
      for (int e = 0; e < N_CH; ++e)
        smB[buf][nn_n + e][nn_k] = pf_nn[e];
    }

    if constexpr (TransA) {
      /* Transpose TA_KR x TA_MC in registers: column j of the loaded rows is
       * half (j&1) of word j/2 in each row; pack rows (0,1) and (2,3). */
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
#pragma unroll
        for (int j = 0; j < TA_MC; ++j) {
          const uint32_t w0 = pf_ta[S][c][0][j / 2];
          const uint32_t w1 = pf_ta[S][c][1][j / 2];
          const uint32_t w2 = pf_ta[S][c][2][j / 2];
          const uint32_t w3 = pf_ta[S][c][3][j / 2];
          u32x2 v;
          if (j & 1) {
            v[0] = (w0 >> 16) | (w1 & 0xffff0000u);
            v[1] = (w2 >> 16) | (w3 & 0xffff0000u);
          } else {
            v[0] = (w0 & 0xffffu) | (w1 << 16);
            v[1] = (w2 & 0xffffu) | (w3 << 16);
          }
          *reinterpret_cast<u32x2 *>(&smA[buf][ta_m[c] + j][ta_k[c]]) = v;
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        T *dst = &smA[buf][a_row[c]][a_koff[c]];
        if constexpr (A_CH == 8)
          *reinterpret_cast<u32x4 *>(dst) =
              *reinterpret_cast<const u32x4 *>(pf_a[S][c]);
        else
          *reinterpret_cast<u32x2 *>(dst) =
              *reinterpret_cast<const u32x2 *>(pf_a[S][c]);
      }
    }
  };

  auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      const int k_off = hipdnn_wmma_k_off(sub);
      if constexpr (std::is_same<T, _Float16>::value) {
        half16 b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          half16 a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] =
                HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        }
      } else {
        bf16xN b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          bf16xN a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                       acc[wm][wn]);
        }
      }
    }
  };

  /* Scheduling fence after the next tile's global loads, only on the 128x256
   * WT4x2 BK=32 PGR1 tile: without it the StaggerU build sinks every one of
   * them below the tile's 16 WMMAs, right before the barrier whose ds_store
   * waits for them, so the load latency is never hidden (4096^3 on that tile
   * with padded B: 4.13-4.44 ms vs 3.77 with the fence). A fence on every
   * tile is not a win (bench3 stage3 pin1: phi3 -5%, qwen7b-up +7%,
   * 128x4096^2 +14%). */
  constexpr bool PIN_LD = BM == 128 && BN == 256 && WT_M == 4 && WT_N == 2 &&
                          BK == 32 && PGRV == 1 && !TransA;
  if (k_begin < k_end) {
    using S0 = std::integral_constant<int, 0>;
    auto nextK = [&](int k) __attribute__((always_inline)) {
      k += BK;
      return k >= k_end ? k_begin : k;
    };
    int kc = k_first;
    issueLoads(kc, S0{});
    if constexpr (PIN_LD)
      GEMM_SCHED_FENCE();
    if constexpr (PGRV == 2) {
      using S1 = std::integral_constant<int, 1>;
      if (niter > 1) {
        /* PGR2: tile t lives in slot t%2; each step issues tile it+1 and
         * stores tile it (issued one step earlier), so a tile has two compute
         * steps to arrive. The final step issues one wrapped, valid tile that
         * is never stored, which keeps every step's load count identical. */
        kc = nextK(kc);
        issueLoads(kc, S1{});
        if constexpr (PIN_LD)
          GEMM_SCHED_FENCE();
        storeToShared(0, S0{});
        HIPDNN_LDS_BARRIER();
        int buf = 0;
        int it = 1;
        while (true) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S1{});
            HIPDNN_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            HIPDNN_LDS_BARRIER();
            storeToShared(0, S1{});
            HIPDNN_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
          kc = nextK(kc);
          issueLoads(kc, S1{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S0{});
            HIPDNN_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            HIPDNN_LDS_BARRIER();
            storeToShared(0, S0{});
            HIPDNN_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
        }
        computeWMMA(DB ? buf : 0);
      } else {
        storeToShared(0, S0{});
        HIPDNN_LDS_BARRIER();
        computeWMMA(0);
      }
    } else {
      storeToShared(0, S0{});
      HIPDNN_LDS_BARRIER();

      if constexpr (DB) {
        int buf = 0;
        for (int it = 1; it < niter; ++it) {
          const int nxt = 1 - buf;
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          computeWMMA(buf);
          storeToShared(nxt, S0{});
          HIPDNN_LDS_BARRIER();
          buf = nxt;
        }
        computeWMMA(buf);
      } else {
        /* Single-buffer: the next tile's global loads (issueLoads) still
         * overlap compute via registers, but the LDS is reused, so a barrier
         * must bracket the overwrite -- one after compute finishes reading,
         * one after the store makes the new tile visible. */
        for (int it = 1; it < niter; ++it) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          computeWMMA(0);
          HIPDNN_LDS_BARRIER();
          storeToShared(0, S0{});
          HIPDNN_LDS_BARRIER();
        }
        computeWMMA(0);
      }
    }
  }

  const bool fast = !sk_acc && (beta == 0.0f || !C) && alpha == 1.0f;
  const size_t sk_off =
      sk_acc ? (static_cast<size_t>(blockIdx.z) * static_cast<size_t>(M) * N)
             : 0;

#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  /* Paired store (BC=false, plain Y = A*B, even N): lanes 2j and 2j+1 hold
   * columns c and c+1 of the same accumulator rows, so one DPP swap gives the
   * even lane both columns of row e and the odd lane both columns of row e+1:
   * one 4-byte store per two outputs instead of two 2-byte stores. bench3
   * stage3 c3a, hb/ours with / without: siglip-up 0.91 / 0.85, minilm-up
   * 0.67 / 0.60, bf16 vitb-up 0.91 / 0.86; large shapes within 1%. */
  /* PGR bit 8: the same paired store for Y = alpha * A*B + beta * C with C a
   * row-broadcast vector (cDim0 == 1, the bias epilogue). A lane's columns
   * do not change with the row, so it reads its C value once per fragment
   * column instead of once per output in gemmStoreY. */
  if constexpr (!BC && sizeof(T) == 2 && (PGR & 8) != 0) {
    if (!sk_acc && C && cDim0 == 1 && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
        const int cc = nbase + (lane & ~1);
        const float bc =
            nbase + lane < N
                ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
                : 0.0f;
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = fmaf(alpha, acc[wm][wn][e], bc);
            const float v1 = fmaf(alpha, acc[wm][wn][e + 1], bc);
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
  if constexpr (!BC && sizeof(T) == 2) {
    if (fast && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
          const int cc = nbase + (lane & ~1);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = acc[wm][wn][e];
            const float v1 = acc[wm][wn][e + 1];
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
#endif

#pragma unroll
  for (int wm = 0; wm < WT_M; ++wm) {
    const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = hipdnn_wmma_acc_row(sub, e);
        const int rr = mbase + r;
        const int cc = nbase + lane;
        const float v = acc[wm][wn][e];
        if constexpr (!BC) {
          /* BC=false requires K%BK==0; M and N may be padded up to whole tiles
           * by the caller (ragged fast path: launch full tiles, mask the row /
           * column overhang here). The store is the epilogue -- one cheap
           * compare per element, outside the K-loop -- so it does not touch
           * the hot path. */
          if (cc < N && rr < M) {
            if (sk_acc)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
            else if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        } else {
          const bool full = (rr < M && cc < N);
          if (sk_acc) {
            if (full)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
          } else if (full) {
            if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        }
      }
    }
  }
}

#if defined(__HIP_DEVICE_COMPILE__) && HIPDNN_WMMA_FRAG_ELEMS == 16
#define GEMM_OVERLAP_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while (0)
#else
#define GEMM_OVERLAP_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, bool TransB, typename T, int BK,
          bool BC = true, bool DB = true, int PGR = 1, bool TransA = false, int SCH = 1>
__device__ __forceinline__ void GemmWmmaOverlapImpl(const T *__restrict__ A,
                                              const T *__restrict__ B,
                                              const T *__restrict__ C,
                                              T *__restrict__ Y, int M, int N,
                                              int K, float alpha, float beta,
                                              int cDim0, int cDim1,
                                              int swizzle_n,
                                              float *__restrict__ sk_acc,
                                              int ldb, int n_base, int lda,
                                              int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int PAD_K = 8;
  constexpr int K_STR = BK + PAD_K;
  constexpr int K_STEPS = BK / WMMA_TILE;

  constexpr int A_HPT = (BM * BK) / THR;
  /* 16-byte A chunks when they tile a thread's share; 8-byte ones otherwise
   * (BM = 96 at BK = 16 / 32 gives 12 halfs per thread). */
  constexpr int A_CH = (A_HPT >= 8 && A_HPT % 8 == 0) ? 8 : 4;
  constexpr int A_NCH = A_HPT / A_CH;
  constexpr int A_ROW_SPLIT = BK / A_CH;

  constexpr int B_HPT = (BN * BK) / THR;
  constexpr int B_U32 = B_HPT / 8; /* TransB: 8 halfs per u32x4 */
  constexpr int B_TPR = BK / B_HPT;
  /* Non-power-of-two BN (96 / 80, hipBLASLt's MT96 / MT80 N-tiles): one
   * B_HPT-wide run per thread no longer tiles the BN x BK panel, so B is
   * staged like A -- B_NCH chunks of B_CH halfs per thread, row = chunk /
   * (BK / B_CH). PGR bit 4 asks for the same staging on a power-of-two tile
   * (BK / 8 lanes per B row: one load instruction covers whole 128 B lines
   * of a few rows instead of 16-32 B pieces of many rows). Every other tile
   * keeps the one-run mapping. NT only. */
  constexpr bool B_GEN =
      (PGR & 4) != 0 || (B_TPR * BN != THR) || (BK % B_HPT) != 0;
  constexpr int B_CH = ((BN * BK) % (8 * THR)) == 0 ? 8 : 4;
  constexpr int B_NCH = B_GEN ? (BN * BK) / (B_CH * THR) : 1;
  constexpr int B_ROW_SPLIT = BK / B_CH;
  constexpr int PF_B = B_NCH > 8 ? B_NCH : 8;
  static_assert(!B_GEN || (TransB && B_NCH * B_CH * THR == BN * BK),
                "B_GEN chunking");

  static_assert(A_NCH >= 1, "A_NCH");
  static_assert(A_HPT % A_CH == 0, "A_HPT");
  static_assert(BK % A_CH == 0, "BK/A_CH");
  static_assert(BK % WMMA_TILE == 0, "BK/WMMA");
  static_assert((K_STR * 2) % 16 == 0, "LDS stride 16B");
  static_assert(B_GEN || (B_HPT >= 4 && B_HPT % 4 == 0), "B_HPT");
  static_assert(B_GEN || B_HPT <= 64, "pf_b cap");

  /* TransA: A is stored [K, M] (lda = row stride of that layout, normally M).
   * A thread loads TA_KR consecutive k-rows of one TA_MC-wide m vector (one
   * 4/8/16-byte load per row) and writes the TA_MC transposed k-runs as packed
   * TA_KR-half (8-byte) LDS stores, so the LDS tile keeps the [m][k] layout the
   * WMMA fragment reads expect -- no 2-byte scalar ds_write. Consecutive
   * threads take consecutive m vectors, so each row load is coalesced. */
  constexpr int TA_KR = 4;
  constexpr int TA_MC = (A_HPT / TA_KR >= 8) ? 8 : A_HPT / TA_KR;
  constexpr int TA_NCH = TransA ? A_HPT / (TA_KR * TA_MC) : 1;
  constexpr int TA_MG = BM / TA_MC;
  static_assert(!TransA || (TA_MC == 2 || TA_MC == 4 || TA_MC == 8), "TA_MC");
  static_assert(!TransA || (BK % TA_KR == 0 && BM % TA_MC == 0 &&
                            TA_NCH * TA_KR * TA_MC == A_HPT),
                "TransA chunking");
  /* NBUF LDS buffers: 2 = double-buffered (default, hides global-load latency
   * behind compute); 1 = single-buffered, half the LDS so two workgroups fit
   * per CU and hide the K-loop barriers across each other -- a win for large
   * LDS-bound tiles (256x128) where double-buffer caps occupancy at 1 WG/CU. */
  /* Double-buffer only when it actually fits in the 64 KB LDS. launchWmma picks
   * db=false for any tile whose double-buffered LDS is too big, so it is always
   * launched single-buffered -- but the dispatch macro still instantiates that
   * tile's dead DB=true variant. Without this cap, a large tile (e.g. a 256-wide
   * BK=64 tile at 108 KB double-buffered) fails the static_assert at compile
   * time even though it is never launched double-buffered. Falling NBUF back to
   * 1 keeps it compilable and never changes a launched tile (the runtime db flag
   * matches this predicate). */
  constexpr int NBUF = (DB && (BM + BN) * K_STR * 2 * 2 <= 65536) ? 2 : 1;
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  /* n_base shifts this launch's column origin. The N-edge grid-split launches
   * the aligned interior with n_base=0 (BC=false) and the ragged edge column
   * tile with n_base=floor(N/BN)*BN (BC=true), so the interior keeps the fast
   * path even when N%BN!=0. */
  const int col0 = n_base + bx * BN;

  constexpr int B_COL_SHIFT = __builtin_ctz(B_TPR > 0 ? B_TPR : 1);
  const int b_col = tid >> B_COL_SHIFT;
  const int b_k8 = (tid & (B_TPR - 1)) * B_HPT;
  const int b_n_g_raw = col0 + b_col;
  const bool b_valid = b_n_g_raw < N;
  const int b_n_g = b_valid ? b_n_g_raw : 0;

  /* NN (TransB=0): one thread owns an N-contiguous chunk of one K-row. */
  constexpr int N_CH = B_HPT;
  constexpr int N_CHUNKS = BN / N_CH;
  const int nn_k = tid / N_CHUNKS;
  const int nn_n = (tid % N_CHUNKS) * N_CH;

  int k_begin = 0;
  int k_end = K;
  if (gridDim.z > 1) {
    const int nkt = (K + BK - 1) / BK;
    const int z = static_cast<int>(blockIdx.z);
    const int sk = static_cast<int>(gridDim.z);
    const int kt0 = static_cast<int>((static_cast<long long>(z) * nkt) / sk);
    const int kt1 =
        static_cast<int>((static_cast<long long>(z + 1) * nkt) / sk);
    k_begin = kt0 * BK;
    k_end = kt1 * BK;
    if (k_end > K)
      k_end = K;
  }

  /* StaggerU: each workgroup starts its K loop `shift` tiles into [k_begin,
   * k_end) and wraps, so workgroups that run in lockstep do not hit the same
   * power-of-two-strided B/A rows in the same cache sets at the same time.
   * stag packs (SU-1) | step_tiles << 8 | map << 20; 0 = off. Only when the K
   * range is a whole number of BK tiles (the wrap needs an exact tile edge). */
  const int span = k_end - k_begin;
  const int niter = (span + BK - 1) / BK;
  int shift = 0;
  if (stag != 0 && (span % BK) == 0 && niter > 1) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter);
  }
  const int k_first = k_begin + shift * BK;

  int a_row[A_NCH];
  int a_koff[A_NCH];
  const T *a_src[A_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int idx = tid + c * THR;
    a_row[c] = idx / A_ROW_SPLIT;
    a_koff[c] = (idx % A_ROW_SPLIT) * A_CH;
    /* BC=false with a ragged M (NT, launchWmma): rows past M re-read the last
     * row (their outputs are masked in the epilogue). */
    const int ar = row0 + a_row[c];
    a_src[c] = A + static_cast<size_t>(BC || ar < M ? ar : M - 1) * lda +
               k_first + a_koff[c];
  }

  /* ldb is the B row stride in elements (K for a dense [N, K] B). Only the
   * row base scales by ldb -- the within-row k range is still [0, K). */
  const T *b_src_nt = B + static_cast<size_t>(b_n_g) * ldb + k_first +
                      b_k8; /* TransB=1 [N,K] */
  int b_row[B_NCH];
  int b_koff[B_NCH];
  bool b_ok[B_NCH];
  const T *b_src_g[B_NCH];
  if constexpr (B_GEN) {
#pragma unroll
    for (int c = 0; c < B_NCH; ++c) {
      const int idx = tid + c * THR;
      b_row[c] = idx / B_ROW_SPLIT;
      b_koff[c] = (idx % B_ROW_SPLIT) * B_CH;
      const int n = col0 + b_row[c];
      b_ok[c] = n < N;
      b_src_g[c] = B + static_cast<size_t>(b_ok[c] ? n : 0) * ldb + k_first +
                   b_koff[c];
    }
  }

  int ta_m[TA_NCH];
  int ta_k[TA_NCH];
  const T *ta_src[TA_NCH];
  if constexpr (TransA) {
#pragma unroll
    for (int c = 0; c < TA_NCH; ++c) {
      const int q = tid + c * THR;
      ta_m[c] = (q % TA_MG) * TA_MC;
      ta_k[c] = (q / TA_MG) * TA_KR;
      ta_src[c] = A + static_cast<size_t>(k_first + ta_k[c]) * lda + row0 +
                  ta_m[c];
    }
  }
  /* Vector A loads need every row start TA_MC-aligned (BC path only; the
   * launcher keeps lda%8==0 for BC=false). */
  const bool ta_vec = (lda % TA_MC) == 0;

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  /* PGR (prefetch global read depth): 1 = the next tile's global loads are in
   * flight during one compute step; 2 = two tiles in flight (register slot
   * per tile), hipBLASLt's PGR2. Only instantiated for the BK=32 NT tiles the
   * aliasing-K launch rule uses (launchWmma); BK=64 tiles lose with it. */
  constexpr int PGRV = PGR & 3; /* bit 4: B staging, bit 8: bias epilogue */
  static_assert(PGRV == 1 || PGRV == 2, "PGR");
  constexpr int NSLOT = PGRV;
  uint32_t pf_a[NSLOT][A_NCH][A_CH / 2];
  uint32_t pf_ta[NSLOT][TA_NCH][TA_KR][TA_MC / 2];
  u32x4 pf_b[NSLOT][PF_B];
  alignas(16) T pf_nn[N_CH];

  auto issueLoads = [&](int k0, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    /* After the last tile of the range the source pointers wrap back to
     * k_begin (a no-op without stagger: nothing is loaded after the last tile). */
    const int adv = (k0 + BK >= k_end) ? (BK - span) : BK;
    if constexpr (TransA) {
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
        const int m = row0 + ta_m[c];
#pragma unroll
        for (int r = 0; r < TA_KR; ++r) {
          const T *src = ta_src[c] + static_cast<size_t>(r) * lda;
          const int k = k0 + ta_k[c] + r;
          if (!BC || (ta_vec && k < K && m + TA_MC <= M)) {
            if constexpr (TA_MC == 8)
              *reinterpret_cast<u32x4 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x4 *>(src);
            else if constexpr (TA_MC == 4)
              *reinterpret_cast<u32x2 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x2 *>(src);
            else
              pf_ta[S][c][r][0] = *reinterpret_cast<const uint32_t *>(src);
          } else {
#pragma unroll
            for (int e = 0; e < TA_MC; ++e)
              reinterpret_cast<T *>(pf_ta[S][c][r])[e] =
                  (k < K && m + e < M) ? src[e] : (T)0;
          }
        }
        ta_src[c] += static_cast<ptrdiff_t>(adv) * lda;
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        if constexpr (!BC) {
          /* Tile fully covers M and K%BK==0: no row/K bounds check. */
          if constexpr (A_CH == 8)
            *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x4 *>(a_src[c]);
          else
            *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x2 *>(a_src[c]);
        } else {
          const int row = row0 + a_row[c];
          const int k = k0 + a_koff[c];
          const bool ok = (row < M) && (k + A_CH <= K);
          if (ok) {
            if constexpr (A_CH == 8)
              *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x4 *>(a_src[c]);
            else
              *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x2 *>(a_src[c]);
          } else {
#pragma unroll
            for (int e = 0; e < A_CH / 2; ++e)
              pf_a[S][c][e] = 0;
            if (row < M) {
#pragma unroll
              for (int e = 0; e < A_CH; ++e) {
                const int kk = k + e;
                const T v =
                    (kk < K) ? A[static_cast<size_t>(row) * lda + kk] : (T)0;
                reinterpret_cast<T *>(pf_a[S][c])[e] = v;
              }
            }
          }
        }
        a_src[c] += adv;
      }
    }

    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        if (!BC || (b_ok[c] && k0 + b_koff[c] + B_CH <= K)) {
          if constexpr (B_CH == 8)
            pf_b[S][c] = *reinterpret_cast<const u32x4 *>(b_src_g[c]);
          else
            *reinterpret_cast<u32x2 *>(&pf_b[S][c]) =
                *reinterpret_cast<const u32x2 *>(b_src_g[c]);
        } else {
          pf_b[S][c] = u32x4{0, 0, 0, 0};
          if (b_ok[c]) {
#pragma unroll
            for (int e = 0; e < B_CH; ++e)
              if (k0 + b_koff[c] + e < K)
                reinterpret_cast<T *>(&pf_b[S][c])[e] = b_src_g[c][e];
          }
        }
        b_src_g[c] += adv;
      }
    } else if constexpr (TransB) {
      if (!BC || (b_valid && k0 + b_k8 + B_HPT <= K)) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(pf_b[S]) =
              *reinterpret_cast<const u32x2 *>(b_src_nt);
        } else if constexpr (B_U32 == 1) {
          pf_b[S][0] = *reinterpret_cast<const u32x4 *>(b_src_nt);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            pf_b[S][w] = reinterpret_cast<const u32x4 *>(b_src_nt)[w];
        }
      } else {
#pragma unroll
        for (int w = 0; w < 8; ++w)
          pf_b[S][w] = u32x4{0, 0, 0, 0};
        if (b_valid) {
#pragma unroll
          for (int e = 0; e < B_HPT; ++e) {
            const int kk = k0 + b_k8 + e;
            const T v = (kk < K) ? B[static_cast<size_t>(b_n_g) * ldb + kk]
                                  : (T)0;
            reinterpret_cast<T *>(pf_b[S])[e] = v;
          }
        }
      }
      b_src_nt += adv;
    } else {
      const int gk = k0 + nn_k;
      const int gn = col0 + nn_n;
      if (gn + N_CH <= N && (!BC || gk < K)) {
        if constexpr (N_CH >= 8) {
          constexpr int NV = N_CH / 8;
#pragma unroll
          for (int w = 0; w < NV; ++w)
            reinterpret_cast<u32x4 *>(pf_nn)[w] =
                reinterpret_cast<const u32x4 *>(
                    &B[static_cast<size_t>(gk) * N + gn])[w];
          if constexpr ((N_CH % 8) == 4)
            reinterpret_cast<u32x2 *>(pf_nn)[NV * 2] =
                *reinterpret_cast<const u32x2 *>(
                    &B[static_cast<size_t>(gk) * N + gn + NV * 8]);
        } else {
          *reinterpret_cast<u32x2 *>(pf_nn) =
              *reinterpret_cast<const u32x2 *>(
                  &B[static_cast<size_t>(gk) * N + gn]);
        }
      } else {
#pragma unroll
        for (int e = 0; e < N_CH; ++e)
          pf_nn[e] = (T)0;
        if (gk < K) {
#pragma unroll
          for (int e = 0; e < N_CH; ++e) {
            const int n = gn + e;
            if (n < N)
              pf_nn[e] = B[static_cast<size_t>(gk) * N + n];
          }
        }
      }
    }
  };

  auto storeToShared = [&](int buf, auto slot)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        T *dst = &smB[buf][b_row[c]][b_koff[c]];
        if (!BC || b_ok[c]) {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = pf_b[S][c];
          else
            *reinterpret_cast<u32x2 *>(dst) =
                *reinterpret_cast<const u32x2 *>(&pf_b[S][c]);
        } else {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = u32x4{0, 0, 0, 0};
          else
            *reinterpret_cast<u32x2 *>(dst) = u32x2{0, 0};
        }
      }
    } else if constexpr (TransB) {
      if (!BC || b_valid) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) =
              *reinterpret_cast<const u32x2 *>(pf_b[S]);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = pf_b[S][w];
        }
      } else {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) = u32x2{0, 0};
        } else {
          const u32x4 z{0, 0, 0, 0};
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = z;
        }
      }
    } else {
#pragma unroll
      for (int e = 0; e < N_CH; ++e)
        smB[buf][nn_n + e][nn_k] = pf_nn[e];
    }

    if constexpr (TransA) {
      /* Transpose TA_KR x TA_MC in registers: column j of the loaded rows is
       * half (j&1) of word j/2 in each row; pack rows (0,1) and (2,3). */
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
#pragma unroll
        for (int j = 0; j < TA_MC; ++j) {
          const uint32_t w0 = pf_ta[S][c][0][j / 2];
          const uint32_t w1 = pf_ta[S][c][1][j / 2];
          const uint32_t w2 = pf_ta[S][c][2][j / 2];
          const uint32_t w3 = pf_ta[S][c][3][j / 2];
          u32x2 v;
          if (j & 1) {
            v[0] = (w0 >> 16) | (w1 & 0xffff0000u);
            v[1] = (w2 >> 16) | (w3 & 0xffff0000u);
          } else {
            v[0] = (w0 & 0xffffu) | (w1 << 16);
            v[1] = (w2 & 0xffffu) | (w3 << 16);
          }
          *reinterpret_cast<u32x2 *>(&smA[buf][ta_m[c] + j][ta_k[c]]) = v;
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        T *dst = &smA[buf][a_row[c]][a_koff[c]];
        if constexpr (A_CH == 8)
          *reinterpret_cast<u32x4 *>(dst) =
              *reinterpret_cast<const u32x4 *>(pf_a[S][c]);
        else
          *reinterpret_cast<u32x2 *>(dst) =
              *reinterpret_cast<const u32x2 *>(pf_a[S][c]);
      }
    }
  };

  auto overlapFence = [&]() __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (SCH == 7)
      __builtin_amdgcn_sched_barrier(0x106);
    else
      __builtin_amdgcn_sched_barrier(0);
#endif
  };

  auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      const int k_off = hipdnn_wmma_k_off(sub);

      if constexpr (std::is_same<T, _Float16>::value) {
        half16 b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr (SCH == 7)
          __builtin_amdgcn_sched_barrier(0x006);
#endif
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          half16 a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] =
                HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        }
      } else {
        bf16xN b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr (SCH == 7)
          __builtin_amdgcn_sched_barrier(0x006);
#endif
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          bf16xN a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                       acc[wm][wn]);
        }
      }
    }
  };

  /* Preserve issueLoads before WMMA. Schedule 1 fences every instruction;
   * schedule 7 permits address ALU and LDS reads to cross, while retaining
   * VMEM and WMMA order. These masks describe permitted crossings. */
  constexpr bool PIN_LD = true;
  if (k_begin < k_end) {
    using S0 = std::integral_constant<int, 0>;
    auto nextK = [&](int k) __attribute__((always_inline)) {
      k += BK;
      return k >= k_end ? k_begin : k;
    };
    int kc = k_first;
    issueLoads(kc, S0{});
    if constexpr (PIN_LD)
      overlapFence();
    if constexpr (PGRV == 2) {
      using S1 = std::integral_constant<int, 1>;
      if (niter > 1) {
        /* PGR2: tile t lives in slot t%2; each step issues tile it+1 and
         * stores tile it (issued one step earlier), so a tile has two compute
         * steps to arrive. The final step issues one wrapped, valid tile that
         * is never stored, which keeps every step's load count identical. */
        kc = nextK(kc);
        issueLoads(kc, S1{});
        if constexpr (PIN_LD)
          overlapFence();
        storeToShared(0, S0{});
        GEMM_OVERLAP_LDS_BARRIER();
        int buf = 0;
        int it = 1;
        while (true) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            overlapFence();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S1{});
            GEMM_OVERLAP_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            GEMM_OVERLAP_LDS_BARRIER();
            storeToShared(0, S1{});
            GEMM_OVERLAP_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
          kc = nextK(kc);
          issueLoads(kc, S1{});
          if constexpr (PIN_LD)
            overlapFence();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S0{});
            GEMM_OVERLAP_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            GEMM_OVERLAP_LDS_BARRIER();
            storeToShared(0, S0{});
            GEMM_OVERLAP_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
        }
        computeWMMA(DB ? buf : 0);
      } else {
        storeToShared(0, S0{});
        GEMM_OVERLAP_LDS_BARRIER();
        computeWMMA(0);
      }
    } else {
      storeToShared(0, S0{});
      GEMM_OVERLAP_LDS_BARRIER();

      if constexpr (DB) {
        int buf = 0;
        for (int it = 1; it < niter; ++it) {
          const int nxt = 1 - buf;
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            overlapFence();
          computeWMMA(buf);
          storeToShared(nxt, S0{});
          GEMM_OVERLAP_LDS_BARRIER();
          buf = nxt;
        }
        computeWMMA(buf);
      } else {
        /* Single-buffer: the next tile's global loads (issueLoads) still
         * overlap compute via registers, but the LDS is reused, so a barrier
         * must bracket the overwrite -- one after compute finishes reading,
         * one after the store makes the new tile visible. */
        for (int it = 1; it < niter; ++it) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            overlapFence();
          computeWMMA(0);
          GEMM_OVERLAP_LDS_BARRIER();
          storeToShared(0, S0{});
          GEMM_OVERLAP_LDS_BARRIER();
        }
        computeWMMA(0);
      }
    }
  }

  const bool fast = !sk_acc && (beta == 0.0f || !C) && alpha == 1.0f;
  const size_t sk_off =
      sk_acc ? (static_cast<size_t>(blockIdx.z) * static_cast<size_t>(M) * N)
             : 0;

#if defined(__HIP_DEVICE_COMPILE__) && HIPDNN_WMMA_FRAG_ELEMS == 16
  /* Paired store (BC=false, plain Y = A*B, even N): lanes 2j and 2j+1 hold
   * columns c and c+1 of the same accumulator rows, so one DPP swap gives the
   * even lane both columns of row e and the odd lane both columns of row e+1:
   * one 4-byte store per two outputs instead of two 2-byte stores. bench3
   * stage3 c3a, hb/ours with / without: siglip-up 0.91 / 0.85, minilm-up
   * 0.67 / 0.60, bf16 vitb-up 0.91 / 0.86; large shapes within 1%. */
  /* PGR bit 8: the same paired store for Y = alpha * A*B + beta * C with C a
   * row-broadcast vector (cDim0 == 1, the bias epilogue). A lane's columns
   * do not change with the row, so it reads its C value once per fragment
   * column instead of once per output in gemmStoreY. */
  if constexpr (!BC && sizeof(T) == 2 && (PGR & 8) != 0) {
    if (!sk_acc && C && cDim0 == 1 && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
        const int cc = nbase + (lane & ~1);
        const float bc =
            nbase + lane < N
                ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
                : 0.0f;
#if defined(__HIP_DEVICE_COMPILE__)
        if constexpr (SCH == 7)
          __builtin_amdgcn_sched_barrier(0x006);
#endif
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = fmaf(alpha, acc[wm][wn][e], bc);
            const float v1 = fmaf(alpha, acc[wm][wn][e + 1], bc);
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
  if constexpr (!BC && sizeof(T) == 2) {
    if (fast && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
          const int cc = nbase + (lane & ~1);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = acc[wm][wn][e];
            const float v1 = acc[wm][wn][e + 1];
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
#endif

#pragma unroll
  for (int wm = 0; wm < WT_M; ++wm) {
    const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = hipdnn_wmma_acc_row(sub, e);
        const int rr = mbase + r;
        const int cc = nbase + lane;
        const float v = acc[wm][wn][e];
        if constexpr (!BC) {
          /* BC=false requires K%BK==0; M and N may be padded up to whole tiles
           * by the caller (ragged fast path: launch full tiles, mask the row /
           * column overhang here). The store is the epilogue -- one cheap
           * compare per element, outside the K-loop -- so it does not touch
           * the hot path. */
          if (cc < N && rr < M) {
            if (sk_acc)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
            else if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        } else {
          const bool full = (rr < M && cc < N);
          if (sk_acc) {
            if (full)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
          } else if (full) {
            if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        }
      }
    }
  }
}


#undef GEMM_OVERLAP_LDS_BARRIER

#define GEMM_WG_SZ(BM, BN, WM, WN)                                             \
  ((BM / (WM * 16)) * (BN / (WN * 16)) * 32)

template <int BM, int BN, int WT_M, int WT_N, bool TransB, typename T, int BK,
          bool BC = true, bool DB = true, int PGR = 1, bool TransA = false>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM, BN, WT_M, WT_N), GEMM_WG_SZ(BM, BN, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_kernel(const T *__restrict__ A, const T *__restrict__ B,
                      const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                      int K, float alpha, float beta, int cDim0, int cDim1,
                      int swizzle_n, float *__restrict__ sk_acc, int ldb,
                      int n_base, int lda, int stag) {
  GemmWmmaImpl<BM, BN, WT_M, WT_N, TransB, T, BK, BC, DB, PGR, TransA>(
      A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, swizzle_n, sk_acc, ldb,
      n_base, lda, stag);
}


/* Source scheduling controls for the measured plain NT shapes. Independent
 * instances keep every existing WMMA kernel and the online table unchanged. */
template <int BM, int BN, int WM, int WN, typename T, int BK, bool DB, int SCH>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM, BN, WM, WN), GEMM_WG_SZ(BM, BN, WM, WN))))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_overlap_kernel(const T *A, const T *B, const T *C, T *Y,
                              int M, int N, int K, float alpha, float beta,
                              int d0, int d1, int sw, int lda, int ldb,
                              int stag) {
  GemmWmmaOverlapImpl<BM, BN, WM, WN, true, T, BK, false, DB, 1, false, SCH>(
      A, B, C, Y, M, N, K, alpha, beta, d0, d1, sw, nullptr, ldb, 0, lda, stag);
}

/* =========================================================================
 * WMMA GEMM with a buffer-load read path (NT only; launchWmma's BUFLOAD rule
 * picks it).
 *
 * Same LDS layout and WMMA step as GemmWmmaImpl, a different K loop:
 *   - A and B are read with buffer_load_b128 through one buffer resource per
 *     operand whose base is the workgroup's first row (word3 0x31004000,
 *     hipBLASLt's gfx11 value). Each lane's byte offset is set up once before
 *     the loop and the K position is one SGPR offset per tile, so the loop
 *     has no 64-bit address arithmetic (GemmWmmaImpl: 4-6 v_add_co per tile
 *     plus the StaggerU wrap select).
 *   - Both operands are staged in 16-byte chunks, BK / 8 lanes per row (a
 *     whole 128 B line per row and instruction at BK = 64).
 *   - The loop is unrolled by two, so the LDS buffer and the register slot
 *     are compile-time offsets. StaggerU's wrap is a second copy of the loop
 *     with a second resource (base moved back by K), not a select in every
 *     trip; the shift is rounded down to an even tile count for that.
 * RP & 3 is the pipeline:
 *   1: the next tile's loads are issued before the step's WMMAs and stored
 *      after them (GemmWmmaImpl's PGR1 order);
 *   2: two register slots: tile t+2 is issued while tile t computes and tile
 *      t+1 is stored after it, so a load has two steps to arrive.
 * RP bit 4: sched_group_barrier order -- the first k-step's fragment loads,
 * then the WMMAs with the step's global loads spread between them and the
 * next k-steps' fragment loads in one batch each. RP bit 16: the same with
 * the fragment loads spread one share per WMMA. RP bit 8: Y = alpha * A*B +
 * beta * C with C a row-broadcast vector (cDim0 == 1, GemmWmmaImpl's PGR bit
 * 8 epilogue); without it Y = A*B.
 * Requires K % (2 * BK) == 0, the whole K range (no split-K) and even N
 * (paired 4-byte stores). M / N may be ragged: rows past M / N re-read the
 * last row and their outputs are masked.
 * On the gfx12-style WMMA encoding (gfx1170) the loads are plain global loads
 * from the same offsets (the buffer resource format there is not verified)
 * and the epilogue stores per element.
 * ========================================================================= */
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__device__ __forceinline__ void
GemmWmmaBufImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = BM / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = MODE == 2 ? 2 : 1;
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BM % RPI == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !HIPDNN_WMMA_GFX12
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if HIPDNN_WMMA_GFX12
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH)
        *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][lk]) = pf[S][i];
      else
        *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][lk]) =
            pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    if constexpr (std::is_same<T, _Float16>::value) {
      half16 b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &smB[P][noff + lane][ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        half16 a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &smA[P][moff + lane][ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] =
              HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    } else {
      bf16xN b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &smB[P][noff + lane][ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        bf16xN a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &smA[P][moff + lane][ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                     acc[wm][wn]);
      }
    }
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, MODE == 2 ? 1 - Q : 0>;
    if constexpr (IS)
      issueAll(SI{}, rsel);
#pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks)
      computeKs(PC{}, ks);
    schedStep(IS);
    if constexpr (ST) {
      if constexpr (!DB) {
        HIPDNN_LDS_BARRIER();
        GEMM_SCHED_FENCE();
      }
      storeAll(SS{}, PN{});
      HIPDNN_LDS_BARRIER();
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  HIPDNN_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = fmaf(al, acc[wm][wn][e], bc);
        const float v1 = fmaf(al, acc[wm][wn][e + 1], bc);
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (cc < N && rr < M)
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (cc < N && rr < M)
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf(fmaf(al, acc[wm][wn][e], bc));
      }
    }
  }
#endif
}

template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM, BN, WT_M, WT_N), GEMM_WG_SZ(BM, BN, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_buf_kernel(const T *__restrict__ A, const T *__restrict__ B,
                          const T *__restrict__ C, T *__restrict__ Y, int M,
                          int N, int K, float alpha, float beta, int cDim0,
                          int cDim1, int swizzle_n, int ldb, int lda,
                          int stag) {
  GemmWmmaBufImpl<BM, BN, WT_M, WT_N, T, BK, DB, RP>(
      A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, swizzle_n, ldb, lda,
      stag);
}

/* Rolling prefetch keeps one register slot per load group: store tile t+1
 * into alternate LDS, then reuse that slot for tile t+2. The original buffer
 * template remains unchanged so its existing instances keep their ISA. */
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__device__ __forceinline__ void
GemmWmmaBufRollImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = BM / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr bool ROLL = (RP & 32) != 0;
  constexpr bool PAD32 = (RP & 64) != 0;
  constexpr bool GROUP = (RP & 128) != 0;
  constexpr bool HB_GROUP = (RP & 256) != 0;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1;
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BM % RPI == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  static_assert(!ROLL || DB, "rolling stores require a second LDS buffer");
  constexpr int LSIZE = PAD32 ? 16384 : (BM + BN) * K_STR;
  constexpr int LALLOC = (NBUF - 1) * LSIZE + (BM + BN) * K_STR;
  static_assert(LSIZE >= (BM + BN) * K_STR && LALLOC * ES <= 65536, "LDS");
  __shared__ __align__(16) T sm[LALLOC];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !HIPDNN_WMMA_GFX12
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if HIPDNN_WMMA_GFX12
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH)
        *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + (lr + i * RPI) * K_STR + lk]) = pf[S][i];
      else
        *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + BM * K_STR + (lr + (i - A_NCH) * RPI) * K_STR + lk]) =
            pf[S][i];
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH)
      *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + (lr + i * RPI) * K_STR + lk]) = pf[S][i];
    else
      *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + BM * K_STR + (lr + (i - A_NCH) * RPI) * K_STR + lk]) = pf[S][i];
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    if constexpr (std::is_same<T, _Float16>::value) {
      half16 b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + BM * K_STR + (noff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        half16 a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + (moff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] =
              HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    } else {
      bf16xN b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + BM * K_STR + (noff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        bf16xN a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + (moff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                     acc[wm][wn]);
      }
    }
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((HB_GROUP && (i < K_STEPS - 1 ? i : K_STEPS - 1) == ks) ||
              (!HB_GROUP && ((!GROUP && ks == 0) || (GROUP && i * K_STEPS / NL == ks)))) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS) kb += KB;
    } else {
      if constexpr (IS) issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) computeKs(PC{}, ks);
      schedStep(IS);
    }
    if constexpr (ST && !ROLL) {
      if constexpr (!DB) {
        HIPDNN_LDS_BARRIER();
        GEMM_SCHED_FENCE();
      }
      storeAll(SS{}, PN{});
      HIPDNN_LDS_BARRIER();
    }
    if constexpr (ST && ROLL) HIPDNN_LDS_BARRIER();
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  HIPDNN_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = fmaf(al, acc[wm][wn][e], bc);
        const float v1 = fmaf(al, acc[wm][wn][e + 1], bc);
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (cc < N && rr < M)
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (cc < N && rr < M)
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf(fmaf(al, acc[wm][wn][e], bc));
      }
    }
  }
#endif
}

template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__global__ __launch_bounds__(GEMM_WG_SZ(BM, BN, WT_M, WT_N))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_bufroll_kernel(const T *__restrict__ A, const T *__restrict__ B,
                          const T *__restrict__ C, T *__restrict__ Y, int M,
                          int N, int K, float alpha, float beta, int cDim0,
                          int cDim1, int swizzle_n, int ldb, int lda,
                          int stag) {
  GemmWmmaBufRollImpl<BM, BN, WT_M, WT_N, T, BK, DB, RP>(
      A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, swizzle_n, ldb, lda,
      stag);
}

/* split_k is always 1: split-K needs a device workspace for its partial sums,
 * and gemm allocates none. The field stays because the LUT schema carries it,
 * so a LUT point recorded with split-K matches no entry and the search moves
 * on. */
struct WmmaCfg {
  int bm, bn, wt_m, wt_n, swizzle_n, split_k, bk;
};


/* Correct fallback for device encodings outside the measured gfx1151 route.
 * Each thread owns output cells. No shared workspace or device allocation. */
template <int BM, int BN, typename T>
__device__ __forceinline__ void GemmNnWidePortable(
    const T *A, const T *B, T *Y, int M, int N, int K) {
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  for (int q = threadIdx.x; q < BM * BN; q += blockDim.x) {
    const int r = row0 + q / BN, c = col0 + q % BN;
    if (r < M && c < N) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k)
        acc = fmaf(GemmElem<T>::tof(A[static_cast<size_t>(r) * K + k]),
                   GemmElem<T>::tof(B[static_cast<size_t>(k) * N + c]), acc);
      Y[static_cast<size_t>(r) * N + c] = GemmElem<T>::fromf(acc);
    }
  }
}

template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, bool AT = true>
__device__ __forceinline__ void
GemmNnWideManualImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = BM / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int BCGR = BN / 8;
  constexpr int B_NL = (BK * BCGR + THR - 1) / THR;
  constexpr int NL = (AT ? A_NCH : 0) + B_NL;
  constexpr int A_ST = AT ? A_NCH : 0;
  constexpr int B_SIZE = BK * BN + (BK / 16) * 8;
  constexpr bool ROLL = (RP & 32) != 0;
  static_assert(!ROLL || DB, "rolling needs double LDS");
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = !ROLL && MODE == 2 ? 2 : 1;
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BM % RPI == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * ((AT ? BM * K_STR : 0) + B_SIZE) * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][AT ? BM : 1][AT ? K_STR : 1];
  __shared__ __align__(16) T smB[NBUF][B_SIZE];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  const int jw = niter;
  constexpr int BS = BK;
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NL], bd[B_NL];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NL; ++c) {
    const int q = c * THR + tid;
    const int cg = q % BCGR, kq = q / BCGR;
    const int nc = col0 + cg * 8;
    const int nv = nc + 8 <= N ? nc : N - 8;
    voffB[c] = (kq * N + nv - col0) * ES;
    bd[c] = kq * BN + (kq / 16) * 8 + cg * 8;
  }
  const char *Ab = reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb = reinterpret_cast<const char *>(B + col0);
  const int kspan = K * ES;
#if !HIPDNN_WMMA_GFX12
  const auto ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab), 0, 0x7fffffff, 0x31004000);
  const auto rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb), 0, 0x7fffffff, 0x31004000);
#endif

  const uint64_t aptr = reinterpret_cast<uint64_t>(Ab);
  const uint64_t bptr = reinterpret_cast<uint64_t>(Bb);
  const u32x4 arsrc = {static_cast<uint32_t>(aptr), static_cast<uint32_t>(aptr >> 32), 0x7fffffffu, 0x31004000u};
  const u32x4 brsrc = {static_cast<uint32_t>(bptr), static_cast<uint32_t>(bptr >> 32), 0x7fffffffu, 0x31004000u};

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int ckb = 0;
  int kb = 0; /* byte offset of the next tile to issue */

  auto issueOne = [&](auto slot, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if (i < A_ST) {
#if HIPDNN_WMMA_GFX12
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Ab + voffA[i] + kb);
#else
      if constexpr((RP & 32768)!=0) {
        const int vo=voffA[i], so=kb; const u32x4 rs=arsrc; u32x4 val;
        asm volatile("buffer_load_b128 %0, %1, %2, %3 offen slc" : "=&v"(val) : "v"(vo), "s"(rs), "s"(so) : "memory");
        pf[S][i]=val;
      } else pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, voffA[i], kb, 2);
#endif
    } else {
      const int c = i - A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
#if HIPDNN_WMMA_GFX12
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Bb + voffB[c] + kb * N);
#else
      if constexpr((RP & 32768)!=0) {
        const int vo=voffB[c], so=kb*N; const u32x4 rs=brsrc; u32x4 val;
        asm volatile("buffer_load_b128 %0, %1, %2, %3 offen slc" : "=&v"(val) : "v"(vo), "s"(rs), "s"(so) : "memory");
        pf[S][i]=val;
      } else pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, voffB[c], kb * N, 2);
#endif
    }
  };
  auto issueAll = [&](auto slot) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i) {
      const int ii=((RP & 32768)!=0 && (RP & 16384)!=0 && AT) ? (i<B_NL?i+A_ST:i-B_NL) : i;
      issueOne(slot,ii);
    }
    kb += KB;
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value, P = decltype(buf)::value;
    if (i < A_ST) {
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
      if constexpr((RP & 32768)!=0) {
        const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smA[P][lr + i * RPI][lk]));
        const u32x4 val=pf[S][i];
        asm volatile("ds_store_b128 %0, %1" : : "v"(addr), "v"(val) : "memory");
      } else
#endif
      *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][lk]) = pf[S][i];
    }
    else {
      const int c=i-A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
      if constexpr((RP & 32768)!=0) {
        const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smB[P][bd[c]]));
        const u32x4 val=pf[S][i];
        asm volatile("ds_store_b128 %0, %1" : : "v"(addr), "v"(val) : "memory");
      } else
#endif
      *reinterpret_cast<u32x4 *>(&smB[P][bd[c]])=pf[S][i];
    }
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i)storeOne(slot,buf,i);
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
    if constexpr((RP & 4096)!=0) {
      const int k0=ks*16+k_off;
      const uint32_t da=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smA[P][wrow*16+lane][k0]));
      const int nc=wcol*16+lane;
      const uint32_t db=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&smB[P][k0*BN+(k0/16)*8+nc]));
      if constexpr(BM==128 && BN==96 && WT_M==4 && WT_N==3) {
        if constexpr(std::is_same<T,_Float16>::value) {
          asm volatile(
          "ds_load_u16 v224, %13 offset:0\n\t"
          "ds_load_u16_d16_hi v224, %13 offset:192\n\t"
          "ds_load_u16 v225, %13 offset:384\n\t"
          "ds_load_u16_d16_hi v225, %13 offset:576\n\t"
          "ds_load_u16 v226, %13 offset:768\n\t"
          "ds_load_u16_d16_hi v226, %13 offset:960\n\t"
          "ds_load_u16 v227, %13 offset:1152\n\t"
          "ds_load_u16_d16_hi v227, %13 offset:1344\n\t"
          "ds_load_u16 v228, %13 offset:1536\n\t"
          "ds_load_u16_d16_hi v228, %13 offset:1728\n\t"
          "ds_load_u16 v229, %13 offset:1920\n\t"
          "ds_load_u16_d16_hi v229, %13 offset:2112\n\t"
          "ds_load_u16 v230, %13 offset:2304\n\t"
          "ds_load_u16_d16_hi v230, %13 offset:2496\n\t"
          "ds_load_u16 v231, %13 offset:2688\n\t"
          "ds_load_u16_d16_hi v231, %13 offset:2880\n\t"
          "ds_load_u16 v232, %13 offset:64\n\t"
          "ds_load_u16_d16_hi v232, %13 offset:256\n\t"
          "ds_load_u16 v233, %13 offset:448\n\t"
          "ds_load_u16_d16_hi v233, %13 offset:640\n\t"
          "ds_load_u16 v234, %13 offset:832\n\t"
          "ds_load_u16_d16_hi v234, %13 offset:1024\n\t"
          "ds_load_u16 v235, %13 offset:1216\n\t"
          "ds_load_u16_d16_hi v235, %13 offset:1408\n\t"
          "ds_load_u16 v236, %13 offset:1600\n\t"
          "ds_load_u16_d16_hi v236, %13 offset:1792\n\t"
          "ds_load_u16 v237, %13 offset:1984\n\t"
          "ds_load_u16_d16_hi v237, %13 offset:2176\n\t"
          "ds_load_u16 v238, %13 offset:2368\n\t"
          "ds_load_u16_d16_hi v238, %13 offset:2560\n\t"
          "ds_load_u16 v239, %13 offset:2752\n\t"
          "ds_load_u16_d16_hi v239, %13 offset:2944\n\t"
          "ds_load_u16 v240, %13 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %13 offset:320\n\t"
          "ds_load_u16 v241, %13 offset:512\n\t"
          "ds_load_u16_d16_hi v241, %13 offset:704\n\t"
          "ds_load_u16 v242, %13 offset:896\n\t"
          "ds_load_u16_d16_hi v242, %13 offset:1088\n\t"
          "ds_load_u16 v243, %13 offset:1280\n\t"
          "ds_load_u16_d16_hi v243, %13 offset:1472\n\t"
          "ds_load_u16 v244, %13 offset:1664\n\t"
          "ds_load_u16_d16_hi v244, %13 offset:1856\n\t"
          "ds_load_u16 v245, %13 offset:2048\n\t"
          "ds_load_u16_d16_hi v245, %13 offset:2240\n\t"
          "ds_load_u16 v246, %13 offset:2432\n\t"
          "ds_load_u16_d16_hi v246, %13 offset:2624\n\t"
          "ds_load_u16 v247, %13 offset:2816\n\t"
          "ds_load_u16_d16_hi v247, %13 offset:3008\n\t"
          "ds_load_b128 v[192:195], %12 offset:0\n\t"
          "ds_load_b128 v[196:199], %12 offset:16\n\t"
          "ds_load_b128 v[200:203], %12 offset:4608\n\t"
          "ds_load_b128 v[204:207], %12 offset:4624\n\t"
          "ds_load_b128 v[208:211], %12 offset:9216\n\t"
          "ds_load_b128 v[212:215], %12 offset:9232\n\t"
          "ds_load_b128 v[216:219], %12 offset:13824\n\t"
          "ds_load_b128 v[220:223], %12 offset:13840\n\t"
          "s_waitcnt lgkmcnt(6)\n\t"
          "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
          "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
          "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
          "s_waitcnt lgkmcnt(4)\n\t"
          "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
          "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
          "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
          "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
          "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
          "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
          "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[0][2]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[1][2]),
            "+v"(acc[2][0]),
            "+v"(acc[2][1]),
            "+v"(acc[2][2]),
            "+v"(acc[3][0]),
            "+v"(acc[3][1]),
            "+v"(acc[3][2])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        } else {
          asm volatile(
          "ds_load_u16 v224, %13 offset:0\n\t"
          "ds_load_u16_d16_hi v224, %13 offset:192\n\t"
          "ds_load_u16 v225, %13 offset:384\n\t"
          "ds_load_u16_d16_hi v225, %13 offset:576\n\t"
          "ds_load_u16 v226, %13 offset:768\n\t"
          "ds_load_u16_d16_hi v226, %13 offset:960\n\t"
          "ds_load_u16 v227, %13 offset:1152\n\t"
          "ds_load_u16_d16_hi v227, %13 offset:1344\n\t"
          "ds_load_u16 v228, %13 offset:1536\n\t"
          "ds_load_u16_d16_hi v228, %13 offset:1728\n\t"
          "ds_load_u16 v229, %13 offset:1920\n\t"
          "ds_load_u16_d16_hi v229, %13 offset:2112\n\t"
          "ds_load_u16 v230, %13 offset:2304\n\t"
          "ds_load_u16_d16_hi v230, %13 offset:2496\n\t"
          "ds_load_u16 v231, %13 offset:2688\n\t"
          "ds_load_u16_d16_hi v231, %13 offset:2880\n\t"
          "ds_load_u16 v232, %13 offset:64\n\t"
          "ds_load_u16_d16_hi v232, %13 offset:256\n\t"
          "ds_load_u16 v233, %13 offset:448\n\t"
          "ds_load_u16_d16_hi v233, %13 offset:640\n\t"
          "ds_load_u16 v234, %13 offset:832\n\t"
          "ds_load_u16_d16_hi v234, %13 offset:1024\n\t"
          "ds_load_u16 v235, %13 offset:1216\n\t"
          "ds_load_u16_d16_hi v235, %13 offset:1408\n\t"
          "ds_load_u16 v236, %13 offset:1600\n\t"
          "ds_load_u16_d16_hi v236, %13 offset:1792\n\t"
          "ds_load_u16 v237, %13 offset:1984\n\t"
          "ds_load_u16_d16_hi v237, %13 offset:2176\n\t"
          "ds_load_u16 v238, %13 offset:2368\n\t"
          "ds_load_u16_d16_hi v238, %13 offset:2560\n\t"
          "ds_load_u16 v239, %13 offset:2752\n\t"
          "ds_load_u16_d16_hi v239, %13 offset:2944\n\t"
          "ds_load_u16 v240, %13 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %13 offset:320\n\t"
          "ds_load_u16 v241, %13 offset:512\n\t"
          "ds_load_u16_d16_hi v241, %13 offset:704\n\t"
          "ds_load_u16 v242, %13 offset:896\n\t"
          "ds_load_u16_d16_hi v242, %13 offset:1088\n\t"
          "ds_load_u16 v243, %13 offset:1280\n\t"
          "ds_load_u16_d16_hi v243, %13 offset:1472\n\t"
          "ds_load_u16 v244, %13 offset:1664\n\t"
          "ds_load_u16_d16_hi v244, %13 offset:1856\n\t"
          "ds_load_u16 v245, %13 offset:2048\n\t"
          "ds_load_u16_d16_hi v245, %13 offset:2240\n\t"
          "ds_load_u16 v246, %13 offset:2432\n\t"
          "ds_load_u16_d16_hi v246, %13 offset:2624\n\t"
          "ds_load_u16 v247, %13 offset:2816\n\t"
          "ds_load_u16_d16_hi v247, %13 offset:3008\n\t"
          "ds_load_b128 v[192:195], %12 offset:0\n\t"
          "ds_load_b128 v[196:199], %12 offset:16\n\t"
          "ds_load_b128 v[200:203], %12 offset:4608\n\t"
          "ds_load_b128 v[204:207], %12 offset:4624\n\t"
          "ds_load_b128 v[208:211], %12 offset:9216\n\t"
          "ds_load_b128 v[212:215], %12 offset:9232\n\t"
          "ds_load_b128 v[216:219], %12 offset:13824\n\t"
          "ds_load_b128 v[220:223], %12 offset:13840\n\t"
          "s_waitcnt lgkmcnt(6)\n\t"
          "v_wmma_f32_16x16x16_bf16 %0, v[192:199], v[224:231], %0\n\t"
          "v_wmma_f32_16x16x16_bf16 %1, v[192:199], v[232:239], %1\n\t"
          "v_wmma_f32_16x16x16_bf16 %2, v[192:199], v[240:247], %2\n\t"
          "s_waitcnt lgkmcnt(4)\n\t"
          "v_wmma_f32_16x16x16_bf16 %3, v[200:207], v[224:231], %3\n\t"
          "v_wmma_f32_16x16x16_bf16 %4, v[200:207], v[232:239], %4\n\t"
          "v_wmma_f32_16x16x16_bf16 %5, v[200:207], v[240:247], %5\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_bf16 %6, v[208:215], v[224:231], %6\n\t"
          "v_wmma_f32_16x16x16_bf16 %7, v[208:215], v[232:239], %7\n\t"
          "v_wmma_f32_16x16x16_bf16 %8, v[208:215], v[240:247], %8\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_bf16 %9, v[216:223], v[224:231], %9\n\t"
          "v_wmma_f32_16x16x16_bf16 %10, v[216:223], v[232:239], %10\n\t"
          "v_wmma_f32_16x16x16_bf16 %11, v[216:223], v[240:247], %11\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[0][2]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[1][2]),
            "+v"(acc[2][0]),
            "+v"(acc[2][1]),
            "+v"(acc[2][2]),
            "+v"(acc[3][0]),
            "+v"(acc[3][1]),
            "+v"(acc[3][2])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        }
      }
      else if constexpr(BM==80 && BN==128 && WT_M==5 && WT_N==2) {
        if constexpr(std::is_same<T,_Float16>::value) {
          asm volatile(
          "ds_load_u16 v232, %11 offset:0\n\t"
          "ds_load_u16_d16_hi v232, %11 offset:256\n\t"
          "ds_load_u16 v233, %11 offset:512\n\t"
          "ds_load_u16_d16_hi v233, %11 offset:768\n\t"
          "ds_load_u16 v234, %11 offset:1024\n\t"
          "ds_load_u16_d16_hi v234, %11 offset:1280\n\t"
          "ds_load_u16 v235, %11 offset:1536\n\t"
          "ds_load_u16_d16_hi v235, %11 offset:1792\n\t"
          "ds_load_u16 v236, %11 offset:2048\n\t"
          "ds_load_u16_d16_hi v236, %11 offset:2304\n\t"
          "ds_load_u16 v237, %11 offset:2560\n\t"
          "ds_load_u16_d16_hi v237, %11 offset:2816\n\t"
          "ds_load_u16 v238, %11 offset:3072\n\t"
          "ds_load_u16_d16_hi v238, %11 offset:3328\n\t"
          "ds_load_u16 v239, %11 offset:3584\n\t"
          "ds_load_u16_d16_hi v239, %11 offset:3840\n\t"
          "ds_load_u16 v240, %11 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %11 offset:384\n\t"
          "ds_load_u16 v241, %11 offset:640\n\t"
          "ds_load_u16_d16_hi v241, %11 offset:896\n\t"
          "ds_load_u16 v242, %11 offset:1152\n\t"
          "ds_load_u16_d16_hi v242, %11 offset:1408\n\t"
          "ds_load_u16 v243, %11 offset:1664\n\t"
          "ds_load_u16_d16_hi v243, %11 offset:1920\n\t"
          "ds_load_u16 v244, %11 offset:2176\n\t"
          "ds_load_u16_d16_hi v244, %11 offset:2432\n\t"
          "ds_load_u16 v245, %11 offset:2688\n\t"
          "ds_load_u16_d16_hi v245, %11 offset:2944\n\t"
          "ds_load_u16 v246, %11 offset:3200\n\t"
          "ds_load_u16_d16_hi v246, %11 offset:3456\n\t"
          "ds_load_u16 v247, %11 offset:3712\n\t"
          "ds_load_u16_d16_hi v247, %11 offset:3968\n\t"
          "ds_load_b128 v[192:195], %10 offset:0\n\t"
          "ds_load_b128 v[196:199], %10 offset:16\n\t"
          "ds_load_b128 v[200:203], %10 offset:2304\n\t"
          "ds_load_b128 v[204:207], %10 offset:2320\n\t"
          "ds_load_b128 v[208:211], %10 offset:4608\n\t"
          "ds_load_b128 v[212:215], %10 offset:4624\n\t"
          "ds_load_b128 v[216:219], %10 offset:6912\n\t"
          "ds_load_b128 v[220:223], %10 offset:6928\n\t"
          "ds_load_b128 v[224:227], %10 offset:9216\n\t"
          "ds_load_b128 v[228:231], %10 offset:9232\n\t"
          "s_waitcnt lgkmcnt(8)\n\t"
          "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[232:239], %0\n\t"
          "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[240:247], %1\n\t"
          "s_waitcnt lgkmcnt(6)\n\t"
          "v_wmma_f32_16x16x16_f16 %2, v[200:207], v[232:239], %2\n\t"
          "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[240:247], %3\n\t"
          "s_waitcnt lgkmcnt(4)\n\t"
          "v_wmma_f32_16x16x16_f16 %4, v[208:215], v[232:239], %4\n\t"
          "v_wmma_f32_16x16x16_f16 %5, v[208:215], v[240:247], %5\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_f16 %6, v[216:223], v[232:239], %6\n\t"
          "v_wmma_f32_16x16x16_f16 %7, v[216:223], v[240:247], %7\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_f16 %8, v[224:231], v[232:239], %8\n\t"
          "v_wmma_f32_16x16x16_f16 %9, v[224:231], v[240:247], %9\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[2][0]),
            "+v"(acc[2][1]),
            "+v"(acc[3][0]),
            "+v"(acc[3][1]),
            "+v"(acc[4][0]),
            "+v"(acc[4][1])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        } else {
          asm volatile(
          "ds_load_u16 v232, %11 offset:0\n\t"
          "ds_load_u16_d16_hi v232, %11 offset:256\n\t"
          "ds_load_u16 v233, %11 offset:512\n\t"
          "ds_load_u16_d16_hi v233, %11 offset:768\n\t"
          "ds_load_u16 v234, %11 offset:1024\n\t"
          "ds_load_u16_d16_hi v234, %11 offset:1280\n\t"
          "ds_load_u16 v235, %11 offset:1536\n\t"
          "ds_load_u16_d16_hi v235, %11 offset:1792\n\t"
          "ds_load_u16 v236, %11 offset:2048\n\t"
          "ds_load_u16_d16_hi v236, %11 offset:2304\n\t"
          "ds_load_u16 v237, %11 offset:2560\n\t"
          "ds_load_u16_d16_hi v237, %11 offset:2816\n\t"
          "ds_load_u16 v238, %11 offset:3072\n\t"
          "ds_load_u16_d16_hi v238, %11 offset:3328\n\t"
          "ds_load_u16 v239, %11 offset:3584\n\t"
          "ds_load_u16_d16_hi v239, %11 offset:3840\n\t"
          "ds_load_u16 v240, %11 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %11 offset:384\n\t"
          "ds_load_u16 v241, %11 offset:640\n\t"
          "ds_load_u16_d16_hi v241, %11 offset:896\n\t"
          "ds_load_u16 v242, %11 offset:1152\n\t"
          "ds_load_u16_d16_hi v242, %11 offset:1408\n\t"
          "ds_load_u16 v243, %11 offset:1664\n\t"
          "ds_load_u16_d16_hi v243, %11 offset:1920\n\t"
          "ds_load_u16 v244, %11 offset:2176\n\t"
          "ds_load_u16_d16_hi v244, %11 offset:2432\n\t"
          "ds_load_u16 v245, %11 offset:2688\n\t"
          "ds_load_u16_d16_hi v245, %11 offset:2944\n\t"
          "ds_load_u16 v246, %11 offset:3200\n\t"
          "ds_load_u16_d16_hi v246, %11 offset:3456\n\t"
          "ds_load_u16 v247, %11 offset:3712\n\t"
          "ds_load_u16_d16_hi v247, %11 offset:3968\n\t"
          "ds_load_b128 v[192:195], %10 offset:0\n\t"
          "ds_load_b128 v[196:199], %10 offset:16\n\t"
          "ds_load_b128 v[200:203], %10 offset:2304\n\t"
          "ds_load_b128 v[204:207], %10 offset:2320\n\t"
          "ds_load_b128 v[208:211], %10 offset:4608\n\t"
          "ds_load_b128 v[212:215], %10 offset:4624\n\t"
          "ds_load_b128 v[216:219], %10 offset:6912\n\t"
          "ds_load_b128 v[220:223], %10 offset:6928\n\t"
          "ds_load_b128 v[224:227], %10 offset:9216\n\t"
          "ds_load_b128 v[228:231], %10 offset:9232\n\t"
          "s_waitcnt lgkmcnt(8)\n\t"
          "v_wmma_f32_16x16x16_bf16 %0, v[192:199], v[232:239], %0\n\t"
          "v_wmma_f32_16x16x16_bf16 %1, v[192:199], v[240:247], %1\n\t"
          "s_waitcnt lgkmcnt(6)\n\t"
          "v_wmma_f32_16x16x16_bf16 %2, v[200:207], v[232:239], %2\n\t"
          "v_wmma_f32_16x16x16_bf16 %3, v[200:207], v[240:247], %3\n\t"
          "s_waitcnt lgkmcnt(4)\n\t"
          "v_wmma_f32_16x16x16_bf16 %4, v[208:215], v[232:239], %4\n\t"
          "v_wmma_f32_16x16x16_bf16 %5, v[208:215], v[240:247], %5\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_bf16 %6, v[216:223], v[232:239], %6\n\t"
          "v_wmma_f32_16x16x16_bf16 %7, v[216:223], v[240:247], %7\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_bf16 %8, v[224:231], v[232:239], %8\n\t"
          "v_wmma_f32_16x16x16_bf16 %9, v[224:231], v[240:247], %9\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[2][0]),
            "+v"(acc[2][1]),
            "+v"(acc[3][0]),
            "+v"(acc[3][1]),
            "+v"(acc[4][0]),
            "+v"(acc[4][1])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        }
      }
      else if constexpr(BM==128 && BN==80 && WT_M==2 && WT_N==5) {
        if constexpr(std::is_same<T,_Float16>::value) {
          asm volatile(
          "ds_load_u16 v208, %11 offset:0\n\t"
          "ds_load_u16_d16_hi v208, %11 offset:160\n\t"
          "ds_load_u16 v209, %11 offset:320\n\t"
          "ds_load_u16_d16_hi v209, %11 offset:480\n\t"
          "ds_load_u16 v210, %11 offset:640\n\t"
          "ds_load_u16_d16_hi v210, %11 offset:800\n\t"
          "ds_load_u16 v211, %11 offset:960\n\t"
          "ds_load_u16_d16_hi v211, %11 offset:1120\n\t"
          "ds_load_u16 v212, %11 offset:1280\n\t"
          "ds_load_u16_d16_hi v212, %11 offset:1440\n\t"
          "ds_load_u16 v213, %11 offset:1600\n\t"
          "ds_load_u16_d16_hi v213, %11 offset:1760\n\t"
          "ds_load_u16 v214, %11 offset:1920\n\t"
          "ds_load_u16_d16_hi v214, %11 offset:2080\n\t"
          "ds_load_u16 v215, %11 offset:2240\n\t"
          "ds_load_u16_d16_hi v215, %11 offset:2400\n\t"
          "ds_load_u16 v216, %11 offset:32\n\t"
          "ds_load_u16_d16_hi v216, %11 offset:192\n\t"
          "ds_load_u16 v217, %11 offset:352\n\t"
          "ds_load_u16_d16_hi v217, %11 offset:512\n\t"
          "ds_load_u16 v218, %11 offset:672\n\t"
          "ds_load_u16_d16_hi v218, %11 offset:832\n\t"
          "ds_load_u16 v219, %11 offset:992\n\t"
          "ds_load_u16_d16_hi v219, %11 offset:1152\n\t"
          "ds_load_u16 v220, %11 offset:1312\n\t"
          "ds_load_u16_d16_hi v220, %11 offset:1472\n\t"
          "ds_load_u16 v221, %11 offset:1632\n\t"
          "ds_load_u16_d16_hi v221, %11 offset:1792\n\t"
          "ds_load_u16 v222, %11 offset:1952\n\t"
          "ds_load_u16_d16_hi v222, %11 offset:2112\n\t"
          "ds_load_u16 v223, %11 offset:2272\n\t"
          "ds_load_u16_d16_hi v223, %11 offset:2432\n\t"
          "ds_load_u16 v224, %11 offset:64\n\t"
          "ds_load_u16_d16_hi v224, %11 offset:224\n\t"
          "ds_load_u16 v225, %11 offset:384\n\t"
          "ds_load_u16_d16_hi v225, %11 offset:544\n\t"
          "ds_load_u16 v226, %11 offset:704\n\t"
          "ds_load_u16_d16_hi v226, %11 offset:864\n\t"
          "ds_load_u16 v227, %11 offset:1024\n\t"
          "ds_load_u16_d16_hi v227, %11 offset:1184\n\t"
          "ds_load_u16 v228, %11 offset:1344\n\t"
          "ds_load_u16_d16_hi v228, %11 offset:1504\n\t"
          "ds_load_u16 v229, %11 offset:1664\n\t"
          "ds_load_u16_d16_hi v229, %11 offset:1824\n\t"
          "ds_load_u16 v230, %11 offset:1984\n\t"
          "ds_load_u16_d16_hi v230, %11 offset:2144\n\t"
          "ds_load_u16 v231, %11 offset:2304\n\t"
          "ds_load_u16_d16_hi v231, %11 offset:2464\n\t"
          "ds_load_u16 v232, %11 offset:96\n\t"
          "ds_load_u16_d16_hi v232, %11 offset:256\n\t"
          "ds_load_u16 v233, %11 offset:416\n\t"
          "ds_load_u16_d16_hi v233, %11 offset:576\n\t"
          "ds_load_u16 v234, %11 offset:736\n\t"
          "ds_load_u16_d16_hi v234, %11 offset:896\n\t"
          "ds_load_u16 v235, %11 offset:1056\n\t"
          "ds_load_u16_d16_hi v235, %11 offset:1216\n\t"
          "ds_load_u16 v236, %11 offset:1376\n\t"
          "ds_load_u16_d16_hi v236, %11 offset:1536\n\t"
          "ds_load_u16 v237, %11 offset:1696\n\t"
          "ds_load_u16_d16_hi v237, %11 offset:1856\n\t"
          "ds_load_u16 v238, %11 offset:2016\n\t"
          "ds_load_u16_d16_hi v238, %11 offset:2176\n\t"
          "ds_load_u16 v239, %11 offset:2336\n\t"
          "ds_load_u16_d16_hi v239, %11 offset:2496\n\t"
          "ds_load_u16 v240, %11 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %11 offset:288\n\t"
          "ds_load_u16 v241, %11 offset:448\n\t"
          "ds_load_u16_d16_hi v241, %11 offset:608\n\t"
          "ds_load_u16 v242, %11 offset:768\n\t"
          "ds_load_u16_d16_hi v242, %11 offset:928\n\t"
          "ds_load_u16 v243, %11 offset:1088\n\t"
          "ds_load_u16_d16_hi v243, %11 offset:1248\n\t"
          "ds_load_u16 v244, %11 offset:1408\n\t"
          "ds_load_u16_d16_hi v244, %11 offset:1568\n\t"
          "ds_load_u16 v245, %11 offset:1728\n\t"
          "ds_load_u16_d16_hi v245, %11 offset:1888\n\t"
          "ds_load_u16 v246, %11 offset:2048\n\t"
          "ds_load_u16_d16_hi v246, %11 offset:2208\n\t"
          "ds_load_u16 v247, %11 offset:2368\n\t"
          "ds_load_u16_d16_hi v247, %11 offset:2528\n\t"
          "ds_load_b128 v[192:195], %10 offset:0\n\t"
          "ds_load_b128 v[196:199], %10 offset:16\n\t"
          "ds_load_b128 v[200:203], %10 offset:9216\n\t"
          "ds_load_b128 v[204:207], %10 offset:9232\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[208:215], %0\n\t"
          "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[216:223], %1\n\t"
          "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[224:231], %2\n\t"
          "v_wmma_f32_16x16x16_f16 %3, v[192:199], v[232:239], %3\n\t"
          "v_wmma_f32_16x16x16_f16 %4, v[192:199], v[240:247], %4\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[208:215], %5\n\t"
          "v_wmma_f32_16x16x16_f16 %6, v[200:207], v[216:223], %6\n\t"
          "v_wmma_f32_16x16x16_f16 %7, v[200:207], v[224:231], %7\n\t"
          "v_wmma_f32_16x16x16_f16 %8, v[200:207], v[232:239], %8\n\t"
          "v_wmma_f32_16x16x16_f16 %9, v[200:207], v[240:247], %9\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[0][2]),
            "+v"(acc[0][3]),
            "+v"(acc[0][4]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[1][2]),
            "+v"(acc[1][3]),
            "+v"(acc[1][4])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        } else {
          asm volatile(
          "ds_load_u16 v208, %11 offset:0\n\t"
          "ds_load_u16_d16_hi v208, %11 offset:160\n\t"
          "ds_load_u16 v209, %11 offset:320\n\t"
          "ds_load_u16_d16_hi v209, %11 offset:480\n\t"
          "ds_load_u16 v210, %11 offset:640\n\t"
          "ds_load_u16_d16_hi v210, %11 offset:800\n\t"
          "ds_load_u16 v211, %11 offset:960\n\t"
          "ds_load_u16_d16_hi v211, %11 offset:1120\n\t"
          "ds_load_u16 v212, %11 offset:1280\n\t"
          "ds_load_u16_d16_hi v212, %11 offset:1440\n\t"
          "ds_load_u16 v213, %11 offset:1600\n\t"
          "ds_load_u16_d16_hi v213, %11 offset:1760\n\t"
          "ds_load_u16 v214, %11 offset:1920\n\t"
          "ds_load_u16_d16_hi v214, %11 offset:2080\n\t"
          "ds_load_u16 v215, %11 offset:2240\n\t"
          "ds_load_u16_d16_hi v215, %11 offset:2400\n\t"
          "ds_load_u16 v216, %11 offset:32\n\t"
          "ds_load_u16_d16_hi v216, %11 offset:192\n\t"
          "ds_load_u16 v217, %11 offset:352\n\t"
          "ds_load_u16_d16_hi v217, %11 offset:512\n\t"
          "ds_load_u16 v218, %11 offset:672\n\t"
          "ds_load_u16_d16_hi v218, %11 offset:832\n\t"
          "ds_load_u16 v219, %11 offset:992\n\t"
          "ds_load_u16_d16_hi v219, %11 offset:1152\n\t"
          "ds_load_u16 v220, %11 offset:1312\n\t"
          "ds_load_u16_d16_hi v220, %11 offset:1472\n\t"
          "ds_load_u16 v221, %11 offset:1632\n\t"
          "ds_load_u16_d16_hi v221, %11 offset:1792\n\t"
          "ds_load_u16 v222, %11 offset:1952\n\t"
          "ds_load_u16_d16_hi v222, %11 offset:2112\n\t"
          "ds_load_u16 v223, %11 offset:2272\n\t"
          "ds_load_u16_d16_hi v223, %11 offset:2432\n\t"
          "ds_load_u16 v224, %11 offset:64\n\t"
          "ds_load_u16_d16_hi v224, %11 offset:224\n\t"
          "ds_load_u16 v225, %11 offset:384\n\t"
          "ds_load_u16_d16_hi v225, %11 offset:544\n\t"
          "ds_load_u16 v226, %11 offset:704\n\t"
          "ds_load_u16_d16_hi v226, %11 offset:864\n\t"
          "ds_load_u16 v227, %11 offset:1024\n\t"
          "ds_load_u16_d16_hi v227, %11 offset:1184\n\t"
          "ds_load_u16 v228, %11 offset:1344\n\t"
          "ds_load_u16_d16_hi v228, %11 offset:1504\n\t"
          "ds_load_u16 v229, %11 offset:1664\n\t"
          "ds_load_u16_d16_hi v229, %11 offset:1824\n\t"
          "ds_load_u16 v230, %11 offset:1984\n\t"
          "ds_load_u16_d16_hi v230, %11 offset:2144\n\t"
          "ds_load_u16 v231, %11 offset:2304\n\t"
          "ds_load_u16_d16_hi v231, %11 offset:2464\n\t"
          "ds_load_u16 v232, %11 offset:96\n\t"
          "ds_load_u16_d16_hi v232, %11 offset:256\n\t"
          "ds_load_u16 v233, %11 offset:416\n\t"
          "ds_load_u16_d16_hi v233, %11 offset:576\n\t"
          "ds_load_u16 v234, %11 offset:736\n\t"
          "ds_load_u16_d16_hi v234, %11 offset:896\n\t"
          "ds_load_u16 v235, %11 offset:1056\n\t"
          "ds_load_u16_d16_hi v235, %11 offset:1216\n\t"
          "ds_load_u16 v236, %11 offset:1376\n\t"
          "ds_load_u16_d16_hi v236, %11 offset:1536\n\t"
          "ds_load_u16 v237, %11 offset:1696\n\t"
          "ds_load_u16_d16_hi v237, %11 offset:1856\n\t"
          "ds_load_u16 v238, %11 offset:2016\n\t"
          "ds_load_u16_d16_hi v238, %11 offset:2176\n\t"
          "ds_load_u16 v239, %11 offset:2336\n\t"
          "ds_load_u16_d16_hi v239, %11 offset:2496\n\t"
          "ds_load_u16 v240, %11 offset:128\n\t"
          "ds_load_u16_d16_hi v240, %11 offset:288\n\t"
          "ds_load_u16 v241, %11 offset:448\n\t"
          "ds_load_u16_d16_hi v241, %11 offset:608\n\t"
          "ds_load_u16 v242, %11 offset:768\n\t"
          "ds_load_u16_d16_hi v242, %11 offset:928\n\t"
          "ds_load_u16 v243, %11 offset:1088\n\t"
          "ds_load_u16_d16_hi v243, %11 offset:1248\n\t"
          "ds_load_u16 v244, %11 offset:1408\n\t"
          "ds_load_u16_d16_hi v244, %11 offset:1568\n\t"
          "ds_load_u16 v245, %11 offset:1728\n\t"
          "ds_load_u16_d16_hi v245, %11 offset:1888\n\t"
          "ds_load_u16 v246, %11 offset:2048\n\t"
          "ds_load_u16_d16_hi v246, %11 offset:2208\n\t"
          "ds_load_u16 v247, %11 offset:2368\n\t"
          "ds_load_u16_d16_hi v247, %11 offset:2528\n\t"
          "ds_load_b128 v[192:195], %10 offset:0\n\t"
          "ds_load_b128 v[196:199], %10 offset:16\n\t"
          "ds_load_b128 v[200:203], %10 offset:9216\n\t"
          "ds_load_b128 v[204:207], %10 offset:9232\n\t"
          "s_waitcnt lgkmcnt(2)\n\t"
          "v_wmma_f32_16x16x16_bf16 %0, v[192:199], v[208:215], %0\n\t"
          "v_wmma_f32_16x16x16_bf16 %1, v[192:199], v[216:223], %1\n\t"
          "v_wmma_f32_16x16x16_bf16 %2, v[192:199], v[224:231], %2\n\t"
          "v_wmma_f32_16x16x16_bf16 %3, v[192:199], v[232:239], %3\n\t"
          "v_wmma_f32_16x16x16_bf16 %4, v[192:199], v[240:247], %4\n\t"
          "s_waitcnt lgkmcnt(0)\n\t"
          "v_wmma_f32_16x16x16_bf16 %5, v[200:207], v[208:215], %5\n\t"
          "v_wmma_f32_16x16x16_bf16 %6, v[200:207], v[216:223], %6\n\t"
          "v_wmma_f32_16x16x16_bf16 %7, v[200:207], v[224:231], %7\n\t"
          "v_wmma_f32_16x16x16_bf16 %8, v[200:207], v[232:239], %8\n\t"
          "v_wmma_f32_16x16x16_bf16 %9, v[200:207], v[240:247], %9\n\t"
          : "+v"(acc[0][0]),
            "+v"(acc[0][1]),
            "+v"(acc[0][2]),
            "+v"(acc[0][3]),
            "+v"(acc[0][4]),
            "+v"(acc[1][0]),
            "+v"(acc[1][1]),
            "+v"(acc[1][2]),
            "+v"(acc[1][3]),
            "+v"(acc[1][4])
          : "v"(da), "v"(db) : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "memory");
        }
      }
    } else
#endif
    {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn=0;wn<WT_N;++wn) {
      const int nc=(wcol*WT_N+wn)*16+lane;
      if constexpr((RP & 64)!=0) {
#pragma unroll
        for(int e=0;e<HIPDNN_WMMA_FRAG_ELEMS/2;++e) {
          const int k=ks*16+k_off+2*e;
          unsigned short lo,hi;
          const T v0=smB[P][k*BN+(k/16)*8+nc];
          const T v1=smB[P][(k+1)*BN+((k+1)/16)*8+nc];
          __builtin_memcpy(&lo,&v0,2);__builtin_memcpy(&hi,&v1,2);
          reinterpret_cast<uint32_t*>(&b_frag[wn])[e]=(uint32_t)lo|((uint32_t)hi<<16);
        }
      } else {
#pragma unroll
        for(int e=0;e<HIPDNN_WMMA_FRAG_ELEMS;++e) {
          const int k=ks*16+k_off+e;
          const T val=smB[P][k*BN+(k/16)*8+nc];
          if constexpr(std::is_same<T,_Float16>::value)b_frag[wn][e]=val;
          else {unsigned short u;__builtin_memcpy(&u,&val,2);b_frag[wn][e]=u;}
        }
      }
    }
#pragma unroll
    for(int wm=0;wm<WT_M;++wm) {
      const int r=(wrow*WT_M+wm)*16+lane;
      Frag a_frag;
      if constexpr(AT) {
        const u32x4* src=reinterpret_cast<const u32x4*>(&smA[P][r][ks*16+k_off]);
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v)
          reinterpret_cast<u32x4*>(&a_frag)[v]=src[v];
      } else {
        const int gr=row0+r;
        const int vo=((gr<M?gr:M-1)-row0)*lda*ES+(ks*16+k_off)*ES;
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v) {
#if HIPDNN_WMMA_GFX12
          reinterpret_cast<u32x4*>(&a_frag)[v]=*reinterpret_cast<const u32x4*>(Ab+vo+v*16+ckb);
#else
          reinterpret_cast<u32x4*>(&a_frag)[v]=__builtin_amdgcn_raw_buffer_load_b128(ra0,vo+v*16,ckb,0);
#endif
        }
      }
#pragma unroll
      for(int wn=0;wn<WT_N;++wn) {
        if constexpr(std::is_same<T,_Float16>::value)
          acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_F16(a_frag,b_frag[wn],acc[wm][wn]);
        else acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_BF16(a_frag,b_frag[wn],acc[wm][wn]);
      }
      if constexpr((RP&128)!=0)GEMM_SCHED_FENCE();
    }
    }
    if constexpr((RP&256)!=0)GEMM_SCHED_FENCE();
  };
  auto schedStep = [&](bool) {};
  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr(ROLL) {
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
      if constexpr((RP & 32768)!=0 && ST && !IS) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks) {
#pragma unroll
        for(int i=0;i<NL;++i) {
          if(i*K_STEPS/NL==ks) {
            const int ii=((RP & 16384)!=0 && AT) ? (i<B_NL?i+A_ST:i-B_NL) : i;
            if constexpr(ST) {
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
              if constexpr((RP & 32768)!=0 && IS) asm volatile("s_waitcnt vmcnt(%0)" : : "n"(NL-1) : "memory");
#endif
              storeOne(SS{},PN{},ii);GEMM_SCHED_FENCE();
            }
            if constexpr(IS) {issueOne(SI{},ii);GEMM_SCHED_FENCE();}
          }
        }
        computeKs(PC{},ks);
      }
      if constexpr(IS)kb+=KB;
      if constexpr(ST) HIPDNN_LDS_BARRIER();
    } else {
      if constexpr(IS)issueAll(SI{});
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks)computeKs(PC{},ks);
      if constexpr(ST) {
        if constexpr(!DB) {HIPDNN_LDS_BARRIER();GEMM_SCHED_FENCE();}
        storeAll(SS{},PN{});HIPDNN_LDS_BARRIER();
      }
    }
    ckb += KB;
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{});
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  if constexpr((RP & 32768)!=0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
  storeAll(I0{}, I0{});
  HIPDNN_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  if constexpr(ROLL) {issueAll(I0{});GEMM_SCHED_FENCE();}

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + ((RP & 4096)!=0 ? wn*WN_L+wcol : wcol*WT_N+wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + ((RP & 4096)!=0 ? wm*WM_L+wrow : wrow*WT_M+wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = fmaf(al, acc[wm][wn][e], bc);
        const float v1 = fmaf(al, acc[wm][wn][e + 1], bc);
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (cc < N && rr < M)
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + ((RP & 4096)!=0 ? wn*WN_L+wcol : wcol*WT_N+wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + ((RP & 4096)!=0 ? wm*WM_L+wrow : wrow*WT_M+wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (cc < N && rr < M)
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf(fmaf(al, acc[wm][wn][e], bc));
      }
    }
  }
#endif
}

template<int BM,int BN,int WT_M,int WT_N,typename T,int BK,bool DB,int RP,bool AT>
__device__ __forceinline__ void GemmNnWideRelativeImpl(const T* A,const T* B,
 const T* C,T* Y,int M,int N,int K,float alpha,float beta,int cDim0,int cDim1,
 int swizzle_n,int ldb,int lda,int stag) {
 static_assert(std::is_same<T,_Float16>::value,"FP16 only");
 static_assert(BM==128&&BN==96&&WT_M==4&&WT_N==3&&BK==64&&DB&&AT,"isolated 128x96");
 constexpr int WM_L=2,WN_L=2,K_STR=72,ASZ=128*72,BSZ=64*96+32,LSLOT=16384;
 constexpr int ABO=BSZ+32,BBO=0;
 __shared__ __align__(16) T lds[LSLOT+ASZ+BSZ+32];
 const int tid=threadIdx.x,lid=tid%32,lane=lid%16,sub=lid/16;
 const int wid=__builtin_amdgcn_readfirstlane(tid/32),wrow=wid/2,wcol=wid%2;
 int by,bx;gemmSwizzleBlock(gridDim.x,gridDim.y,swizzle_n,by,bx);
 const int row0=by*128,col0=bx*96,lr=tid/8,lk=(tid%8)*8;
 int va[8],vb[6];
#pragma unroll
 for(int i=0;i<8;++i){const int r=row0+lr+i*16;va[i]=((r<M?r:M-1)-row0)*K*2+lk*2;}
#pragma unroll
 for(int j=0;j<6;++j){const int nc=col0+(tid%4)*8+(j%3)*32;vb[j]=((tid/4+(j/3)*32)*N+(nc+8<=N?nc:N-8)-col0)*2;}
 const uint64_t ap=reinterpret_cast<uint64_t>(A+static_cast<size_t>(row0)*K),bp=reinterpret_cast<uint64_t>(B+col0);
 const u32x4 ar={static_cast<uint32_t>(ap),static_cast<uint32_t>(ap>>32),0x7fffffffu,0x31004000u};
 const u32x4 br={static_cast<uint32_t>(bp),static_cast<uint32_t>(bp>>32),0x7fffffffu,0x31004000u};
 float8 acc[4][3];
#pragma unroll
 for(int i=0;i<4;++i)
#pragma unroll
 for(int j=0;j<3;++j)acc[i][j]=(float8)0.f;
 u32x4 pf[14];
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
 // Initial tile: B-first FIFO and constant LDS offsets, identical load map.
 const int kq=tid/4,ng=(tid%4)*8;
 uint32_t sb=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[BBO+kq*96+(kq/16)*8+ng]));
 uint32_t sa=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[ABO+lr*72+lk]));
 uint32_t db=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[BBO+(wcol*16+lane)]));
 uint32_t da=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&lds[ABO+(wrow*16+lane)*72]));
#pragma unroll
 for(int q=0;q<14;++q){u32x4 v;const int vo=q<6?vb[q]:va[q-6];const u32x4 rs=q<6?br:ar;
   asm volatile("buffer_load_b128 %0, %1, %2, 0 offen":"=&v"(v):"v"(vo),"s"(rs):"memory");pf[q]=v;}
 asm volatile("s_waitcnt vmcnt(0)":::"memory");
#pragma unroll
 for(int q=0;q<14;++q){
   const uint32_t ad=q<6?sb:sa;const u32x4 v=pf[q];
   if(q<6){const int off=((q%3)*32+(q/3)*(32*96+16))*2;
     asm volatile("ds_store_b128 %0, %1 offset:%2"::"v"(ad),"v"(v),"n"(off):"memory");}
   else {const int off=(q-6)*16*72*2;
     asm volatile("ds_store_b128 %0, %1 offset:%2"::"v"(ad),"v"(v),"n"(off):"memory");}}
 HIPDNN_LDS_BARRIER();GEMM_SCHED_FENCE();
#pragma unroll
 for(int q=0;q<14;++q){u32x4 v;const int vo=q<6?vb[q]:va[q-6];const u32x4 rs=q<6?br:ar;const int so=q<6?128*N:128;
   asm volatile("buffer_load_b128 %0, %1, %2, %3 offen":"=&v"(v):"v"(vo),"s"(rs),"s"(so):"memory");pf[q]=v;}
 sb^=0x8000;sa^=0x8000;
 int it=K/64-2,ak=256,bk=256*N;
asm volatile(
    ".Lnnloop_%=:\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_barrier\n\t"
    "ds_load_u16 v224, %29 offset:0\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:192\n\t"
    "ds_load_u16 v225, %29 offset:384\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:576\n\t"
    "ds_load_u16 v226, %29 offset:768\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:960\n\t"
    "ds_load_u16 v227, %29 offset:1152\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:1344\n\t"
    "ds_load_u16 v228, %29 offset:1536\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:1728\n\t"
    "ds_load_u16 v229, %29 offset:1920\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:2112\n\t"
    "ds_load_u16 v230, %29 offset:2304\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:2496\n\t"
    "ds_load_u16 v231, %29 offset:2688\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:2880\n\t"
    "ds_load_u16 v232, %29 offset:64\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:256\n\t"
    "ds_load_u16 v233, %29 offset:448\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:640\n\t"
    "ds_load_u16 v234, %29 offset:832\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:1024\n\t"
    "ds_load_u16 v235, %29 offset:1216\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:1408\n\t"
    "ds_load_u16 v236, %29 offset:1600\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:1792\n\t"
    "ds_load_u16 v237, %29 offset:1984\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:2176\n\t"
    "ds_load_u16 v238, %29 offset:2368\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:2560\n\t"
    "ds_load_u16 v239, %29 offset:2752\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:2944\n\t"
    "ds_load_u16 v240, %29 offset:128\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:320\n\t"
    "ds_load_u16 v241, %29 offset:512\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:704\n\t"
    "ds_load_u16 v242, %29 offset:896\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:1088\n\t"
    "ds_load_u16 v243, %29 offset:1280\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:1472\n\t"
    "ds_load_u16 v244, %29 offset:1664\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:1856\n\t"
    "ds_load_u16 v245, %29 offset:2048\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:2240\n\t"
    "ds_load_u16 v246, %29 offset:2432\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:2624\n\t"
    "ds_load_u16 v247, %29 offset:2816\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:3008\n\t"
    "ds_load_b128 v[192:195], %30 offset:0\n\t"
    "ds_load_b128 v[196:199], %30 offset:16\n\t"
    "ds_load_b128 v[200:203], %30 offset:4608\n\t"
    "ds_load_b128 v[204:207], %30 offset:4624\n\t"
    "ds_load_b128 v[208:211], %30 offset:9216\n\t"
    "ds_load_b128 v[212:215], %30 offset:9232\n\t"
    "ds_load_b128 v[216:219], %30 offset:13824\n\t"
    "ds_load_b128 v[220:223], %30 offset:13840\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %12 offset:0\n\t"
    "buffer_load_b128 %12, %36, %33, %32 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %13 offset:64\n\t"
    "buffer_load_b128 %13, %37, %33, %32 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %14 offset:128\n\t"
    "buffer_load_b128 %14, %38, %33, %32 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %15 offset:6176\n\t"
    "buffer_load_b128 %15, %39, %33, %32 offen\n\t"
    "s_waitcnt lgkmcnt(4)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:3088\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:3280\n\t"
    "ds_load_u16 v225, %29 offset:3472\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:3664\n\t"
    "ds_load_u16 v226, %29 offset:3856\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:4048\n\t"
    "ds_load_u16 v227, %29 offset:4240\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:4432\n\t"
    "ds_load_u16 v228, %29 offset:4624\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:4816\n\t"
    "ds_load_u16 v229, %29 offset:5008\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:5200\n\t"
    "ds_load_u16 v230, %29 offset:5392\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:5584\n\t"
    "ds_load_u16 v231, %29 offset:5776\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:5968\n\t"
    "ds_load_u16 v232, %29 offset:3152\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:3344\n\t"
    "ds_load_u16 v233, %29 offset:3536\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:3728\n\t"
    "ds_load_u16 v234, %29 offset:3920\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:4112\n\t"
    "ds_load_u16 v235, %29 offset:4304\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:4496\n\t"
    "ds_load_u16 v236, %29 offset:4688\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:4880\n\t"
    "ds_load_u16 v237, %29 offset:5072\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:5264\n\t"
    "ds_load_u16 v238, %29 offset:5456\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:5648\n\t"
    "ds_load_u16 v239, %29 offset:5840\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:6032\n\t"
    "ds_load_u16 v240, %29 offset:3216\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:3408\n\t"
    "ds_load_u16 v241, %29 offset:3600\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:3792\n\t"
    "ds_load_u16 v242, %29 offset:3984\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:4176\n\t"
    "ds_load_u16 v243, %29 offset:4368\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:4560\n\t"
    "ds_load_u16 v244, %29 offset:4752\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:4944\n\t"
    "ds_load_u16 v245, %29 offset:5136\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:5328\n\t"
    "ds_load_u16 v246, %29 offset:5520\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:5712\n\t"
    "ds_load_u16 v247, %29 offset:5904\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:6096\n\t"
    "ds_load_b128 v[192:195], %30 offset:32\n\t"
    "ds_load_b128 v[196:199], %30 offset:48\n\t"
    "ds_load_b128 v[200:203], %30 offset:4640\n\t"
    "ds_load_b128 v[204:207], %30 offset:4656\n\t"
    "ds_load_b128 v[208:211], %30 offset:9248\n\t"
    "ds_load_b128 v[212:215], %30 offset:9264\n\t"
    "ds_load_b128 v[216:219], %30 offset:13856\n\t"
    "ds_load_b128 v[220:223], %30 offset:13872\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %16 offset:6240\n\t"
    "buffer_load_b128 %16, %40, %33, %32 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %27, %17 offset:6304\n\t"
    "buffer_load_b128 %17, %41, %33, %32 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %18 offset:0\n\t"
    "buffer_load_b128 %18, %42, %34, %31 offen\n\t"
    "s_waitcnt lgkmcnt(3)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:6176\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:6368\n\t"
    "ds_load_u16 v225, %29 offset:6560\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:6752\n\t"
    "ds_load_u16 v226, %29 offset:6944\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:7136\n\t"
    "ds_load_u16 v227, %29 offset:7328\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:7520\n\t"
    "ds_load_u16 v228, %29 offset:7712\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:7904\n\t"
    "ds_load_u16 v229, %29 offset:8096\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:8288\n\t"
    "ds_load_u16 v230, %29 offset:8480\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:8672\n\t"
    "ds_load_u16 v231, %29 offset:8864\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:9056\n\t"
    "ds_load_u16 v232, %29 offset:6240\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:6432\n\t"
    "ds_load_u16 v233, %29 offset:6624\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:6816\n\t"
    "ds_load_u16 v234, %29 offset:7008\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:7200\n\t"
    "ds_load_u16 v235, %29 offset:7392\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:7584\n\t"
    "ds_load_u16 v236, %29 offset:7776\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:7968\n\t"
    "ds_load_u16 v237, %29 offset:8160\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:8352\n\t"
    "ds_load_u16 v238, %29 offset:8544\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:8736\n\t"
    "ds_load_u16 v239, %29 offset:8928\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:9120\n\t"
    "ds_load_u16 v240, %29 offset:6304\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:6496\n\t"
    "ds_load_u16 v241, %29 offset:6688\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:6880\n\t"
    "ds_load_u16 v242, %29 offset:7072\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:7264\n\t"
    "ds_load_u16 v243, %29 offset:7456\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:7648\n\t"
    "ds_load_u16 v244, %29 offset:7840\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:8032\n\t"
    "ds_load_u16 v245, %29 offset:8224\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:8416\n\t"
    "ds_load_u16 v246, %29 offset:8608\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:8800\n\t"
    "ds_load_u16 v247, %29 offset:8992\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:9184\n\t"
    "ds_load_b128 v[192:195], %30 offset:64\n\t"
    "ds_load_b128 v[196:199], %30 offset:80\n\t"
    "ds_load_b128 v[200:203], %30 offset:4672\n\t"
    "ds_load_b128 v[204:207], %30 offset:4688\n\t"
    "ds_load_b128 v[208:211], %30 offset:9280\n\t"
    "ds_load_b128 v[212:215], %30 offset:9296\n\t"
    "ds_load_b128 v[216:219], %30 offset:13888\n\t"
    "ds_load_b128 v[220:223], %30 offset:13904\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %19 offset:2304\n\t"
    "buffer_load_b128 %19, %43, %34, %31 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %20 offset:4608\n\t"
    "buffer_load_b128 %20, %44, %34, %31 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %21 offset:6912\n\t"
    "buffer_load_b128 %21, %45, %34, %31 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %22 offset:9216\n\t"
    "buffer_load_b128 %22, %46, %34, %31 offen\n\t"
    "s_waitcnt lgkmcnt(4)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:9264\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:9456\n\t"
    "ds_load_u16 v225, %29 offset:9648\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:9840\n\t"
    "ds_load_u16 v226, %29 offset:10032\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:10224\n\t"
    "ds_load_u16 v227, %29 offset:10416\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:10608\n\t"
    "ds_load_u16 v228, %29 offset:10800\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:10992\n\t"
    "ds_load_u16 v229, %29 offset:11184\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:11376\n\t"
    "ds_load_u16 v230, %29 offset:11568\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:11760\n\t"
    "ds_load_u16 v231, %29 offset:11952\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:12144\n\t"
    "ds_load_u16 v232, %29 offset:9328\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:9520\n\t"
    "ds_load_u16 v233, %29 offset:9712\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:9904\n\t"
    "ds_load_u16 v234, %29 offset:10096\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:10288\n\t"
    "ds_load_u16 v235, %29 offset:10480\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:10672\n\t"
    "ds_load_u16 v236, %29 offset:10864\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:11056\n\t"
    "ds_load_u16 v237, %29 offset:11248\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:11440\n\t"
    "ds_load_u16 v238, %29 offset:11632\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:11824\n\t"
    "ds_load_u16 v239, %29 offset:12016\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:12208\n\t"
    "ds_load_u16 v240, %29 offset:9392\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:9584\n\t"
    "ds_load_u16 v241, %29 offset:9776\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:9968\n\t"
    "ds_load_u16 v242, %29 offset:10160\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:10352\n\t"
    "ds_load_u16 v243, %29 offset:10544\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:10736\n\t"
    "ds_load_u16 v244, %29 offset:10928\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:11120\n\t"
    "ds_load_u16 v245, %29 offset:11312\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:11504\n\t"
    "ds_load_u16 v246, %29 offset:11696\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:11888\n\t"
    "ds_load_u16 v247, %29 offset:12080\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:12272\n\t"
    "ds_load_b128 v[192:195], %30 offset:96\n\t"
    "ds_load_b128 v[196:199], %30 offset:112\n\t"
    "ds_load_b128 v[200:203], %30 offset:4704\n\t"
    "ds_load_b128 v[204:207], %30 offset:4720\n\t"
    "ds_load_b128 v[208:211], %30 offset:9312\n\t"
    "ds_load_b128 v[212:215], %30 offset:9328\n\t"
    "ds_load_b128 v[216:219], %30 offset:13920\n\t"
    "ds_load_b128 v[220:223], %30 offset:13936\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %23 offset:11520\n\t"
    "buffer_load_b128 %23, %47, %34, %31 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %24 offset:13824\n\t"
    "buffer_load_b128 %24, %48, %34, %31 offen\n\t"
    "s_waitcnt vmcnt(13)\n\t"
    "ds_store_b128 %28, %25 offset:16128\n\t"
    "buffer_load_b128 %25, %49, %34, %31 offen\n\t"
    "v_xor_b32_e32 %27, 0x8000, %27\n\t"
    "v_xor_b32_e32 %28, 0x8000, %28\n\t"
    "v_xor_b32_e32 %29, 0x8000, %29\n\t"
    "v_xor_b32_e32 %30, 0x8000, %30\n\t"
    "s_waitcnt lgkmcnt(3)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "s_add_i32 %31, %31, 128\n\t"
    "s_add_i32 %32, %32, %50\n\t"
    "s_sub_u32 %26, %26, 1\n\t"
    "s_cmp_gt_i32 %26, 0\n\t"
    "s_cbranch_scc1 .Lnnloop_%=\n\t"
    "s_waitcnt vmcnt(0)\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_barrier\n\t"
    "ds_load_u16 v224, %29 offset:0\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:192\n\t"
    "ds_load_u16 v225, %29 offset:384\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:576\n\t"
    "ds_load_u16 v226, %29 offset:768\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:960\n\t"
    "ds_load_u16 v227, %29 offset:1152\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:1344\n\t"
    "ds_load_u16 v228, %29 offset:1536\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:1728\n\t"
    "ds_load_u16 v229, %29 offset:1920\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:2112\n\t"
    "ds_load_u16 v230, %29 offset:2304\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:2496\n\t"
    "ds_load_u16 v231, %29 offset:2688\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:2880\n\t"
    "ds_load_u16 v232, %29 offset:64\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:256\n\t"
    "ds_load_u16 v233, %29 offset:448\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:640\n\t"
    "ds_load_u16 v234, %29 offset:832\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:1024\n\t"
    "ds_load_u16 v235, %29 offset:1216\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:1408\n\t"
    "ds_load_u16 v236, %29 offset:1600\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:1792\n\t"
    "ds_load_u16 v237, %29 offset:1984\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:2176\n\t"
    "ds_load_u16 v238, %29 offset:2368\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:2560\n\t"
    "ds_load_u16 v239, %29 offset:2752\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:2944\n\t"
    "ds_load_u16 v240, %29 offset:128\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:320\n\t"
    "ds_load_u16 v241, %29 offset:512\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:704\n\t"
    "ds_load_u16 v242, %29 offset:896\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:1088\n\t"
    "ds_load_u16 v243, %29 offset:1280\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:1472\n\t"
    "ds_load_u16 v244, %29 offset:1664\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:1856\n\t"
    "ds_load_u16 v245, %29 offset:2048\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:2240\n\t"
    "ds_load_u16 v246, %29 offset:2432\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:2624\n\t"
    "ds_load_u16 v247, %29 offset:2816\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:3008\n\t"
    "ds_load_b128 v[192:195], %30 offset:0\n\t"
    "ds_load_b128 v[196:199], %30 offset:16\n\t"
    "ds_load_b128 v[200:203], %30 offset:4608\n\t"
    "ds_load_b128 v[204:207], %30 offset:4624\n\t"
    "ds_load_b128 v[208:211], %30 offset:9216\n\t"
    "ds_load_b128 v[212:215], %30 offset:9232\n\t"
    "ds_load_b128 v[216:219], %30 offset:13824\n\t"
    "ds_load_b128 v[220:223], %30 offset:13840\n\t"
    "ds_store_b128 %27, %12 offset:0\n\t"
    "ds_store_b128 %27, %13 offset:64\n\t"
    "ds_store_b128 %27, %14 offset:128\n\t"
    "ds_store_b128 %27, %15 offset:6176\n\t"
    "s_waitcnt lgkmcnt(4)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:3088\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:3280\n\t"
    "ds_load_u16 v225, %29 offset:3472\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:3664\n\t"
    "ds_load_u16 v226, %29 offset:3856\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:4048\n\t"
    "ds_load_u16 v227, %29 offset:4240\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:4432\n\t"
    "ds_load_u16 v228, %29 offset:4624\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:4816\n\t"
    "ds_load_u16 v229, %29 offset:5008\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:5200\n\t"
    "ds_load_u16 v230, %29 offset:5392\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:5584\n\t"
    "ds_load_u16 v231, %29 offset:5776\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:5968\n\t"
    "ds_load_u16 v232, %29 offset:3152\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:3344\n\t"
    "ds_load_u16 v233, %29 offset:3536\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:3728\n\t"
    "ds_load_u16 v234, %29 offset:3920\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:4112\n\t"
    "ds_load_u16 v235, %29 offset:4304\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:4496\n\t"
    "ds_load_u16 v236, %29 offset:4688\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:4880\n\t"
    "ds_load_u16 v237, %29 offset:5072\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:5264\n\t"
    "ds_load_u16 v238, %29 offset:5456\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:5648\n\t"
    "ds_load_u16 v239, %29 offset:5840\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:6032\n\t"
    "ds_load_u16 v240, %29 offset:3216\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:3408\n\t"
    "ds_load_u16 v241, %29 offset:3600\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:3792\n\t"
    "ds_load_u16 v242, %29 offset:3984\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:4176\n\t"
    "ds_load_u16 v243, %29 offset:4368\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:4560\n\t"
    "ds_load_u16 v244, %29 offset:4752\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:4944\n\t"
    "ds_load_u16 v245, %29 offset:5136\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:5328\n\t"
    "ds_load_u16 v246, %29 offset:5520\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:5712\n\t"
    "ds_load_u16 v247, %29 offset:5904\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:6096\n\t"
    "ds_load_b128 v[192:195], %30 offset:32\n\t"
    "ds_load_b128 v[196:199], %30 offset:48\n\t"
    "ds_load_b128 v[200:203], %30 offset:4640\n\t"
    "ds_load_b128 v[204:207], %30 offset:4656\n\t"
    "ds_load_b128 v[208:211], %30 offset:9248\n\t"
    "ds_load_b128 v[212:215], %30 offset:9264\n\t"
    "ds_load_b128 v[216:219], %30 offset:13856\n\t"
    "ds_load_b128 v[220:223], %30 offset:13872\n\t"
    "ds_store_b128 %27, %16 offset:6240\n\t"
    "ds_store_b128 %27, %17 offset:6304\n\t"
    "ds_store_b128 %28, %18 offset:0\n\t"
    "s_waitcnt lgkmcnt(3)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:6176\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:6368\n\t"
    "ds_load_u16 v225, %29 offset:6560\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:6752\n\t"
    "ds_load_u16 v226, %29 offset:6944\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:7136\n\t"
    "ds_load_u16 v227, %29 offset:7328\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:7520\n\t"
    "ds_load_u16 v228, %29 offset:7712\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:7904\n\t"
    "ds_load_u16 v229, %29 offset:8096\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:8288\n\t"
    "ds_load_u16 v230, %29 offset:8480\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:8672\n\t"
    "ds_load_u16 v231, %29 offset:8864\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:9056\n\t"
    "ds_load_u16 v232, %29 offset:6240\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:6432\n\t"
    "ds_load_u16 v233, %29 offset:6624\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:6816\n\t"
    "ds_load_u16 v234, %29 offset:7008\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:7200\n\t"
    "ds_load_u16 v235, %29 offset:7392\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:7584\n\t"
    "ds_load_u16 v236, %29 offset:7776\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:7968\n\t"
    "ds_load_u16 v237, %29 offset:8160\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:8352\n\t"
    "ds_load_u16 v238, %29 offset:8544\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:8736\n\t"
    "ds_load_u16 v239, %29 offset:8928\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:9120\n\t"
    "ds_load_u16 v240, %29 offset:6304\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:6496\n\t"
    "ds_load_u16 v241, %29 offset:6688\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:6880\n\t"
    "ds_load_u16 v242, %29 offset:7072\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:7264\n\t"
    "ds_load_u16 v243, %29 offset:7456\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:7648\n\t"
    "ds_load_u16 v244, %29 offset:7840\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:8032\n\t"
    "ds_load_u16 v245, %29 offset:8224\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:8416\n\t"
    "ds_load_u16 v246, %29 offset:8608\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:8800\n\t"
    "ds_load_u16 v247, %29 offset:8992\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:9184\n\t"
    "ds_load_b128 v[192:195], %30 offset:64\n\t"
    "ds_load_b128 v[196:199], %30 offset:80\n\t"
    "ds_load_b128 v[200:203], %30 offset:4672\n\t"
    "ds_load_b128 v[204:207], %30 offset:4688\n\t"
    "ds_load_b128 v[208:211], %30 offset:9280\n\t"
    "ds_load_b128 v[212:215], %30 offset:9296\n\t"
    "ds_load_b128 v[216:219], %30 offset:13888\n\t"
    "ds_load_b128 v[220:223], %30 offset:13904\n\t"
    "ds_store_b128 %28, %19 offset:2304\n\t"
    "ds_store_b128 %28, %20 offset:4608\n\t"
    "ds_store_b128 %28, %21 offset:6912\n\t"
    "ds_store_b128 %28, %22 offset:9216\n\t"
    "s_waitcnt lgkmcnt(4)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:9264\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:9456\n\t"
    "ds_load_u16 v225, %29 offset:9648\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:9840\n\t"
    "ds_load_u16 v226, %29 offset:10032\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:10224\n\t"
    "ds_load_u16 v227, %29 offset:10416\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:10608\n\t"
    "ds_load_u16 v228, %29 offset:10800\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:10992\n\t"
    "ds_load_u16 v229, %29 offset:11184\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:11376\n\t"
    "ds_load_u16 v230, %29 offset:11568\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:11760\n\t"
    "ds_load_u16 v231, %29 offset:11952\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:12144\n\t"
    "ds_load_u16 v232, %29 offset:9328\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:9520\n\t"
    "ds_load_u16 v233, %29 offset:9712\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:9904\n\t"
    "ds_load_u16 v234, %29 offset:10096\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:10288\n\t"
    "ds_load_u16 v235, %29 offset:10480\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:10672\n\t"
    "ds_load_u16 v236, %29 offset:10864\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:11056\n\t"
    "ds_load_u16 v237, %29 offset:11248\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:11440\n\t"
    "ds_load_u16 v238, %29 offset:11632\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:11824\n\t"
    "ds_load_u16 v239, %29 offset:12016\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:12208\n\t"
    "ds_load_u16 v240, %29 offset:9392\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:9584\n\t"
    "ds_load_u16 v241, %29 offset:9776\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:9968\n\t"
    "ds_load_u16 v242, %29 offset:10160\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:10352\n\t"
    "ds_load_u16 v243, %29 offset:10544\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:10736\n\t"
    "ds_load_u16 v244, %29 offset:10928\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:11120\n\t"
    "ds_load_u16 v245, %29 offset:11312\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:11504\n\t"
    "ds_load_u16 v246, %29 offset:11696\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:11888\n\t"
    "ds_load_u16 v247, %29 offset:12080\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:12272\n\t"
    "ds_load_b128 v[192:195], %30 offset:96\n\t"
    "ds_load_b128 v[196:199], %30 offset:112\n\t"
    "ds_load_b128 v[200:203], %30 offset:4704\n\t"
    "ds_load_b128 v[204:207], %30 offset:4720\n\t"
    "ds_load_b128 v[208:211], %30 offset:9312\n\t"
    "ds_load_b128 v[212:215], %30 offset:9328\n\t"
    "ds_load_b128 v[216:219], %30 offset:13920\n\t"
    "ds_load_b128 v[220:223], %30 offset:13936\n\t"
    "ds_store_b128 %28, %23 offset:11520\n\t"
    "ds_store_b128 %28, %24 offset:13824\n\t"
    "ds_store_b128 %28, %25 offset:16128\n\t"
    "v_xor_b32_e32 %27, 0x8000, %27\n\t"
    "v_xor_b32_e32 %28, 0x8000, %28\n\t"
    "v_xor_b32_e32 %29, 0x8000, %29\n\t"
    "v_xor_b32_e32 %30, 0x8000, %30\n\t"
    "s_waitcnt lgkmcnt(3)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_barrier\n\t"
    "ds_load_u16 v224, %29 offset:0\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:192\n\t"
    "ds_load_u16 v225, %29 offset:384\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:576\n\t"
    "ds_load_u16 v226, %29 offset:768\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:960\n\t"
    "ds_load_u16 v227, %29 offset:1152\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:1344\n\t"
    "ds_load_u16 v228, %29 offset:1536\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:1728\n\t"
    "ds_load_u16 v229, %29 offset:1920\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:2112\n\t"
    "ds_load_u16 v230, %29 offset:2304\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:2496\n\t"
    "ds_load_u16 v231, %29 offset:2688\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:2880\n\t"
    "ds_load_u16 v232, %29 offset:64\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:256\n\t"
    "ds_load_u16 v233, %29 offset:448\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:640\n\t"
    "ds_load_u16 v234, %29 offset:832\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:1024\n\t"
    "ds_load_u16 v235, %29 offset:1216\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:1408\n\t"
    "ds_load_u16 v236, %29 offset:1600\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:1792\n\t"
    "ds_load_u16 v237, %29 offset:1984\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:2176\n\t"
    "ds_load_u16 v238, %29 offset:2368\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:2560\n\t"
    "ds_load_u16 v239, %29 offset:2752\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:2944\n\t"
    "ds_load_u16 v240, %29 offset:128\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:320\n\t"
    "ds_load_u16 v241, %29 offset:512\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:704\n\t"
    "ds_load_u16 v242, %29 offset:896\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:1088\n\t"
    "ds_load_u16 v243, %29 offset:1280\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:1472\n\t"
    "ds_load_u16 v244, %29 offset:1664\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:1856\n\t"
    "ds_load_u16 v245, %29 offset:2048\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:2240\n\t"
    "ds_load_u16 v246, %29 offset:2432\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:2624\n\t"
    "ds_load_u16 v247, %29 offset:2816\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:3008\n\t"
    "ds_load_b128 v[192:195], %30 offset:0\n\t"
    "ds_load_b128 v[196:199], %30 offset:16\n\t"
    "ds_load_b128 v[200:203], %30 offset:4608\n\t"
    "ds_load_b128 v[204:207], %30 offset:4624\n\t"
    "ds_load_b128 v[208:211], %30 offset:9216\n\t"
    "ds_load_b128 v[212:215], %30 offset:9232\n\t"
    "ds_load_b128 v[216:219], %30 offset:13824\n\t"
    "ds_load_b128 v[220:223], %30 offset:13840\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:3088\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:3280\n\t"
    "ds_load_u16 v225, %29 offset:3472\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:3664\n\t"
    "ds_load_u16 v226, %29 offset:3856\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:4048\n\t"
    "ds_load_u16 v227, %29 offset:4240\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:4432\n\t"
    "ds_load_u16 v228, %29 offset:4624\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:4816\n\t"
    "ds_load_u16 v229, %29 offset:5008\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:5200\n\t"
    "ds_load_u16 v230, %29 offset:5392\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:5584\n\t"
    "ds_load_u16 v231, %29 offset:5776\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:5968\n\t"
    "ds_load_u16 v232, %29 offset:3152\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:3344\n\t"
    "ds_load_u16 v233, %29 offset:3536\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:3728\n\t"
    "ds_load_u16 v234, %29 offset:3920\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:4112\n\t"
    "ds_load_u16 v235, %29 offset:4304\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:4496\n\t"
    "ds_load_u16 v236, %29 offset:4688\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:4880\n\t"
    "ds_load_u16 v237, %29 offset:5072\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:5264\n\t"
    "ds_load_u16 v238, %29 offset:5456\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:5648\n\t"
    "ds_load_u16 v239, %29 offset:5840\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:6032\n\t"
    "ds_load_u16 v240, %29 offset:3216\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:3408\n\t"
    "ds_load_u16 v241, %29 offset:3600\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:3792\n\t"
    "ds_load_u16 v242, %29 offset:3984\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:4176\n\t"
    "ds_load_u16 v243, %29 offset:4368\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:4560\n\t"
    "ds_load_u16 v244, %29 offset:4752\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:4944\n\t"
    "ds_load_u16 v245, %29 offset:5136\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:5328\n\t"
    "ds_load_u16 v246, %29 offset:5520\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:5712\n\t"
    "ds_load_u16 v247, %29 offset:5904\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:6096\n\t"
    "ds_load_b128 v[192:195], %30 offset:32\n\t"
    "ds_load_b128 v[196:199], %30 offset:48\n\t"
    "ds_load_b128 v[200:203], %30 offset:4640\n\t"
    "ds_load_b128 v[204:207], %30 offset:4656\n\t"
    "ds_load_b128 v[208:211], %30 offset:9248\n\t"
    "ds_load_b128 v[212:215], %30 offset:9264\n\t"
    "ds_load_b128 v[216:219], %30 offset:13856\n\t"
    "ds_load_b128 v[220:223], %30 offset:13872\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:6176\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:6368\n\t"
    "ds_load_u16 v225, %29 offset:6560\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:6752\n\t"
    "ds_load_u16 v226, %29 offset:6944\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:7136\n\t"
    "ds_load_u16 v227, %29 offset:7328\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:7520\n\t"
    "ds_load_u16 v228, %29 offset:7712\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:7904\n\t"
    "ds_load_u16 v229, %29 offset:8096\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:8288\n\t"
    "ds_load_u16 v230, %29 offset:8480\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:8672\n\t"
    "ds_load_u16 v231, %29 offset:8864\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:9056\n\t"
    "ds_load_u16 v232, %29 offset:6240\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:6432\n\t"
    "ds_load_u16 v233, %29 offset:6624\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:6816\n\t"
    "ds_load_u16 v234, %29 offset:7008\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:7200\n\t"
    "ds_load_u16 v235, %29 offset:7392\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:7584\n\t"
    "ds_load_u16 v236, %29 offset:7776\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:7968\n\t"
    "ds_load_u16 v237, %29 offset:8160\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:8352\n\t"
    "ds_load_u16 v238, %29 offset:8544\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:8736\n\t"
    "ds_load_u16 v239, %29 offset:8928\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:9120\n\t"
    "ds_load_u16 v240, %29 offset:6304\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:6496\n\t"
    "ds_load_u16 v241, %29 offset:6688\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:6880\n\t"
    "ds_load_u16 v242, %29 offset:7072\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:7264\n\t"
    "ds_load_u16 v243, %29 offset:7456\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:7648\n\t"
    "ds_load_u16 v244, %29 offset:7840\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:8032\n\t"
    "ds_load_u16 v245, %29 offset:8224\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:8416\n\t"
    "ds_load_u16 v246, %29 offset:8608\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:8800\n\t"
    "ds_load_u16 v247, %29 offset:8992\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:9184\n\t"
    "ds_load_b128 v[192:195], %30 offset:64\n\t"
    "ds_load_b128 v[196:199], %30 offset:80\n\t"
    "ds_load_b128 v[200:203], %30 offset:4672\n\t"
    "ds_load_b128 v[204:207], %30 offset:4688\n\t"
    "ds_load_b128 v[208:211], %30 offset:9280\n\t"
    "ds_load_b128 v[212:215], %30 offset:9296\n\t"
    "ds_load_b128 v[216:219], %30 offset:13888\n\t"
    "ds_load_b128 v[220:223], %30 offset:13904\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
    "ds_load_u16 v224, %29 offset:9264\n\t"
    "ds_load_u16_d16_hi v224, %29 offset:9456\n\t"
    "ds_load_u16 v225, %29 offset:9648\n\t"
    "ds_load_u16_d16_hi v225, %29 offset:9840\n\t"
    "ds_load_u16 v226, %29 offset:10032\n\t"
    "ds_load_u16_d16_hi v226, %29 offset:10224\n\t"
    "ds_load_u16 v227, %29 offset:10416\n\t"
    "ds_load_u16_d16_hi v227, %29 offset:10608\n\t"
    "ds_load_u16 v228, %29 offset:10800\n\t"
    "ds_load_u16_d16_hi v228, %29 offset:10992\n\t"
    "ds_load_u16 v229, %29 offset:11184\n\t"
    "ds_load_u16_d16_hi v229, %29 offset:11376\n\t"
    "ds_load_u16 v230, %29 offset:11568\n\t"
    "ds_load_u16_d16_hi v230, %29 offset:11760\n\t"
    "ds_load_u16 v231, %29 offset:11952\n\t"
    "ds_load_u16_d16_hi v231, %29 offset:12144\n\t"
    "ds_load_u16 v232, %29 offset:9328\n\t"
    "ds_load_u16_d16_hi v232, %29 offset:9520\n\t"
    "ds_load_u16 v233, %29 offset:9712\n\t"
    "ds_load_u16_d16_hi v233, %29 offset:9904\n\t"
    "ds_load_u16 v234, %29 offset:10096\n\t"
    "ds_load_u16_d16_hi v234, %29 offset:10288\n\t"
    "ds_load_u16 v235, %29 offset:10480\n\t"
    "ds_load_u16_d16_hi v235, %29 offset:10672\n\t"
    "ds_load_u16 v236, %29 offset:10864\n\t"
    "ds_load_u16_d16_hi v236, %29 offset:11056\n\t"
    "ds_load_u16 v237, %29 offset:11248\n\t"
    "ds_load_u16_d16_hi v237, %29 offset:11440\n\t"
    "ds_load_u16 v238, %29 offset:11632\n\t"
    "ds_load_u16_d16_hi v238, %29 offset:11824\n\t"
    "ds_load_u16 v239, %29 offset:12016\n\t"
    "ds_load_u16_d16_hi v239, %29 offset:12208\n\t"
    "ds_load_u16 v240, %29 offset:9392\n\t"
    "ds_load_u16_d16_hi v240, %29 offset:9584\n\t"
    "ds_load_u16 v241, %29 offset:9776\n\t"
    "ds_load_u16_d16_hi v241, %29 offset:9968\n\t"
    "ds_load_u16 v242, %29 offset:10160\n\t"
    "ds_load_u16_d16_hi v242, %29 offset:10352\n\t"
    "ds_load_u16 v243, %29 offset:10544\n\t"
    "ds_load_u16_d16_hi v243, %29 offset:10736\n\t"
    "ds_load_u16 v244, %29 offset:10928\n\t"
    "ds_load_u16_d16_hi v244, %29 offset:11120\n\t"
    "ds_load_u16 v245, %29 offset:11312\n\t"
    "ds_load_u16_d16_hi v245, %29 offset:11504\n\t"
    "ds_load_u16 v246, %29 offset:11696\n\t"
    "ds_load_u16_d16_hi v246, %29 offset:11888\n\t"
    "ds_load_u16 v247, %29 offset:12080\n\t"
    "ds_load_u16_d16_hi v247, %29 offset:12272\n\t"
    "ds_load_b128 v[192:195], %30 offset:96\n\t"
    "ds_load_b128 v[196:199], %30 offset:112\n\t"
    "ds_load_b128 v[200:203], %30 offset:4704\n\t"
    "ds_load_b128 v[204:207], %30 offset:4720\n\t"
    "ds_load_b128 v[208:211], %30 offset:9312\n\t"
    "ds_load_b128 v[212:215], %30 offset:9328\n\t"
    "ds_load_b128 v[216:219], %30 offset:13920\n\t"
    "ds_load_b128 v[220:223], %30 offset:13936\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_nop 1\n\t"
    "v_wmma_f32_16x16x16_f16 %0, v[192:199], v[224:231], %0\n\t"
    "v_wmma_f32_16x16x16_f16 %1, v[192:199], v[232:239], %1\n\t"
    "v_wmma_f32_16x16x16_f16 %2, v[192:199], v[240:247], %2\n\t"
    "v_wmma_f32_16x16x16_f16 %3, v[200:207], v[224:231], %3\n\t"
    "v_wmma_f32_16x16x16_f16 %4, v[200:207], v[232:239], %4\n\t"
    "v_wmma_f32_16x16x16_f16 %5, v[200:207], v[240:247], %5\n\t"
    "v_wmma_f32_16x16x16_f16 %6, v[208:215], v[224:231], %6\n\t"
    "v_wmma_f32_16x16x16_f16 %7, v[208:215], v[232:239], %7\n\t"
    "v_wmma_f32_16x16x16_f16 %8, v[208:215], v[240:247], %8\n\t"
    "v_wmma_f32_16x16x16_f16 %9, v[216:223], v[224:231], %9\n\t"
    "v_wmma_f32_16x16x16_f16 %10, v[216:223], v[232:239], %10\n\t"
    "v_wmma_f32_16x16x16_f16 %11, v[216:223], v[240:247], %11\n\t"
 : "+v"(acc[0][0]),
   "+v"(acc[0][1]),
   "+v"(acc[0][2]),
   "+v"(acc[1][0]),
   "+v"(acc[1][1]),
   "+v"(acc[1][2]),
   "+v"(acc[2][0]),
   "+v"(acc[2][1]),
   "+v"(acc[2][2]),
   "+v"(acc[3][0]),
   "+v"(acc[3][1]),
   "+v"(acc[3][2]),
   "+&v"(pf[0]),
   "+&v"(pf[1]),
   "+&v"(pf[2]),
   "+&v"(pf[3]),
   "+&v"(pf[4]),
   "+&v"(pf[5]),
   "+&v"(pf[6]),
   "+&v"(pf[7]),
   "+&v"(pf[8]),
   "+&v"(pf[9]),
   "+&v"(pf[10]),
   "+&v"(pf[11]),
   "+&v"(pf[12]),
   "+&v"(pf[13]),
   "+&s"(it),
   "+&v"(sb),
   "+&v"(sa),
   "+&v"(db),
   "+&v"(da),
   "+&s"(ak),
   "+&s"(bk)
 : "s"(br),
   "s"(ar),
   "s"(0),
   "v"(vb[0]),
   "v"(vb[1]),
   "v"(vb[2]),
   "v"(vb[3]),
   "v"(vb[4]),
   "v"(vb[5]),
   "v"(va[0]),
   "v"(va[1]),
   "v"(va[2]),
   "v"(va[3]),
   "v"(va[4]),
   "v"(va[5]),
   "v"(va[6]),
   "v"(va[7]),
   "s"(128*N)
 : "v192", "v193", "v194", "v195", "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244", "v245", "v246", "v247", "scc", "memory");
#else
 GemmNnWidePortable<128,96,T>(A,B,Y,M,N,K);
 return;
#endif
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wn*WN_L+wcol) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm*WM_L+wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = fmaf(al, acc[wm][wn][e], bc);
        const float v1 = fmaf(al, acc[wm][wn][e + 1], bc);
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (cc < N && rr < M)
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wn*WN_L+wcol) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm*WM_L+wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (cc < N && rr < M)
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf(fmaf(al, acc[wm][wn][e], bc));
      }
    }
  }
#endif
}

/* NN weights resident in the 32 MiB Infinity Cache. Independent instances,
 * outside kWmma: DU64, four waves, natural B LDS, store-before-refill reads.
 * The two routes are limited to measured FP16, Y=A*B shapes.
 * The80x128 route uses slc reads: retained throughput gain under both tested
 * block lengths and product preheat; see autotune/gemm/README.md. */
template <int Route>
__global__ void __launch_bounds__(128)
gemm_nnwide_kernel(const _Float16 *A, const _Float16 *B, _Float16 *Y,
                   int M, int N, int K, int sw) {
  if constexpr (Route == 0)
    GemmNnWideRelativeImpl<128,96,4,3,_Float16,64,true,4,true>(
        A,B,nullptr,Y,M,N,K,1.f,0.f,1,1,sw,N,K,0);
  else {
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
    GemmNnWideManualImpl<80,128,5,2,_Float16,64,true,37154,true>(
        A,B,nullptr,Y,M,N,K,1.f,0.f,1,1,sw,N,K,0);
#else
    GemmNnWidePortable<80,128,_Float16>(A,B,Y,M,N,K);
#endif
  }
}

static bool gemmNnWideGfx1151() {
  static std::atomic<int> cached[16];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16)
    return false;
  int answer = cached[dev].load(std::memory_order_relaxed);
  if (answer == 0) {
    hipDeviceProp_t props;
    answer = hipGetDeviceProperties(&props, dev) == hipSuccess &&
                     std::strncmp(props.gcnArchName, "gfx1151", 7) == 0 &&
                     (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':')
                 ? 1 : -1;
    cached[dev].store(answer, std::memory_order_relaxed);
  }
  return answer == 1;
}

static bool launchNnWide(hipStream_t s, const _Float16 *A, const _Float16 *B,
                         _Float16 *Y, int M, int N, int K, float alpha,
                         float beta) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE", 1) != 0;
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG", -1);
  if (!on || alpha != 1.f || beta != 0.f ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) & 15) != 0)
    return false;
  int id = M == 128 && N == 3072 && K == 3072 ? 0 :
           (M == 80 && N == 4096 && K == 4096 ? 1 : -1);
  if (id < 0 || (forced >= 0 && forced != id) || !gemmNnWideGfx1151())
    return false;
  (void)hipGetLastError();
  const int bn = id == 0 ? 96 : 128;
  const dim3 grid(N / bn, 1);
  if (id == 0)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_nnwide_kernel<0>), grid,
                       dim3(128), 0, s, A,B,Y,M,N,K,8);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_nnwide_kernel<1>), grid,
                       dim3(128), 0, s, A,B,Y,M,N,K,8);
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma-nnwide M=%d N=%d K=%d -> wide-cfg[%d] bn=%d bk=64 "
      "waves=4 sw=8 stag=0 wgs=%d\n", M,N,K,id,bn,static_cast<int>(grid.x));
  return true;
}

/* Streaming NN weights at the measured 8192-square stride. Natural B LDS
 * rows, BK32, two LDS buffers and one rolling register slot: store tile t+1
 * before reusing its register group to issue t+2. A loads directly into WMMA
 * fragments. Independent templates preserve all earlier device instances. */
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, bool AT = true>
__device__ __forceinline__ void
GemmNnStreamImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int BCGR = BN / 8;
  constexpr int B_NL = (BK * BCGR + THR - 1) / THR;
  constexpr int NL = (AT ? A_NCH : 0) + B_NL;
  constexpr int A_ST = AT ? A_NCH : 0;
  constexpr int B_SIZE = BK * BN + (BK / 16) * 8;
  constexpr bool ROLL = (RP & 32) != 0;
  static_assert(!ROLL || DB, "rolling needs double LDS");
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = !ROLL && MODE == 2 ? 2 : 1;
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * ((AT ? BM * K_STR : 0) + B_SIZE) * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][AT ? A_NCH * RPI : 1][AT ? K_STR : 1];
  __shared__ __align__(16) T smB[NBUF][B_SIZE];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  const int jw = niter;
  constexpr int BS = BK;
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NL], bd[B_NL];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NL; ++c) {
    const int q = c * THR + tid;
    const int cg = q % BCGR, kq = q / BCGR;
    const int nc = col0 + cg * 8;
    const int nv = nc + 8 <= N ? nc : N - 8;
    voffB[c] = (kq * N + nv - col0) * ES;
    bd[c] = kq * BN + (kq / 16) * 8 + cg * 8;
  }
  const char *Ab = reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb = reinterpret_cast<const char *>(B + col0);
  const int kspan = K * ES;
#if !HIPDNN_WMMA_GFX12
  const auto ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab), 0, 0x7fffffff, 0x31004000);
  const auto rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int ckb = 0;
  int kb = 0; /* byte offset of the next tile to issue */

  auto issueOne = [&](auto slot, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if (i < A_ST) {
#if HIPDNN_WMMA_GFX12
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Ab + voffA[i] + kb);
#else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, voffA[i], kb, 0);
#endif
    } else {
      const int c = i - A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
#if HIPDNN_WMMA_GFX12
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Bb + voffB[c] + kb * N);
#else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, voffB[c], kb * N, (RP & 64) ? 2 : 0);
#endif
    }
  };
  auto issueAll = [&](auto slot) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i) issueOne(slot,i);
    kb += KB;
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value, P = decltype(buf)::value;
    if (i < A_ST)
      *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][lk]) = pf[S][i];
    else {
      const int c=i-A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
      *reinterpret_cast<u32x4 *>(&smB[P][bd[c]])=pf[S][i];
    }
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i)storeOne(slot,buf,i);
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn=0;wn<WT_N;++wn) {
      const int nc=(wcol*WT_N+wn)*16+lane;
#pragma unroll
      for(int e=0;e<HIPDNN_WMMA_FRAG_ELEMS;++e) {
        const int k=ks*16+k_off+e;
        const T val=smB[P][k*BN+(k/16)*8+nc];
        if constexpr(std::is_same<T,_Float16>::value)b_frag[wn][e]=val;
        else { unsigned short u; __builtin_memcpy(&u,&val,2); b_frag[wn][e]=u; }
      }
    }
#pragma unroll
    for(int wm=0;wm<WT_M;++wm) {
      const int r=(wrow*WT_M+wm)*16+lane;
      Frag a_frag;
      if constexpr(AT) {
        const u32x4* src=reinterpret_cast<const u32x4*>(&smA[P][r][ks*16+k_off]);
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v)
          reinterpret_cast<u32x4*>(&a_frag)[v]=src[v];
      } else {
        const int gr=row0+r;
        const int vo=((gr<M?gr:M-1)-row0)*lda*ES+(ks*16+k_off)*ES;
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v) {
#if HIPDNN_WMMA_GFX12
          reinterpret_cast<u32x4*>(&a_frag)[v]=*reinterpret_cast<const u32x4*>(Ab+vo+v*16+ckb);
#else
          reinterpret_cast<u32x4*>(&a_frag)[v]=__builtin_amdgcn_raw_buffer_load_b128(ra0,vo+v*16,ckb,0);
#endif
        }
      }
#pragma unroll
      for(int wn=0;wn<WT_N;++wn) {
        if constexpr(std::is_same<T,_Float16>::value)
          acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_F16(a_frag,b_frag[wn],acc[wm][wn]);
        else acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_BF16(a_frag,b_frag[wn],acc[wm][wn]);
      }
    }
  };
  auto schedStep = [&](bool) {};
  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr(ROLL) {
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks) {
#pragma unroll
        for(int i=0;i<NL;++i) {
          if(i*K_STEPS/NL==ks) {
            if constexpr(ST) {storeOne(SS{},PN{},i);GEMM_SCHED_FENCE();}
            if constexpr(IS) {issueOne(SI{},i);GEMM_SCHED_FENCE();}
          }
        }
        computeKs(PC{},ks);
      }
      if constexpr(IS)kb+=KB;
      if constexpr(ST) HIPDNN_LDS_BARRIER();
    } else {
      if constexpr(IS)issueAll(SI{});
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks)computeKs(PC{},ks);
      if constexpr(ST) {
        if constexpr(!DB) {HIPDNN_LDS_BARRIER();GEMM_SCHED_FENCE();}
        storeAll(SS{},PN{});HIPDNN_LDS_BARRIER();
      }
    }
    ckb += KB;
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{});
  storeAll(I0{}, I0{});
  HIPDNN_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  if constexpr(ROLL) {issueAll(I0{});GEMM_SCHED_FENCE();}

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

#pragma unroll
  for(int i=0;i<WT_M;++i)
#pragma unroll
    for(int j=0;j<WT_N;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        int r=row0+(wrow*WT_M+i)*16+hipdnn_wmma_acc_row(sub,e);
        int c=col0+(wcol*WT_N+j)*16+lane;
        if(r<M && c<N)Y[static_cast<size_t>(r)*N+c]=GemmElem<T>::fromf(acc[i][j][e]);
      }
}

template <typename T, int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__((BM / (WM * 16)) * (BN / (WN * 16)) * 32)
gemm_nnstream_kernel(const T *A, const T *B, T *Y,
                      int M, int N, int K, int sw) {
  GemmNnStreamImpl<BM,BN,WM,WN,T,32,true,98,false>(
      A,B,nullptr,Y,M,N,K,1.f,0.f,1,1,sw,N,K,0);
}

static bool gemmNnStreamGfx1151() {
  static std::atomic<int> cached[16];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16)
    return false;
  int answer = cached[dev].load(std::memory_order_relaxed);
  if (answer == 0) {
    hipDeviceProp_t props;
    answer = hipGetDeviceProperties(&props, dev) == hipSuccess &&
                     std::strncmp(props.gcnArchName, "gfx1151", 7) == 0 &&
                     (props.gcnArchName[7] == '\0' || props.gcnArchName[7] == ':')
                 ? 1 : -1;
    cached[dev].store(answer, std::memory_order_relaxed);
  }
  return answer == 1;
}

template <typename T>
static bool launchNnStream(hipStream_t s, const T *A, const T *B, T *Y,
                           int M, int N, int K, float alpha, float beta) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM", 1) != 0;
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG", -1);
  constexpr bool f16 = std::is_same<T, _Float16>::value;
  constexpr int id = f16 ? 0 : 1;
  if (!on || N != 8192 || K != 8192 || M != (f16 ? 64 : 48) ||
      alpha != 1.f || beta != 0.f ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) & 15) != 0 ||
      (forced >= 0 && forced != id) || !gemmNnStreamGfx1151())
    return false;
  (void)hipGetLastError();
  if constexpr (f16)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_nnstream_kernel<T,64,128,1,8>),
                       dim3(64), dim3(128), 0, s, A,B,Y,M,N,K,8);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_nnstream_kernel<T,48,64,3,1>),
                       dim3(128), dim3(128), 0, s, A,B,Y,M,N,K,8);
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma-nnstream M=%d N=%d K=%d -> stream-cfg[%d] bn=%d bk=32 "
      "waves=4 sw=8 stag=0 wgs=%d\n", M,N,K,id,f16?128:64,f16?64:128);
  return true;
}


/* Exact NN2048 specialization: B stays in padded [k][n] LDS, while A
 * uses padded [m][k] LDS. Compiler-visible barriers preserve read overlap.
 * Measured routes and original device payloads are documented in autotune/gemm.
 */
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define NN_LARGE_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while (0)
#else
#define NN_LARGE_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, bool AT = true>
__device__ __forceinline__ void
GemmNnLargeNaturalImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int BCGR = BN / 8;
  constexpr int B_NL = (BK * BCGR + THR - 1) / THR;
  constexpr int NL = (AT ? A_NCH : 0) + B_NL;
  constexpr int A_ST = AT ? A_NCH : 0;
  constexpr int B_SIZE = BK * BN + (BK / 16) * 8;
  constexpr bool ROLL = (RP & 32) != 0;
  static_assert(!ROLL || DB, "rolling needs double LDS");
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = !ROLL && MODE == 2 ? 2 : 1;
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * ((AT ? BM * K_STR : 0) + B_SIZE) * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][AT ? A_NCH * RPI : 1][AT ? K_STR : 1];
  __shared__ __align__(16) T smB[NBUF][B_SIZE];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  const int jw = niter;
  constexpr int BS = BK;
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NL], bd[B_NL];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NL; ++c) {
    const int q = c * THR + tid;
    const int cg = q % BCGR, kq = q / BCGR;
    const int nc = col0 + cg * 8;
    const int nv = nc + 8 <= N ? nc : N - 8;
    voffB[c] = (kq * N + nv - col0) * ES;
    bd[c] = kq * BN + (kq / 16) * 8 + cg * 8;
  }
  const char *Ab = reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb = reinterpret_cast<const char *>(B + col0);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const auto ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab), 0, 0x7fffffff, 0x31004000);
  const auto rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int ckb = 0;
  int kb = 0; /* byte offset of the next tile to issue */

  auto issueOne = [&](auto slot, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if (i < A_ST) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Ab + voffA[i] + kb);
#else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, voffA[i], kb, 0);
#endif
    } else {
      const int c = i - A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
      pf[S][i] = *reinterpret_cast<const u32x4 *>(Bb + voffB[c] + kb * N);
#else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, voffB[c], kb * N, (RP & 64) ? 2 : 0);
#endif
    }
  };
  auto issueAll = [&](auto slot) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i) issueOne(slot,i);
    kb += KB;
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value, P = decltype(buf)::value;
    if (i < A_ST)
      *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][lk]) = pf[S][i];
    else {
      const int c=i-A_ST;
      if constexpr ((BK * BCGR) % THR != 0) { if(c * THR + tid >= BK * BCGR) return; }
      *reinterpret_cast<u32x4 *>(&smB[P][bd[c]])=pf[S][i];
    }
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
#pragma unroll
    for(int i=0;i<NL;++i)storeOne(slot,buf,i);
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn=0;wn<WT_N;++wn) {
      const int nc=(wcol*WT_N+wn)*16+lane;
#pragma unroll
      for(int e=0;e<HIPDNN_WMMA_FRAG_ELEMS;++e) {
        const int k=ks*16+k_off+e;
        const T val=smB[P][k*BN+(k/16)*8+nc];
        if constexpr(std::is_same<T,_Float16>::value)b_frag[wn][e]=val;
        else { unsigned short u; __builtin_memcpy(&u,&val,2); b_frag[wn][e]=u; }
      }
    }
#pragma unroll
    for(int wm=0;wm<WT_M;++wm) {
      const int r=(wrow*WT_M+wm)*16+lane;
      Frag a_frag;
      if constexpr(AT) {
        const u32x4* src=reinterpret_cast<const u32x4*>(&smA[P][r][ks*16+k_off]);
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v)
          reinterpret_cast<u32x4*>(&a_frag)[v]=src[v];
      } else {
        const int gr=row0+r;
        const int vo=((gr<M?gr:M-1)-row0)*lda*ES+(ks*16+k_off)*ES;
#pragma unroll
        for(int v=0;v<HIPDNN_WMMA_FRAG_ELEMS/8;++v) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
          reinterpret_cast<u32x4*>(&a_frag)[v]=*reinterpret_cast<const u32x4*>(Ab+vo+v*16+ckb);
#else
          reinterpret_cast<u32x4*>(&a_frag)[v]=__builtin_amdgcn_raw_buffer_load_b128(ra0,vo+v*16,ckb,0);
#endif
        }
      }
#pragma unroll
      for(int wn=0;wn<WT_N;++wn) {
        if constexpr(std::is_same<T,_Float16>::value)
          acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_F16(a_frag,b_frag[wn],acc[wm][wn]);
        else acc[wm][wn]=HIPDNN_WMMA_F32_16X16X16_BF16(a_frag,b_frag[wn],acc[wm][wn]);
      }
    }
  };
  auto schedStep = [&](bool) {};
  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr(ROLL) {
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks) {
#pragma unroll
        for(int i=0;i<NL;++i) {
          if(i*K_STEPS/NL==ks) {
            if constexpr(ST) {storeOne(SS{},PN{},i);GEMM_SCHED_FENCE();}
            if constexpr(IS) {issueOne(SI{},i);GEMM_SCHED_FENCE();}
          }
        }
        computeKs(PC{},ks);
      }
      if constexpr(IS)kb+=KB;
      if constexpr(ST) NN_LARGE_LDS_BARRIER();
    } else {
      if constexpr(IS)issueAll(SI{});
#pragma unroll
      for(int ks=0;ks<K_STEPS;++ks)computeKs(PC{},ks);
      if constexpr(ST) {
        if constexpr(!DB) {NN_LARGE_LDS_BARRIER();GEMM_SCHED_FENCE();}
        storeAll(SS{},PN{});NN_LARGE_LDS_BARRIER();
      }
    }
    ckb += KB;
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{});
  storeAll(I0{}, I0{});
  NN_LARGE_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  if constexpr(ROLL) {issueAll(I0{});GEMM_SCHED_FENCE();}

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

#pragma unroll
  for(int i=0;i<WT_M;++i)
#pragma unroll
    for(int j=0;j<WT_N;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        int r=row0+(wrow*WT_M+i)*16+hipdnn_wmma_acc_row(sub,e);
        int c=col0+(wcol*WT_N+j)*16+lane;
        if(r<M && c<N)Y[static_cast<size_t>(r)*N+c]=GemmElem<T>::fromf(acc[i][j][e]);
      }
}


template <typename T, int BM, int BN, int WM, int WN, int BK, bool DB, int RP, bool AT, int SW>
__global__ void __launch_bounds__((BM/(WM*16))*(BN/(WN*16))*32)
gemm_wmma_nnlarge_kernel(const T *A,const T *B,T *Y,int M,int N,int K) {
  GemmNnLargeNaturalImpl<BM,BN,WM,WN,T,BK,DB,RP,AT>(A,B,nullptr,Y,M,N,K,1.f,0.f,1,1,SW,N,K,0);
}

/* Five-process generic/O2 screening, old/new/HB ms (64 launches):
 * f16 2.0104/1.7999/1.8597; exact NN2048 only.
 * bf16 2.0727/1.7899/1.9018; exact NN2048 only.
 */
template <typename T>
static bool launchMeasuredNnLarge(hipStream_t s, const T *A, const T *B, T *Y,
                                  int M,int N,int K,int ta,int tb,float alpha,float beta) {
  if constexpr (HIPDNN_WMMA_FRAG_ELEMS == 16) {
    if(M!=2048 || N!=4096 || K!=4096 || ta || tb || alpha!=1.f || beta!=0.f ||
       ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15)!=0)
      return false;
    static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE",1)!=0;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1);
    static const bool pinned=gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
       gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
       gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
       gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0;
    if(!on || (forced>=0 && forced!=0) || pinned || !gemmNnStreamGfx1151()) return false;
    (void)hipGetLastError();
    if constexpr(std::is_same<T,_Float16>::value) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<T,96,128,6,2,32,false,2,true,32>),
                         dim3((N+128-1)/128,(M+96-1)/96),dim3(128),0,s,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] wmma-nnlarge M=%d N=%d K=%d -> nnlarge-cfg[0] bm=96 bn=128 wt=6x2 bk=32 db=0 rp=2 at=1 sw=32 wgs=704\n",M,N,K);
    }
    else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<T,96,128,6,2,32,false,2,true,32>),
                         dim3((N+128-1)/128,(M+96-1)/96),dim3(128),0,s,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] wmma-nnlarge M=%d N=%d K=%d -> nnlarge-cfg[0] bm=96 bn=128 wt=6x2 bk=32 db=0 rp=2 at=1 sw=32 wgs=704\n",M,N,K);
    }
    return true;
  }
  return false;
}
#undef NN_LARGE_LDS_BARRIER
/* Tile table: occupancy vs arithmetic intensity on gfx1151 (~40 CUs).
 * BK=64 raises arithmetic intensity when (BM+BN)*(BK+8)*4 fits in 64 KB LDS. */
static constexpr WmmaCfg kWmma[] = {
    {64, 64, 2, 2, 8, 1, 32},    {64, 128, 2, 2, 8, 1, 32},
    {128, 64, 2, 2, 8, 1, 32},   {128, 128, 2, 2, 8, 1, 32},
    {128, 256, 2, 2, 8, 1, 32},  {256, 128, 4, 2, 8, 1, 32},
    {128, 128, 2, 4, 8, 1, 32},  {64, 32, 2, 1, 8, 1, 32},
    {128, 32, 2, 1, 8, 1, 32},   {64, 256, 2, 2, 8, 1, 32},
    {64, 128, 2, 4, 8, 1, 32},   {128, 256, 2, 4, 8, 1, 32},
    {128, 128, 2, 2, 4, 1, 32},  {128, 128, 2, 2, 16, 1, 32},
    {128, 256, 2, 2, 16, 1, 32}, {64, 128, 2, 4, 8, 1, 64},
    {64, 128, 2, 2, 8, 1, 64},   {128, 64, 2, 2, 8, 1, 64},
    {64, 64, 2, 2, 8, 1, 64},
    /* High Matrix-Core:LDS ratio tiles (matmul_nbits): WT4x4/WT4x2 BK=32
     * issue 8-16 WMMA per fragment-load batch. Best for compute-bound
     * large/square GEMM once the grid fills gfx1151's CUs. */
    {128, 128, 4, 4, 8, 1, 32},  {128, 128, 4, 2, 8, 1, 32},
    {128, 256, 4, 2, 8, 1, 32},  {256, 128, 4, 4, 8, 1, 32},
    {128, 128, 4, 4, 16, 1, 32},
    /* BK=16 tiles for K%16==0 but K%32!=0 (e.g. vision K=4304): K is a multiple
     * of BK=16, so the whole launch runs the aligned BC=false loop with no
     * ragged K-tail. Costs 2x barriers vs BK=32, so used only when BK=32 would
     * be ragged. */
    {256, 128, 4, 2, 8, 1, 16},  {128, 128, 2, 4, 8, 1, 16},
    {64, 128, 2, 4, 8, 1, 16},
    /* BK=64 large tiles: a K-tile consumes each B/A row's whole 128 B cache
     * line, so an unpadded K%1024==0 row stride cannot evict the line's second
     * half before the next K-tile reads it. Single-buffered (LDS). */
    {128, 128, 2, 4, 8, 1, 64},  {128, 128, 2, 2, 8, 1, 64},
    {128, 128, 4, 2, 8, 1, 64},  {256, 128, 4, 2, 8, 1, 64},
    {128, 256, 4, 2, 8, 1, 64},  {128, 64, 4, 2, 8, 1, 64},
    /* 4-wave 128x64 (hipBLASLt's MT64x128 pick for 1024x4096x4096). */
    {128, 64, 4, 2, 8, 1, 32},
};
static constexpr int kNumWmma =
    static_cast<int>(sizeof(kWmma) / sizeof(kWmma[0]));

/* BK=32 tiles instantiated with PGR2 (GEMM_WMMA_TRY_PGR in launchWmma). */
static bool wmmaHasPgr2(const WmmaCfg &c) {
  if (c.bk != 32)
    return false;
  return (c.bm == 128 && c.bn == 128 && c.wt_m == 2 && c.wt_n == 4) ||
         (c.bm == 256 && c.bn == 128 && c.wt_m == 4 && c.wt_n == 2) ||
         (c.bm == 128 && c.bn == 128 && c.wt_m == 4 && c.wt_n == 2) ||
         (c.bm == 128 && c.bn == 256 && c.wt_m == 4 && c.wt_n == 2);
}

static bool wmmaApplies(const WmmaCfg &c, int M, int N, int K) {
  if (c.bn < 64 && N > 2 * c.bn)
    return false;
  return true;
}

static int findWmmaCfg(int bm, int bn, int wtm, int wtn, int sw, int sk,
                       int bk = 32) {
  for (int i = 0; i < kNumWmma; ++i) {
    const WmmaCfg &c = kWmma[i];
    if (c.bm == bm && c.bn == bn && c.wt_m == wtm && c.wt_n == wtn &&
        c.swizzle_n == sw && c.split_k == sk && c.bk == bk)
      return i;
  }
  return 3;
}

/* Geometry <-> config-index bridge for the WMMA LUT (see gemm_lut). A geometry
 * the live table no longer contains, or that is illegal for this shape,
 * resolves to -1 so the LUT search moves on rather than return an unlaunchable
 * config. */
struct GemmWmmaLutCtx {
  int M, N, K;
};
static int wmmaConfigFromGeom(const gemm_lut::WmmaAnswer &g, int M, int N,
                              int K) {
  for (int i = 0; i < kNumWmma; ++i) {
    const WmmaCfg &c = kWmma[i];
    if (c.bm == g.bm && c.bn == g.bn && c.wt_m == g.wt_m && c.wt_n == g.wt_n &&
        c.swizzle_n == g.swizzle_n && c.split_k == g.split_k && c.bk == g.bk) {
      return wmmaApplies(c, M, N, K) ? i : -1;
    }
  }
  return -1;
}
static bool wmmaLutValidator(void *ctx, const gemm_lut::WmmaAnswer &g) {
  const auto *c = static_cast<const GemmWmmaLutCtx *>(ctx);
  return wmmaConfigFromGeom(g, c->M, c->N, c->K) >= 0;
}
template <typename T>
static int gemmLutWmmaConfig(int M, int N, int K, int transB) {
  if (!gemmLutLookupEnabled())
    return -1; /* online mode: skip the LUT and sweep */
  gemm_lut::Request req;
  req.kind = gemm_lut::Kind::Wmma;
  req.act_dtype = req.wts_dtype = req.out_dtype = gemmDType<T>();
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  GemmWmmaLutCtx ctx{M, N, K};
  const gemm_lut::Result r =
      gemm_lut::resolve(req, wmmaLutValidator, nullptr, nullptr, &ctx);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return wmmaConfigFromGeom(r.wmma, M, N, K);
}

static int pickWmmaHeuristic(int M, int N, int K) {
  auto wgs = [&](int bm, int bn, int sk) {
    return ((M + bm - 1) / bm) * ((N + bn - 1) / bn) * sk;
  };
  /* K%32!=0 but K%16==0 (e.g. vision K=4304): a BK=32 tile has a ragged K-tail
   * that forces the whole launch onto the slow bounds-checked path. A BK=16
   * tile makes K%BK==0 so the launch stays on the fast BC=false loop -- measured
   * ~+11% over the BK=32 bounds-checked path at 256x128. Costs 2x barriers, so
   * used only for these K%32!=0 shapes. */
  if ((K % 32 != 0) && (K % 16 == 0)) {
    if (M >= 256 && wgs(256, 128, 1) >= 24)
      return findWmmaCfg(256, 128, 4, 2, 8, 1, 16);
    if (wgs(128, 128, 1) >= 16)
      return findWmmaCfg(128, 128, 2, 4, 8, 1, 16);
    return findWmmaCfg(64, 128, 2, 4, 8, 1, 16);
  }
  if (N <= 32)
    return findWmmaCfg(64, 32, 2, 1, 8, 1);
  if (N <= 64)
    return findWmmaCfg(64, 64, 2, 2, 8, 1);
  /* Power-of-two K >= 8192: the NT row stride (K*sizeof(T) bytes) is a multiple
   * of the 16 KB L1, so a wide 256x128 tile's tile-rows all map to the same few
   * L1 sets and evict each other -- throughput collapses (~24k vs ~37k at the
   * non-aliasing K=4096). Two regimes, split by the M-tile count:
   *
   *   - Large M (>= 1024, i.e. >= 8 row-tiles): DRAM traffic is dominated by
   *     A re-reads (A is read N/BN times). The narrow BN=64 tile therefore
   *     re-streams A ~N/64 times and hits a bandwidth wall -- measured 8192^2
   *     M=4096 at 128x64 = 24.3 ms (-41% vs hipBLAS) and M=2048 = 12.0 ms
   *     (-45%). A wide 128x256 WT4x2 tile cuts A traffic 4x (N/256 reads) and
   *     wins outright: M=4096 -> 13.7 ms (+4% over hipBLAS), M=2048 -> 7.1 ms
   *     (parity), M=1024 -> 4.07 ms (was 4.46). The L1 aliasing on this tile
   *     is handled at launch by the K%1024 alias rule in launchWmma.
   *   - Small M (257..512): too few row-tiles to amortise the wide tile; the
   *     128x64 BK=64 tile (half the aliasing tile-rows on both operands, half
   *     the K-step/barrier count) is the workhorse and beats the wide tile
   *     (M=512 8192^2: 2.17 vs 2.29 ms). Small M<=256 falls through below.
   *
   * Non-power-of-two large K (14336/17408) does not alias, so the big square
   * tiles in the block below still win. */
  if (K >= 8192 && (K & (K - 1)) == 0) {
    if (M >= 1024) {
      /* Same BN choice as the A-panel rule below: wide BN=256 (cfg32) only when
       * there are enough N-tiles (N>=4096) or the shape is very K-heavy (K>=3N);
       * a small/mid N (e.g. 2816/3840 at K=8192) has too few 256-wide columns and
       * the 128x128 tile wins (2816x8192 cfg12 1.37 vs cfg32 2.0). The 8192^2
       * squares (N>=4096) keep cfg32 + the sw=4 raster fix. */
      if (N >= 4096 || static_cast<size_t>(K) >= static_cast<size_t>(N) * 3)
        return findWmmaCfg(128, 256, 4, 2, 8, 1, 32);
      return findWmmaCfg(128, 128, 2, 2, 4, 1, 32);
    }
    if (M > 256)
      return findWmmaCfg(128, 64, 2, 2, 8, 1, 64);
  }
  /* Wide-N tile when the A-panel (M*K) is large: A is read N/BN times, so the
   * wide 128x256 WT4x2 tile (N/256 A-reads) halves A DRAM traffic vs the 256x128
   * cfg5 (N/128). The win scales with the A-panel size M*K*2 bytes, NOT with M
   * alone -- so it fires both for large-M squares AND for mid-M K-heavy shapes
   * (MLP down-projection, large K). Faithful-autotune + fair ours-vs-ours:
   *   - 4096^3           cfg32 3.36 vs cfg5 3.73  (+9.7%, beats hipBLASLt)
   *   - 2048x4096x4096   +3% ; 2048/4096 x 8192^2 win/parity
   *   - 1024x4096x14336  cfg32 3.68 vs cfg5 5.69  (+35%!, -28% -> +16% vs hb)
   * Threshold A>=16 MB with N>=2048: at A<16 MB the mid squares (4608^2/5120^2/
   * 6144^2 at M=1024, ~9-13 MB) still prefer cfg5 (A-panel reuses in L2, and the
   * extra BN just thins the grid), and narrow N (<2048) has too few 256-wide
   * column tiles. */
  if (M >= 1024 && N >= 2048 &&
      static_cast<size_t>(M) * K * 2 >= (16ull << 20)) {
    /* Large A-panel -> BM=128 (fits/reuses L2; BM=256 spills at large K). Pick
     * BN by how K-heavy the shape is: wide BN=256 (cfg32) only pays off when
     * there are enough N-tiles (N>=4096) OR the shape is very K-heavy (K>=3N,
     * where cutting A-reads dominates, e.g. 2560x10240 cfg32 1.89). Otherwise
     * the mid-N shape has too few 256-wide columns and the 128x128 tile wins
     * (2816x8192 cfg12 1.37 vs cfg32 2.0; 3840x8192 cfg12 1.89 vs 2.11). */
    if (N >= 4096 || static_cast<size_t>(K) >= static_cast<size_t>(N) * 3)
      return findWmmaCfg(128, 256, 4, 2, 8, 1, 32);
    return findWmmaCfg(128, 128, 2, 2, 4, 1, 32);
  }
  /* Small N (<2048) with a large A-panel and large K: BM=256 (cfg5) spills the
   * L2 A-panel, and BN=256 has far too few column tiles, so the 128x128 tile
   * wins (2048x1536x6144 cfg12 1.31 vs cfg5 1.95). */
  if (M >= 1024 && N < 2048 && K >= 6144 &&
      static_cast<size_t>(M) * K * 2 >= (16ull << 20))
    return findWmmaCfg(128, 128, 2, 2, 4, 1, 32);
  /* K-heavy at mid M (512..1023): the wide-N tile needs BM=128 (too few M-tiles
   * here), so instead the 256x128 WT4x4 tile (cfg33) -- higher Matrix-Core:LDS
   * ratio for the long compute-bound K-loop -- wins. Fair bench 512x4096x14336:
   * cfg33 2.02 vs cfg5 2.20 (-9% -> parity vs hipBLASLt). Gated K>=8192 so only
   * genuinely K-heavy shapes qualify (pow2 K>=8192 is already handled above). */
  if (M >= 512 && M < 1024 && N >= 2048 && K >= 8192 &&
      static_cast<size_t>(M) * K * 2 >= (8ull << 20))
    return findWmmaCfg(256, 128, 4, 4, 8, 1, 32);
  /* Small square-ish GEMM at large M (N,K <= 4096): the 256x128 cfg5 tile
   * under-fills the grid (few M-tiles) and its extra per-tile traffic buys
   * nothing at small K, so the smaller 128x128 WT2x4 tile wins on occupancy.
   * Fair bench 1024x4096x4096: 0.90 vs cfg5 0.95 (-4% -> parity). Gated N,K<=4096
   * only: at K>=5120 the bigger tile's lower DRAM traffic wins (1024x5120^2 cfg6
   * is -12%), and wide N stays on the N-heavy 256x128 path below. Large-A shapes
   * already took the cfg32 wide-N route above. */
  if (M >= 1024 && N <= 4096 && K <= 4096)
    return findWmmaCfg(128, 128, 2, 4, 8, 1, 32);
  /* Tile size is chosen by M and grid fill (measured on gfx1151):
   *   - Large M reuses the B tile across many A-rows, so a 256-row tile
   *     (256x128 WT4x2) wins once the grid still covers the CUs -- measured
   *     ~2.8x over 128x128 WT2x4 at M=512 N=4096 K=4096.
   *   - Mid M (128..255) has too few row-tiles for BM=256, so 128x128 WT2x4
   *     (8-WMMA / 6-load ratio) is the workhorse.
   * The grid-fill guard (wgs >= ~24) keeps a big tile from starving the CUs;
   * on a small grid we fall to smaller tiles for more workgroups. */
  if (M >= 256 && wgs(256, 128, 1) >= 24)
    return findWmmaCfg(256, 128, 4, 2, 8, 1);
  if (M >= 256 && wgs(128, 256, 1) >= 24)
    return findWmmaCfg(128, 256, 2, 2, 8, 1);
  if (M >= 128 && wgs(128, 128, 1) >= 16)
    return findWmmaCfg(128, 128, 2, 4, 8, 1, 32);
  if (K >= 512 && wgs(64, 128, 1) >= 32)
    return findWmmaCfg(64, 128, 2, 4, 8, 1, 64);
  if (wgs(128, 256, 1) >= 24)
    return findWmmaCfg(128, 256, 2, 2, 8, 1);
  if (wgs(128, 128, 1) >= 16)
    return findWmmaCfg(128, 128, 2, 2, 8, 1);
  if (wgs(64, 256, 1) >= 24)
    return findWmmaCfg(64, 256, 2, 2, 8, 1);
  if (wgs(64, 128, 1) >= 16)
    return findWmmaCfg(64, 128, 2, 2, 8, 1);
  return findWmmaCfg(64, 64, 2, 2, 8, 1);
}

static constexpr int kStaggerNtDefault = 1;
static constexpr int kStaggerNnDefault = 0;
static constexpr int kStaggerU = 32;
static constexpr int kStaggerBytes = 256;
static constexpr int kStaggerMap = 0;

/* Aliasing-K launch rule (NT fp16/bf16, row stride a multiple of
 * kAliasKMod elements): M-first raster with kAliasWgm M-tiles per group plus
 * PGR2 on the BK=32 tiles that have a PGR2 instance (StaggerU is already on
 * for every NT launch). See launchWmma. HIPDNN_EP_GEMM_ALIASK=0 disables it
 * (A/B only); HIPDNN_EP_GEMM_SWIZZLE still overrides the raster. */
static constexpr int kAliasKMod = 1024;
static constexpr int kGridFillWgs = 40;
static constexpr int kAliasWgm = 8;

static bool gemmAliasRuleOn() {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_ALIASK", 1) != 0;
  return on;
}

/* StaggerU parameters (env read once). HIPDNN_EP_GEMM_STAGGER=0/1 (NT default
 * on, NN default per kStaggerNnDefault), _U = number of distinct start offsets
 * (power of two, hipBLASLt uses 32), _BYTES = start-offset step in bytes of K
 * (hipBLASLt 256), _MAP = which tile index selects the offset (0 M-tile,
 * 1 N-tile, 2 serial). Returns the packed value the kernel decodes. */
static int gemmStaggerParam(int bk, int elem_bytes, bool transB) {
  struct Cfg {
    int en_nt, en_nn, su, bytes, map;
  };
  static const Cfg cfg = [] {
    Cfg c;
    const int en = gemmEnvInt("HIPDNN_EP_GEMM_STAGGER", -1);
    c.en_nt = en < 0 ? kStaggerNtDefault : en;
    c.en_nn = en < 0 ? kStaggerNnDefault : en;
    c.su = gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U", kStaggerU);
    c.bytes = gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES", kStaggerBytes);
    c.map = gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP", kStaggerMap);
    return c;
  }();
  if (!(transB ? cfg.en_nt : cfg.en_nn) || cfg.su < 2 || cfg.su > 256 ||
      (cfg.su & (cfg.su - 1)) != 0)
    return 0;
  int step = cfg.bytes / (bk * elem_bytes);
  if (step < 1)
    step = 1;
  if (step > 0xfff)
    step = 0xfff;
  return (cfg.su - 1) | (step << 8) | ((cfg.map & 3) << 20);
}

/* Skinny-M WMMA configs {MT, NTL, KW, KU, SB}: tile (16*MT) x (16*NTL), KW
 * waves splitting K, KU 16-wide k-steps per chunk, SB = B through the LDS row
 * copy. X-macros so each table and its dispatch switch cannot drift apart.
 * GEMM_SKINNY_CFGS: NT; 0-3 are the routed ones (pickSkinny), 4-8
 * alternatives for A/B (HIPDNN_EP_GEMM_SKINNY_CFG). GEMM_SKINNY_TCFGS: an
 * operand transposed through LDS (NN B, transA A; SB = the 128-byte-line form
 * of the NN B staging), with a last column LAY = the layouts a config is
 * built for (1 NN, 2 transA with B = [N, K], 4 transA with B = [K, N]; the
 * staged operand's F * 16*KU / 8 must be 4, 8 or 16 for each). 0-4 serve
 * M < 16 (0-3 routed, 4 for A/B), 5+ M = 16..128 (pickSkinnyT); any of them
 * can be pinned with HIPDNN_EP_GEMM_SKINNY_TCFG. */
#define GEMM_SKINNY_CFGS(X)                                                    \
  X(0, 1, 1, 8, 4, 1)                                                          \
  X(1, 2, 2, 8, 2, 0)                                                          \
  X(2, 2, 1, 4, 4, 1)                                                          \
  X(3, 4, 2, 4, 1, 0)                                                          \
  X(4, 1, 4, 2, 2, 0)                                                          \
  X(5, 4, 1, 4, 1, 0)                                                          \
  X(6, 4, 1, 8, 1, 0)                                                          \
  X(7, 1, 1, 4, 4, 1)                                                          \
  X(8, 4, 1, 4, 2, 1)
#define GEMM_SKINNY_TCFGS(X)                                                   \
  X(0, 1, 4, 2, 2, 0, 7)                                                       \
  X(1, 1, 4, 4, 2, 0, 7)                                                       \
  X(2, 1, 2, 8, 2, 0, 7)                                                       \
  X(3, 1, 4, 2, 2, 1, 7)                                                       \
  X(4, 1, 2, 4, 2, 0, 7)                                                       \
  X(5, 2, 1, 4, 2, 0, 2)                                                       \
  X(6, 2, 1, 8, 2, 0, 2)                                                       \
  X(7, 4, 1, 4, 2, 0, 2)                                                       \
  X(8, 4, 1, 8, 2, 0, 2)                                                       \
  X(9, 2, 2, 4, 2, 0, 2)                                                       \
  X(10, 2, 4, 2, 1, 0, 1)                                                      \
  X(11, 1, 4, 2, 1, 0, 1)                                                      \
  X(12, 2, 2, 4, 1, 0, 1)                                                      \
  X(13, 2, 4, 1, 1, 0, 1)                                                      \
  X(14, 4, 2, 1, 1, 0, 1)                                                      \
  X(15, 2, 2, 1, 1, 0, 1)                                                      \
  X(16, 3, 2, 1, 1, 0, 1)

struct SkinnyCfg {
  int mt, ntl, kw, ku, sb, lay;
};
#define GEMM_SKINNY_ENTRY(ID, MT, NTL, KW, KU, SB) {MT, NTL, KW, KU, SB, 0},
#define GEMM_SKINNY_TENTRY(ID, MT, NTL, KW, KU, SB, LAY)                        \
  {MT, NTL, KW, KU, SB, LAY},
static constexpr SkinnyCfg kSkinny[] = {GEMM_SKINNY_CFGS(GEMM_SKINNY_ENTRY)};
static constexpr SkinnyCfg kSkinnyT[] = {GEMM_SKINNY_TCFGS(GEMM_SKINNY_TENTRY)};
#undef GEMM_SKINNY_TENTRY
#undef GEMM_SKINNY_ENTRY
static constexpr int kNumSkinny =
    static_cast<int>(sizeof(kSkinny) / sizeof(kSkinny[0]));
static constexpr int kNumSkinnyT =
    static_cast<int>(sizeof(kSkinnyT) / sizeof(kSkinnyT[0]));

/* Skinny route (NT fp16/bf16, 2 <= M <= 128): the config to launch, or -1
 * to stay on the WMMA tile path. bench3 stage3 sk2 / sk3 / c1k (one round
 * per config) then v7nt (2 rounds ABBA, slower round); hb = hipBLASLt,
 * fp16 / bf16 ms:
 *   - B <= 32 MB (Infinity-Cache resident): M <= 16 -> cfg 0 (8x4096^2
 *     0.040 / 0.039, hb 0.050 / 0.038), M <= 32 -> cfg 1 (32x4096^2 0.071 /
 *     0.070, hb 0.068). Larger M re-reads A once per N-tile and loses to the
 *     LDS tiles.
 *   - B > 32 MB: M <= 32 -> cfg 2 (16x4096x14336 0.62 / 0.60, hb 0.79 / 0.61;
 *     32x14336x4096 bf16 0.63, hb 0.57); M <= 64 -> cfg 0 when N >= K
 *     (64x14336x4096 bf16 0.70, hb 0.66), else cfg 2 (64x4096x14336 bf16
 *     0.75, hb 0.67); M <= 128 with N >= K -> cfg 3 (128x14336x4096 0.89 /
 *     0.89, hb 0.97 / 1.02). */
static int pickSkinny(int M, int N, int K, int esize) {
  const bool big = static_cast<size_t>(N) * K * esize > (32ull << 20);
  if (!big)
    return M <= 16 ? 0 : (M <= 32 ? 1 : -1);
  if (M <= 32)
    return 2;
  if (M <= 64)
    return N >= K ? 0 : 2;
  return (M <= 128 && N >= K) ? 3 : -1;
}

/* Transposed-operand route (NN B = [K, N] or transA with B = [N, K]; TCFGS
 * index or -1 = keep the GEMV (M < 16) / WMMA tile path). transA with
 * B = [K, N] would transpose both operands and stays put (4x14336x4096
 * 0.550 vs 0.798 ms on the GEMV).
 * M < 16: bench3 stage3 sk2 / v7st, fp16 / bf16 ms: B <= 32 MB -> tcfg 0
 * transA, tcfg 1 NN (8x4096^2 0.046 / 0.044, hb 0.086 / 0.083; 0.043 /
 * 0.042, hb 0.048 / 0.048); larger -> tcfg 2 transA (8x14336x4096 0.671 /
 * 0.670, hb 0.861 / 0.869), tcfg 3 NN (0.619 / 0.618, hb 0.603 / 0.556; the
 * default B staging 0.630 / 0.629, but on 4096^2 it is 0.043 vs 0.056),
 * except NN M <= 4, which keeps the [K, N] GEMV (4x14336x4096 0.555 vs
 * 0.633).
 * M = 16..128: bench3 stage3 p2_sw_nn / p2_sw_ta / p2_rd_nn / p2_sw_kw1
 * (each config forced, 2 rounds ABBA, slower round; fp16 unless noted; old =
 * the WMMA tile route, hb = hipBLASLt), ms:
 *   - transA, B <= 32 MB -> tcfg 9 (16x4096^2 0.048, old 0.205, hb 0.084;
 *     64x4096^2 0.110, old 0.229; 128x4096^2 0.197, old 0.258).
 *   - transA, B > 32 MB: K > N -> tcfg 6 up to M = 32 (16x4096x14336 0.627,
 *     old 1.456, hb 0.986), tcfg 8 up to 64 (64x4096x14336 0.735, old
 *     1.806, hb 1.080); otherwise tcfg 5 up to 32 (16x14336x4096 0.606, old
 *     1.553, hb 0.919) and tcfg 7 (64x14336x4096 0.663, old 1.703;
 *     128x14336x4096 0.893, old 1.834; 128x4096x14336 1.002, old 1.918).
 *   - NN, B <= 32 MB: M <= 16 -> tcfg 1 (16x4096^2 0.045, old 0.103, hb
 *     0.048), M <= 64 -> tcfg 12 (32x4096^2 0.056, old 0.113; 64x4096^2
 *     0.114, old 0.128; 64x3072^2 0.063, old 0.071); larger M stays on the
 *     tile (128x4096^2 0.159 vs 0.246, 96x4096^2 0.163 vs 0.171).
 *   - NN, B > 32 MB and N >= 2K (short K, many N-tiles): one wave per
 *     workgroup (KW = 1) streams the whole K, faster here than splitting
 *     K. fp16 / bf16: M <= 32 -> tcfg 13 (32x14336x4096 0.553 / 0.554,
 *     K-split tcfg 10 0.607 / 0.608, old 0.777, hb 0.577 / 0.563;
 *     32x11008x4096 0.402 / 0.404, old 0.459),
 *     M <= 48 -> tcfg 14 (48x14336x4096 0.571 / 0.570, old 0.787, hb 0.594
 *     / 0.590), M <= 64 -> tcfg 15 (64x14336x4096 0.600 / 0.603, old 0.756;
 *     64x11008x4096 0.423 / 0.423, old 0.419), M <= 96 -> tcfg 16
 *     (96x14336x4096 0.724 / 0.723, old 1.095; 96x11008x4096 0.448 / 0.449,
 *     old 0.538); larger M stays on the tile (128x14336x4096 old 0.838,
 *     tcfg 10 0.867). On 14336x4096 the KW = 1 configs read 0.72-0.83 in
 *     some processes outside the M range they are routed to (tcfg 13 at
 *     M = 48 / 64, tcfg 14 at M <= 32 / 64); inside it they were fast in
 *     every process (12/12, 4/4, 4/4, 4/4).
 *   - NN, B > 32 MB, otherwise: M <= 16 -> tcfg 11 (16x4096x14336 0.524,
 *     old 0.561, hb 0.500), then tcfg 10 up to M = 96, or 128 when K > N
 *     (64x4096x14336 0.583, old 0.658; 128x4096x14336 0.959, old 0.970). */
static int pickSkinnyT(int M, int N, int K, int esize, bool transA,
                       bool transB) {
  if (transA && !transB)
    return -1;
  const bool big = static_cast<size_t>(N) * K * esize > (32ull << 20);
  /* NN M = 2..4 on the ring GEMV (pickKnRing), through launchSmallM. */
  if (!transA && M <= 4 && gemmKnRingOn() && (N % 8) == 0 &&
      static_cast<size_t>(N) * K >= (size_t{1} << 16) &&
      pickKnRing(M, N, K, esize) >= 0)
    return -1;
  if (M < 16) {
    if (!big)
      return transA ? 0 : 1;
    if (transA)
      return 2;
    return M <= 4 ? -1 : 3;
  }
  if (transA) {
    if (!big)
      return 9;
    if (K > N && M <= 64)
      return M <= 32 ? 6 : 8;
    return M <= 32 ? 5 : 7;
  }
  if (!big)
    return M <= 16 ? 1 : (M <= 64 ? 12 : -1);
  if (N >= 2 * K) {
    if (M <= 32)
      return 13;
    if (M <= 48)
      return 14;
    if (M <= 64)
      return 15;
    return M <= 96 ? 16 : -1;
  }
  if (M <= 16)
    return 11;
  return (M <= 96 || K > N) ? 10 : -1;
}

/* Returns false (nothing launched) when the shape does not meet the kernel's
 * preconditions or the route keeps it on its existing path. */
template <typename T, bool TA, bool TB>
static bool launchSkinnyL(hipStream_t s, const T *A, const T *B, const T *C,
                          T *Y, int M, int N, int K, float alpha, float beta,
                          int cDim0, int cDim1) {
  constexpr bool staged = TA || !TB;
  if (((!TA || TB) && (K % 8) != 0) ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) != 0)
    return false;
  static const int forced = gemmEnvInt(
      staged ? "HIPDNN_EP_GEMM_SKINNY_TCFG" : "HIPDNN_EP_GEMM_SKINNY_CFG", -1);
  const int nc = staged ? kNumSkinnyT : kNumSkinny;
  const int esize = static_cast<int>(sizeof(T));
  const int id = (forced >= 0 && forced < nc)
                     ? forced
                     : (staged ? pickSkinnyT(M, N, K, esize, TA, TB)
                               : pickSkinny(M, N, K, esize));
  /* Layout bit of this instantiation in a TCFGS entry's LAY mask. */
  constexpr int lay = TA ? (TB ? 2 : 4) : 1;
  if (id < 0 || (staged && (kSkinnyT[id].lay & lay) == 0))
    return false;
  const SkinnyCfg c = staged ? kSkinnyT[id] : kSkinny[id];
  (void)hipGetLastError();
  /* Several M-tiles (M > 16*MT) are adjacent in dispatch order and share
   * each N-tile's B panel in L2; the transposed A^T staging takes row0 as
   * its column start. */
  const int mtiles = (M + 16 * c.mt - 1) / (16 * c.mt);
  const int ntiles = (N + 16 * c.ntl - 1) / (16 * c.ntl);
  /* StaggerU as for the WMMA tiles: on for B = [N, K], off for B = [K, N]
   * (HIPDNN_EP_GEMM_STAGGER). On NN it spreads the N-tiles of one k-row over
   * different rows. bench3 stage3 p2_stag, fp16 ms on / off: 32x4096^2 tcfg
   * 12 0.0699 / 0.0567, 64x14336x4096 {4,2,4,1} 0.710 / 0.680,
   * 96x4096x14336 tcfg 10 0.843 / 0.813; the M < 16 routes move <= 1.5%
   * (8x4096x14336 tcfg 3 0.622 / 0.632, 8x14336x4096 0.623 / 0.620). */
  const int stag =
      (K % kAliasKMod) == 0
          ? gemmStaggerParam(16 * c.ku, static_cast<int>(sizeof(T)), TB)
          : 0;
  const dim3 grid(mtiles * ntiles);
  const dim3 block(c.kw * 32);
#define GEMM_SKINNY_CASE(ID, MT, NTL, KW, KU, SB)                              \
  case ID:                                                                     \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(                                                       \
            gemm_wmma_skinny_kernel<T, MT, NTL, KW, KU, TA, TB, SB != 0>),     \
        grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,     \
        mtiles, stag);                                                         \
    break;
#define GEMM_SKINNY_TCASE(ID, MT, NTL, KW, KU, SB, LAY)                        \
  case ID:                                                                     \
    if constexpr (((LAY) & lay) != 0) {                                        \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_skinny_kernel<T, MT, NTL, KW, KU, TA, TB, SB != 0>),   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          mtiles, stag);                                                       \
    }                                                                          \
    break;
  if constexpr (staged) {
    switch (id) {
      GEMM_SKINNY_TCFGS(GEMM_SKINNY_TCASE)
    default:
      return false;
    }
  } else {
    switch (id) {
      GEMM_SKINNY_CFGS(GEMM_SKINNY_CASE)
    default:
      return false;
    }
  }
#undef GEMM_SKINNY_TCASE
#undef GEMM_SKINNY_CASE
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma-skinny M=%d N=%d K=%d ta=%d tb=%d -> sk-cfg[%d%s] mt=%d "
      "ntl=%d kw=%d ku=%d sb=%d\n",
      M, N, K, TA ? 1 : 0, TB ? 1 : 0, id, staged ? "t" : "", c.mt, c.ntl,
      c.kw, c.ku, c.sb);
  return true;
}

template <typename T>
static bool launchSkinny(hipStream_t s, const T *A, const T *B, const T *C,
                         T *Y, int M, int N, int K, int transA, int transB,
                         float alpha, float beta, int cDim0, int cDim1) {
  if (transA && transB)
    return launchSkinnyL<T, true, true>(s, A, B, C, Y, M, N, K, alpha, beta,
                                        cDim0, cDim1);
  if (transA)
    return launchSkinnyL<T, true, false>(s, A, B, C, Y, M, N, K, alpha, beta,
                                         cDim0, cDim1);
  if (transB)
    return launchSkinnyL<T, false, true>(s, A, B, C, Y, M, N, K, alpha, beta,
                                         cDim0, cDim1);
  return launchSkinnyL<T, false, false>(s, A, B, C, Y, M, N, K, alpha, beta,
                                        cDim0, cDim1);
}

/* gemm_wmma_nnmid_kernel configs {MW, NTL, NWM, KC, PF} (X-macro: table and
 * dispatch cannot drift apart); rows = 16 * MW * NWM, columns = 16 * NTL. 0-4
 * Infinity-Cache-resident B (32 columns, KC 64), 5-9 a larger B (KC 32).
 * {2, 4, 4, 64, 2} matched cfg 9 on 11008x4096 but needs 244 VGPRs. */
#define GEMM_NNMID_CFGS(X)                                                     \
  X(0, 1, 2, 3, 64, 2)                                                         \
  X(1, 1, 2, 4, 64, 2)                                                         \
  X(2, 2, 2, 2, 64, 2)                                                         \
  X(3, 2, 2, 3, 64, 2)                                                         \
  X(4, 2, 2, 4, 64, 2)                                                         \
  X(5, 1, 6, 4, 32, 2)                                                         \
  X(6, 1, 4, 3, 32, 2)                                                         \
  X(7, 1, 4, 6, 32, 1)                                                         \
  X(8, 1, 6, 6, 32, 2)                                                         \
  X(9, 1, 4, 8, 32, 2)

/* pickNnMid's answer when the NN 128x64 WT4x2 tile (gemm_wmma_kernel, the
 * full-tile BC=false instance with rows past M masked) beats the nnmid
 * configs: BK 32 double-buffered or BK 64 single-buffered. */
static constexpr int kNnMidTileBk32 = -2;
static constexpr int kNnMidTileBk64 = -3;

/* NN mid-M route (fp16/bf16, 33 <= M <= 128): a GEMM_NNMID_CFGS index, one of
 * the two tile answers, or -1 to keep the skinny kernel / WMMA tile route.
 * Screening harness (the product route and the candidates in one process,
 * 2 rounds ABBA, slower round, fp16 / bf16 within a few %), ms old route ->
 * new (bench3 in the commit):
 *   - B <= 32 MB (Infinity-Cache resident): M <= 48 -> cfg 0 (48x3072^2
 *     0.0636 -> 0.0253), M <= 64 -> cfg 1, or cfg 2 on a 32 MB B (64x4096^2
 *     0.115 -> 0.090), M <= 96 -> cfg 3 (96x3072^2 0.091 -> 0.055,
 *     80x2048^2 0.051 -> 0.020), M <= 128 -> cfg 4 (128x2048^2 0.060 ->
 *     0.026) or the BK 64 tile on a 32 MB B (128x4096^2 0.187 -> 0.141; cfg
 *     4 0.98).
 *   - B > 32 MB, N >= 2K (short K, many column strips), a k-row stride of a
 *     4 KB multiple (14336x4096): M <= 64 keeps the KW = 1 skinny configs
 *     (cfg 5 read 0.63 ms at M = 48 in bench3 against their 0.58; they are
 *     bimodal per process and the harness had caught a 0.76 one), then the
 *     BK 32 tile (96x14336x4096 0.722 -> 0.680, 112x14336x4096 1.101 ->
 *     0.680: the ragged-M tile ran the bounds-checked K loop). Another k-row stride (11008x4096): up to
 *     M = 96 the skinny / tile route (the candidates 0.95-1.03), then cfg 9
 *     (112x11008x4096 0.538 -> 0.458).
 *   - B > 32 MB, N < 2K: N < K (4096x14336) keeps the skinny route up to
 *     M = 64, then cfg 7 (96x4096x14336 0.816 -> 0.656) and the BK 32 tile
 *     above M = 96 (112x4096x14336 0.967 -> 0.692); N >= K (8192^2): M <= 48
 *     -> cfg 6 (0.855 -> 0.745), M <= 64 -> cfg 5 (0.875 -> 0.765), M <= 96
 *     -> cfg 8 (0.972 -> 0.786), then the BK 32 tile (112x8192^2 1.797 ->
 *     0.80).
 * The M < 48 / M > 128 neighbours were not measured and keep their routes.
 * N % 64 and K % 64 are required for the tiles (launchNnMid checks). */
static int pickNnMid(int M, int N, int K, int esize) {
  if (M < 33 || M > 128 || esize != 2)
    return -1;
  const size_t wb = static_cast<size_t>(N) * K * esize;
  if (wb <= (32ull << 20)) {
    const bool full = wb > (24ull << 20);
    if (M <= 48)
      return 0;
    if (M <= 64)
      return full ? 2 : 1;
    if (M <= 96)
      return 3;
    return full ? kNnMidTileBk64 : 4;
  }
  if (N >= 2 * K) {
    if ((static_cast<size_t>(N) * esize) % 4096 == 0)
      return M <= 64 ? -1 : kNnMidTileBk32;
    return M <= 96 ? -1 : 9;
  }
  if (N < K) {
    if (M <= 64)
      return -1;
    return M <= 96 ? 7 : kNnMidTileBk32;
  }
  if (M <= 48)
    return 6;
  if (M <= 64)
    return 5;
  return M <= 96 ? 8 : kNnMidTileBk32;
}

struct NnMidCfg {
  int mw, ntl, nwm, kc, pf;
};
#define GEMM_NNMID_ENTRY(ID, MW, NTL, NWM, KC, PF) {MW, NTL, NWM, KC, PF},
static constexpr NnMidCfg kNnMid[] = {GEMM_NNMID_CFGS(GEMM_NNMID_ENTRY)};
#undef GEMM_NNMID_ENTRY
static constexpr int kNumNnMid =
    static_cast<int>(sizeof(kNnMid) / sizeof(kNnMid[0]));

/* Returns false (nothing launched) when the shape does not meet the
 * kernel's preconditions (fp16/bf16 NN, K % KC == 0, N % 8 == 0, 16-byte
 * aligned A and B, M within the config's rows) or the route keeps it on its
 * existing path. HIPDNN_EP_GEMM_NNMID=0 disables it;
 * HIPDNN_EP_GEMM_NNMID_CFG=<id> forces a config for every eligible NN shape
 * with 2 <= M <= its rows. */
template <typename T>
static bool launchNnMid(hipStream_t s, const T *A, const T *B, const T *C,
                        T *Y, int M, int N, int K, float alpha, float beta,
                        int cDim0, int cDim1) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_NNMID", 1) != 0;
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG", -1);
  if (!on || M < 2 || N < 8 || (N % 8) != 0 ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) != 0)
    return false;
  int id = (forced >= 0 && forced < kNumNnMid)
               ? forced
               : pickNnMid(M, N, K, static_cast<int>(sizeof(T)));
  if (id == kNnMidTileBk32 || id == kNnMidTileBk64) {
    /* the existing NN 128x64 WT4x2 instances, full tiles (BC=false: A rows
     * past M re-read row M-1, the epilogue masks them) */
    const int bk = id == kNnMidTileBk32 ? 32 : 64;
    if ((N % 64) != 0 || (K % bk) != 0 || M > 128)
      return false;
    (void)hipGetLastError();
    const dim3 grid(N / 64, 1);
    if (bk == 32)
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              gemm_wmma_kernel<128, 64, 4, 2, false, T, 32, false, true, 1>),
          grid, dim3(128), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
          cDim1, 8, nullptr, N, 0, K, 0);
    else
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(
              gemm_wmma_kernel<128, 64, 4, 2, false, T, 64, false, false, 1>),
          grid, dim3(128), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
          cDim1, 8, nullptr, N, 0, K, 0);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] wmma-nnmid M=%d N=%d K=%d -> tile 128x64 wt=4x2 bk=%d db=%d "
        "bc=0 wgs=%d\n",
        M, N, K, bk, bk == 32 ? 1 : 0, static_cast<int>(grid.x));
    return true;
  }
  if (id < 0)
    return false;
  const NnMidCfg &c = kNnMid[id];
  if ((K % c.kc) != 0 || K < c.kc || M > 16 * c.mw * c.nwm)
    return false;
  (void)hipGetLastError();
  const dim3 grid((N + 16 * c.ntl - 1) / (16 * c.ntl));
#define GEMM_NNMID_CASE(ID, MW, NTL, NWM, KC, PF)                              \
  case ID:                                                                     \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(gemm_wmma_nnmid_kernel<T, MW, NTL, NWM, KC, PF>), grid, \
        dim3(NWM * 32), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1); \
    break;
  switch (id) {
    GEMM_NNMID_CFGS(GEMM_NNMID_CASE)
  default:
    return false;
  }
#undef GEMM_NNMID_CASE
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma-nnmid M=%d N=%d K=%d -> nnmid-cfg[%d] mw=%d ntl=%d nwm=%d "
      "kc=%d pf=%d wgs=%d\n",
      M, N, K, id, c.mw, c.ntl, c.nwm, c.kc, c.pf, static_cast<int>(grid.x));
  return true;
}

/* Launch-side route to the skinny kernel: NT, NN and transA with B = [N, K]
 * 2 <= M <= 128, transA with B = [K, N] 2 <= M < 16 (pickSkinny /
 * pickSkinnyT may still keep a shape on its WMMA path;
 * HIPDNN_EP_GEMM_SKINNY=0 disables it). */
static bool gemmSkinnyEligible(int M, int transA, int transB) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_SKINNY", 1) != 0;
  if (!on || M < 2)
    return false;
  return (transA && !transB) ? M < 16 : M <= 128;
}

template <typename T, bool TransB>
static void launchWmma(int id, hipStream_t s, const T *A, const T *B,
                       const T *C, T *Y, int M, int N, int K, float alpha,
                       float beta, int cDim0, int cDim1, int ldb,
                       int lda = -1) {
  /* A row stride: defaults to K. */
  const int lda_eff = (lda < 0) ? K : lda;
  /* Aliasing-K rule. With a row stride that is a multiple of 2 KB (K%1024 at
   * 2 bytes) every B row a workgroup streams maps to the same cache sets, and
   * the default N-super-column raster puts many N-tiles (= many distinct B
   * rows at the same k) in flight together. hipBLASLt's fix, reproduced here:
   * an M-first raster (neighbouring workgroups share one N-tile's B panel and
   * walk M), StaggerU on the K start (gemmStaggerParam) and PGR2 (two tiles of
   * global loads in flight) on the BK=32 tiles. Applies to LUT-selected and
   * heuristic configs alike, because it lives here and not in the picker.
   * The current LUT was measured without this rule, so its BK<=32 picks are
   * remapped to the PGR2 tile that ranks best under it (online V2 ranking +
   * fair 2-round re-measure, bench3 stage2, fp16 and bf16):
   *   - WT4x4 BK=32 -> WT4x2, same BMxBN (the 4x4 PGR2 instance spills);
   *   - BK=16 -> BK=64, same BMxBN/WT (K%1024 is BK=64-aligned; BK=16 has 4x
   *     the barriers): bf16 64x14336x4096 cfg37 4.84 ms, BK=32 2.39 ms, the
   *     BK=64 twin (fp16's LUT pick for the same shape) 1.25 ms;
   *   - 128x256 -> 256x128 (M >= 256): the 256-row B panel puts twice the
   *     same-set rows in flight. 2048x4096^2 2.46 -> 1.92 ms, bf16 512x4096^2
   *     >0.64 -> 0.59 ms;
   *   - BK=32 tiles with no PGR2 instance (the BM=64 / WT2x2 family), M >= 128
   *     -> 128x128 WT4x2: 128x4096x14336 64x256 2.21 -> 1.07 ms;
   *   - M >= 1024 (>= 8 row tiles): the BK=32 PGR2 tiles stall at the
   *     un-ruled level (4096x4096^2 fp16 4.27 / bf16 4.27 ms, a per-call
   *     row-padded B 4.19 / 4.09); 128x128 WT4x2 BK=64 with a 16-wide
   *     M-first group runs 3.86 /
   *     3.82 (8-wide 3.98 / 3.99; 16-wide on the BK=32 tiles 5.9). At M=2048
   *     the same tile: 2048x4096^2 fp16 1.96 (BK=32 rule) -> 1.86 (16-wide) /
   *     2.05 (8-wide); 2048x4096x14336 8.94 -> 8.41 (8-wide) / 9.15 (16-wide);
   *     2048x14336x4096 6.09 vs 6.13 (16-wide). The 16-wide group only pays
   *     while its 16 A panels (16 * BM * K * elem) fit half the 32 MB
   *     Infinity Cache; K=14336 (58 MB) keeps the 8-wide group. M=1024:
   *     1024x4096^2 fp16 1.059 -> 1.007, bf16 1.067 -> 1.022; phi3 1024x16384x
   *     3072 unchanged (2.64 / 2.63). Known miss: fp16 2048x4096^2 with a bias
   *     C 1.94 -> 2.03 (bf16 1.91 -> 1.88). M=512 keeps BK=32: the BK=64 tiles
   *     lose 5-15% there.
   * Drop these remaps once the LUT is re-measured with the rule in place. */
  const bool alias_k = TransB && gemmAliasRuleOn() &&
                       (ldb % kAliasKMod) == 0 && (lda_eff % kAliasKMod) == 0;
  WmmaCfg c = kWmma[id];
  /* Small M (<= 128) under the rule: the alias-era LUT hands these shapes
   * anything from a split-K 64x128 to a 128x256 tile with half its rows idle
   * (64x4096x14336). With StaggerU keyed by N-tile (below) the BK=64 tiles
   * win: M <= 64 -> 64x128 WT2x2 BK64 (64x4096^2 0.080 ms, hb 0.101); M <= 128
   * -> 128x64 WT4x2 BK64 (128x4096^2 0.1425, hb 0.169; 128x4096x14336 0.994,
   * hb 1.038). bench3 stage3 c1m / c1k. Shapes the skinny kernel takes never
   * get here. */
  if (alias_k && M <= 128 && (K % 64) == 0) {
    if (M > 64)
      c = WmmaCfg{128, 64, 4, 2, 8, 1, 64};
    else
      c = WmmaCfg{64, 128, 2, 2, 8, 1, 64};
  }
  if (alias_k) {
    if (c.bk == 16 && (K % 64) == 0)
      c.bk = 64;
    if (c.bk == 32 && c.wt_m == 4 && c.wt_n == 4)
      c.wt_n = 2;
    if (c.bk == 32 && c.bm == 128 && c.bn == 256 && c.wt_m == 4 && M >= 256) {
      c.bm = 256;
      c.bn = 128;
    }
    if (c.bk == 32 && M >= 128 && !wmmaHasPgr2(c))
      c = WmmaCfg{128, 128, 4, 2, c.swizzle_n, 1, 32};
    if (c.bk == 32 && M >= 1024 && (K % 64) == 0)
      c = WmmaCfg{128, 128, 4, 2, c.swizzle_n, 1, 64};
    /* The 64-row BK=64 LUT picks at M >= 1024 (lm_head 2048x128256x8192 gets
     * 64x128 WT2x4) re-read B M/512 times under the 8-wide M-first group:
     * fp16 190.9 -> 111.3 ms with 128x128 WT4x2 (hipBLASLt 100.5). */
    if (c.bk == 64 && c.bm == 64 && M >= 1024)
      c = WmmaCfg{128, 128, 4, 2, c.swizzle_n, 1, 64};
  }
  /* Mid M with a very long aliasing K: the 4-wave 128x64 WT4x2 BK64 tile,
   * double-buffered (55 KB). bench3 WP1 E1 (gemm_wpA_wpB_gfx1151_v2), fp16,
   * slower of 2 rounds: 512x4096x14336 2.51 -> 2.01 ms, 1024x1536x12288
   * 1.50 -> 1.31 (hipBLASLt 1.87 / 1.01). M >= 2048 keeps the 128x128 BK64
   * tile: every 4-wave variant was 14-48% slower there.
   * HIPDNN_EP_GEMM_BIGK_MID=0 disables. */
  static const bool bigk_mid_on = gemmEnvInt("HIPDNN_EP_GEMM_BIGK_MID", 1) != 0;
  const bool bigk_mid = alias_k && bigk_mid_on && M >= 256 && M <= 1024 &&
                        K >= 12288 && (K % 64) == 0;
  if (bigk_mid)
    c = WmmaCfg{128, 64, 4, 2, c.swizzle_n, 1, 64};
  /* Grid fill (outside the K%1024 rule): a tile that puts fewer workgroups
   * than CUs on the 40-CU part steps down a ladder of smaller tiles until the
   * grid covers them (BK=16 tiles when K%32 == 16). bench3 stage3 c3a, fp16
   * ms with / without: 729x1152x4304 (LUT 128x256, 30 WGs) 0.206 / 0.295,
   * 512x1536x384 0.021 / 0.029, 512x384x1536 0.023 / 0.034 (hipBLASLt 0.182 /
   * 0.014 / 0.015). HIPDNN_EP_GEMM_GRIDFILL=0 disables. */
  static const bool gridfill_on = gemmEnvInt("HIPDNN_EP_GEMM_GRIDFILL", 1) != 0;
  if (gridfill_on && !alias_k) {
    auto wgs = [&](const WmmaCfg &t) {
      return ((M + t.bm - 1) / t.bm) * ((N + t.bn - 1) / t.bn);
    };
    if (wgs(c) < kGridFillWgs) {
      const bool k16 = (K % 32) != 0 && (K % 16) == 0;
      static constexpr WmmaCfg kLadder[] = {
          {128, 128, 4, 2, 8, 1, 32}, {64, 128, 2, 2, 8, 1, 32},
          {128, 64, 2, 2, 8, 1, 32},  {64, 64, 2, 2, 8, 1, 32}};
      static constexpr WmmaCfg kLadder16[] = {{128, 128, 2, 4, 8, 1, 16},
                                              {64, 128, 2, 4, 8, 1, 16}};
      const WmmaCfg *lad = k16 ? kLadder16 : kLadder;
      const int nl = k16 ? 2 : 4;
      for (int i = 0; i < nl; ++i) {
        if (lad[i].bm * lad[i].bn >= c.bm * c.bn)
          continue; /* only ever smaller */
        c = lad[i];
        if (wgs(c) >= kGridFillWgs)
          break;
      }
    }
  }
  /* Mid-size NT shapes (vision / encoder layers). The LUT and the grid-fill
   * ladder only know power-of-two tiles, which leave 512 <= M <= 2048 grids
   * unevenly spread over the 20 WGPs or re-read A too often; these measured
   * faster (bench3 g2_c1, 2 ABBA rounds, slower round, fp16 / bf16 ms,
   * old -> new; hipBLASLt fastest round):
   *   - K % 32 == 16, M 512..1024, N <= 2048 -> 96x128 WT3x4 BK16 (72 WGs
   *     instead of 54): 729x1152x4304 0.213 / 0.206 -> 0.197 / 0.196
   *     (0.186 / 0.184);
   *   - K % 32 == 0, K <= 2048, M 512..1024, N 2048..8192 -> 128x128 WT4x4
   *     BK32 (4 waves, LUT: 64x128 WT2x4 BK64 / 128x256 WT4x2):
   *     729x4304x1152 0.223 / 0.210 -> 0.205 / 0.205 (4 rounds; 0.197 /
   *     0.157);
   *   - K % 32 == 0, K <= 1024, M 1025..2048, N 2048..4096 -> 96x96
   *     WT3x3 BK32: 1576x3072x768 0.216 / 0.229 -> 0.205 / 0.207 (0.198 /
   *     0.219);
   *   - K%1024 alias, M 1024..2048, N <= 1024 -> 128x256 WT4x2 BK64 (A read
   *     N/256 times instead of N/128): 1576x768x3072 0.208 / 0.212 -> 0.187 /
   *     0.184 (0.192 / 0.191);
   *   - M 256..512, N 256..512, K >= 1024 -> 64x64 WT2x2 BK64: 512x384x1536
   *     0.0237 / 0.0234 -> 0.0228 / 0.0220 (0.0156 / 0.0154);
   *   - M 256..512, N 1024..4096, K <= 512 -> 128x128 WT4x4 BK32 (4 waves):
   *     512x1536x384 0.0215 / 0.0208 -> 0.0199 / 0.0201 (0.0204 / 0.0152).
   * Not with a C term (bias): these tiles' generic epilogue reads C per output
   * element (bias-vitb-up on 96x96: 0.226 -> 0.241 ms). HIPDNN_EP_GEMM_MIDTILE=0
   * disables. */
  static const bool midtile_on = gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE", 1) != 0;
  /* Row-broadcast C (the bias epilogue, PGR bit 8): the 96x96 and the alias
   * 128x128 WT4x2 BK64 tiles have an instance that reads C once per column
   * (see biasepi below). */
  static const bool biasepi_on = gemmEnvInt("HIPDNN_EP_GEMM_BIASEPI", 1) != 0;
  const bool bias_epi_ok = TransB && biasepi_on && C != nullptr &&
                           beta != 0.0f && cDim0 == 1 && (N & 1) == 0;
  /* With a C term only the 96x96 rule, and only when the bias epilogue
   * applies (bias_epi_ok). */
  int mid_pick = 0; /* the two minilm branches, for the BUFLOAD rule */
  if (TransB && midtile_on && M >= 256 && M <= 2048 &&
      (!(C != nullptr && beta != 0.0f) || bias_epi_ok)) {
    const bool c_term = C != nullptr && beta != 0.0f;
    const bool k16 = (K % 32) != 0 && (K % 16) == 0;
    if (c_term) {
      if (!alias_k && (K % 32) == 0 && K <= 1024 && M > 1024 && N >= 2048 &&
          N <= 4096)
        c = WmmaCfg{96, 96, 3, 3, 8, 1, 32};
    } else if (k16 && M >= 512 && M <= 1024 && N <= 2048)
      c = WmmaCfg{96, 128, 3, 4, 8, 1, 16};
    else if (!alias_k && (K % 32) == 0 && K <= 2048 && M >= 512 &&
             M <= 1024 && N >= 2048 && N <= 8192)
      c = WmmaCfg{128, 128, 4, 4, 8, 1, 32};
    else if (!alias_k && (K % 32) == 0 && K <= 1024 && M > 1024 &&
             N >= 2048 && N <= 4096)
      c = WmmaCfg{96, 96, 3, 3, 8, 1, 32};
    else if (alias_k && M >= 1024 && N <= 1024 && (K % 64) == 0)
      c = WmmaCfg{128, 256, 4, 2, c.swizzle_n, 1, 64};
    else if (!alias_k && M <= 512 && N >= 256 && N <= 512 && K >= 1024 &&
             (K % 64) == 0) {
      c = WmmaCfg{64, 64, 2, 2, 8, 1, 64};
      mid_pick = 1;
    } else if (!alias_k && M <= 512 && N >= 1024 && N <= 4096 && K <= 512 &&
               (K % 32) == 0) {
      c = WmmaCfg{128, 128, 4, 4, 8, 1, 32};
      mid_pick = 2;
    }
  }
  /* Adaptive threadblock rasterization width (L2 A-panel reuse). A wide tile's
   * A-panel is re-touched by every N-tile in its rasterization super-column, so
   * once the launch has MANY M-tile rows the default 8-wide super-column evicts
   * the A-panels from the 2 MB L2 before they are reused, and throughput drops
   * (N=K=4096 M=4096, 16 M-tiles: 28 TFLOP at sw=8 -> 36 at sw=32, = hipBLASLt).
   * Widen the super-column when the M-tile count is high AND the weight B is
   * small enough that its streaming does not already thrash the 2 MB L2. For a
   * big weight (8192^2 = 128 MB, 6144^2 = 72 MB) the B stream evicts the reused
   * A-panels regardless of scheduling, and sw=32 is a large regression there
   * (8192^2 M=4096: -54%), so keep the default 8. Measured win: N=K=4096
   * M=4096/8192 (16/32 M-tiles, 32 MB weight) 28 -> 37 TFLOP (= hipBLASLt).
   * gemmSwizzleBlock clamps to n_tiles, so a wide value is safe on a small grid. */
  const int my_tiles = (M + c.bm - 1) / c.bm;
  int sw_eff = c.swizzle_n;
  if (my_tiles >= 12 &&
      static_cast<size_t>(N) * K * 2 <= (48ull << 20))
    sw_eff = 32;
  /* Big-weight wide-N tile with FEW M-tile rows (e.g. 1024x8192^2 on cfg32:
   * 128 MB B, 8 M-tiles): a narrow super-column raster (sw=4) keeps the reused
   * A block-row hotter in L2 across the N-sweep and raises the L2 hit rate on
   * the DRAM-bound stream. Measured 1024x8192^2 4.19 -> 3.82 ms (+9%, -17% ->
   * -11% vs hipBLASLt). Only helps when the grid has few M-tiles: at 16/32
   * M-tiles (M=2048/4096) it is neutral-to-slightly-worse, so keep the default
   * sw there. Gated to the wide tile (bn>=256) so the narrow pow2-K cfg27 path
   * is untouched. */
  if (c.bn >= 256 && my_tiles <= 8 &&
      static_cast<size_t>(N) * K * 2 > (96ull << 20))
    sw_eff = 4;
  if (alias_k)
    sw_eff = (c.bk == 64 && c.bm == 128 && my_tiles >= 2 * kAliasWgm &&
              static_cast<size_t>(2 * kAliasWgm) * c.bm * K * sizeof(T) <=
                  (16ull << 20))
                 ? -2 * kAliasWgm
                 : -kAliasWgm;
  const bool pgr2 = alias_k && c.bk == 32;
  /* Alias rule's 128x128 WT4x2 BK64 tile with line-wide B loads (PGR bit 4):
   * its default B staging gives each lane 64 contiguous bytes of one row, so a
   * load instruction touches 16 rows x 32 B; the line-wide staging reads 4
   * rows x 128 B per instruction. These shapes are bound by L2-miss traffic
   * (RGP, fp16 2048x4096x14336: memory unit stalled 98.7% of the time at 354
   * GB/s, hipBLASLt 69% at 432). bench3 g2_c1, slower of 2 ABBA rounds,
   * fp16 / bf16 ms without -> with: 2048x4096x14336 8.39 / 8.32 -> 8.17 /
   * 8.14, 2048x14336x4096 6.18 / 6.11 -> 6.12 / 5.93, 2048x4096^2 1.90 /
   * 1.86 -> 1.82 / 1.84, 4096^3 3.89 / 3.85 -> 3.81 / 3.81; single shapes
   * move within the run-to-run noise (the g3 session had 2048x4096x14336
   * fp16 8.29 -> 8.47), the geomean over the six K%1024 shapes and both
   * sessions is 0.987. HIPDNN_EP_GEMM_LINEB=0 disables. */
  static const bool lineb_on = gemmEnvInt("HIPDNN_EP_GEMM_LINEB", 1) != 0;
  const bool lineb = TransB && alias_k && lineb_on && c.bm == 128 &&
                     c.bn == 128 && c.wt_m == 4 && c.wt_n == 2 && c.bk == 64;
  /* Bias epilogue (PGR bit 8) on the tiles it is built for: with a
   * row-broadcast C the default epilogue reads C per output element through
   * gemmStoreY and writes 2-byte stores; this one reads C once per fragment
   * column and keeps the paired 4-byte stores. bench3 g3, slower of 2 ABBA
   * rounds, fp16 / bf16 ms without -> with (hipBLASLt): 2048x4096^2 + bias
   * 1.988 / 2.075 -> 1.869 / 1.963 (1.887 / 1.894); 1576x3072x768 + bias
   * (old tile -> 96x96) 0.260 / 0.243 -> 0.235 / 0.217 (0.207 / 0.211).
   * HIPDNN_EP_GEMM_BIASEPI=0 disables. */
  const bool biasepi =
      bias_epi_ok && ((lineb) || (c.bm == 96 && c.bn == 96 && c.wt_m == 3 &&
                                  c.wt_n == 3 && c.bk == 32));
  /* Buffer-load read path (gemm_wmma_buf_kernel): buffer_load_b128 with the
   * K position in one SGPR offset, no 64-bit address VALU in the K loop,
   * loads spread between the WMMAs or issued two steps ahead. Replaces the
   * minilm-class tiles of the mid-shape rule (K % 128 == 0, even N, Y = A*B).
   * bench3 c1_scan (2 ABBA rounds, slower round, fp16 / bf16 ms, old ->
   * new; hipBLASLt fastest round):
   *   - M 256..512, N 256..512, K 1024..2048, <= 48 WGs of 64x64: the same
   *     64x64 WT2x2 BK64 tile with the loads spread between the WMMAs (RP 5):
   *     512x384x1536 0.0242 / 0.0227 -> 0.0199 / 0.0199 (0.0193 / 0.0148);
   *     256x384x1536 0.0149 / 0.0150 -> 0.0138 / 0.0142; 512x256x1152
   *     0.0115 / 0.0116 -> 0.0105 / 0.0105. Not at 64 WGs (512x512x1536
   *     1.018 / 1.000, 512x512x4352 1.091 / 1.032 of the old time);
   *   - M 256..512, N 1024..4096, K <= 512: hipBLASLt's 128x80 tile (WT2x5
   *     BK64, single-buffered 30 KB: 4 WGs per WGP, 80 WGs on 512x1536)
   *     with two tiles of loads in flight (RP 2) instead of 128x128 WT4x4
   *     BK32: 512x1536x384 0.0203 / 0.0199 -> 0.0179 / 0.0171 (0.0163 /
   *     0.0144); 256x1536x384 0.0132 / 0.0136 -> 0.0095 / 0.0100;
   *     384x2048x512 0.0322 / 0.0327 -> 0.0291 / 0.0294.
   * The 20 us shapes run ~30% faster still when the GPU has been busy for
   * >~20 ms without a break (stage5 screen: 512x384x1536 0.014 ms); bench3's
   * 10 ms blocks do not reach that state for either side.
   * Also replaces the K%1024 alias rule's 128x128 WT4x2 BK64 line-wide-B
   * tile (M >= 1024; Y = A*B or the bias epilogue) with 128x128 WT2x4 BK64,
   * single-buffered, two tiles of loads in flight (RP 2; bias RP 10; the
   * WT4x2 tile with the same loop was 1-4% behind in the stage5 screen).
   * For K >= 12288 the fragment loads are also spread one share per WMMA
   * (RP 18; screen, 2048x4096x14336: 0.916 / 0.911 of the old tile, RP 2
   * 0.953 / 0.950). bench3 c2_scan (2 ABBA rounds, slower round, fp16 /
   * bf16 ms, old -> new; hipBLASLt fastest): 2048x4096x14336 8.201 / 8.249
   * -> 7.417 / 7.540 (8.014 / 7.631); 1024x16384x3072 2.611 / 2.598 ->
   * 2.470 / 2.544 (2.494 / 2.483); 2048x4096^2 1.823 / 1.816 -> 1.791 /
   * 1.774, with a bias 1.860 / 1.876 -> 1.805 / 1.822; 2048x14336x4096
   * 5.996 / 5.960 -> 5.861 / 5.824; 4096^3 3.800 / 3.759 -> 3.729 / 3.804
   * (3.693 / 4.087); geomean over 15 K%1024 shapes 0.961 / 0.968.
   * bl= in the final line. HIPDNN_EP_GEMM_BUFLOAD=0 disables. */
  static const bool bufload_on = gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD", 1) != 0;
  const bool c_term = C != nullptr && beta != 0.0f;
  const bool plain = alpha == 1.0f && !c_term;
  int bl = 0;
  if (TransB && bufload_on && (N & 1) == 0 && (K % 128) == 0) {
    if (plain && mid_pick == 1 && K <= 2048 &&
        ((M + 63) / 64) * ((N + 63) / 64) <= 48) {
      bl = 1;
    } else if (plain && mid_pick == 2) {
      bl = 2;
      c = WmmaCfg{128, 80, 2, 5, 8, 1, 64};
    } else if (lineb && (plain || bias_epi_ok)) {
      bl = (plain && K >= 12288) ? 4 : 3;
      c = WmmaCfg{128, 128, 2, 4, c.swizzle_n, 1, 64};
    }
  }
  /* Long-K NT down projections. A separate rolling-prefetch template stores
   * tile t+1 into the alternate LDS slot before issuing t+2, with row groups
   * spread across the four WMMA K steps. Halo13 screening, slower of two
   * opposite-order rounds: pre512 2.079/2.064 -> 1.668/1.628 ms;
   * qwen7b 7.084/7.101 -> 6.511/6.423 ms (fp16/bf16).
   * Other measured down pairs use the existing two-register-slot buffer
   * loop. Only measured M/N/K pairs enter; all instances stay outside kWmma.
   * HIPDNN_EP_GEMM_LONGK=0 disables; LONGK_CFG=0..4 forces an entry
   * inside this measured set, with BUFLOAD enabled and Y=A*B. */
  static const bool longk_on = gemmEnvInt("HIPDNN_EP_GEMM_LONGK", 1) != 0;
  static const int longk_forced = gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG", -1);
  int lk = -1;
  if (TransB && bufload_on && longk_on && plain && (N & 1) == 0 &&
      (K % 128) == 0) {
    if (M == 512 && N == 4096 && K == 14336)
      lk = 0;
    else if (M == 512 && N == 896 && K == 4864)
      lk = 1;
    else if ((M == 512 && N == 2048 && K == 11008) ||
             ((M == 512 || M == 2048) && N == 5120 && K == 13824))
      lk = 2;
    else if (M == 2048 && N == 3584 && K == 18944)
      lk = 4;
    else if (M == 2048 && ((N == 1536 && K == 8960) ||
                           (N == 2048 && K == 11008)))
      lk = 3;
    if (lk >= 0) {
      if (longk_forced >= 0 && longk_forced <= 4)
        lk = longk_forced;
      static const WmmaCfg longk_tiles[] = {
          {128, 64, 4, 2, 8, 1, 64}, {128, 192, 2, 6, 8, 1, 64},
          {256, 128, 4, 2, 8, 1, 64}, {128, 128, 2, 4, 32, 1, 64},
          {128, 80, 2, 5, 8, 1, 64}};
      c = longk_tiles[lk];
      bl = lk + 5;
      sw_eff = lk == 3 ? 32 : (alias_k ? -8 : 8);
    }
  }
  {
    /* <0 selects the M-first raster (see gemmSwizzleBlock). */
    static const int sw_env = gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE", 0);
    if (sw_env != 0)
      sw_eff = sw_env;
  }
  /* StaggerU only for an aliasing K (rows a multiple of 2 KB apart): on other
   * K it costs 2-3% on short-K shapes (bench3 stage3 c0e,
   * 4 rounds ABBA, slower round: f16 729x4304x1152 0.296 -> 0.268 ms,
   * 512x1536x384 0.0319 -> 0.0297 with it off). */
  int stag = (K % kAliasKMod) == 0
                 ? gemmStaggerParam(c.bk, static_cast<int>(sizeof(T)), TransB)
                 : 0;
  /* With fewer M-tile rows than one M-first raster group, StaggerU keyed by
   * M-tile hands every workgroup the same K offset, so all N-tiles hit the
   * same aliasing B sets together; key it by N-tile instead (unless
   * HIPDNN_EP_GEMM_STAGGER_MAP pins the map). bench3 stage3 c1m, fp16:
   * 64x4096^2 64x128 WT2x2 BK64 0.113 -> 0.0795 ms, 128x4096^2 128x64 WT4x2
   * BK64 0.164 -> 0.1425 (hipBLASLt 0.101 / 0.169). */
  static const bool map_pinned =
      gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP", -1) >= 0;
  if (stag != 0 && !map_pinned && my_tiles < kAliasWgm)
    stag = (stag & ~(3 << 20)) | (1 << 20);
  /* The tile actually launched, after every remap above (the "LUT hit" /
   * "heuristic" lines print the config before them). pgr2 = 1 only when the
   * tile has a PGR2 instance. */
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma final M=%d N=%d K=%d -> %dx%d wt=%dx%d bk=%d pgr2=%d lb=%d "
      "be=%d bl=%d sw=%d stag=%#x wgs=%d\n",
      M, N, K, c.bm, c.bn, c.wt_m, c.wt_n, c.bk,
      (TransB && pgr2 && wmmaHasPgr2(c)) ? 1 : 0, (lineb && !bl) ? 1 : 0,
      (bl ? (bl >= 3 && c_term) : biasepi) ? 1 : 0, bl, sw_eff, stag,
      ((M + c.bm - 1) / c.bm) * ((N + c.bn - 1) / c.bn));
  const int th = (c.bm / (c.wt_m * 16)) * (c.bn / (c.wt_n * 16)) * 32;
  /* The kernel keeps its split-K partial-sum argument; nothing launches
   * split-K, so it is always null. */
  float *const skp = nullptr;
  dim3 block(th);

  /* Single-buffer (DB=false) when double-buffered LDS would exceed ~48 KB, so
   * two workgroups fit per CU and hide the K-loop barriers across each other.
   * Below that the tile already reaches useful occupancy, so keep the
   * latency-hiding double buffer. */
  const int db_lds = 2 * (c.bm + c.bn) * (c.bk + 8) * 2;
  /* Single-buffer (2 WG/CU) is the default: it hides the K-loop barrier across
   * co-resident workgroups and wins when the B/A loads hit the 2 MB L2. But once
   * the streaming footprint spills L2 (large pow2-K weight, e.g. the 128 MB B of
   * 8192x8192), the now-exposed DRAM load latency dominates and the latency-
   * hiding double buffer wins despite dropping to 1 WG/CU -- provided it still
   * fits in 64 KB LDS. Only the 128x64 BK=64 tile (used for pow2 K>=8192) lands
   * in the 49-57 KB double-buffer window; larger tiles stay single. */
  const bool spills_l2 = static_cast<size_t>(N) * K * 2 > (96ull << 20);
  const bool db = db_lds <= 49152 ||
                  (c.bk == 64 && db_lds <= 57344 && (spills_l2 || bigk_mid));

  /* Sub-launches. Normally one launch covering the whole [M,N] grid with a
   * single compile-time BC. When only N is ragged (M,K tile-aligned), launch ceil(N/BN) FULL tiles all on the fast BC=false path: the
   * B-load is N-safe (NT reads a dummy row for the column overhang, NN
   * loads only the column chunks below N) and the store masks cc>=N (a cheap epilogue compare). Measured on
   * vision mlp-up N=4304 M=512: the old interior+ragged-edge split was 0.185 ms
   * (a separate ragged-edge launch costs more than a full extra tile); one
   * BC=false launch of 34 full tiles matches the aligned N=4352 (0.141 ms) --
   * i.e. the N%BN!=0 tax disappears. */
  struct SubLaunch {
    dim3 grid;
    bool bc;
    int nb;
  };
  SubLaunch subs[2];
  int nsub;
  const int my = (M + c.bm - 1) / c.bm;
  const bool n_ragged = (N % c.bn) != 0;
  /* A ragged M takes the same full-tile BC=false launch for NT (A rows past M
   * are clamped in the kernel, their outputs masked): vision M = 1576 / 729
   * otherwise ran the whole K loop on the bounds-checked path. bench3 stage3
   * c3a, 2 rounds ABBA, fp16 ms with / without: 1576x3072x768 0.216 / 0.250,
   * 729x4304x1152 0.215 / 0.258 (hipBLASLt 0.201 / 0.195).
   * HIPDNN_EP_GEMM_MRAGGED=0 disables. */
  static const bool m_ragged_on = gemmEnvInt("HIPDNN_EP_GEMM_MRAGGED", 1) != 0;
  const bool m_ragged = (M % c.bm) != 0;
  if ((K % c.bk == 0) &&
      ((n_ragged && !m_ragged) || (TransB && m_ragged && m_ragged_on))) {
    subs[0] = SubLaunch{dim3((N + c.bn - 1) / c.bn, my, 1), false, 0};
    nsub = 1;
  } else {
    const bool bc = !(M % c.bm == 0 && N % c.bn == 0 && K % c.bk == 0);
    subs[0] = SubLaunch{dim3((N + c.bn - 1) / c.bn, my, 1), bc, 0};
    nsub = 1;
  }

#define GEMM_WMMA_ONE(BM, BN, WM, WN, BK)                                      \
  GEMM_WMMA_ONE_P(BM, BN, WM, WN, BK, 1)

#define GEMM_WMMA_ONE_P(BM, BN, WM, WN, BK, PG)                                \
  do {                                                                         \
    if (bc && db)                                                              \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, true, true, PG>),    \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff, stag);                                               \
    else if (bc && !db)                                                        \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, true, false, PG>),   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff, stag);                                               \
    else if (!bc && db)                                                        \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, false, true, PG>),   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff, stag);                                               \
    else                                                                       \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, false, false, PG>),  \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff, stag);                                               \
  } while (0)

#define GEMM_WMMA_TRY(BM, BN, WM, WN, BK)                                      \
  if (c.bm == BM && c.bn == BN && c.wt_m == WM && c.wt_n == WN &&              \
      c.bk == BK) {                                                            \
    GEMM_WMMA_ONE(BM, BN, WM, WN, BK);                                         \
  } else

  /* BK=32 tile with a PGR2 instance (NT only; the alias rule sets pgr2).
   * Not the WT 4x4 tiles: their PGR2 instance spills (scratch > 0). */
#define GEMM_WMMA_TRY_PGR(BM, BN, WM, WN)                                      \
  if (c.bm == BM && c.bn == BN && c.wt_m == WM && c.wt_n == WN &&              \
      c.bk == 32) {                                                            \
    if constexpr (TransB) {                                                    \
      if (pgr2)                                                                \
        GEMM_WMMA_ONE_P(BM, BN, WM, WN, 32, 2);                                \
      else                                                                     \
        GEMM_WMMA_ONE(BM, BN, WM, WN, 32);                                     \
    } else {                                                                   \
      GEMM_WMMA_ONE(BM, BN, WM, WN, 32);                                       \
    }                                                                          \
  } else

  /* BUFLOAD rule: one full-tile launch (the kernel clamps ragged rows). */
  if constexpr (TransB && sizeof(T) == 2) {
    if (bl != 0) {
      const dim3 grid((N + c.bn - 1) / c.bn, (M + c.bm - 1) / c.bm);
#define GEMM_WMMA_BUF(BM, BN, WM, WN, DB, RP)                                  \
  hipLaunchKernelGGL(                                                          \
      HIP_KERNEL_NAME(gemm_wmma_buf_kernel<BM, BN, WM, WN, T, 64, DB, RP>),    \
      grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,       \
      sw_eff, ldb, lda_eff, stag)
      if (bl == 1)
        GEMM_WMMA_BUF(64, 64, 2, 2, true, 5);
      else if (bl == 2)
        GEMM_WMMA_BUF(128, 80, 2, 5, false, 2);
      else if (bl == 5)
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_wmma_bufroll_kernel<128, 64, 4, 2, T, 64,
                                                      true, 161>),
            grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
            sw_eff, ldb, lda_eff, stag);
      else if (bl == 9)
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_wmma_bufroll_kernel<128, 80, 2, 5, T, 64,
                                                      true, 161>),
            grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
            sw_eff, ldb, lda_eff, stag);
      else if (bl == 6)
        GEMM_WMMA_BUF(128, 192, 2, 6, false, 2);
      else if (bl == 7)
        GEMM_WMMA_BUF(256, 128, 4, 2, false, 2);
      else if (bl == 8)
        GEMM_WMMA_BUF(128, 128, 2, 4, false, 2);
      else if (bl == 4)
        GEMM_WMMA_BUF(128, 128, 2, 4, false, 18);
      else if (c_term)
        GEMM_WMMA_BUF(128, 128, 2, 4, false, 10);
      else
        GEMM_WMMA_BUF(128, 128, 2, 4, false, 2);
#undef GEMM_WMMA_BUF
      return;
    }
  }

  /* Exact measured routes only: generic compilation retains the early
   * next-tile loads in separate registers. Respect explicit route overrides. */
  if constexpr (TransB && sizeof(T) == 2 && HIPDNN_WMMA_FRAG_ELEMS == 16) {
    static const bool overlap = gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP", 1) != 0;
    static const int forced_overlap = gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG", -1);
    const bool eligible = overlap && alpha == 1.0f && beta == 0.0f && !C &&
                          nsub == 1 && !subs[0].bc && !bl && !lineb && !pgr2 &&
                          !skp && lda_eff == K && ldb == K &&
                          (((reinterpret_cast<uintptr_t>(A) |
                             reinterpret_cast<uintptr_t>(B)) & 15) == 0);
    if (eligible) {
      int cfg = -1;
      if (M == 2048 && N == 18944 && K == 3584 && c.bm == 256 && c.bn == 128 &&
          c.wt_m == 4 && c.wt_n == 2 && c.bk == 32 && !db &&
          sw_eff == 8 && stag == 0)
        cfg = 0;
      else if (M == 729 && N == 1152 && K == 4304 && c.bm == 96 && c.bn == 128 &&
               c.wt_m == 3 && c.wt_n == 4 && c.bk == 16 && db &&
               sw_eff == 8 && stag == 0)
        cfg = 1;
      else if (M == 1576 && N == 768 && K == 3072 && c.bm == 128 && c.bn == 256 &&
               c.wt_m == 4 && c.wt_n == 2 && c.bk == 64 && !db &&
               sw_eff == -8 && stag == 0x21f)
        cfg = 2;
      if (cfg >= 0 && (forced_overlap < 0 || forced_overlap == cfg)) {
        CUSTOM_KERNELS_DEBUG_LOG("[gemm] wmma overlap cfg=%d M=%d N=%d K=%d\n",
                                 cfg, M, N, K);
        if (cfg == 0)
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_wmma_overlap_kernel<256, 128, 4, 2, T, 32,
                                                         false, 7>),
              subs[0].grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
              cDim0, cDim1, sw_eff, lda_eff, ldb, stag);
        else if (cfg == 1)
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_wmma_overlap_kernel<96, 128, 3, 4, T, 16,
                                                         true, 7>),
              subs[0].grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
              cDim0, cDim1, sw_eff, lda_eff, ldb, stag);
        else
          hipLaunchKernelGGL(
              HIP_KERNEL_NAME(gemm_wmma_overlap_kernel<128, 256, 4, 2, T, 64,
                                                         false, 1>),
              subs[0].grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
              cDim0, cDim1, sw_eff, lda_eff, ldb, stag);
        return;
      }
    }
  }

  for (int sidx = 0; sidx < nsub; ++sidx) {
    const dim3 grid = subs[sidx].grid;
    const bool bc = subs[sidx].bc;
    const int nb = subs[sidx].nb;

  /* Tiles only the mid-shape rule above picks (not in kWmma, so the LUT,
   * the online sweep and HIPDNN_EP_GEMM_WMMA_CFG never reach them); NT only:
   * the NN B staging needs a power-of-two BN. */
#define GEMM_WMMA_TRY_NT(BM, BN, WM, WN, BK)                                   \
  if (c.bm == BM && c.bn == BN && c.wt_m == WM && c.wt_n == WN &&              \
      c.bk == BK) {                                                            \
    if constexpr (TransB) {                                                    \
      GEMM_WMMA_ONE(BM, BN, WM, WN, BK);                                       \
    }                                                                          \
  } else
  GEMM_WMMA_TRY_NT(96, 128, 3, 4, 16)
  if (biasepi && c.bm == 96) {
    if constexpr (TransB) {
      GEMM_WMMA_ONE_P(96, 96, 3, 3, 32, 9);
    }
  } else
  GEMM_WMMA_TRY_NT(96, 96, 3, 3, 32)
  if (lineb) {
    if constexpr (TransB) {
      if (biasepi)
        GEMM_WMMA_ONE_P(128, 128, 4, 2, 64, 13);
      else
        GEMM_WMMA_ONE_P(128, 128, 4, 2, 64, 5);
    }
  } else
  /* BK=64 tiles */
  GEMM_WMMA_TRY(64, 128, 2, 4, 64)
  GEMM_WMMA_TRY(64, 128, 2, 2, 64)
  GEMM_WMMA_TRY(128, 64, 2, 2, 64)
  GEMM_WMMA_TRY(64, 64, 2, 2, 64)
  GEMM_WMMA_TRY(128, 128, 2, 4, 64)
  GEMM_WMMA_TRY(128, 128, 2, 2, 64)
  GEMM_WMMA_TRY(128, 128, 4, 2, 64)
  GEMM_WMMA_TRY(256, 128, 4, 2, 64)
  GEMM_WMMA_TRY(128, 256, 4, 2, 64)
  GEMM_WMMA_TRY(128, 64, 4, 2, 64)
  /* BK=32 WT2x2 */
  GEMM_WMMA_TRY(64, 64, 2, 2, 32)
  GEMM_WMMA_TRY(64, 128, 2, 2, 32)
  GEMM_WMMA_TRY(128, 64, 2, 2, 32)
  GEMM_WMMA_TRY(128, 128, 2, 2, 32)
  GEMM_WMMA_TRY(128, 256, 2, 2, 32)
  GEMM_WMMA_TRY(64, 256, 2, 2, 32)
  /* BK=32 WT with N=1 (small N) */
  GEMM_WMMA_TRY(64, 32, 2, 1, 32)
  GEMM_WMMA_TRY(128, 32, 2, 1, 32)
  /* BK=32 WT_N=4 */
  GEMM_WMMA_TRY_PGR(128, 128, 2, 4)
  GEMM_WMMA_TRY(64, 128, 2, 4, 32)
  GEMM_WMMA_TRY(128, 256, 2, 4, 32)
  /* BK=32 WT_M=4 (high Matrix-Core:LDS ratio) */
  GEMM_WMMA_TRY_PGR(256, 128, 4, 2)
  GEMM_WMMA_TRY(128, 128, 4, 4, 32)
  GEMM_WMMA_TRY_PGR(128, 128, 4, 2)
  GEMM_WMMA_TRY_PGR(128, 256, 4, 2)
  GEMM_WMMA_TRY(256, 128, 4, 4, 32)
  GEMM_WMMA_TRY(128, 64, 4, 2, 32)
  GEMM_WMMA_TRY(128, 128, 4, 4, 16)
  /* BK=16 (K%16==0, K%32!=0) */
  GEMM_WMMA_TRY(256, 128, 4, 2, 16)
  GEMM_WMMA_TRY(128, 128, 2, 4, 16)
  GEMM_WMMA_TRY(64, 128, 2, 4, 16)
  {
    GEMM_WMMA_ONE(128, 128, 2, 2, 32);
  }
  } /* sub-launch loop */
#undef GEMM_WMMA_TRY_NT
#undef GEMM_WMMA_TRY_PGR
#undef GEMM_WMMA_TRY
#undef GEMM_WMMA_ONE_P
#undef GEMM_WMMA_ONE

}

static std::unordered_map<std::string, int> g_wmma_cache;

template <typename T, bool TransB>
static int tuneWmma(hipStream_t s, const T *A, const T *B, const T *C, T *Y,
                     int M, int N, int K, float alpha, float beta, int cDim0,
                     int cDim1, int ldb) {
  /* D1 fix: this key used to be sizeof(T), so an fp16 GEMM and a bf16 GEMM on
   * the same shape collided (both 2 bytes) and the SECOND dtype to run in a
   * process silently reused the FIRST's config -- resolve() was never called
   * for it. Stage E.4 pins this with a same-process fp16-then-bf16 regression
   * test. */
  std::string key = std::string(TransB ? "tb1_" : "tb0_") +
                    hipdnn_ep::common::toString(gemmDType<T>()) + "_" +
                    std::to_string(M) + "_" + std::to_string(N) + "_" +
                    std::to_string(K);
  {
    std::lock_guard<std::mutex> lk(g_tune_mu);
    auto it = g_wmma_cache.find(key);
    if (it != g_wmma_cache.end())
      return it->second;
  }

  int best = pickWmmaHeuristic(M, N, K);
  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG", -1);
  const int lut = gemmLutWmmaConfig<T>(M, N, K, TransB ? 1 : 0);
  if (forced >= 0 && forced < kNumWmma && wmmaApplies(kWmma[forced], M, N, K)) {
    best = forced;
  } else if (lut >= 0) {
    /* Offline LUT hit: skip the sweep. Written into the in-process map below
     * so the lookup happens once per shape per process. */
    best = lut;
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] wmma LUT hit M=%d N=%d K=%d tb=%d -> cfg[%d] %dx%d wt=%dx%d "
        "sw=%d sk=%d bk=%d\n",
        M, N, K, (int)TransB, best, kWmma[best].bm, kWmma[best].bn,
        kWmma[best].wt_m, kWmma[best].wt_n, kWmma[best].swizzle_n,
        kWmma[best].split_k, kWmma[best].bk);
  } else if (gemmLutLookupEnabled()) {
    /* Default (lookup) mode, LUT miss: use the static occupancy heuristic, NOT
     * the on-device sweep. On this gfx1151 APU compute-bound tiles throttle
     * hard, and the throttling correlates with tile size, so a runtime sweep
     * systematically mis-ranks (it picked 128x64 @ ~4.5 TFLOP/s over 128x128
     * WT2x4 @ ~9.3 TFLOP/s on N=4096 K=14336). The heuristic already returns
     * the WT2x4 workhorse for these shapes. The sweep is available only in
     * `online` mode, whose purpose is to generate the offline LUT under
     * controlled conditions -- see HIPDNN_GEMM_AUTOTUNE_MODE. */
    best = pickWmmaHeuristic(M, N, K);
    /* NN (B = [K, N]) has no LUT points and the TN-tuned heuristic picks the
     * wide 128x128 / 128x256 BK32 tiles, which re-read the strided B rows per
     * K-step. The 4-wave 128x64 BK64 tile (cfg 128x64 wt=4x2 bk=64) reads a
     * 64-row x 128 B B-panel per K-tile instead; forced A/B on gfx1151, normal
     * mode, slower of 2 rounds: 2048x4096x4096 fp16 3.67 -> 2.03 ms, bf16
     * 3.67 -> 2.07; 128x4096x4096 fp16 0.212 -> 0.165, bf16 0.215 -> 0.149.
     * Drop once the LUT carries NN points. */
    if (!TransB && M >= 128 && (K % 64) == 0) {
      const int nn = findWmmaCfg(128, 64, 4, 2, 8, 1, 64);
      if (wmmaApplies(kWmma[nn], M, N, K))
        best = nn;
    }
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] wmma heuristic M=%d N=%d K=%d tb=%d -> cfg[%d] %dx%d wt=%dx%d "
        "sw=%d sk=%d bk=%d\n",
        M, N, K, (int)TransB, best, kWmma[best].bm, kWmma[best].bn,
        kWmma[best].wt_m, kWmma[best].wt_n, kWmma[best].swizzle_n,
        kWmma[best].split_k, kWmma[best].bk);
  } else if (gemmAutotuneOn() && gemmSweepSafe(C, Y, beta)) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *out = Y;
    /* Warm the clocks to a sustained state before timing: compute-bound tiles
     * on gfx1151 boost for the first few launches, so a short sweep would rank
     * configs on boost clocks that do not hold under a real inference loop. */
    const int kInitWarm = gemmEnvInt("HIPDNN_GEMM_TUNE_INITWARM", 40);
    const int kCoarseWarm = gemmEnvInt("HIPDNN_GEMM_TUNE_CWARM", 8);
    for (int w = 0; w < kInitWarm; ++w)
      launchWmma<T, TransB>(best, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                            cDim1, ldb);
    hipStreamSynchronize(s);

    float times[64];
    int ids[64];
    int nc = 0;
    for (int cid = 0; cid < kNumWmma; ++cid) {
      if (!wmmaApplies(kWmma[cid], M, N, K))
        continue;
      /* Coarse pre-filter to pick the top-N finalists. A short per-candidate
       * warmup (not just one launch) keeps the prune from dropping a good tile
       * that was measured on the previous tile's hot clocks; the finalists are
       * then measured faithfully below. */
      for (int w = 0; w < kCoarseWarm; ++w)
        launchWmma<T, TransB>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                              cDim1, ldb);
      hipEventRecord(e0, s);
      for (int i = 0; i < 3; ++i)
        launchWmma<T, TransB>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                             cDim1, ldb);
      hipEventRecord(e1, s);
      if (hipEventSynchronize(e1) != hipSuccess)
        continue;
      float ms = 0.f;
      hipEventElapsedTime(&ms, e0, e1);
      times[nc] = ms / 3.f;
      ids[nc] = cid;
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] wmma cfg[%d] %dx%d wt=%dx%d sw=%d sk=%d bk=%d : %.4f ms\n",
          cid, kWmma[cid].bm, kWmma[cid].bn, kWmma[cid].wt_m, kWmma[cid].wt_n,
          kWmma[cid].swizzle_n, kWmma[cid].split_k, kWmma[cid].bk, times[nc]);
      ++nc;
    }
    int order[64];
    for (int i = 0; i < nc; ++i)
      order[i] = i;
    std::sort(order, order + nc,
              [&](int a, int b) { return times[a] < times[b]; });
    const int n2 = nc < 12 ? nc : 12;

    /* Faithful per-candidate measurement -- mirror the production/hipBLAS fair
     * bench so both sides' numbers are comparable.
     *
     * The earlier round-robin approach gave each candidate only ONE warmup
     * launch before timing, so a candidate inherited the PREVIOUS tile's
     * thermal/clock state instead of reaching its own. On this APU compute-bound
     * tiles throttle by ~2x, and the throttle correlates with tile size, so
     * timing a tile on the wrong thermal state mis-ranked it (it "showed" cfg30
     * beating cfg5 at 4608^2 and cfg33 beating cfg5 at the vision mlp-down,
     * whereas the fair sustained bench has cfg5 tied-or-ahead on both).
     *
     * Fix: warm EACH candidate to its OWN sustained state (a per-candidate
     * warmup burst re-equilibrates the part to that tile's power draw), then
     * time the peak of several sustained reps (best-of-reps, like both the
     * final hip_gemm number and the hipBLAS algo pick). Now the autotune's per
     * config number is the same "peak sustained" quantity we compare against
     * hipBLAS, so the ranking is trustworthy. */
    /* Env-tunable so bulk offline LUT generation can trade a little ranking
     * robustness for throughput (hundreds of shapes). Defaults keep the full
     * faithful measurement. Lighter values (e.g. WARM=12 REPS=2 ITERS=8) still
     * reach steady state on this APU -- validate the winner is unchanged before
     * trusting a lighter setting on a new shape class. */
    const int kWarm = gemmEnvInt("HIPDNN_GEMM_TUNE_WARM", 25);
    const int kReps = gemmEnvInt("HIPDNN_GEMM_TUNE_REPS", 3);
    const int kIters = gemmEnvInt("HIPDNN_GEMM_TUNE_ITERS", 10);
    float best_min[64];
    for (int r = 0; r < n2; ++r) {
      const int cid = ids[order[r]];
      for (int w = 0; w < kWarm; ++w)
        launchWmma<T, TransB>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                              cDim1, ldb);
      hipStreamSynchronize(s);
      float cand_min = 1e30f;
      for (int rep = 0; rep < kReps; ++rep) {
        hipEventRecord(e0, s);
        for (int i = 0; i < kIters; ++i)
          launchWmma<T, TransB>(cid, s, A, B, C, out, M, N, K, alpha, beta,
                                cDim0, cDim1, ldb);
        hipEventRecord(e1, s);
        if (hipEventSynchronize(e1) != hipSuccess)
          continue;
        float ms = 0.f;
        hipEventElapsedTime(&ms, e0, e1);
        ms /= kIters;
        if (ms < cand_min)
          cand_min = ms;
      }
      best_min[r] = cand_min;
    }
    float best_ms = 1e30f;
    for (int r = 0; r < n2; ++r) {
      const int cid = ids[order[r]];
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] wmma retune cfg[%d] : %.4f ms (peak)\n",
                               cid, best_min[r]);
      if (best_min[r] < best_ms) {
        best_ms = best_min[r];
        best = cid;
      }
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] wmma autotune M=%d N=%d K=%d tb=%d -> cfg[%d] %dx%d wt=%dx%d "
        "sw=%d sk=%d bk=%d\n",
        M, N, K, (int)TransB, best, kWmma[best].bm, kWmma[best].bn,
        kWmma[best].wt_m, kWmma[best].wt_n, kWmma[best].swizzle_n,
        kWmma[best].split_k, kWmma[best].bk);
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_wmma_cache[key] = best;
  return best;
}

template <typename T>
static int launchWmmaGemm(hipStream_t s, const T *A, const T *B, const T *C,
                           T *Y, int M, int N, int K, int transB, float alpha,
                           float beta, int cDim0, int cDim1, int ldb,
                           int lda = -1) {
  (void)hipGetLastError();
  if (transB) {
    int id = tuneWmma<T, true>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                               cDim1, ldb);
    launchWmma<T, true>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
                        ldb, lda);
  } else {
    int id = tuneWmma<T, false>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                                cDim1, ldb);
    launchWmma<T, false>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
                         ldb, lda);
  }
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* transA=1 fp16/bf16, M >= 16: WMMA with A stored [K, M]. GemmWmmaImpl's
 * TransA staging transposes A into the usual [m][k] LDS tile with packed
 * stores; B is either layout. A small fixed tile ladder (no LUT entry exists
 * for transA): the largest tile that still puts >= ~40 workgroups on the
 * 40-CU part, then smaller ones. HIPDNN_EP_GEMM_WMMA_TA_CFG=<idx> forces a
 * table entry (A/B only). */
static constexpr WmmaCfg kWmmaTa[] = {
    {128, 128, 2, 4, 8, 1, 32},
    {64, 128, 2, 2, 8, 1, 32},
    {128, 64, 2, 2, 8, 1, 32},
    {64, 64, 2, 2, 8, 1, 32},
};
static constexpr int kNumWmmaTa =
    static_cast<int>(sizeof(kWmmaTa) / sizeof(kWmmaTa[0]));

static int pickWmmaTa(int M, int N) {
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG", -1);
  if (forced >= 0 && forced < kNumWmmaTa)
    return forced;
  for (int i = 0; i < kNumWmmaTa - 1; ++i) {
    const WmmaCfg &c = kWmmaTa[i];
    if (((M + c.bm - 1) / c.bm) * ((N + c.bn - 1) / c.bn) >= 40)
      return i;
  }
  return kNumWmmaTa - 1;
}

template <typename T, bool TransB>
static void launchWmmaTa(int id, hipStream_t s, const T *A, const T *B,
                         const T *C, T *Y, int M, int N, int K, float alpha,
                         float beta, int cDim0, int cDim1) {
  const WmmaCfg c = kWmmaTa[id];
  const int th = (c.bm / (c.wt_m * 16)) * (c.bn / (c.wt_n * 16)) * 32;
  const int stag = gemmStaggerParam(c.bk, static_cast<int>(sizeof(T)), TransB);
  /* The BC=false fast path needs whole tiles and 16-byte-aligned A rows
   * (lda = M, TA_MC <= 8 halves). */
  const bool bc = !(M % c.bm == 0 && N % c.bn == 0 && K % c.bk == 0 &&
                    M % 8 == 0);
  const dim3 grid((N + c.bn - 1) / c.bn, (M + c.bm - 1) / c.bm, 1);
  const dim3 block(th);
#define GEMM_WMMA_TA_ONE(BM, BN, WM, WN)                                       \
  if (c.bm == BM && c.bn == BN && c.wt_m == WM && c.wt_n == WN) {              \
    if (bc)                                                                    \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, 32,      \
                                           true, true, 1, true>),              \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          c.swizzle_n, nullptr, K, 0, M, stag);                                \
    else                                                                       \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, 32,      \
                                           false, true, 1, true>),             \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          c.swizzle_n, nullptr, K, 0, M, stag);                                \
  } else
  GEMM_WMMA_TA_ONE(128, 128, 2, 4)
  GEMM_WMMA_TA_ONE(64, 128, 2, 2)
  GEMM_WMMA_TA_ONE(128, 64, 2, 2)
  GEMM_WMMA_TA_ONE(64, 64, 2, 2) {}
#undef GEMM_WMMA_TA_ONE
}

template <typename T>
static int launchWmmaGemmTa(hipStream_t s, const T *A, const T *B, const T *C,
                            T *Y, int M, int N, int K, int transB, float alpha,
                            float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
  const int id = pickWmmaTa(M, N);
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] wmma transA M=%d N=%d K=%d tb=%d -> ta-cfg[%d] %dx%d wt=%dx%d\n",
      M, N, K, transB, id, kWmmaTa[id].bm, kWmmaTa[id].bn, kWmmaTa[id].wt_m,
      kWmmaTa[id].wt_n);
  if (transB)
    launchWmmaTa<T, true>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                          cDim1);
  else
    launchWmmaTa<T, false>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                           cDim1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* =========================================================================
 * fp32 / fp64 tiled FMA (no WMMA). Also serves wave64 fp16/bf16 (no WMMA on
 * that hardware). Templated on tile geometry (Stage A8): BM/BN/BK/TM/TN pick
 * the tile, THREADS the block size (THREADS == (BM/TM)*(BN/TN)). Covers all
 * four (transA, transB) combinations. Accumulation is always float regardless
 * of T (matches the original untemplated kernel this replaces).
 * ========================================================================= */

template <typename T, bool TransA, bool TransB, int BM, int BN, int BK,
          int TM, int TN, int THREADS>
__global__ void __launch_bounds__(THREADS)
    gemm_tiled_fma_kernel(const T *__restrict__ A, const T *__restrict__ B,
                         const T *__restrict__ C, T *__restrict__ Y, int M,
                         int N, int K, float alpha, float beta, int cDim0,
                         int cDim1) {
  constexpr int THREADS_X = BN / TN;
  constexpr int THREADS_Y = BM / TM;
  static_assert(THREADS_X * THREADS_Y == THREADS, "tile/thread mismatch");
  constexpr int A_PER_THREAD = (BM * BK) / THREADS;
  constexpr int B_PER_THREAD = (BK * BN) / THREADS;
  static_assert(A_PER_THREAD >= 1 && (BM * BK) % THREADS == 0, "A load tile");
  static_assert(B_PER_THREAD >= 1 && (BK * BN) % THREADS == 0, "B load tile");

  __shared__ float smA[BM][BK + 1];
  __shared__ float smB[BK][BN + 1];

  const int tx = threadIdx.x % THREADS_X;
  const int ty = threadIdx.x / THREADS_X;
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  const int lda = TransA ? M : K;
  const int ldb = TransB ? K : N;

  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int i = 0; i < A_PER_THREAD; ++i) {
      const int idx = threadIdx.x + i * THREADS;
      const int ar = idx / BK;
      const int ak = idx % BK;
      float av = 0.f;
      const int gr = row0 + ar;
      const int gk = k0 + ak;
      if (gr < M && gk < K) {
        av = GemmElem<T>::tof(TransA ? A[static_cast<size_t>(gk) * lda + gr]
                                      : A[static_cast<size_t>(gr) * lda + gk]);
      }
      smA[ar][ak] = av;
    }
#pragma unroll
    for (int i = 0; i < B_PER_THREAD; ++i) {
      const int idx = threadIdx.x + i * THREADS;
      const int bk = idx / BN;
      const int bn = idx % BN;
      float bv = 0.f;
      const int gk = k0 + bk;
      const int gc = col0 + bn;
      if (gk < K && gc < N) {
        bv = GemmElem<T>::tof(TransB ? B[static_cast<size_t>(gc) * ldb + gk]
                                      : B[static_cast<size_t>(gk) * ldb + gc]);
      }
      smB[bk][bn] = bv;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const float av = smA[ty * TM + i][kk];
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(av, smB[kk][tx * TN + j], acc[i][j]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = row0 + ty * TM + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = col0 + tx * TN + j;
      if (r < M && c < N)
        gemmStoreY(Y, C, r, c, N, cDim0, cDim1, acc[i][j], alpha, beta);
    }
  }
}

struct TiledFmaCfg {
  int bm, bn, bk, tm, tn, threads;
};

/* Config table (Stage A8). Index 0 is the original hardcoded 64x64x16/4x4/256
 * geometry, kept as entry 0 so the pre-LUT behaviour is exactly reproducible
 * (HIPDNN_EP_GEMM_TILEDFMA_CFG=0 or any miss/no-table path). The other three
 * give real occupancy/traffic tradeoffs to tune between (a smaller 32x32 tile
 * for small/grid-starved shapes, two rectangular 512-thread tiles). Every
 * entry needs a matching GEMM_TILEDFMA_TRY dispatch branch below or it
 * silently falls through to the default (plan.md R10). */
static constexpr TiledFmaCfg kTiledFma[] = {
    {64, 64, 16, 4, 4, 256},
    {128, 64, 16, 4, 4, 512},
    {64, 128, 16, 4, 4, 512},
    {32, 32, 16, 2, 2, 256},
};
static constexpr int kNumTiledFma =
    static_cast<int>(sizeof(kTiledFma) / sizeof(kTiledFma[0]));

template <typename T>
static int launchTiledFma(int id, hipStream_t s, const T *A, const T *B,
                          const T *C, T *Y, int M, int N, int K, int transA,
                          int transB, float alpha, float beta, int cDim0,
                          int cDim1) {
  (void)hipGetLastError();
  const TiledFmaCfg &c = kTiledFma[id];
  dim3 grid((N + c.bn - 1) / c.bn, (M + c.bm - 1) / c.bm);
  dim3 block(c.threads);

#define GEMM_TILEDFMA_ONE(BM, BN, BK, TM, TN, THR)                            \
  do {                                                                       \
    if (!transA && !transB)                                                  \
      hipLaunchKernelGGL(                                                    \
          HIP_KERNEL_NAME(                                                   \
              gemm_tiled_fma_kernel<T, false, false, BM, BN, BK, TM, TN,     \
                                    THR>),                                   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,        \
          cDim1);                                                            \
    else if (!transA && transB)                                              \
      hipLaunchKernelGGL(                                                    \
          HIP_KERNEL_NAME(                                                   \
              gemm_tiled_fma_kernel<T, false, true, BM, BN, BK, TM, TN,      \
                                    THR>),                                   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,        \
          cDim1);                                                            \
    else if (transA && !transB)                                              \
      hipLaunchKernelGGL(                                                    \
          HIP_KERNEL_NAME(                                                   \
              gemm_tiled_fma_kernel<T, true, false, BM, BN, BK, TM, TN,      \
                                    THR>),                                   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,        \
          cDim1);                                                            \
    else                                                                     \
      hipLaunchKernelGGL(                                                    \
          HIP_KERNEL_NAME(                                                   \
              gemm_tiled_fma_kernel<T, true, true, BM, BN, BK, TM, TN,       \
                                    THR>),                                   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,        \
          cDim1);                                                            \
  } while (0)

#define GEMM_TILEDFMA_TRY(BM, BN, BK, TM, TN, THR)                            \
  if (c.bm == BM && c.bn == BN && c.bk == BK && c.tm == TM && c.tn == TN &&   \
      c.threads == THR) {                                                    \
    GEMM_TILEDFMA_ONE(BM, BN, BK, TM, TN, THR);                              \
  } else

  GEMM_TILEDFMA_TRY(64, 64, 16, 4, 4, 256)
  GEMM_TILEDFMA_TRY(128, 64, 16, 4, 4, 512)
  GEMM_TILEDFMA_TRY(64, 128, 16, 4, 4, 512)
  GEMM_TILEDFMA_TRY(32, 32, 16, 2, 2, 256)
  {
    /* Unreachable for any id in kTiledFma[]; kept as a safe, explicit default
     * (never a silent wrong-grid launch) if a stale LUT geometry index ever
     * reached here -- it cannot, since launchTiledFma is only ever called
     * with an id from kTiledFma or from tiledFmaConfigFromGeom(), which
     * returns -1 (never dispatched) for anything not in this exact list. */
    GEMM_TILEDFMA_ONE(64, 64, 16, 4, 4, 256);
  }
#undef GEMM_TILEDFMA_TRY
#undef GEMM_TILEDFMA_ONE

  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* Geometry <-> config-index bridge for the TiledFma LUT (Stage A8). */
static int tiledFmaConfigFromGeom(const gemm_lut::TiledFmaAnswer &g) {
  for (int cid = 0; cid < kNumTiledFma; ++cid) {
    const TiledFmaCfg &c = kTiledFma[cid];
    if (c.bm == g.bm && c.bn == g.bn && c.bk == g.bk && c.tm == g.tm &&
        c.tn == g.tn && c.threads == g.threads)
      return cid;
  }
  return -1;
}
static bool tiledFmaLutValidator(void *, const gemm_lut::TiledFmaAnswer &g) {
  return tiledFmaConfigFromGeom(g) >= 0;
}
template <typename T>
static int gemmLutTiledFmaConfig(int M, int N, int K, int transB) {
  if (!gemmLutLookupEnabled())
    return -1;
  gemm_lut::Request req;
  req.kind = gemm_lut::Kind::TiledFma;
  req.act_dtype = req.wts_dtype = req.out_dtype = gemmDType<T>();
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  const gemm_lut::Result r = gemm_lut::resolve(req, nullptr, nullptr,
                                               tiledFmaLutValidator, nullptr);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return tiledFmaConfigFromGeom(r.tiled_fma);
}

static std::unordered_map<std::string, int> g_tiled_fma_cache;

/* tuneTiledFma follows the same in-process-map -> LUT -> heuristic-default ->
 * (online mode only) settle+peak sweep dispatch every other tuner here uses.
 * transA is folded into the cache key (the FMA kernel has real transA support,
 * unlike the WMMA/GEMV paths which route transA into launchSmallM before
 * reaching here for fp16/bf16 -- only fp32/fp64 can reach tuneTiledFma with
 * transA=1). transA=1 is NOT looked up in the LUT or online-tuned this round
 * (plan.md's Stage B matrix only measures transA=0; transposed-A prefill does
 * not occur in the real models this table targets): it always keeps `best`
 * at its cfg-0 default, i.e. exactly this kernel's pre-LUT behaviour, so a
 * transA=1 caller is never worse off than before the LUT existed. */
template <typename T>
static int tuneTiledFma(hipStream_t s, const T *A, const T *B, const T *C,
                        T *Y, int M, int N, int K, int transA, int transB,
                        float alpha, float beta, int cDim0, int cDim1) {
  std::string key = std::string(transA ? "ta1_" : "ta0_") +
                    (transB ? "tb1_" : "tb0_") +
                    hipdnn_ep::common::toString(gemmDType<T>()) + "_" +
                    std::to_string(M) + "_" + std::to_string(N) + "_" +
                    std::to_string(K);
  {
    std::lock_guard<std::mutex> lk(g_tune_mu);
    auto it = g_tiled_fma_cache.find(key);
    if (it != g_tiled_fma_cache.end())
      return it->second;
  }

  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG", -1);
  int best = 0; /* 64x64x16 4x4, the original hardcoded geometry */
  if (forced >= 0 && forced < kNumTiledFma) {
    best = forced;
  } else if (!transA) {
    if (int lut = gemmLutTiledFmaConfig<T>(M, N, K, transB); lut >= 0) {
      best = lut;
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] tiledfma LUT hit M=%d N=%d K=%d ta=%d tb=%d -> cfg[%d] "
          "%dx%d bk=%d tm=%d tn=%d th=%d\n",
          M, N, K, transA, transB, best, kTiledFma[best].bm,
          kTiledFma[best].bn, kTiledFma[best].bk, kTiledFma[best].tm,
          kTiledFma[best].tn, kTiledFma[best].threads);
    } else if (gemmLutLookupEnabled()) {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] tiledfma default M=%d N=%d K=%d ta=%d tb=%d -> cfg[%d] "
          "%dx%d bk=%d tm=%d tn=%d th=%d\n",
          M, N, K, transA, transB, best, kTiledFma[best].bm,
          kTiledFma[best].bn, kTiledFma[best].bk, kTiledFma[best].tm,
          kTiledFma[best].tn, kTiledFma[best].threads);
    } else if (gemmAutotuneOn() && gemmSweepSafe(C, Y, beta)) {
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      T *out = Y;

      const int kInitWarm = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_INITWARM", 40);
      const int kWarm = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_WARM", 5);
      const int kIters = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_ITERS", 5);

      gemmSettleClock(
          s, e0, e1,
          [&] {
            launchTiledFma<T>(best, s, A, B, C, out, M, N, K, transA, transB,
                              alpha, beta, cDim0, cDim1);
          },
          kInitWarm);

      float best_ms = 1e30f;
      for (int cid = 0; cid < kNumTiledFma; ++cid) {
        const float ms = gemmTimeConfigPeak(
            s, e0, e1,
            [&] {
              launchTiledFma<T>(cid, s, A, B, C, out, M, N, K, transA, transB,
                                alpha, beta, cDim0, cDim1);
            },
            kWarm, kIters);
        CUSTOM_KERNELS_DEBUG_LOG(
            "[gemm] tiledfma cfg[%d] %dx%d bk=%d tm=%d tn=%d th=%d : %.4f ms "
            "(peak)\n",
            cid, kTiledFma[cid].bm, kTiledFma[cid].bn, kTiledFma[cid].bk,
            kTiledFma[cid].tm, kTiledFma[cid].tn, kTiledFma[cid].threads, ms);
        if (ms < best_ms) {
          best_ms = ms;
          best = cid;
        }
      }
      hipEventDestroy(e0);
      hipEventDestroy(e1);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] tiledfma autotune M=%d N=%d K=%d ta=%d tb=%d -> cfg[%d] "
          "%dx%d bk=%d tm=%d tn=%d th=%d\n",
          M, N, K, transA, transB, best, kTiledFma[best].bm,
          kTiledFma[best].bn, kTiledFma[best].bk, kTiledFma[best].tm,
          kTiledFma[best].tn, kTiledFma[best].threads);
    }
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_tiled_fma_cache[key] = best;
  return best;
}

/* =========================================================================
 * fp32 tile GEMM (NT / NN, M >= 16, no transA) -- the fp32 path on wave32.
 *
 * gemm_tiled_fma_kernel puts every 4-byte global load behind its own bounds
 * branch and waits for it before the LDS store, so a wave has one load in
 * flight and a K tile costs six memory round trips. Here each thread loads
 * 16-byte pieces of the next K tile with no branches -- rows past M or N read
 * a clamped valid row and their outputs are masked -- and all of them are in
 * flight while the current tile is computed out of the other half of a
 * double-buffered LDS tile: one barrier per K tile.
 *
 * LDS holds both tiles k-major ([BK][BM+4], [BK][BN+4]); A and the NT B are
 * transposed by the store. A loader lane pair reads 32 contiguous bytes of
 * one row, so a wave's 4-byte transposed stores hit 16 rows x 2 k and no two
 * share a bank. A thread owns TM x TN outputs as (TM/4) x (TN/4) blocks of
 * 4x4 spaced BM/(TM/4) rows and BN/(TN/4) columns apart: for each k,
 * neighbouring threads read neighbouring 16-byte pieces of the B row
 * (ds_load_b128 without bank conflicts) and the A pieces are broadcasts. At
 * 8x8 that is 64 FMAs per 64 bytes of LDS reads. Outputs go out as 16-byte
 * stores.
 *
 * K runs in whole BK tiles plus at most one zero-filled tail tile. Needs
 * K % 4 == 0, N % 4 == 0 for NN and 16-byte aligned A and B
 * (f32TileEligible). The accumulation order differs from
 * gemm_tiled_fma_kernel, so the two do not give bit-identical results.
 * ========================================================================= */
/* Native vector type: HIP's float4 struct keeps the register arrays below in
 * scratch (allocas SROA cannot split). */
typedef float gemmF4 __attribute__((ext_vector_type(4)));
template <bool TB, int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ void __launch_bounds__(THREADS)
    gemm_f32_tile_kernel(const float *__restrict__ A,
                         const float *__restrict__ B,
                         const float *__restrict__ C, float *__restrict__ Y,
                         int M, int N, int K, float alpha, float beta,
                         int cDim0, int cDim1, int yvec, int stag,
                         int swz) {
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;
  static_assert(TX * TY == THREADS, "thread tile");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "4x4 output blocks");
  static_assert(BK % 8 == 0 && BM >= 16 && (!TB || BN >= 16),
                "loader lane pairs cover 16 rows x 2 k-pieces");
  constexpr int GM = TM / 4;
  constexpr int GN = TN / 4;
  constexpr int SM = BM / GM;
  constexpr int SN = BN / GN;
  constexpr int LDA = BM + 4;
  constexpr int LDB = BN + 4;
  constexpr int KQ = BK / 4; /* 4-float k pieces per row */
  constexpr int A_V4 = BM * KQ / THREADS;
  static_assert(A_V4 >= 1 && A_V4 * THREADS == BM * KQ, "A load split");
  constexpr int BQ = TB ? KQ : BN / 4;
  constexpr int B_ROWS = TB ? BN : BK;
  constexpr int B_V4 = B_ROWS * BQ / THREADS;
  static_assert(B_V4 >= 1 && B_V4 * THREADS == B_ROWS * BQ, "B load split");

  __shared__ __align__(16) float smA[2][BK][LDA];
  __shared__ __align__(16) float smB[2][BK][LDB];

  const int tid = threadIdx.x;
  const int tx = tid % TX;
  const int ty = tid / TX;
  int by = blockIdx.y, bx = blockIdx.x;
  if (swz != 0)
    gemmSwizzleBlock(gridDim.x, gridDim.y, swz, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  /* k-contiguous operand piece idx -> (row, k): lane pairs take two adjacent
   * k pieces of one row, 16 rows per wave. */
  auto pairPiece = [](int idx, int rows, int &r, int &k)
      __attribute__((always_inline)) {
    r = (idx >> 1) % rows;
    k = ((idx & 1) | ((idx / (2 * rows)) << 1)) * 4;
  };

  const float *a_src[A_V4];
  int a_r[A_V4], a_k[A_V4];
#pragma unroll
  for (int v = 0; v < A_V4; ++v) {
    pairPiece(tid + v * THREADS, BM, a_r[v], a_k[v]);
    const int gr = min(row0 + a_r[v], M - 1);
    a_src[v] = A + static_cast<size_t>(gr) * K + a_k[v];
  }
  /* NT: b_r is the tile's n row and b_c its k offset. NN: b_r is the k row
   * and b_c the n offset (16-byte pieces along N, stored as they are). */
  const float *b_src[B_V4];
  int b_r[B_V4], b_c[B_V4];
#pragma unroll
  for (int v = 0; v < B_V4; ++v) {
    const int idx = tid + v * THREADS;
    if constexpr (TB) {
      pairPiece(idx, BN, b_r[v], b_c[v]);
      const int gn = min(col0 + b_r[v], N - 1);
      b_src[v] = B + static_cast<size_t>(gn) * K + b_c[v];
    } else {
      b_r[v] = idx / BQ;
      b_c[v] = (idx % BQ) * 4;
      const int gn = min(col0 + b_c[v], N - 4);
      b_src[v] = B + static_cast<size_t>(b_r[v]) * N + gn;
    }
  }

  gemmF4 ra[A_V4], rb[B_V4];
  /* TAIL: the tile may run past K. K % 4 == 0, so a 4-float k piece is either
   * all inside K or all outside. */
  auto load = [&](int k0, auto tail) __attribute__((always_inline)) {
    constexpr bool TAIL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int v = 0; v < A_V4; ++v)
      ra[v] = (!TAIL || k0 + a_k[v] < K)
                  ? *reinterpret_cast<const gemmF4 *>(a_src[v] + k0)
                  : zero;
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB)
        rb[v] = (!TAIL || k0 + b_c[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(b_src[v] + k0)
                    : zero;
      else
        rb[v] = (!TAIL || k0 + b_r[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(
                          b_src[v] + static_cast<size_t>(k0) * N)
                    : zero;
    }
  };

  auto store = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int v = 0; v < A_V4; ++v) {
      smA[buf][a_k[v] + 0][a_r[v]] = ra[v].x;
      smA[buf][a_k[v] + 1][a_r[v]] = ra[v].y;
      smA[buf][a_k[v] + 2][a_r[v]] = ra[v].z;
      smA[buf][a_k[v] + 3][a_r[v]] = ra[v].w;
    }
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB) {
        smB[buf][b_c[v] + 0][b_r[v]] = rb[v].x;
        smB[buf][b_c[v] + 1][b_r[v]] = rb[v].y;
        smB[buf][b_c[v] + 2][b_r[v]] = rb[v].z;
        smB[buf][b_c[v] + 3][b_r[v]] = rb[v].w;
      } else {
        *reinterpret_cast<gemmF4 *>(&smB[buf][b_r[v]][b_c[v]]) = rb[v];
      }
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  auto compute = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int g = 0; g < GM; ++g) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smA[buf][kk][g * SM + ty * 4]);
        a[g * 4 + 0] = v.x;
        a[g * 4 + 1] = v.y;
        a[g * 4 + 2] = v.z;
        a[g * 4 + 3] = v.w;
      }
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smB[buf][kk][h * SN + tx * 4]);
        b[h * 4 + 0] = v.x;
        b[h * 4 + 1] = v.y;
        b[h * 4 + 2] = v.z;
        b[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      GEMM_SCHED_FENCE();
    }
    /* Keep the FMAs here: without this the compiler sinks the last tile's
     * FMAs into the epilogue branches and hoists all its LDS reads (256 VGPRs
     * and scratch). */
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
        __asm__ volatile("" : "+v"(acc[i][j]));
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int nfull = K / BK;
  const int ntiles = nfull + ((K % BK) != 0 ? 1 : 0);
  /* StaggerU over the whole tiles (stag = tiles per step, 0 = off): dispatch
   * neighbours start their K loop at different offsets and wrap, so a K whose
   * row stride is a large power of two does not send every workgroup's rows
   * to the same memory channels at once. The tail tile always comes last. */
  const int shift =
      (stag > 0 && nfull > 1)
          ? ((static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x) & 31) *
             stag) %
                nfull
          : 0;
  auto kofs = [&](int i) __attribute__((always_inline)) {
    int kt = i + shift;
    if (kt >= nfull)
      kt -= nfull;
    return kt * BK;
  };
  if (nfull > 0)
    load(kofs(0), Full{});
  else
    load(0, Tail{});
  store(0);
  HIPDNN_LDS_BARRIER();
  /* Whole tiles first so the loop body is one straight-line load / compute /
   * store / barrier; the zero-filled tail tile and the last tile follow it. */
  int t = 0;
  for (; t + 1 < nfull; ++t) {
    load(kofs(t + 1), Full{});
    compute(t & 1);
    store((t + 1) & 1);
    HIPDNN_LDS_BARRIER();
  }
  if (t + 1 < ntiles) {
    load((t + 1) * BK, Tail{});
    compute(t & 1);
    store((t + 1) & 1);
    HIPDNN_LDS_BARRIER();
    ++t;
  }
  compute(t & 1);

  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f && yvec;
#pragma unroll
  for (int g = 0; g < GM; ++g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = row0 + g * SM + ty * 4 + i;
      if (r >= M)
        continue;
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const int c = col0 + h * SN + tx * 4;
        const int ai = g * 4 + i;
        if (fast && c + 4 <= N) {
          *reinterpret_cast<gemmF4 *>(&Y[static_cast<size_t>(r) * N + c]) =
              gemmF4{acc[ai][h * 4 + 0], acc[ai][h * 4 + 1],
                          acc[ai][h * 4 + 2], acc[ai][h * 4 + 3]};
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j)
            if (c + j < N)
              gemmStoreY(Y, C, r, c + j, N, cDim0, cDim1, acc[ai][h * 4 + j],
                         alpha, beta);
        }
      }
    }
  }
}

/* {id, BM, BN, BK, TM, TN, threads}. One X-macro feeds both the table and the
 * launch switch so they cannot drift apart. */
#define GEMM_F32_TILE_CFGS(X)                                                  \
  X(0, 128, 128, 16, 8, 8, 256)                                                \
  X(1, 64, 128, 16, 8, 8, 128)                                                 \
  X(2, 128, 64, 16, 8, 8, 128)                                                 \
  X(3, 64, 64, 16, 4, 8, 128)                                                  \
  X(4, 32, 64, 32, 4, 4, 128)                                                  \
  X(5, 16, 64, 32, 4, 4, 64)                                                   \
  X(6, 128, 64, 32, 8, 8, 128)

struct F32TileCfg {
  int bm, bn, bk, tm, tn, threads;
};
#define GEMM_F32_TILE_ENTRY(ID, BM, BN, BK, TM, TN, THR)                       \
  {BM, BN, BK, TM, TN, THR},
static constexpr F32TileCfg kF32Tile[] = {
    GEMM_F32_TILE_CFGS(GEMM_F32_TILE_ENTRY)};
#undef GEMM_F32_TILE_ENTRY
static constexpr int kNumF32Tile =
    static_cast<int>(sizeof(kF32Tile) / sizeof(kF32Tile[0]));

static bool f32TileEligible(const float *A, const float *B, int M, int N,
                            int K, int transA, int transB) {
  const bool aligned = ((reinterpret_cast<uintptr_t>(A) |
                         reinterpret_cast<uintptr_t>(B)) &
                        15) == 0;
  return !transA && M >= 16 && K % 4 == 0 && (transB || N % 4 == 0) &&
         aligned;
}

template <bool TB>
static void launchF32TileT(int id, hipStream_t s, const float *A,
                           const float *B, const float *C, float *Y, int M,
                           int N, int K, float alpha, float beta, int cDim0,
                           int cDim1) {
  const F32TileCfg &c = kF32Tile[id];
  const dim3 grid((N + c.bn - 1) / c.bn, (M + c.bm - 1) / c.bm);
  const int yvec =
      (N % 4) == 0 && (reinterpret_cast<uintptr_t>(Y) & 15) == 0 ? 1 : 0;
  /* StaggerU step of 256 bytes of K, NT only (the B rows are what alias). */
  static const bool stag_on = gemmEnvInt("HIPDNN_EP_GEMM_F32_STAGGER", 1) != 0;
  const int stag = (TB && stag_on) ? std::max(1, 64 / c.bk) : 0;
  /* Raster (NT, where the B rows alias): an M-first group of 8 M-tiles while
   * its A panels (8 * BM * K floats) fit 16 MB of the Infinity Cache, else
   * 8-wide N-first super-columns. bench3 exp5, fp32 NT 128x64 BK32, slower of
   * 2 rounds, plain order -> this: 512x4096^2 2.17 -> 1.64 ms, 2048x4096^2
   * 8.63 -> 6.62, 4096^3 17.4 -> 13.3, 1024x16384x3072 12.1 -> 9.81,
   * 2048x4096x14336 32.0 -> 28.5 (N-first; M-first 32.3). A 4-wide M-first
   * group when only that fits: 2048x4096x8192 15.5 -> 13.4. Short K keeps the
   * plain order (1576x3072x768: -3.6% M-first).
   * HIPDNN_EP_GEMM_F32_SWIZZLE (nonzero) overrides. */
  static const int swz_env = gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE", 0);
  int swz = 0;
  if (TB && K >= 2048) {
    const size_t panel = static_cast<size_t>(c.bm) * K * sizeof(float);
    swz = 8 * panel <= (16ull << 20) ? -8 : (4 * panel <= (16ull << 20) ? -4 : 8);
  }
  if (swz_env != 0)
    swz = swz_env;
  switch (id) {
#define GEMM_F32_TILE_CASE(ID, BM, BN, BK, TM, TN, THR)                        \
  case ID:                                                                     \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(gemm_f32_tile_kernel<TB, BM, BN, BK, TM, TN, THR>),    \
        grid, dim3(THR), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, \
        yvec, stag, swz);                                                           \
    break;
    GEMM_F32_TILE_CFGS(GEMM_F32_TILE_CASE)
#undef GEMM_F32_TILE_CASE
  default:
    break;
  }
}

static int launchF32Tile(int id, hipStream_t s, const float *A, const float *B,
                         const float *C, float *Y, int M, int N, int K,
                         int transB, float alpha, float beta, int cDim0,
                         int cDim1) {
  (void)hipGetLastError();
  if (transB)
    launchF32TileT<true>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                         cDim1);
  else
    launchF32TileT<false>(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                          cDim1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* Compute units of the current device, read once per device. */
static int gemmCuCount() {
  static std::atomic<int> cached[16];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16)
    dev = 0;
  int cus = cached[dev].load(std::memory_order_relaxed);
  if (cus <= 0) {
    if (hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount,
                              dev) != hipSuccess ||
        cus <= 0)
      cus = 40;
    cached[dev].store(cus, std::memory_order_relaxed);
  }
  return cus;
}

/* Static pick: the largest tile that still gives every CU two workgroups;
 * below 128 rows the tile height follows M. NT above 32 rows takes the
 * BK=32 128x64 tile (a K tile covers a whole 128-byte line of every A/B row)
 * when its grid is big enough, else 64x64; never 64x128, which is slow on NT
 * at BK=16 and 32 alike. bench3, fp32 NT, slower of 2 rounds:
 * 512x4096^2 2.49 -> 2.14 ms, 1024x16384x3072 15.9 -> 12.2,
 * 2048x4096x14336 41.4 -> 32.2 (vs 128x128 BK16); 64x14336x4096 3.97 -> 1.61
 * (vs 64x128 BK16). */
static int pickF32Tile(int M, int N, int transB) {
  const int target = 2 * gemmCuCount();
  auto wgs = [&](int id) {
    return ((M + kF32Tile[id].bm - 1) / kF32Tile[id].bm) *
           ((N + kF32Tile[id].bn - 1) / kF32Tile[id].bn);
  };
  if (M <= 16)
    return 5;
  if (M <= 32)
    return 4;
  if (transB)
    return wgs(6) >= target ? 6 : 3;
  if (M < 128)
    return wgs(1) >= target ? 1 : 3;
  for (int id = 0; id < 3; ++id)
    if (wgs(id) >= target)
      return id;
  return 3;
}

/* The LUT keeps tile geometry under the TiledFma kind. Points measured on
 * gemm_tiled_fma_kernel have geometries kF32Tile does not contain, so they
 * are rejected here until the table is re-measured with this kernel. */
static int f32TileConfigFromGeom(const gemm_lut::TiledFmaAnswer &g) {
  for (int cid = 0; cid < kNumF32Tile; ++cid) {
    const F32TileCfg &c = kF32Tile[cid];
    if (c.bm == g.bm && c.bn == g.bn && c.bk == g.bk && c.tm == g.tm &&
        c.tn == g.tn && c.threads == g.threads)
      return cid;
  }
  return -1;
}
static bool f32TileLutValidator(void *, const gemm_lut::TiledFmaAnswer &g) {
  return f32TileConfigFromGeom(g) >= 0;
}
static int gemmLutF32TileConfig(int M, int N, int K, int transB) {
  if (!gemmLutLookupEnabled())
    return -1;
  gemm_lut::Request req;
  req.kind = gemm_lut::Kind::TiledFma;
  req.act_dtype = req.wts_dtype = req.out_dtype = gemmDType<float>();
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  const gemm_lut::Result r = gemm_lut::resolve(req, nullptr, nullptr,
                                               f32TileLutValidator, nullptr);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return f32TileConfigFromGeom(r.tiled_fma);
}

static std::unordered_map<std::string, int> g_f32_tile_cache;

/* Same map -> forced -> LUT -> static pick -> (online mode) sweep order as
 * tuneTiledFma. The sweep logs in tuneTiledFma's "[gemm] tiledfma" format,
 * which update_lut.py parses into TiledFma points. */
static int tuneF32Tile(hipStream_t s, const float *A, const float *B,
                       const float *C, float *Y, int M, int N, int K,
                       int transB, float alpha, float beta, int cDim0,
                       int cDim1) {
  std::string key = std::string(transB ? "tb1_" : "tb0_") +
                    std::to_string(M) + "_" + std::to_string(N) + "_" +
                    std::to_string(K);
  {
    std::lock_guard<std::mutex> lk(g_tune_mu);
    auto it = g_f32_tile_cache.find(key);
    if (it != g_f32_tile_cache.end())
      return it->second;
  }

  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG", -1);
  int best = pickF32Tile(M, N, transB);
  if (forced >= 0 && forced < kNumF32Tile) {
    best = forced;
  } else if (int lut = gemmLutF32TileConfig(M, N, K, transB); lut >= 0) {
    best = lut;
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] f32tile LUT hit M=%d N=%d K=%d tb=%d -> cfg[%d] %dx%d bk=%d "
        "tm=%d tn=%d th=%d\n",
        M, N, K, transB, best, kF32Tile[best].bm, kF32Tile[best].bn,
        kF32Tile[best].bk, kF32Tile[best].tm, kF32Tile[best].tn,
        kF32Tile[best].threads);
  } else if (gemmLutLookupEnabled()) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] f32tile default M=%d N=%d K=%d tb=%d -> cfg[%d] %dx%d bk=%d "
        "tm=%d tn=%d th=%d\n",
        M, N, K, transB, best, kF32Tile[best].bm, kF32Tile[best].bn,
        kF32Tile[best].bk, kF32Tile[best].tm, kF32Tile[best].tn,
        kF32Tile[best].threads);
  } else if (gemmAutotuneOn() && gemmSweepSafe(C, Y, beta)) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    const int kInitWarm = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_INITWARM", 40);
    const int kWarm = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_WARM", 5);
    const int kIters = gemmEnvInt("HIPDNN_GEMM_TILEDFMA_TUNE_ITERS", 5);
    gemmSettleClock(
        s, e0, e1,
        [&] {
          launchF32Tile(best, s, A, B, C, Y, M, N, K, transB, alpha, beta,
                        cDim0, cDim1);
        },
        kInitWarm);
    float best_ms = 1e30f;
    for (int cid = 0; cid < kNumF32Tile; ++cid) {
      const float ms = gemmTimeConfigPeak(
          s, e0, e1,
          [&] {
            launchF32Tile(cid, s, A, B, C, Y, M, N, K, transB, alpha, beta,
                          cDim0, cDim1);
          },
          kWarm, kIters);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] tiledfma cfg[%d] %dx%d bk=%d tm=%d tn=%d th=%d : %.4f ms "
          "(peak)\n",
          cid, kF32Tile[cid].bm, kF32Tile[cid].bn, kF32Tile[cid].bk,
          kF32Tile[cid].tm, kF32Tile[cid].tn, kF32Tile[cid].threads, ms);
      if (ms < best_ms) {
        best_ms = ms;
        best = cid;
      }
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] tiledfma autotune M=%d N=%d K=%d ta=0 tb=%d -> cfg[%d] %dx%d "
        "bk=%d tm=%d tn=%d th=%d\n",
        M, N, K, transB, best, kF32Tile[best].bm, kF32Tile[best].bn,
        kF32Tile[best].bk, kF32Tile[best].tm, kF32Tile[best].tn,
        kF32Tile[best].threads);
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_f32_tile_cache[key] = best;
  return best;
}

/* =========================================================================
 * fp32 skinny-M NT (B = [N, K], transA = 0, small M): the structure of the
 * fp16/bf16 skinny kernel and ROCm/hip-ep #1120's small-M MatMulNBits kernel,
 * on VALU FMAs. A workgroup owns MT rows x WC columns of Y and its KW waves
 * split K (wave w takes the KC-wide chunks w, w+KW, ...), so the K loop has
 * no workgroup barrier and every wave keeps DB-1 chunks of loads in flight
 * behind the one it computes (a register ring rotated at compile time; the
 * loads are unconditional, as in gemm_wmma_skinny_kernel). The KW partial
 * tiles meet in LDS once at the end.
 *
 * B: LPR adjacent lanes share one column (B row): lane (r, q) =
 * (lid / LPR, lid % LPR), R = 32 / LPR, reads the 16-byte pieces q, q + LPR,
 * ... of row col0 + j*R + r (j < CL), so one load instruction covers R rows x
 * LPR*16 contiguous bytes (LPR = 8: whole 128-byte lines). The LPR partial
 * sums of a column meet by lane shuffles after the K loop.
 * A (MT rows, L2-resident and read by every workgroup): each chunk's MT x KC
 * slice is copied into a wave-private LDS tile with 16-byte loads issued
 * with the B loads, then read back as A[m][k..k+3] (LPR distinct addresses
 * per instruction) for all CL columns of the lane. The tile is private to
 * the wave and one wave's LDS operations complete in order, so no barrier.
 * Rows past M read row M-1 and columns past N read row N-1; neither is
 * stored. StaggerU (stag = chunks per step, keyed by N-tile) rotates each
 * N-tile's chunk order. Needs K % 4 == 0 and 16-byte aligned A and B; the
 * partial last chunk (K % KC) is one wave's, zero-filled past K.
 * ========================================================================= */
template <int MT, int CL, int LPR, int KW, int KC, int DB>
__global__ void __launch_bounds__(KW * 32)
    gemm_f32_skinny_kernel(const float *__restrict__ A,
                           const float *__restrict__ B,
                           const float *__restrict__ C, float *__restrict__ Y,
                           int M, int N, int K, float alpha, float beta,
                           int cDim0, int cDim1, int mtiles, int stag) {
  constexpr int R = 32 / LPR;        /* B rows per load instruction */
  constexpr int WC = R * CL;         /* Y columns per workgroup */
  constexpr int NL = KC / (4 * LPR); /* 16-byte B loads per column per chunk */
  constexpr int KQ = KC / 4;         /* 16-byte pieces per A row per chunk */
  constexpr int PA = MT * KQ / 32;   /* A pieces per lane per chunk */
  static_assert(32 % LPR == 0 && NL >= 1 && NL * 4 * LPR == KC, "B map");
  static_assert(PA >= 1 && PA * 32 == MT * KQ, "A map");
  static_assert(DB >= 2, "register ring");
  /* The A tiles (K loop) and the split-K reduction (after it) share one LDS
   * block. */
  constexpr int SM_T = KW * MT * KC;
  constexpr int SM_R = KW > 1 ? KW * MT * WC : 1;
  __shared__ __align__(16) float sm[SM_T > SM_R ? SM_T : SM_R];

  const int tid = threadIdx.x;
  /* Wave index as a scalar (see gemm_wmma_skinny_kernel). */
  const int w = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int r = lid / LPR;
  const int q = lid % LPR;
  const int mt = static_cast<int>(blockIdx.x) % mtiles;
  const int nt = static_cast<int>(blockIdx.x) / mtiles;
  const int row0 = mt * MT;
  const int col0 = nt * WC;
  float *tA = sm + w * MT * KC;

  const float *bp[CL];
#pragma unroll
  for (int j = 0; j < CL; ++j) {
    const int c = col0 + j * R + r;
    bp[j] = B + static_cast<size_t>(c < N ? c : N - 1) * K + 4 * q;
  }
  /* A piece p = lid + 32*u: row p / KQ, k offset 4 * (p % KQ); its tile
   * offset is 4 * p (row stride KC = 4 * KQ). */
  const float *ap[PA];
#pragma unroll
  for (int u = 0; u < PA; ++u) {
    const int p = lid + 32 * u;
    const int ar = row0 + p / KQ;
    ap[u] = A + static_cast<size_t>(ar < M ? ar : M - 1) * K + 4 * (p % KQ);
  }

  const int nfull = K / KC;
  const int shift = (stag > 0 && nfull > 1) ? ((nt & 31) * stag) % nfull : 0;

  gemmF4 ra[DB][PA];
  gemmF4 rb[DB][CL][NL];
  /* CL == 1: two partial sums per output (k pieces .x/.z and .y/.w), so the
   * FMAs of one row form two independent chains the compiler can pair into
   * v_dual_fmac_f32 (with CL > 1 the columns are independent already). */
  constexpr int AP = CL == 1 ? 2 : 1;
  float acc[MT][CL][AP];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < CL; ++j)
#pragma unroll
      for (int h = 0; h < AP; ++h)
        acc[m][j][h] = 0.0f;

  /* Loads chunk k0 .. k0+KC into slot S. TAIL zero-fills k >= K (K % 4 == 0,
   * so a 16-byte piece is all inside K or all outside). */
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < PA; ++u)
      ra[S][u] = (!TL || k0 + 4 * ((lid + 32 * u) % KQ) < K)
                     ? *reinterpret_cast<const gemmF4 *>(ap[u] + k0)
                     : zero;
#pragma unroll
    for (int j = 0; j < CL; ++j)
#pragma unroll
      for (int i = 0; i < NL; ++i)
        rb[S][j][i] = (!TL || k0 + 4 * (i * LPR + q) < K)
                          ? *reinterpret_cast<const gemmF4 *>(
                                bp[j] + k0 + 4 * i * LPR)
                          : zero;
  };
  /* Computes the chunk in slot S. MK (the last group only): a padding chunk
   * (live = false) stores zeros for its A slice instead of branching around
   * the compute. The fence after each 16-byte k step keeps the compiler from
   * hoisting every A read of the chunk at once (MT*NL*4 live registers). */
  auto compute = [&](auto slot, auto mk, bool live)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool MK = decltype(mk)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < PA; ++u)
      *reinterpret_cast<gemmF4 *>(tA + 4 * (lid + 32 * u)) =
          (!MK || live) ? ra[S][u] : zero;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const gemmF4 a =
            *reinterpret_cast<const gemmF4 *>(tA + m * KC + 4 * (i * LPR + q));
#pragma unroll
        for (int j = 0; j < CL; ++j) {
          const gemmF4 b = rb[S][j][i];
          acc[m][j][0] = fmaf(a.x, b.x, acc[m][j][0]);
          acc[m][j][AP - 1] = fmaf(a.y, b.y, acc[m][j][AP - 1]);
          acc[m][j][0] = fmaf(a.z, b.z, acc[m][j][0]);
          acc[m][j][AP - 1] = fmaf(a.w, b.w, acc[m][j][AP - 1]);
        }
      }
      GEMM_SCHED_FENCE();
    }
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  using NoMask = std::integral_constant<bool, false>;
  using Mask = std::integral_constant<bool, true>;
  /* The wave's c-th whole chunk, StaggerU-rotated. */
  auto kofs = [&](int c) __attribute__((always_inline)) {
    int ch = w + c * KW + shift;
    if (ch >= nfull)
      ch -= nfull;
    return ch * KC;
  };
  const int n = nfull > w ? (nfull - w + KW - 1) / KW : 0;
  if (n > 0) {
    /* The wave walks its n chunks padded to a multiple of DB: a padding
     * chunk loads the last real one again and is computed masked, so every
     * load is unconditional and the compiler's vmcnt waits keep DB-1 chunks
     * in flight (a load that may or may not be issued before a compute makes
     * it wait for that load either way). Slots 0 .. DB-2 hold chunks
     * i .. i+DB-2 at the top of each group. */
    const int npad = (n + DB - 1) / DB * DB;
    auto kofsc = [&](int c) __attribute__((always_inline)) {
      return kofs(c < n ? c : n - 1);
    };
    gemmStaticFor<0, DB - 1>([&](auto s) __attribute__((always_inline)) {
      load(kofsc(decltype(s)::value), s, Full{});
    });
    int i = 0;
    for (; i + DB < npad; i += DB) {
      gemmStaticFor<0, DB>([&](auto s) __attribute__((always_inline)) {
        constexpr int S = decltype(s)::value;
        load(kofsc(i + S + DB - 1),
             std::integral_constant<int, (S + DB - 1) % DB>{}, Full{});
        GEMM_SCHED_FENCE();
        compute(s, NoMask{}, true);
      });
    }
    /* Last group: one load left, then the DB computes. */
    load(kofsc(i + DB - 1), std::integral_constant<int, DB - 1>{}, Full{});
    GEMM_SCHED_FENCE();
    gemmStaticFor<0, DB>([&](auto s) __attribute__((always_inline)) {
      compute(s, Mask{}, i + decltype(s)::value < n);
    });
  }
  /* Partial last chunk (K % KC != 0): one wave, zero-filled past K. */
  if (nfull * KC < K && w == nfull % KW) {
    load(nfull * KC, std::integral_constant<int, 0>{}, Tail{});
    compute(std::integral_constant<int, 0>{}, NoMask{}, true);
  }

  /* The LPR lanes of a column hold partial sums over different k pieces. */
  float sum[MT][CL];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < CL; ++j) {
      sum[m][j] = AP == 2 ? acc[m][j][0] + acc[m][j][AP - 1] : acc[m][j][0];
#pragma unroll
      for (int off = 1; off < LPR; off *= 2)
        sum[m][j] += __shfl_xor(sum[m][j], off);
    }

  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f;
  auto store = [&](int m, int c, float v) __attribute__((always_inline)) {
    const int rr = row0 + m;
    const int cc = col0 + c;
    if (rr < M && cc < N) {
      if (fast)
        Y[static_cast<size_t>(rr) * N + cc] = v;
      else
        gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
    }
  };
  if constexpr (KW == 1) {
    if (q == 0) {
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < CL; ++j)
          store(m, j * R + r, sum[m][j]);
    }
  } else {
    __syncthreads(); /* other waves may still read their A tiles */
    if (q == 0) {
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < CL; ++j)
          sm[(w * MT + m) * WC + j * R + r] = sum[m][j];
    }
    __syncthreads();
    for (int idx = tid; idx < MT * WC; idx += KW * 32) {
      const int m = idx / WC;
      const int c = idx % WC;
      float v = 0.0f;
#pragma unroll
      for (int wq = 0; wq < KW; ++wq)
        v += sm[(wq * MT + m) * WC + c];
      store(m, c, v);
    }
  }
}

/* {id, MT, CL, LPR, KW, KC, DB}: tile MT x (32 / LPR * CL), KW waves splitting
 * K in KC-wide chunks, DB chunks of loads per wave (DB-1 in flight behind the
 * computed one). X-macro so the table and its launch switch cannot drift. */
#define GEMM_F32_SKINNY_CFGS(X)                                                \
  X(0, 16, 1, 8, 8, 32, 2)                                                     \
  X(1, 8, 1, 8, 8, 32, 2)                                                      \
  X(2, 8, 2, 8, 8, 32, 2)                                                      \
  X(3, 16, 2, 8, 8, 32, 2)                                                     \
  X(4, 16, 2, 8, 4, 32, 2)

struct F32SkinnyCfg {
  int mt, cl, lpr, kw, kc, db;
};
#define GEMM_F32_SKINNY_ENTRY(ID, MT, CL, LPR, KW, KC, DB)                     \
  {MT, CL, LPR, KW, KC, DB},
static constexpr F32SkinnyCfg kF32Skinny[] = {
    GEMM_F32_SKINNY_CFGS(GEMM_F32_SKINNY_ENTRY)};
#undef GEMM_F32_SKINNY_ENTRY
static constexpr int kNumF32Skinny =
    static_cast<int>(sizeof(kF32Skinny) / sizeof(kF32Skinny[0]));

static void launchF32SkinnyCfg(int id, hipStream_t s, const float *A,
                               const float *B, const float *C, float *Y, int M,
                               int N, int K, float alpha, float beta, int cDim0,
                               int cDim1, int stag_on) {
  const F32SkinnyCfg &c = kF32Skinny[id];
  const int wc = 32 / c.lpr * c.cl;
  const int mtiles = (M + c.mt - 1) / c.mt;
  const int ntiles = (N + wc - 1) / wc;
  /* StaggerU step of 256 bytes of K. */
  const int stag = stag_on ? std::max(1, 64 / c.kc) : 0;
  switch (id) {
#define GEMM_F32_SKINNY_CASE(ID, MT, CL, LPR, KW, KC, DB)                      \
  case ID:                                                                     \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(gemm_f32_skinny_kernel<MT, CL, LPR, KW, KC, DB>),      \
        dim3(mtiles * ntiles), dim3(KW * 32), 0, s, A, B, C, Y, M, N, K,       \
        alpha, beta, cDim0, cDim1, mtiles, stag);                              \
    break;
    GEMM_F32_SKINNY_CFGS(GEMM_F32_SKINNY_CASE)
#undef GEMM_F32_SKINNY_CASE
  default:
    break;
  }
}

/* fp32 skinny route (NT, transA = 0): the config to launch, or -1 to stay on
 * the multi-row GEMV (M < 16) / gemm_f32_tile_kernel. bench3 stage3 sm1_sweep
 * (every config forced on 45 shapes, 2 rounds ABBA, slower round) and sm1_re;
 * old = the GEMV / tile route, hb = hipBLASLt, ms:
 *   - M <= 4 keeps the GEMV (2x14336x4096 1.072 vs 1.103 with cfg 1).
 *   - B <= 48 MB (3072^2 here, partly Infinity-Cache resident) and
 *     M <= 24 -> cfg 2 (16x3072^2 0.116, old 0.157, cfg 0 0.135;
 *     24x3072^2 0.134, old 0.167, cfg 1 0.172).
 *   - M <= 8 -> cfg 1 (8x4096^2 0.318, old 0.345, hb 0.327; 6x14336x4096
 *     1.098, old 1.130), M <= 16 -> cfg 0 (16x4096^2 0.324, old 0.422, hb
 *     0.356; 16x14336x4096 1.103, old 1.514, hb 1.164; 16x4096x14336 1.132,
 *     old 1.508, hb 1.220), M <= 24 -> cfg 1 (24x14336x4096 1.132, old 1.446).
 *   - M <= 32 -> cfg 4 (32x4096^2 0.346, old 0.429; 32x14336x4096 1.215,
 *     old 1.457).
 *   - M <= 48 -> cfg 3 when N >= 8192 and K <= 8192 (48x14336x4096 1.243,
 *     old 1.569; cfg 4 1.375), else cfg 4 (48x4096x14336 1.568, old 1.793).
 *   - M <= 64 -> cfg 4 (64x14336x4096 1.484, old 1.609; 64x4096^2 0.439,
 *     old 0.504).
 *   - Not when A (M*K floats) exceeds 3 MB: every workgroup re-reads A, and
 *     past the 2 MB L2 that costs more than B (64x4096x14336, A 3.7 MB: 2.06
 *     vs the tile's 1.77). Nor below K = 256 or N*K = 64K (not measured).
 * The 512-thread twin of cfg 0 (KW = 16) is 3-5% faster on an idle GPU but
 * slowed 4-40x (hipBLASLt 2x) while another process held the GPU. */
static int pickF32Skinny(int M, int N, int K) {
  if (M <= 4 || M > 64 || K < 256 ||
      static_cast<size_t>(N) * K < (size_t{1} << 16) ||
      static_cast<size_t>(M) * K * sizeof(float) > (3ull << 20))
    return -1;
  if (M <= 24 && static_cast<size_t>(N) * K * sizeof(float) <= (48ull << 20))
    return 2;
  if (M <= 8)
    return 1;
  if (M <= 16)
    return 0;
  if (M <= 24)
    return 1;
  if (M <= 32)
    return 4;
  if (M <= 48)
    return (N >= 8192 && K <= 8192) ? 3 : 4;
  return 4;
}

/* Returns false (nothing launched) when the shape does not meet the kernel's
 * preconditions (fp32 NT, transA = 0, K % 4 == 0, 16-byte aligned A and B)
 * or the route keeps it on its existing path. HIPDNN_EP_GEMM_F32_SKINNY=0
 * disables the route; HIPDNN_EP_GEMM_F32_SKINNY_CFG=<id> forces a config for
 * every eligible shape. */
static bool launchF32Skinny(hipStream_t s, const float *A, const float *B,
                            const float *C, float *Y, int M, int N, int K,
                            int transA, int transB, float alpha, float beta,
                            int cDim0, int cDim1) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY", 1) != 0;
  static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG", -1);
  static const bool stag_on = gemmEnvInt("HIPDNN_EP_GEMM_F32_STAGGER", 1) != 0;
  if (!on || transA || !transB || M < 2 || (K % 4) != 0 ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) != 0)
    return false;
  const int id = (forced >= 0 && forced < kNumF32Skinny)
                     ? forced
                     : pickF32Skinny(M, N, K);
  if (id < 0)
    return false;
  (void)hipGetLastError();
  launchF32SkinnyCfg(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
                     stag_on ? 1 : 0);
  const F32SkinnyCfg &c = kF32Skinny[id];
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] f32-skinny M=%d N=%d K=%d -> cfg[%d] mt=%d cl=%d lpr=%d kw=%d "
      "kc=%d db=%d\n",
      M, N, K, id, c.mt, c.cl, c.lpr, c.kw, c.kc, c.db);
  return true;
}

/* =========================================================================
 * fp32 skinny-M NN (B = [K, N], transA = 0, small M). The K-split structure
 * of gemm_f32_skinny_kernel, for a weight whose k-rows run along N: lane l
 * reads columns col0 + 32*VW*j + VW*l .. +VW-1 (j < CL) of one k-row with one
 * VW*4-byte load, so one load instruction streams 32*VW*4 contiguous bytes
 * of the row (VW = 1: one 128-byte line) and no lane shares a column (no
 * shuffle reduction). A workgroup owns MT rows x 32*VW*CL columns of Y; its
 * KW waves split K (wave w takes the KC-row chunks w, w+KW, ...), with no
 * workgroup barrier in the K loop and a DB-deep register ring keeping DB-1
 * chunks of loads in flight behind the one being computed (unconditional
 * loads, padding chunks computed masked, as in gemm_f32_skinny_kernel).
 * MT should cover M: M-tiles in separate workgroups did not reliably share
 * the B panel through L2 on a long K.
 * A (MT rows, L2-resident): each chunk's MT x KC slice is copied with 16-byte
 * loads into a wave-private LDS tile, then read back as A[m][k..k+3]: every
 * lane of the wave reads the same address (an LDS broadcast). Per 4 k-rows a
 * lane does 4 x MT x VW*CL FMAs into its MT x VW*CL accumulators. The KW
 * partial tiles meet in LDS at the end, RG rows at a time.
 * Rows past M read row M-1 and columns past N read the last VW columns;
 * neither is stored. StaggerU (stag = chunks per step, keyed by N-tile)
 * rotates each N-tile's chunk order. Needs N % VW == 0, K % 4 == 0, 16-byte
 * aligned A and VW*4-byte aligned B; the partial last chunk (K % KC) is one
 * wave's, zero-filled past K.
 * ========================================================================= */
template <int N_> struct GemmF32Vec;
template <> struct GemmF32Vec<1> {
  typedef float type;
};
template <> struct GemmF32Vec<2> {
  typedef float type __attribute__((ext_vector_type(2)));
};
template <> struct GemmF32Vec<4> {
  typedef float type __attribute__((ext_vector_type(4)));
};

template <int MT, int VW, int CL, int KW, int KC, int DB>
__global__ void __launch_bounds__(KW * 32)
    gemm_f32_skinny_nn_kernel(const float *__restrict__ A,
                              const float *__restrict__ B,
                              const float *__restrict__ C,
                              float *__restrict__ Y, int M, int N, int K,
                              float alpha, float beta, int cDim0, int cDim1,
                              int mtiles, int stag) {
  using V = typename GemmF32Vec<VW>::type;
  constexpr int WS = 32 * VW;        /* columns of one load instruction */
  constexpr int WC = WS * CL;        /* Y columns per workgroup */
  constexpr int KQ = KC / 4;         /* 16-byte pieces per A row per chunk */
  constexpr int NP = MT * KQ;        /* A pieces per chunk */
  constexpr int PA = (NP + 31) / 32; /* A pieces per lane */
  static_assert(KC % 4 == 0 && KC >= 4 && DB >= 2, "chunk / ring");
  /* Reduction: RG rows of the KW partial tiles at a time (<= 16 KB). */
  constexpr int RG0 = 4096 / (KW * WC);
  constexpr int RG = RG0 < 1 ? 1 : (RG0 > MT ? MT : RG0);
  static_assert(MT % RG == 0, "reduction groups");
  constexpr int SM_T = KW * MT * KC;
  constexpr int SM_R = KW > 1 ? KW * RG * WC : 1;
  __shared__ __align__(16) float sm[SM_T > SM_R ? SM_T : SM_R];

  const int tid = threadIdx.x;
  /* Wave index as a scalar (see gemm_wmma_skinny_kernel). */
  const int w = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int mt = static_cast<int>(blockIdx.x) % mtiles;
  const int nt = static_cast<int>(blockIdx.x) / mtiles;
  const int row0 = mt * MT;
  const int col0 = nt * WC;
  float *tA = sm + w * MT * KC;

  int bc[CL];
#pragma unroll
  for (int j = 0; j < CL; ++j) {
    const int c = col0 + j * WS + VW * lid;
    bc[j] = c + VW <= N ? c : N - VW;
  }
  /* A piece p = lid + 32*u (pieces past NP repeat one inside it): row
   * p / KQ, k offset 4 * (p % KQ), tile offset 4 * p. */
  const float *ap[PA];
  int ao[PA];
#pragma unroll
  for (int u = 0; u < PA; ++u) {
    const int p0 = lid + 32 * u;
    const int p = p0 < NP ? p0 : p0 % NP;
    const int ar = row0 + p / KQ;
    ap[u] = A + static_cast<size_t>(ar < M ? ar : M - 1) * K + 4 * (p % KQ);
    ao[u] = 4 * p;
  }

  const int nfull = K / KC;
  const int shift = (stag > 0 && nfull > 1) ? ((nt & 31) * stag) % nfull : 0;

  gemmF4 ra[DB][PA];
  V vz;
  __builtin_memset(&vz, 0, sizeof(vz));
  V rb[DB][CL][KC];
  V acc[MT][CL];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < CL; ++j)
      acc[m][j] = vz;

  /* Loads chunk k0 .. k0+KC into slot S. TAIL zero-fills k >= K (K % 4 == 0,
   * so a 16-byte A piece is all inside K or all outside). */
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < PA; ++u)
      ra[S][u] = (!TL || k0 + 4 * (((lid + 32 * u) % NP) % KQ) < K)
                     ? *reinterpret_cast<const gemmF4 *>(ap[u] + k0)
                     : zero;
    const float *bk = B + static_cast<size_t>(k0) * N;
#pragma unroll
    for (int r = 0; r < KC; ++r)
#pragma unroll
      for (int j = 0; j < CL; ++j)
        rb[S][j][r] = (!TL || k0 + r < K)
                          ? *reinterpret_cast<const V *>(
                                bk + static_cast<size_t>(r) * N + bc[j])
                          : vz;
  };
  /* Computes the chunk in slot S. MK (the last group only): a padding chunk
   * (live = false) stores zeros for its A slice instead of branching around
   * the compute. The fence after each 4-row k step keeps the compiler from
   * hoisting every A read of the chunk at once. */
  auto compute = [&](auto slot, auto mk, bool live)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool MK = decltype(mk)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < PA; ++u)
      *reinterpret_cast<gemmF4 *>(tA + ao[u]) =
          (!MK || live) ? ra[S][u] : zero;
#pragma unroll
    for (int q = 0; q < KQ; ++q) {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const gemmF4 a =
            *reinterpret_cast<const gemmF4 *>(tA + m * KC + 4 * q);
#pragma unroll
        for (int j = 0; j < CL; ++j) {
          acc[m][j] += a.x * rb[S][j][4 * q];
          acc[m][j] += a.y * rb[S][j][4 * q + 1];
          acc[m][j] += a.z * rb[S][j][4 * q + 2];
          acc[m][j] += a.w * rb[S][j][4 * q + 3];
        }
      }
      GEMM_SCHED_FENCE();
    }
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  using NoMask = std::integral_constant<bool, false>;
  using Mask = std::integral_constant<bool, true>;
  /* The wave's c-th whole chunk, StaggerU-rotated. */
  auto kofs = [&](int c) __attribute__((always_inline)) {
    int ch = w + c * KW + shift;
    if (ch >= nfull)
      ch -= nfull;
    return ch * KC;
  };
  const int n = nfull > w ? (nfull - w + KW - 1) / KW : 0;
  if (n > 0) {
    /* Padded to a multiple of DB chunks, as in gemm_f32_skinny_kernel. */
    const int npad = (n + DB - 1) / DB * DB;
    auto kofsc = [&](int c) __attribute__((always_inline)) {
      return kofs(c < n ? c : n - 1);
    };
    gemmStaticFor<0, DB - 1>([&](auto s) __attribute__((always_inline)) {
      load(kofsc(decltype(s)::value), s, Full{});
    });
    int i = 0;
    for (; i + DB < npad; i += DB) {
      gemmStaticFor<0, DB>([&](auto s) __attribute__((always_inline)) {
        constexpr int S = decltype(s)::value;
        load(kofsc(i + S + DB - 1),
             std::integral_constant<int, (S + DB - 1) % DB>{}, Full{});
        GEMM_SCHED_FENCE();
        compute(s, NoMask{}, true);
      });
    }
    load(kofsc(i + DB - 1), std::integral_constant<int, DB - 1>{}, Full{});
    GEMM_SCHED_FENCE();
    gemmStaticFor<0, DB>([&](auto s) __attribute__((always_inline)) {
      compute(s, Mask{}, i + decltype(s)::value < n);
    });
  }
  /* Partial last chunk (K % KC != 0): one wave, zero-filled past K. */
  if (nfull * KC < K && w == nfull % KW) {
    load(nfull * KC, std::integral_constant<int, 0>{}, Tail{});
    compute(std::integral_constant<int, 0>{}, NoMask{}, true);
  }

  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f;
  auto store = [&](int rr, int cc, V v) __attribute__((always_inline)) {
    if (rr < M && cc < N) {
      float e[VW];
      __builtin_memcpy(e, &v, sizeof(e));
#pragma unroll
      for (int q = 0; q < VW; ++q) {
        if (fast)
          Y[static_cast<size_t>(rr) * N + cc + q] = e[q];
        else
          gemmStoreY(Y, C, rr, cc + q, N, cDim0, cDim1, e[q], alpha, beta);
      }
    }
  };
  if constexpr (KW == 1) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < CL; ++j)
        store(row0 + m, col0 + j * WS + VW * lid, acc[m][j]);
  } else {
    V *red = reinterpret_cast<V *>(sm);
    gemmStaticFor<0, MT / RG>([&](auto g) __attribute__((always_inline)) {
      constexpr int G = decltype(g)::value;
      __syncthreads(); /* A tiles / the previous group are still being read */
#pragma unroll
      for (int i = 0; i < RG; ++i)
#pragma unroll
        for (int j = 0; j < CL; ++j)
          red[((w * RG + i) * CL + j) * 32 + lid] = acc[G * RG + i][j];
      __syncthreads();
      for (int idx = tid; idx < RG * CL * 32; idx += KW * 32) {
        V v = red[idx];
#pragma unroll
        for (int wq = 1; wq < KW; ++wq)
          v += red[wq * RG * CL * 32 + idx];
        store(row0 + G * RG + idx / (CL * 32),
              col0 + (idx / 32) % CL * WS + VW * (idx % 32), v);
      }
    });
  }
}

/* {id, MT, VW, CL, KW, KC, DB}: tile MT x 32*VW*CL, KW waves splitting K in
 * KC-row chunks, DB chunks of loads per wave (DB-1 in flight behind the
 * computed one). X-macro so the table and its launch switch cannot drift. */
#define GEMM_F32_SKINNY_NN_CFGS(X)                                             \
  X(0, 8, 4, 2, 4, 8, 2)                                                       \
  X(1, 16, 2, 2, 4, 4, 2)                                                      \
  X(2, 32, 2, 2, 4, 4, 2)

struct F32SkinnyNnCfg {
  int mt, vw, cl, kw, kc, db;
};
#define GEMM_F32_SKINNY_NN_ENTRY(ID, MT, VW, CL, KW, KC, DB)                   \
  {MT, VW, CL, KW, KC, DB},
static constexpr F32SkinnyNnCfg kF32SkinnyNn[] = {
    GEMM_F32_SKINNY_NN_CFGS(GEMM_F32_SKINNY_NN_ENTRY)};
#undef GEMM_F32_SKINNY_NN_ENTRY
static constexpr int kNumF32SkinnyNn =
    static_cast<int>(sizeof(kF32SkinnyNn) / sizeof(kF32SkinnyNn[0]));

static void launchF32SkinnyNnCfg(int id, hipStream_t s, const float *A,
                                 const float *B, const float *C, float *Y,
                                 int M, int N, int K, float alpha, float beta,
                                 int cDim0, int cDim1, int stag_on) {
  const F32SkinnyNnCfg &c = kF32SkinnyNn[id];
  const int wc = 32 * c.vw * c.cl;
  const int mtiles = (M + c.mt - 1) / c.mt;
  const int ntiles = (N + wc - 1) / wc;
  /* StaggerU step of 64 k-rows. */
  const int stag = stag_on ? std::max(1, 64 / c.kc) : 0;
  switch (id) {
#define GEMM_F32_SKINNY_NN_CASE(ID, MT, VW, CL, KW, KC, DB)                    \
  case ID:                                                                     \
    hipLaunchKernelGGL(                                                        \
        HIP_KERNEL_NAME(gemm_f32_skinny_nn_kernel<MT, VW, CL, KW, KC, DB>),    \
        dim3(mtiles * ntiles), dim3(KW * 32), 0, s, A, B, C, Y, M, N, K,       \
        alpha, beta, cDim0, cDim1, mtiles, stag);                              \
    break;
    GEMM_F32_SKINNY_NN_CFGS(GEMM_F32_SKINNY_NN_CASE)
#undef GEMM_F32_SKINNY_NN_CASE
  default:
    break;
  }
}

/* fp32 NN skinny route (B = [K, N], transA = 0): the config to launch, or -1
 * to stay on the [K, N] GEMV (M < 16) / gemm_f32_tile_kernel. bench3 stage3
 * p3_sw_a / p3_sw_b (each config forced, 2 rounds ABBA, slower round; old =
 * the GEMV / tile route, hb = hipBLASLt), ms:
 *   - M <= 8 -> cfg 0 when B <= 48 MB or N >= 2K (8x3072^2 0.105, old
 *     0.201, hb 0.129; 6x11008x4096 0.763, old 0.854; 2x14336x4096 1.031,
 *     old GEMV 1.045), else cfg 1 (8x4096^2 0.303, old 0.424, hb 0.298;
 *     8x4096x14336 1.049, old 1.411, hb 1.042).
 *   - M <= 16 -> cfg 1 (12x4096x14336 1.055, old 2.508, hb 1.017;
 *     12x14336x4096 1.131, old 2.119, hb 1.218; 16x4096^2 0.298, old
 *     0.386).
 *   - M <= 32 -> cfg 2 (32x4096x14336 1.078, old 1.380, hb 1.089;
 *     32x14336x4096 1.113, old 1.360, hb 1.118; 24x4096^2 0.302, old
 *     0.399).
 *   - M <= 64 only when B <= 48 MB -> cfg 2 (48x3072^2 0.163, old 0.180,
 *     hb 0.154). Above 32 rows the tile wins on the larger weights: one
 *     workgroup per column strip needs MT * VW * CL accumulators per lane
 *     (MT 64 was 1.32-1.80 on the DRAM shapes), and M-tiles in separate
 *     workgroups re-read B (48x4096x14336 2.18 with two MT 32 tiles, tile
 *     1.24).
 *   - Not when A (M*K floats) exceeds 3 MB, K < 256 or N*K < 64K, as for
 *     the NT kernel. */
static int pickF32SkinnyNn(int M, int N, int K) {
  /* M = 2..8 on an Infinity-Cache-resident B: the ring GEMV (pickKnRing). */
  if (M <= 8 && gemmKnRingOn() && pickKnRing(M, N, K, 4) >= 0 &&
      (K % 4) == 0 && static_cast<size_t>(N) * K >= (size_t{1} << 16))
    return -1;
  if (M < 2 || M > 64 || K < 256 ||
      static_cast<size_t>(N) * K < (size_t{1} << 16) ||
      static_cast<size_t>(M) * K * sizeof(float) > (3ull << 20))
    return -1;
  const bool small =
      static_cast<size_t>(N) * K * sizeof(float) <= (48ull << 20);
  if (M > 32)
    return small ? 2 : -1;
  if (M <= 8)
    return (small || N >= 2 * K) ? 0 : 1;
  return M <= 16 ? 1 : 2;
}

/* Returns false (nothing launched) when the shape does not meet the kernel's
 * preconditions (fp32 NN, transA = 0, N % 4 == 0, K % 4 == 0, 16-byte
 * aligned A and B) or the route keeps it on its existing path.
 * HIPDNN_EP_GEMM_F32_SKINNY=0 disables it (and the NT kernel);
 * HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG=<id> forces a config for every eligible
 * shape. */
static bool launchF32SkinnyNn(hipStream_t s, const float *A, const float *B,
                              const float *C, float *Y, int M, int N, int K,
                              int transA, int transB, float alpha, float beta,
                              int cDim0, int cDim1) {
  static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY", 1) != 0;
  static const int forced =
      gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG", -1);
  static const int stag_on = gemmEnvInt("HIPDNN_EP_GEMM_F32_STAGGER", 1);
  if (!on || transA || transB || M < 2 || (K % 4) != 0 || (N % 4) != 0 ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) != 0)
    return false;
  const int id = (forced >= 0 && forced < kNumF32SkinnyNn)
                     ? forced
                     : pickF32SkinnyNn(M, N, K);
  if (id < 0)
    return false;
  (void)hipGetLastError();
  launchF32SkinnyNnCfg(id, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,
                       stag_on != 0 ? 1 : 0);
  const F32SkinnyNnCfg &c = kF32SkinnyNn[id];
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] f32-skinny-nn M=%d N=%d K=%d -> cfg[%d] mt=%d vw=%d cl=%d kw=%d "
      "kc=%d db=%d\n",
      M, N, K, id, c.mt, c.vw, c.cl, c.kw, c.kc, c.db);
  return true;
}

template <typename T>
__global__ void gemm_kzero_kernel(const T *__restrict__ C, T *__restrict__ Y,
                                 int M, int N, float beta, int cDim0,
                                 int cDim1) {
  const size_t total = static_cast<size_t>(M) * N;
  const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += step) {
    const int r = static_cast<int>(i / N);
    const int c = static_cast<int>(i - static_cast<size_t>(r) * N);
    gemmStoreY(Y, C, r, c, N, cDim0, cDim1, 0.0f, 1.0f, beta);
  }
}

template <typename T>
static int launchKZero(hipStream_t s, const T *C, T *Y, int M, int N,
                       float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
  const size_t total = static_cast<size_t>(M) * N;
  const size_t blocks = std::min<size_t>((total + 255) / 256, size_t{1} << 20);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kzero_kernel<T>),
                     dim3(static_cast<unsigned>(blocks)), dim3(256), 0, s, C,
                     Y, M, N, beta, cDim0, cDim1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* Wave-private small-M rolling tiles. Stores precede register refill;
 * the one-wave instances need no workgroup rendezvous in the K loop. */
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__device__ __forceinline__ void
GemmSmallMRollImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + 8;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = BM / RPI;
  constexpr int B_NCH = BN / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr bool ROLL = (RP & 32) != 0;
  constexpr bool PAD32 = (RP & 64) != 0;
  constexpr bool GROUP = (RP & 128) != 0;
  constexpr bool HB_GROUP = (RP & 256) != 0;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1;
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && BM % RPI == 0 && BN % RPI == 0, "chunks");
  static_assert(MODE == 1 || MODE == 2, "RP pipeline");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  static_assert(!ROLL || DB, "rolling stores require a second LDS buffer");
  constexpr int LSIZE = PAD32 ? 16384 : (BM + BN) * K_STR;
  constexpr int LALLOC = (NBUF - 1) * LSIZE + (BM + BN) * K_STR;
  static_assert(LSIZE >= (BM + BN) * K_STR && LALLOC * ES <= 65536, "LDS");
  __shared__ __align__(16) T sm[LALLOC];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if HIPDNN_WMMA_FRAG_ELEMS == 16
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if HIPDNN_WMMA_GFX12
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH)
        *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + (lr + i * RPI) * K_STR + lk]) = pf[S][i];
      else
        *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + BM * K_STR + (lr + (i - A_NCH) * RPI) * K_STR + lk]) =
            pf[S][i];
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH)
      *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + (lr + i * RPI) * K_STR + lk]) = pf[S][i];
    else
      *reinterpret_cast<u32x4 *>(&sm[P * LSIZE + BM * K_STR + (lr + (i - A_NCH) * RPI) * K_STR + lk]) = pf[S][i];
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    if constexpr (std::is_same<T, _Float16>::value) {
      half16 b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + BM * K_STR + (noff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        half16 a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + (moff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] =
              HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    } else {
      bf16xN b_frag[WT_N];
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int noff = (wcol * WT_N + wn) * WMMA_TILE;
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + BM * K_STR + (noff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
      }
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int moff = (wrow * WT_M + wm) * WMMA_TILE;
        bf16xN a_frag{};
        const u32x4 *src = reinterpret_cast<const u32x4 *>(
            &sm[P * LSIZE + (moff + lane) * K_STR + ks * WMMA_TILE + k_off]);
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = src[i];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                     acc[wm][wn]);
      }
    }
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((HB_GROUP && (i < K_STEPS - 1 ? i : K_STEPS - 1) == ks) ||
              (!HB_GROUP && ((!GROUP && ks == 0) || (GROUP && i * K_STEPS / NL == ks)))) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS) kb += KB;
    } else {
      if constexpr (IS) issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) computeKs(PC{}, ks);
      schedStep(IS);
    }
    if constexpr (ST && !ROLL) {
      if constexpr (!DB) {
        if constexpr (THR > 32) __syncthreads();
        GEMM_SCHED_FENCE();
      }
      storeAll(SS{}, PN{});
      if constexpr (THR > 32) __syncthreads();
    }
    if constexpr (ST && ROLL) if constexpr (THR > 32) __syncthreads();
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  if constexpr (THR > 32) __syncthreads();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && HIPDNN_WMMA_FRAG_ELEMS == 16
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = fmaf(al, acc[wm][wn][e], bc);
        const float v1 = fmaf(al, acc[wm][wn][e + 1], bc);
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (cc < N && rr < M)
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (cc < N && rr < M)
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf(fmaf(al, acc[wm][wn][e], bc));
      }
    }
  }
#endif
}

template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP>
__global__ __launch_bounds__(GEMM_WG_SZ(BM, BN, WT_M, WT_N))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_smallroll_kernel(const T *__restrict__ A, const T *__restrict__ B,
                          const T *__restrict__ C, T *__restrict__ Y, int M,
                          int N, int K, float alpha, float beta, int cDim0,
                          int cDim1, int swizzle_n, int ldb, int lda,
                          int stag) {
  GemmSmallMRollImpl<BM, BN, WT_M, WT_N, T, BK, DB, RP>(
      A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, swizzle_n, ldb, lda,
      stag);
}



template <typename T, int BLOCK_SIZE, int TILE_N>
__global__ void __launch_bounds__(BLOCK_SIZE)
    gemm_gemv_decode_nt_kernel(const T *__restrict__ A, const T *__restrict__ B,
                       const T *__restrict__ C, T *__restrict__ Y, int M,
                       int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  constexpr int WARP = HIPDNN_WAVE_SIZE;
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP;

  const int n_base = blockIdx.x * TILE_N;
  const int m = blockIdx.y;
  const int tid = threadIdx.x;
  if (m >= M)
    return;

  const T *a_base = A + static_cast<size_t>(m) * K;
  const T *b_row[TILE_N];
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn) {
    const int n = n_base + tn;
    const int sn = (n < N) ? n : 0;
    b_row[tn] = B + static_cast<size_t>(sn) * K;
  }

  float partial[TILE_N];
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn)
    partial[tn] = 0.0f;

  const int num_vec = K / 8;
  for (int idx = tid; idx < num_vec; idx += BLOCK_SIZE) {
    const int k = idx * 8;
    float af[8];
    if constexpr (std::is_same<T, _Float16>::value) {
      u32x4 a_pk = *reinterpret_cast<const u32x4 *>(a_base + k);
      const _Float16 *ah = reinterpret_cast<const _Float16 *>(&a_pk);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        af[i] = (float)ah[i];
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        af[i] = GemmElem<T>::tof(a_base[k + i]);
    }

#pragma unroll
    for (int tn = 0; tn < TILE_N; ++tn) {
      if constexpr (std::is_same<T, _Float16>::value) {
        u32x4 b_pk = *reinterpret_cast<const u32x4 *>(b_row[tn] + k);
        const _Float16 *bh = reinterpret_cast<const _Float16 *>(&b_pk);
#pragma unroll
        for (int i = 0; i < 8; ++i)
          partial[tn] = __fmaf_rn(af[i], (float)bh[i], partial[tn]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          partial[tn] = __fmaf_rn(af[i], GemmElem<T>::tof(b_row[tn][k + i]),
                                  partial[tn]);
      }
    }
  }

  const int k8 = num_vec * 8;
  for (int k = k8 + tid; k < K; k += BLOCK_SIZE) {
    const float av = GemmElem<T>::tof(a_base[k]);
#pragma unroll
    for (int tn = 0; tn < TILE_N; ++tn)
      partial[tn] =
          __fmaf_rn(av, GemmElem<T>::tof(b_row[tn][k]), partial[tn]);
  }

  const int lane = tid % WARP;
  const int warp = tid / WARP;
#pragma unroll
  for (int tn = 0; tn < TILE_N; ++tn) {
    for (int off = WARP >> 1; off > 0; off >>= 1)
      partial[tn] += __shfl_down(partial[tn], off);
  }

  if constexpr (BLOCK_SIZE <= WARP) {
    if (lane == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn) {
        const int n = n_base + tn;
        if (n < N)
          gemmStoreY(Y, C, m, n, N, cDim0, cDim1, partial[tn], alpha, beta);
      }
    }
  } else {
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    if (lane == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn)
        warp_sums[tn * NUM_WARPS + warp] = partial[tn];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
      for (int tn = 0; tn < TILE_N; ++tn) {
        float val = (lane < NUM_WARPS) ? warp_sums[tn * NUM_WARPS + lane] : 0.f;
        for (int off = WARP >> 1; off > 0; off >>= 1)
          val += __shfl_down(val, off);
        if (lane == 0) {
          const int n = n_base + tn;
          if (n < N)
            gemmStoreY(Y, C, m, n, N, cDim0, cDim1, val, alpha, beta);
        }
      }
    }
  }
}

/* Exact measured FP16 decode override, independent of the GEMV LUT. */
template <typename T>
static bool launchMeasuredDecodeNt(hipStream_t s, const T *A, const T *B,
                                   const T *C, T *Y, int M, int N, int K,
                                   int transA, int transB, float alpha,
                                   float beta, int cDim0, int cDim1) {
  if constexpr (HIPDNN_WMMA_FRAG_ELEMS == 16 &&
                std::is_same<T, _Float16>::value) {
    if (M != 1 || N != 4096 || K != 4096 || transA || !transB ||
        alpha != 1.f || beta != 0.f ||
        ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
          reinterpret_cast<uintptr_t>(Y)) & 15) != 0)
      return false;
    static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_DECODENT", 1) != 0;
    static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG", -1);
    static const bool legacy_forced =
        gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG", -1) >= 0;
    if (!on || (forced >= 0 && forced != 0) || legacy_forced ||
        !gemmNnStreamGfx1151())
      return false;
    /* Four outputs share A; 512 threads cover K4096 in one vector pass.
     * Checked five-process screen: 0.0421 -> 0.0356 ms (old -> geometry).
     * Keep an independent instance and only route this measured request. */
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemm_gemv_decode_nt_kernel<T, 512, 4>),
        dim3(N / 4), dim3(512), 0, s, A, B, C, Y, M, N, K, alpha, beta,
        cDim0, cDim1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-decode-nt M=%d N=%d K=%d -> decode-cfg[0] th=512 "
        "tn=4 wgs=%d\n", M, N, K, N / 4);
    return true;
  }
  return false;
}

/* Measured small-M NT routes. Keep these instances out of kWmma so online
 * tuning and its remaps retain their original device code and route ladder. */
template <typename T>
static bool launchMeasuredSmallM(hipStream_t s, const T *A, const T *B,
                                 const T *C, T *Y, int M, int N, int K,
                                 int transA, int transB, float alpha,
                                 float beta, int cDim0, int cDim1) {
  if constexpr (HIPDNN_WMMA_FRAG_ELEMS == 16) {
    if (transA || !transB || alpha != 1.f || beta != 0.f ||
        ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
          reinterpret_cast<uintptr_t>(Y)) & 15) != 0)
      return false;
    static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_SMALLM", 1) != 0;
    static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG", -1);
    static const bool legacy_on =
        gemmEnvInt("HIPDNN_EP_GEMM_SKINNY", 1) != 0 &&
        gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD", 1) != 0;
    static const bool legacy_forced =
        gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG", -1) >= 0;
    if (!on || !legacy_on || legacy_forced || !gemmNnStreamGfx1151())
      return false;
    static const int sw_env = gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE", 0);
    static const bool map_pinned =
        gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP", -1) >= 0;
    const int sw = sw_env != 0 ? sw_env : -8;
    if (M == 32 && N == 4096 && K == 4096) {
      if (forced >= 0 && forced != 0)
        return false;
      const dim3 grid(N / 64, M / 32);
      int stag = gemmStaggerParam(64, sizeof(T), true);
      if (stag != 0 && !map_pinned)
        stag = (stag & ~(3 << 20)) | (1 << 20);
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_bufroll_kernel<32, 64, 2, 2, T, 64, true, 161>),
                         grid, dim3(64), 0, s, A, B, C, Y, M, N, K,
                         alpha, beta, cDim0, cDim1, sw, K, K, stag);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] wmma-smallm M=%d N=%d K=%d -> smallm-cfg[0] "
          "32x64 wt=2x2 bk=64 rp=161 sw=%d stag=%#x wgs=%d\n",
          M, N, K, sw, stag, static_cast<int>(grid.x * grid.y));
      return true;
    }
    if (M == 64 && N == 14336 && K == 4096) {
      if (forced >= 0 && forced != 1)
        return false;
      const dim3 grid(N / 16, M / 16);
      int stag = gemmStaggerParam(64, sizeof(T), true);
      if (stag != 0 && !map_pinned)
        stag = (stag & ~(3 << 20)) | (1 << 20);
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_bufroll_kernel<16, 16, 1, 1, T, 64, true, 161>),
                         grid, dim3(32), 0, s, A, B, C, Y, M, N, K,
                         alpha, beta, cDim0, cDim1, sw, K, K, stag);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] wmma-smallm M=%d N=%d K=%d -> smallm-cfg[1] "
          "16x16 wt=1x1 bk=64 rp=161 sw=%d stag=%#x wgs=%d\n",
          M, N, K, sw, stag, static_cast<int>(grid.x * grid.y));
      return true;
    }
    if constexpr (std::is_same<T, _Float16>::value) {
      if (M == 16 && N == 4096 && K == 4096) {
        if (forced >= 0 && forced != 2)
          return false;
        const dim3 grid(N / 32, M / 16);
        int stag = gemmStaggerParam(64, sizeof(T), true);
        if (stag != 0 && !map_pinned)
          stag = (stag & ~(3 << 20)) | (1 << 20);
        (void)hipGetLastError();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, T, 64, true, 161>),
                           grid, dim3(32), 0, s, A, B, C, Y, M, N, K,
                           alpha, beta, cDim0, cDim1, sw, K, K, stag);
        CUSTOM_KERNELS_DEBUG_LOG(
            "[gemm] wmma-smallm M=%d N=%d K=%d -> smallm-cfg[2] "
            "16x32 wt=1x2 bk=64 rp=161 sw=%d stag=%#x wgs=%d\n",
            M, N, K, sw, stag, static_cast<int>(grid.x * grid.y));
        return true;
      }
    }
    if (M == 128 && N == 4096 && K == 14336) {
      if (forced >= 0 && forced != 3)
        return false;
      const dim3 grid(N / 16, M / 64);
      int stag = gemmStaggerParam(64, sizeof(T), true);
      if (stag != 0 && !map_pinned)
        stag = (stag & ~(3 << 20)) | (1 << 20);
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_bufroll_kernel<64, 16, 1, 1, T, 64, true, 161>),
                         grid, dim3(128), 0, s, A, B, C, Y, M, N, K,
                         alpha, beta, cDim0, cDim1, sw, K, K, stag);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] wmma-smallm M=%d N=%d K=%d -> smallm-cfg[3] "
          "64x16 wt=1x1 bk=64 rp=161 sw=%d stag=%#x wgs=%d\n",
          M, N, K, sw, stag, static_cast<int>(grid.x * grid.y));
      return true;
    }

  }
  return false;
}

/* Independent small-M GEMV template; old device code untouched. */
template <typename T, bool TA, int MR, int TN, int BT>
__global__ void __launch_bounds__(BT)
    gemm_gemv_mr_nt_r2_kernel(const T *__restrict__ A, const T *__restrict__ B,
                           const T *__restrict__ C, T *__restrict__ Y, int M,
                           int N, int K, float alpha, float beta, int cDim0,
                           int cDim1) {
  constexpr int WARP = HIPDNN_WAVE_SIZE;
  constexpr int BLOCK = BT;
  constexpr int KC = BLOCK * 8;
  constexpr int NW = BLOCK / WARP;
  __shared__ __align__(16) T smA[TA ? KC * MR : 8];
  __shared__ float red[NW][MR * TN];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * TN;

  const T *brow[TN];
#pragma unroll
  for (int tn = 0; tn < TN; ++tn)
    brow[tn] = B + static_cast<size_t>(n0 + tn < N ? n0 + tn : 0) * K;
  const bool kvec = (K % 8) == 0;

  float acc[MR][TN];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn)
      acc[m][tn] = 0.0f;

  for (int kc = 0; kc < K; kc += KC) {
    const int klen = (K - kc < KC) ? K - kc : KC;
    if constexpr (TA) {
      /* kc*M*2 bytes = 2048*M*(kc/KC): every chunk start is 16-byte
       * aligned. */
      const T *src = A + static_cast<size_t>(kc) * M;
      const int cnt = klen * M;
      const int cnt8 = cnt / 8;
      for (int i = tid; i < cnt8; i += BLOCK)
        reinterpret_cast<u32x4 *>(smA)[i] =
            reinterpret_cast<const u32x4 *>(src)[i];
      for (int i = cnt8 * 8 + tid; i < cnt; i += BLOCK)
        smA[i] = src[i];
      __syncthreads();
    }

    const int k = tid * 8;
    if (k < klen) {
      const bool full = kvec && k + 8 <= klen;
      float bf[TN][8];
      if (full) {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
          gemmLoad8<T>(brow[tn] + kc + k, bf[tn]);
      } else {
#pragma unroll
        for (int tn = 0; tn < TN; ++tn)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            bf[tn][i] = (k + i < klen) ? GemmElem<T>::tof(brow[tn][kc + k + i])
                                       : 0.0f;
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        if (m < M) {
          float af[8];
          if constexpr (TA) {
#pragma unroll
            for (int i = 0; i < 8; ++i)
              af[i] = (k + i < klen) ? GemmElem<T>::tof(smA[(k + i) * M + m])
                                     : 0.0f;
          } else {
            const T *arow = A + static_cast<size_t>(m) * K + kc + k;
            if (full) {
              gemmLoad8<T>(arow, af);
            } else {
#pragma unroll
              for (int i = 0; i < 8; ++i)
                af[i] = (k + i < klen) ? GemmElem<T>::tof(arow[i]) : 0.0f;
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int tn = 0; tn < TN; ++tn)
              acc[m][tn] = __fmaf_rn(af[i], bf[tn][i], acc[m][tn]);
        }
      }
    }
    if constexpr (TA)
      __syncthreads();
  }

  const int lane = tid % WARP;
  const int warp = tid / WARP;
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int tn = 0; tn < TN; ++tn) {
      float v = acc[m][tn];
      for (int off = WARP >> 1; off > 0; off >>= 1)
        v += __shfl_down(v, off);
      if (lane == 0)
        red[warp][m * TN + tn] = v;
    }
  __syncthreads();
  if (tid < MR * TN) {
    const int m = tid / TN;
    const int n = n0 + tid % TN;
    if (m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; ++w)
        v += red[w][tid];
      gemmStoreY(Y, C, m, n, N, cDim0, cDim1, v, alpha, beta);
    }
  }
}



/* Exact NN packed B rows and wave-local fragment redistribution.
 * No explicit B transpose array or workgroup barrier occurs in the K loop.
 * PIPE selects one register slot or the existing two-slot load/compute order.
 * Compiler shuffles may use LDS instructions; this is not an LDS-free claim. */
template<typename T,int MM,int NN,int KK,int NTL,int KW,int KU,int PIPE>
__global__ void __launch_bounds__(KW*32)
gemm_wmma_nn_direct_r2_kernel(const T* __restrict__ A,const T* __restrict__ B,
                             const T* __restrict__ C,T* __restrict__ Y,
                             int M,int N,int K,float alpha,float beta,int c0,int c1) {
  constexpr int MT=(MM+15)/16,KC=16*KU,STEPS=KK/(KC*KW);
  static_assert(KK%(KC*KW)==0 && NN%(16*NTL)==0,"exact NN chunks");
  static_assert(NTL==4 && (PIPE==0 || PIPE==1),"one complete 128-byte B row");
  static_assert(HIPDNN_WMMA_FRAG_ELEMS==16 || sizeof(T)==0,"gfx11 wave32 fragments");
  using Frag=typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
  const int tid=threadIdx.x,lid=tid%32,lane=lid%16;
  const int w=__builtin_amdgcn_readfirstlane(tid/32),col0=blockIdx.x*16*NTL;
  Frag fa[2][MT][KU],fb[2][NTL][KU];
  float8 acc[MT][NTL];
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)acc[i][j]=float8{};
  auto load=[&](int step,auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    const int k0=(step*KW+w)*KC;
#pragma unroll
    for(int q=0;q<KU;++q) {
#pragma unroll
      for(int i=0;i<MT;++i) {
        const int r=i*16+lane;
        const T* src=A+size_t(r<MM?r:0)*KK+k0+q*16;
        reinterpret_cast<u32x4*>(&fa[S][i][q])[0]=reinterpret_cast<const u32x4*>(src)[0];
        reinterpret_cast<u32x4*>(&fa[S][i][q])[1]=reinterpret_cast<const u32x4*>(src)[1];
      }

#pragma unroll
      for(int e=0;e<16;++e) {
        // Thirty-two distinct aligned dwords cover sixty-four half columns.
        const unsigned word=*reinterpret_cast<const unsigned*>(
            B+size_t(k0+q*16+e)*NN+col0+2*lid);
#pragma unroll
        for(int j=0;j<NTL;++j) {
          const unsigned selected=__shfl(word,(j*16+lane)/2,32);
          const unsigned short bits=static_cast<unsigned short>(selected>>(16*(lane&1)));
          if constexpr(std::is_same<T,_Float16>::value) {
            _Float16 v;__builtin_memcpy(&v,&bits,2);fb[S][j][q][e]=v;
          } else fb[S][j][q][e]=bits;
        }
      }
    }
  };
  auto compute=[&](auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
#pragma unroll
    for(int q=0;q<KU;++q)
#pragma unroll
      for(int i=0;i<MT;++i)
#pragma unroll
        for(int j=0;j<NTL;++j) {
          if constexpr(std::is_same<T,_Float16>::value)
            acc[i][j]=HIPDNN_WMMA_F32_16X16X16_F16(fa[S][i][q],fb[S][j][q],acc[i][j]);
          else acc[i][j]=HIPDNN_WMMA_F32_16X16X16_BF16(fa[S][i][q],fb[S][j][q],acc[i][j]);
        }
  };
  using S0=std::integral_constant<int,0>;
  using S1=std::integral_constant<int,1>;
  if constexpr(PIPE) {
  load(0,S0{});
  int step=0;
#pragma unroll 1
  for(;step+2<STEPS;step+=2) {
    load(step+1,S1{});GEMM_SCHED_FENCE();compute(S0{});
    load(step+2,S0{});GEMM_SCHED_FENCE();compute(S1{});
  }
  if constexpr(STEPS%2==0) {load(step+1,S1{});GEMM_SCHED_FENCE();compute(S0{});compute(S1{});}
  else compute(S0{});
  } else {
#pragma unroll 1
    for(int step=0;step<STEPS;++step) {load(step,S0{});compute(S0{});}
  }
  __shared__ float red[KW>1?KW:1][MT][NTL][8][32];
  if constexpr(KW>1) {
#pragma unroll
    for(int i=0;i<MT;++i)
#pragma unroll
      for(int j=0;j<NTL;++j)
#pragma unroll
        for(int e=0;e<8;++e)red[w][i][j][e][lid]=acc[i][j][e];
    __syncthreads();
  }
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        const int r=i*16+hipdnn_wmma_acc_row(lid/16,e);
        if(r<MM && w==0) {
          float v=acc[i][j][e];
          if constexpr(KW>1) {
            v=0.f;
#pragma unroll
            for(int z=0;z<KW;++z)v+=red[z][i][j][e][lid];
          }
          Y[size_t(r)*NN+col0+j*16+lane]=GemmElem<T>::fromf(v);
        }
      }
}

/* Independent exact NT overload retaining declared StaggerU. */
/* Independent exact-shape native overload; old eight-parameter body is untouched. */
template <typename T, int MT, int NTL, int KW, int KU, bool TA, bool TB,
          bool SB, int MM, int NN, int KK, int SS>
__global__ void __launch_bounds__(KW * 32)
    gemm_wmma_skinny_kernel(const T *__restrict__ A, const T *__restrict__ B,
                            const T *__restrict__ C_arg, T *__restrict__ Y, int M_arg,
                            int N_arg, int K_arg, float alpha_arg, float beta_arg, int cDim0,
                            int cDim1, int mtiles_arg, int stag_arg) {
  static_assert(!TA && TB && KK % (16*KU*KW) == 0 && NN % (16*NTL) == 0,
                "exact native NT chunks");
  constexpr int M=MM,N=NN,K=KK,mtiles=1,stag=SS;
  constexpr float alpha=1.f,beta=0.f;
  const T *const C=nullptr;
  constexpr int FE = HIPDNN_WMMA_FRAG_ELEMS;
  constexpr int FV = FE / 8; /* u32x4 per fragment */
  constexpr int KC = 16 * KU;
  using Frag = typename std::conditional<std::is_same<T, _Float16>::value,
                                         half16, bf16xN>::type;
  /* SB: NT -> B through the wave's LDS tile as a plain row copy; NN -> the
   * 128-byte-line form of B's transposed staging (SkinnyT L128). */
  constexpr bool LB128 = !TB && SB;
  using PA = SkinnyT<T, MT, KC>;
  using PB = SkinnyT<T, NTL, KC, LB128>;
  constexpr bool BCP = TB && SB;
  constexpr int SEG = KC / 8;
  constexpr int RPI = 32 / SEG;
  constexpr int NCP = BCP ? 16 * NTL / RPI : 1;
  static_assert(!BCP || (32 % SEG == 0 && (16 * NTL) % RPI == 0), "SB map");
  constexpr int LA = TA ? 16 * MT * PA::STR : 0;
  constexpr int LB = (TB && !SB) ? 0 : 16 * NTL * PB::STR;
  constexpr int SLOTS = MT * NTL * 8;
  /* The staging tiles (K loop) and the split-K reduction (after it) share
   * one LDS block. */
  constexpr size_t SM_T = static_cast<size_t>(KW) * (LA + LB) * sizeof(T);
  constexpr size_t SM_R = KW > 1 ? static_cast<size_t>(KW) * SLOTS * 32 * 4 : 0;
  __shared__ __align__(16) char sm_raw[(SM_T > SM_R ? SM_T : SM_R) + 16];

  const int tid = threadIdx.x;
  /* Wave index as a scalar: the K loop below is driven by it, and a value the
   * compiler treats as per-lane turns that loop into exec-masked branches
   * with a full vmcnt(0) wait before every chunk's WMMAs. */
  const int w = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int mt = static_cast<int>(blockIdx.x) % mtiles;
  const int nt = static_cast<int>(blockIdx.x) / mtiles;
  const int row0 = mt * 16 * MT;
  const int col0 = nt * 16 * NTL;
  const int koff = hipdnn_wmma_k_off(sub);
  T *tA = reinterpret_cast<T *>(sm_raw) + w * (LA + LB);
  T *tB = tA + LA;
  /* Packed vector loads of a [k][c] operand need every row start aligned to
   * the per-lane column run. */
  const bool vecA = (M % PA::CW) == 0 && M >= PA::CW;
  const bool vecB = (N % PB::CW) == 0 && N >= PB::CW;

  const T *ap[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int r = row0 + i * 16 + lane;
    ap[i] = A + static_cast<size_t>(r < M ? r : 0) * K + koff;
  }
  const T *bp[NTL];
#pragma unroll
  for (int j = 0; j < NTL; ++j) {
    const int c = col0 + j * 16 + lane;
    bp[j] = B + static_cast<size_t>(c < N ? c : 0) * K + koff;
  }
  const T *cp[NCP];
  int cp_lds[NCP];
  const int cp_seg = (lid % SEG) * 8;
#pragma unroll
  for (int q = 0; q < NCP; ++q) {
    const int r = q * RPI + lid / SEG;
    const int c = col0 + r;
    cp[q] = B + static_cast<size_t>(c < N ? c : 0) * K + cp_seg;
    cp_lds[q] = r * PB::STR + cp_seg;
  }

  const int nfull = K / KC;
  int shift = 0;
  if (stag != 0 && nfull > 1)
    shift = ((nt & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % nfull;

  u32x4 fa[2][TA ? 1 : MT][KU][FV];
  u32x4 fb[2][TB ? NTL : 1][KU][FV];
  SkinnyReg<T, TA ? MT : 1, TA ? KC : 32, false> ra[2];
  SkinnyReg<T, TB ? 1 : NTL, TB ? 32 : KC, LB128> rb[2];
  u32x4 rc[2][NCP];
  float8 acc[MT][NTL];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NTL; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  /* Loads chunk k0 .. k0+KC into slot S. TAIL zero-fills k >= K. */
  auto load = [&](int k0, auto slot, auto tail) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr bool TL = decltype(tail)::value;
    if constexpr (TA)
      skinnyLoadT<T, MT, KC, false, TL>(A, M, M, K, k0, row0, vecA, ra[S]);
    if constexpr (!TB)
      skinnyLoadT<T, NTL, KC, LB128, TL>(B, N, N, K, k0, col0, vecB,
                                         rb[S]);
    if constexpr (BCP) {
      const bool kin = !TL || (k0 + cp_seg + 8 <= K);
#pragma unroll
      for (int q = 0; q < NCP; ++q) {
        rc[S][q] = *reinterpret_cast<const u32x4 *>(cp[q] + (kin ? k0 : 0));
        if (TL && !kin)
          rc[S][q] = u32x4{0, 0, 0, 0};
      }
    }
#pragma unroll
    for (int kk = 0; kk < KU; ++kk) {
#pragma unroll
      for (int v = 0; v < FV; ++v) {
        const int k = k0 + kk * 16 + v * 8;
        const bool kin = !TL || (k + koff + 8 <= K);
        const int ke = kin ? k : 0;
        if constexpr (!TA) {
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            fa[S][i][kk][v] = *reinterpret_cast<const u32x4 *>(ap[i] + ke);
            if (TL && !kin)
              fa[S][i][kk][v] = u32x4{0, 0, 0, 0};
          }
        }
        if constexpr (TB && !SB) {
#pragma unroll
          for (int j = 0; j < NTL; ++j) {
            fb[S][j][kk][v] = *reinterpret_cast<const u32x4 *>(bp[j] + ke);
            if (TL && !kin)
              fb[S][j][kk][v] = u32x4{0, 0, 0, 0};
          }
        }
      }
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TA)
      skinnyStoreT<T, MT, KC, false>(tA, ra[S]);
    if constexpr (!TB)
      skinnyStoreT<T, NTL, KC, LB128>(tB, rb[S]);
    if constexpr (BCP) {
#pragma unroll
      for (int q = 0; q < NCP; ++q)
        *reinterpret_cast<u32x4 *>(&tB[cp_lds[q]]) = rc[S][q];
    }
#pragma unroll
    for (int kk = 0; kk < KU; ++kk) {
#pragma unroll
      for (int j = 0; j < NTL; ++j) {
        Frag b;
        if constexpr (TB && !SB) {
          __builtin_memcpy(&b, fb[S][j][kk], sizeof(Frag));
        } else {
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &tB[(j * 16 + lane) * PB::STR + kk * 16 + koff]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b);
#pragma unroll
          for (int v = 0; v < FV; ++v)
            dst[v] = src[v];
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          Frag a;
          if constexpr (TA) {
            const u32x4 *src = reinterpret_cast<const u32x4 *>(
                &tA[(i * 16 + lane) * PA::STR + kk * 16 + koff]);
            u32x4 *dst = reinterpret_cast<u32x4 *>(&a);
#pragma unroll
            for (int v = 0; v < FV; ++v)
              dst[v] = src[v];
          } else {
            __builtin_memcpy(&a, fa[S][i][kk], sizeof(Frag));
          }
          if constexpr (std::is_same<T, _Float16>::value)
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_F16(a, b, acc[i][j]);
          else
            acc[i][j] = HIPDNN_WMMA_F32_16X16X16_BF16(a, b, acc[i][j]);
        }
      }
    }
  };

  using S0 = std::integral_constant<int, 0>;
  using S1 = std::integral_constant<int, 1>;
  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  auto kofs = [&](int c) __attribute__((always_inline)) {
    int ch = c + shift;
    if (ch >= nfull)
      ch -= nfull;
    return ch * KC;
  };
  /* Whole chunks w, w+KW, ... (n of them) with one register double buffer.
   * The steady-state loop issues its loads unconditionally and the last one
   * or two chunks are peeled: a load that may or may not be issued before a
   * compute makes the compiler's vmcnt bookkeeping wait for it either way. */
  const int n = nfull > w ? (nfull - w + KW - 1) / KW : 0;
  if (n > 0) {
    load(kofs(w), S0{}, Full{});
    int i = 0;
    for (; i + 2 < n; i += 2) {
      load(kofs(w + (i + 1) * KW), S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      load(kofs(w + (i + 2) * KW), S0{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S1{});
    }
    if (n - i == 2) {
      load(kofs(w + (i + 1) * KW), S1{}, Full{});
      GEMM_SCHED_FENCE();
      compute(S0{});
      compute(S1{});
    } else {
      compute(S0{});
    }
  }
  /* Partial last chunk (K % KC != 0): one wave, zero-filled past K. */
  if (nfull * KC < K && w == nfull % KW) {
    load(nfull * KC, S0{}, Tail{});
    compute(S0{});
  }

  const bool fast = (beta == 0.0f || !C) && alpha == 1.0f;
  auto store = [&](int i, int j, int e, int l, float v)
      __attribute__((always_inline)) {
    const int r = row0 + i * 16 + hipdnn_wmma_acc_row(l / 16, e);
    const int cc = col0 + j * 16 + l % 16;
    if (r < M && cc < N) {
      if (fast)
        Y[static_cast<size_t>(r) * N + cc] = GemmElem<T>::fromf(v);
      else
        gemmStoreY(Y, C, r, cc, N, cDim0, cDim1, v, alpha, beta);
    }
  };
  if constexpr (KW == 1) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NTL; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          store(i, j, e, lid, acc[i][j][e]);
  } else {
    float(*red)[SLOTS][32] = reinterpret_cast<float(*)[SLOTS][32]>(sm_raw);
    if constexpr (SM_T > 0)
      __syncthreads(); /* other waves may still read their staging tiles */
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NTL; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          red[w][(i * NTL + j) * 8 + e][lid] = acc[i][j][e];
    __syncthreads();
#pragma unroll
    for (int idx = tid; idx < SLOTS * 32; idx += KW * 32) {
      const int slot = idx / 32;
      const int l = idx % 32;
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < KW; ++q)
        v += red[q][slot][l];
      store(slot / (NTL * 8), (slot / 8) % NTL, slot % 8, l, v);
    }
  }
}






/* Exact NN small-M, direct strided B fragments, two register slots.
 * Each wave owns the complete M tile and an interleaved K partition.
 * Scalar half loads are coalesced over the wave's sixteen columns.
 * No B transpose tile and no workgroup barrier occur in the K loop. */
template<typename T,int MM,int NN,int KK,int NTL,int KW,int KU>
__global__ void __launch_bounds__(KW*32)
gemm_wmma_nn_direct_r2_kernel(const T* __restrict__ A,const T* __restrict__ B,
                             const T* __restrict__ C,T* __restrict__ Y,
                             int M,int N,int K,float alpha,float beta,int c0,int c1) {
  constexpr int MT=(MM+15)/16,KC=16*KU,STEPS=KK/(KC*KW);
  static_assert(KK%(KC*KW)==0 && NN%(16*NTL)==0,"exact NN chunks");
  static_assert(HIPDNN_WMMA_FRAG_ELEMS==16 || sizeof(T)==0,"gfx11 wave32 fragments");
  using Frag=typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
  const int tid=threadIdx.x,lid=tid%32,lane=lid%16;
  const int w=__builtin_amdgcn_readfirstlane(tid/32),col0=blockIdx.x*16*NTL;
  Frag fa[2][MT][KU],fb[2][NTL][KU];
  float8 acc[MT][NTL];
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)acc[i][j]=float8{};
  auto load=[&](int step,auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    const int k0=(step*KW+w)*KC;
#pragma unroll
    for(int q=0;q<KU;++q) {
#pragma unroll
      for(int i=0;i<MT;++i) {
        const int r=i*16+lane;
        const T* src=A+size_t(r<MM?r:0)*KK+k0+q*16;
        reinterpret_cast<u32x4*>(&fa[S][i][q])[0]=reinterpret_cast<const u32x4*>(src)[0];
        reinterpret_cast<u32x4*>(&fa[S][i][q])[1]=reinterpret_cast<const u32x4*>(src)[1];
      }
#pragma unroll
      for(int j=0;j<NTL;++j)
#pragma unroll
        for(int e=0;e<16;++e) {
          T v=B[size_t(k0+q*16+e)*NN+col0+j*16+lane];
          if constexpr(std::is_same<T,_Float16>::value)fb[S][j][q][e]=v;
          else {unsigned short bits;__builtin_memcpy(&bits,&v,2);fb[S][j][q][e]=bits;}
        }
    }
  };
  auto compute=[&](auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
#pragma unroll
    for(int q=0;q<KU;++q)
#pragma unroll
      for(int i=0;i<MT;++i)
#pragma unroll
        for(int j=0;j<NTL;++j) {
          if constexpr(std::is_same<T,_Float16>::value)
            acc[i][j]=HIPDNN_WMMA_F32_16X16X16_F16(fa[S][i][q],fb[S][j][q],acc[i][j]);
          else acc[i][j]=HIPDNN_WMMA_F32_16X16X16_BF16(fa[S][i][q],fb[S][j][q],acc[i][j]);
        }
  };
  using S0=std::integral_constant<int,0>;
  using S1=std::integral_constant<int,1>;
  load(0,S0{});
  int step=0;
#pragma unroll 1
  for(;step+2<STEPS;step+=2) {
    load(step+1,S1{});GEMM_SCHED_FENCE();compute(S0{});
    load(step+2,S0{});GEMM_SCHED_FENCE();compute(S1{});
  }
  if constexpr(STEPS%2==0) {load(step+1,S1{});GEMM_SCHED_FENCE();compute(S0{});compute(S1{});}
  else compute(S0{});
  __shared__ float red[KW>1?KW:1][MT][NTL][8][32];
  if constexpr(KW>1) {
#pragma unroll
    for(int i=0;i<MT;++i)
#pragma unroll
      for(int j=0;j<NTL;++j)
#pragma unroll
        for(int e=0;e<8;++e)red[w][i][j][e][lid]=acc[i][j][e];
    __syncthreads();
  }
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        const int r=i*16+hipdnn_wmma_acc_row(lid/16,e);
        if(r<MM && w==0) {
          float v=acc[i][j][e];
          if constexpr(KW>1) {
            v=0.f;
#pragma unroll
            for(int z=0;z<KW;++z)v+=red[z][i][j][e][lid];
          }
          Y[size_t(r)*NN+col0+j*16+lane]=GemmElem<T>::fromf(v);
        }
      }
}

template <typename T, int MR, int LPR, int KW, int U, int NS, int NN, int KK>
__global__ void __launch_bounds__(KW * 32)
    gemm_gemv_kn_raw_r2_kernel(const T *__restrict__ A, const T *__restrict__ B,
                             const T *__restrict__ C, T *__restrict__ Y, int M_arg,
                             int N_arg, int K_arg, float alpha, float beta, int cDim0,
                             int cDim1) {
  constexpr int M=MR,N=NN,K=KK;
  constexpr int VE = 16 / static_cast<int>(sizeof(T));
  constexpr int R = 32 / LPR;
  constexpr int NB = LPR * VE;
  constexpr int GR = R * U;
  constexpr int AW = U * static_cast<int>(sizeof(T)) / 4; /* A dwords */
  static_assert(R * LPR == 32 && NS >= 2, "kn ring shape");
  static_assert(AW >= 1 && (AW <= 2 || AW % 4 == 0), "kn ring A width");
  __shared__ float red[KW > 1 ? KW : 1][MR][NB];

  const int lid = threadIdx.x % 32;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
  const int cl = lid % LPR;
  const int rl = lid / LPR;
  const int nblk = blockIdx.x * NB;
  const int m0 = blockIdx.y * MR;
  const int n0 = nblk + cl * VE;
  const int nc = (n0 + VE <= N) ? n0 : N - VE;
  const size_t ldn = static_cast<size_t>(N);
  const T *bp = B + static_cast<size_t>(rl * U) * ldn + nc;
  const T *ap[MR];
#pragma unroll
  for (int m = 0; m < MR; ++m)
    ap[m] = A + static_cast<size_t>(m0 + m < M ? m0 + m : 0) * K + rl * U;

  const int ng = K / GR;
  constexpr int nw=KK/(GR*KW);
  static_assert(KK%(GR*KW)==0 && nw>=NS,"full ring chunks");

  const auto b_resource=__builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(B),0,NN*KK*int(sizeof(T)),0x31004000);
  u32x4 rb[NS][U];
  uint32_t ra[NS][MR][AW];
  float acc[MR][VE];
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int e = 0; e < VE; ++e)
      acc[m][e] = 0.0f;

  /* the wave's j-th group is group w + j * KW */
  auto load = [&](int j, auto slot, auto clamp) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    const int raw_g=w+j*KW;
    const int g=decltype(clamp)::value && raw_g>=ng ? ng-1 : raw_g;
    const T *b = bp + static_cast<size_t>(g) * GR * ldn;
#pragma unroll
    for (int u = 0; u < U; ++u)
      rb[S][u] = __builtin_amdgcn_raw_buffer_load_b128(b_resource,
          ((rl*U+u)*NN+nc)*int(sizeof(T)),g*GR*NN*int(sizeof(T)),0);
#pragma unroll
    for (int m = 0; m < MR; ++m) {
      const T *src = ap[m] + g * GR;
      if constexpr (AW % 4 == 0) {
#pragma unroll
        for (int v = 0; v < AW / 4; ++v)
          *reinterpret_cast<u32x4 *>(&ra[S][m][4 * v]) =
              reinterpret_cast<const u32x4 *>(src)[v];
      } else if constexpr (AW == 2) {
        *reinterpret_cast<u32x2 *>(&ra[S][m][0]) =
            *reinterpret_cast<const u32x2 *>(src);
      } else {
        ra[S][m][0] = *reinterpret_cast<const uint32_t *>(src);
      }
    }
  };
  auto compute = [&](auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float b[VE];
      if constexpr (std::is_same<T, float>::value) {
#pragma unroll
        for (int e = 0; e < VE; ++e)
          b[e] = __uint_as_float(rb[S][u][e]);
      } else {
        gemmUnpack8<T>(rb[S][u], b);
      }
#pragma unroll
      for (int m = 0; m < MR; ++m) {
        const T *av = reinterpret_cast<const T *>(&ra[S][m][0]);
        const float a = GemmElem<T>::tof(av[u]);
#pragma unroll
        for (int e = 0; e < VE; ++e)
          acc[m][e] = __fmaf_rn(a, b[e], acc[m][e]);
      }
    }
  };
  gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
    load(decltype(s)::value, s, std::false_type{});
  });
  int j = 0;
  for (; j + 2 * NS <= nw; j += NS) {
    gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
      compute(s);
      load(j + NS + decltype(s)::value, s, std::false_type{});
      GEMM_SCHED_FENCE();
    });
  }
  if (j + NS <= nw) {
    gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
      compute(s);
      load(j + NS + decltype(s)::value, s, std::true_type{});
      GEMM_SCHED_FENCE();
    });
    j += NS;
  }
  gemmStaticFor<0, NS>([&](auto s) __attribute__((always_inline)) {
    if (j + decltype(s)::value < nw)
      compute(s);
  });

  /* the R lane rows of a column -> lane row 0 */
#pragma unroll
  for (int m = 0; m < MR; ++m)
#pragma unroll
    for (int e = 0; e < VE; ++e) {
      float v = acc[m][e];
#pragma unroll
      for (int off = LPR; off < 32; off <<= 1)
        v += __shfl_xor(v, off);
      acc[m][e] = v;
    }
  if (rl == 0) {
#pragma unroll
    for (int m = 0; m < MR; ++m)
#pragma unroll
      for (int e = 0; e < VE; ++e)
        red[w][m][cl * VE + e] = acc[m][e];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < MR * NB; i += KW * 32) {
    const int m = i / NB, c = i % NB;
    const int n = nblk + c;
    if (m0 + m < M && n < N) {
      float v = 0.0f;
#pragma unroll
      for (int q = 0; q < KW; ++q)
        v += red[q][m][c];
      gemmStoreY(Y, C, m0 + m, n, N, cDim0, cDim1, v, alpha, beta);
    }
  }
}


template <int BM, int BN, int WT_M, int WT_N, bool TransB, typename T, int BK,
          bool BC = true, bool DB = true, int PGR = 1, bool TransA = false>
__device__ __forceinline__ void GemmWmmaTa6R2Impl(const T *__restrict__ A,
                                              const T *__restrict__ B,
                                              const T *__restrict__ C,
                                              T *__restrict__ Y, int M, int N,
                                              int K, float alpha, float beta,
                                              int cDim0, int cDim1,
                                              int swizzle_n,
                                              float *__restrict__ sk_acc,
                                              int ldb, int n_base, int lda,
                                              int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int PAD_K = 8;
  constexpr int K_STR = BK + PAD_K;
  constexpr int K_STEPS = BK / WMMA_TILE;

  constexpr int A_HPT = (BM * BK) / THR;
  /* 16-byte A chunks when they tile a thread's share; 8-byte ones otherwise
   * (BM = 96 at BK = 16 / 32 gives 12 halfs per thread). */
  constexpr int A_CH = (A_HPT >= 8 && A_HPT % 8 == 0) ? 8 : 4;
  constexpr int A_NCH = A_HPT / A_CH;
  constexpr int A_ROW_SPLIT = BK / A_CH;

  constexpr int B_HPT = (BN * BK) / THR;
  constexpr int B_U32 = B_HPT / 8; /* TransB: 8 halfs per u32x4 */
  constexpr int B_TPR = BK / B_HPT;
  /* Non-power-of-two BN (96 / 80, hipBLASLt's MT96 / MT80 N-tiles): one
   * B_HPT-wide run per thread no longer tiles the BN x BK panel, so B is
   * staged like A -- B_NCH chunks of B_CH halfs per thread, row = chunk /
   * (BK / B_CH). PGR bit 4 asks for the same staging on a power-of-two tile
   * (BK / 8 lanes per B row: one load instruction covers whole 128 B lines
   * of a few rows instead of 16-32 B pieces of many rows). Every other tile
   * keeps the one-run mapping. NT only. */
  constexpr bool B_GEN =
      (PGR & 4) != 0 || (B_TPR * BN != THR) || (BK % B_HPT) != 0;
  constexpr int B_CH = ((BN * BK) % (8 * THR)) == 0 ? 8 : 4;
  constexpr int B_NCH = B_GEN ? (BN * BK) / (B_CH * THR) : 1;
  constexpr int B_ROW_SPLIT = BK / B_CH;
  constexpr int PF_B = B_NCH > 8 ? B_NCH : 8;
  static_assert(!B_GEN || (TransB && B_NCH * B_CH * THR == BN * BK),
                "B_GEN chunking");

  static_assert(A_NCH >= 1, "A_NCH");
  static_assert(A_HPT % A_CH == 0, "A_HPT");
  static_assert(BK % A_CH == 0, "BK/A_CH");
  static_assert(BK % WMMA_TILE == 0, "BK/WMMA");
  static_assert((K_STR * 2) % 16 == 0, "LDS stride 16B");
  static_assert(B_GEN || (B_HPT >= 4 && B_HPT % 4 == 0), "B_HPT");
  static_assert(B_GEN || B_HPT <= 64, "pf_b cap");

  /* TransA: A is stored [K, M] (lda = row stride of that layout, normally M).
   * A thread loads TA_KR consecutive k-rows of one TA_MC-wide m vector (one
   * 4/8/16-byte load per row) and writes the TA_MC transposed k-runs as packed
   * TA_KR-half (8-byte) LDS stores, so the LDS tile keeps the [m][k] layout the
   * WMMA fragment reads expect -- no 2-byte scalar ds_write. Consecutive
   * threads take consecutive m vectors, so each row load is coalesced. */
  constexpr int TA_KR = 4;
  constexpr int TA_MC = (A_HPT / TA_KR >= 8) ? 8 : A_HPT / TA_KR;
  constexpr int TA_NCH = TransA ? A_HPT / (TA_KR * TA_MC) : 1;
  constexpr int TA_MG = BM / TA_MC;
  static_assert(!TransA || (TA_MC == 2 || TA_MC == 4 || TA_MC == 6 || TA_MC == 8), "TA_MC");
  static_assert(!TransA || (BK % TA_KR == 0 && BM % TA_MC == 0 &&
                            TA_NCH * TA_KR * TA_MC == A_HPT),
                "TransA chunking");
  /* NBUF LDS buffers: 2 = double-buffered (default, hides global-load latency
   * behind compute); 1 = single-buffered, half the LDS so two workgroups fit
   * per CU and hide the K-loop barriers across each other -- a win for large
   * LDS-bound tiles (256x128) where double-buffer caps occupancy at 1 WG/CU. */
  /* Double-buffer only when it actually fits in the 64 KB LDS. launchWmma picks
   * db=false for any tile whose double-buffered LDS is too big, so it is always
   * launched single-buffered -- but the dispatch macro still instantiates that
   * tile's dead DB=true variant. Without this cap, a large tile (e.g. a 256-wide
   * BK=64 tile at 108 KB double-buffered) fails the static_assert at compile
   * time even though it is never launched double-buffered. Falling NBUF back to
   * 1 keeps it compilable and never changes a launched tile (the runtime db flag
   * matches this predicate). */
  constexpr int NBUF = (DB && (BM + BN) * K_STR * 2 * 2 <= 65536) ? 2 : 1;
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  /* n_base shifts this launch's column origin. The N-edge grid-split launches
   * the aligned interior with n_base=0 (BC=false) and the ragged edge column
   * tile with n_base=floor(N/BN)*BN (BC=true), so the interior keeps the fast
   * path even when N%BN!=0. */
  const int col0 = n_base + bx * BN;

  constexpr int B_COL_SHIFT = __builtin_ctz(B_TPR > 0 ? B_TPR : 1);
  const int b_col = tid >> B_COL_SHIFT;
  const int b_k8 = (tid & (B_TPR - 1)) * B_HPT;
  const int b_n_g_raw = col0 + b_col;
  const bool b_valid = b_n_g_raw < N;
  const int b_n_g = b_valid ? b_n_g_raw : 0;

  /* NN (TransB=0): one thread owns an N-contiguous chunk of one K-row. */
  constexpr int N_CH = B_HPT;
  constexpr int N_CHUNKS = BN / N_CH;
  const int nn_k = tid / N_CHUNKS;
  const int nn_n = (tid % N_CHUNKS) * N_CH;

  int k_begin = 0;
  int k_end = K;
  if (gridDim.z > 1) {
    const int nkt = (K + BK - 1) / BK;
    const int z = static_cast<int>(blockIdx.z);
    const int sk = static_cast<int>(gridDim.z);
    const int kt0 = static_cast<int>((static_cast<long long>(z) * nkt) / sk);
    const int kt1 =
        static_cast<int>((static_cast<long long>(z + 1) * nkt) / sk);
    k_begin = kt0 * BK;
    k_end = kt1 * BK;
    if (k_end > K)
      k_end = K;
  }

  /* StaggerU: each workgroup starts its K loop `shift` tiles into [k_begin,
   * k_end) and wraps, so workgroups that run in lockstep do not hit the same
   * power-of-two-strided B/A rows in the same cache sets at the same time.
   * stag packs (SU-1) | step_tiles << 8 | map << 20; 0 = off. Only when the K
   * range is a whole number of BK tiles (the wrap needs an exact tile edge). */
  const int span = k_end - k_begin;
  const int niter = (span + BK - 1) / BK;
  int shift = 0;
  if (stag != 0 && (span % BK) == 0 && niter > 1) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter);
  }
  const int k_first = k_begin + shift * BK;

  int a_row[A_NCH];
  int a_koff[A_NCH];
  const T *a_src[A_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int idx = tid + c * THR;
    a_row[c] = idx / A_ROW_SPLIT;
    a_koff[c] = (idx % A_ROW_SPLIT) * A_CH;
    /* BC=false with a ragged M (NT, launchWmma): rows past M re-read the last
     * row (their outputs are masked in the epilogue). */
    const int ar = row0 + a_row[c];
    a_src[c] = A + static_cast<size_t>(BC || ar < M ? ar : M - 1) * lda +
               k_first + a_koff[c];
  }

  /* ldb is the B row stride in elements (K for a dense [N, K] B). Only the
   * row base scales by ldb -- the within-row k range is still [0, K). */
  const T *b_src_nt = B + static_cast<size_t>(b_n_g) * ldb + k_first +
                      b_k8; /* TransB=1 [N,K] */
  int b_row[B_NCH];
  int b_koff[B_NCH];
  bool b_ok[B_NCH];
  const T *b_src_g[B_NCH];
  if constexpr (B_GEN) {
#pragma unroll
    for (int c = 0; c < B_NCH; ++c) {
      const int idx = tid + c * THR;
      b_row[c] = idx / B_ROW_SPLIT;
      b_koff[c] = (idx % B_ROW_SPLIT) * B_CH;
      const int n = col0 + b_row[c];
      b_ok[c] = n < N;
      b_src_g[c] = B + static_cast<size_t>(b_ok[c] ? n : 0) * ldb + k_first +
                   b_koff[c];
    }
  }

  int ta_m[TA_NCH];
  int ta_k[TA_NCH];
  const T *ta_src[TA_NCH];
  if constexpr (TransA) {
#pragma unroll
    for (int c = 0; c < TA_NCH; ++c) {
      const int q = tid + c * THR;
      ta_m[c] = (q % TA_MG) * TA_MC;
      ta_k[c] = (q / TA_MG) * TA_KR;
      ta_src[c] = A + static_cast<size_t>(k_first + ta_k[c]) * lda + row0 +
                  ta_m[c];
    }
  }
  /* Vector A loads need every row start TA_MC-aligned (BC path only; the
   * launcher keeps lda%8==0 for BC=false). */
  const bool ta_vec = (lda % TA_MC) == 0;

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  /* PGR (prefetch global read depth): 1 = the next tile's global loads are in
   * flight during one compute step; 2 = two tiles in flight (register slot
   * per tile), hipBLASLt's PGR2. Only instantiated for the BK=32 NT tiles the
   * aliasing-K launch rule uses (launchWmma); BK=64 tiles lose with it. */
  constexpr int PGRV = PGR & 3; /* bit 4: B staging, bit 8: bias epilogue */
  static_assert(PGRV == 1 || PGRV == 2, "PGR");
  constexpr int NSLOT = PGRV;
  uint32_t pf_a[NSLOT][A_NCH][A_CH / 2];
  uint32_t pf_ta[NSLOT][TA_NCH][TA_KR][TA_MC / 2];
  u32x4 pf_b[NSLOT][PF_B];
  alignas(16) T pf_nn[N_CH];

  auto issueLoads = [&](int k0, auto slot) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    /* After the last tile of the range the source pointers wrap back to
     * k_begin (a no-op without stagger: nothing is loaded after the last tile). */
    const int adv = (k0 + BK >= k_end) ? (BK - span) : BK;
    if constexpr (TransA) {
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
        const int m = row0 + ta_m[c];
#pragma unroll
        for (int r = 0; r < TA_KR; ++r) {
          const T *src = ta_src[c] + static_cast<size_t>(r) * lda;
          const int k = k0 + ta_k[c] + r;
          if (!BC || (ta_vec && k < K && m + TA_MC <= M)) {
            if constexpr (TA_MC == 8)
              *reinterpret_cast<u32x4 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x4 *>(src);
            else if constexpr (TA_MC == 6)
              __builtin_memcpy(pf_ta[S][c][r], src, 12);
            else if constexpr (TA_MC == 4)
              *reinterpret_cast<u32x2 *>(pf_ta[S][c][r]) =
                  *reinterpret_cast<const u32x2 *>(src);
            else
              pf_ta[S][c][r][0] = *reinterpret_cast<const uint32_t *>(src);
          } else {
#pragma unroll
            for (int e = 0; e < TA_MC; ++e)
              reinterpret_cast<T *>(pf_ta[S][c][r])[e] =
                  (k < K && m + e < M) ? src[e] : (T)0;
          }
        }
        ta_src[c] += static_cast<ptrdiff_t>(adv) * lda;
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        if constexpr (!BC) {
          /* Tile fully covers M and K%BK==0: no row/K bounds check. */
          if constexpr (A_CH == 8)
            *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x4 *>(a_src[c]);
          else
            *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                *reinterpret_cast<const u32x2 *>(a_src[c]);
        } else {
          const int row = row0 + a_row[c];
          const int k = k0 + a_koff[c];
          const bool ok = (row < M) && (k + A_CH <= K);
          if (ok) {
            if constexpr (A_CH == 8)
              *reinterpret_cast<u32x4 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x4 *>(a_src[c]);
            else
              *reinterpret_cast<u32x2 *>(pf_a[S][c]) =
                  *reinterpret_cast<const u32x2 *>(a_src[c]);
          } else {
#pragma unroll
            for (int e = 0; e < A_CH / 2; ++e)
              pf_a[S][c][e] = 0;
            if (row < M) {
#pragma unroll
              for (int e = 0; e < A_CH; ++e) {
                const int kk = k + e;
                const T v =
                    (kk < K) ? A[static_cast<size_t>(row) * lda + kk] : (T)0;
                reinterpret_cast<T *>(pf_a[S][c])[e] = v;
              }
            }
          }
        }
        a_src[c] += adv;
      }
    }

    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        if (!BC || (b_ok[c] && k0 + b_koff[c] + B_CH <= K)) {
          if constexpr (B_CH == 8)
            pf_b[S][c] = *reinterpret_cast<const u32x4 *>(b_src_g[c]);
          else
            *reinterpret_cast<u32x2 *>(&pf_b[S][c]) =
                *reinterpret_cast<const u32x2 *>(b_src_g[c]);
        } else {
          pf_b[S][c] = u32x4{0, 0, 0, 0};
          if (b_ok[c]) {
#pragma unroll
            for (int e = 0; e < B_CH; ++e)
              if (k0 + b_koff[c] + e < K)
                reinterpret_cast<T *>(&pf_b[S][c])[e] = b_src_g[c][e];
          }
        }
        b_src_g[c] += adv;
      }
    } else if constexpr (TransB) {
      if (!BC || (b_valid && k0 + b_k8 + B_HPT <= K)) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(pf_b[S]) =
              *reinterpret_cast<const u32x2 *>(b_src_nt);
        } else if constexpr (B_U32 == 1) {
          pf_b[S][0] = *reinterpret_cast<const u32x4 *>(b_src_nt);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            pf_b[S][w] = reinterpret_cast<const u32x4 *>(b_src_nt)[w];
        }
      } else {
#pragma unroll
        for (int w = 0; w < 8; ++w)
          pf_b[S][w] = u32x4{0, 0, 0, 0};
        if (b_valid) {
#pragma unroll
          for (int e = 0; e < B_HPT; ++e) {
            const int kk = k0 + b_k8 + e;
            const T v = (kk < K) ? B[static_cast<size_t>(b_n_g) * ldb + kk]
                                  : (T)0;
            reinterpret_cast<T *>(pf_b[S])[e] = v;
          }
        }
      }
      b_src_nt += adv;
    } else {
      const int gk = k0 + nn_k;
      const int gn = col0 + nn_n;
      if (gn + N_CH <= N && (!BC || gk < K)) {
        if constexpr (N_CH >= 8) {
          constexpr int NV = N_CH / 8;
#pragma unroll
          for (int w = 0; w < NV; ++w)
            reinterpret_cast<u32x4 *>(pf_nn)[w] =
                reinterpret_cast<const u32x4 *>(
                    &B[static_cast<size_t>(gk) * N + gn])[w];
          if constexpr ((N_CH % 8) == 4)
            reinterpret_cast<u32x2 *>(pf_nn)[NV * 2] =
                *reinterpret_cast<const u32x2 *>(
                    &B[static_cast<size_t>(gk) * N + gn + NV * 8]);
        } else {
          *reinterpret_cast<u32x2 *>(pf_nn) =
              *reinterpret_cast<const u32x2 *>(
                  &B[static_cast<size_t>(gk) * N + gn]);
        }
      } else {
#pragma unroll
        for (int e = 0; e < N_CH; ++e)
          pf_nn[e] = (T)0;
        if (gk < K) {
#pragma unroll
          for (int e = 0; e < N_CH; ++e) {
            const int n = gn + e;
            if (n < N)
              pf_nn[e] = B[static_cast<size_t>(gk) * N + n];
          }
        }
      }
    }
  };

  auto storeToShared = [&](int buf, auto slot)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    if constexpr (TransB && B_GEN) {
#pragma unroll
      for (int c = 0; c < B_NCH; ++c) {
        T *dst = &smB[buf][b_row[c]][b_koff[c]];
        if (!BC || b_ok[c]) {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = pf_b[S][c];
          else
            *reinterpret_cast<u32x2 *>(dst) =
                *reinterpret_cast<const u32x2 *>(&pf_b[S][c]);
        } else {
          if constexpr (B_CH == 8)
            *reinterpret_cast<u32x4 *>(dst) = u32x4{0, 0, 0, 0};
          else
            *reinterpret_cast<u32x2 *>(dst) = u32x2{0, 0};
        }
      }
    } else if constexpr (TransB) {
      if (!BC || b_valid) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) =
              *reinterpret_cast<const u32x2 *>(pf_b[S]);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = pf_b[S][w];
        }
      } else {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) = u32x2{0, 0};
        } else {
          const u32x4 z{0, 0, 0, 0};
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = z;
        }
      }
    } else {
#pragma unroll
      for (int e = 0; e < N_CH; ++e)
        smB[buf][nn_n + e][nn_k] = pf_nn[e];
    }

    if constexpr (TransA) {
      /* Transpose TA_KR x TA_MC in registers: column j of the loaded rows is
       * half (j&1) of word j/2 in each row; pack rows (0,1) and (2,3). */
#pragma unroll
      for (int c = 0; c < TA_NCH; ++c) {
#pragma unroll
        for (int j = 0; j < TA_MC; ++j) {
          const uint32_t w0 = pf_ta[S][c][0][j / 2];
          const uint32_t w1 = pf_ta[S][c][1][j / 2];
          const uint32_t w2 = pf_ta[S][c][2][j / 2];
          const uint32_t w3 = pf_ta[S][c][3][j / 2];
          u32x2 v;
          if (j & 1) {
            v[0] = (w0 >> 16) | (w1 & 0xffff0000u);
            v[1] = (w2 >> 16) | (w3 & 0xffff0000u);
          } else {
            v[0] = (w0 & 0xffffu) | (w1 << 16);
            v[1] = (w2 & 0xffffu) | (w3 << 16);
          }
          *reinterpret_cast<u32x2 *>(&smA[buf][ta_m[c] + j][ta_k[c]]) = v;
        }
      }
    } else {
#pragma unroll
      for (int c = 0; c < A_NCH; ++c) {
        T *dst = &smA[buf][a_row[c]][a_koff[c]];
        if constexpr (A_CH == 8)
          *reinterpret_cast<u32x4 *>(dst) =
              *reinterpret_cast<const u32x4 *>(pf_a[S][c]);
        else
          *reinterpret_cast<u32x2 *>(dst) =
              *reinterpret_cast<const u32x2 *>(pf_a[S][c]);
      }
    }
  };

  auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks) {
      const int k_off = hipdnn_wmma_k_off(sub);
      if constexpr (std::is_same<T, _Float16>::value) {
        half16 b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          half16 a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] =
                HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        }
      } else {
        bf16xN b_frag[WT_N];
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int noff = (wcol * WT_N + wn) * WMMA_TILE;
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int moff = (wrow * WT_M + wm) * WMMA_TILE;
          bf16xN a_frag{};
          const u32x4 *src = reinterpret_cast<const u32x4 *>(
              &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
          u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = src[i];
#pragma unroll
          for (int wn = 0; wn < WT_N; ++wn)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn],
                                                       acc[wm][wn]);
        }
      }
    }
  };

  /* Scheduling fence after the next tile's global loads, only on the 128x256
   * WT4x2 BK=32 PGR1 tile: without it the StaggerU build sinks every one of
   * them below the tile's 16 WMMAs, right before the barrier whose ds_store
   * waits for them, so the load latency is never hidden (4096^3 on that tile
   * with padded B: 4.13-4.44 ms vs 3.77 with the fence). A fence on every
   * tile is not a win (bench3 stage3 pin1: phi3 -5%, qwen7b-up +7%,
   * 128x4096^2 +14%). */
  constexpr bool PIN_LD = BM == 128 && BN == 256 && WT_M == 4 && WT_N == 2 &&
                          BK == 32 && PGRV == 1 && !TransA;
  if (k_begin < k_end) {
    using S0 = std::integral_constant<int, 0>;
    auto nextK = [&](int k) __attribute__((always_inline)) {
      k += BK;
      return k >= k_end ? k_begin : k;
    };
    int kc = k_first;
    issueLoads(kc, S0{});
    if constexpr (PIN_LD)
      GEMM_SCHED_FENCE();
    if constexpr (PGRV == 2) {
      using S1 = std::integral_constant<int, 1>;
      if (niter > 1) {
        /* PGR2: tile t lives in slot t%2; each step issues tile it+1 and
         * stores tile it (issued one step earlier), so a tile has two compute
         * steps to arrive. The final step issues one wrapped, valid tile that
         * is never stored, which keeps every step's load count identical. */
        kc = nextK(kc);
        issueLoads(kc, S1{});
        if constexpr (PIN_LD)
          GEMM_SCHED_FENCE();
        storeToShared(0, S0{});
        HIPDNN_LDS_BARRIER();
        int buf = 0;
        int it = 1;
        while (true) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S1{});
            HIPDNN_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            HIPDNN_LDS_BARRIER();
            storeToShared(0, S1{});
            HIPDNN_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
          kc = nextK(kc);
          issueLoads(kc, S1{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          if constexpr (DB) {
            computeWMMA(buf);
            storeToShared(1 - buf, S0{});
            HIPDNN_LDS_BARRIER();
            buf = 1 - buf;
          } else {
            computeWMMA(0);
            HIPDNN_LDS_BARRIER();
            storeToShared(0, S0{});
            HIPDNN_LDS_BARRIER();
          }
          if (++it >= niter)
            break;
        }
        computeWMMA(DB ? buf : 0);
      } else {
        storeToShared(0, S0{});
        HIPDNN_LDS_BARRIER();
        computeWMMA(0);
      }
    } else {
      storeToShared(0, S0{});
      HIPDNN_LDS_BARRIER();

      if constexpr (DB) {
        int buf = 0;
        for (int it = 1; it < niter; ++it) {
          const int nxt = 1 - buf;
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          computeWMMA(buf);
          storeToShared(nxt, S0{});
          HIPDNN_LDS_BARRIER();
          buf = nxt;
        }
        computeWMMA(buf);
      } else {
        /* Single-buffer: the next tile's global loads (issueLoads) still
         * overlap compute via registers, but the LDS is reused, so a barrier
         * must bracket the overwrite -- one after compute finishes reading,
         * one after the store makes the new tile visible. */
        for (int it = 1; it < niter; ++it) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
          if constexpr (PIN_LD)
            GEMM_SCHED_FENCE();
          computeWMMA(0);
          HIPDNN_LDS_BARRIER();
          storeToShared(0, S0{});
          HIPDNN_LDS_BARRIER();
        }
        computeWMMA(0);
      }
    }
  }

  const bool fast = !sk_acc && (beta == 0.0f || !C) && alpha == 1.0f;
  const size_t sk_off =
      sk_acc ? (static_cast<size_t>(blockIdx.z) * static_cast<size_t>(M) * N)
             : 0;

#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  /* Paired store (BC=false, plain Y = A*B, even N): lanes 2j and 2j+1 hold
   * columns c and c+1 of the same accumulator rows, so one DPP swap gives the
   * even lane both columns of row e and the odd lane both columns of row e+1:
   * one 4-byte store per two outputs instead of two 2-byte stores. bench3
   * stage3 c3a, hb/ours with / without: siglip-up 0.91 / 0.85, minilm-up
   * 0.67 / 0.60, bf16 vitb-up 0.91 / 0.86; large shapes within 1%. */
  /* PGR bit 8: the same paired store for Y = alpha * A*B + beta * C with C a
   * row-broadcast vector (cDim0 == 1, the bias epilogue). A lane's columns
   * do not change with the row, so it reads its C value once per fragment
   * column instead of once per output in gemmStoreY. */
  if constexpr (!BC && sizeof(T) == 2 && (PGR & 8) != 0) {
    if (!sk_acc && C && cDim0 == 1 && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
        const int cc = nbase + (lane & ~1);
        const float bc =
            nbase + lane < N
                ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
                : 0.0f;
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = fmaf(alpha, acc[wm][wn][e], bc);
            const float v1 = fmaf(alpha, acc[wm][wn][e + 1], bc);
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
  if constexpr (!BC && sizeof(T) == 2) {
    if (fast && (N & 1) == 0) {
      const bool odd = (lane & 1) != 0;
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
          const int cc = nbase + (lane & ~1);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float v0 = acc[wm][wn][e];
            const float v1 = acc[wm][wn][e + 1];
            const float x = __int_as_float(__builtin_amdgcn_update_dpp(
                0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
            const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
            const T lo = GemmElem<T>::fromf(odd ? x : v0);
            const T hi = GemmElem<T>::fromf(odd ? v1 : x);
            uint32_t pk = static_cast<uint32_t>(
                              *reinterpret_cast<const uint16_t *>(&lo)) |
                          (static_cast<uint32_t>(
                               *reinterpret_cast<const uint16_t *>(&hi))
                           << 16);
            if (cc < N && rr < M)
              *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N +
                                               cc]) = pk;
          }
        }
      }
      return;
    }
  }
#endif

#pragma unroll
  for (int wm = 0; wm < WT_M; ++wm) {
    const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = hipdnn_wmma_acc_row(sub, e);
        const int rr = mbase + r;
        const int cc = nbase + lane;
        const float v = acc[wm][wn][e];
        if constexpr (!BC) {
          /* BC=false requires K%BK==0; M and N may be padded up to whole tiles
           * by the caller (ragged fast path: launch full tiles, mask the row /
           * column overhang here). The store is the epilogue -- one cheap
           * compare per element, outside the K-loop -- so it does not touch
           * the hot path. */
          if (cc < N && rr < M) {
            if (sk_acc)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
            else if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        } else {
          const bool full = (rr < M && cc < N);
          if (sk_acc) {
            if (full)
              sk_acc[sk_off + static_cast<size_t>(rr) * N + cc] = v;
          } else if (full) {
            if (fast)
              Y[static_cast<size_t>(rr) * N + cc] = GemmElem<T>::fromf(v);
            else
              gemmStoreY(Y, C, rr, cc, N, cDim0, cDim1, v, alpha, beta);
          }
        }
      }
    }
  }
}

/* Independent TA loader: six M columns x four K rows/thread. Twelve bytes
 * per K row are copied without imposing eight/sixteen-byte source alignment.
 * Existing packed four-K LDS stores and fragment/epilogue mapping are retained. */
template<int BM,int BN,int WM,int WN,bool TB,typename T,int BK,
         bool BC=true,bool DB=true,int PGR=1,bool TA=true>
__global__ __attribute__((amdgpu_flat_work_group_size(GEMM_WG_SZ(BM,BN,WM,WN),GEMM_WG_SZ(BM,BN,WM,WN))))
__attribute__((amdgpu_waves_per_eu(2,8)))
void gemm_wmma_ta6_r2_kernel(const T* A,const T* B,const T* C,T* Y,
                            int M,int N,int K,float alpha,float beta,int c0,int c1,
                            int sw,float* sk,int ldb,int nbase,int lda,int stag) {
  GemmWmmaTa6R2Impl<BM,BN,WM,WN,TB,T,BK,BC,DB,PGR,TA>(A,B,C,Y,M,N,K,alpha,beta,c0,c1,sw,sk,ldb,nbase,lda,stag);
}












/* Exact stage1 cells; independent instances stay outside kWmma. */
static bool gemmLagR2Allowed(const void*A,const void*B,const void*C,void*Y,
                            float alpha,float beta) {
  if(alpha!=1.f || beta!=0.f || C ||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2",1)!=0;
  static const bool pinned=
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_TCFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP",-1)>=0;
  return on&&!pinned&&gemmNnStreamGfx1151();
}

template<typename T>
static bool launchMeasuredLagR2Half(hipStream_t st,const T*A,const T*B,const T*C,T*Y,
                                  int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    if(!gemmLagR2Allowed(A,B,C,Y,alpha,beta))return false;
    static const bool skinny=gemmEnvInt("HIPDNN_EP_GEMM_SKINNY",1)!=0;
    static const bool smallm=gemmEnvInt("HIPDNN_EP_GEMM_SMALLM",1)!=0;
    static const bool bufload=gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)!=0;
    static const bool nnmid=gemmEnvInt("HIPDNN_EP_GEMM_NNMID",1)!=0;
    static const bool nnstream=gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM",1)!=0;
    if(!skinny || (!ta && tb && (!smallm || !bufload)) || (!ta && !tb && (!nnmid || !nnstream)))return false;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1);
    if constexpr(std::is_same<T,_Float16>::value) {
    /* f16_128_11008_4096_0_0_0: screened P/new/HB 0.471198531/
     * 0.452243898/0.437850282 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==128 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=19)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 64, 64, 4, 2, 32, 0, 2, 0, 1>),dim3(172,2,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=19 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_128_4096_14336_0_0_0: screened P/new/HB 0.728947438/
     * 0.705813058/0.645059339 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==128 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=20)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 128, 128, 4, 2, 32, 0, 2, 1, 1>),dim3(32,1,1),dim3(256,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=20 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_16_11008_4096_0_0_0: screened P/new/HB 0.399176881/
     * 0.383920446/0.385218168 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==16 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=21)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_kernel<16, 128, 1, 2, 0, _Float16, 32, 0, 1, 1, 0>),dim3(86,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,nullptr,11008,0,4096,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=21 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_16_14336_4096_0_1_0: screened P/new/HB 0.616208359/
     * 0.597738125/0.583919688 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==16 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=22)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_skinny_kernel<_Float16, 1, 1, 4, 4, 0, 1, 1, 16, 14336, 4096, 543>),dim3(896,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,543);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=22 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_24_11008_4096_0_0_0: screened P/new/HB 0.400236308/
     * 0.388712000/0.392662667 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==24 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=23)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 32, 64, 2, 1, 32, 0, 2, 0, 1>),dim3(172,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=23 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_24_4096_14336_0_0_0: screened P/new/HB 0.533187597/
     * 0.517300844/0.504857143 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==24 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=24)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 24, 4096, 14336, 4, 2, 1>),dim3(64,1,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=24 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_32_4096_14336_0_0_0: screened P/new/HB 0.539171467/
     * 0.521122267/0.510669000 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==32 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=25)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 32, 4096, 14336, 4, 2, 1>),dim3(64,1,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=25 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_48_11008_4096_0_0_0: screened P/new/HB 0.445754508/
     * 0.403041192/0.398022539 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=26)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 48, 96, 3, 3, 32, 1, 1, 1, 1>),dim3(115,1,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=26 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_48_14336_4096_0_0_0: screened P/new/HB 0.806952556/
     * 0.580010451/0.589054436 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=27)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 48, 14336, 4096, 4, 1, 1, 0>),dim3(224,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=27 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_4_4096_4096_0_1_0: screened P/new/HB 0.039339910/
     * 0.038157005/0.038587680 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==4 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=28)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, _Float16, 64, 1, 1185>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,-8,4096,4096,1049119);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=28 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_64_11008_4096_0_0_0: screened P/new/HB 0.420760263/
     * 0.412173158/0.402620105 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=29)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 64, 96, 4, 3, 32, 0, 1, 1, 1>),dim3(115,1,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=29 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_64_14336_4096_0_0_0: screened P/new/HB 0.737115379/
     * 0.627993712/0.595538409 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=30)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 64, 14336, 4096, 4, 2, 1, 0>),dim3(224,1,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=30 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_64_4096_4096_0_0_0: screened P/new/HB 0.089877292/
     * 0.083507569/0.089715716 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=31)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 64, 64, 4, 2, 64, 0, 2, 1, 1>),dim3(64,1,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=31 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_8_14336_4096_0_1_0: screened P/new/HB 0.604791562/
     * 0.592081797/0.578431250 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=32)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_skinny_kernel<_Float16, 1, 1, 4, 4, 0, 1, 1, 8, 14336, 4096, 543>),dim3(896,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,543);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=32 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_8_4096_4096_0_1_0: screened P/new/HB 0.039891310/
     * 0.038534705/0.038944105 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=33)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, _Float16, 64, 1, 1281>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,-8,4096,4096,1049119);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=33 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_96_11008_4096_0_0_0: screened P/new/HB 0.446442486/
     * 0.440502373/0.441671243 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=34)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 64, 64, 4, 2, 32, 0, 2, 0, 1>),dim3(172,2,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=34 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_96_14336_4096_0_0_0: screened P/new/HB 0.675545794/
     * 0.648451587/0.624892143 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=35)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 96, 96, 3, 3, 32, 0, 1, 1, 1>),dim3(150,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=35 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_96_3072_3072_1_1_0: screened P/new/HB 0.094590652/
     * 0.074399979/0.083832934 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==3072 && K==3072 && ta==1 && tb==1) {
      if(forced>=0 && forced!=36)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_ta6_r2_kernel<96, 96, 3, 3, 1, _Float16, 32, 0, 1, 2, 1>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,nullptr,3072,0,96,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=36 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_96_4096_4096_0_0_0: screened P/new/HB 0.112774650/
     * 0.108872055/0.111824883 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=37)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<_Float16, 96, 64, 3, 2, 32, 1, 2, 1, 1>),dim3(64,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=37 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }

    } else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
    /* bf16_128_11008_4096_0_0_0: screened P/new/HB 0.474289943/
     * 0.452734310/0.439252931 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==128 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 64, 64, 4, 2, 32, 0, 2, 0, 1>),dim3(172,2,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=0 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_128_4096_14336_0_0_0: screened P/new/HB 0.727052222/
     * 0.705886923/0.646921282 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==128 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=1)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 128, 128, 4, 2, 32, 0, 2, 1, 1>),dim3(32,1,1),dim3(256,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=1 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_16_14336_4096_0_1_0: screened P/new/HB 0.613070391/
     * 0.597426953/0.585669531 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==16 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=2)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_skinny_kernel<__hip_bfloat16, 1, 1, 4, 4, 0, 1, 1, 16, 14336, 4096, 543>),dim3(896,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,543);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=2 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_16_4096_4096_0_1_0: screened P/new/HB 0.041343845/
     * 0.038790600/0.039308715 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==16 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=3)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, __hip_bfloat16, 64, 1, 1281>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,-8,4096,4096,1049119);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=3 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_24_11008_4096_0_0_0: screened P/new/HB 0.399811606/
     * 0.387749741/0.393129741 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==24 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=4)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 32, 64, 2, 1, 32, 0, 2, 0, 1>),dim3(172,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=4 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_24_4096_14336_0_0_0: screened P/new/HB 0.534499038/
     * 0.517299551/0.504713910 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==24 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=5)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 24, 4096, 14336, 4, 2, 1>),dim3(64,1,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=5 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_32_11008_4096_0_0_0: screened P/new/HB 0.402359430/
     * 0.391170674/0.394986373 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==32 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=6)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 32, 64, 2, 1, 32, 0, 2, 0, 1>),dim3(172,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=6 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_32_4096_14336_0_0_0: screened P/new/HB 0.539129470/
     * 0.521157748/0.510520132 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==32 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=7)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 32, 4096, 14336, 4, 2, 1>),dim3(64,1,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=7 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_48_11008_4096_0_0_0: screened P/new/HB 0.445651263/
     * 0.404002105/0.398128632 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=8)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 48, 96, 3, 3, 32, 1, 1, 1, 1>),dim3(115,1,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=8 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_48_14336_4096_0_0_0: screened P/new/HB 0.806334511/
     * 0.577363233/0.586746767 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=9)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 48, 14336, 4096, 4, 1, 1, 0>),dim3(224,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=9 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_4_4096_4096_0_1_0: screened P/new/HB 0.037435960/
     * 0.036964495/0.037989710 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==4 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=10)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, __hip_bfloat16, 64, 1, 1185>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,-8,4096,4096,1049119);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=10 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_64_11008_4096_0_0_0: screened P/new/HB 0.421125885/
     * 0.412663385/0.403417083 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=11)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 64, 96, 4, 3, 32, 0, 1, 1, 1>),dim3(115,1,1),dim3(64,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=11 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_64_14336_4096_0_0_0: screened P/new/HB 0.737300534/
     * 0.594643817/0.596002977 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=12)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 64, 14336, 4096, 4, 1, 1, 0>),dim3(224,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=12 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_8_14336_4096_0_1_0: screened P/new/HB 0.606433984/
     * 0.591763359/0.579428125 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=13)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_skinny_kernel<__hip_bfloat16, 1, 1, 4, 4, 0, 1, 1, 8, 14336, 4096, 543>),dim3(896,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,543);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=13 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_8_4096_4096_0_1_0: screened P/new/HB 0.038571305/
     * 0.037355570/0.038333015 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=14)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_smallroll_kernel<16, 32, 1, 2, __hip_bfloat16, 64, 1, 1185>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,-8,4096,4096,1049119);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=14 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_96_11008_4096_0_0_0: screened P/new/HB 0.447317273/
     * 0.442293807/0.445793466 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=15)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 96, 64, 3, 2, 32, 0, 2, 1, 1>),dim3(172,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=15 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_96_14336_4096_0_0_0: screened P/new/HB 0.678626260/
     * 0.656096504/0.624134228 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=16)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 96, 96, 3, 3, 32, 0, 1, 1, 1>),dim3(150,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=16 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_96_3072_3072_1_1_0: screened P/new/HB 0.093629609/
     * 0.073724520/0.083454245 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==3072 && K==3072 && ta==1 && tb==1) {
      if(forced>=0 && forced!=17)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_ta6_r2_kernel<96, 96, 3, 3, 1, __hip_bfloat16, 32, 0, 1, 2, 1>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,nullptr,3072,0,96,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=17 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_96_4096_4096_0_0_0: screened P/new/HB 0.112120801/
     * 0.107490000/0.112275994 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=18)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nnlarge_kernel<__hip_bfloat16, 96, 64, 3, 2, 32, 1, 2, 1, 1>),dim3(64,1,1),dim3(128,1,1),0,st,A,B,Y,M,N,K);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=18 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }

    }
  }
  return false;
}

static bool launchMeasuredLagR2F32(hipStream_t st,const float*A,const float*B,const float*C,float*Y,
                                 int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if(!gemmLagR2Allowed(A,B,C,Y,alpha,beta))return false;
  static const bool nt=gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY",1)!=0;
  static const bool nn=gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN",1)!=0;
  static const bool mr=gemmEnvInt("HIPDNN_EP_GEMM_GEMV_MR",1)!=0;
  static const bool ring=gemmEnvInt("HIPDNN_EP_GEMM_KN_RING",1)!=0;
  if((tb&&!nt)||(!tb&&!nn))return false;
  if((tb&&M<=8&&!mr)||(!tb&&M<=8&&int64_t(N)*K*4<=32*1024*1024&&!ring))return false;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1);
    /* f32_12_14336_4096_0_0_0: screened P/new/HB 1.129020000/
     * 1.010845352/1.079680986 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==12 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=38)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_skinny_nn_kernel<12, 4, 2, 4, 4, 2>),dim3(56,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=38 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_16_14336_4096_0_0_0: screened P/new/HB 1.131834521/
     * 1.025604110/1.078938219 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==16 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=39)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_skinny_nn_kernel<16, 4, 2, 4, 4, 2>),dim3(56,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=39 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_2_4096_14336_0_0_0: screened P/new/HB 1.030479625/
     * 1.002459500/0.984054250 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==2 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=40)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_raw_r2_kernel<float, 2, 32, 4, 2, 2, 4096, 14336>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=40 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_2_4096_4096_0_0_0: screened P/new/HB 0.296517033/
     * 0.290367106/0.284157070 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==2 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=41)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_raw_r2_kernel<float, 2, 32, 4, 2, 2, 4096, 4096>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=41 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_32_4096_14336_0_0_0: screened P/new/HB 1.079035333/
     * 1.046600800/1.040566400 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==32 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=42)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_skinny_nn_kernel<32, 4, 1, 4, 4, 2>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=42 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_48_11008_4096_0_0_0: screened P/new/HB 0.849663196/
     * 0.775169691/0.815746186 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=43)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 64, 192, 16, 8, 12, 128>),dim3(58,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=43 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_48_14336_4096_0_0_0: screened P/new/HB 1.328705156/
     * 1.147179063/1.258499688 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=44)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 64, 256, 16, 8, 8, 256>),dim3(56,1,1),dim3(256,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=44 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_48_4096_4096_0_0_0: screened P/new/HB 0.370114054/
     * 0.354571802/0.355537387 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=45)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>),dim3(128,2,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,-4);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=45 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_4_4096_4096_0_0_0: screened P/new/HB 0.296572074/
     * 0.289831667/0.284978222 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==4 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=46)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_raw_r2_kernel<float, 4, 32, 4, 2, 2, 4096, 4096>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=46 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_4_4096_4096_0_1_0: screened P/new/HB 0.323113690/
     * 0.312508340/0.321901695 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==4 && N==4096 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=47)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_mr_nt_r2_kernel<float, 0, 4, 4, 256>),dim3(1024,1,1),dim3(256,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=47 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_64_11008_4096_0_0_0: screened P/new/HB 0.816636495/
     * 0.787001753/0.813128763 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==11008 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=48)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 64, 192, 16, 8, 12, 128>),dim3(58,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=48 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_64_14336_4096_0_0_0: screened P/new/HB 1.375854844/
     * 1.159105000/1.270747188 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==14336 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=49)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 64, 256, 16, 8, 8, 256>),dim3(56,1,1),dim3(256,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=49 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_64_4096_14336_0_0_0: screened P/new/HB 1.294305781/
     * 1.261564062/1.229542656 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=50)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>),dim3(128,2,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,-4);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=50 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_64_4096_4096_0_0_0: screened P/new/HB 0.376009545/
     * 0.356969818/0.355373455 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=51)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>),dim3(128,2,1),dim3(64,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,-4);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=51 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_6_4096_4096_0_0_0: screened P/new/HB 0.296352230/
     * 0.287543011/0.292898067 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==6 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=52)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_raw_r2_kernel<float, 6, 32, 4, 2, 2, 4096, 4096>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=52 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_8_14336_4096_0_1_0: screened P/new/HB 1.096753828/
     * 1.037559844/1.094079766 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=53)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_f32_skinny_kernel<8, 1, 8, 16, 32, 3>),dim3(3584,1,1),dim3(512,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=53 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_8_4096_4096_0_0_0: screened P/new/HB 0.296214599/
     * 0.287665839/0.288873613 ms, one clean process.
     * Final five-process product results are recorded in autotune/gemm/README.md. */
    if(M==8 && N==4096 && K==4096 && ta==0 && tb==0) {
      if(forced>=0 && forced!=54)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_kn_raw_r2_kernel<float, 8, 32, 4, 2, 2, 4096, 4096>),dim3(32,1,1),dim3(128,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2 route=54 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }

  return false;
}


/* Independent exact measured llm routes. */
/* Exact plain NT LLM scratch hypotheses. The overlap helper uses builtins. */
template<int BM,int BN,int WT_M,int WT_N,typename T,int BK,bool DB,int PGR,
         int SCH,int FIXM,int FIXN,int FIXK,int SW,int STAG>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM,BN,WT_M,WT_N),GEMM_WG_SZ(BM,BN,WT_M,WT_N))))
__attribute__((amdgpu_waves_per_eu(2,8)))
void gemm_wmma_lagr2b_llm_kernel(const T* A,const T* B,const T* C,T* Y,
    int M,int N,int K,float alpha,float beta,int c0,int c1) {
  GemmWmmaOverlapImpl<BM,BN,WT_M,WT_N,true,T,BK,false,DB,PGR,false,SCH>(
      A,B,nullptr,Y,FIXM,FIXN,FIXK,1.f,0.f,1,1,SW,nullptr,FIXK,0,FIXK,STAG);
}

/* Exact Qwen hypotheses: independent fixed-shape entry points. */
template<int BM,int BN,int WT_M,int WT_N,typename T,int BK,bool DB,int PGR,
         int SCH,int FIXM,int FIXN,int FIXK,int SW,int STAG>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM,BN,WT_M,WT_N),GEMM_WG_SZ(BM,BN,WT_M,WT_N))))
__attribute__((amdgpu_waves_per_eu(2,8)))
void gemm_wmma_lagr2b_qwen_kernel(const T* A,const T* B,const T* C,T* Y,
    int M,int N,int K,float alpha,float beta,int c0,int c1) {
  GemmWmmaOverlapImpl<BM,BN,WT_M,WT_N,true,T,BK,false,DB,PGR,false,SCH>(
      A,B,nullptr,Y,FIXM,FIXN,FIXK,1.f,0.f,1,1,SW,nullptr,FIXK,0,FIXK,STAG);
}

static bool gemmLagR2bLlmAllowed(const void*A,const void*B,const void*C,void*Y,
                            float alpha,float beta) {
  if(alpha!=1.f || beta!=0.f || C ||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM",1)!=0;
  static const bool pinned=
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_TCFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP",-1)>=0;
  return on&&!pinned&&gemmNnStreamGfx1151();
}


template<typename T>
static bool launchMeasuredLagR2bLlmHalf(hipStream_t st,const T*A,const T*B,const T*C,T*Y,
                                     int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    bool exact=false;
    if constexpr(std::is_same<T,_Float16>::value)exact=(M==2048 && N==14336 && K==4096 && ta==0 && tb==1) || (M==2048 && N==18944 && K==3584 && ta==0 && tb==1);
    else if constexpr(std::is_same<T,__hip_bfloat16>::value)exact=(M==2048 && N==14336 && K==4096 && ta==0 && tb==1);
    if(!exact)return false;
    if(!gemmLagR2bLlmAllowed(A,B,C,Y,alpha,beta))return false;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",-1);
    if constexpr(std::is_same<T,_Float16>::value) {
    /* f16_2048_14336_4096_0_1_0: screened P/new/HB 6.330459688/
     * 5.841151250/6.174032500 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==2048 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_ALIASK",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_lagr2b_llm_kernel<128,128,2,4,_Float16,64,0,1,7,2048,14336,4096,-16,543>),dim3(112,16),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_llm route=0 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_2048_18944_3584_0_1_0: screened P/new/HB 6.539491563/
     * 6.426931875/6.449810937 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==2048 && N==18944 && K==3584 && ta==0 && tb==1) {
      if(forced>=0 && forced!=2)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_lagr2b_qwen_kernel<128,128,2,4,_Float16,64,0,1,7,2048,18944,3584,8,0>),dim3(148,16),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_llm route=2 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    } else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
    /* bf16_2048_14336_4096_0_1_0: screened P/new/HB 6.308140938/
     * 5.842740313/6.135011250 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==2048 && N==14336 && K==4096 && ta==0 && tb==1) {
      if(forced>=0 && forced!=1)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_ALIASK",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_lagr2b_llm_kernel<128,128,2,4,__hip_bfloat16,64,0,1,7,2048,14336,4096,-16,543>),dim3(112,16),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_llm route=1 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    }
  }
  return false;
}

static bool launchMeasuredLagR2bLlmF32(hipStream_t st,const float*A,const float*B,const float*C,float*Y,
                                    int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if(!(false))return false;
  if(!gemmLagR2bLlmAllowed(A,B,C,Y,alpha,beta))return false;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",-1);
  return false;
}


/* Independent exact measured mid routes. */

template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmMid4bC2Impl(const T *__restrict__ A, const T *__restrict__ B,
               const T *__restrict__ C, T *__restrict__ Y, int M, int N,
               int K, float alpha, float beta, int cDim0, int cDim1,
               int swizzle_n, int ldb, int lda, int stag);

template <typename T, int FM, int FN, int FK, int BM, int BN,
          int WT_M, int WT_N, int BK, bool DB, int RP, int PAD,
          bool XOR, int SW, int STAG>
__global__ void __launch_bounds__((BM / (WT_M * 16)) * (BN / (WT_N * 16)) * 32)
gemm_lagr2b_mid_half_kernel(const T *__restrict__ A, const T *__restrict__ B,
                          const T *__restrict__ C, T *__restrict__ Y,
                          int M, int N, int K, float alpha, float beta,
                          int cDim0, int cDim1) {
  GemmMid4bC2Impl<BM,BN,WT_M,WT_N,T,BK,DB,RP,PAD,XOR,FK,FM,FN,
                  SW,0,1,0>(A,B,C,Y,FM,FN,FK,1.f,0.f,1,1,SW,FK,FK,STAG);
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAGR2B_MID_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while (0)
#else
#define LAGR2B_MID_LDS_BARRIER() __syncthreads()
#endif

__device__ __forceinline__ void gemmMid4bSwizzleC2(int n_tiles, int m_tiles,
                                                int swizzle_n, int &by, int &bx);
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmLagr2bMidPackedImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");


  static_assert(BK == 16 && PAD == 0 && !XOR, "fixed packed BK16 control");
  static_assert(BM * BK % 64 == 0 && BN * BK % 64 == 0, "whole padded blocks");
  constexpr int M_ELS = (BM * BK / 64) * 72;
  constexpr int N_ELS = (BN * BK / 64) * 72;
  static_assert(NBUF * (M_ELS + N_ELS) * 2 <= 65536, "packed LDS 64KB");
  __shared__ __align__(16) T smA[NBUF][M_ELS];
  __shared__ __align__(16) T smB[NBUF][N_ELS];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC2((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (K / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.

  auto ldsOff = [](int r, int k) __attribute__((always_inline)) {
    const int logical = r * BK + k;
    return logical + (logical / 64) * 8;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][ldsOff(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][ldsOff(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][ldsOff(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][ldsOff(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wm * WM_L + wrow) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ldsOff(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wm * WM_L + wrow) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ldsOff(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAGR2B_MID_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAGR2B_MID_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAGR2B_MID_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAGR2B_MID_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAGR2B_MID_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAGR2B_MID_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        const char *base = i < A_NCH ? Ab : Bb;
        pf[0][i] = *reinterpret_cast<const u32x4 *>(base + vo + START * ES);
#else
        if (i < A_NCH)
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, vo, START * ES, 0);
        else
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, vo, START * ES, 0);
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAGR2B_MID_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAGR2B_MID_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wn * WN_L + wcol) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm * WM_L + wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wn * WN_L + wcol) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm * WM_L + wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}

template <typename T, int FM, int FN, int FK, int BM, int BN,
          int WT_M, int WT_N, int BK, bool DB, int RP, int PAD,
          bool XOR, int SW, int STAG>
__global__ void __launch_bounds__((BM / (WT_M * 16)) * (BN / (WT_N * 16)) * 32)
gemm_lagr2b_mid_packed_kernel(const T *__restrict__ A, const T *__restrict__ B,
                            const T *__restrict__ C, T *__restrict__ Y,
                            int M, int N, int K, float alpha, float beta,
                            int cDim0, int cDim1) {
  static_assert(STAG == 0, "unshifted PGR0 and PGR2 controls");
  GemmLagr2bMidPackedImpl<BM,BN,WT_M,WT_N,T,BK,DB,RP,PAD,XOR,FK,FM,FN,
                        SW,0,1,0>(A,B,C,Y,FM,FN,FK,1.f,0.f,1,1,SW,FK,FK,0);
}

template <int FM, int FN, int FK, bool TB, int BM, int BN, int BK,
          int TM, int TN, int THREADS, int SW, int STAG>
__global__ void __launch_bounds__(THREADS)
gemm_lagr2b_mid_f32_kernel(const float *__restrict__ A,
                         const float *__restrict__ B,
                         const float *__restrict__ C, float *__restrict__ Y,
                         int inputM, int inputN, int inputK,
                         float alpha, float beta, int cDim0, int cDim1)
{
  const int M = FM;
  const int N = FN != 0 ? FN : inputN;
  const int K = FK != 0 ? FK : inputK;
  constexpr int swz = SW;
  constexpr int stag = STAG;
  static_assert(FM % BM == 0 && (FN == 0 || FN % BN == 0), "full output tiles");
  static_assert(FK == 0 || FK % BK == 0, "whole K tiles");
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;
  static_assert(TX * TY == THREADS, "thread tile");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "4x4 output blocks");
  static_assert(BK % 8 == 0 && BM >= 16 && (!TB || BN >= 16),
                "loader lane pairs cover 16 rows x 2 k-pieces");
  constexpr int GM = TM / 4;
  constexpr int GN = TN / 4;
  constexpr int SM = BM / GM;
  constexpr int SN = BN / GN;
  constexpr int LDA = BM + 4;
  constexpr int LDB = BN + 4;
  constexpr int KQ = BK / 4; /* 4-float k pieces per row */
  constexpr int A_V4 = BM * KQ / THREADS;
  static_assert(A_V4 >= 1 && A_V4 * THREADS == BM * KQ, "A load split");
  constexpr int BQ = TB ? KQ : BN / 4;
  constexpr int B_ROWS = TB ? BN : BK;
  constexpr int B_V4 = B_ROWS * BQ / THREADS;
  static_assert(B_V4 >= 1 && B_V4 * THREADS == B_ROWS * BQ, "B load split");

  __shared__ __align__(16) float smA[2][BK][LDA];
  __shared__ __align__(16) float smB[2][BK][LDB];

  const int tid = threadIdx.x;
  const int tx = tid % TX;
  const int ty = tid / TX;
  int by = blockIdx.y, bx = blockIdx.x;
  if (swz != 0)
    gemmSwizzleBlock(gridDim.x, gridDim.y, swz, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  /* k-contiguous operand piece idx -> (row, k): lane pairs take two adjacent
   * k pieces of one row, 16 rows per wave. */
  auto pairPiece = [](int idx, int rows, int &r, int &k)
      __attribute__((always_inline)) {
    r = (idx >> 1) % rows;
    k = ((idx & 1) | ((idx / (2 * rows)) << 1)) * 4;
  };

  const float *a_src[A_V4];
  int a_r[A_V4], a_k[A_V4];
#pragma unroll
  for (int v = 0; v < A_V4; ++v) {
    pairPiece(tid + v * THREADS, BM, a_r[v], a_k[v]);
    const int gr = row0 + a_r[v];
    a_src[v] = A + static_cast<size_t>(gr) * K + a_k[v];
  }
  /* NT: b_r is the tile's n row and b_c its k offset. NN: b_r is the k row
   * and b_c the n offset (16-byte pieces along N, stored as they are). */
  const float *b_src[B_V4];
  int b_r[B_V4], b_c[B_V4];
#pragma unroll
  for (int v = 0; v < B_V4; ++v) {
    const int idx = tid + v * THREADS;
    if constexpr (TB) {
      pairPiece(idx, BN, b_r[v], b_c[v]);
      const int gn = col0 + b_r[v];
      b_src[v] = B + static_cast<size_t>(gn) * K + b_c[v];
    } else {
      b_r[v] = idx / BQ;
      b_c[v] = (idx % BQ) * 4;
      const int gn = col0 + b_c[v];
      b_src[v] = B + static_cast<size_t>(b_r[v]) * N + gn;
    }
  }

  gemmF4 ra[A_V4], rb[B_V4];
  /* TAIL: the tile may run past K. K % 4 == 0, so a 4-float k piece is either
   * all inside K or all outside. */
  auto load = [&](int k0, auto tail) __attribute__((always_inline)) {
    constexpr bool TAIL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int v = 0; v < A_V4; ++v)
      ra[v] = (!TAIL || k0 + a_k[v] < K)
                  ? *reinterpret_cast<const gemmF4 *>(a_src[v] + k0)
                  : zero;
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB)
        rb[v] = (!TAIL || k0 + b_c[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(b_src[v] + k0)
                    : zero;
      else
        rb[v] = (!TAIL || k0 + b_r[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(
                          b_src[v] + static_cast<size_t>(k0) * N)
                    : zero;
    }
  };

  auto store = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int v = 0; v < A_V4; ++v) {
      smA[buf][a_k[v] + 0][a_r[v]] = ra[v].x;
      smA[buf][a_k[v] + 1][a_r[v]] = ra[v].y;
      smA[buf][a_k[v] + 2][a_r[v]] = ra[v].z;
      smA[buf][a_k[v] + 3][a_r[v]] = ra[v].w;
    }
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB) {
        smB[buf][b_c[v] + 0][b_r[v]] = rb[v].x;
        smB[buf][b_c[v] + 1][b_r[v]] = rb[v].y;
        smB[buf][b_c[v] + 2][b_r[v]] = rb[v].z;
        smB[buf][b_c[v] + 3][b_r[v]] = rb[v].w;
      } else {
        *reinterpret_cast<gemmF4 *>(&smB[buf][b_r[v]][b_c[v]]) = rb[v];
      }
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  auto compute = [&](int buf) __attribute__((always_inline)) {
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int g = 0; g < GM; ++g) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smA[buf][kk][g * SM + ty * 4]);
        a[g * 4 + 0] = v.x;
        a[g * 4 + 1] = v.y;
        a[g * 4 + 2] = v.z;
        a[g * 4 + 3] = v.w;
      }
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smB[buf][kk][h * SN + tx * 4]);
        b[h * 4 + 0] = v.x;
        b[h * 4 + 1] = v.y;
        b[h * 4 + 2] = v.z;
        b[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      GEMM_SCHED_FENCE();
    }
    /* Keep the FMAs here: without this the compiler sinks the last tile's
     * FMAs into the epilogue branches and hoists all its LDS reads (256 VGPRs
     * and scratch). */
    __builtin_amdgcn_sched_barrier(0);
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int nfull = K / BK;
  const int ntiles = nfull;
  /* StaggerU over the whole tiles (stag = tiles per step, 0 = off): dispatch
   * neighbours start their K loop at different offsets and wrap, so a K whose
   * row stride is a large power of two does not send every workgroup's rows
   * to the same memory channels at once. The tail tile always comes last. */
  const int shift =
      (stag > 0 && nfull > 1)
          ? ((static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x) & 31) *
             stag) %
                nfull
          : 0;
  auto kofs = [&](int i) __attribute__((always_inline)) {
    int kt = i + shift;
    if (kt >= nfull)
      kt -= nfull;
    return kt * BK;
  };
  load(kofs(0), Full{});
  store(0);
  LAGR2B_MID_LDS_BARRIER();
  /* Whole tiles first so the loop body is one straight-line load / compute /
   * store / barrier; the zero-filled tail tile and the last tile follow it. */
  int t = 0;
  for (; t + 1 < nfull; ++t) {
    load(kofs(t + 1), Full{});
    compute(t & 1);
    store((t + 1) & 1);
    LAGR2B_MID_LDS_BARRIER();
  }
  compute(t & 1);


#pragma unroll
  for (int g = 0; g < GM; ++g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = row0 + g * SM + ty * 4 + i;
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const int c = col0 + h * SN + tx * 4;
        const gemmF4 out = gemmF4{acc[g*4+i][h*4], acc[g*4+i][h*4+1],
                                 acc[g*4+i][h*4+2], acc[g*4+i][h*4+3]};
        *reinterpret_cast<gemmF4 *>(Y + static_cast<size_t>(r) * N + c) = out;
      }
    }
  }
}

#undef LAGR2B_MID_LDS_BARRIER


template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmMid4bC2Impl(const T *__restrict__ A, const T *__restrict__ B,
               const T *__restrict__ C, T *__restrict__ Y, int M, int N,
               int K, float alpha, float beta, int cDim0, int cDim1,
               int swizzle_n, int ldb, int lda, int stag);
__device__ __forceinline__ void gemmMid4bSwizzleC2(int n_tiles, int m_tiles,
                                                int swizzle_n, int &by, int &bx);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAGR2B_MID02A_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while (0)
#else
#define LAGR2B_MID02A_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmLagr2bMidPacked02aImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");


  static_assert(BK == 16 && PAD == 0 && !XOR, "fixed packed BK16 control");
  static_assert(BM * BK % 64 == 0 && BN * BK % 64 == 0, "whole padded blocks");
  constexpr int M_ELS = (BM * BK / 64) * 72;
  constexpr int N_ELS = (BN * BK / 64) * 72;
  static_assert(NBUF * (M_ELS + N_ELS) * 2 <= 65536, "packed LDS 64KB");
  __shared__ __align__(16) T smA[NBUF][M_ELS];
  __shared__ __align__(16) T smB[NBUF][N_ELS];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC2((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (K / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.

  auto ldsOff = [](int r, int k) __attribute__((always_inline)) {
    const int logical = r * BK + k;
    return logical + (logical / 64) * 8;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][ldsOff(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][ldsOff(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][ldsOff(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][ldsOff(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wm * WM_L + wrow) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ldsOff(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wn * WN_L + wcol) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][ldsOff(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wm * WM_L + wrow) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ldsOff(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAGR2B_MID02A_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAGR2B_MID02A_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAGR2B_MID02A_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAGR2B_MID02A_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAGR2B_MID02A_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAGR2B_MID02A_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        const char *base = i < A_NCH ? Ab : Bb;
        pf[0][i] = *reinterpret_cast<const u32x4 *>(base + vo + START * ES);
#else
        if (i < A_NCH)
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, vo, START * ES, 0);
        else
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, vo, START * ES, 0);
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAGR2B_MID02A_BARRIER();
    storeAll(I0{}, I0{});
    LAGR2B_MID02A_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wn * WN_L + wcol) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm * WM_L + wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wn * WN_L + wcol) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wm * WM_L + wrow) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T, int FM, int FN, int FK, int BM, int BN,
          int WT_M, int WT_N, int BK, bool DB, int RP, int PAD,
          bool XOR, int SW, int STAG, int CF, int SF, bool PACKED>
__global__ void __launch_bounds__((BM / (WT_M * 16)) * (BN / (WT_N * 16)) * 32)
gemm_lagr2b_mid02a_half_kernel(const T *__restrict__ A, const T *__restrict__ B,
                            const T *__restrict__ C, T *__restrict__ Y,
                            int M, int N, int K, float alpha, float beta,
                            int cDim0, int cDim1) {
  if constexpr (PACKED)
    GemmLagr2bMidPacked02aImpl<BM,BN,WT_M,WT_N,T,BK,DB,RP,PAD,XOR,FK,FM,FN,
                              SW,CF,SF,0>(A,B,C,Y,FM,FN,FK,1.f,0.f,1,1,SW,FK,FK,STAG);
  else
    GemmMid4bC2Impl<BM,BN,WT_M,WT_N,T,BK,DB,RP,PAD,XOR,FK,FM,FN,
                     SW,CF,SF,0>(A,B,C,Y,FM,FN,FK,1.f,0.f,1,1,SW,FK,FK,STAG);
}
template <int FM, int FN, int FK, bool TB, int BM, int BN, int BK,
          int TM, int TN, int THREADS, int SW, int STAG, int SPLIT, bool DB>
__global__ void __launch_bounds__(THREADS * SPLIT)
gemm_lagr2b_mid02a_f32_kernel(const float *__restrict__ A,
                         const float *__restrict__ B,
                         const float *__restrict__ C, float *__restrict__ Y,
                         int inputM, int inputN, int inputK,
                         float alpha, float beta, int cDim0, int cDim1)
{
  const int M = FM;
  const int N = FN != 0 ? FN : inputN;

  const int totalK = FK != 0 ? FK : inputK;
  const int K = totalK / SPLIT;
  static_assert(SPLIT == 1 || SPLIT == 2, "one or two intra-WG K groups");
  static_assert(THREADS == 128 && (FK == 0 || FK % (SPLIT * BK) == 0), "whole K partitions");
  const int group = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x) / THREADS);
  const int groupK = group * K;

  constexpr int swz = SW;
  constexpr int stag = STAG;
  static_assert(FM % BM == 0 && (FN == 0 || FN % BN == 0), "full output tiles");
  static_assert(FK == 0 || FK % BK == 0, "whole K tiles");
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;
  static_assert(TX * TY == THREADS, "thread tile");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "4x4 output blocks");
  static_assert(BK % 8 == 0 && BM >= 16 && (!TB || BN >= 16),
                "loader lane pairs cover 16 rows x 2 k-pieces");
  constexpr int GM = TM / 4;
  constexpr int GN = TN / 4;
  constexpr int SM = BM / GM;
  constexpr int SN = BN / GN;
  constexpr int LDA = BM + 4;
  constexpr int LDB = BN + 4;
  constexpr int KQ = BK / 4; /* 4-float k pieces per row */
  constexpr int A_V4 = BM * KQ / THREADS;
  static_assert(A_V4 >= 1 && A_V4 * THREADS == BM * KQ, "A load split");
  constexpr int BQ = TB ? KQ : BN / 4;
  constexpr int B_ROWS = TB ? BN : BK;
  constexpr int B_V4 = B_ROWS * BQ / THREADS;
  static_assert(B_V4 >= 1 && B_V4 * THREADS == B_ROWS * BQ, "B load split");


  constexpr int NBUF = DB ? 2 : 1;
  constexpr int STAGING = NBUF * BK * (LDA + LDB);
  constexpr int PARTIALS = BM * BN;
  constexpr int SLAB = SPLIT > 1 && PARTIALS > STAGING ? PARTIALS : STAGING;
  static_assert(SPLIT * SLAB * sizeof(float) <= 65536, "intra-WG LDS budget");
  __shared__ __align__(16) float storage[SPLIT][SLAB];
  float (*smA)[BK][LDA] = reinterpret_cast<float (*)[BK][LDA]>(storage[group]);
  float (*smB)[BK][LDB] = reinterpret_cast<float (*)[BK][LDB]>(storage[group] + NBUF * BK * LDA);


  const int tid = static_cast<int>(threadIdx.x) % THREADS;
  const int tx = tid % TX;
  const int ty = tid / TX;
  int by = blockIdx.y, bx = blockIdx.x;
  if (swz != 0)
    gemmSwizzleBlock(gridDim.x, gridDim.y, swz, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  /* k-contiguous operand piece idx -> (row, k): lane pairs take two adjacent
   * k pieces of one row, 16 rows per wave. */
  auto pairPiece = [](int idx, int rows, int &r, int &k)
      __attribute__((always_inline)) {
    /* Each wave has sixteen row pairs. Keep the row-group quotient
     * uniform so non-power-of-two rows do not narrow lane arithmetic. */
    const int lane = idx & 31;
    const int wave = __builtin_amdgcn_readfirstlane(idx >> 5);
    const int rowGroups = rows / 16;
    r = (wave % rowGroups) * 16 + (lane >> 1);
    k = ((lane & 1) | ((wave / rowGroups) << 1)) * 4;
  };

  const float *a_src[A_V4];
  int a_r[A_V4], a_k[A_V4];
#pragma unroll
  for (int v = 0; v < A_V4; ++v) {
    pairPiece(tid + v * THREADS, BM, a_r[v], a_k[v]);
    const int gr = row0 + a_r[v];
    a_src[v] = A + static_cast<size_t>(gr) * totalK + groupK + a_k[v];
  }
  /* NT: b_r is the tile's n row and b_c its k offset. NN: b_r is the k row
   * and b_c the n offset (16-byte pieces along N, stored as they are). */
  const float *b_src[B_V4];
  int b_r[B_V4], b_c[B_V4];
#pragma unroll
  for (int v = 0; v < B_V4; ++v) {
    const int idx = tid + v * THREADS;
    if constexpr (TB) {
      pairPiece(idx, BN, b_r[v], b_c[v]);
      const int gn = col0 + b_r[v];
      b_src[v] = B + static_cast<size_t>(gn) * totalK + groupK + b_c[v];
    } else {
      b_r[v] = idx / BQ;
      b_c[v] = (idx % BQ) * 4;
      const int gn = col0 + b_c[v];
      b_src[v] = B + static_cast<size_t>(groupK + b_r[v]) * N + gn;
    }
  }

  gemmF4 ra[A_V4], rb[B_V4];
  /* TAIL: the tile may run past K. K % 4 == 0, so a 4-float k piece is either
   * all inside K or all outside. */
  auto load = [&](int k0, auto tail) __attribute__((always_inline)) {
    constexpr bool TAIL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int v = 0; v < A_V4; ++v)
      ra[v] = (!TAIL || k0 + a_k[v] < K)
                  ? *reinterpret_cast<const gemmF4 *>(a_src[v] + k0)
                  : zero;
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB)
        rb[v] = (!TAIL || k0 + b_c[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(b_src[v] + k0)
                    : zero;
      else
        rb[v] = (!TAIL || k0 + b_r[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(
                          b_src[v] + static_cast<size_t>(k0) * N)
                    : zero;
    }
  };

  auto store = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int v = 0; v < A_V4; ++v) {
      smA[buf][a_k[v] + 0][a_r[v]] = ra[v].x;
      smA[buf][a_k[v] + 1][a_r[v]] = ra[v].y;
      smA[buf][a_k[v] + 2][a_r[v]] = ra[v].z;
      smA[buf][a_k[v] + 3][a_r[v]] = ra[v].w;
    }
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB) {
        smB[buf][b_c[v] + 0][b_r[v]] = rb[v].x;
        smB[buf][b_c[v] + 1][b_r[v]] = rb[v].y;
        smB[buf][b_c[v] + 2][b_r[v]] = rb[v].z;
        smB[buf][b_c[v] + 3][b_r[v]] = rb[v].w;
      } else {
        *reinterpret_cast<gemmF4 *>(&smB[buf][b_r[v]][b_c[v]]) = rb[v];
      }
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  auto compute = [&](int buf) __attribute__((always_inline)) {
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int g = 0; g < GM; ++g) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smA[buf][kk][g * SM + ty * 4]);
        a[g * 4 + 0] = v.x;
        a[g * 4 + 1] = v.y;
        a[g * 4 + 2] = v.z;
        a[g * 4 + 3] = v.w;
      }
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smB[buf][kk][h * SN + tx * 4]);
        b[h * 4 + 0] = v.x;
        b[h * 4 + 1] = v.y;
        b[h * 4 + 2] = v.z;
        b[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      GEMM_SCHED_FENCE();
    }
    /* Keep the FMAs here: without this the compiler sinks the last tile's
     * FMAs into the epilogue branches and hoists all its LDS reads (256 VGPRs
     * and scratch). */
    __builtin_amdgcn_sched_barrier(0);
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int nfull = K / BK;
  const int ntiles = nfull;
  /* StaggerU over the whole tiles (stag = tiles per step, 0 = off): dispatch
   * neighbours start their K loop at different offsets and wrap, so a K whose
   * row stride is a large power of two does not send every workgroup's rows
   * to the same memory channels at once. The tail tile always comes last. */
  const int shift =
      (stag > 0 && nfull > 1)
          ? ((static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x) & 31) *
             stag) %
                nfull
          : 0;
  auto kofs = [&](int i) __attribute__((always_inline)) {
    int kt = i + shift;
    if (kt >= nfull)
      kt -= nfull;
    return kt * BK;
  };
  load(kofs(0), Full{});
  store(0);
  LAGR2B_MID02A_BARRIER();
  /* Whole tiles first so the loop body is one straight-line load / compute /
   * store / barrier; the zero-filled tail tile and the last tile follow it. */
  int t = 0;
  for (; t + 1 < nfull; ++t) {
    load(kofs(t + 1), Full{});
    compute(DB ? (t & 1) : 0);
    if constexpr (!DB) {
      LAGR2B_MID02A_BARRIER();
      GEMM_SCHED_FENCE();
    }
    store(DB ? ((t + 1) & 1) : 0);
    LAGR2B_MID02A_BARRIER();
  }
  compute(DB ? (t & 1) : 0);


  if constexpr (SPLIT > 1) {
    LAGR2B_MID02A_BARRIER();
    GEMM_SCHED_FENCE();
    if (group == 1) {
#pragma unroll
      for (int g = 0; g < GM; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int h = 0; h < GN; ++h) {
            const int r = g * SM + ty * 4 + i;
            const int c = h * SN + tx * 4;
            const gemmF4 v{acc[g*4+i][h*4], acc[g*4+i][h*4+1],
                           acc[g*4+i][h*4+2], acc[g*4+i][h*4+3]};
            *reinterpret_cast<gemmF4 *>(&storage[1][r * BN + c]) = v;
          }
    }
    LAGR2B_MID02A_BARRIER();
    GEMM_SCHED_FENCE();
    if (group != 0) return;
#pragma unroll
    for (int g = 0; g < GM; ++g)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int h = 0; h < GN; ++h) {
          const int r = g * SM + ty * 4 + i;
          const int c = h * SN + tx * 4;
          const gemmF4 v = *reinterpret_cast<const gemmF4 *>(&storage[1][r * BN + c]);
          acc[g*4+i][h*4] += v.x;
          acc[g*4+i][h*4+1] += v.y;
          acc[g*4+i][h*4+2] += v.z;
          acc[g*4+i][h*4+3] += v.w;
        }
  }

#pragma unroll
  for (int g = 0; g < GM; ++g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = row0 + g * SM + ty * 4 + i;
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const int c = col0 + h * SN + tx * 4;
        const gemmF4 out = gemmF4{acc[g*4+i][h*4], acc[g*4+i][h*4+1],
                                 acc[g*4+i][h*4+2], acc[g*4+i][h*4+3]};
        *reinterpret_cast<gemmF4 *>(Y + static_cast<size_t>(r) * N + c) = out;
      }
    }
  }
}

#undef LAGR2B_MID02A_BARRIER


__device__ __forceinline__ void gemmMid4bSwizzleC2(int n_tiles, int m_tiles,
                                                int swizzle_n, int &by, int &bx);
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmMid4bC2Impl(const T *__restrict__ A, const T *__restrict__ B,
               const T *__restrict__ C, T *__restrict__ Y, int M, int N,
               int K, float alpha, float beta, int cDim0, int cDim1,
               int swizzle_n, int ldb, int lda, int stag);
template <typename T, int CF, int SF>
__global__ void __launch_bounds__(128)
gemm_lagr2b_mid11_vit_kernel(const T *__restrict__ A, const T *__restrict__ B,
                           const T *__restrict__ C, T *__restrict__ Y,
                           int M, int N, int K, float alpha, float beta,
                           int cDim0, int cDim1) {
  GemmMid4bC2Impl<160,96,5,3,T,64,false,5,0,true,3072,1576,768,-8,CF,SF,0>(
      A,B,C,Y,1576,768,3072,1.f,0.f,1,1,-8,3072,3072,543);
}


template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmMid4bC2Impl(const T *__restrict__ A, const T *__restrict__ B,
               const T *__restrict__ C, T *__restrict__ Y, int M, int N,
               int K, float alpha, float beta, int cDim0, int cDim1,
               int swizzle_n, int ldb, int lda, int stag);
template <typename T, int FM, int FN, int FK, int BM, int BN,
          int WT_M, int WT_N, int BK, int CF, int STAG>
__global__ void __launch_bounds__((BM / (WT_M * 16)) * (BN / (WT_N * 16)) * 32)
gemm_lagr2b_mid05_vit_kernel(const T *__restrict__ A, const T *__restrict__ B,
                           const T *__restrict__ C, T *__restrict__ Y,
                           int M, int N, int K, float alpha, float beta,
                           int cDim0, int cDim1) {
  GemmMid4bC2Impl<BM,BN,WT_M,WT_N,T,BK,true,2,0,true,FK,FM,FN,-8,CF,3,2>(
      A,B,C,Y,FM,FN,FK,1.f,0.f,1,1,-8,FK,FK,STAG);
}

static bool gemmLagR2bMidAllowed(const void*A,const void*B,const void*C,void*Y,
                            float alpha,float beta) {
  if(alpha!=1.f || beta!=0.f || C ||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID",1)!=0;
  static const bool pinned=
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_TCFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP",-1)>=0;
  return on&&!pinned&&gemmNnStreamGfx1151();
}


template<typename T>
static bool launchMeasuredLagR2bMidHalf(hipStream_t st,const T*A,const T*B,const T*C,T*Y,
                                     int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    bool exact=false;
    if constexpr(std::is_same<T,_Float16>::value)exact=(M==512 && N==1536 && K==384 && ta==0 && tb==1) || (M==1576 && N==768 && K==3072 && ta==0 && tb==1);
    else if constexpr(std::is_same<T,__hip_bfloat16>::value)exact=(M==512 && N==1536 && K==384 && ta==0 && tb==1) || (M==729 && N==4304 && K==1152 && ta==0 && tb==1) || (M==1576 && N==768 && K==3072 && ta==0 && tb==1);
    if(!exact)return false;
    if(!gemmLagR2bMidAllowed(A,B,C,Y,alpha,beta))return false;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID_CFG",-1);
    if constexpr(std::is_same<T,_Float16>::value) {
    /* f16_512_1536_384_0_1_0: screened P/new/HB 0.019855810/
     * 0.018981940/0.019905910 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==512 && N==1536 && K==384 && ta==0 && tb==1) {
      if(forced>=0 && forced!=0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid_half_kernel<_Float16,512,1536,384,128,80,2,5,64,0,2,8,0,8,0>),dim3(20,4),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=0 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_1576_768_3072_0_1_0: screened P/new/HB 0.178570000/
     * 0.171824063/0.174874375 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==1576 && N==768 && K==3072 && ta==0 && tb==1) {
      if(forced>=0 && forced!=5)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid11_vit_kernel<_Float16,0,3>),dim3(8,10),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=5 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    } else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
    /* bf16_512_1536_384_0_1_0: screened P/new/HB 0.019532910/
     * 0.018903485/0.019830205 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==512 && N==1536 && K==384 && ta==0 && tb==1) {
      if(forced>=0 && forced!=1)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid_half_kernel<__hip_bfloat16,512,1536,384,128,80,2,5,64,0,2,8,0,8,0>),dim3(20,4),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=1 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_729_4304_1152_0_1_0: screened P/new/HB 0.201372500/
     * 0.187465156/0.196385625 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==729 && N==4304 && K==1152 && ta==0 && tb==1) {
      if(forced>=0 && forced!=2)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid_half_kernel<__hip_bfloat16,729,4304,1152,128,96,4,3,32,0,2,0,1,8,0>),dim3(45,6),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=2 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_1576_768_3072_0_1_0: screened P/new/HB 0.178227812/
     * 0.170659375/0.169285313 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==1576 && N==768 && K==3072 && ta==0 && tb==1) {
      if(forced>=0 && forced!=6)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_MIDTILE",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid05_vit_kernel<__hip_bfloat16,1576,768,3072,128,256,4,4,32,1,1055>),dim3(3,13),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=6 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    }
  }
  return false;
}

static bool launchMeasuredLagR2bMidF32(hipStream_t st,const float*A,const float*B,const float*C,float*Y,
                                    int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if(!((M==512 && N==1536 && K==384 && ta==0 && tb==1) || (M==512 && N==384 && K==1536 && ta==0 && tb==1)))return false;
  if(!gemmLagR2bMidAllowed(A,B,C,Y,alpha,beta))return false;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID_CFG",-1);
    /* f32_512_1536_384_0_1_0: screened P/new/HB 0.077425820/
     * 0.060021595/0.072939230 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==512 && N==1536 && K==384 && ta==0 && tb==1) {
      if(forced>=0 && forced!=3)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_F32_STAGGER",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid_f32_kernel<512,0,0,1,128,64,16,8,8,128,0,0>),dim3(24,4),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=3 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f32_512_384_1536_0_1_0: screened P/new/HB 0.089460525/
     * 0.079412935/0.080323885 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==512 && N==384 && K==1536 && ta==0 && tb==1) {
      if(forced>=0 && forced!=4)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_F32_STAGGER",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_lagr2b_mid02a_f32_kernel<512,384,1536,1,64,96,16,4,12,128,0,0,2,1>),dim3(4,8),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_mid route=4 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
  return false;
}


/* Independent exact measured nn routes. */

// This helper is independent of the parent transpose implementation.
// FLAGS bit0 uses byte permutation; bit1 removes unneeded LDS end padding.
template<typename T,int F,int KC,bool L128,int PERM>
__device__ __forceinline__ void
lagr2bNnStore06(T* __restrict__ tile,const SkinnyReg<T,F,KC,L128>& r) {
  if constexpr(!PERM)skinnyStoreT<T,F,KC,L128>(tile,r);
  else {
    using P=SkinnyT<T,F,KC,L128>;
    const int lid=threadIdx.x%32;
#pragma unroll
    for(int pass=0;pass<P::R;++pass) {
      const int slot=pass*32+lid,cb=(slot%P::CG)*P::CW,kb=(slot/P::CG)*4;
#pragma unroll
      for(int j=0;j<P::CW;++j) {
        const unsigned w0=r[pass][0][j/2],w1=r[pass][1][j/2];
        const unsigned w2=r[pass][2][j/2],w3=r[pass][3][j/2];
        constexpr unsigned LO=0x05040100u,HI=0x07060302u;
        u32x2 v;
        v[0]=__builtin_amdgcn_perm(w1,w0,(j&1)?HI:LO);
        v[1]=__builtin_amdgcn_perm(w3,w2,(j&1)?HI:LO);
        *reinterpret_cast<u32x2*>(&tile[(cb+j)*P::STR+kb])=v;
      }
    }
  }
}

/* Independent exact NN transpose loader. The old skinny/direct templates
 * are untouched. Per-lane vector offsets are fixed before the loop; only
 * scalar K offsets advance. NS selects a two/three-slot register pipeline.
 * Each wave stages legal64-byte B row segments (CW4,8-byte lane loads); the K loop has
 * no WG barrier. Wave-private staging and final reduction reuse one LDS. */
template<typename T,int MM,int NN,int KK,int MT,int NTL,int KW,int KU,int NS,int CACHE,int FLAGS,int LOADPATH,int RESERVED,int VERSION,int NARROW>
__global__ void __launch_bounds__(KW*32)
gemm_wmma_nn_direct_r2_kernel(const T* __restrict__ A,const T* __restrict__ B,
                             const T* __restrict__ C,T* __restrict__ Y,
                             int M_arg,int N_arg,int K_arg,float alpha,float beta,int c0,int c1) {
  constexpr int KC=16*KU,BN=16*NTL,STEPS=KK/(KC*KW),MTILES=(MM+16*MT-1)/(16*MT);
  static_assert(NTL==2 && NS==2 && KU==1 && NARROW==1 && KK%(KC*KW)==0 && NN%BN==0,"legal64-byte row chunks");
  static_assert(LOADPATH>=1 && LOADPATH<=3 && CACHE==0 && RESERVED==0 && VERSION==6,"vector/order control");
  static_assert(HIPDNN_WMMA_FRAG_ELEMS==16 || sizeof(T)==0,"gfx11 wave32 fragments");
  using P=SkinnyT<T,NTL,KC,true>;
  using Frag=typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
  constexpr int LB=BN*P::STR;
  constexpr int STAGING=KW*LB*2,RED=KW>1?KW*MT*NTL*8*32*4:0;
  __shared__ __align__(16) char sm_raw[(STAGING>RED?STAGING:RED)+((FLAGS&2)?0:16)];
  const int tid=threadIdx.x,lid=tid%32,lane=lid%16;
  const int wid=__builtin_amdgcn_readfirstlane(tid/32);
  const int row0=(blockIdx.x%MTILES)*16*MT,col0=(blockIdx.x/MTILES)*BN;
  T* tB=reinterpret_cast<T*>(sm_raw)+wid*LB;
  const T* Ab=A+size_t(row0)*KK;
  const T* Bb=B+col0;
  const auto ar=__builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(Ab),0,MM*KK*2,0x31004000);
  const auto br=__builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(Bb),0,NN*KK*2,0x31004000);
  int av[MT][KU][2],bv[P::R][4];
#pragma unroll
  for(int i=0;i<MT;++i) {
    const int r=row0+i*16+lane;
    const int rc=r<MM?r:MM-1;
#pragma unroll
    for(int q=0;q<KU;++q)
#pragma unroll
      for(int v=0;v<2;++v)av[i][q][v]=((rc-row0)*KK+q*16+v*8)*2;
  }
#pragma unroll
  for(int pass=0;pass<P::R;++pass) {
    const int slot=pass*32+lid;
#pragma unroll
    for(int q=0;q<4;++q)bv[pass][q]=((slot/P::CG*4+q)*NN+(slot%P::CG)*P::CW)*2;
  }
  u32x4 fa[NS][MT][KU][2];
  SkinnyReg<T,NTL,KC,true> rb[NS];
  float8 acc[MT][NTL];
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)acc[i][j]=float8{};
  int ka=wid*KC*2,kb=ka*NN;

  // LOADPATH: 1 raw-buffer B/A, 2 global-vector A/B, 3 global-vector B/A.
  // Offsets remain fixed, K advances in scalars, and the NS-1 pipeline,
  // transpose packing, staging allocation and reduction are unchanged.
  auto issue=[&](auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    auto loadA=[&]() __attribute__((always_inline)) {
#pragma unroll
      for(int i=0;i<MT;++i)
#pragma unroll
        for(int q=0;q<KU;++q)
#pragma unroll
          for(int v=0;v<2;++v) {
            if constexpr(LOADPATH>=2) {
              const unsigned offset=static_cast<unsigned>(av[i][q][v]+ka);
              fa[S][i][q][v]=*reinterpret_cast<const u32x4*>(
                  reinterpret_cast<const char*>(Ab)+offset);
            } else fa[S][i][q][v]=__builtin_amdgcn_raw_buffer_load_b128(ar,av[i][q][v],ka,0);
          }
    };
    auto loadB=[&]() __attribute__((always_inline)) {
#pragma unroll
      for(int pass=0;pass<P::R;++pass)
#pragma unroll
        for(int q=0;q<4;++q) {
          if constexpr(LOADPATH>=2) {
            const unsigned offset=static_cast<unsigned>(bv[pass][q]+kb);
            *reinterpret_cast<u32x2*>(rb[S][pass][q])=*reinterpret_cast<const u32x2*>(
                reinterpret_cast<const char*>(Bb)+offset);
          } else *reinterpret_cast<u32x2*>(rb[S][pass][q])=
              __builtin_amdgcn_raw_buffer_load_b64(br,bv[pass][q],kb,CACHE);
        }
    };
    if constexpr(LOADPATH==2) {loadA();loadB();}
    else {loadB();loadA();}
    ka+=KW*KC*2;kb+=KW*KC*NN*2;
  };
  auto compute=[&](auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    lagr2bNnStore06<T,NTL,KC,true,(FLAGS&1)>(tB,rb[S]);
#pragma unroll
    for(int q=0;q<KU;++q)
#pragma unroll
      for(int j=0;j<NTL;++j) {
        Frag b;
        const u32x4* bp=reinterpret_cast<const u32x4*>(&tB[(j*16+lane)*P::STR+q*16]);
        reinterpret_cast<u32x4*>(&b)[0]=bp[0];reinterpret_cast<u32x4*>(&b)[1]=bp[1];
#pragma unroll
        for(int i=0;i<MT;++i) {
          Frag a;__builtin_memcpy(&a,fa[S][i][q],sizeof(Frag));
          if constexpr(std::is_same<T,_Float16>::value)
            acc[i][j]=HIPDNN_WMMA_F32_16X16X16_F16(a,b,acc[i][j]);
          else acc[i][j]=HIPDNN_WMMA_F32_16X16X16_BF16(a,b,acc[i][j]);
        }
      }
    GEMM_SCHED_FENCE();
  };
  using S0=std::integral_constant<int,0>;
  using S1=std::integral_constant<int,1>;
  using S2=std::integral_constant<int,2>;

  // Prefill NS-1 slots. Issue the free slot before consuming the oldest.
  // All main-loop loads are unconditional; the remaining chunks are peeled.
  issue(S0{});if constexpr(NS==3)issue(S1{});
  GEMM_SCHED_FENCE();
  constexpr int NISS=STEPS-(NS-1),NROUND=NISS/NS,LEFT=NISS%NS;
#pragma unroll 1
  for(int g=0;g<NROUND;++g) {
    if constexpr(NS==2) {
      issue(S1{});GEMM_SCHED_FENCE();compute(S0{});
      issue(S0{});GEMM_SCHED_FENCE();compute(S1{});
    } else {
      issue(S2{});GEMM_SCHED_FENCE();compute(S0{});
      issue(S0{});GEMM_SCHED_FENCE();compute(S1{});
      issue(S1{});GEMM_SCHED_FENCE();compute(S2{});
    }
  }
  if constexpr(NS==2) {
    if constexpr(LEFT==1) {
      issue(S1{});GEMM_SCHED_FENCE();compute(S0{});compute(S1{});
    } else compute(S0{});
  } else {
    if constexpr(LEFT>=1) {issue(S2{});GEMM_SCHED_FENCE();}
    compute(S0{});
    if constexpr(LEFT>=2) {issue(S0{});GEMM_SCHED_FENCE();}
    compute(S1{});
    if constexpr(LEFT>=1)compute(S2{});
    if constexpr(LEFT>=2)compute(S0{});
  }
  if constexpr(KW>1)__syncthreads(); // all waves finish staging before LDS reuse
  auto red=reinterpret_cast<float(*)[MT][NTL][8][32]>(sm_raw);
  if constexpr(KW>1) {
#pragma unroll
    for(int i=0;i<MT;++i)
#pragma unroll
      for(int j=0;j<NTL;++j)
#pragma unroll
        for(int e=0;e<8;++e)red[wid][i][j][e][lid]=acc[i][j][e];
    __syncthreads();
  }
#pragma unroll
  for(int i=0;i<MT;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        const int r=row0+i*16+hipdnn_wmma_acc_row(lid/16,e);
        if(r<MM && wid==0) {
          float v=acc[i][j][e];
          if constexpr(KW>1) {
            v=0.f;
#pragma unroll
            for(int z=0;z<KW;++z)v+=red[z][i][j][e][lid];
          }
          Y[size_t(r)*NN+col0+j*16+lane]=GemmElem<T>::fromf(v);
        }
      }
}


#if defined(__HIP_DEVICE_COMPILE__) && (HIPDNN_WMMA_FRAG_ELEMS == 16)
#define LAGR2B_SHARED03_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAGR2B_SHARED03_BARRIER() __syncthreads()
#endif

/* Independent exact NN loop: one transposed B tile shared by all M waves,
 * direct A fragments, scalar advancing K offsets, two B register slots and
 * two alternating LDS buffers. Each wave owns distinct output rows.
 * Twelve integer template parameters distinguish this wrapper from every
 * preceding direct, native and natural overload. */
template<typename T,int MM,int NN,int KK,int BM,int MW,int NTL,int NWM,int KC,int PF,int PACK,int CACHE,int PAD>
__global__ void __launch_bounds__(NWM*32)
gemm_wmma_nn_direct_r2_kernel(const T* __restrict__ A,const T* __restrict__ B,
                             const T* __restrict__ C,T* __restrict__ Y,
                             int M,int N,int K,float alpha,float beta,int c0,int c1) {
  constexpr int THR=NWM*32,BN=16*NTL,KS=KC/16,STR=KC+PAD;
  constexpr int CGN=BN/8,NSL=(KC/4)*CGN,SPT=(NSL+THR-1)/THR;
  constexpr int MTILES=(MM+BM-1)/BM,CHUNKS=KK/KC;
  static_assert(BM==16*MW*NWM && NN%BN==0 && KK%KC==0 && CHUNKS%2==0,"exact shared geometry");
  static_assert((PF==1 || PF==2) && (PACK==0 || PACK==1) && PAD%8==0,"shared pipeline");
  static_assert(HIPDNN_WMMA_FRAG_ELEMS==16 || sizeof(T)==0,"wave32 full fragments");
  using Frag=typename std::conditional<std::is_same<T,_Float16>::value,half16,bf16xN>::type;
  __shared__ __align__(16) T tB[2][BN*STR];
  const int tid=threadIdx.x,lid=tid%32,lane=lid%16;
  const int wv=__builtin_amdgcn_readfirstlane(tid/32);
  const int row0=(blockIdx.x%MTILES)*BM,col0=(blockIdx.x/MTILES)*BN;
  const int wrow0=row0+wv*MW*16;
  const bool wact=wrow0<MM;
  const auto ar=__builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(A+size_t(row0)*KK),0,MM*KK*2,0x31004000);
  const auto br=__builtin_amdgcn_make_buffer_rsrc(const_cast<T*>(B+col0),0,NN*KK*2,0x31004000);
  int av[MW][KS][2],bv[SPT][4],sq[SPT];
#pragma unroll
  for(int i=0;i<MW;++i) {
    const int r=wrow0+i*16+lane,rc=r<MM?r:MM-1;
#pragma unroll
    for(int q=0;q<KS;++q)
#pragma unroll
      for(int v=0;v<2;++v)av[i][q][v]=((rc-row0)*KK+q*16+v*8)*2;
  }
#pragma unroll
  for(int p=0;p<SPT;++p) {
    const int q=p*THR+tid,cg=q%CGN,kq=(q/CGN)%(KC/4);
    sq[p]=cg*8*STR+kq*4;
#pragma unroll
    for(int r=0;r<4;++r)bv[p][r]=((kq*4+r)*NN+cg*8)*2;
  }
  u32x4 rb[PF][SPT][4],fa[2][MW][KS][2];
  float8 acc[MW][NTL];
#pragma unroll
  for(int i=0;i<MW;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)acc[i][j]=float8{};
  auto active=[&](int p) __attribute__((always_inline)) {
    return NSL%THR==0 || p*THR+tid<NSL;
  };
  auto loadB=[&](int chunk,auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    const int clipped=chunk<CHUNKS?chunk:CHUNKS-1;
    const int soff=clipped*KC*NN*2;
#pragma unroll
    for(int p=0;p<SPT;++p)if(active(p)) {
#pragma unroll
      for(int r=0;r<4;++r)rb[S][p][r]=__builtin_amdgcn_raw_buffer_load_b128(br,bv[p][r],soff,CACHE);
    }
  };
  auto loadA=[&](int chunk,auto slot) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value;
    const int clipped=chunk<CHUNKS?chunk:CHUNKS-1,soff=clipped*KC*2;
#pragma unroll
    for(int i=0;i<MW;++i)
#pragma unroll
      for(int q=0;q<KS;++q)
#pragma unroll
        for(int v=0;v<2;++v)fa[S][i][q][v]=__builtin_amdgcn_raw_buffer_load_b128(ar,av[i][q][v],soff,0);
  };
  auto storeB=[&](auto slot,auto buf) __attribute__((always_inline)) {
    constexpr int S=decltype(slot)::value,P=decltype(buf)::value;
#pragma unroll
    for(int p=0;p<SPT;++p)if(active(p)) {
#pragma unroll
      for(int j=0;j<8;++j) {
        const unsigned w0=rb[S][p][0][j/2],w1=rb[S][p][1][j/2];
        const unsigned w2=rb[S][p][2][j/2],w3=rb[S][p][3][j/2];
        u32x2 v;
        if constexpr(PACK) {
          const unsigned selector=(j&1)?0x07060302u:0x05040100u;
          v[0]=__builtin_amdgcn_perm(w1,w0,selector);
          v[1]=__builtin_amdgcn_perm(w3,w2,selector);
        } else if(j&1) {
          v[0]=(w0>>16)|(w1&0xffff0000u);v[1]=(w2>>16)|(w3&0xffff0000u);
        } else {
          v[0]=(w0&0xffffu)|(w1<<16);v[1]=(w2&0xffffu)|(w3<<16);
        }
        *reinterpret_cast<u32x2*>(&tB[P][sq[p]+j*STR])=v;
      }
    }
  };
  auto compute=[&](auto aslot,auto buf) __attribute__((always_inline)) {
    constexpr int S=decltype(aslot)::value,P=decltype(buf)::value;
#pragma unroll
    for(int q=0;q<KS;++q)
#pragma unroll
      for(int j=0;j<NTL;++j) {
        Frag b;
        const u32x4* bp=reinterpret_cast<const u32x4*>(&tB[P][(j*16+lane)*STR+q*16]);
        reinterpret_cast<u32x4*>(&b)[0]=bp[0];reinterpret_cast<u32x4*>(&b)[1]=bp[1];
#pragma unroll
        for(int i=0;i<MW;++i) {
          Frag a;__builtin_memcpy(&a,fa[S][i][q],sizeof(Frag));
          if constexpr(std::is_same<T,_Float16>::value)
            acc[i][j]=HIPDNN_WMMA_F32_16X16X16_F16(a,b,acc[i][j]);
          else acc[i][j]=HIPDNN_WMMA_F32_16X16X16_BF16(a,b,acc[i][j]);
        }
      }
  };
  using S0=std::integral_constant<int,0>;
  using S1=std::integral_constant<int,1>;
  auto step=[&](int t,auto par) __attribute__((always_inline)) {
    constexpr int P=decltype(par)::value;
    using PC=std::integral_constant<int,P>;
    using PN=std::integral_constant<int,1-P>;
    if constexpr(PF==2)loadB(t+2,PC{});else loadB(t+1,S0{});
    loadA(t+1,PN{});GEMM_SCHED_FENCE();
    if(wact)compute(PC{},PC{});
    GEMM_SCHED_FENCE();
    if constexpr(PF==2)storeB(PN{},PN{});else storeB(S0{},PN{});
    LAGR2B_SHARED03_BARRIER();
  };
  loadB(0,S0{});loadA(0,S0{});storeB(S0{},S0{});
  if constexpr(PF==2)loadB(1,S1{});
  GEMM_SCHED_FENCE();LAGR2B_SHARED03_BARRIER();
#pragma unroll 1
  for(int t=0;t<CHUNKS;t+=2) {step(t,S0{});step(t+1,S1{});}
#pragma unroll
  for(int i=0;i<MW;++i)
#pragma unroll
    for(int j=0;j<NTL;++j)
#pragma unroll
      for(int e=0;e<8;++e) {
        const int r=wrow0+i*16+hipdnn_wmma_acc_row(lid/16,e);
        if(r<MM)Y[size_t(r)*NN+col0+j*16+lane]=GemmElem<T>::fromf(acc[i][j][e]);
      }
}
#undef LAGR2B_SHARED03_BARRIER

static bool gemmLagR2bNnAllowed(const void*A,const void*B,const void*C,void*Y,
                            float alpha,float beta) {
  if(alpha!=1.f || beta!=0.f || C ||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN",1)!=0;
  static const bool pinned=
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_TCFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP",-1)>=0;
  return on&&!pinned&&gemmNnStreamGfx1151();
}


template<typename T>
static bool launchMeasuredLagR2bNnHalf(hipStream_t st,const T*A,const T*B,const T*C,T*Y,
                                     int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    bool exact=false;
    if constexpr(std::is_same<T,_Float16>::value)exact=(M==48 && N==4096 && K==14336 && ta==0 && tb==0) || (M==64 && N==4096 && K==14336 && ta==0 && tb==0) || (M==96 && N==4096 && K==14336 && ta==0 && tb==0);
    else if constexpr(std::is_same<T,__hip_bfloat16>::value)exact=(M==48 && N==4096 && K==14336 && ta==0 && tb==0) || (M==64 && N==4096 && K==14336 && ta==0 && tb==0) || (M==96 && N==4096 && K==14336 && ta==0 && tb==0);
    if(!exact)return false;
    if(!gemmLagR2bNnAllowed(A,B,C,Y,alpha,beta))return false;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN_CFG",-1);
    if constexpr(std::is_same<T,_Float16>::value) {
    /* f16_48_4096_14336_0_0_0: screened P/new/HB 0.574519722/
     * 0.558162083/0.538326875 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=3)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_SKINNY",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 48, 4096, 14336, 3, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=3 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_64_4096_14336_0_0_0: screened P/new/HB 0.582462837/
     * 0.566161560/0.549148085 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=4)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_SKINNY",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 64, 4096, 14336, 4, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=4 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* f16_96_4096_14336_0_0_0: screened P/new/HB 0.657267907/
     * 0.646664961/0.606576047 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=5)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_NNMID",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<_Float16, 96, 4096, 14336, 96, 1, 4, 6, 32, 1, 1, 0, 8>),dim3(64,1,1),dim3(192,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=5 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    } else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
    /* bf16_48_4096_14336_0_0_0: screened P/new/HB 0.573667123/
     * 0.557208836/0.538208836 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==48 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_SKINNY",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 48, 4096, 14336, 3, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=0 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_64_4096_14336_0_0_0: screened P/new/HB 0.581982837/
     * 0.566524894/0.549928865 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==64 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=1)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_SKINNY",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 64, 4096, 14336, 4, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>),dim3(128,1,1),dim3(32,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=1 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    /* bf16_96_4096_14336_0_0_0: screened P/new/HB 0.656886357/
     * 0.646738760/0.601842326 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==96 && N==4096 && K==14336 && ta==0 && tb==0) {
      if(forced>=0 && forced!=2)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_NNMID",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_nn_direct_r2_kernel<__hip_bfloat16, 96, 4096, 14336, 96, 1, 4, 6, 32, 1, 1, 0, 8>),dim3(64,1,1),dim3(192,1,1),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_nn route=2 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    }
  }
  return false;
}

static bool launchMeasuredLagR2bNnF32(hipStream_t st,const float*A,const float*B,const float*C,float*Y,
                                    int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if(!(false))return false;
  if(!gemmLagR2bNnAllowed(A,B,C,Y,alpha,beta))return false;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN_CFG",-1);
  return false;
}


/* Independent exact measured bigb routes. */

/* Independent fixed-dimension buffer entry for measured large-B controls. */
template<int BM,int BN,int WT_M,int WT_N,typename T,int BK,bool DB,int RP,
         int FIXM,int FIXN,int FIXK,int SW,int STAG>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM,BN,WT_M,WT_N),GEMM_WG_SZ(BM,BN,WT_M,WT_N))))
__attribute__((amdgpu_waves_per_eu(2,8)))
void gemm_wmma_lagr2b_bigb_buffer_kernel(const T* A,const T* B,const T* C,T* Y,
    int M,int N,int K,float alpha,float beta,int c0,int c1) {
  GemmWmmaBufImpl<BM,BN,WT_M,WT_N,T,BK,DB,RP>(
      A,B,nullptr,Y,FIXM,FIXN,FIXK,1.f,0.f,1,1,SW,FIXK,FIXK,STAG);
}

static bool gemmLagR2bBigbAllowed(const void*A,const void*B,const void*C,void*Y,
                            float alpha,float beta) {
  if(alpha!=1.f || beta!=0.f || C ||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB",1)!=0;
  static const bool pinned=
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_MID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LAGR2_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_TCFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_KN_RING_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_LONGK_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNMID_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNWIDE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNSTREAM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_NNLARGE_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_OVERLAP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_DECODENT_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SMALLM_CFG",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_U",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_BYTES",-1)>=0 ||
    gemmEnvInt("HIPDNN_EP_GEMM_STAGGER_MAP",-1)>=0;
  return on&&!pinned&&gemmNnStreamGfx1151();
}


template<typename T>
static bool launchMeasuredLagR2bBigbHalf(hipStream_t st,const T*A,const T*B,const T*C,T*Y,
                                     int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    bool exact=false;
    if constexpr(std::is_same<T,_Float16>::value)exact=false;
    else if constexpr(std::is_same<T,__hip_bfloat16>::value)exact=(M==2048 && N==128256 && K==8192 && ta==0 && tb==1);
    if(!exact)return false;
    if(!gemmLagR2bBigbAllowed(A,B,C,Y,alpha,beta))return false;
    static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG",-1);
    if constexpr(std::is_same<T,_Float16>::value) {
    } else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
    /* bf16_2048_128256_8192_0_1_0: screened P/new/HB 108.255143333/
     * 102.826956667/100.800523333 ms, five checked processes.
     * Actual product results are recorded in autotune/gemm/README.md. */
    if(M==2048 && N==128256 && K==8192 && ta==0 && tb==1) {
      if(forced>=0 && forced!=0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_ALIASK",1)==0)return false;
      if(gemmEnvInt("HIPDNN_EP_GEMM_BUFLOAD",1)==0)return false;
      (void)hipGetLastError();
      hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_wmma_lagr2b_llm_kernel<128,128,2,4,__hip_bfloat16,64,0,1,7,2048,128256,8192,-8,543>),dim3(1002,16),dim3(256),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1);
      CUSTOM_KERNELS_DEBUG_LOG("[gemm] lagr2b_bigb route=0 M=%d N=%d K=%d ta=%d tb=%d\n",M,N,K,ta,tb);
      return true;
    }
    }
  }
  return false;
}

static bool launchMeasuredLagR2bBigbF32(hipStream_t st,const float*A,const float*B,const float*C,float*Y,
                                    int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if(!(false))return false;
  if(!gemmLagR2bBigbAllowed(A,B,C,Y,alpha,beta))return false;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG",-1);
  return false;
}

/* =========================================================================
 * Public entry
 * ========================================================================= */


/* Exact vitb-up NT product additions. Separate instances preserve all
 * pre-existing device payloads. See autotune/gemm/README.md for evidence.
 */

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define VIT_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while(0)
#else
#define VIT_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR>
__device__ __forceinline__ void
GemmVitFastDenseImpl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = MODE == 2 ? 2 : 1;
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!XOR || BK == 32, "Measured compact XOR layout uses BK32");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmSwizzleBlock(gridDim.x, gridDim.y, swizzle_n, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = K / BK;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !HIPDNN_WMMA_GFX12
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if HIPDNN_WMMA_GFX12
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, MODE == 2 ? 1 - Q : 0>;
    if constexpr (IS)
      issueAll(SI{}, rsel);
#pragma unroll
    for (int ks = 0; ks < K_STEPS; ++ks)
      computeKs(PC{}, ks);
    schedStep(IS);
    if constexpr (ST) {
      if constexpr (!DB) {
        VIT_LDS_BARRIER();
        GEMM_SCHED_FENCE();
      }
      storeAll(SS{}, PN{});
      VIT_LDS_BARRIER();
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      VIT_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      VIT_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  VIT_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  /* niter is even (the launcher's K % (2 * BK) == 0), so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !HIPDNN_WMMA_GFX12
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T,bool BIAS,int BM,int BN,int WM,int WN,int BK,bool DB,int RP,int SCH,int SW,int FAMILY,int PAD,bool XOR,bool FIXED_MN,bool FIXED_K>
__global__ void __launch_bounds__((BM/(WM*16))*(BN/(WN*16))*32)
gemm_vitup_wmma_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
  const int gm=FIXED_MN?1576:M, gn=FIXED_MN?3072:N, gk=FIXED_K?768:K;
  if constexpr(FAMILY==8)
    GemmVitFastDenseImpl<BM,BN,WM,WN,T,BK,DB,RP+(BIAS?8:0),PAD,XOR>(A,B,C,Y,gm,gn,gk,1.f,BIAS?1.f:0.f,1,BIAS?gn:1,SW,gk,gk,0);
}

static bool vitupEligible(const void*A,const void*B,const void*C,void*Y,int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1,int cfg) {
  if(M!=1576||N!=3072||K!=768||ta||!tb||alpha!=1.f||
     ((reinterpret_cast<uintptr_t>(A)|reinterpret_cast<uintptr_t>(B)|reinterpret_cast<uintptr_t>(Y))&15))return false;
  if(cfg<2 ? beta!=0.f : (beta!=1.f||!C||c0!=1||c1!=N||(reinterpret_cast<uintptr_t>(C)&15)))return false;
  static const bool on=gemmEnvInt("HIPDNN_EP_GEMM_VITUP",1)!=0;
  static const int forced=gemmEnvInt("HIPDNN_EP_GEMM_VITUP_CFG",-1);
  static const bool pinned=gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG",-1)>=0||
      gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG",-1)>=0||gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG",-1)>=0||
      gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG",-1)>=0||gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG",-1)>=0||
      gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE",0)!=0||gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE",0)!=0;
  return on&&!pinned&&(forced<0||forced==cfg)&&gemmNnStreamGfx1151();
}
template <typename T>
static bool launchMeasuredVitup(hipStream_t st,const T*A,const T*B,const T*C,T*Y,int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    const bool bias=beta!=0.f;
    const int cfg=(std::is_same<T,_Float16>::value?0:1)+(bias?2:0);
    if(!vitupEligible(A,B,C,Y,M,N,K,ta,tb,alpha,beta,c0,c1,cfg))return false;
    (void)hipGetLastError();
    if(!bias) {
      if constexpr(std::is_same<T,_Float16>::value) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vitup_wmma_kernel<T,false,80,128,5,2,32,true,2,0,8,8,0,true,true,true>),dim3((N+128-1)/128,(M+80-1)/80),dim3(128),0,st,A,B,C,Y,M,N,K);
      }
      else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vitup_wmma_kernel<T,false,80,128,5,2,32,true,2,0,8,8,0,true,true,true>),dim3((N+128-1)/128,(M+80-1)/80),dim3(128),0,st,A,B,C,Y,M,N,K);
      }
    } else {
      if constexpr(std::is_same<T,_Float16>::value) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vitup_wmma_kernel<T,true,80,128,5,2,32,true,2,0,8,8,0,true,true,true>),dim3((N+128-1)/128,(M+80-1)/80),dim3(128),0,st,A,B,C,Y,M,N,K);
      }
      else if constexpr(std::is_same<T,__hip_bfloat16>::value) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vitup_wmma_kernel<T,true,80,128,5,2,32,true,2,0,8,8,0,true,true,true>),dim3((N+128-1)/128,(M+80-1)/80),dim3(128),0,st,A,B,C,Y,M,N,K);
      }
    }
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] vitup M=%d N=%d K=%d -> vitup-cfg[%d] bias=%d\n",M,N,K,cfg,int(bias));
    return true;
  }
  return false;
}
template <bool TB, int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ void __launch_bounds__(THREADS)
    gemm_vit_f32_u59_kernel(const float *__restrict__ A,
                         const float *__restrict__ B,
                         const float *__restrict__ C, float *__restrict__ Y,
                         int M, int N, int K, float alpha, float beta,
                         int cDim0, int cDim1, int yvec, int stag,
                         int swz) {
  constexpr int TX = BN / TN;
  constexpr int TY = BM / TM;
  static_assert(TX * TY == THREADS, "thread tile");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "4x4 output blocks");
  static_assert(BK % 8 == 0 && BM >= 16 && (!TB || BN >= 16),
                "loader lane pairs cover 16 rows x 2 k-pieces");
  constexpr int GM = TM / 4;
  constexpr int GN = TN / 4;
  constexpr int SM = BM / GM;
  constexpr int SN = BN / GN;
  constexpr int LDA = BM + 4;
  constexpr int LDB = BN + 4;
  constexpr int KQ = BK / 4; /* 4-float k pieces per row */
  constexpr int A_V4 = BM * KQ / THREADS;
  static_assert(A_V4 >= 1 && A_V4 * THREADS == BM * KQ, "A load split");
  constexpr int BQ = TB ? KQ : BN / 4;
  constexpr int B_ROWS = TB ? BN : BK;
  constexpr int B_V4 = B_ROWS * BQ / THREADS;
  static_assert(B_V4 >= 1 && B_V4 * THREADS == B_ROWS * BQ, "B load split");

  __shared__ __align__(16) float smA[2][BK][LDA];
  __shared__ __align__(16) float smB[2][BK][LDB];

  const int tid = threadIdx.x;
  const int tx = tid % TX;
  const int ty = tid / TX;
  int by = blockIdx.y, bx = blockIdx.x;
  if (swz != 0)
    gemmSwizzleBlock(gridDim.x, gridDim.y, swz, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  /* k-contiguous operand piece idx -> (row, k): lane pairs take two adjacent
   * k pieces of one row, 16 rows per wave. */
  auto pairPiece = [](int idx, int rows, int &r, int &k)
      __attribute__((always_inline)) {
    r = (idx >> 1) % rows;
    k = ((idx & 1) | ((idx / (2 * rows)) << 1)) * 4;
  };

  const float *a_src[A_V4];
  int a_r[A_V4], a_k[A_V4];
#pragma unroll
  for (int v = 0; v < A_V4; ++v) {
    pairPiece(tid + v * THREADS, BM, a_r[v], a_k[v]);
    const int gr = min(row0 + a_r[v], M - 1);
    a_src[v] = A + static_cast<size_t>(gr) * K + a_k[v];
  }
  /* NT: b_r is the tile's n row and b_c its k offset. NN: b_r is the k row
   * and b_c the n offset (16-byte pieces along N, stored as they are). */
  const float *b_src[B_V4];
  int b_r[B_V4], b_c[B_V4];
#pragma unroll
  for (int v = 0; v < B_V4; ++v) {
    const int idx = tid + v * THREADS;
    if constexpr (TB) {
      pairPiece(idx, BN, b_r[v], b_c[v]);
      const int gn = min(col0 + b_r[v], N - 1);
      b_src[v] = B + static_cast<size_t>(gn) * K + b_c[v];
    } else {
      b_r[v] = idx / BQ;
      b_c[v] = (idx % BQ) * 4;
      const int gn = min(col0 + b_c[v], N - 4);
      b_src[v] = B + static_cast<size_t>(b_r[v]) * N + gn;
    }
  }

  gemmF4 ra[A_V4], rb[B_V4];
  /* TAIL: the tile may run past K. K % 4 == 0, so a 4-float k piece is either
   * all inside K or all outside. */
  auto load = [&](int k0, auto tail) __attribute__((always_inline)) {
    constexpr bool TAIL = decltype(tail)::value;
    const gemmF4 zero = gemmF4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int v = 0; v < A_V4; ++v)
      ra[v] = (!TAIL || k0 + a_k[v] < K)
                  ? *reinterpret_cast<const gemmF4 *>(a_src[v] + k0)
                  : zero;
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB)
        rb[v] = (!TAIL || k0 + b_c[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(b_src[v] + k0)
                    : zero;
      else
        rb[v] = (!TAIL || k0 + b_r[v] < K)
                    ? *reinterpret_cast<const gemmF4 *>(
                          b_src[v] + static_cast<size_t>(k0) * N)
                    : zero;
    }
  };

  auto store = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
    for (int v = 0; v < A_V4; ++v) {
      smA[buf][a_k[v] + 0][a_r[v]] = ra[v].x;
      smA[buf][a_k[v] + 1][a_r[v]] = ra[v].y;
      smA[buf][a_k[v] + 2][a_r[v]] = ra[v].z;
      smA[buf][a_k[v] + 3][a_r[v]] = ra[v].w;
    }
#pragma unroll
    for (int v = 0; v < B_V4; ++v) {
      if constexpr (TB) {
        smB[buf][b_c[v] + 0][b_r[v]] = rb[v].x;
        smB[buf][b_c[v] + 1][b_r[v]] = rb[v].y;
        smB[buf][b_c[v] + 2][b_r[v]] = rb[v].z;
        smB[buf][b_c[v] + 3][b_r[v]] = rb[v].w;
      } else {
        *reinterpret_cast<gemmF4 *>(&smB[buf][b_r[v]][b_c[v]]) = rb[v];
      }
    }
  };

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  auto compute = [&](int buf) __attribute__((always_inline)) {
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int g = 0; g < GM; ++g) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smA[buf][kk][g * SM + ty * 4]);
        a[g * 4 + 0] = v.x;
        a[g * 4 + 1] = v.y;
        a[g * 4 + 2] = v.z;
        a[g * 4 + 3] = v.w;
      }
#pragma unroll
      for (int h = 0; h < GN; ++h) {
        const gemmF4 v = *reinterpret_cast<const gemmF4 *>(
            &smB[buf][kk][h * SN + tx * 4]);
        b[h * 4 + 0] = v.x;
        b[h * 4 + 1] = v.y;
        b[h * 4 + 2] = v.z;
        b[h * 4 + 3] = v.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
      GEMM_SCHED_FENCE();
    }
    /* Keep the FMAs here: without this the compiler sinks the last tile's
     * FMAs into the epilogue branches and hoists all its LDS reads (256 VGPRs
     * and scratch). */
    __builtin_amdgcn_sched_barrier(0);
  };

  using Full = std::integral_constant<bool, false>;
  using Tail = std::integral_constant<bool, true>;
  const int nfull = K / BK;
  const int ntiles = nfull + ((K % BK) != 0 ? 1 : 0);
  /* StaggerU over the whole tiles (stag = tiles per step, 0 = off): dispatch
   * neighbours start their K loop at different offsets and wrap, so a K whose
   * row stride is a large power of two does not send every workgroup's rows
   * to the same memory channels at once. The tail tile always comes last. */
  const int shift =
      (stag > 0 && nfull > 1)
          ? ((static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x) & 31) *
             stag) %
                nfull
          : 0;
  auto kofs = [&](int i) __attribute__((always_inline)) {
    int kt = i + shift;
    if (kt >= nfull)
      kt -= nfull;
    return kt * BK;
  };
  if (nfull > 0)
    load(kofs(0), Full{});
  else
    load(0, Tail{});
  store(0);
  VIT_LDS_BARRIER();
  /* Whole tiles first so the loop body is one straight-line load / compute /
   * store / barrier; the zero-filled tail tile and the last tile follow it. */
  int t = 0;
  for (; t + 1 < nfull; ++t) {
    load(kofs(t + 1), Full{});
    compute(t & 1);
    store((t + 1) & 1);
    VIT_LDS_BARRIER();
  }
  if (t + 1 < ntiles) {
    load((t + 1) * BK, Tail{});
    compute(t & 1);
    store((t + 1) & 1);
    VIT_LDS_BARRIER();
    ++t;
  }
  compute(t & 1);

  gemmF4 bc[GN];
#pragma unroll
  for(int h=0;h<GN;++h) {
    const int c=col0+h*SN+tx*4;
    bc[h]=*reinterpret_cast<const gemmF4 *>(C+c);
  }
#pragma unroll
  for(int g=0;g<GM;++g) {
#pragma unroll
    for(int i=0;i<4;++i) {
      const int r=row0+g*SM+ty*4+i;
      if(r>=M)continue;
      float *dst=Y+static_cast<size_t>(r)*N+col0+tx*4;
#pragma unroll
      for(int h=0;h<GN;++h) {
        const int ai=g*4+i;
        *reinterpret_cast<gemmF4 *>(dst+h*SN)=gemmF4{
          acc[ai][h*4+0]+bc[h].x,acc[ai][h*4+1]+bc[h].y,
          acc[ai][h*4+2]+bc[h].z,acc[ai][h*4+3]+bc[h].w};
      }
    }
  }
}

template <typename T>
static bool launchMeasuredVitupF32(hipStream_t st,const T*A,const T*B,const T*C,T*Y,int M,int N,int K,int ta,int tb,float alpha,float beta,int c0,int c1) {
  if constexpr(HIPDNN_WMMA_FRAG_ELEMS==16) {
    if(!vitupEligible(A,B,C,Y,M,N,K,ta,tb,alpha,beta,c0,c1,4))return false;
    (void)hipGetLastError();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_vit_f32_u59_kernel<std::is_same<T,float>::value,128,64,16,8,8,128>),dim3((N+64-1)/64,(M+128-1)/128),dim3(128),0,st,A,B,C,Y,M,N,K,alpha,beta,c0,c1,1,0,0);
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] vitup-f32 M=%d N=%d K=%d -> vitup-cfg[4] bias=1\n",M,N,K);return true;
  }
  return false;
}
#undef VIT_LDS_BARRIER

/* Stage4b: independent measured SigLIP/MiniLM NT instances. Existing
 * kernels and kWmma stay unchanged; exact runtime gfx1151 eligibility is
 * enforced below. The compile-time geometry also pins raster and K order.
 */

__device__ __forceinline__ void gemmMid4bSwizzleC0(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

#undef LAG4B_LDS_BARRIER

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAG4B_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); if constexpr (THR > 32) __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAG4B_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL, int SRDS>
__device__ __forceinline__ void
GemmMid4bC0Impl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR == 64 || THR == 128, "Two or four output waves");
  static_assert(SRDS == 1 || SRDS == 2, "B-only or A+B chunk resources");
  static_assert(FIXK == 4304 && FIXM == 729 && FIXN == 1152 && FIXN % BN == 0, "Exact Down");
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC0((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (K / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  const int voffAShared = lr * lda * ES + lk * ES;
  const int voffBShared = lr * ldb * ES + lk * ES;
  int voffA[A_NCH];
  if constexpr (SRDS == 1) {
#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      const int r = row0 + lr + c * RPI;
      voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
    }
  }
  const char *Ab = reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb = reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab), 0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  Rsrc rac0[A_NCH], rac1[A_NCH], rbc0[B_NCH], rbc1[B_NCH];
  if constexpr (SRDS == 2) {
#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      const int first = row0 + c * RPI;
      const int bytes = (first < M ? M - first : 0) * lda * ES;
      char *base = const_cast<char *>(Ab + static_cast<size_t>(c * RPI) * lda * ES);
      rac0[c] = __builtin_amdgcn_make_buffer_rsrc(base, 0, bytes, 0x31004000);
      rac1[c] = __builtin_amdgcn_make_buffer_rsrc(base - kspan, 0, bytes + kspan, 0x31004000);
    }
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    char *base = const_cast<char *>(Bb + static_cast<size_t>(c * RPI) * ldb * ES);
    rbc0[c] = __builtin_amdgcn_make_buffer_rsrc(base, 0, 0x7fffffff, 0x31004000);
    rbc1[c] = __builtin_amdgcn_make_buffer_rsrc(base - kspan, 0, 0x7fffffff, 0x31004000);
  }
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    if (i < A_NCH) {
      const int vo = SRDS == 2 ? voffAShared + i * RPI * lda * ES : voffA[i];
      pf[S][i] = SRDS == 2 && row0 + lr + i * RPI >= M ? u32x4{} :
          *reinterpret_cast<const u32x4 *>((R == 0 ? Ab : Ab - kspan) + vo + kb);
    } else {
      const int vo = voffBShared + (i - A_NCH) * RPI * ldb * ES;
      pf[S][i] = *reinterpret_cast<const u32x4 *>((R == 0 ? Bb : Bb - kspan) + vo + kb);
    }
#else
    if (i < A_NCH) {
      if constexpr (SRDS == 2)
        pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? rac0[i] : rac1[i], voffAShared, kb, 0);
      else
        pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1, voffA[i], kb, 0);
    } else {
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? rbc0[i - A_NCH] : rbc1[i - A_NCH], voffBShared, kb, 0);
    }
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR && BK >= 32 ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAG4B_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAG4B_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAG4B_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        if (i < A_NCH) {
          const int vo = SRDS == 2 ? voffAShared + i * RPI * lda * ES : voffA[i];
          pf[0][i] = SRDS == 2 && row0 + lr + i * RPI >= M ? u32x4{} :
              *reinterpret_cast<const u32x4 *>(Ab + vo + START * ES);
        } else {
          const int vo = voffBShared + (i - A_NCH) * RPI * ldb * ES;
          pf[0][i] = *reinterpret_cast<const u32x4 *>(Bb + vo + START * ES);
        }
#else
        if (i < A_NCH) {
          if constexpr (SRDS == 2)
            pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rac0[i], voffAShared, START * ES, 0);
          else
            pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, voffA[i], START * ES, 0);
        } else {
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rbc0[i - A_NCH], voffBShared, START * ES, 0);
        }
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAG4B_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAG4B_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T> __global__ void __launch_bounds__(128)
gemm_mid4b_wmma_c0_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
 GemmMid4bC0Impl<96,96,3,3,T,32,true,2,0,true,4304,729,1152,8,0,2,0,1>(A,B,C,Y,729,1152,4304,1.f,0.f,1,1,8,4304,4304,0);
}

__device__ __forceinline__ void gemmMid4bSwizzleC1(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

#undef LAG4B_LDS_BARRIER

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAG4B_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); if constexpr (THR > 32) __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAG4B_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL, int SRDS>
__device__ __forceinline__ void
GemmMid4bC1Impl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR == 64 || THR == 128, "Two or four output waves");
  static_assert(SRDS == 1 || SRDS == 2, "B-only or A+B chunk resources");
  static_assert(FIXK == 4304 && FIXM == 729 && FIXN == 1152 && FIXN % BN == 0, "Exact Down");
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC1((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (K / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  const int voffAShared = lr * lda * ES + lk * ES;
  const int voffBShared = lr * ldb * ES + lk * ES;
  int voffA[A_NCH];
  if constexpr (SRDS == 1) {
#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      const int r = row0 + lr + c * RPI;
      voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
    }
  }
  const char *Ab = reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb = reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab), 0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  Rsrc rac0[A_NCH], rac1[A_NCH], rbc0[B_NCH], rbc1[B_NCH];
  if constexpr (SRDS == 2) {
#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      const int first = row0 + c * RPI;
      const int bytes = (first < M ? M - first : 0) * lda * ES;
      char *base = const_cast<char *>(Ab + static_cast<size_t>(c * RPI) * lda * ES);
      rac0[c] = __builtin_amdgcn_make_buffer_rsrc(base, 0, bytes, 0x31004000);
      rac1[c] = __builtin_amdgcn_make_buffer_rsrc(base - kspan, 0, bytes + kspan, 0x31004000);
    }
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    char *base = const_cast<char *>(Bb + static_cast<size_t>(c * RPI) * ldb * ES);
    rbc0[c] = __builtin_amdgcn_make_buffer_rsrc(base, 0, 0x7fffffff, 0x31004000);
    rbc1[c] = __builtin_amdgcn_make_buffer_rsrc(base - kspan, 0, 0x7fffffff, 0x31004000);
  }
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    if (i < A_NCH) {
      const int vo = SRDS == 2 ? voffAShared + i * RPI * lda * ES : voffA[i];
      pf[S][i] = SRDS == 2 && row0 + lr + i * RPI >= M ? u32x4{} :
          *reinterpret_cast<const u32x4 *>((R == 0 ? Ab : Ab - kspan) + vo + kb);
    } else {
      const int vo = voffBShared + (i - A_NCH) * RPI * ldb * ES;
      pf[S][i] = *reinterpret_cast<const u32x4 *>((R == 0 ? Bb : Bb - kspan) + vo + kb);
    }
#else
    if (i < A_NCH) {
      if constexpr (SRDS == 2)
        pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? rac0[i] : rac1[i], voffAShared, kb, 0);
      else
        pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1, voffA[i], kb, 0);
    } else {
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? rbc0[i - A_NCH] : rbc1[i - A_NCH], voffBShared, kb, 0);
    }
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR && BK >= 32 ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAG4B_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAG4B_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAG4B_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        if (i < A_NCH) {
          const int vo = SRDS == 2 ? voffAShared + i * RPI * lda * ES : voffA[i];
          pf[0][i] = SRDS == 2 && row0 + lr + i * RPI >= M ? u32x4{} :
              *reinterpret_cast<const u32x4 *>(Ab + vo + START * ES);
        } else {
          const int vo = voffBShared + (i - A_NCH) * RPI * ldb * ES;
          pf[0][i] = *reinterpret_cast<const u32x4 *>(Bb + vo + START * ES);
        }
#else
        if (i < A_NCH) {
          if constexpr (SRDS == 2)
            pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rac0[i], voffAShared, START * ES, 0);
          else
            pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, voffA[i], START * ES, 0);
        } else {
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rbc0[i - A_NCH], voffBShared, START * ES, 0);
        }
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAG4B_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAG4B_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T> __global__ void __launch_bounds__(128)
gemm_mid4b_wmma_c1_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
 GemmMid4bC1Impl<96,96,3,3,T,32,true,2,0,true,4304,729,1152,8,0,2,0,1>(A,B,C,Y,729,1152,4304,1.f,0.f,1,1,8,4304,4304,0);
}

__device__ __forceinline__ void gemmMid4bSwizzleC2(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

#undef LAG4B_LDS_BARRIER

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAG4B_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAG4B_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL>
__device__ __forceinline__ void
GemmMid4bC2Impl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][BN][K_STR];

  const int tid = threadIdx.x;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC2((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (K / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda);
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb);
  const int kspan = K * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR && BK >= 32 ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAG4B_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAG4B_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAG4B_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        const char *base = i < A_NCH ? Ab : Bb;
        pf[0][i] = *reinterpret_cast<const u32x4 *>(base + vo + START * ES);
#else
        if (i < A_NCH)
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, vo, START * ES, 0);
        else
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, vo, START * ES, 0);
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAG4B_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAG4B_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T> __global__ void __launch_bounds__(128) gemm_mid4b_wmma_c2_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
  GemmMid4bC2Impl<128,96,4,3,T,32,true,2,0,true,1152,729,4304,8,0,1,0>(A,B,C,Y,729,4304,1152,1.f,0.f,1,1,8,1152,1152,0);
}

__device__ __forceinline__ void gemmMid4bSwizzleC3(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

#undef LAG4B_LDS_BARRIER

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAG4B_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAG4B_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL, int NWK>
__device__ __forceinline__ void
GemmMid4bC3Impl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NWK * NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][NWK][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][NWK][BN][K_STR];

  static_assert(NWK == 2 && FIXK % (2 * NWK * BK) == 0, "Two complete equal K groups");
  static_assert(NBUF * NWK * BN * K_STR * 2 >= BM * BN * 4, "Reuse B LDS for one FP32 partial tile");
  const int group = __builtin_amdgcn_readfirstlane(threadIdx.x / THR);
  const int tid = threadIdx.x % THR;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC3((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (FIXK / NWK / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda + group * (FIXK / NWK));
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb + group * (FIXK / NWK));
  const int kspan = (FIXK / NWK) * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR && BK >= 32 ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][group][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][group][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][group][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][group][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][group][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][group][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAG4B_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAG4B_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAG4B_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / NWK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK / NWK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        const char *base = i < A_NCH ? Ab : Bb;
        pf[0][i] = *reinterpret_cast<const u32x4 *>(base + vo + START * ES);
#else
        if (i < A_NCH)
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, vo, START * ES, 0);
        else
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, vo, START * ES, 0);
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAG4B_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAG4B_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Both equal K groups finish before B staging is reused. Group 1
  // writes one FP32 partial tile into that existing LDS allocation; group 0
  // reads and adds it, then performs the ordinary SDK output conversion.
  // There is no cross-workgroup reduction or additional device allocation.
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  float *partial = reinterpret_cast<float *>(smB);
  if (group == 1) {
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm)
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int rr = (wrow * WT_M + wm) * WMMA_TILE + hipdnn_wmma_acc_row(sub, e);
          const int cc = (wcol * WT_N + wn) * WMMA_TILE + lane;
          partial[rr * BN + cc] = acc[wm][wn][e];
        }
  }
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  if (group != 0)
    return;
#pragma unroll
  for (int wm = 0; wm < WT_M; ++wm)
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = (wrow * WT_M + wm) * WMMA_TILE + hipdnn_wmma_acc_row(sub, e);
        const int cc = (wcol * WT_N + wn) * WMMA_TILE + lane;
        acc[wm][wn][e] += partial[rr * BN + cc];
      }
  GEMM_SCHED_FENCE();

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T> __global__ void __launch_bounds__(256)
__attribute__((amdgpu_waves_per_eu(2, 8)))
gemm_mid4b_wmma_c3_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
  GemmMid4bC3Impl<64,96,2,3,T,32,true,2,0,true,1536,512,384,8,0,2,2,2>(A,B,C,Y,512,384,1536,1.f,0.f,1,1,8,1536,1536,0);
}

__device__ __forceinline__ void gemmMid4bSwizzleC4(int n_tiles, int m_tiles,
                                                  int swizzle_n, int &by,
                                                  int &bx) {
  const int block_id = blockIdx.y * n_tiles + blockIdx.x;
  if (swizzle_n < 0) {
    /* M-first raster (Tensile WGM>0): dispatch-order neighbours walk up to
     * -swizzle_n M-tiles of one N-tile before moving to the next N-tile. */
    const int wgm = (-swizzle_n < m_tiles) ? -swizzle_n : m_tiles;
    const int group = block_id / (wgm * n_tiles);
    const int rows = (m_tiles - group * wgm < wgm) ? m_tiles - group * wgm : wgm;
    const int rem = block_id - group * wgm * n_tiles;
    by = group * wgm + rem % rows;
    bx = rem / rows;
    return;
  }
  const int sw = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
  const int main_cols = (n_tiles / sw) * sw;
  const int main_blocks = main_cols * m_tiles;
  if (block_id < main_blocks) {
    const int super = block_id / (sw * m_tiles);
    const int rem = block_id % (sw * m_tiles);
    by = rem / sw;
    bx = super * sw + rem % sw;
  } else {
    const int tail_id = block_id - main_blocks;
    const int tail_cols = n_tiles - main_cols;
    by = tail_id / tail_cols;
    bx = main_cols + (tail_id % tail_cols);
  }
}

#undef LAG4B_LDS_BARRIER

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#define LAG4B_LDS_BARRIER() do { __builtin_amdgcn_s_waitcnt(0xfc07); __builtin_amdgcn_s_barrier(); } while(0)
#else
#define LAG4B_LDS_BARRIER() __syncthreads()
#endif
template <int BM, int BN, int WT_M, int WT_N, typename T, int BK, bool DB,
          int RP, int PAD, bool XOR, int FIXK, int FIXM, int FIXN,
          int SW, int CF, int SF, int ROLL, int NWK>
__device__ __forceinline__ void
GemmMid4bC4Impl(const T *__restrict__ A, const T *__restrict__ B,
                const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                int K, float alpha, float beta, int cDim0, int cDim1,
                int swizzle_n, int ldb, int lda, int stag) {
  constexpr int WM_L = BM / (WT_M * 16);
  constexpr int WN_L = BN / (WT_N * 16);
  constexpr int THR = WM_L * WN_L * 32;
  constexpr int K_STR = BK + PAD;
  constexpr int K_STEPS = BK / WMMA_TILE;
  constexpr int LPR = BK / 8;    /* lanes per row, 16 B each */
  constexpr int RPI = THR / LPR; /* rows one WG-wide load instruction covers */
  constexpr int A_NCH = (BM + RPI - 1) / RPI;
  constexpr int B_NCH = (BN + RPI - 1) / RPI;
  constexpr int NL = A_NCH + B_NCH;
  constexpr int MODE = RP & 3;
  constexpr int NSLOT = ROLL ? 1 : (MODE == 2 ? 2 : 1);
  constexpr int NBUF = DB ? 2 : 1;
  constexpr int D = ROLL || MODE == 2 ? 2 : 1; /* step t issues tile t + D */
  constexpr int KB = BK * static_cast<int>(sizeof(T));
  constexpr int ES = static_cast<int>(sizeof(T));
  static_assert(THR % LPR == 0 && PAD % 8 == 0, "chunks");
  static_assert(MODE >= 0 && MODE <= 2, "RP pipeline");
  static_assert(!ROLL || (DB && MODE == 2), "rolling DB/PGR2");
  static_assert(NWK * NBUF * (BM + BN) * K_STR * 2 <= 65536, "LDS 64KB");

  __shared__ __align__(16) T smA[NBUF][NWK][BM][K_STR];
  __shared__ __align__(16) T smB[NBUF][NWK][BN][K_STR];

  static_assert(NWK == 2 && FIXK % (2 * NWK * BK) == 0, "Two complete equal K groups");
  static_assert(NBUF * NWK * BN * K_STR * 2 >= BM * BN * 4, "Reuse B LDS for one FP32 partial tile");
  const int group = __builtin_amdgcn_readfirstlane(threadIdx.x / THR);
  const int tid = threadIdx.x % THR;
  const int wid = __builtin_amdgcn_readfirstlane(tid / 32);
  const int lid = tid % 32;
  const int lane = lid % 16;
  const int sub = lid / 16;
  const int wrow = wid / WN_L;
  const int wcol = wid % WN_L;

  int by, bx;
  gemmMid4bSwizzleC4((FIXN + BN - 1) / BN, (FIXM + BM - 1) / BM, SW, by, bx);
  const int row0 = by * BM;
  const int col0 = bx * BN;

  const int niter = (FIXK / NWK / BK) & ~1;
  int shift = 0;
  if (stag != 0) {
    const int map = (stag >> 20) & 3;
    const int sid_src =
        map == 0 ? by
                 : (map == 1 ? bx : by * static_cast<int>(gridDim.x) + bx);
    shift = (((sid_src & (stag & 0xff)) * ((stag >> 8) & 0xfff)) % niter) & ~1;
  }
  const int jw = niter - shift; /* first tile index past the wrap */

  /* Thread -> (row lr + c * RPI, k chunk lk); rows past M / N clamp. */
  const int lr = tid / LPR;
  const int lk = (tid % LPR) * 8;
  int voffA[A_NCH];
  int voffB[B_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int r = row0 + lr + c * RPI;
    voffA[c] = ((r < M ? r : M - 1) - row0) * lda * ES + lk * ES;
  }
#pragma unroll
  for (int c = 0; c < B_NCH; ++c) {
    const int r = col0 + lr + c * RPI;
    voffB[c] = ((r < N ? r : N - 1) - col0) * ldb * ES + lk * ES;
  }
  const char *Ab =
      reinterpret_cast<const char *>(A + static_cast<size_t>(row0) * lda + group * (FIXK / NWK));
  const char *Bb =
      reinterpret_cast<const char *>(B + static_cast<size_t>(col0) * ldb + group * (FIXK / NWK));
  const int kspan = (FIXK / NWK) * ES;
#if !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  using Rsrc = __amdgpu_buffer_rsrc_t;
  const Rsrc ra0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Ab),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc rb0 = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(Bb),
                                                     0, 0x7fffffff, 0x31004000);
  const Rsrc ra1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Ab - kspan), 0, 0x7fffffff, 0x31004000);
  const Rsrc rb1 = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<char *>(Bb - kspan), 0, 0x7fffffff, 0x31004000);
#endif

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  u32x4 pf[NSLOT][NL];
  int kb = shift * KB; /* byte offset of the next tile to issue */

  /* rsel picks the resource pair at compile time (the host pass cannot
   * mangle __amdgpu_buffer_rsrc_t in a lambda signature): 0 = before the
   * wrap, 1 = after it. */
  auto issueOne = [&](auto slot, int i, auto rsel)
      __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int R = decltype(rsel)::value;
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
    const char *base = i < A_NCH ? (R == 0 ? Ab : Ab - kspan)
                                 : (R == 0 ? Bb : Bb - kspan);
    const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
    pf[S][i] = *reinterpret_cast<const u32x4 *>(base + vo + kb);
#else
    if (i < A_NCH)
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(R == 0 ? ra0 : ra1,
                                                       voffA[i], kb, 0);
    else
      pf[S][i] = __builtin_amdgcn_raw_buffer_load_b128(
          R == 0 ? rb0 : rb1, voffB[i - A_NCH], kb, 0);
#endif
  };
  auto issueAll = [&](auto slot, auto rsel) __attribute__((always_inline)) {
#pragma unroll
    for (int i = 0; i < NL; ++i)
      issueOne(slot, i, rsel);
    kb += KB;
  };
  // XOR whole eight-half chunks: each b128 address remains 16B aligned.
  auto ldsK = [](int r, int k) __attribute__((always_inline)) {
    return XOR && BK >= 32 ? (k ^ (((r >> 1) & 3) << 3)) : k;
  };
  auto storeAll = [&](auto slot, auto buf) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      if (i < A_NCH) {
        if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
          *reinterpret_cast<u32x4 *>(&smA[P][group][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
      } else {
        if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
          *reinterpret_cast<u32x4 *>(&smB[P][group][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
      }
    }
  };
  auto storeOne = [&](auto slot, auto buf, int i) __attribute__((always_inline)) {
    constexpr int S = decltype(slot)::value;
    constexpr int P = decltype(buf)::value;
    if (i < A_NCH) {
      if ((i + 1) * RPI <= BM || lr + i * RPI < BM)
        *reinterpret_cast<u32x4 *>(&smA[P][group][lr + i * RPI][ldsK(lr + i * RPI, lk)]) = pf[S][i];
    } else {
      if ((i - A_NCH + 1) * RPI <= BN || lr + (i - A_NCH) * RPI < BN)
        *reinterpret_cast<u32x4 *>(&smB[P][group][lr + (i - A_NCH) * RPI][ldsK(lr + (i - A_NCH) * RPI, lk)]) = pf[S][i];
    }
  };
  auto computeKs = [&](auto buf, int ks) __attribute__((always_inline)) {
    constexpr int P = decltype(buf)::value;
    const int k_off = hipdnn_wmma_k_off(sub);
    using Frag = typename std::conditional<std::is_same<T, _Float16>::value, half16, bf16xN>::type;
    if constexpr (CF != 0) {
      Frag a_frag[WT_M];
#pragma unroll
      for (int wm = 0; wm < WT_M; ++wm) {
        const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
        u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag[wm]);
#pragma unroll
        for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
          dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][group][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
      }
      Frag b_frag[WT_N];
      if constexpr (CF == 2) {
#pragma unroll
        for (int wn = 0; wn < WT_N; ++wn) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
      }
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (CF == 1) {
          const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
          u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
          for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
            dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
        }
#pragma unroll
        for (int wm = 0; wm < WT_M; ++wm) {
          if constexpr (std::is_same<T, _Float16>::value)
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag[wm], b_frag[wn], acc[wm][wn]);
          else
            acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag[wm], b_frag[wn], acc[wm][wn]);
        }
      }
    } else {
    Frag b_frag[WT_N];
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn) {
      const int br = (wcol * WT_N + wn) * WMMA_TILE + lane;
      u32x4 *dst = reinterpret_cast<u32x4 *>(&b_frag[wn]);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smB[P][group][br][ldsK(br, ks * WMMA_TILE + k_off + i * 8)]);
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int ar = (wrow * WT_M + wm) * WMMA_TILE + lane;
      Frag a_frag{};
      u32x4 *dst = reinterpret_cast<u32x4 *>(&a_frag);
#pragma unroll
      for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 8; ++i)
        dst[i] = *reinterpret_cast<const u32x4 *>(&smA[P][group][ar][ldsK(ar, ks * WMMA_TILE + k_off + i * 8)]);
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn) {
        if constexpr (std::is_same<T, _Float16>::value)
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(a_frag, b_frag[wn], acc[wm][wn]);
        else
          acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_BF16(a_frag, b_frag[wn], acc[wm][wn]);
      }
    }
    }
    if constexpr ((SF & 2) != 0)
      GEMM_SCHED_FENCE();
  };
  /* RP bits 4 / 16: the sched_group_barrier order of one step (masks: 0x8
   * WMMA, 0x20 global load, 0x100 LDS read; group sizes must be literals). */
  auto schedStep = [&](bool is) __attribute__((always_inline)) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr ((RP & 20) != 0) {
      constexpr int WMK = WT_M * WT_N;
      constexpr int NW = K_STEPS * WMK;
      constexpr int NDR = (WT_M + WT_N) * (HIPDNN_WMMA_FRAG_ELEMS / 8);
      constexpr int WSTEP = NW / NL > 0 ? NW / NL : 1;
      int nis = is ? NL : 0;
      __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
        const int ks = w / WMK, wk = w % WMK;
        if (ks + 1 < K_STEPS) {
          if constexpr ((RP & 16) != 0) {
            const int nd = ((wk + 1) * NDR) / WMK - (wk * NDR) / WMK;
            if (nd == 1)
              __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
            else if (nd == 2)
              __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
            else if (nd == 3)
              __builtin_amdgcn_sched_group_barrier(0x100, 3, 0);
            else if (nd >= 4)
              __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
          } else if (wk == 0) {
            __builtin_amdgcn_sched_group_barrier(0x100, NDR, 0);
          }
        }
        if (w % WSTEP == WSTEP - 1 && nis > 0) {
          __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
          --nis;
        }
      }
    }
#endif
  };

  /* One K step on tile t in LDS buffer P (register slot parity Q = t & 1).
   * ST: a next tile exists to store; IS: a new tile is issued. A scheduling
   * fence ends every step: the unrolled pair is one basic block, and without
   * it the scheduler sinks half of a step's WMMAs below the barrier, keeping
   * two steps' fragments live (128x128 WT4x4: 256 VGPR + scratch). */
  auto step = [&](auto qc, auto stc, auto isc, auto rsel)
      __attribute__((always_inline)) {
    constexpr int Q = decltype(qc)::value;
    constexpr bool ST = decltype(stc)::value;
    constexpr bool IS = decltype(isc)::value;
    constexpr int P = DB ? Q : 0;
    using PC = std::integral_constant<int, P>;
    using PN = std::integral_constant<int, DB ? 1 - P : 0>;
    /* MODE 1: issue t+1 into slot 0 and store it after the WMMAs. MODE 2:
     * issue t+2 into slot Q, store t+1 from slot 1-Q. */
    using SI = std::integral_constant<int, !ROLL && MODE == 2 ? Q : 0>;
    using SS = std::integral_constant<int, !ROLL && MODE == 2 ? 1 - Q : 0>;
    if constexpr (ROLL != 0) {
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks) {
#pragma unroll
        for (int i = 0; i < NL; ++i) {
          if ((ROLL == 1 && ks == 0) || (ROLL == 2 && i * K_STEPS / NL == ks)) {
            if constexpr (ST) {
              storeOne(SS{}, PN{}, i);
              GEMM_SCHED_FENCE();
            }
            if constexpr (IS) {
              issueOne(SI{}, i, rsel);
              GEMM_SCHED_FENCE();
            }
          }
        }
        computeKs(PC{}, ks);
      }
      if constexpr (IS)
        kb += KB;
      if constexpr (ST)
        LAG4B_LDS_BARRIER();
    } else {
      if constexpr (IS)
        issueAll(SI{}, rsel);
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(PC{}, ks);
      schedStep(IS);
      if constexpr ((SF & 1) != 0)
        GEMM_SCHED_FENCE();
      if constexpr (ST) {
        if constexpr (!DB) {
          LAG4B_LDS_BARRIER();
          GEMM_SCHED_FENCE();
        }
        storeAll(SS{}, PN{});
        LAG4B_LDS_BARRIER();
      }
    }
    GEMM_SCHED_FENCE();
  };

  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using BT = std::integral_constant<bool, true>;
  using BF = std::integral_constant<bool, false>;
  using R0 = std::integral_constant<int, 0>;
  using R1 = std::integral_constant<int, 1>;
  if constexpr (MODE == 0) {
    // Current tile only: no extra global-prefetch slot or alternate LDS.
    // PAD is a multiple of eight half elements, preserving b128 alignment.
    for (int t = 0; t < niter; ++t) {
      issueAll(I0{}, R0{});
      storeAll(I0{}, I0{});
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
#pragma unroll
      for (int ks = 0; ks < K_STEPS; ++ks)
        computeKs(I0{}, ks);
      LAG4B_LDS_BARRIER();
      GEMM_SCHED_FENCE();
    }
  } else {
  /* Prologue: tile 0 in LDS buffer 0; MODE 2 also has tile 1 in flight. */
  issueAll(I0{}, R0{});
  if constexpr (!ROLL && MODE == 2)
    issueAll(I1{}, R0{});
  storeAll(I0{}, I0{});
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();

  if constexpr (ROLL != 0) {
    issueAll(I0{}, R0{});
    GEMM_SCHED_FENCE();
  }

  /* The main complete-BK tile count is even, so every step below is
   * straight-line code: the tail has no runtime cases (a branchy tail made
   * the register allocator shuffle and spill the accumulators). MODE 1 peels
   * step 0 so that each pair starts on an even issued tile index: with the
   * even wrap point jw a pair never straddles the wrap. Tiles 0 .. D are
   * issued before the wrap (jw >= 2). */
  int t = 0;
  if constexpr (D == 1) {
    step(I0{}, BT{}, BT{}, R0{});
    t = 1;
  }
  using QA = std::integral_constant<int, D == 1 ? 1 : 0>;
  using QB = std::integral_constant<int, D == 1 ? 0 : 1>;
  while (t + 1 + D < niter && t + 1 + D < jw) {
    step(QA{}, BT{}, BT{}, R0{});
    step(QB{}, BT{}, BT{}, R0{});
    t += 2;
  }
  /* Past the wrap: kb keeps counting, the resources are moved back by K. */
  while (t + 1 + D < niter) {
    step(QA{}, BT{}, BT{}, R1{});
    step(QB{}, BT{}, BT{}, R1{});
    t += 2;
  }
  /* Tail: MODE 1 has one step left (compute only), MODE 2 two (store the
   * last tile, then compute it). */
  if constexpr (D == 1) {
    step(QA{}, BF{}, BF{}, R0{});
  } else {
    step(QA{}, BT{}, BF{}, R0{});
    step(QB{}, BF{}, BF{}, R0{});
  }

  }

  // The main pipeline consumes an even number of complete BK tiles.
  // A bounded suffix handles at most one complete tile and one K16 tile.
  // It neither reads across an operand row nor changes accumulation order.
  constexpr int MAIN_K = ((FIXK / NWK / BK) & ~1) * BK;
  constexpr int TAIL_K = FIXK / NWK - MAIN_K;
  auto tailTile = [&](auto startc, auto validc) __attribute__((always_inline)) {
    constexpr int START = decltype(startc)::value;
    constexpr int VALID = decltype(validc)::value;
#pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int vo = i < A_NCH ? voffA[i] : voffB[i - A_NCH];
      if (lk < VALID) {
#if (HIPDNN_WMMA_FRAG_ELEMS == 8)
        const char *base = i < A_NCH ? Ab : Bb;
        pf[0][i] = *reinterpret_cast<const u32x4 *>(base + vo + START * ES);
#else
        if (i < A_NCH)
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(ra0, vo, START * ES, 0);
        else
          pf[0][i] = __builtin_amdgcn_raw_buffer_load_b128(rb0, vo, START * ES, 0);
#endif
      } else {
        pf[0][i] = u32x4{};
      }
    }
    LAG4B_LDS_BARRIER();
    storeAll(I0{}, I0{});
    LAG4B_LDS_BARRIER();
    GEMM_SCHED_FENCE();
#pragma unroll
    for (int ks = 0; ks < VALID / WMMA_TILE; ++ks)
      computeKs(I0{}, ks);
    GEMM_SCHED_FENCE();
  };
  if constexpr (TAIL_K >= BK)
    tailTile(std::integral_constant<int, MAIN_K>{}, std::integral_constant<int, BK>{});
  if constexpr (TAIL_K % BK != 0)
    tailTile(std::integral_constant<int, MAIN_K + (TAIL_K >= BK ? BK : 0)>{},
             std::integral_constant<int, TAIL_K % BK>{});

  // Both equal K groups finish before B staging is reused. Group 1
  // writes one FP32 partial tile into that existing LDS allocation; group 0
  // reads and adds it, then performs the ordinary SDK output conversion.
  // There is no cross-workgroup reduction or additional device allocation.
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  float *partial = reinterpret_cast<float *>(smB);
  if (group == 1) {
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm)
#pragma unroll
      for (int wn = 0; wn < WT_N; ++wn)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int rr = (wrow * WT_M + wm) * WMMA_TILE + hipdnn_wmma_acc_row(sub, e);
          const int cc = (wcol * WT_N + wn) * WMMA_TILE + lane;
          partial[rr * BN + cc] = acc[wm][wn][e];
        }
  }
  LAG4B_LDS_BARRIER();
  GEMM_SCHED_FENCE();
  if (group != 0)
    return;
#pragma unroll
  for (int wm = 0; wm < WT_M; ++wm)
#pragma unroll
    for (int wn = 0; wn < WT_N; ++wn)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = (wrow * WT_M + wm) * WMMA_TILE + hipdnn_wmma_acc_row(sub, e);
        const int cc = (wcol * WT_N + wn) * WMMA_TILE + lane;
        acc[wm][wn][e] += partial[rr * BN + cc];
      }
  GEMM_SCHED_FENCE();

  // Workgroups wholly inside Y avoid per-output execution-mask branches.
  auto epilogue = [&](auto full) __attribute__((always_inline)) {
    constexpr bool FULL = decltype(full)::value;
  /* Epilogue: GemmWmmaImpl's paired 4-byte stores (lanes 2j / 2j+1 swap one
   * value by DPP). One path per instance: the generic per-element
   * gemmStoreY path raised the register pressure past the K loop's and
   * spilled, so the launcher only routes Y = A*B and the bias case. */
#if defined(__HIP_DEVICE_COMPILE__) && !(HIPDNN_WMMA_FRAG_ELEMS == 8)
  const bool odd = (lane & 1) != 0;
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
    const int cc = nbase + (lane & ~1);
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || nbase + lane < N
               ? beta * gemmLoadC(C, 0, nbase + lane, cDim0, cDim1)
               : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; e += 2) {
        const float v0 = (RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e];
        const float v1 = (RP & 8) ? fmaf(al, acc[wm][wn][e + 1], bc) : acc[wm][wn][e + 1];
        const float x = __int_as_float(__builtin_amdgcn_update_dpp(
            0, __float_as_int(odd ? v0 : v1), 0xB1, 0xF, 0xF, false));
        const int rr = mbase + hipdnn_wmma_acc_row(sub, odd ? e + 1 : e);
        const T lo = GemmElem<T>::fromf(odd ? x : v0);
        const T hi = GemmElem<T>::fromf(odd ? v1 : x);
        uint32_t pk =
            static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
            (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi))
             << 16);
        if (FULL || (cc < N && rr < M))
          *reinterpret_cast<uint32_t *>(&Y[static_cast<size_t>(rr) * N + cc]) =
              pk;
      }
    }
  }
#else
#pragma unroll
  for (int wn = 0; wn < WT_N; ++wn) {
    const int cc = col0 + (wcol * WT_N + wn) * WMMA_TILE + lane;
    float bc = 0.0f, al = 1.0f;
    if constexpr ((RP & 8) != 0) {
      al = alpha;
      bc = FULL || cc < N ? beta * gemmLoadC(C, 0, cc, cDim0, cDim1) : 0.0f;
    }
#pragma unroll
    for (int wm = 0; wm < WT_M; ++wm) {
      const int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rr = mbase + hipdnn_wmma_acc_row(sub, e);
        if (FULL || (cc < N && rr < M))
          Y[static_cast<size_t>(rr) * N + cc] =
              GemmElem<T>::fromf((RP & 8) ? fmaf(al, acc[wm][wn][e], bc) : acc[wm][wn][e]);
      }
    }
  }
#endif
  };
  if (row0 + BM <= M && col0 + BN <= N)
    epilogue(std::true_type{});
  else
    epilogue(std::false_type{});

}


template <typename T> __global__ void __launch_bounds__(256)
__attribute__((amdgpu_waves_per_eu(2, 8)))
gemm_mid4b_wmma_c4_kernel(const T*A,const T*B,const T*C,T*Y,int M,int N,int K) {
  GemmMid4bC4Impl<64,96,2,3,T,32,true,2,0,true,1536,512,384,8,1,2,0,2>(A,B,C,Y,512,384,1536,1.f,0.f,1,1,8,1536,1536,0);
}
template <typename T>
static bool launchMeasuredMid4b(hipStream_t st, const T *A, const T *B,
                               const T *C, T *Y, int M, int N, int K,
                               int ta, int tb, float alpha, float beta) {
  if constexpr (HIPDNN_WMMA_FRAG_ELEMS == 16) {
    if (ta || !tb || alpha != 1.f || beta != 0.f ||
        ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
          reinterpret_cast<uintptr_t>(Y)) & 15))
      return false;
    constexpr bool F16 = std::is_same<T, _Float16>::value;
    int cfg = -1;
    if (M == 729 && N == 1152 && K == 4304)
      cfg = F16 ? 0 : 1;
    else if (F16 && M == 729 && N == 4304 && K == 1152)
      cfg = 2;
    else if (M == 512 && N == 384 && K == 1536)
      cfg = F16 ? 3 : 4;
    if (cfg < 0)
      return false;
    static const bool on = gemmEnvInt("HIPDNN_EP_GEMM_MID4B", 1) != 0;
    static const int forced = gemmEnvInt("HIPDNN_EP_GEMM_MID4B_CFG", -1);
    static const bool pinned =
        gemmEnvInt("HIPDNN_EP_GEMM_WMMA_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_SKINNY_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_GEMV_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_TILEDFMA_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_F32_TILE_CFG", -1) >= 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE", 0) != 0 ||
        gemmEnvInt("HIPDNN_EP_GEMM_F32_SWIZZLE", 0) != 0;
    if (!on || pinned || (forced >= 0 && forced != cfg) ||
        !gemmNnStreamGfx1151())
      return false;
    (void)hipGetLastError();
    switch (cfg) {
    // Candidate 180: five-process selection 0.176928203 ms,
    // fixed HB/candidate 1.017124; product results in README.
    case 0:
      if constexpr (std::is_same<T, _Float16>::value) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_mid4b_wmma_c1_kernel<T>),
            dim3((N + 96 - 1) / 96, (M + 96 - 1) / 96),
            dim3(128), 0, st, A, B, C, Y, M, N, K);
      } else {
        return false;
      }
      break;
    // Candidate 180: five-process selection 0.174172734 ms,
    // fixed HB/candidate 1.013783; product results in README.
    case 1:
      if constexpr (std::is_same<T, __hip_bfloat16>::value) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_mid4b_wmma_c1_kernel<T>),
            dim3((N + 96 - 1) / 96, (M + 96 - 1) / 96),
            dim3(128), 0, st, A, B, C, Y, M, N, K);
      } else {
        return false;
      }
      break;
    // Candidate 51: five-process selection 0.193918203 ms,
    // fixed HB/candidate 1.014391; product results in README.
    case 2:
      if constexpr (std::is_same<T, _Float16>::value) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_mid4b_wmma_c2_kernel<T>),
            dim3((N + 96 - 1) / 96, (M + 128 - 1) / 128),
            dim3(128), 0, st, A, B, C, Y, M, N, K);
      } else {
        return false;
      }
      break;
    // Candidate 124: five-process selection 0.019190110 ms,
    // fixed HB/candidate 1.067578; product results in README.
    case 3:
      if constexpr (std::is_same<T, _Float16>::value) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_mid4b_wmma_c3_kernel<T>),
            dim3((N + 96 - 1) / 96, (M + 64 - 1) / 64),
            dim3(256), 0, st, A, B, C, Y, M, N, K);
      } else {
        return false;
      }
      break;
    // Candidate 123: five-process selection 0.019119225 ms,
    // fixed HB/candidate 1.036827; product results in README.
    case 4:
      if constexpr (std::is_same<T, __hip_bfloat16>::value) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemm_mid4b_wmma_c4_kernel<T>),
            dim3((N + 96 - 1) / 96, (M + 64 - 1) / 64),
            dim3(256), 0, st, A, B, C, Y, M, N, K);
      } else {
        return false;
      }
      break;
    default:
      return false;
    }
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] mid4b M=%d N=%d K=%d -> mid4b-cfg[%d]\n", M, N, K, cfg);
    return true;
  }
  return false;
}
#undef LAG4B_LDS_BARRIER

// Host dispatch only: a stored legal geometry takes precedence over measured
// and skinny defaults. The GPU kernels, tile tables and launch remaps below
// are unchanged. Cache misses as well as hits so normal repeated model calls
// do not repeat a nearest-neighbour search. Online mode retains the frozen
// dispatch/timing behaviour for offline selection and its correctness leaves.
struct GemmLookupRoute {
  int cfg = -1;
};
static std::unordered_map<std::string, GemmLookupRoute> g_lookup_first_cache;

static bool gemmForcedRoute() {
  static const bool forced = [] {
    const char *names[] = {
      "HIPDNN_EP_GEMM_DECODENT_CFG", "HIPDNN_EP_GEMM_F32_SKINNY_CFG",
      "HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG", "HIPDNN_EP_GEMM_F32_TILE_CFG",
      "HIPDNN_EP_GEMM_GEMV_CFG", "HIPDNN_EP_GEMM_GEMV_KN_CFG",
      "HIPDNN_EP_GEMM_GEMV_NN_CFG", "HIPDNN_EP_GEMM_KN_RING_CFG",
      "HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG", "HIPDNN_EP_GEMM_LAGR2B_LLM_CFG",
      "HIPDNN_EP_GEMM_LAGR2B_MID_CFG", "HIPDNN_EP_GEMM_LAGR2B_NN_CFG",
      "HIPDNN_EP_GEMM_LAGR2_CFG", "HIPDNN_EP_GEMM_LONGK_CFG",
      "HIPDNN_EP_GEMM_MID4B_CFG",
      "HIPDNN_EP_GEMM_NNLARGE_CFG", "HIPDNN_EP_GEMM_NNMID_CFG",
      "HIPDNN_EP_GEMM_NNSTREAM_CFG", "HIPDNN_EP_GEMM_NNWIDE_CFG",
      "HIPDNN_EP_GEMM_OVERLAP_CFG", "HIPDNN_EP_GEMM_SKINNY_CFG",
      "HIPDNN_EP_GEMM_SKINNY_TCFG", "HIPDNN_EP_GEMM_SMALLM_CFG",
      "HIPDNN_EP_GEMM_TILEDFMA_CFG", "HIPDNN_EP_GEMM_VITUP_CFG",
      "HIPDNN_EP_GEMM_WMMA_CFG", "HIPDNN_EP_GEMM_WMMA_TA_CFG"
    };
    for (const char *name : names)
      if (gemmEnvInt(name, -1) >= 0)
        return true;
    return false;
  }();
  return forced;
}

template <typename T>
static int launchLookupFirst(hipStream_t s, const T *A, const T *B,
                               const T *C, T *Y, int M, int N, int K,
                               int transA, int transB, bool wave64,
                               float alpha, float beta, int cDim0, int cDim1) {
  constexpr int kMiss = -2;
  if (!gemmLutLookupEnabled())
    return kMiss;
  constexpr bool half = std::is_same<T, _Float16>::value ||
                        std::is_same<T, __hip_bfloat16>::value;
  gemm_lut::Kind kind = gemm_lut::Kind::Unconfigured;
  bool f32Tile = false;
  if (!transA && M == 1)
    kind = transB ? gemm_lut::Kind::GemvNt : gemm_lut::Kind::GemvNn;
  else if (!transA && M >= 16) {
    kind = half && !wave64 ? gemm_lut::Kind::Wmma : gemm_lut::Kind::TiledFma;
    if constexpr (std::is_same<T, float>::value)
      f32Tile = !wave64 && f32TileEligible(A, B, M, N, K, transA, transB);
  }
  const std::string key = std::to_string(static_cast<int>(kind)) + "_" +
      std::to_string(f32Tile) + "_" +
      hipdnn_ep::common::toString(gemmDType<T>()) + "_" +
      std::to_string(transA) + "_" + std::to_string(transB) + "_" +
      std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K);
  GemmLookupRoute route;
  bool cached = false;
  {
    std::lock_guard<std::mutex> lock(g_tune_mu);
    auto found = g_lookup_first_cache.find(key);
    if (found != g_lookup_first_cache.end()) {
      route = found->second;
      cached = true;
    }
  }
  if (!cached) {
    gemm_lut::Request request;
    request.kind = kind;
    request.act_dtype = request.wts_dtype = request.out_dtype = gemmDType<T>();
    request.trans_b = transB;
    request.m = M; request.n = N; request.k = K;
    GemmWmmaLutCtx context{M, N, K};
    const auto answer = gemm_lut::resolve(
        request, wmmaLutValidator,
        transB ? gemvNtLutValidator : gemvNnLutValidator,
        f32Tile ? f32TileLutValidator : tiledFmaLutValidator, &context);
    if (answer.source != gemm_lut::Source::None) {
      if (kind == gemm_lut::Kind::Wmma)
        route.cfg = wmmaConfigFromGeom(answer.wmma, M, N, K);
      else if (kind == gemm_lut::Kind::GemvNt)
        route.cfg = gemvNtConfigFromGeom(answer.gemv);
      else if (kind == gemm_lut::Kind::GemvNn)
        route.cfg = gemvNnConfigFromGeom(answer.gemv);
      else if (kind == gemm_lut::Kind::TiledFma)
        route.cfg = f32Tile ? f32TileConfigFromGeom(answer.tiled_fma)
                            : tiledFmaConfigFromGeom(answer.tiled_fma);
    }
    std::lock_guard<std::mutex> lock(g_tune_mu);
    g_lookup_first_cache.emplace(key, route);
  }
  // Explicit diagnostic configs retain priority, including measured families.
  // Their operation is still custom GEMM; there is no vendor retry.
  if (gemmForcedRoute() || route.cfg < 0) {
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] LUT-first kind=%d cfg=%d -> default resolver%s\n",
        static_cast<int>(kind), route.cfg, gemmForcedRoute() ? " (explicit config)" : "");
    return kMiss;
  }
  CUSTOM_KERNELS_DEBUG_LOG("[gemm] LUT-first kind=%d cfg=%d M=%d N=%d K=%d\n",
      static_cast<int>(kind), route.cfg, M, N, K);
  (void)hipGetLastError();
  if (kind == gemm_lut::Kind::Wmma) {
    if constexpr (half) {
      if (transB)
        launchWmma<T, true>(route.cfg, s, A, B, C, Y, M, N, K, alpha, beta,
                            cDim0, cDim1, K);
      else
        launchWmma<T, false>(route.cfg, s, A, B, C, Y, M, N, K, alpha, beta,
                             cDim0, cDim1, K);
      return hipGetLastError() == hipSuccess ? 0 : 1;
    }
  } else if (kind == gemm_lut::Kind::GemvNt) {
    launchGemvNt<T>(route.cfg, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
    return hipGetLastError() == hipSuccess ? 0 : 1;
  } else if (kind == gemm_lut::Kind::GemvNn) {
    launchGemvNn<T>(route.cfg, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
    return hipGetLastError() == hipSuccess ? 0 : 1;
  } else if (kind == gemm_lut::Kind::TiledFma) {
    if constexpr (std::is_same<T, float>::value)
      if (f32Tile)
        return launchF32Tile(route.cfg, s, A, B, C, Y, M, N, K, transB,
                                 alpha, beta, cDim0, cDim1);
    return launchTiledFma<T>(route.cfg, s, A, B, C, Y, M, N, K, transA, transB,
                                alpha, beta, cDim0, cDim1);
  }
  return kMiss;
}

extern "C" int hip_gemm(void *stream, const void *A, const void *B,
                        const void *C, void *Y, int M, int N, int K,
                        float alpha, float beta, int transA, int transB,
                        int type_code, int cDim0, int cDim1) {
  if (!stream || !A || !B || !Y || M < 0 || N < 0 || K < 0)
    return -1;
  if (M == 0 || N == 0)
    return 0;

  hipStream_t s = static_cast<hipStream_t>(stream);
  transA = transA ? 1 : 0;
  transB = transB ? 1 : 0;
  if (cDim0 <= 0)
    cDim0 = 1;
  if (cDim1 <= 0)
    cDim1 = 1;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] hip_gemm M=%d N=%d K=%d ta=%d tb=%d ty=%d a=%g b=%g C=%s "
      "c=[%d,%d]\n",
      M, N, K, transA, transB, type_code, alpha, beta, C ? "yes" : "null",
      cDim0, cDim1);

  if (K == 0) {
    if (type_code == kTypeFloat16)
      return launchKZero<_Float16>(s, static_cast<const _Float16 *>(C),
                                   static_cast<_Float16 *>(Y), M, N, beta,
                                   cDim0, cDim1);
    if (type_code == kTypeBFloat16)
      return launchKZero<__hip_bfloat16>(
          s, static_cast<const __hip_bfloat16 *>(C),
          static_cast<__hip_bfloat16 *>(Y), M, N, beta, cDim0, cDim1);
    if (type_code == kTypeFloat32)
      return launchKZero<float>(s, static_cast<const float *>(C),
                                static_cast<float *>(Y), M, N, beta, cDim0,
                                cDim1);
    if (type_code == kTypeFloat64)
      return launchKZero<double>(s, static_cast<const double *>(C),
                                 static_cast<double *>(Y), M, N, beta, cDim0,
                                 cDim1);
    return -1;
  }

  const bool wave64 = hipdnn_device_is_wave64();
  constexpr int kWmmaM = 16; /* llama.cpp: VEC below this */

#define GEMM_LOOKUP_FIRST(T)                                                   \
  do {                                                                        \
    int rc = launchLookupFirst<T>(s, static_cast<const T *>(A),                 \
        static_cast<const T *>(B), static_cast<const T *>(C),                   \
        static_cast<T *>(Y), M, N, K, transA, transB, wave64,                    \
        alpha, beta, cDim0, cDim1);                                            \
    if (rc != -2) return rc;                                                   \
  } while (0)
  if (type_code == kTypeFloat16) { GEMM_LOOKUP_FIRST(_Float16); }
  else if (type_code == kTypeBFloat16) { GEMM_LOOKUP_FIRST(__hip_bfloat16); }
  else if (type_code == kTypeFloat32) { GEMM_LOOKUP_FIRST(float); }
  else if (type_code == kTypeFloat64) { GEMM_LOOKUP_FIRST(double); }
#undef GEMM_LOOKUP_FIRST

  auto go_f16 = [&]() -> int {
    using T = _Float16;
    auto *a = static_cast<const T *>(A);
    auto *b = static_cast<const T *>(B);
    auto *c = static_cast<const T *>(C);
    auto *y = static_cast<T *>(Y);
    if(!wave64 && launchMeasuredLagR2bBigbHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bNnHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bMidHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bLlmHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2Half<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if (!wave64 && launchMeasuredDecodeNt<T>(s, a, b, c, y, M, N, K,
                                            transA, transB, alpha, beta,
                                            cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredMid4b<T>(s, a, b, c, y, M, N, K,
                                         transA, transB, alpha, beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredVitup<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredNnLarge<T>(s,a,b,y,M,N,K,transA,transB,alpha,beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredSmallM<T>(s, a, b, c, y, M, N, K,
                                          transA, transB, alpha, beta,
                                          cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && !transA && !transB &&
        launchNnWide(s,a,b,y,M,N,K,alpha,beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && !transA && !transB &&
        launchNnStream<T>(s,a,b,y,M,N,K,alpha,beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && !transA && !transB &&
        launchNnMid<T>(s, a, b, c, y, M, N, K, alpha, beta, cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && gemmSkinnyEligible(M, transA, transB) &&
        launchSkinny<T>(s, a, b, c, y, M, N, K, transA, transB, alpha, beta,
                        cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (transA && M >= kWmmaM && !wave64)
      return launchWmmaGemmTa<T>(s, a, b, c, y, M, N, K, transB, alpha, beta,
                                 cDim0, cDim1);
    if (M < kWmmaM || transA)
      return launchSmallM<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                             beta, cDim0, cDim1);
    if (wave64) {
      int cfg = tuneTiledFma<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                                beta, cDim0, cDim1);
      return launchTiledFma<T>(cfg, s, a, b, c, y, M, N, K, transA, transB,
                               alpha, beta, cDim0, cDim1);
    }
    return launchWmmaGemm<T>(s, a, b, c, y, M, N, K, transB, alpha, beta,
                             cDim0, cDim1, K);
  };

  auto go_bf16 = [&]() -> int {
    using T = __hip_bfloat16;
    auto *a = static_cast<const T *>(A);
    auto *b = static_cast<const T *>(B);
    auto *c = static_cast<const T *>(C);
    auto *y = static_cast<T *>(Y);
    if(!wave64 && launchMeasuredLagR2bBigbHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bNnHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bMidHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bLlmHalf<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2Half<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if (!wave64 && launchMeasuredMid4b<T>(s, a, b, c, y, M, N, K,
                                         transA, transB, alpha, beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredVitup<T>(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredNnLarge<T>(s,a,b,y,M,N,K,transA,transB,alpha,beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchMeasuredSmallM<T>(s, a, b, c, y, M, N, K,
                                          transA, transB, alpha, beta,
                                          cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && !transA && !transB &&
        launchNnStream<T>(s,a,b,y,M,N,K,alpha,beta))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && !transA && !transB &&
        launchNnMid<T>(s, a, b, c, y, M, N, K, alpha, beta, cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && gemmSkinnyEligible(M, transA, transB) &&
        launchSkinny<T>(s, a, b, c, y, M, N, K, transA, transB, alpha, beta,
                        cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (transA && M >= kWmmaM && !wave64)
      return launchWmmaGemmTa<T>(s, a, b, c, y, M, N, K, transB, alpha, beta,
                                 cDim0, cDim1);
    if (M < kWmmaM || transA)
      return launchSmallM<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                             beta, cDim0, cDim1);
    if (wave64) {
      int cfg = tuneTiledFma<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                                beta, cDim0, cDim1);
      return launchTiledFma<T>(cfg, s, a, b, c, y, M, N, K, transA, transB,
                               alpha, beta, cDim0, cDim1);
    }
    return launchWmmaGemm<T>(s, a, b, c, y, M, N, K, transB, alpha, beta,
                             cDim0, cDim1, K);
  };

  if (type_code == kTypeFloat16)
    return go_f16();
  if (type_code == kTypeBFloat16)
    return go_bf16();
  if (type_code == kTypeFloat32) {
    if (!wave64 && launchMeasuredVitupF32(s,static_cast<const float*>(A),static_cast<const float*>(B),static_cast<const float*>(C),static_cast<float*>(Y),M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;

    auto *a = static_cast<const float *>(A);
    auto *b = static_cast<const float *>(B);
    auto *c = static_cast<const float *>(C);
    auto *y = static_cast<float *>(Y);
    if(!wave64 && launchMeasuredLagR2bBigbF32(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bNnF32(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bMidF32(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2bLlmF32(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if(!wave64 && launchMeasuredLagR2F32(s,a,b,c,y,M,N,K,transA,transB,alpha,beta,cDim0,cDim1))
      return hipGetLastError()==hipSuccess?0:1;
    if (!wave64 && launchF32Skinny(s, a, b, c, y, M, N, K, transA, transB,
                                   alpha, beta, cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (!wave64 && launchF32SkinnyNn(s, a, b, c, y, M, N, K, transA, transB,
                                     alpha, beta, cDim0, cDim1))
      return hipGetLastError() == hipSuccess ? 0 : 1;
    if (M < kWmmaM)
      return launchSmallM<float>(s, a, b, c, y, M, N, K, transA, transB,
                                 alpha, beta, cDim0, cDim1);
    if (!wave64 && f32TileEligible(a, b, M, N, K, transA, transB)) {
      const int cfg = tuneF32Tile(s, a, b, c, y, M, N, K, transB, alpha, beta,
                                  cDim0, cDim1);
      return launchF32Tile(cfg, s, a, b, c, y, M, N, K, transB, alpha, beta,
                           cDim0, cDim1);
    }
    {
      int cfg = tuneTiledFma<float>(s, a, b, c, y, M, N, K, transA, transB,
                                    alpha, beta, cDim0, cDim1);
      return launchTiledFma<float>(cfg, s, a, b, c, y, M, N, K, transA, transB,
                                   alpha, beta, cDim0, cDim1);
    }
  }
  if (type_code == kTypeFloat64) {
    auto *a = static_cast<const double *>(A);
    auto *b = static_cast<const double *>(B);
    auto *c = static_cast<const double *>(C);
    auto *y = static_cast<double *>(Y);
    if (M < kWmmaM)
      return launchSmallM<double>(s, a, b, c, y, M, N, K, transA, transB,
                                  alpha, beta, cDim0, cDim1);
    {
      int cfg = tuneTiledFma<double>(s, a, b, c, y, M, N, K, transA, transB,
                                     alpha, beta, cDim0, cDim1);
      return launchTiledFma<double>(cfg, s, a, b, c, y, M, N, K, transA,
                                    transB, alpha, beta, cDim0, cDim1);
    }
  }
  return -1;
}
