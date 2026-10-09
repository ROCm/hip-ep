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
  return GemmElem<T>::tof(C[cr * cDim1 + cc]);
}

template <typename T>
__device__ __forceinline__ void gemmStoreY(T *Y, const T *C, int r, int c,
                                           int N, int cDim0, int cDim1,
                                           float acc, float alpha, float beta) {
  float v = acc * alpha;
  if (beta != 0.0f && C)
    v = fmaf(beta, gemmLoadC(C, r, c, cDim0, cDim1), v);
  Y[r * N + c] = GemmElem<T>::fromf(v);
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
  float acc = 0.0f;
  int k = 0;
  for (; k + 4 <= K; k += 4) {
    acc = __fmaf_rn(GemmElem<T>::tof(a[k]), GemmElem<T>::tof(b[k * N]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 1]),
                    GemmElem<T>::tof(b[(k + 1) * N]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 2]),
                    GemmElem<T>::tof(b[(k + 2) * N]), acc);
    acc = __fmaf_rn(GemmElem<T>::tof(a[k + 3]),
                    GemmElem<T>::tof(b[(k + 3) * N]), acc);
  }
  for (; k < K; ++k)
    acc = __fmaf_rn(GemmElem<T>::tof(a[k]), GemmElem<T>::tof(b[k * N]), acc);

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

/* Multi-row NT GEMV (fp16/bf16, 2 <= M < 16, B = [N, K]): one block owns TN
 * output columns for ALL M rows, so B -- the large operand -- is streamed
 * once (16-byte loads, 8 k per thread per chunk) instead of once per row as
 * in gemm_gemv_nt_kernel. TA=false: A is [M, K], each row's 8-k run is one
 * 16-byte load (L2-resident, A is tiny). TA=true: A is [K, M]; for a chunk of
 * KC k-rows the slice A[kc .. kc+KC)[0 .. M) is one contiguous run, copied to
 * LDS with 16-byte loads and read back per (k, m). Requires M <= MR. */
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
          gemmUnpack8<T>(*reinterpret_cast<const u32x4 *>(brow[tn] + kc + k),
                         bf[tn]);
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
              gemmUnpack8<T>(*reinterpret_cast<const u32x4 *>(arow), af);
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

/* B = [K, N] GEMV (fp16/bf16, small M): each thread owns 8 consecutive output
 * columns (one 16-byte B load per k-row) and a 1/KS share of K; the KS
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
      gemmUnpack8<T>(*reinterpret_cast<const u32x4 *>(brow), b);
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
 *                      the in-process sweep only on a miss.
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
static void *g_tune_out = nullptr;
static size_t g_tune_out_bytes = 0;

static void *ensureTuneOut(size_t bytes) {
  if (g_tune_out && g_tune_out_bytes >= bytes)
    return g_tune_out;
  if (g_tune_out)
    hipFree(g_tune_out);
  g_tune_out = nullptr;
  g_tune_out_bytes = 0;
  if (hipMalloc(&g_tune_out, bytes) != hipSuccess)
    return nullptr;
  g_tune_out_bytes = bytes;
  return g_tune_out;
}

static void *g_sk_buf = nullptr;
static size_t g_sk_bytes = 0;

static void *ensureSkBuf(size_t bytes) {
  if (g_sk_buf && g_sk_bytes >= bytes)
    return g_sk_buf;
  if (g_sk_buf)
    hipFree(g_sk_buf);
  g_sk_buf = nullptr;
  g_sk_bytes = 0;
  if (hipMalloc(&g_sk_buf, bytes) != hipSuccess)
    return nullptr;
  g_sk_bytes = bytes;
  return g_sk_buf;
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
  } else if (gemmAutotuneOn()) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *scratch = static_cast<T *>(
        ensureTuneOut(static_cast<size_t>(M) * N * sizeof(T)));
    T *out = scratch ? scratch : Y;

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
  } else if (gemmAutotuneOn()) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *scratch = static_cast<T *>(
        ensureTuneOut(static_cast<size_t>(M) * N * sizeof(T)));
    T *out = scratch ? scratch : Y;

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
  int id = gemmEnvInt("HIPDNN_EP_GEMM_GEMV_KN_CFG", -1);
  if (id < 0 || id > 2) {
    id = 2;
    for (int i = 0; i < 2; ++i)
      if (((N + kCols[i] - 1) / kCols[i]) * ((M + MR - 1) / MR) >= 80) {
        id = i;
        break;
      }
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
  constexpr int A_CH = (A_HPT >= 8) ? 8 : 4;
  constexpr int A_NCH = A_HPT / A_CH;
  constexpr int A_ROW_SPLIT = BK / A_CH;

  constexpr int B_HPT = (BN * BK) / THR;
  constexpr int B_U32 = B_HPT / 8; /* TransB: 8 halfs per u32x4 */
  constexpr int B_TPR = BK / B_HPT;

  static_assert(A_NCH >= 1, "A_NCH");
  static_assert(A_HPT % A_CH == 0, "A_HPT");
  static_assert(BK % A_CH == 0, "BK/A_CH");
  static_assert(BK % WMMA_TILE == 0, "BK/WMMA");
  static_assert((K_STR * 2) % 16 == 0, "LDS stride 16B");
  static_assert(B_HPT >= 4 && B_HPT % 4 == 0, "B_HPT");
  static_assert(B_HPT <= 64, "pf_b cap");

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

  constexpr int B_COL_SHIFT = __builtin_ctz(B_TPR);
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
    a_src[c] = A + static_cast<size_t>(row0 + a_row[c]) * lda + k_first +
               a_koff[c];
  }

  /* ldb is the B row stride in elements; == K unless the caller passed a
   * row-padded weight (padrow, power-of-two-K L1 conflict fix). Only the row
   * base scales by ldb -- the within-row k range is still [0, K). */
  const T *b_src_nt = B + static_cast<size_t>(b_n_g) * ldb + k_first +
                      b_k8; /* TransB=1 [N,K] */

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
  static_assert(PGR == 1 || PGR == 2, "PGR");
  constexpr int NSLOT = PGR;
  uint32_t pf_a[NSLOT][A_NCH][A_CH / 2];
  uint32_t pf_ta[NSLOT][TA_NCH][TA_KR][TA_MC / 2];
  u32x4 pf_b[NSLOT][8];
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

    if constexpr (TransB) {
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
      if (!BC || (gk < K && gn + N_CH <= N)) {
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
    if constexpr (TransB) {
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

  if (k_begin < k_end) {
    using S0 = std::integral_constant<int, 0>;
    auto nextK = [&](int k) __attribute__((always_inline)) {
      k += BK;
      return k >= k_end ? k_begin : k;
    };
    int kc = k_first;
    issueLoads(kc, S0{});
    if constexpr (PGR == 2) {
      using S1 = std::integral_constant<int, 1>;
      if (niter > 1) {
        /* PGR2: tile t lives in slot t%2; each step issues tile it+1 and
         * stores tile it (issued one step earlier), so a tile has two compute
         * steps to arrive. The final step issues one wrapped, valid tile that
         * is never stored, which keeps every step's load count identical. */
        kc = nextK(kc);
        issueLoads(kc, S1{});
        storeToShared(0, S0{});
        HIPDNN_LDS_BARRIER();
        int buf = 0;
        int it = 1;
        while (true) {
          kc = nextK(kc);
          issueLoads(kc, S0{});
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
          /* Tile covers M and K (BC=false path requires M%BM==0, K%BK==0), but
           * N may be padded up to a BN multiple by the caller (N-ragged fast
           * path: launch full tiles, mask the column overhang here). The store
           * is the epilogue -- one cheap `cc < N` compare per element, outside
           * the K-loop -- so it does not touch the hot path. For aligned N it is
           * a predicted-taken branch. */
          if (cc < N) {
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

template <typename T>
__global__ void __launch_bounds__(256)
    gemm_reduce_sk_kernel(const float *__restrict__ parts, T *__restrict__ Y,
                          const T *__restrict__ C, int M, int N, int sk,
                          float alpha, float beta, int cDim0, int cDim1) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int total = M * N;
  if (i >= total)
    return;
  float acc = 0.0f;
#pragma unroll 1
  for (int z = 0; z < sk; ++z)
    acc += parts[static_cast<size_t>(z) * static_cast<size_t>(total) + i];
  const int r = i / N;
  const int c = i - r * N;
  gemmStoreY(Y, C, r, c, N, cDim0, cDim1, acc, alpha, beta);
}

struct WmmaCfg {
  int bm, bn, wt_m, wt_n, swizzle_n, split_k, bk;
};

/* Tile table: occupancy vs arithmetic intensity on gfx1151 (~40 CUs).
 * split_k grows the grid when ceil(M/BM)*ceil(N/BN) would otherwise underfill.
 * BK=64 raises arithmetic intensity when (BM+BN)*(BK+8)*4 fits in 64 KB LDS. */
static constexpr WmmaCfg kWmma[] = {
    {64, 64, 2, 2, 8, 1, 32},    {64, 128, 2, 2, 8, 1, 32},
    {128, 64, 2, 2, 8, 1, 32},   {128, 128, 2, 2, 8, 1, 32},
    {128, 256, 2, 2, 8, 1, 32},  {256, 128, 4, 2, 8, 1, 32},
    {128, 128, 2, 4, 8, 1, 32},  {64, 32, 2, 1, 8, 1, 32},
    {128, 32, 2, 1, 8, 1, 32},   {64, 256, 2, 2, 8, 1, 32},
    {64, 128, 2, 4, 8, 1, 32},   {128, 256, 2, 4, 8, 1, 32},
    {128, 128, 2, 2, 4, 1, 32},  {128, 128, 2, 2, 16, 1, 32},
    {128, 256, 2, 2, 16, 1, 32}, {64, 128, 2, 2, 8, 2, 32},
    {128, 128, 2, 2, 8, 2, 32},  {128, 256, 2, 2, 8, 2, 32},
    {64, 256, 2, 2, 8, 2, 32},   {256, 128, 4, 2, 8, 2, 32},
    {64, 64, 2, 2, 8, 4, 32},    {64, 128, 2, 2, 8, 4, 32},
    {128, 64, 2, 2, 8, 4, 32},   {128, 128, 2, 2, 8, 4, 32},
    {64, 256, 2, 2, 8, 4, 32},   {64, 128, 2, 4, 8, 1, 64},
    {64, 128, 2, 2, 8, 1, 64},   {128, 64, 2, 2, 8, 1, 64},
    {64, 64, 2, 2, 8, 1, 64},    {64, 128, 2, 4, 8, 2, 64},
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
  if (c.split_k > 1) {
    const int nkt = (K + c.bk - 1) / c.bk;
    if (nkt < c.split_k || K < 256)
      return false;
    const size_t elems =
        static_cast<size_t>(c.split_k) * static_cast<size_t>(M) * N;
    if (elems > (16ull << 20))
      return false;
  }
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
   *     is handled at launch by the K%1024 alias rule in launchWmma (padrow is
   *     opt-in only now).
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

template <typename T, bool TransB>
static void launchWmma(int id, hipStream_t s, const T *A, const T *B,
                       const T *C, T *Y, int M, int N, int K, float alpha,
                       float beta, int cDim0, int cDim1, int ldb,
                       int lda = -1) {
  /* A row stride: defaults to K. pad-K mode passes lda=K_real with K=K_pad so
   * the loop/B use the BK-aligned padded K while A keeps its real stride. */
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
   *     un-ruled level (4096x4096^2 fp16 4.27 / bf16 4.27 ms, PADROW 4.19 /
   *     4.09); 128x128 WT4x2 BK=64 with a 16-wide M-first group runs 3.86 /
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
  if (alias_k && c.split_k == 1) {
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
  if (my_tiles >= 12 && c.split_k == 1 &&
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
  if (c.split_k == 1 && c.bn >= 256 && my_tiles <= 8 &&
      static_cast<size_t>(N) * K * 2 > (96ull << 20))
    sw_eff = 4;
  if (alias_k)
    sw_eff = (c.bk == 64 && c.bm == 128 && my_tiles >= 2 * kAliasWgm &&
              static_cast<size_t>(2 * kAliasWgm) * c.bm * K * sizeof(T) <=
                  (16ull << 20))
                 ? -2 * kAliasWgm
                 : -kAliasWgm;
  const bool pgr2 = alias_k && c.bk == 32;
  {
    /* <0 selects the M-first raster (see gemmSwizzleBlock). */
    const int sw_env = gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE", 0);
    if (sw_env != 0)
      sw_eff = sw_env;
  }
  const int stag = gemmStaggerParam(c.bk, static_cast<int>(sizeof(T)), TransB);
  const int th = (c.bm / (c.wt_m * 16)) * (c.bn / (c.wt_n * 16)) * 32;
  int sk = c.split_k;
  float *skp = nullptr;
  if (sk > 1) {
    const size_t bytes =
        static_cast<size_t>(sk) * static_cast<size_t>(M) * N * sizeof(float);
    skp = static_cast<float *>(ensureSkBuf(bytes));
    if (!skp)
      sk = 1;
  }
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
  const bool db =
      db_lds <= 49152 || (c.bk == 64 && db_lds <= 57344 && spills_l2);

  /* Sub-launches. Normally one launch covering the whole [M,N] grid with a
   * single compile-time BC. When only N is ragged (M,K tile-aligned, no
   * split-K), launch ceil(N/BN) FULL tiles all on the fast BC=false path: the
   * B-load is already N-safe (b_valid reads a dummy row for the column
   * overhang) and the store masks cc>=N (a cheap epilogue compare). Measured on
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
  if (sk == 1 && n_ragged && (M % c.bm == 0) && (K % c.bk == 0)) {
    subs[0] = SubLaunch{dim3((N + c.bn - 1) / c.bn, my, 1), false, 0};
    nsub = 1;
  } else {
    const bool bc =
        !(M % c.bm == 0 && N % c.bn == 0 && K % c.bk == 0) || sk > 1;
    subs[0] =
        SubLaunch{dim3((N + c.bn - 1) / c.bn, my, sk > 1 ? sk : 1), bc, 0};
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

  for (int sidx = 0; sidx < nsub; ++sidx) {
    const dim3 grid = subs[sidx].grid;
    const bool bc = subs[sidx].bc;
    const int nb = subs[sidx].nb;

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
#undef GEMM_WMMA_TRY_PGR
#undef GEMM_WMMA_TRY
#undef GEMM_WMMA_ONE_P
#undef GEMM_WMMA_ONE

  if (sk > 1 && skp) {
    const int total = M * N;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_reduce_sk_kernel<T>),
                       dim3((total + 255) / 256), dim3(256), 0, s, skp, Y, C, M,
                       N, sk, alpha, beta, cDim0, cDim1);
  }
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
  } else if (gemmAutotuneOn()) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *scratch = static_cast<T *>(
        ensureTuneOut(static_cast<size_t>(M) * N * sizeof(T)));
    T *out = scratch ? scratch : Y;
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
  const int forced = gemmEnvInt("HIPDNN_EP_GEMM_WMMA_TA_CFG", -1);
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
    } else if (gemmAutotuneOn()) {
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      T *scratch = static_cast<T *>(
          ensureTuneOut(static_cast<size_t>(M) * N * sizeof(T)));
      T *out = scratch ? scratch : Y;

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

template <typename T>
__global__ void gemm_kzero_kernel(const T *__restrict__ C, T *__restrict__ Y,
                                 int M, int N, float beta, int cDim0,
                                 int cDim1) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (i >= total)
    return;
  const int r = i / N;
  const int c = i - r * N;
  gemmStoreY(Y, C, r, c, N, cDim0, cDim1, 0.0f, 1.0f, beta);
}

template <typename T>
static int launchKZero(hipStream_t s, const T *C, T *Y, int M, int N,
                       float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
  const int total = M * N;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_kzero_kernel<T>),
                     dim3((total + 255) / 256), dim3(256), 0, s, C, Y, M, N,
                     beta, cDim0, cDim1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

/* =========================================================================
 * Padrow: L1 set-conflict fix for power-of-two-K NT weights
 *
 * NT reads BN rows of B[N,K] that are K*sizeof(T) bytes apart. A power-of-two
 * K makes 128 tile-rows land in a handful of L1 sets and evict each other
 * (measured ~27% loss at K=2048). Padding the row stride by one cache line
 * spreads them over every set. The padded copy is NOT cached (the old
 * B-device-pointer cache and its hit-rate guard are gone): it is repacked into
 * one shared scratch on every call, so padrow is opt-in
 * (HIPDNN_EP_GEMM_PADROW=1) for A/B only. The default fix for the same
 * aliasing is the launch-side K%1024 rule in launchWmma (M-first raster +
 * StaggerU + PGR2, HIPDNN_EP_GEMM_ALIASK), which needs no copy: measured
 * 4096x4096^2 fp16 3.86 ms vs 4.19 with PADROW=1 (bf16 3.82 vs 4.09).
 * ========================================================================= */

/* Pad the row stride by one 128 B cache line, in elements. */
static constexpr int kGemmPadRowBytes = 128;

static bool gemmIsPow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

static bool gemmPadRowEnabled() {
  static const bool on = [] {
    std::string s = gemmEnvStr("HIPDNN_EP_GEMM_PADROW");
    return !s.empty() && s[0] == '1'; /* opt-in: per-call copy of B, A/B only */
  }();
  return on;
}

/* Eligible: NT, fp16/bf16 (WMMA path), and a power-of-two K >= 2048 (the
 * measured L1-conflict range).
 *
 * The remaining gate is memory-bound vs compute-bound, NOT weight size: the
 * B-load L1 conflict only bottlenecks when the GEMM is memory-bound. Measured:
 *   - Small M (<=256): padrow always wins, even for a 128 MB weight
 *     (M=128 N=8192 K=8192 is ~5x faster with padrow) -- the tile re-reads B
 *     with little compute to hide the conflict.
 *   - Large M (>256), mid-size weight (<=64 MB): still wins with the default
 *     256x128 tile (M=512 N=4096 K=4096, 32 MB: +39%).
 *   - Large M (>256), large weight (>64 MB) with a non-severe pow2 stride: the
 *     conflict is hidden by compute and the padded footprint costs more than it
 *     saves with the wide 256x128 tile, so it is gated off there.
 *   - Bad K stride (K % 1024 == 0): the B/A row stride (K*2 bytes, a multiple of
 *     2 KB) aliases few L1 sets and collapses cfg5 throughput. Measured (cfg5
 *     M=1024, cooled): K=6144 27->41 TFLOP with padrow (-38% -> -4% vs hipBLASLt),
 *     K=5120 37->44 (+1% lead), while K=7680 (%1024=512) is already 43 and padrow
 *     is a no-op (+1%). This is a superset of the power-of-two case and catches
 *     the non-pow2 dips (6144, 3072). These numbers predate the per-call copy
 *     (they assumed a cached weight); with the copy paid on every call padrow
 *     stays opt-in, and a bad-K weight is eligible regardless of size.
 * The 64 MB bound is empirical, to be refined by the offline LUT. */
static bool gemmPadRowEligible(int M, int N, int K, int transA, int transB,
                               int type_bytes) {
  const size_t weight_bytes =
      static_cast<size_t>(N) * K * static_cast<size_t>(type_bytes);
  static const bool force_bound = [] {
    std::string s = gemmEnvStr("HIPDNN_EP_GEMM_PADROW_FORCE");
    return !s.empty() && s[0] != '0';
  }();
  /* K*2 a multiple of 2 KB -> row stride lands on few L1 sets -> aliasing. This
   * catches every K that collapses cfg5, and padrow recovers it at any weight
   * size, so a bad-K weight bypasses the compute-bound size bound. */
  const bool bad_stride = (K % 1024) == 0 && K >= 2048;
  const bool bound_ok = force_bound || (M <= 256) ||
                        (weight_bytes <= (64ull << 20)) || bad_stride;
  return gemmPadRowEnabled() && transB && !transA && type_bytes == 2 &&
         M >= 16 && N >= 64 && bad_stride && bound_ok;
}

template <typename T>
__global__ void gemm_padrow_copy_kernel(const T *__restrict__ src,
                                        T *__restrict__ dst, int N, int K,
                                        int ldb) {
  const int row = blockIdx.y;
  if (row >= N)
    return;
  const size_t s0 = static_cast<size_t>(row) * K;
  const size_t d0 = static_cast<size_t>(row) * ldb;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < K;
       k += gridDim.x * blockDim.x)
    dst[d0 + k] = src[s0 + k];
}

/* One reusable scratch for the padded B copy (row-pad or K-extend), shared
 * across every gemm call and grown to the largest single shape it sees -- the
 * same pattern MatMulNBits uses (g_bpad_buf / ensureBPadBuffer). The padded B
 * is a transient WMMA input, not persistent state, so it is REPACKED from the
 * current B on every call. This deliberately replaces the former g_pad_map
 * keyed by the raw B device pointer: that cache returned a stale padded copy
 * whenever a caller freed B and the allocator recycled the address for a
 * different tensor (a clean unit test that malloc/frees B per case, or any
 * dynamic-B use), silently computing on the wrong weights. Repacking each call
 * is a single strided copy, cheap next to the GEMM, and bounds the footprint to
 * one shape instead of the sum of every padrow weight. */
static std::mutex g_pad_mu;
static void *g_pad_buf = nullptr;
static size_t g_pad_bytes = 0;

static void *gemmEnsurePadScratch(size_t bytes) {
  if (g_pad_buf && g_pad_bytes >= bytes)
    return g_pad_buf;
  if (g_pad_buf)
    hipFree(g_pad_buf);
  if (hipMalloc(&g_pad_buf, bytes) != hipSuccess) {
    g_pad_buf = nullptr;
    g_pad_bytes = 0;
    return nullptr;
  }
  g_pad_bytes = bytes;
  return g_pad_buf;
}

static size_t gemmPadRowMaxBytes() {
  static const size_t cap = [] {
    std::string s = gemmEnvStr("HIPDNN_EP_GEMM_PADROW_MAX_MB");
    if (s.empty())
      return static_cast<size_t>(512) * 1024 * 1024;
    return static_cast<size_t>(strtoull(s.c_str(), nullptr, 10)) * 1024 * 1024;
  }();
  return cap;
}

/* Return a padded (row-stride = ldb) copy of B[N,K], or nullptr to fall back to
 * the arrival layout. Sets *out_ldb on success. */
template <typename T>
static const T *gemmEnsurePaddedB(hipStream_t s, const T *B, int N, int K,
                                  int *out_ldb) {
  const int pad = kGemmPadRowBytes / static_cast<int>(sizeof(T));
  const int ldb = K + pad;
  const size_t bytes = static_cast<size_t>(N) * ldb * sizeof(T);
  const size_t cap = gemmPadRowMaxBytes();
  if (cap == 0 || bytes > cap)
    return nullptr; /* padrow off, or one weight bigger than the whole cap */

  std::lock_guard<std::mutex> lk(g_pad_mu);
  void *buf = gemmEnsurePadScratch(bytes);
  if (!buf)
    return nullptr;
  /* Repack the CURRENT B into the shared scratch. The pad columns [K,ldb) only
   * change the row stride and are never read by the GEMM, so they need no
   * zeroing. */
  dim3 grid((K + 255) / 256 < 1 ? 1 : (K + 255) / 256, N);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_padrow_copy_kernel<T>), grid,
                     dim3(256), 0, s, B, static_cast<T *>(buf), N, K, ldb);
  if (hipGetLastError() != hipSuccess)
    return nullptr;
  *out_ldb = ldb;
  return static_cast<const T *>(buf);
}

/* K-extend a reused NT weight B[N,K] to B'[N,K_pad] with K_pad a multiple of 32
 * and the tail [K,K_pad) zero-filled, so a K%32!=0 shape (vision K=4304) can run
 * the fast BK=32 BC=false path instead of the 2x-barrier BK=16 tile. A keeps its
 * real stride (K): the A-load reads garbage in the [K,K_pad) tail, but B's zeros
 * null it (sum_k A*0 = 0), so no A copy is needed. Repacked into the shared
 * padrow scratch each call; returns nullptr to fall back. Sets *out_kpad. The A over-read is <=
 * (K_pad-K)*2 bytes past the last row -- within the pooled/aligned A slack. */
template <typename T>
static const T *gemmEnsureKPaddedB(hipStream_t s, const T *B, int N, int K,
                                   int *out_kpad) {
  const int kpad = ((K + 31) / 32) * 32;
  if (kpad == K)
    return nullptr; /* already aligned; caller uses the normal path */
  const size_t bytes = static_cast<size_t>(N) * kpad * sizeof(T);
  const size_t cap = gemmPadRowMaxBytes();
  if (cap == 0 || bytes > cap)
    return nullptr;

  std::lock_guard<std::mutex> lk(g_pad_mu);
  void *buf = gemmEnsurePadScratch(bytes);
  if (!buf)
    return nullptr;
  /* K-extend: the GEMM reads all kpad columns, so the tail [K,kpad) must be
   * zero (sum_k A*0 = 0). Zero then repack the current B into the shared
   * scratch each call. */
  if (hipMemsetAsync(buf, 0, bytes, s) != hipSuccess)
    return nullptr;
  dim3 grid((K + 255) / 256 < 1 ? 1 : (K + 255) / 256, N);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_padrow_copy_kernel<T>), grid,
                     dim3(256), 0, s, B, static_cast<T *>(buf), N, K, kpad);
  if (hipGetLastError() != hipSuccess)
    return nullptr;
  *out_kpad = kpad;
  return static_cast<const T *>(buf);
}

/* Eligible: NT fp16/bf16, K a multiple of 16 but not 32 (vision 4304), M and N
 * tile-aligned enough that the aligned BK=32 path is a clear win. Opt-in with
 * padrow (HIPDNN_EP_GEMM_PADROW=1): the K-extended copy is paid every call. */
static bool gemmPadKEligible(int M, int N, int K, int transA, int transB,
                             int type_bytes) {
  return gemmPadRowEnabled() && transB && !transA && type_bytes == 2 &&
         (K % 16 == 0) && (K % 32 != 0) && K >= 64 && M >= 256 && N >= 128;
}

/* =========================================================================
 * Public entry
 * ========================================================================= */

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

  auto go_f16 = [&]() -> int {
    using T = _Float16;
    auto *a = static_cast<const T *>(A);
    auto *b = static_cast<const T *>(B);
    auto *c = static_cast<const T *>(C);
    auto *y = static_cast<T *>(Y);
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
    int ldb = K, kk = K, lda = -1;
    if (gemmPadKEligible(M, N, K, transA, transB, sizeof(T))) {
      int kpad = K;
      if (const T *bp = gemmEnsureKPaddedB<T>(s, b, N, K, &kpad)) {
        b = bp;
        kk = kpad; /* loop + B run the BK-aligned padded K (fast BK=32 path) */
        ldb = kpad;
        lda = K; /* A keeps its real stride; the tail is nulled by B's zeros */
      }
    } else if (gemmPadRowEligible(M, N, K, transA, transB, sizeof(T))) {
      if (const T *bp = gemmEnsurePaddedB<T>(s, b, N, K, &ldb))
        b = bp;
    }
    return launchWmmaGemm<T>(s, a, b, c, y, M, N, kk, transB, alpha, beta,
                             cDim0, cDim1, ldb, lda);
  };

  auto go_bf16 = [&]() -> int {
    using T = __hip_bfloat16;
    auto *a = static_cast<const T *>(A);
    auto *b = static_cast<const T *>(B);
    auto *c = static_cast<const T *>(C);
    auto *y = static_cast<T *>(Y);
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
    int ldb = K, kk = K, lda = -1;
    if (gemmPadKEligible(M, N, K, transA, transB, sizeof(T))) {
      int kpad = K;
      if (const T *bp = gemmEnsureKPaddedB<T>(s, b, N, K, &kpad)) {
        b = bp;
        kk = kpad; /* loop + B run the BK-aligned padded K (fast BK=32 path) */
        ldb = kpad;
        lda = K; /* A keeps its real stride; the tail is nulled by B's zeros */
      }
    } else if (gemmPadRowEligible(M, N, K, transA, transB, sizeof(T))) {
      if (const T *bp = gemmEnsurePaddedB<T>(s, b, N, K, &ldb))
        b = bp;
    }
    return launchWmmaGemm<T>(s, a, b, c, y, M, N, kk, transB, alpha, beta,
                             cDim0, cDim1, ldb, lda);
  };

  if (type_code == kTypeFloat16)
    return go_f16();
  if (type_code == kTypeBFloat16)
    return go_bf16();
  if (type_code == kTypeFloat32) {
    auto *a = static_cast<const float *>(A);
    auto *b = static_cast<const float *>(B);
    auto *c = static_cast<const float *>(C);
    auto *y = static_cast<float *>(Y);
    if (M < kWmmaM)
      return launchSmallM<float>(s, a, b, c, y, M, N, K, transA, transB,
                                 alpha, beta, cDim0, cDim1);
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
