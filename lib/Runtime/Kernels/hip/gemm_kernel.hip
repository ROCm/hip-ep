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

template <typename T>
static void launchGemvNn(hipStream_t s, const T *A, const T *B, const T *C,
                          T *Y, int M, int N, int K, float alpha, float beta,
                          int cDim0, int cDim1) {
  constexpr int BLOCK = 256;
  dim3 grid((N + BLOCK - 1) / BLOCK, M);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_gemv_nn_kernel<T, BLOCK>), grid,
                     dim3(BLOCK), 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0,
                     cDim1);
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

/* ---- Offline LUT bridge (scaffold) ------------------------------------
 *
 * The autotune winner for a shape is durable knowledge that belongs in an
 * offline table, exactly like matmul_nbits' hipdnn_ep::matmul_nbits_autotune.
 * The table stores tile *geometry* (not a config index), and a validator maps
 * the geometry back onto the live config table at load: a geometry the table
 * no longer contains is skipped so a stale LUT is never worse than no LUT.
 *
 * The lookup is wired into the dispatch path now (map -> LUT -> sweep) so the
 * table can be dropped in later without touching call sites. Until a table is
 * generated and embedded (see lib/Runtime/Kernels/hip/autotune/), resolve()
 * reports a miss and the sweep runs, i.e. "lookup" behaves like "online". */
namespace gemm_lut {

enum class Kind : uint8_t { Wmma = 1, GemvNt = 2 };

struct Request {
  Kind kind = Kind::Wmma;
  int type_bytes = 2; /* sizeof(T): 2=f16/bf16, 4=f32, 8=f64 */
  int trans_b = 1;
  int m = 0, n = 0, k = 0;
};

/* WMMA tile geometry, matching WmmaCfg fields the dispatch keys on. */
struct WmmaAnswer {
  int bm = 0, bn = 0, wt_m = 0, wt_n = 0, swizzle_n = 0, split_k = 1, bk = 32;
};

/* (threads, tile_n) for the GEMV NT decode path. */
struct GemvAnswer {
  int threads = 0, tile_n = 0;
};

enum class Source : uint8_t { None = 0, Exact = 1, Nearest = 2 };

struct Result {
  Source source = Source::None;
  WmmaAnswer wmma;
  GemvAnswer gemv;
  float distance = 0.0f;
};

using WmmaValidator = bool (*)(void *ctx, const WmmaAnswer &);
using GemvValidator = bool (*)(void *ctx, const GemvAnswer &);

/* Stub: no table embedded yet. Returns a miss so the caller sweeps. When the
 * offline table lands, this forwards to the embedded resolver and the shape of
 * this call does not change. */
static Result resolve(const Request &, WmmaValidator, GemvValidator, void *) {
  return Result{};
}

} // namespace gemm_lut

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
  req.type_bytes = sizeof(T);
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  const gemm_lut::Result r =
      gemm_lut::resolve(req, nullptr, gemvNtLutValidator, nullptr);
  if (r.source == gemm_lut::Source::None)
    return -1;
  return gemvNtConfigFromGeom(r.gemv);
}

template <typename T>
static int tuneGemvNt(hipStream_t s, const T *A, const T *B, const T *C, T *Y,
                       int M, int N, int K, float alpha, float beta, int cDim0,
                       int cDim1) {
  std::string key = std::string("nt_") + std::to_string(sizeof(T)) + "_" +
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
    CUSTOM_KERNELS_DEBUG_LOG("[gemm] gemv-nt LUT hit M=%d N=%d K=%d -> cfg[%d]\n",
                             M, N, K, best);
  } else if (gemmLutLookupEnabled()) {
    /* Default (lookup) mode, LUT miss: keep the static default (best=6). The
     * on-device sweep runs only in `online` mode, for LUT generation. */
  } else if (gemmAutotuneOn()) {
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    T *scratch = static_cast<T *>(
        ensureTuneOut(static_cast<size_t>(M) * N * sizeof(T)));
    T *out = scratch ? scratch : Y;
    float best_ms = 1e30f;
    for (int cid = 0; cid < kNumGemvNt; ++cid) {
      launchGemvNt<T>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                      cDim1);
      hipEventRecord(e0, s);
      for (int i = 0; i < 3; ++i)
        launchGemvNt<T>(cid, s, A, B, C, out, M, N, K, alpha, beta, cDim0,
                        cDim1);
      hipEventRecord(e1, s);
      hipEventSynchronize(e1);
      float ms = 0.f;
      hipEventElapsedTime(&ms, e0, e1);
      CUSTOM_KERNELS_DEBUG_LOG(
          "[gemm] gemv-nt cfg[%d] th=%d tn=%d : %.4f ms\n", cid,
          kGemvNt[cid].threads, kGemvNt[cid].tile_n, ms / 3.f);
      if (ms < best_ms) {
        best_ms = ms;
        best = cid;
      }
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] gemv-nt autotune M=%d N=%d K=%d -> cfg[%d]\n", M, N, K, best);
  }

  std::lock_guard<std::mutex> lk(g_tune_mu);
  g_gemv_nt_cache[key] = best;
  return best;
}

template <typename T>
static int launchSmallM(hipStream_t s, const T *A, const T *B, const T *C,
                         T *Y, int M, int N, int K, int transA, int transB,
                         float alpha, float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
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
    launchGemvNn<T>(s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1);
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
          bool BC = true, bool DB = true>
__device__ __forceinline__ void GemmWmmaImpl(const T *__restrict__ A,
                                              const T *__restrict__ B,
                                              const T *__restrict__ C,
                                              T *__restrict__ Y, int M, int N,
                                              int K, float alpha, float beta,
                                              int cDim0, int cDim1,
                                              int swizzle_n,
                                              float *__restrict__ sk_acc,
                                              int ldb, int n_base, int lda) {
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

  int a_row[A_NCH];
  int a_koff[A_NCH];
  const T *a_src[A_NCH];
#pragma unroll
  for (int c = 0; c < A_NCH; ++c) {
    const int idx = tid + c * THR;
    a_row[c] = idx / A_ROW_SPLIT;
    a_koff[c] = (idx % A_ROW_SPLIT) * A_CH;
    a_src[c] = A + static_cast<size_t>(row0 + a_row[c]) * lda + k_begin +
               a_koff[c];
  }

  /* ldb is the B row stride in elements; == K unless the caller passed a
   * row-padded weight (padrow, power-of-two-K L1 conflict fix). Only the row
   * base scales by ldb -- the within-row k range is still [0, K). */
  const T *b_src_nt = B + static_cast<size_t>(b_n_g) * ldb + k_begin +
                      b_k8; /* TransB=1 [N,K] */

  float8 acc[WT_M][WT_N];
#pragma unroll
  for (int i = 0; i < WT_M; ++i)
#pragma unroll
    for (int j = 0; j < WT_N; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        acc[i][j][e] = 0.0f;

  uint32_t pf_a[A_NCH][A_CH / 2];
  u32x4 pf_b[8];
  alignas(16) T pf_nn[N_CH];

  auto issueLoads = [&](int k0) __attribute__((always_inline)) {
#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      if constexpr (!BC) {
        /* Tile fully covers M and K%BK==0: no row/K bounds check. */
        if constexpr (A_CH == 8)
          *reinterpret_cast<u32x4 *>(pf_a[c]) =
              *reinterpret_cast<const u32x4 *>(a_src[c]);
        else
          *reinterpret_cast<u32x2 *>(pf_a[c]) =
              *reinterpret_cast<const u32x2 *>(a_src[c]);
      } else {
        const int row = row0 + a_row[c];
        const int k = k0 + a_koff[c];
        const bool ok = (row < M) && (k + A_CH <= K);
        if (ok) {
          if constexpr (A_CH == 8)
            *reinterpret_cast<u32x4 *>(pf_a[c]) =
                *reinterpret_cast<const u32x4 *>(a_src[c]);
          else
            *reinterpret_cast<u32x2 *>(pf_a[c]) =
                *reinterpret_cast<const u32x2 *>(a_src[c]);
        } else {
#pragma unroll
          for (int e = 0; e < A_CH / 2; ++e)
            pf_a[c][e] = 0;
          if (row < M) {
#pragma unroll
            for (int e = 0; e < A_CH; ++e) {
              const int kk = k + e;
              const T v =
                  (kk < K) ? A[static_cast<size_t>(row) * lda + kk] : (T)0;
              reinterpret_cast<T *>(pf_a[c])[e] = v;
            }
          }
        }
      }
      a_src[c] += BK;
    }

    if constexpr (TransB) {
      if (!BC || (b_valid && k0 + b_k8 + B_HPT <= K)) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(pf_b) =
              *reinterpret_cast<const u32x2 *>(b_src_nt);
        } else if constexpr (B_U32 == 1) {
          pf_b[0] = *reinterpret_cast<const u32x4 *>(b_src_nt);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            pf_b[w] = reinterpret_cast<const u32x4 *>(b_src_nt)[w];
        }
      } else {
#pragma unroll
        for (int w = 0; w < 8; ++w)
          pf_b[w] = u32x4{0, 0, 0, 0};
        if (b_valid) {
#pragma unroll
          for (int e = 0; e < B_HPT; ++e) {
            const int kk = k0 + b_k8 + e;
            const T v = (kk < K) ? B[static_cast<size_t>(b_n_g) * ldb + kk]
                                  : (T)0;
            reinterpret_cast<T *>(pf_b)[e] = v;
          }
        }
      }
      b_src_nt += BK;
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

  auto storeToShared = [&](int buf) __attribute__((always_inline)) {
    if constexpr (TransB) {
      if (!BC || b_valid) {
        if constexpr (B_HPT == 4) {
          *reinterpret_cast<u32x2 *>(&smB[buf][b_col][b_k8]) =
              *reinterpret_cast<const u32x2 *>(pf_b);
        } else {
#pragma unroll
          for (int w = 0; w < B_U32; ++w)
            reinterpret_cast<u32x4 *>(&smB[buf][b_col][b_k8])[w] = pf_b[w];
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

#pragma unroll
    for (int c = 0; c < A_NCH; ++c) {
      T *dst = &smA[buf][a_row[c]][a_koff[c]];
      if constexpr (A_CH == 8)
        *reinterpret_cast<u32x4 *>(dst) =
            *reinterpret_cast<const u32x4 *>(pf_a[c]);
      else
        *reinterpret_cast<u32x2 *>(dst) =
            *reinterpret_cast<const u32x2 *>(pf_a[c]);
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
    issueLoads(k_begin);
    storeToShared(0);
    HIPDNN_LDS_BARRIER();

    if constexpr (DB) {
      int buf = 0;
      for (int t = k_begin + BK; t < k_end; t += BK) {
        const int nxt = 1 - buf;
        issueLoads(t);
        computeWMMA(buf);
        storeToShared(nxt);
        HIPDNN_LDS_BARRIER();
        buf = nxt;
      }
      computeWMMA(buf);
    } else {
      /* Single-buffer: the next tile's global loads (issueLoads) still overlap
       * compute via registers, but the LDS is reused, so a barrier must bracket
       * the overwrite -- one after compute finishes reading, one after the
       * store makes the new tile visible. */
      for (int t = k_begin + BK; t < k_end; t += BK) {
        issueLoads(t);
        computeWMMA(0);
        HIPDNN_LDS_BARRIER();
        storeToShared(0);
        HIPDNN_LDS_BARRIER();
      }
      computeWMMA(0);
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
          bool BC = true, bool DB = true>
__global__ __attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM, BN, WT_M, WT_N), GEMM_WG_SZ(BM, BN, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(2, 8)))
void gemm_wmma_kernel(const T *__restrict__ A, const T *__restrict__ B,
                      const T *__restrict__ C, T *__restrict__ Y, int M, int N,
                      int K, float alpha, float beta, int cDim0, int cDim1,
                      int swizzle_n, float *__restrict__ sk_acc, int ldb,
                      int n_base, int lda) {
  GemmWmmaImpl<BM, BN, WT_M, WT_N, TransB, T, BK, BC, DB>(
      A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1, swizzle_n, sk_acc, ldb,
      n_base, lda);
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
};
static constexpr int kNumWmma =
    static_cast<int>(sizeof(kWmma) / sizeof(kWmma[0]));

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
  req.type_bytes = sizeof(T);
  req.trans_b = transB;
  req.m = M;
  req.n = N;
  req.k = K;
  GemmWmmaLutCtx ctx{M, N, K};
  const gemm_lut::Result r =
      gemm_lut::resolve(req, wmmaLutValidator, nullptr, &ctx);
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
   *     (parity), M=1024 -> 4.07 ms (was 4.46). padrow still fixes the L1
   *     aliasing on this tile (bad_stride forces it for pow2 K).
   *   - Small M (257..512): too few row-tiles to amortise the wide tile; the
   *     128x64 BK=64 tile (half the aliasing tile-rows on both operands, half
   *     the K-step/barrier count) is the workhorse and beats the wide tile
   *     (M=512 8192^2: 2.17 vs 2.29 ms). Small M<=256 uses padrow below.
   *
   * Non-power-of-two large K (14336/17408) does not alias, so the big square
   * tiles in the block below still win. */
  if (K >= 8192 && (K & (K - 1)) == 0) {
    if (M >= 1024)
      return findWmmaCfg(128, 256, 4, 2, 8, 1, 32);
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
      static_cast<size_t>(M) * K * 2 >= (16ull << 20))
    return findWmmaCfg(128, 256, 4, 2, 8, 1, 32);
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

template <typename T, bool TransB>
static void launchWmma(int id, hipStream_t s, const T *A, const T *B,
                       const T *C, T *Y, int M, int N, int K, float alpha,
                       float beta, int cDim0, int cDim1, int ldb,
                       int lda = -1) {
  const WmmaCfg c = kWmma[id];
  /* A row stride: defaults to K. pad-K mode passes lda=K_real with K=K_pad so
   * the loop/B use the BK-aligned padded K while A keeps its real stride. */
  const int lda_eff = (lda < 0) ? K : lda;
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
  {
    const int sw_env = gemmEnvInt("HIPDNN_EP_GEMM_SWIZZLE", 0);
    if (sw_env > 0)
      sw_eff = sw_env;
  }
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
  do {                                                                         \
    if (bc && db)                                                              \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, true, true>),    \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff);                                               \
    else if (bc && !db)                                                        \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, true, false>),   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff);                                               \
    else if (!bc && db)                                                        \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, false, true>),   \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff);                                               \
    else                                                                       \
      hipLaunchKernelGGL(                                                      \
          HIP_KERNEL_NAME(                                                     \
              gemm_wmma_kernel<BM, BN, WM, WN, TransB, T, BK, false, false>),  \
          grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta, cDim0, cDim1,   \
          sw_eff, skp, ldb, nb, lda_eff);                                               \
  } while (0)

#define GEMM_WMMA_TRY(BM, BN, WM, WN, BK)                                      \
  if (c.bm == BM && c.bn == BN && c.wt_m == WM && c.wt_n == WN &&              \
      c.bk == BK) {                                                            \
    GEMM_WMMA_ONE(BM, BN, WM, WN, BK);                                         \
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
  GEMM_WMMA_TRY(128, 128, 2, 4, 32)
  GEMM_WMMA_TRY(64, 128, 2, 4, 32)
  GEMM_WMMA_TRY(128, 256, 2, 4, 32)
  /* BK=32 WT_M=4 (high Matrix-Core:LDS ratio) */
  GEMM_WMMA_TRY(256, 128, 4, 2, 32)
  GEMM_WMMA_TRY(128, 128, 4, 4, 32)
  GEMM_WMMA_TRY(128, 128, 4, 2, 32)
  GEMM_WMMA_TRY(128, 256, 4, 2, 32)
  GEMM_WMMA_TRY(256, 128, 4, 4, 32)
  GEMM_WMMA_TRY(128, 128, 4, 4, 16)
  /* BK=16 (K%16==0, K%32!=0) */
  GEMM_WMMA_TRY(256, 128, 4, 2, 16)
  GEMM_WMMA_TRY(128, 128, 2, 4, 16)
  GEMM_WMMA_TRY(64, 128, 2, 4, 16)
  {
    GEMM_WMMA_ONE(128, 128, 2, 2, 32);
  }
  } /* sub-launch loop */
#undef GEMM_WMMA_TRY
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
  std::string key = std::string(TransB ? "tb1_" : "tb0_") +
                    std::to_string(sizeof(T)) + "_" + std::to_string(M) + "_" +
                    std::to_string(N) + "_" + std::to_string(K);
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
        "[gemm] wmma LUT hit M=%d N=%d K=%d tb=%d -> cfg[%d]\n", M, N, K,
        (int)TransB, best);
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
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] wmma heuristic M=%d N=%d K=%d tb=%d -> cfg[%d]\n", M, N, K,
        (int)TransB, best);
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
        "[gemm] wmma autotune M=%d N=%d K=%d tb=%d -> cfg[%d]\n", M, N, K,
        (int)TransB, best);
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

/* =========================================================================
 * fp32 / fp64 tiled FMA (no WMMA). 64x64x16, 256 threads, 4x4 per thread.
 * Templates cover all four (transA, transB) combinations.
 * ========================================================================= */

template <typename T, bool TransA, bool TransB>
__global__ void __launch_bounds__(256)
    gemm_tiled_fma_kernel(const T *__restrict__ A, const T *__restrict__ B,
                         const T *__restrict__ C, T *__restrict__ Y, int M,
                         int N, int K, float alpha, float beta, int cDim0,
                         int cDim1) {
  constexpr int BM = 64, BN = 64, BK = 16;
  constexpr int TM = 4, TN = 4;
  __shared__ float smA[BM][BK + 1];
  __shared__ float smB[BK][BN + 1];

  const int tx = threadIdx.x % 16;
  const int ty = threadIdx.x / 16;
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
    /* 256 threads fill 64*16 = 1024 A elems (4 each) and 16*64 B. */
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * 256;
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
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * 256;
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

template <typename T>
static int launchTiledFma(hipStream_t s, const T *A, const T *B, const T *C,
                          T *Y, int M, int N, int K, int transA, int transB,
                          float alpha, float beta, int cDim0, int cDim1) {
  (void)hipGetLastError();
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  dim3 block(256);
  if (!transA && !transB)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_tiled_fma_kernel<T, false, false>),
                       grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
                       cDim0, cDim1);
  else if (!transA && transB)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_tiled_fma_kernel<T, false, true>),
                       grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
                       cDim0, cDim1);
  else if (transA && !transB)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_tiled_fma_kernel<T, true, false>),
                       grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
                       cDim0, cDim1);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_tiled_fma_kernel<T, true, true>),
                       grid, block, 0, s, A, B, C, Y, M, N, K, alpha, beta,
                       cDim0, cDim1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
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
 * spreads them over every set. For fp16 the repack copy is too expensive to
 * pay per call (unlike int4), so the padded copy of the *constant weight* is
 * cached and reused across inferences; a bounded LRU caps the memory and a
 * hit-rate guard disables padrow if B turns out to be a dynamic activation.
 * ========================================================================= */

/* Pad the row stride by one 128 B cache line, in elements. */
static constexpr int kGemmPadRowBytes = 128;

static bool gemmIsPow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

static bool gemmPadRowEnabled() {
  static const bool on = [] {
    std::string s = gemmEnvStr("HIPDNN_EP_GEMM_PADROW");
    return s.empty() || s[0] != '0';
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
 *     the non-pow2 dips (6144, 3072). Padrow never regresses (worst +1%), and the
 *     hit-rate guard disables it for a dynamic (non-reused) B, so a bad-K weight
 *     is always eligible regardless of size.
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

struct GemmPadKey {
  const void *ptr;
  int N, K, tb;
  bool operator==(const GemmPadKey &o) const {
    return ptr == o.ptr && N == o.N && K == o.K && tb == o.tb;
  }
};
struct GemmPadKeyHash {
  size_t operator()(const GemmPadKey &k) const {
    size_t h = std::hash<const void *>{}(k.ptr);
    h ^= std::hash<int>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.tb) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};
struct GemmPadEntry {
  void *buf;
  size_t bytes;
  uint64_t lru;
};

static std::mutex g_pad_mu;
static std::unordered_map<GemmPadKey, GemmPadEntry, GemmPadKeyHash> g_pad_map;
static size_t g_pad_total = 0;
static uint64_t g_pad_clock = 0;
static uint64_t g_pad_hits = 0, g_pad_misses = 0;
static bool g_pad_disabled = false;

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

  GemmPadKey key{static_cast<const void *>(B), N, K, (int)sizeof(T)};
  std::lock_guard<std::mutex> lk(g_pad_mu);
  if (g_pad_disabled)
    return nullptr;

  auto it = g_pad_map.find(key);
  if (it != g_pad_map.end()) {
    it->second.lru = ++g_pad_clock;
    ++g_pad_hits;
    *out_ldb = ldb;
    return static_cast<const T *>(it->second.buf);
  }

  /* Miss: hit-rate guard. If, after warming up, the caller keeps handing us
   * new B pointers (B is a dynamic activation, not a reused weight), the copy
   * never amortizes -- disable padrow process-wide and fall back. */
  ++g_pad_misses;
  const uint64_t total = g_pad_hits + g_pad_misses;
  if (total >= 32 && g_pad_hits * 100 < total * 40) { /* <40% hit rate */
    g_pad_disabled = true;
    CUSTOM_KERNELS_DEBUG_LOG(
        "[gemm] padrow disabled: hit-rate %llu/%llu too low (dynamic B)\n",
        (unsigned long long)g_pad_hits, (unsigned long long)total);
    return nullptr;
  }

  /* Evict LRU entries until the new buffer fits under the cap. */
  while (g_pad_total + bytes > cap && !g_pad_map.empty()) {
    auto victim = g_pad_map.begin();
    for (auto i = g_pad_map.begin(); i != g_pad_map.end(); ++i)
      if (i->second.lru < victim->second.lru)
        victim = i;
    hipFree(victim->second.buf);
    g_pad_total -= victim->second.bytes;
    g_pad_map.erase(victim);
  }

  void *buf = nullptr;
  if (hipMalloc(&buf, bytes) != hipSuccess)
    return nullptr;
  dim3 grid((K + 255) / 256 < 1 ? 1 : (K + 255) / 256, N);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_padrow_copy_kernel<T>), grid,
                     dim3(256), 0, s, B, static_cast<T *>(buf), N, K, ldb);
  if (hipGetLastError() != hipSuccess) {
    hipFree(buf);
    return nullptr;
  }
  g_pad_map[key] = GemmPadEntry{buf, bytes, ++g_pad_clock};
  g_pad_total += bytes;
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] padrow: cached B=%p N=%d K=%d -> ldb=%d (%zu MB total)\n",
      (const void *)B, N, K, ldb, g_pad_total >> 20);
  *out_ldb = ldb;
  return static_cast<const T *>(buf);
}

/* K-extend a reused NT weight B[N,K] to B'[N,K_pad] with K_pad a multiple of 32
 * and the tail [K,K_pad) zero-filled, so a K%32!=0 shape (vision K=4304) can run
 * the fast BK=32 BC=false path instead of the 2x-barrier BK=16 tile. A keeps its
 * real stride (K): the A-load reads garbage in the [K,K_pad) tail, but B's zeros
 * null it (sum_k A*0 = 0), so no A copy is needed. Cached like padrow (weight
 * reuse); returns nullptr to fall back. Sets *out_kpad. The A over-read is <=
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

  GemmPadKey key{static_cast<const void *>(B), N, K, -1 /* K-extend mode */};
  std::lock_guard<std::mutex> lk(g_pad_mu);
  if (g_pad_disabled)
    return nullptr;
  auto it = g_pad_map.find(key);
  if (it != g_pad_map.end()) {
    it->second.lru = ++g_pad_clock;
    ++g_pad_hits;
    *out_kpad = kpad;
    return static_cast<const T *>(it->second.buf);
  }
  ++g_pad_misses;
  const uint64_t total = g_pad_hits + g_pad_misses;
  if (total >= 32 && g_pad_hits * 100 < total * 40) {
    g_pad_disabled = true;
    return nullptr;
  }
  while (g_pad_total + bytes > cap && !g_pad_map.empty()) {
    auto victim = g_pad_map.begin();
    for (auto i = g_pad_map.begin(); i != g_pad_map.end(); ++i)
      if (i->second.lru < victim->second.lru)
        victim = i;
    hipFree(victim->second.buf);
    g_pad_total -= victim->second.bytes;
    g_pad_map.erase(victim);
  }
  void *buf = nullptr;
  if (hipMalloc(&buf, bytes) != hipSuccess)
    return nullptr;
  if (hipMemsetAsync(buf, 0, bytes, s) != hipSuccess) {
    hipFree(buf);
    return nullptr;
  }
  dim3 grid((K + 255) / 256 < 1 ? 1 : (K + 255) / 256, N);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_padrow_copy_kernel<T>), grid,
                     dim3(256), 0, s, B, static_cast<T *>(buf), N, K, kpad);
  if (hipGetLastError() != hipSuccess) {
    hipFree(buf);
    return nullptr;
  }
  g_pad_map[key] = GemmPadEntry{buf, bytes, ++g_pad_clock};
  g_pad_total += bytes;
  CUSTOM_KERNELS_DEBUG_LOG(
      "[gemm] padK: cached B=%p N=%d K=%d -> kpad=%d (%zu MB total)\n",
      (const void *)B, N, K, kpad, g_pad_total >> 20);
  *out_kpad = kpad;
  return static_cast<const T *>(buf);
}

/* Eligible: NT fp16/bf16, K a multiple of 16 but not 32 (vision 4304), M and N
 * tile-aligned enough that the aligned BK=32 path is a clear win, and B a reused
 * weight (guarded by the shared padrow hit-rate check). */
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
    if (M < kWmmaM || transA)
      return launchSmallM<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                             beta, cDim0, cDim1);
    if (wave64)
      return launchTiledFma<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                                beta, cDim0, cDim1);
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
    if (M < kWmmaM || transA)
      return launchSmallM<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                             beta, cDim0, cDim1);
    if (wave64)
      return launchTiledFma<T>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                                beta, cDim0, cDim1);
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
    return launchTiledFma<float>(s, a, b, c, y, M, N, K, transA, transB, alpha,
                                  beta, cDim0, cDim1);
  }
  if (type_code == kTypeFloat64) {
    auto *a = static_cast<const double *>(A);
    auto *b = static_cast<const double *>(B);
    auto *c = static_cast<const double *>(C);
    auto *y = static_cast<double *>(Y);
    if (M < kWmmaM)
      return launchSmallM<double>(s, a, b, c, y, M, N, K, transA, transB,
                                  alpha, beta, cDim0, cDim1);
    return launchTiledFma<double>(s, a, b, c, y, M, N, K, transA, transB,
                                   alpha, beta, cDim0, cDim1);
  }
  return -1;
}
