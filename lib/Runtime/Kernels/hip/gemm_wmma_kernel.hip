/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"

// half16 holds one WMMA A/B fragment; its vector width matches
// HIPDNN_WMMA_FRAG_ELEMS (16 on gfx11, 8 on the gfx12-style encoding
// gfx1170/gfx12xx use -- see hip_arch_compat.h) so it binds directly to
// whichever HIPDNN_WMMA_F32_16X16X16_F16 builtin is selected. The name is
// kept for continuity with the existing RDNA3 comments/callers even though
// the width is arch-dependent.
typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Where neither WMMA builtin is available (non-RDNA3/3.5/4 arch), trap
// instead so this file still compiles and a WMMA launch on an unsupported
// arch aborts loudly rather than returning garbage.
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

static constexpr int kWmmaTile = 16;  // WMMA tile dimension (16x16x16)

//===----------------------------------------------------------------------===//
// WMMA GEMM kernel for small-M fp16 MatMuls (M <= 512)
//===----------------------------------------------------------------------===//
//
// Computes C[M,N] = A[M,K] * B[K,N]  (all row-major, fp16, fp32 accumulation)
//
// Uses HIPDNN_WMMA_F32_16X16X16_F16 (hip_arch_compat.h) for hardware-
// accelerated 16x16x16 tile multiply-accumulate on RDNA 3/3.5/4.
//
// Register-tiled: each wavefront computes a (KX*16) x (KY*16) output tile.
// KX A fragments are each reused KY times; KY B fragments each reused KX times.
// Grid: (ceil(N/(KY*16)), ceil(M/(KX*16)))
//
// WMMA fragment layout (wave32; two encodings, selected at compile time by
// HIPDNN_WMMA_GFX12 -- see hip_arch_compat.h for how each was verified):
//   gfx11 (HIPDNN_WMMA_FRAG_ELEMS==16): a_frag[k] at lane m = A_tile[m, k];
//     b_frag[k] at lane n = B_tile[k, n]. Lanes 0-15 and 16-31 hold identical
//     data (required by this encoding), satisfied by indexing with
//     threadIdx.x % 16 == m (or n).
//   gfx12-style (HIPDNN_WMMA_FRAG_ELEMS==8, e.g. gfx1170/gfx12xx): no
//     replication; a_frag[j] at lane L = A_tile[L%16, hipdnn_wmma_k_off(pair)+j],
//     b_frag[j] at lane L = B_tile[hipdnn_wmma_k_off(pair)+j, L%16], where
//     pair = L/16 selects which half of K this lane contributes.
//   c_frag (float8, both encodings): c_frag[ele] -> row hipdnn_wmma_acc_row(pair, ele),
//     col (lane%16). lane = threadIdx.x % 16, pair = threadIdx.x / 16.

template<int KX, int KY>
__global__ void __launch_bounds__(32)
gemm_wmma_fp16_kernel(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M, int K, int N)
{
    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int lane = threadIdx.x % kWmmaTile;
    const int pair = threadIdx.x / kWmmaTile;

    const int m_base = m_tile * KX * kWmmaTile;
    const int n_base = n_tile * KY * kWmmaTile;

    half16 a_frag[KX];
    half16 b_frag[KY];
    float8 c_frag[KY][KX] = {};

    // K-axis base this lane's pair-half contributes: 0 on gfx11 (each lane
    // loads the full 16-wide K range), pair*8 on the gfx12-style encoding
    // (each lane loads only its half, no replication).
    const int k_off = hipdnn_wmma_k_off(pair);

    for (int k = 0; k < K; k += kWmmaTile) {
        for (int ele = 0; ele < HIPDNN_WMMA_FRAG_ELEMS; ++ele) {
            for (int x = 0; x < KX; ++x) {
                int row = m_base + x * kWmmaTile + lane;
                a_frag[x][ele] = (row < M) ? A[row * K + k + k_off + ele]
                                            : (_Float16)0;
            }
            for (int y = 0; y < KY; ++y) {
                int col = n_base + y * kWmmaTile + lane;
                b_frag[y][ele] = (col < N) ? B[(k + k_off + ele) * N + col]
                                            : (_Float16)0;
            }
        }

        for (int y = 0; y < KY; ++y)
            for (int x = 0; x < KX; ++x)
                c_frag[y][x] = HIPDNN_WMMA_F32_16X16X16_F16(
                                    a_frag[x], b_frag[y], c_frag[y][x]);
    }

    for (int y = 0; y < KY; ++y) {
        for (int x = 0; x < KX; ++x) {
            for (int ele = 0; ele < 8; ++ele) {
                int r = m_base + x * kWmmaTile + hipdnn_wmma_acc_row(pair, ele);
                int c = n_base + y * kWmmaTile + lane;
                if (r < M && c < N)
                    C[r * N + c] = (_Float16)c_frag[y][x][ele];
            }
        }
    }
}

extern "C"
int hip_gemm_wmma_fp16(void* stream, const void* A, const void* B,
                       void* C, int M, int K, int N) {
    // WMMA is RDNA-only (wave32); CDNA (wave64, e.g. MI300/MI350) has no WMMA
    // and would otherwise hit the __builtin_trap() fallback above at launch.
    if (hipdnn_device_is_wave64())
        return -1;
    constexpr int KX = 2;
    constexpr int KY = 4;
    dim3 block(32);
    dim3 grid((N + KY * kWmmaTile - 1) / (KY * kWmmaTile),
              (M + KX * kWmmaTile - 1) / (KX * kWmmaTile));
    gemm_wmma_fp16_kernel<KX, KY><<<grid, block, 0, (hipStream_t)stream>>>(
        (const _Float16*)A, (const _Float16*)B, (_Float16*)C, M, K, N);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

//===--------------------------------------------------------------------===//
// Narrow-N fp16 decode GEMV (M == 1)
//===--------------------------------------------------------------------===//
// Built for the MoE router: hidden x num_experts, so K is large (2048) and N is
// tiny (128), at M == 1. That shape is a bad fit for a tiled GEMM library --
// there is no M to tile and only four 32-wide columns of N -- and it shows: the
// Tensile kernel picked for it retiles between context lengths and the same
// 512 KB of work measures anywhere from 0.43 to 0.87 ms per decode step.
//
// The whole operation is one pass over B. At 512 KB and a ~200 GB/s roofline
// its floor is a couple of microseconds, so everything above that is scheduling.
//
// Shape of the launch. Coalescing wants 32 consecutive n per wavefront, which
// caps n-parallelism at N/32 = 4 blocks -- four CUs on a forty-CU part. So K is
// split across blocks as well, grid = (ceil(N/32), splits), and the partials are
// reduced without a second dispatch: every block writes its own slice of
// `partials`, then the last block to arrive (a counter incremented after a
// __threadfence) sums the splits and writes fp16. That keeps this a ONE-dispatch
// replacement for a GEMM that is often two (a tiled kernel plus a GSU reduce).
//
// Partials are indexed per split rather than atomically accumulated, so the
// buffer needs no zeroing between calls. Only the counter carries state, and the
// last block resets it to 0 on its way out, leaving the buffer reusable by the
// next layer with no host-side memset.

static constexpr int kGemvWave = 32;       // n per wavefront: one 64 B cache line
static constexpr int kGemvBlock = 256;     // 8 wavefronts
static constexpr int kGemvMaxN = 256;      // reduction pass is one thread per n

__global__ __launch_bounds__(kGemvBlock) void gemv_fp16_narrow_n_kernel(
    const _Float16* __restrict__ A,   // [K]
    const _Float16* __restrict__ B,   // [K, N] row-major
    _Float16* __restrict__ C,         // [N]
    float* __restrict__ partials,     // [splits, N]
    unsigned int* __restrict__ counter,
    int N, int K, int splits, int k_chunk) {
  const int n = blockIdx.x * kGemvWave + (threadIdx.x % kGemvWave);
  const int wave = threadIdx.x / kGemvWave;
  const int waves = kGemvBlock / kGemvWave;
  const int ks = blockIdx.y;

  int k_begin = ks * k_chunk;
  int k_end = k_begin + k_chunk;
  if (k_end > K) k_end = K;

  // Each wavefront walks its own stride through the split's K range, so the
  // wave's 32 lanes always read one contiguous run of B.
  float acc = 0.0f;
  if (n < N) {
    for (int k = k_begin + wave; k < k_end; k += waves)
      acc += (float)A[k] * (float)B[(long long)k * N + n];
  }

  __shared__ float s_acc[kGemvBlock];
  s_acc[threadIdx.x] = acc;
  __syncthreads();
  // Fold the wavefronts onto wave 0. Lanes stay aligned to n throughout, so
  // this is a plain strided add and never a cross-lane shuffle.
  for (int w = waves / 2; w > 0; w >>= 1) {
    if (wave < w)
      s_acc[threadIdx.x] += s_acc[threadIdx.x + w * kGemvWave];
    __syncthreads();
  }
  if (wave == 0 && n < N)
    partials[(long long)ks * N + n] = s_acc[threadIdx.x];

  if (splits == 1) {
    // Nothing to reduce across blocks; skip the counter handshake entirely.
    if (wave == 0 && n < N) C[n] = (_Float16)s_acc[threadIdx.x];
    return;
  }

  // Make this block's partials visible before announcing it is done.
  __threadfence();
  __shared__ bool s_last;
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicAdd(counter, 1u);
    s_last = (ticket == (unsigned int)(gridDim.x * gridDim.y - 1));
  }
  __syncthreads();
  if (!s_last) return;

  // One thread per n, so N is capped at the block size (kGemvMaxN); the host
  // gate enforces it.
  const int nn = threadIdx.x;
  if (nn < N) {
    float sum = 0.0f;
    for (int s = 0; s < splits; ++s)
      sum += partials[(long long)s * N + nn];
    C[nn] = (_Float16)sum;
  }
  // Leave the counter as the next call expects to find it.
  if (threadIdx.x == 0) *counter = 0u;
}

// Splits are chosen to fill the machine rather than to bound the work per
// block: the operation is latency-bound, so what matters is that enough
// wavefronts are resident to hide the B reads.
extern "C"
int hip_gemv_fp16_narrow_n_splits(int N, int K) {
  const int n_tiles = (N + kGemvWave - 1) / kGemvWave;
  if (n_tiles <= 0) return 1;
  int splits = 64 / n_tiles;
  if (splits < 1) splits = 1;
  // A split that scans fewer than 64 rows of B costs more in reduction than it
  // saves in parallelism.
  int max_by_k = K / 64;
  if (max_by_k < 1) max_by_k = 1;
  if (splits > max_by_k) splits = max_by_k;
  return splits;
}

extern "C"
int hip_gemv_fp16_narrow_n(void* stream, const void* A, const void* B, void* C,
                           int N, int K, void* scratch, size_t scratch_bytes) {
  if (N <= 0 || K <= 0 || N > kGemvMaxN) return -1;
  const int splits = hip_gemv_fp16_narrow_n_splits(N, K);
  const size_t need = hip_gemv_fp16_narrow_n_scratch_bytes(N, K);
  if (!scratch || scratch_bytes < need) return -1;

  float* partials = (float*)scratch;
  // The counter sits after the partials, 64 B-aligned so it does not share a
  // cache line with the data the last block is about to read.
  size_t pbytes = ((size_t)splits * N * sizeof(float) + 63) & ~(size_t)63;
  unsigned int* counter = (unsigned int*)((char*)scratch + pbytes);

  const int n_tiles = (N + kGemvWave - 1) / kGemvWave;
  const int k_chunk = (K + splits - 1) / splits;
  (void)hipGetLastError();
  gemv_fp16_narrow_n_kernel<<<dim3(n_tiles, splits), kGemvBlock, 0,
                              (hipStream_t)stream>>>(
      (const _Float16*)A, (const _Float16*)B, (_Float16*)C, partials, counter,
      N, K, splits, k_chunk);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C"
size_t hip_gemv_fp16_narrow_n_scratch_bytes(int N, int K) {
  if (N <= 0 || K <= 0) return 0;
  const int splits = hip_gemv_fp16_narrow_n_splits(N, K);
  size_t pbytes = ((size_t)splits * N * sizeof(float) + 63) & ~(size_t)63;
  return pbytes + 64;
}
