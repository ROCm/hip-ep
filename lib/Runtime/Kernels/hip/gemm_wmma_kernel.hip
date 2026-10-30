/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_bf16.h>
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

__global__ void gemv_f32_transposed_b_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int N,
                                             int K) {
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int output = blockIdx.x;
    if (wave != 0 || output >= N)
        return;

    const float *weight = B + static_cast<size_t>(output) * K;
    float sum = 0.0f;
    for (int64_t k = lane; k < K; k += warpSize)
        sum = fmaf(A[k], weight[k], sum);
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);
    if (lane == 0)
        C[output] = sum;
}

template <int BlockSize, int Accumulators>
__global__ void
gemv_f32_transposed_b_multiwave_kernel(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C, int N, int K) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int waves_per_block = blockDim.x / warpSize;
  const int output = blockIdx.x;
  if (output >= N)
    return;

  const float *weight = B + static_cast<size_t>(output) * K;
  float sums[Accumulators] = {};
  int k = threadIdx.x;
  for (; k + (Accumulators - 1) * BlockSize < K;
       k += Accumulators * BlockSize) {
#pragma unroll
    for (int accumulator = 0; accumulator < Accumulators; ++accumulator) {
      const int index = k + accumulator * BlockSize;
      sums[accumulator] = fmaf(A[index], weight[index], sums[accumulator]);
    }
  }
  for (; k < K; k += BlockSize)
    sums[0] = fmaf(A[k], weight[k], sums[0]);

  float sum = sums[0];
#pragma unroll
  for (int accumulator = 1; accumulator < Accumulators; ++accumulator)
    sum += sums[accumulator];
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    sum += __shfl_down(sum, offset);

  __shared__ float wave_sums[BlockSize / 32];
  if (lane == 0)
    wave_sums[wave] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float output_sum = 0.0f;
    for (int wave_index = 0; wave_index < waves_per_block; ++wave_index)
      output_sum += wave_sums[wave_index];
    C[output] = output_sum;
  }
}

extern "C" int hip_gemv_f32_transposed_b(void *stream, const float *A,
                                         const float *B, float *C, int N,
                                         int K) {
  if (!stream || !A || !B || !C || N <= 0 || K <= 0)
    return -1;

  // A 64-thread block contains one wave on wave64 targets and two waves on
  // wave32 targets. Only the first wave participates, so the same kernel is
  // valid for either hardware wave size without a host-side device query.
  constexpr int kBlockSize = 64;
  // Clear an error an earlier HIP call left, so the status below is this
  // launch's own.
  (void)hipGetLastError();
  hipLaunchKernelGGL(gemv_f32_transposed_b_kernel, dim3(N), dim3(kBlockSize), 0,
                     static_cast<hipStream_t>(stream), A, B, C, N, K);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

// Same reduction with 16-byte loads: each of the BlockSize threads reads a
// float4 of the row and of the input per step.
template <int BlockSize>
__global__ void
gemv_f32_transposed_b_vec4_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int N, int K) {
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  const int output = blockIdx.x;
  if (output >= N)
    return;

  const float4 *weight =
      reinterpret_cast<const float4 *>(B + static_cast<size_t>(output) * K);
  const float4 *input = reinterpret_cast<const float4 *>(A);
  float sum = 0.0f;
  for (int k = threadIdx.x; k < K / 4; k += BlockSize) {
    const float4 a = input[k];
    const float4 w = weight[k];
    sum = fmaf(a.x, w.x, sum);
    sum = fmaf(a.y, w.y, sum);
    sum = fmaf(a.z, w.z, sum);
    sum = fmaf(a.w, w.w, sum);
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    sum += __shfl_down(sum, offset);

  __shared__ float wave_sums[BlockSize / 32];
  if (lane == 0)
    wave_sums[wave] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float output_sum = 0.0f;
    for (int wave_index = 0; wave_index < BlockSize / warpSize; ++wave_index)
      output_sum += wave_sums[wave_index];
    C[output] = output_sum;
  }
}

extern "C" int hip_gemv_f32_transposed_b_multiwave_gfx1151(
    void *stream, const float *A, const float *B, float *C, int N, int K) {
  if (!stream || !A || !B || !C || N <= 0 || N > 256 || K <= 0)
    return -1;

  // One 256-thread block per output row. With K a multiple of four and
  // 16-byte aligned operands every row is read as float4; otherwise four
  // scalar accumulators per thread. On gfx1151 at N in {128, 256} and K from
  // 1024 to 7168 this runs at 150-220 GB/s, 2.4-4x the one-wave kernel; the
  // 128-thread form it replaces loses up to 1.7x at power-of-two K.
  constexpr int kBlockSize = 256;
  const bool vectorized =
      K % 4 == 0 &&
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       15) == 0;
  (void)hipGetLastError();
  if (vectorized)
    hipLaunchKernelGGL((gemv_f32_transposed_b_vec4_kernel<kBlockSize>), dim3(N),
                       dim3(kBlockSize), 0, static_cast<hipStream_t>(stream), A,
                       B, C, N, K);
  else
    hipLaunchKernelGGL((gemv_f32_transposed_b_multiwave_kernel<kBlockSize, 4>),
                       dim3(N), dim3(kBlockSize), 0,
                       static_cast<hipStream_t>(stream), A, B, C, N, K);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

namespace {

__device__ __forceinline__ float bf16Dot2(__hip_bfloat162 lhs,
                                          __hip_bfloat162 rhs,
                                          float accumulator) {
#if defined(__gfx1151__)
    return __builtin_amdgcn_fdot2_f32_bf16(
        static_cast<__bf16_2>(lhs), static_cast<__bf16_2>(rhs), accumulator,
        false);
#else
    const float2 lhs_f32 = __bfloat1622float2(lhs);
    const float2 rhs_f32 = __bfloat1622float2(rhs);
    return fmaf(lhs_f32.y, rhs_f32.y,
                fmaf(lhs_f32.x, rhs_f32.x, accumulator));
#endif
}

template <int BlockSize, int Accumulators>
__global__ void gemv_bf16_f32_transposed_b_kernel(
    const __hip_bfloat162 *__restrict__ A,
    const __hip_bfloat162 *__restrict__ B, float *__restrict__ C, int N,
    int K2) {
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int waves_per_block = blockDim.x / warpSize;
    const int output = blockIdx.x;
    if (output >= N)
        return;

    const __hip_bfloat162 *weight =
        B + static_cast<size_t>(output) * K2;
    float sums[Accumulators] = {};
    const int wave_stride = waves_per_block * warpSize;
    int k = wave * warpSize + lane;
    for (; k + (Accumulators - 1) * wave_stride < K2;
         k += Accumulators * wave_stride) {
#pragma unroll
        for (int accumulator = 0; accumulator < Accumulators; ++accumulator) {
            const int index = k + accumulator * wave_stride;
            sums[accumulator] =
                bf16Dot2(A[index], weight[index], sums[accumulator]);
        }
    }
    for (; k < K2; k += wave_stride)
        sums[0] = bf16Dot2(A[k], weight[k], sums[0]);

    float sum;
    if constexpr (Accumulators == 2) {
        sum = sums[0] + sums[1];
    } else {
        static_assert(Accumulators == 4);
        sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    __shared__ float wave_sums[BlockSize / 32];
    if (lane == 0)
        wave_sums[wave] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float output_sum = 0.0f;
        for (int wave_index = 0; wave_index < waves_per_block; ++wave_index)
            output_sum += wave_sums[wave_index];
        C[output] = output_sum;
    }
}

// BF16 x BF16 -> F32 GEMM for prefill-sized transposed-B projections:
// C[M,N] = A[M,K] * transpose(B[N,K]), all row-major, F32 accumulation.
//
// A block of 256 threads (eight waves, WM x WN) computes a BM x BN tile of C;
// each wave owns (BM/WM) x (BN/WN) outputs as 16x16 WMMA tiles. A and B are
// staged through padded shared memory in BK-wide slices, double-buffered: the
// next slice is loaded into registers while the WMMAs run on the current one.
// Blocks walk M fastest, so the blocks sharing one slice of the weights run
// together and read it from DRAM once.
//
// Only the gfx11 WMMA encoding (RDNA3/3.5) is implemented: a_frag at lane l
// holds A_tile[l % 16][0..15] and b_frag holds B_tile[l % 16][0..15] (B rows
// are N, so both are contiguous K runs); accumulator element e is row
// 2 * e + l / 16, column l % 16.
#if defined(__HIP_DEVICE_COMPILE__) &&                                        \
    __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
#define HIPDNN_HAS_BF16_WMMA_GFX11 1
#else
#define HIPDNN_HAS_BF16_WMMA_GFX11 0
#endif

typedef short bf16x16 __attribute__((ext_vector_type(16)));

template <int BM, int BN, int BK, int WM, int WN>
__global__ void __launch_bounds__(256) gemm_bf16_f32_transposed_b_kernel(
    const uint16_t *__restrict__ A, const uint16_t *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
#if HIPDNN_HAS_BF16_WMMA_GFX11
  static_assert(WM * WN == 8, "eight waves per block");
  constexpr int TM = BM / WM / kWmmaTile;
  constexpr int TN = BN / WN / kWmmaTile;
  constexpr int kVector = 8; // BF16 values per 16-byte load
  constexpr int kStride = BK + 8;
  constexpr int AVectors = BM * BK / kVector / 256;
  constexpr int BVectors = BN * BK / kVector / 256;
  static_assert(AVectors * 256 * kVector == BM * BK &&
                BVectors * 256 * kVector == BN * BK);
  __shared__ __attribute__((aligned(16))) uint16_t As[2][BM * kStride];
  __shared__ __attribute__((aligned(16))) uint16_t Bs[2][BN * kStride];

  const int thread = threadIdx.x;
  const int wave = thread / 32;
  const int lane = thread % 32;
  const int row16 = lane % kWmmaTile;
  const int waveM = wave / WN;
  const int waveN = wave % WN;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;

  uint4 aStage[AVectors];
  uint4 bStage[BVectors];
  auto loadGlobal = [&](int k0) {
#pragma unroll
    for (int i = 0; i < AVectors; ++i) {
      const int v = thread + i * 256;
      const int r = v / (BK / kVector);
      const int c = (v % (BK / kVector)) * kVector;
      aStage[i] = m0 + r < M && k0 + c < K
                      ? *reinterpret_cast<const uint4 *>(
                            A + static_cast<size_t>(m0 + r) * K + k0 + c)
                      : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int i = 0; i < BVectors; ++i) {
      const int v = thread + i * 256;
      const int r = v / (BK / kVector);
      const int c = (v % (BK / kVector)) * kVector;
      bStage[i] = n0 + r < N && k0 + c < K
                      ? *reinterpret_cast<const uint4 *>(
                            B + static_cast<size_t>(n0 + r) * K + k0 + c)
                      : make_uint4(0, 0, 0, 0);
    }
  };
  auto storeShared = [&](int buffer) {
#pragma unroll
    for (int i = 0; i < AVectors; ++i) {
      const int v = thread + i * 256;
      *reinterpret_cast<uint4 *>(
          &As[buffer][(v / (BK / kVector)) * kStride +
                      (v % (BK / kVector)) * kVector]) = aStage[i];
    }
#pragma unroll
    for (int i = 0; i < BVectors; ++i) {
      const int v = thread + i * 256;
      *reinterpret_cast<uint4 *>(
          &Bs[buffer][(v / (BK / kVector)) * kStride +
                      (v % (BK / kVector)) * kVector]) = bStage[i];
    }
  };
  auto fragment = [](const uint16_t *row) {
    bf16x16 value;
    const uint4 low = *reinterpret_cast<const uint4 *>(row);
    const uint4 high = *reinterpret_cast<const uint4 *>(row + 8);
    __builtin_memcpy(&value, &low, sizeof(low));
    __builtin_memcpy(reinterpret_cast<char *>(&value) + sizeof(low), &high,
                     sizeof(high));
    return value;
  };

  float8 acc[TM][TN] = {};
  loadGlobal(0);
  storeShared(0);
  __syncthreads();
  int buffer = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool more = k0 + BK < K;
    if (more)
      loadGlobal(k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += kWmmaTile) {
      bf16x16 a[TM];
      bf16x16 b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        a[i] = fragment(&As[buffer][((waveM * TM + i) * kWmmaTile + row16) *
                                        kStride +
                                    kk]);
#pragma unroll
      for (int j = 0; j < TN; ++j)
        b[j] = fragment(&Bs[buffer][((waveN * TN + j) * kWmmaTile + row16) *
                                        kStride +
                                    kk]);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] =
              __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[i], b[j], acc[i][j]);
    }
    if (more) {
      storeShared(buffer ^ 1);
      __syncthreads();
      buffer ^= 1;
    }
  }

  const int pair = lane / kWmmaTile;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = m0 + (waveM * TM + i) * kWmmaTile + 2 * e + pair;
        const int c = n0 + (waveN * TN + j) * kWmmaTile + row16;
        if (r < M && c < N)
          C[static_cast<size_t>(r) * N + c] = acc[i][j][e];
      }
#else
  __builtin_trap();
#endif
}

} // namespace

extern "C" int hip_gemv_bf16_f32_transposed_b(void *stream, const void *A,
                                                const void *B, float *C,
                                                int N, int K) {
    constexpr uintptr_t kPairAlignmentMask = alignof(uint32_t) - 1;
    if (!stream || !A || !B || !C || N <= 0 || K <= 0 || K % 2 != 0 ||
        ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
         kPairAlignmentMask) != 0)
        return -1;

    const auto *input = static_cast<const __hip_bfloat162 *>(A);
    const auto *weight = static_cast<const __hip_bfloat162 *>(B);
    // Clear an error an earlier HIP call left, so the status below is this
    // launch's own.
    (void)hipGetLastError();
    // gfx1151 benefits from more waves on long rows. Independent accumulator
    // chains hide dot-product latency while every wave contributes useful
    // work to the same output row.
    if (K >= 8192) {
        hipLaunchKernelGGL((gemv_bf16_f32_transposed_b_kernel<512, 2>), dim3(N),
                           dim3(512), 0, static_cast<hipStream_t>(stream), input,
                           weight, C, N, K / 2);
    } else {
        hipLaunchKernelGGL((gemv_bf16_f32_transposed_b_kernel<128, 4>), dim3(N),
                           dim3(128), 0, static_cast<hipStream_t>(stream), input,
                           weight, C, N, K / 2);
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

extern "C" int hip_gemm_bf16_f32_transposed_b(void *stream, const void *A,
                                              const void *B, float *C, int M,
                                              int N, int K) {
  constexpr uintptr_t kVectorAlignmentMask = alignof(uint4) - 1;
  if (!stream || !A || !B || !C || M <= 0 || N <= 0 || K <= 0 || K % 8 != 0 ||
      ((reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B)) &
       kVectorAlignmentMask) != 0 ||
      hipdnn_device_is_wave64())
    return -1;
  const auto *input = static_cast<const uint16_t *>(A);
  const auto *weight = static_cast<const uint16_t *>(B);
  const auto hipStream = static_cast<hipStream_t>(stream);
  // Clear an error an earlier HIP call left, so the status below is this
  // launch's own.
  (void)hipGetLastError();
  // 128 x 64 tiles keep eight waves busy on 2x2 WMMA tiles each for prompt-
  // sized M; short M takes 64 x 64 tiles so fewer rows are padding.
  if (M >= 128) {
    constexpr int BM = 128, BN = 64;
    hipLaunchKernelGGL((gemm_bf16_f32_transposed_b_kernel<BM, BN, 64, 4, 2>),
                       dim3((M + BM - 1) / BM, (N + BN - 1) / BN), dim3(256),
                       0, hipStream, input, weight, C, M, N, K);
  } else {
    constexpr int BM = 64, BN = 64;
    hipLaunchKernelGGL((gemm_bf16_f32_transposed_b_kernel<BM, BN, 64, 2, 4>),
                       dim3((M + BM - 1) / BM, (N + BN - 1) / BN), dim3(256),
                       0, hipStream, input, weight, C, M, N, K);
  }
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
