/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

// fp16 GEMV out[N] = a[K] . B[K, N], B row-major (N contiguous), for the
// narrow MatMuls of a decode step (an MoE router: K=2880, N=128).
//
// Each workgroup owns LANES_N * kVec columns over all of K: LANES_N lanes each
// load one 16-byte vector (8 columns) of a row, kThreads / LANES_N row groups
// stride over K, and the row groups reduce through LDS. Every workgroup reads
// the same rows, so a row's 2*N bytes are fetched from DRAM once and served to
// the rest from L2. Wider outputs take more lanes per row so the workgroup
// count, and with it the number of busy CUs, grows more slowly than N.
static constexpr int kVec = 8;
static constexpr int kThreads = 256;

template <int LANES_N>
__global__ void __launch_bounds__(kThreads)
gemv_f16_kn_kernel(const __half* __restrict__ a, const __half* __restrict__ B,
                   __half* __restrict__ out, int K, int N) {
  constexpr int kRows = kThreads / LANES_N;
  constexpr int kCols = kVec * LANES_N;
  const int ln = threadIdx.x % LANES_N;
  const int rg = threadIdx.x / LANES_N;
  const int col = (blockIdx.x * LANES_N + ln) * kVec;

  float acc[kVec];
#pragma unroll
  for (int v = 0; v < kVec; v++) acc[v] = 0.0f;
  if (col < N) {
#pragma unroll 4
    for (int k = rg; k < K; k += kRows) {
      const float av = __half2float(a[k]);
      const uint4 w =
          *reinterpret_cast<const uint4*>(B + static_cast<size_t>(k) * N + col);
      const __half2* h = reinterpret_cast<const __half2*>(&w);
#pragma unroll
      for (int v = 0; v < kVec / 2; v++) {
        const float2 f = __half22float2(h[v]);
        acc[2 * v] += av * f.x;
        acc[2 * v + 1] += av * f.y;
      }
    }
  }

  __shared__ float red[kRows][kCols + 1];
#pragma unroll
  for (int v = 0; v < kVec; v++) red[rg][ln * kVec + v] = acc[v];
  __syncthreads();
  for (int i = threadIdx.x; i < kCols; i += blockDim.x) {
    float s = 0.0f;
    for (int r = 0; r < kRows; r++) s += red[r][i];
    const int c = blockIdx.x * kCols + i;
    if (c < N) out[c] = __float2half(s);
  }
}

template <int LANES_N>
static void launch_gemv_f16_kn(hipStream_t s, const __half* a, const __half* B,
                               __half* out, int64_t K, int64_t N) {
  constexpr int kCols = kVec * LANES_N;
  const unsigned grid = static_cast<unsigned>((N + kCols - 1) / kCols);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_f16_kn_kernel<LANES_N>), dim3(grid),
                     dim3(kThreads), 0, s, a, B, out, static_cast<int>(K),
                     static_cast<int>(N));
}

extern "C" int hip_gemv_f16_kn(void* stream, const void* a, const void* B,
                               void* out, int64_t K, int64_t N) {
  if (K <= 0 || N <= 0 || K > INT32_MAX || N > INT32_MAX || (N % kVec) != 0 ||
      (reinterpret_cast<uintptr_t>(B) % 16) != 0) {
    fprintf(stderr, "[custom_kernels] hip_gemv_f16_kn: unsupported K=%lld "
                    "N=%lld (N must be a multiple of %d, B 16-byte aligned)\n",
            (long long)K, (long long)N, kVec);
    return -1;
  }
  hipStream_t s = static_cast<hipStream_t>(stream);
  const auto* pa = static_cast<const __half*>(a);
  const auto* pB = static_cast<const __half*>(B);
  auto* po = static_cast<__half*>(out);
  if (N <= 32)
    launch_gemv_f16_kn<1>(s, pa, pB, po, K, N);
  else if (N <= 128)
    launch_gemv_f16_kn<2>(s, pa, pB, po, K, N);
  else if (N <= 256)
    launch_gemv_f16_kn<4>(s, pa, pB, po, K, N);
  else
    launch_gemv_f16_kn<8>(s, pa, pB, po, K, N);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_gemv_f16_kn failed: %s\n",
            hipGetErrorString(e));
    return static_cast<int>(e);
  }
  return 0;
}
