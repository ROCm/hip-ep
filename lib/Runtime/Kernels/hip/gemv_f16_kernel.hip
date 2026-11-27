/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

// GEMV out[N] = a[K] . B[K, N], B row-major (N contiguous), fp32 accumulation,
// for the narrow MatMuls of a decode step: MoE routers (fp16 K=2880, N=128;
// fp32 K=2048, N=256 and K=4096, N=288).
//
// Each workgroup owns LANES_N 16-byte vectors of columns over all of K:
// LANES_N lanes each load one vector of a row, kThreads / LANES_N row groups
// stride over K with U rows per thread in flight, and the row groups reduce
// through LDS. Every workgroup reads the same rows, so a row's bytes are
// fetched from DRAM once and served to the rest from L2.
static constexpr int kThreads = 256;

template <typename T> struct GemvElem;

template <> struct GemvElem<__half> {
  static constexpr int kVec = 8;
  __device__ static float load(const __half* p) { return __half2float(*p); }
  __device__ static __half store(float x) { return __float2half(x); }
  __device__ static void fma(float (&acc)[kVec], float av, const uint4& w) {
    const __half2* h = reinterpret_cast<const __half2*>(&w);
#pragma unroll
    for (int v = 0; v < kVec / 2; v++) {
      const float2 f = __half22float2(h[v]);
      acc[2 * v] += av * f.x;
      acc[2 * v + 1] += av * f.y;
    }
  }
};

template <> struct GemvElem<float> {
  static constexpr int kVec = 4;
  __device__ static float load(const float* p) { return *p; }
  __device__ static float store(float x) { return x; }
  __device__ static void fma(float (&acc)[kVec], float av, const uint4& w) {
    const float* f = reinterpret_cast<const float*>(&w);
#pragma unroll
    for (int v = 0; v < kVec; v++) acc[v] += av * f[v];
  }
};

template <typename T, int LANES_N, int U = 8>
__global__ void __launch_bounds__(kThreads)
gemv_kn_kernel(const T* __restrict__ a, const T* __restrict__ B,
               T* __restrict__ out, int K, int N) {
  using E = GemvElem<T>;
  constexpr int kVec = E::kVec;
  constexpr int kRows = kThreads / LANES_N;
  constexpr int kCols = kVec * LANES_N;
  const int ln = threadIdx.x % LANES_N;
  const int rg = threadIdx.x / LANES_N;
  const int col = (blockIdx.x * LANES_N + ln) * kVec;

  float acc[kVec];
#pragma unroll
  for (int v = 0; v < kVec; v++) acc[v] = 0.0f;
  if (col < N) {
    for (int k0 = rg; k0 < K; k0 += kRows * U) {
      uint4 w[U];
      float av[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int k = k0 + u * kRows;
        w[u] = k < K ? *reinterpret_cast<const uint4*>(B + static_cast<size_t>(k) * N + col)
                     : make_uint4(0u, 0u, 0u, 0u);
        av[u] = k < K ? E::load(a + k) : 0.0f;
      }
#pragma unroll
      for (int u = 0; u < U; u++) E::fma(acc, av[u], w[u]);
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
    if (c < N) out[c] = E::store(s);
  }
}

template <typename T, int LANES_N, int U = 8>
static void launch_gemv_kn(hipStream_t s, const T* a, const T* B, T* out,
                           int64_t K, int64_t N) {
  constexpr int kCols = GemvElem<T>::kVec * LANES_N;
  const unsigned grid = static_cast<unsigned>((N + kCols - 1) / kCols);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_kn_kernel<T, LANES_N, U>), dim3(grid),
                     dim3(kThreads), 0, s, a, B, out, static_cast<int>(K),
                     static_cast<int>(N));
}

template <typename T>
static int gemv_kn(const char* name, void* stream, const void* a, const void* B,
                   void* out, int64_t K, int64_t N) {
  constexpr int kVec = GemvElem<T>::kVec;
  if (K <= 0 || N <= 0 || K > INT32_MAX || N > INT32_MAX || (N % kVec) != 0 ||
      (reinterpret_cast<uintptr_t>(B) % 16) != 0) {
    fprintf(stderr, "[custom_kernels] %s: unsupported K=%lld N=%lld (N must be "
                    "a multiple of %d, B 16-byte aligned)\n",
            name, (long long)K, (long long)N, kVec);
    return -1;
  }
  hipStream_t s = static_cast<hipStream_t>(stream);
  const auto* pa = static_cast<const T*>(a);
  const auto* pB = static_cast<const T*>(B);
  auto* po = static_cast<T*>(out);
  // At these sizes the workgroup count, not the workgroup width, sets the
  // bandwidth. gfx1151, cold weights: fp32 K=2048 N=256 11.0 us at 2 lanes vs
  // 17.5 us at 8; fp32 K=4096 N=288 23.0 vs 36.2 us; fp16 K=2880 N=128 6.2 us
  // at 1 lane vs 7.4 us at 2 and 11.8 us at 4.
  const int64_t vecs = N / kVec;
  (void)hipGetLastError();
  if (vecs >= 64)
    launch_gemv_kn<T, 2>(s, pa, pB, po, K, N);
  else
    launch_gemv_kn<T, 1>(s, pa, pB, po, K, N);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    fprintf(stderr, "[custom_kernels] %s failed: %s\n", name,
            hipGetErrorString(e));
    return static_cast<int>(e);
  }
  return 0;
}

extern "C" int hip_gemv_f16_kn(void* stream, const void* a, const void* B,
                               void* out, int64_t K, int64_t N) {
  return gemv_kn<__half>("hip_gemv_f16_kn", stream, a, B, out, K, N);
}

extern "C" int hip_gemv_f32_kn(void* stream, const void* a, const void* B,
                               void* out, int64_t K, int64_t N) {
  return gemv_kn<float>("hip_gemv_f32_kn", stream, a, B, out, K, N);
}
