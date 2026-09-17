/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include <hip/hip_runtime.h>

#include "hip_custom_kernels.h"

#include <cstdint>

// fp16 GEMV: y[1, N] = a[1, K] @ b[K, N], row-major, no bias, alpha = 1.
//
// The tuned instances in ck_gemm.hip are GEMM tiles, so at M == 1 the tile's M
// extent sits idle and the shape runs far below bandwidth: on a 201088x2880
// weight matrix the fastest of the 64 instances reaches ~28 GB/s where this
// kernel reaches ~129 GB/s.
//
// One thread owns VEC adjacent output columns and walks K, so each k step reads
// VEC * blockDim.x contiguous halves of b. Accumulation is fp32, matching what
// the GEMM path computes.

namespace {

constexpr int kBlock = 256;
constexpr int kKChunk = 1024;

typedef _Float16 half4 __attribute__((ext_vector_type(4)));

template <int VEC>
__device__ __forceinline__ void loadCols(const _Float16 *p, float (&out)[VEC]);

template <>
__device__ __forceinline__ void loadCols<1>(const _Float16 *p,
                                            float (&out)[1]) {
  out[0] = static_cast<float>(p[0]);
}

template <>
__device__ __forceinline__ void loadCols<4>(const _Float16 *p,
                                            float (&out)[4]) {
  const half4 v = *reinterpret_cast<const half4 *>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<float>(v[i]);
  }
}

template <int VEC>
__device__ __forceinline__ void storeCols(_Float16 *p, const float (&in)[VEC]);

template <>
__device__ __forceinline__ void storeCols<1>(_Float16 *p,
                                             const float (&in)[1]) {
  p[0] = static_cast<_Float16>(in[0]);
}

template <>
__device__ __forceinline__ void storeCols<4>(_Float16 *p,
                                             const float (&in)[4]) {
  half4 v;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[i] = static_cast<_Float16>(in[i]);
  }
  *reinterpret_cast<half4 *>(p) = v;
}

template <int VEC>
__global__ __launch_bounds__(kBlock) void gemvF16Kernel(
    const _Float16 *__restrict__ a, const _Float16 *__restrict__ b,
    _Float16 *__restrict__ y, int64_t n, int64_t k) {
  __shared__ _Float16 aTile[kKChunk];

  const int64_t col =
      (static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x) * VEC;
  // VEC divides n and col, so one bound check covers all VEC lanes.
  const bool active = col < n;

  float acc[VEC];
#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    acc[v] = 0.0f;
  }

  for (int64_t k0 = 0; k0 < k; k0 += kKChunk) {
    const int kn = static_cast<int>(min(static_cast<int64_t>(kKChunk), k - k0));
    for (int i = threadIdx.x; i < kn; i += kBlock) {
      aTile[i] = a[k0 + i];
    }
    __syncthreads();

    if (active) {
      const _Float16 *row = b + k0 * n + col;
#pragma unroll 4
      for (int kk = 0; kk < kn; ++kk) {
        float cols[VEC];
        loadCols<VEC>(row, cols);
        const float av = static_cast<float>(aTile[kk]);
#pragma unroll
        for (int v = 0; v < VEC; ++v) {
          acc[v] = fmaf(av, cols[v], acc[v]);
        }
        row += n;
      }
    }
    __syncthreads();
  }

  if (active) {
    storeCols<VEC>(y + col, acc);
  }
}

int deviceCuCount() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return 0;
  }
  int cu = 0;
  if (hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev) !=
      hipSuccess) {
    return 0;
  }
  return cu;
}

} // namespace

extern "C" int hip_gemv_fp16(void *stream, const void *a, const void *b,
                             void *y, int64_t n, int64_t k) {
  if (!a || !b || !y || n <= 0 || k <= 0) {
    return 1;
  }

  // A vector load needs b's row stride (n halves) and the column offset to keep
  // the 8-byte alignment, which holds exactly when 4 divides n.
  const int vec = (n % 4 == 0) ? 4 : 1;
  const int64_t tile = static_cast<int64_t>(kBlock) * vec;
  const int64_t blocks = (n + tile - 1) / tile;

  // Only the column axis carries parallelism here, so a narrow output would run
  // on a handful of CUs. Decline and let the caller keep its GEMM route.
  const int cu = deviceCuCount();
  if (cu <= 0 || blocks < cu || blocks > INT32_MAX) {
    return 1;
  }

  const auto s = static_cast<hipStream_t>(stream);
  const dim3 grid(static_cast<unsigned>(blocks));
  const dim3 block(kBlock);
  const auto *ah = static_cast<const _Float16 *>(a);
  const auto *bh = static_cast<const _Float16 *>(b);
  auto *yh = static_cast<_Float16 *>(y);

  if (vec == 4) {
    gemvF16Kernel<4><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
  } else {
    gemvF16Kernel<1><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
  }
  return static_cast<int>(hipGetLastError());
}
