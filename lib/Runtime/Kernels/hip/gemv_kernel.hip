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
// weight matrix the fastest tuned instance reaches ~28 GB/s where this kernel
// reaches ~129 GB/s.
//
// One thread owns VEC adjacent output columns and walks K, so each k step reads
// VEC * blockDim.x contiguous halves of b. Accumulation is fp32, matching what
// the GEMM path computes.
//
// Columns are the only parallel axis in that scheme, so a narrow output leaves
// most of the device idle. Below the width that fills it, a second kernel adds
// K as a block-local axis: blockDim.y threads walk K cyclically and are reduced
// through LDS, which keeps the reduction inside the block and needs no
// workspace. The MoE router m=1,n=128,k=2880 runs 5x faster that way than as a
// CK GEMM tile.

namespace {

constexpr int kBlock = 256;
constexpr int kKChunk = 1024;

constexpr int kNarrowThreads = 512;
// Threads to keep resident, so the reads stay deep enough to cover memory
// latency while each thread still walks a long enough K run to amortize the
// reduction.
constexpr int64_t kNarrowTargetThreads = 4096;
constexpr int64_t kNarrowMinCols = 64;
constexpr int64_t kNarrowMinK = 256;

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

template <int VEC>
__global__ __launch_bounds__(kNarrowThreads) void gemvF16NarrowKernel(
    const _Float16 *__restrict__ a, const _Float16 *__restrict__ b,
    _Float16 *__restrict__ y, int64_t n, int64_t k) {
  __shared__ float partial[kNarrowThreads * VEC];

  const int cols = static_cast<int>(blockDim.x) * VEC;
  const int lane = static_cast<int>(threadIdx.x) * VEC;
  const int64_t col =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC;
  // VEC divides n and col, so one bound check covers all VEC lanes.
  const bool active = col < n;

  float acc[VEC];
#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    acc[v] = 0.0f;
  }

  if (active) {
    const _Float16 *row = b + static_cast<int64_t>(threadIdx.y) * n + col;
    const int64_t step = static_cast<int64_t>(blockDim.y) * n;
#pragma unroll 4
    for (int64_t kk = threadIdx.y; kk < k; kk += blockDim.y) {
      float bv[VEC];
      loadCols<VEC>(row, bv);
      const float av = static_cast<float>(a[kk]);
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        acc[v] = fmaf(av, bv[v], acc[v]);
      }
      row += step;
    }
  }

#pragma unroll
  for (int v = 0; v < VEC; ++v) {
    partial[threadIdx.y * cols + lane + v] = acc[v];
  }
  __syncthreads();

  // blockDim.y is a power of two, so the tree halves down to row 0 exactly.
  for (unsigned s = blockDim.y >> 1; s > 0; s >>= 1) {
    if (threadIdx.y < s) {
#pragma unroll
      for (int v = 0; v < VEC; ++v) {
        partial[threadIdx.y * cols + lane + v] +=
            partial[(threadIdx.y + s) * cols + lane + v];
      }
    }
    __syncthreads();
  }

  if (threadIdx.y == 0 && active) {
    float out[VEC];
#pragma unroll
    for (int v = 0; v < VEC; ++v) {
      out[v] = partial[lane + v];
    }
    storeCols<VEC>(y + col, out);
  }
}

int pow2Floor(int64_t v) {
  int r = 1;
  while (static_cast<int64_t>(r) * 2 <= v) {
    r *= 2;
  }
  return r;
}

struct NarrowSplit {
  int colThreads;
  int kSlices;
  int64_t blocks;
};

// Every bound here is a power of two, which the kernel's LDS tree reduction
// requires of kSlices.
NarrowSplit chooseNarrowSplit(int64_t colGroups, int64_t k) {
  int kSlices = pow2Floor(kNarrowTargetThreads / colGroups);
  if (kSlices < 32) {
    kSlices = 32;
  }
  if (kSlices > 256) {
    kSlices = 256;
  }
  const int kCap = pow2Floor(k);
  if (kSlices > kCap) {
    kSlices = kCap;
  }
  const int colThreads = kNarrowThreads / kSlices;
  return {colThreads, kSlices, (colGroups + colThreads - 1) / colThreads};
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

  const int cu = deviceCuCount();
  if (cu <= 0 || blocks > INT32_MAX) {
    return 1;
  }

  const auto s = static_cast<hipStream_t>(stream);
  const auto *ah = static_cast<const _Float16 *>(a);
  const auto *bh = static_cast<const _Float16 *>(b);
  auto *yh = static_cast<_Float16 *>(y);

  if (blocks >= cu) {
    const dim3 grid(static_cast<unsigned>(blocks));
    const dim3 block(kBlock);
    if (vec == 4) {
      gemvF16Kernel<4><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
    } else {
      gemvF16Kernel<1><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
    }
    return static_cast<int>(hipGetLastError());
  }

  // Narrower than the column axis alone can fill. Decline the widths and depths
  // the K split was not measured on and let the caller keep its GEMM route.
  if (n < kNarrowMinCols || k < kNarrowMinK) {
    return 1;
  }

  const NarrowSplit split = chooseNarrowSplit(n / vec, k);
  const dim3 grid(static_cast<unsigned>(split.blocks));
  const dim3 block(static_cast<unsigned>(split.colThreads),
                   static_cast<unsigned>(split.kSlices));
  if (vec == 4) {
    gemvF16NarrowKernel<4><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
  } else {
    gemvF16NarrowKernel<1><<<grid, block, 0, s>>>(ah, bh, yh, n, k);
  }
  return static_cast<int>(hipGetLastError());
}
