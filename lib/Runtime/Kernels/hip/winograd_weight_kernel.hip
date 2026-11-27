/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- winograd_weight_kernel.hip - Winograd F(2,3) filter layout ---------===//
//
// Rewrites a [K, C, 3, 3] fp16 conv filter into the store that the Winograd
// F(2x2, 3x3) conv kernel reads (third_party/migraphx winograd_conv.hpp), with
// C innermost so the kernel's weight loads coalesce:
//   full_transform = 0: T = G g, shape [4, 3, K, C]; the kernel applies G^T.
//   full_transform = 1: the raw g, shape [3, 3, K, C]; the kernel applies both.
// G has rows [1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1].
//
// The filter is a model constant, so the runtime runs this once per op state.
//===----------------------------------------------------------------------===//

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "hip_custom_kernels.h"

namespace {

__global__ void winograd_f23_weight_kernel(const __half* __restrict__ g,
                                           __half* __restrict__ u, int64_t kc,
                                           int full_transform) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= kc)
    return;
  // g is [K, C, 3, 3]: (k, c) owns 9 contiguous values at idx * 9. The store
  // is [rows, 3, K, C], so (i, j, k, c) sits at (i * 3 + j) * K * C + idx.
  const __half* src = g + idx * 9;
  float w[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      w[i][j] = __half2float(src[i * 3 + j]);
  auto store = [&](int i, int j, float v) {
    u[(static_cast<int64_t>(i) * 3 + j) * kc + idx] = __float2half(v);
  };
  for (int j = 0; j < 3; ++j) {
    if (full_transform) {
      for (int i = 0; i < 3; ++i)
        store(i, j, w[i][j]);
    } else {
      store(0, j, w[0][j]);
      store(1, j, 0.5f * (w[0][j] + w[1][j] + w[2][j]));
      store(2, j, 0.5f * (w[0][j] - w[1][j] + w[2][j]));
      store(3, j, w[2][j]);
    }
  }
}

} // namespace

extern "C" int hip_winograd_f23_weight_transform_f16(void* stream,
                                                     const void* weights,
                                                     void* store, int64_t K,
                                                     int64_t C,
                                                     int full_transform) {
  if (!weights || !store || K <= 0 || C <= 0)
    return -1;
  const int64_t kc = K * C;
  constexpr int kThreads = 256;
  const int64_t blocks = (kc + kThreads - 1) / kThreads;
  (void)hipGetLastError();
  hipLaunchKernelGGL(winograd_f23_weight_kernel, dim3(static_cast<unsigned>(blocks)),
                     dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                     static_cast<const __half*>(weights),
                     static_cast<__half*>(store), kc, full_transform);
  return hipGetLastError();
}
