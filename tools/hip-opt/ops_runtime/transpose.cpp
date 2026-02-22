/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- transpose.cpp - hip.transpose runtime
//-------------------------------===//
//
// N-D transpose swapping two specified dimensions.
//
// Signature from MLIR lowering:
//   hip_transpose(handle, input, output, rank, dim0, dim1, s0, s1, s2)
//
// Shape is padded to 3 dims with trailing 1s.
// For 3D [B,S,D] transposing dims 1,2 -> [B,D,S].
// For 2D [M,N] transposing dims 0,1 -> [N,M] (classic transpose).
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime_api.h>
#include <vector>

extern "C" void hip_transpose(void* /*handle*/, void* input, void* output,
                              int64_t rank, int64_t dim0, int64_t dim1,
                              int64_t s0, int64_t s1, int64_t s2) {
  int64_t shape[3] = {s0, s1, s2};
  fprintf(
      stderr,
      "[transpose] rank=%lld, swap dims %lld<->%lld, shape=[%lld,%lld,%lld]\n",
      (long long)rank, (long long)dim0, (long long)dim1, (long long)s0,
      (long long)s1, (long long)s2);

  int64_t total = s0 * s1 * s2;
  std::vector<float> h_in(total), h_out(total);
  hipMemcpy(h_in.data(), input, total * sizeof(float), hipMemcpyDeviceToHost);

  // Compute input strides (row-major)
  int64_t in_stride[3] = {s1 * s2, s2, 1};

  // Output shape: swap dim0 and dim1
  int64_t out_shape[3] = {s0, s1, s2};
  std::swap(out_shape[dim0], out_shape[dim1]);
  int64_t out_stride[3] = {out_shape[1] * out_shape[2], out_shape[2], 1};

  for (int64_t i0 = 0; i0 < s0; i0++)
    for (int64_t i1 = 0; i1 < s1; i1++)
      for (int64_t i2 = 0; i2 < s2; i2++) {
        int64_t idx[3] = {i0, i1, i2};
        int64_t src = i0 * in_stride[0] + i1 * in_stride[1] + i2 * in_stride[2];
        std::swap(idx[dim0], idx[dim1]);
        int64_t dst = idx[0] * out_stride[0] + idx[1] * out_stride[1] +
                      idx[2] * out_stride[2];
        h_out[dst] = h_in[src];
      }

  hipMemcpy(output, h_out.data(), total * sizeof(float), hipMemcpyHostToDevice);
}
