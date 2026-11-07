/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Operation shapes the DynamicDispatch (NPU) runtime wrappers can execute.
// Shared by the HipToLLVM lowering, which keeps unsupported operations on the
// GPU, and by wrap_dd_matmul / wrap_dd_conv2d, which map the shapes onto DD
// kernels. Keep the two in agreement by changing only this header.

#ifndef HIP_DYNAMIC_DISPATCH_SUPPORT_H
#define HIP_DYNAMIC_DISPATCH_SUPPORT_H

#include <cstdint>

namespace hip_dd {

// combined_gemm only runs exact {Bin, B, M, K, N} shapes. With Bin == 1 and
// M == 1 the [B, 1, N] output is byte-identical to a 2-D [1, B * N] result, so
// a 2-D GEMM [1, K] x [K, B * N] maps onto these kernels.
struct GemmShape {
  int bin, b, m, k, n;
};

inline constexpr GemmShape kGemmShapes[] = {
    {1, 3, 1, 1024, 64},
    {1, 6, 1, 1024, 64},
};

inline const GemmShape *findGemmShape(int64_t M, int64_t N, int64_t K) {
  for (const GemmShape &s : kGemmShapes) {
    if (M == s.m && K == s.k && N == int64_t(s.b) * s.n) {
      return &s;
    }
  }
  return nullptr;
}

// iconv reads and writes activations padded to a multiple of 8 channels; the
// EP's buffers are unpadded, so other channel counts would overrun them. Bias
// and padding are not expressible in dd_iconv_config_t. The spatial shapes
// iconv supports are only known to DD at run time.
inline bool isConvSupported(int64_t n, int64_t c, int64_t k, int64_t dilation_h,
                            int64_t dilation_w, int64_t group, bool has_bias,
                            bool has_padding) {
  return n == 1 && dilation_h == 1 && dilation_w == 1 && group == 1 &&
         c % 8 == 0 && k % 8 == 0 && !has_bias && !has_padding;
}

} // namespace hip_dd

#endif // HIP_DYNAMIC_DISPATCH_SUPPORT_H
