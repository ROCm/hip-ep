/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GPU-free numeric reference for hip.qlpnormalization:
//   y = Q( L2(DQ(x)) )  with  L2 = RMS * (1/sqrt(N)), epsilon = 0
// Matches the fused wrap's math (not the GPU kernels).

#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, llvm::StringRef what) {
  if (cond)
    llvm::outs() << "[ OK ] " << what << "\n";
  else {
    llvm::errs() << "[FAIL] " << what << "\n";
    ++g_failures;
  }
}

float dequant(uint16_t x, float scale, int64_t zp) {
  return scale * (static_cast<float>(x) - static_cast<float>(zp));
}

uint16_t quant(float x, float scale, int64_t zp) {
  float q = std::round(x / scale) + static_cast<float>(zp);
  if (q < 0.0f)
    q = 0.0f;
  if (q > 65535.0f)
    q = 65535.0f;
  return static_cast<uint16_t>(q);
}

std::vector<uint16_t> fused_l2(const std::vector<uint16_t> &x, int64_t n,
                               float s_in, int64_t z_in, float s_out,
                               int64_t z_out) {
  const int64_t numel = static_cast<int64_t>(x.size());
  std::vector<float> dq(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i)
    dq[static_cast<size_t>(i)] = dequant(x[static_cast<size_t>(i)], s_in, z_in);

  std::vector<uint16_t> y(static_cast<size_t>(numel));
  for (int64_t row = 0; row < numel / n; ++row) {
    float sum_sq = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
      float v = dq[static_cast<size_t>(row * n + j)];
      sum_sq += v * v;
    }
    for (int64_t j = 0; j < n; ++j) {
      float v = dq[static_cast<size_t>(row * n + j)];
      float l2 = (sum_sq == 0.0f) ? 0.0f : v / std::sqrt(sum_sq);
      y[static_cast<size_t>(row * n + j)] = quant(l2, s_out, z_out);
    }
  }
  return y;
}

} // namespace

int main() {
  // One row of 4, asymmetric UINT16 QDQ with different in/out scale+zp.
  const int64_t n = 4;
  const float s_in = 0.1f;
  const int64_t z_in = 100;
  const float s_out = 0.05f;
  const int64_t z_out = 200;
  const std::vector<uint16_t> x = {110, 120, 90, 130};

  std::vector<float> dq(4);
  float sum_sq = 0.0f;
  for (int i = 0; i < 4; ++i) {
    dq[i] = dequant(x[i], s_in, z_in);
    sum_sq += dq[i] * dq[i];
  }
  std::vector<uint16_t> expected(4);
  for (int i = 0; i < 4; ++i)
    expected[i] = quant(dq[i] / std::sqrt(sum_sq), s_out, z_out);

  auto got = fused_l2(x, n, s_in, z_in, s_out, z_out);
  bool match = got == expected;
  check(match, "Q(L2(DQ(x))) matches fused reference (different in/out scale)");
  if (!match) {
    for (int i = 0; i < 4; ++i)
      llvm::errs() << "  [" << i << "] got " << got[i] << " want "
                   << expected[i] << "\n";
  }

  // RMS * 1/sqrt(N) equals L2 on this row.
  float rms = std::sqrt(sum_sq / 4.0f);
  float inv_sqrt_n = 1.0f / std::sqrt(4.0f);
  bool rms_is_l2 = true;
  for (int i = 0; i < 4; ++i) {
    float l2 = dq[i] / std::sqrt(sum_sq);
    float via_rms = dq[i] / rms * inv_sqrt_n;
    if (std::fabs(l2 - via_rms) > 1e-5f)
      rms_is_l2 = false;
  }
  check(rms_is_l2, "L2(x) == RMS(x) * 1/sqrt(N)");

  return g_failures ? 1 : 0;
}
