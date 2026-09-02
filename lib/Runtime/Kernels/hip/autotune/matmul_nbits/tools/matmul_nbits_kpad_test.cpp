/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Single-kernel correctness check for the K%32!=0 int4 WMMA K-padding path.
 *
 * hip_matmul_nbits pads K up to a multiple of 32 and runs the unchanged WMMA
 * kernel for K%32!=0 prefill (e.g. a down_proj with K=4304). This exercises
 * that path against a CPU reference, and a K%32==0 control shape to confirm the
 * ordinary WMMA path is unchanged.
 *
 * GPU required (runs the kernel). Build from the repo root, e.g.:
 *   flatc --cpp -o <gen> lib/Runtime/Kernels/hip/autotune/matmul_nbits/matmul_nbits_autotune.fbs
 *   clang++ -x hip --offload-arch=gfx1151 -O3 -std=c++17 -w \
 *     -I lib/Runtime/Kernels/include \
 *     -I lib/Runtime/Kernels/hip/autotune/matmul_nbits \
 *     -I <gen> -I <flatbuffers include> \
 *     lib/Runtime/Kernels/hip/autotune/matmul_nbits/tools/matmul_nbits_kpad_test.cpp \
 *     lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip \
 *     lib/Runtime/Kernels/hip/autotune/matmul_nbits/matmul_nbits_autotune.cpp \
 *     lib/Runtime/Kernels/hip/autotune/matmul_nbits/tools/empty_lut_data.cpp \
 *     -o matmul_nbits_kpad_test
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int hip_matmul_nbits(
    void* stream, const void* A, const void* B, const void* scales,
    const void* zero_points, const void* bias, void* output,
    int64_t M, int64_t N, int64_t K, int64_t batch, int64_t bits,
    int64_t block_size, int64_t element_size_bytes, int64_t zp_elem_size,
    const void* pre_unpacked_zp_u8, const void* pre_unpacked_zp_fp16);

#define HIP_OK(c)                                                             \
  do {                                                                        \
    hipError_t e_ = (c);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP %s at line %d\n", hipGetErrorString(e_),      \
                   __LINE__);                                                 \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static uint16_t f2h(float f) {
  _Float16 h = (_Float16)f;
  uint16_t o;
  std::memcpy(&o, &h, 2);
  return o;
}
static float h2f(uint16_t o) {
  _Float16 h;
  std::memcpy(&h, &o, 2);
  return (float)h;
}

// One shape: row-major A[M,K] fp16, packed int4 B[N, ngk*(gs/2)], fp16
// scales[N,ngk], symmetric (default zp=8). Returns max abs and max rel error
// of hip_matmul_nbits against a CPU fp32 reference.
static bool check(int M, int N, int K, int gs) {
  const int ngk = (K + gs - 1) / gs;
  const int row_bytes = ngk * (gs / 2);

  std::vector<uint16_t> hA((size_t)M * K), hSc((size_t)N * ngk);
  std::vector<uint8_t> hB((size_t)N * row_bytes);
  for (size_t i = 0; i < hA.size(); ++i)
    hA[i] = f2h(float((int)(i % 23) - 11) * 0.0625f);
  for (size_t i = 0; i < hSc.size(); ++i)
    hSc[i] = f2h(0.01f + 0.001f * float(i % 7));
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = uint8_t(i * 31 + 7);

  auto wof = [&](int n, int k) -> int {
    const int g = k / gs, loc = k % gs;
    uint8_t p = hB[(size_t)n * row_bytes + (size_t)g * (gs / 2) + loc / 2];
    return (loc & 1) ? (p >> 4) : (p & 0xF);
  };

  // CPU reference: out[m,n] = sum_k A[m,k] * (w-8) * scale[n, k/gs]
  std::vector<float> ref((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = h2f(hA[(size_t)m * K + k]);
        float sc = h2f(hSc[(size_t)n * ngk + k / gs]);
        acc += a * (float(wof(n, k)) - 8.0f) * sc;
      }
      ref[(size_t)m * N + n] = acc;
    }

  void *dA, *dB, *dSc, *dOut;
  HIP_OK(hipMalloc(&dA, hA.size() * 2));
  HIP_OK(hipMalloc(&dB, hB.size()));
  HIP_OK(hipMalloc(&dSc, hSc.size() * 2));
  HIP_OK(hipMalloc(&dOut, (size_t)M * N * 2));
  HIP_OK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dSc, hSc.data(), hSc.size() * 2, hipMemcpyHostToDevice));

  int rc = hip_matmul_nbits(nullptr, dA, dB, dSc, /*zp=*/nullptr, nullptr, dOut,
                            M, N, K, 1, 4, gs, 2, 2, nullptr, nullptr);
  HIP_OK(hipDeviceSynchronize());
  if (rc != 0) {
    std::printf("  hip_matmul_nbits rc=%d\n", rc);
    return false;
  }

  std::vector<uint16_t> hOut((size_t)M * N);
  HIP_OK(hipMemcpy(hOut.data(), dOut, hOut.size() * 2, hipMemcpyDeviceToHost));

  // Normalize by the peak reference magnitude, not per element: these outputs
  // are sums of thousands of signed terms and individual entries cancel to near
  // zero, where a per-element relative error explodes on fp16 rounding that is
  // absolutely tiny. A wrong K tail (missing or garbage) shows up as a large
  // ABSOLUTE error instead, which this still catches.
  float max_abs = 0.0f, max_ref = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(h2f(hOut[i]) - ref[i]));
    max_ref = std::max(max_ref, std::fabs(ref[i]));
  }
  hipFree(dA); hipFree(dB); hipFree(dSc); hipFree(dOut);

  const float rel = max_abs / (max_ref + 1e-6f);
  const bool ok = rel < 0.02f;   // 2% of signal scale; fp16 tile-order rounding
  std::printf("  M=%-5d N=%-6d K=%-6d gs=%-3d  max_abs=%.4f peak=%.3f "
              "rel=%.4f  %s\n",
              M, N, K, gs, max_abs, max_ref, rel, ok ? "OK" : "FAIL");
  return ok;
}

int main() {
  hipDeviceProp_t p;
  HIP_OK(hipGetDeviceProperties(&p, 0));
  std::printf("device %s\n\n", p.gcnArchName);

  int fails = 0;
  std::printf("-- K%%32!=0 (new K-padding WMMA path) --\n");
  fails += !check(32,  1152, 4304, 32);   // gemma-4 down_proj, prefill
  fails += !check(128, 1152, 4304, 32);
  fails += !check(17,  1152, 4304, 32);   // M not a multiple of the tile

  std::printf("\n-- K%%32==0 control (unchanged WMMA path) --\n");
  fails += !check(32,  4096, 4096, 32);
  fails += !check(128, 14336, 4096, 128);

  std::printf("\n%s\n", fails ? "FAILED" : "ALL PASSED");
  return fails ? 1 : 0;
}
