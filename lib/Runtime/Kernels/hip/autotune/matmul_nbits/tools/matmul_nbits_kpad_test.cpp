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
 * Zero-point coverage matters as much as the shape here. The K-padding path
 * reads zp_for_fp16_paths, which degrades to the packed-nibble zero_points when
 * the caller supplies no pre-unpacked fp16 buffer. An earlier version of this
 * test only ever passed zero_points=nullptr, so has_zp was false, the pointer
 * was never dereferenced, and it reported the path correct while every real
 * asymmetric model (gemma-4's SigLIP, K=4304) produced garbage. Every shape is
 * now run in three zero-point modes -- see Zp below.
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

/* How the caller presents zero_points, mirroring what the runtime wrapper
 * (lib/Runtime/real/matmul_nbits.cpp) hands the kernel:
 *
 *   Symmetric  - no zero_points at all; the kernel's implicit zp of 8.
 *   AsymU8Only - packed nibbles + pre_unpacked_zp_u8, no fp16 buffer. This is
 *                what the wrapper used to pass for K%32!=0, and the case the
 *                K-padding path silently mis-read. The kernel must now decline
 *                the fp16-zp paths and land on one that reads the uint8 buffer.
 *   AsymFp16   - packed nibbles + both pre-unpacked buffers, i.e. what the
 *                wrapper passes now. This is the case that must stay on WMMA.
 *
 * Both asym modes must agree with the same reference: which kernel serves the
 * shape is a performance decision, never a numerical one. */
enum class Zp { Symmetric, AsymU8Only, AsymFp16 };

static const char* zpName(Zp z) {
  switch (z) {
    case Zp::Symmetric:  return "sym";
    case Zp::AsymU8Only: return "asym u8-only";
    default:             return "asym +fp16";
  }
}

// One shape: row-major A[M,K] fp16, packed int4 B[N, ngk*(gs/2)], fp16
// scales[N,ngk], and zero-points per `mode`. Compares hip_matmul_nbits against
// a CPU fp32 reference.
static bool check(int M, int N, int K, int gs, Zp mode) {
  const int ngk = (K + gs - 1) / gs;
  const int row_bytes = ngk * (gs / 2);
  const bool asym = (mode != Zp::Symmetric);

  std::vector<uint16_t> hA((size_t)M * K), hSc((size_t)N * ngk);
  std::vector<uint8_t> hB((size_t)N * row_bytes);
  for (size_t i = 0; i < hA.size(); ++i)
    hA[i] = f2h(float((int)(i % 23) - 11) * 0.0625f);
  for (size_t i = 0; i < hSc.size(); ++i)
    hSc[i] = f2h(0.01f + 0.001f * float(i % 7));
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = uint8_t(i * 31 + 7);

  // Asymmetric zero-points, in all three layouts the kernel can be handed.
  // Packed is [N, (ngk+1)/2] with the low nibble the even group and the high
  // nibble the odd one; the u8 and fp16 forms are one value per group [N, ngk].
  // Values deliberately stray from 8 so a path that ignores them, or reads the
  // packed bytes as fp16, cannot accidentally match.
  const int packed_cols = (ngk + 1) / 2;
  std::vector<uint8_t> hZpPacked(asym ? (size_t)N * packed_cols : 0);
  std::vector<uint8_t> hZpU8(asym ? (size_t)N * ngk : 0);
  std::vector<uint16_t> hZpFp16(asym ? (size_t)N * ngk : 0);
  auto zp_of = [&](int n, int g) -> int {
    if (!asym) return 8;
    return (int)hZpU8[(size_t)n * ngk + g];
  };
  if (asym) {
    for (int n = 0; n < N; ++n)
      for (int g = 0; g < ngk; ++g) {
        const int v = (n * 7 + g * 5 + 1) % 16;
        hZpU8[(size_t)n * ngk + g] = (uint8_t)v;
        hZpFp16[(size_t)n * ngk + g] = f2h((float)v);
        uint8_t& p = hZpPacked[(size_t)n * packed_cols + g / 2];
        if (g % 2 == 0)
          p = (uint8_t)((p & 0xF0) | v);
        else
          p = (uint8_t)((p & 0x0F) | (v << 4));
      }
  }

  auto wof = [&](int n, int k) -> int {
    const int g = k / gs, loc = k % gs;
    uint8_t p = hB[(size_t)n * row_bytes + (size_t)g * (gs / 2) + loc / 2];
    return (loc & 1) ? (p >> 4) : (p & 0xF);
  };

  // CPU reference: out[m,n] = sum_k A[m,k] * (w - zp[n, k/gs]) * scale[n, k/gs].
  // Dequantizing each B row once and reusing it across all M keeps this
  // affordable at the full prefill M (the M=2268 case below is ~11e9 MACs).
  std::vector<float> ref((size_t)M * N, 0.0f);
  std::vector<float> brow((size_t)K);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      const int g = k / gs;
      brow[(size_t)k] =
          (float(wof(n, k)) - float(zp_of(n, g))) * h2f(hSc[(size_t)n * ngk + g]);
    }
    for (int m = 0; m < M; ++m) {
      const uint16_t* arow = &hA[(size_t)m * K];
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += h2f(arow[k]) * brow[(size_t)k];
      ref[(size_t)m * N + n] = acc;
    }
  }

  void *dA, *dB, *dSc, *dOut;
  HIP_OK(hipMalloc(&dA, hA.size() * 2));
  HIP_OK(hipMalloc(&dB, hB.size()));
  HIP_OK(hipMalloc(&dSc, hSc.size() * 2));
  HIP_OK(hipMalloc(&dOut, (size_t)M * N * 2));
  HIP_OK(hipMemcpy(dA, hA.data(), hA.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dSc, hSc.data(), hSc.size() * 2, hipMemcpyHostToDevice));

  void *dZpPacked = nullptr, *dZpU8 = nullptr, *dZpFp16 = nullptr;
  if (asym) {
    HIP_OK(hipMalloc(&dZpPacked, hZpPacked.size()));
    HIP_OK(hipMemcpy(dZpPacked, hZpPacked.data(), hZpPacked.size(),
                     hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&dZpU8, hZpU8.size()));
    HIP_OK(hipMemcpy(dZpU8, hZpU8.data(), hZpU8.size(), hipMemcpyHostToDevice));
    if (mode == Zp::AsymFp16) {
      HIP_OK(hipMalloc(&dZpFp16, hZpFp16.size() * 2));
      HIP_OK(hipMemcpy(dZpFp16, hZpFp16.data(), hZpFp16.size() * 2,
                       hipMemcpyHostToDevice));
    }
  }

  int rc = hip_matmul_nbits(nullptr, dA, dB, dSc, dZpPacked, nullptr, dOut, M, N,
                            K, 1, 4, gs, 2, asym ? 1 : 2, dZpU8, dZpFp16);
  HIP_OK(hipDeviceSynchronize());
  if (rc != 0) {
    std::printf("  M=%-5d N=%-6d K=%-6d gs=%-3d %-13s hip_matmul_nbits rc=%d\n",
                M, N, K, gs, zpName(mode), rc);
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
  if (dZpPacked) hipFree(dZpPacked);
  if (dZpU8) hipFree(dZpU8);
  if (dZpFp16) hipFree(dZpFp16);

  const float rel = max_abs / (max_ref + 1e-6f);
  const bool ok = rel < 0.02f;   // 2% of signal scale; fp16 tile-order rounding
  std::printf("  M=%-5d N=%-6d K=%-6d gs=%-3d %-13s max_abs=%.4f peak=%.3f "
              "rel=%.4f  %s\n",
              M, N, K, gs, zpName(mode), max_abs, max_ref, rel,
              ok ? "OK" : "FAIL");
  return ok;
}

// Every shape in all three zero-point modes. The asym modes are the point:
// AsymU8Only is the combination that used to reach the K-padding path with the
// packed nibbles reinterpreted as fp16.
static int checkAll(int M, int N, int K, int gs) {
  int fails = 0;
  fails += !check(M, N, K, gs, Zp::Symmetric);
  fails += !check(M, N, K, gs, Zp::AsymU8Only);
  fails += !check(M, N, K, gs, Zp::AsymFp16);
  return fails;
}

int main() {
  hipDeviceProp_t p;
  HIP_OK(hipGetDeviceProperties(&p, 0));
  std::printf("device %s\n\n", p.gcnArchName);

  int fails = 0;
  std::printf("-- K%%32!=0 (new K-padding WMMA path) --\n");
  fails += checkAll(32,  1152, 4304, 32);   // gemma-4 down_proj, prefill
  fails += checkAll(128, 1152, 4304, 32);
  fails += checkAll(17,  1152, 4304, 32);   // M not a multiple of the tile

  // The shipped geometry: gemma-4's SigLIP down_proj at a 2268-token image
  // prefill, the shape whose asym output was garbage. Slowest case here by far
  // (the reference is ~11e9 MACs), and the one that actually reproduced the bug
  // end to end.
  std::printf("\n-- shipped SigLIP down_proj geometry --\n");
  fails += checkAll(2268, 1152, 4304, 32);

  std::printf("\n-- K%%32==0 control (unchanged WMMA path) --\n");
  fails += checkAll(32,  4096, 4096, 32);
  fails += checkAll(128, 14336, 4096, 128);

  std::printf("\n%s\n", fails ? "FAILED" : "ALL PASSED");
  return fails ? 1 : 0;
}
