/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Correctness + optional hipBLASLt bench for hip_gemm (gemm_kernel.hip).
 *
 * Unit-test model: this file is entirely self-contained (no example/common/,
 * no python, no on-disk data). All inputs + the CPU reference are generated
 * in-process. The case list below covers every categorical situation this op
 * has (TA/TB transpose, bias on/off, dtype fp16/fp32) paired with a small set
 * of typical (M,N,K) shapes chosen to exercise distinct kernel dispatch paths
 * (decode/GEMV, WMMA, split-K, K-remainder, prefill-scale) -- NOT the full
 * M/N/K shape space. COVERAGE=1|2|3 (default 3) only thins how many of those
 * typical shapes run; every tier still touches every TA/TB/bias/dtype value.
 *
 *   test_gemm.exe [M] [N] [K] [transA] [transB] [type]
 * type: 0=f16  1=f32
 *
 * HIPDNN_EP_DEBUG=1 logs autotune. HIPDNN_EP_GEMM_AUTOTUNE=0 uses heuristics.
 */

#include "hip_custom_kernels.h"
#include "gemm_autotune.h"

#ifndef HIPDNN_LUT_LINKED_EXTERNALLY
// MODE=autotune (default): no real FlatBuffers LUT is linked. Empty resolve()
// lets the kernel link and fall back to its runtime sweep. MODE=lut links the
// real gemm_autotune.cpp resolver instead (see Makefile), which defines these
// symbols, so this stub must not also define them.
namespace hipdnn_ep {
namespace gemm_autotune {
Result resolve(const Request&, WmmaValidator, GemvValidator, void*) { return {}; }
Stats stats() { return {}; }
}  // namespace gemm_autotune
}  // namespace hipdnn_ep
#else
// One-line C23 #embed of the real LUT .fb -- HIPDNN_LUT_FB is defined by a
// tiny Makefile-generated header (a plain #define, not a data file) so the
// path never has to survive hipcc's Windows -D quoting (which mangles
// embedded quote characters).
#include "lut_fb_path.h"
extern "C" const unsigned char kGemmLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kGemmLutData_size = sizeof(kGemmLutData);
#endif  // HIPDNN_LUT_LINKED_EXTERNALLY

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__,                    \
              hipGetErrorString(e));                                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static float h2f(__half h) {
  uint16_t bits;
  memcpy(&bits, &h, 2);
  uint32_t s = (bits >> 15) & 1;
  uint32_t e = (bits >> 10) & 0x1f;
  uint32_t m = bits & 0x3ff;
  uint32_t f;
  if (e == 0) {
    if (m == 0)
      f = s << 31;
    else {
      e = 1;
      while (!(m & 0x400)) {
        m <<= 1;
        e--;
      }
      m &= 0x3ff;
      f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
    }
  } else if (e == 31)
    f = (s << 31) | (0xffu << 23) | (m << 13);
  else
    f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
  float r;
  memcpy(&r, &f, 4);
  return r;
}

static __half f2h(float x) {
  __half h;
  uint32_t f;
  memcpy(&f, &x, 4);
  uint32_t s = (f >> 31) & 1;
  int e = (int)((f >> 23) & 0xff) - 127 + 15;
  uint32_t m = f & 0x7fffff;
  uint16_t bits;
  if (e <= 0) {
    bits = (uint16_t)(s << 15);
  } else if (e >= 31) {
    bits = (uint16_t)((s << 15) | 0x7c00);
  } else {
    bits = (uint16_t)((s << 15) | (e << 10) | (m >> 13));
  }
  memcpy(&h, &bits, 2);
  return h;
}

// ---- Tiny inline coverage-tier resolver (replaces example/common/coverage.h) ----
// Model: categorical situations (TA/TB, bias on/off, dtype fp16/fp32) are
// covered by every case below regardless of tier; COVERAGE only picks how
// many of the typical shapes (rows) run -- see kTierRows.
static int resolveCoverageTier(int argc, char** argv, int default_tier = 3) {
  int tier = default_tier;
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) tier = std::atoi(env);
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      tier = std::atoi(argv[++i]);
  return tier < 1 ? 1 : (tier > 3 ? 3 : tier);
}

static void cpuGemmF32(const float *A, const float *B, const float *C,
                       float *Y, int M, int N, int K, int ta, int tb,
                       float alpha, float beta, int c0, int c1) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        float a = ta ? A[k * M + i] : A[i * K + k];
        float b = tb ? B[j * K + k] : B[k * N + j];
        acc += a * b;
      }
      float v = alpha * acc;
      if (C && beta != 0.f) {
        int cr = (c0 == 1) ? 0 : i;
        int cc = (c1 == 1) ? 0 : j;
        v += beta * C[cr * c1 + cc];
      }
      Y[i * N + j] = v;
    }
  }
}

static int runCase(int M, int N, int K, int ta, int tb, int type, float alpha,
                   float beta, int c0, int c1, bool bench) {
  const size_t aN = ta ? (size_t)K * M : (size_t)M * K;
  const size_t bN = tb ? (size_t)N * K : (size_t)K * N;
  const size_t yN = (size_t)M * N;
  const bool hasC = beta != 0.f;
  const size_t cElems = hasC ? (size_t)c0 * c1 : 0;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> Ah(aN), Bh(bN), Ch(cElems), Yref(yN);
  for (auto &x : Ah) x = dist(rng);
  for (auto &x : Bh) x = dist(rng);
  for (auto &x : Ch) x = dist(rng);

  cpuGemmF32(Ah.data(), Bh.data(), hasC ? Ch.data() : nullptr, Yref.data(), M,
             N, K, ta, tb, alpha, beta, c0, c1);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int rc = 0;

  if (type == 0) {
    std::vector<__half> A16(aN), B16(bN), C16(cElems), Y16(yN);
    for (size_t i = 0; i < aN; ++i) A16[i] = f2h(Ah[i]);
    for (size_t i = 0; i < bN; ++i) B16[i] = f2h(Bh[i]);
    for (size_t i = 0; i < cElems; ++i) C16[i] = f2h(Ch[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dA, aN * 2));
    HIP_CHECK(hipMalloc(&dB, bN * 2));
    HIP_CHECK(hipMalloc(&dY, yN * 2));
    if (hasC) HIP_CHECK(hipMalloc(&dC, cElems * 2));
    HIP_CHECK(hipMemcpy(dA, A16.data(), aN * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B16.data(), bN * 2, hipMemcpyHostToDevice));
    if (hasC) HIP_CHECK(hipMemcpy(dC, C16.data(), cElems * 2, hipMemcpyHostToDevice));

    rc = hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0, c1);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (rc != 0) {
      fprintf(stderr, "hip_gemm rc=%d\n", rc);
      hipFree(dA); hipFree(dB); hipFree(dY); if (dC) hipFree(dC);
      hipStreamDestroy(stream);
      return 1;
    }
    HIP_CHECK(hipMemcpy(Y16.data(), dY, yN * 2, hipMemcpyDeviceToHost));

    int bad = 0;
    double maxe = 0.0;
    for (size_t i = 0; i < yN; ++i) {
      double e = std::abs((double)h2f(Y16[i]) - (double)Yref[i]);
      double den = std::max(1.0, std::abs((double)Yref[i]));
      double rel = e / den;
      maxe = std::max(maxe, rel);
      if (rel > 5e-2) ++bad;
    }
    printf("f16 M=%d N=%d K=%d ta=%d tb=%d bias=%d  max_rel=%.4g  fail=%d/%zu\n",
           M, N, K, ta, tb, hasC, maxe, bad, yN);

    if (bench) {
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      for (int i = 0; i < 30; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0, c1);
      HIP_CHECK(hipStreamSynchronize(stream));
      const int iters = 20;
      hipEventRecord(e0, stream);
      for (int i = 0; i < iters; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0, c1);
      hipEventRecord(e1, stream);
      HIP_CHECK(hipEventSynchronize(e1));
      float t = 0.f;
      hipEventElapsedTime(&t, e0, e1);
      double ms = t / iters;
      const double flops = 2.0 * M * N * K;
      printf("  hip_gemm  %.3f ms  %.1f GFLOP/s\n", ms, flops / (ms * 1e6));
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }

    hipFree(dA); hipFree(dB); hipFree(dY);
    if (dC) hipFree(dC);
    if (bad) rc = 1;
  } else {
    std::vector<float> Yh(yN);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dA, aN * 4));
    HIP_CHECK(hipMalloc(&dB, bN * 4));
    HIP_CHECK(hipMalloc(&dY, yN * 4));
    if (hasC) HIP_CHECK(hipMalloc(&dC, cElems * 4));
    HIP_CHECK(hipMemcpy(dA, Ah.data(), aN * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, Bh.data(), bN * 4, hipMemcpyHostToDevice));
    if (hasC) HIP_CHECK(hipMemcpy(dC, Ch.data(), cElems * 4, hipMemcpyHostToDevice));
    rc = hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 1, c0, c1);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(Yh.data(), dY, yN * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    double maxe = 0.0;
    for (size_t i = 0; i < yN; ++i) {
      double e = std::abs((double)Yh[i] - (double)Yref[i]);
      double rel = e / std::max(1.0, std::abs((double)Yref[i]));
      maxe = std::max(maxe, rel);
      if (rel > 2e-3) ++bad;
    }
    printf("f32 M=%d N=%d K=%d ta=%d tb=%d  max_rel=%.4g  fail=%d/%zu\n", M, N,
           K, ta, tb, maxe, bad, yN);
    hipFree(dA); hipFree(dB); hipFree(dY);
    if (dC) hipFree(dC);
    if (bad) rc = 1;
  }

  hipStreamDestroy(stream);
  return rc;
}

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  for (size_t i = 0; i < args.size();) {
    if (args[i] == "--coverage" && i + 1 < args.size())
      args.erase(args.begin() + i, args.begin() + i + 2);
    else
      ++i;
  }

  int M = 128, N = 512, K = 256, ta = 0, tb = 1, type = 0;
  if (args.size() >= 3) {
    M = atoi(args[0].c_str());
    N = atoi(args[1].c_str());
    K = atoi(args[2].c_str());
  }
  if (args.size() >= 5) {
    ta = atoi(args[3].c_str());
    tb = atoi(args[4].c_str());
  }
  if (args.size() >= 6) type = atoi(args[5].c_str());

  if (args.size() >= 3)
    return runCase(M, N, K, ta, tb, type, 1.f, 0.f, 1, N, /*bench=*/true);

  // ============================================================
  // Categorical situations (TA/TB transpose x bias on/off x dtype fp16/fp32)
  // paired with a small set of typical (M,N,K) shapes -- each row targets a
  // distinct kernel dispatch path (decode/GEMV, WMMA, split-K, K-remainder,
  // prefill-scale). NOT a full M/N/K sweep. kTierRows below picks a subset
  // per COVERAGE tier; every tier still includes at least one case for every
  // TA/TB/bias/dtype value (see the tier1 comment below).
  // ============================================================
  struct Case {
    int m, n, k, ta, tb, ty;
    float alpha, beta;
    int c0, c1;
  };
  static const Case cases[] = {
      {1, 512, 256, 0, 1, 0, 1.f, 0.f, 1, 512},     /* decode NT */
      {1, 512, 256, 0, 0, 0, 1.f, 0.f, 1, 512},     /* decode NN */
      {8, 256, 128, 0, 1, 0, 1.f, 0.f, 1, 256},     /* small-M WMMA skip */
      {32, 256, 128, 0, 1, 0, 1.f, 0.f, 1, 256},    /* WMMA NT */
      {32, 256, 128, 0, 0, 0, 1.f, 0.f, 1, 256},    /* WMMA NN */
      {64, 128, 48, 0, 1, 0, 1.f, 0.f, 1, 128},     /* K remainder */
      {32, 128, 64, 0, 1, 0, 1.f, 1.f, 1, 128},     /* bias [1,N] */
      {16, 64, 32, 0, 1, 1, 1.f, 0.f, 1, 64},       /* f32 GEMV/tiled */
      {128, 256, 128, 0, 1, 0, 1.f, 0.f, 1, 256},
      {32, 256, 256, 0, 1, 0, 1.f, 0.f, 1, 256},  /* split-K eligible NT */
      {32, 256, 256, 0, 0, 0, 1.f, 0.f, 1, 256},  /* split-K eligible NN */
      {64, 512, 512, 1, 1, 0, 1.f, 0.f, 1, 512},  /* TA=1,TB=1 combo */
      {96, 384, 320, 1, 0, 0, 1.f, 1.f, 96, 1},   /* TA=1, bias [M,1] */
      {256, 4096, 2880, 0, 1, 0, 1.f, 0.f, 1, 4096}, /* prefill-scale NT */
      {512, 2880, 4096, 0, 1, 0, 1.f, 0.f, 1, 2880}, /* prefill-scale, K>N */
      {8, 128, 64, 1, 1, 1, 1.f, 1.f, 8, 1},      /* f32 + TA/TB + bias */
  };
  const size_t kNumCases = sizeof(cases) / sizeof(cases[0]);

  // tier1 (5 rows) still touches TA{0,1}, TB{0,1}, bias{0,1}, dtype{fp16,fp32}.
  static const size_t kTier1[] = {0, 1, 7, 12, 15};
  static const size_t kTier2[] = {0, 1, 3, 4, 5, 6, 7, 11, 12, 13, 15};
  const int tier = resolveCoverageTier(argc, argv);
  const size_t* idxs = tier == 1 ? kTier1 : (tier == 2 ? kTier2 : nullptr);
  const size_t n_idxs = tier == 1 ? sizeof(kTier1) / sizeof(kTier1[0])
                       : tier == 2 ? sizeof(kTier2) / sizeof(kTier2[0])
                                   : kNumCases;
  printf("coverage=%d -> running %zu/%zu gemm cases\n", tier, n_idxs, kNumCases);

  int fail = 0;
  for (size_t i = 0; i < n_idxs; ++i) {
    const Case &c = idxs ? cases[idxs[i]] : cases[i];
    fail |= runCase(c.m, c.n, c.k, c.ta, c.tb, c.ty, c.alpha, c.beta, c.c0,
                    c.c1, /*bench=*/c.m >= 32);
  }
  printf("\n%s (%d/%zu case(s) failing)\n", fail ? "SOME FAILED" : "ALL PASS",
         fail, n_idxs);
  return fail ? 1 : 0;
}
