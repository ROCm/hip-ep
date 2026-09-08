/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Correctness + optional hipBLASLt bench for hip_gemm (gemm_kernel.hip).
 *
 *   test_gemm.exe [M] [N] [K] [transA] [transB] [type]
 * type: 0=f16  1=f32  3=bf16
 *
 * HIPDNN_EP_DEBUG=1 logs autotune. HIPDNN_EP_GEMM_AUTOTUNE=0 uses heuristics.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef HIPDNN_GEMM_BENCH_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

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
  for (auto &x : Ah)
    x = dist(rng);
  for (auto &x : Bh)
    x = dist(rng);
  for (auto &x : Ch)
    x = dist(rng);

  cpuGemmF32(Ah.data(), Bh.data(), hasC ? Ch.data() : nullptr, Yref.data(), M,
             N, K, ta, tb, alpha, beta, c0, c1);

#ifdef HIPDNN_GEMM_BENCH_HIPBLASLT
  /* Same wrapping as wrap_gemm: hipBLASLt A=B, B=A, D=[N,M] ld=N.
   * Enumerate + time algos (getAllAlgos) instead of algo=nullptr. */
  auto benchHipBlasLtF16 = [&](hipStream_t stream, void *dA, void *dB, void *dY,
                               int iters, double flops) -> int {
    hipblasLtHandle_t h = nullptr;
    if (hipblasLtCreate(&h) != HIPBLAS_STATUS_SUCCESS)
      return 1;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    if (tb) {
      hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, K, N, K);
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
    } else {
      hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, N, K, N);
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN,
                                      sizeof(opN));
    }
    if (ta) {
      hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, M, K, M);
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT,
                                      sizeof(opT));
    } else {
      hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, M, K);
      hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
    }
    hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, N, M, N);

    float a1 = 1.f, b0 = 0.f;
    std::vector<hipblasLtMatmulHeuristicResult_t> all;
    hipblasOperation_t gaA = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t gaB = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblaslt_ext::getAllAlgos(h, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, gaA,
                               gaB, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F,
                               HIPBLAS_COMPUTE_32F, all);

    struct Cand {
      hipblasLtMatmulAlgo_t algo;
      size_t ws;
    };
    std::vector<Cand> cands;
    size_t maxws = 0;
    for (auto &r : all) {
      if ((int)cands.size() >= 16)
        break;
      size_t need = 0;
      if (hipblaslt_ext::matmulIsAlgoSupported(h, desc, &a1, la, lb, &b0, lc,
                                               lc, r.algo,
                                               need) != HIPBLAS_STATUS_SUCCESS)
        continue;
      cands.push_back({r.algo, need});
      if (need > maxws)
        maxws = need;
    }

    void *ws = nullptr;
    if (maxws > 0)
      hipMalloc(&ws, maxws);

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    int best = -1;
    float best_ms = 1e30f;
    auto run = [&](int i) -> hipblasStatus_t {
      void *wsp = cands[i].ws ? ws : nullptr;
      return hipblasLtMatmul(h, desc, &a1, dB, la, dA, lb, &b0, dY, lc, dY, lc,
                             &cands[i].algo, wsp, cands[i].ws, stream);
    };

    if (cands.empty()) {
      for (int i = 0; i < 3; ++i)
        hipblasLtMatmul(h, desc, &a1, dB, la, dA, lb, &b0, dY, lc, dY, lc,
                        nullptr, nullptr, 0, stream);
      hipStreamSynchronize(stream);
      hipEventRecord(e0, stream);
      for (int i = 0; i < iters; ++i)
        hipblasLtMatmul(h, desc, &a1, dB, la, dA, lb, &b0, dY, lc, dY, lc,
                        nullptr, nullptr, 0, stream);
      hipEventRecord(e1, stream);
      hipEventSynchronize(e1);
      float t = 0.f;
      hipEventElapsedTime(&t, e0, e1);
      printf("  hipBLASLt %.3f ms  %.1f GFLOP/s  (algo=nullptr, %d algos)\n",
             t / iters, flops / ((t / iters) * 1e6), (int)all.size());
    } else {
      for (int i = 0; i < (int)cands.size(); ++i) {
        if (run(i) != HIPBLAS_STATUS_SUCCESS)
          continue;
        hipEventRecord(e0, stream);
        for (int r = 0; r < 3; ++r)
          run(i);
        hipEventRecord(e1, stream);
        if (hipEventSynchronize(e1) != hipSuccess)
          continue;
        float ms = 0.f;
        hipEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) {
          best_ms = ms;
          best = i;
        }
      }
      if (best >= 0) {
        run(best);
        hipStreamSynchronize(stream);
        hipEventRecord(e0, stream);
        for (int i = 0; i < iters; ++i)
          run(best);
        hipEventRecord(e1, stream);
        hipEventSynchronize(e1);
        float t = 0.f;
        hipEventElapsedTime(&t, e0, e1);
        printf("  hipBLASLt %.3f ms  %.1f GFLOP/s  (best of %d algos, ws=%zu)\n",
               t / iters, flops / ((t / iters) * 1e6), (int)cands.size(),
               cands[best].ws);
      } else {
        printf("  hipBLASLt: no runnable algo\n");
      }
    }

    hipEventDestroy(e0);
    hipEventDestroy(e1);
    if (ws)
      hipFree(ws);
    hipblasLtMatrixLayoutDestroy(la);
    hipblasLtMatrixLayoutDestroy(lb);
    hipblasLtMatrixLayoutDestroy(lc);
    hipblasLtMatmulDescDestroy(desc);
    hipblasLtDestroy(h);
    return 0;
  };
#endif

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int rc = 0;
  double ms = 0.0;

  if (type == 0) {
    std::vector<__half> A16(aN), B16(bN), C16(cElems), Y16(yN);
    for (size_t i = 0; i < aN; ++i)
      A16[i] = f2h(Ah[i]);
    for (size_t i = 0; i < bN; ++i)
      B16[i] = f2h(Bh[i]);
    for (size_t i = 0; i < cElems; ++i)
      C16[i] = f2h(Ch[i]);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dA, aN * 2));
    HIP_CHECK(hipMalloc(&dB, bN * 2));
    HIP_CHECK(hipMalloc(&dY, yN * 2));
    if (hasC)
      HIP_CHECK(hipMalloc(&dC, cElems * 2));
    HIP_CHECK(hipMemcpy(dA, A16.data(), aN * 2, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B16.data(), bN * 2, hipMemcpyHostToDevice));
    if (hasC)
      HIP_CHECK(hipMemcpy(dC, C16.data(), cElems * 2, hipMemcpyHostToDevice));

    rc = hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0,
                  c1);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (rc != 0) {
      fprintf(stderr, "hip_gemm rc=%d\n", rc);
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
      if (rel > 5e-2)
        ++bad;
    }
    printf("f16 M=%d N=%d K=%d ta=%d tb=%d bias=%d  max_rel=%.4g  fail=%d/%zu\n",
           M, N, K, ta, tb, hasC, maxe, bad, yN);

    if (bench) {
      hipEvent_t e0, e1;
      hipEventCreate(&e0);
      hipEventCreate(&e1);
      for (int i = 0; i < 30; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0,
                 c1);
      HIP_CHECK(hipStreamSynchronize(stream));
      const int iters = 20;
      hipEventRecord(e0, stream);
      for (int i = 0; i < iters; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0,
                 c1);
      hipEventRecord(e1, stream);
      HIP_CHECK(hipEventSynchronize(e1));
      float t = 0.f;
      hipEventElapsedTime(&t, e0, e1);
      ms = t / iters;
      const double flops = 2.0 * M * N * K;
      printf("  hip_gemm  %.3f ms  %.1f GFLOP/s\n", ms, flops / (ms * 1e6));
#ifdef HIPDNN_GEMM_BENCH_HIPBLASLT
      benchHipBlasLtF16(stream, dA, dB, dY, iters, flops);
      for (int i = 0; i < 10; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0,
                 c1);
      HIP_CHECK(hipStreamSynchronize(stream));
      hipEventRecord(e0, stream);
      for (int i = 0; i < iters; ++i)
        hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 0, c0,
                 c1);
      hipEventRecord(e1, stream);
      HIP_CHECK(hipEventSynchronize(e1));
      hipEventElapsedTime(&t, e0, e1);
      printf("  hip_gemm  %.3f ms  %.1f GFLOP/s  (after hipBLASLt warmup)\n",
             t / iters, flops / ((t / iters) * 1e6));
#endif
      hipEventDestroy(e0);
      hipEventDestroy(e1);
    }

    hipFree(dA);
    hipFree(dB);
    hipFree(dY);
    if (dC)
      hipFree(dC);
    if (bad)
      rc = 1;
  } else if (type == 1) {
    std::vector<float> Yh(yN);
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dA, aN * 4));
    HIP_CHECK(hipMalloc(&dB, bN * 4));
    HIP_CHECK(hipMalloc(&dY, yN * 4));
    if (hasC)
      HIP_CHECK(hipMalloc(&dC, cElems * 4));
    HIP_CHECK(hipMemcpy(dA, Ah.data(), aN * 4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, Bh.data(), bN * 4, hipMemcpyHostToDevice));
    if (hasC)
      HIP_CHECK(hipMemcpy(dC, Ch.data(), cElems * 4, hipMemcpyHostToDevice));
    rc = hip_gemm(stream, dA, dB, dC, dY, M, N, K, alpha, beta, ta, tb, 1, c0,
                  c1);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(Yh.data(), dY, yN * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    double maxe = 0.0;
    for (size_t i = 0; i < yN; ++i) {
      double e = std::abs((double)Yh[i] - (double)Yref[i]);
      double rel = e / std::max(1.0, std::abs((double)Yref[i]));
      maxe = std::max(maxe, rel);
      if (rel > 2e-3)
        ++bad;
    }
    printf("f32 M=%d N=%d K=%d ta=%d tb=%d  max_rel=%.4g  fail=%d/%zu\n", M, N,
           K, ta, tb, maxe, bad, yN);
    hipFree(dA);
    hipFree(dB);
    hipFree(dY);
    if (dC)
      hipFree(dC);
    if (bad)
      rc = 1;
  } else {
    fprintf(stderr, "type %d not in this test binary\n", type);
    rc = 1;
  }

  hipStreamDestroy(stream);
  return rc;
}

int main(int argc, char **argv) {
  int M = 128, N = 512, K = 256, ta = 0, tb = 1, type = 0;
  if (argc >= 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  if (argc >= 6) {
    ta = atoi(argv[4]);
    tb = atoi(argv[5]);
  }
  if (argc >= 7)
    type = atoi(argv[6]);

  bool bench = true;
  int fail = 0;
  if (argc >= 4) {
    fail |= runCase(M, N, K, ta, tb, type, 1.f, 0.f, 1, N, bench);
    return fail;
  }

  struct Case {
    int m, n, k, ta, tb, ty;
    float alpha, beta;
    int c0, c1;
  };
  const Case cases[] = {
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
  };
  for (const auto &c : cases)
    fail |= runCase(c.m, c.n, c.k, c.ta, c.tb, c.ty, c.alpha, c.beta, c.c0,
                    c.c1, /*bench=*/c.m >= 32);
  return fail ? 1 : 0;
}
