/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Self-contained fp16 GEMM kernel-unit-test leaf. This source deliberately
 * instantiates exactly one device/input/output dtype path.
 */
#include "hip_custom_kernels.h"
#include "gemm_autotune.h"

#if !defined(HIPDNN_LUT_LINKED_EXTERNALLY) && !defined(HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS)
namespace hipdnn_ep { namespace gemm_autotune {
Result resolve(const Request&, WmmaValidator, GemvValidator, void*) { return {}; }
Stats stats() { return {}; }
} }
#elif defined(HIPDNN_LUT_LINKED_EXTERNALLY)
#include "lut_fb_path.h"
extern "C" const unsigned char kGemmLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kGemmLutData_size = sizeof(kGemmLutData);
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  std::fprintf(stderr, "HIP %s:%d %s\\n", __FILE__, __LINE__, hipGetErrorString(e)); return 1; } \
} while (0)

static constexpr int kGemmDtype = 0;
static constexpr const char* kDtypeName = "fp16";
static constexpr double kTolerance = 6e-2;
using Elem = __half;

static float toFloat(Elem value) { return __half2float(value); }
static Elem fromFloat(float value) { return __float2half(value); }

static int coverageTier(int argc, char** argv) {
  int tier = 3;
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) tier = std::atoi(env);
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--coverage") == 0) tier = std::atoi(argv[++i]);
  return std::max(1, std::min(3, tier));
}

static void cpuReference(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& c, std::vector<float>& y,
                         int m, int n, int k, int ta, int tb, float alpha,
                         float beta, int c0, int c1) {
  auto rows = [&](int first, int last) {
    for (int i = first; i < last; ++i) for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        const float av = ta ? a[kk * m + i] : a[i * k + kk];
        const float bv = tb ? b[j * k + kk] : b[kk * n + j];
        acc += av * bv;
      }
      float value = alpha * acc;
      if (beta != 0.0f) value += beta * c[(c0 == 1 ? 0 : i) * c1 + (c1 == 1 ? 0 : j)];
      y[i * n + j] = value;
    }
  };
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned threads = std::min<unsigned>(hw ? hw : 4, std::max(1, m));
  if (threads == 1 || m < 8) { rows(0, m); return; }
  std::vector<std::thread> pool;
  const int chunk = (m + static_cast<int>(threads) - 1) / static_cast<int>(threads);
  for (unsigned t = 0; t < threads; ++t) {
    const int first = static_cast<int>(t) * chunk, last = std::min(m, first + chunk);
    if (first < last) pool.emplace_back(rows, first, last);
  }
  for (auto& thread : pool) thread.join();
}

struct Case { int m, n, k, ta, tb, c0, c1; float alpha, beta; const char* phase; };

static int launch(hipStream_t stream, void* a, void* b, void* c, void* y,
                  const Case& test) {
  return hip_gemm(stream, a, b, c, y, test.m, test.n, test.k, test.alpha,
                  test.beta, test.ta, test.tb, kGemmDtype, test.c0, test.c1);
}

static int runCase(const Case& test) {
  const size_t aCount = test.ta ? static_cast<size_t>(test.k) * test.m : static_cast<size_t>(test.m) * test.k;
  const size_t bCount = test.tb ? static_cast<size_t>(test.n) * test.k : static_cast<size_t>(test.k) * test.n;
  const size_t yCount = static_cast<size_t>(test.m) * test.n;
  const size_t cCount = test.beta == 0.0f ? 0 : static_cast<size_t>(test.c0) * test.c1;
  std::mt19937 rng(1234 + test.m + test.n + test.k + test.ta + test.tb);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> af(aCount), bf(bCount), cf(cCount), reference(yCount);
  for (float& value : af) value = dist(rng);
  for (float& value : bf) value = dist(rng);
  for (float& value : cf) value = dist(rng);
  cpuReference(af, bf, cf, reference, test.m, test.n, test.k, test.ta, test.tb,
               test.alpha, test.beta, test.c0, test.c1);

  std::vector<Elem> a(aCount), b(bCount), c(cCount), y(yCount);
  for (size_t i = 0; i < aCount; ++i) a[i] = fromFloat(af[i]);
  for (size_t i = 0; i < bCount; ++i) b[i] = fromFloat(bf[i]);
  for (size_t i = 0; i < cCount; ++i) c[i] = fromFloat(cf[i]);
  void *da = nullptr, *db = nullptr, *dc = nullptr, *dy = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&da, aCount * sizeof(Elem)));
  HIP_CHECK(hipMalloc(&db, bCount * sizeof(Elem)));
  HIP_CHECK(hipMalloc(&dy, yCount * sizeof(Elem)));
  if (cCount) HIP_CHECK(hipMalloc(&dc, cCount * sizeof(Elem)));
  HIP_CHECK(hipMemcpy(da, a.data(), aCount * sizeof(Elem), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, b.data(), bCount * sizeof(Elem), hipMemcpyHostToDevice));
  if (cCount) HIP_CHECK(hipMemcpy(dc, c.data(), cCount * sizeof(Elem), hipMemcpyHostToDevice));
  const int rc = launch(stream, da, db, dc, dy, test);
  HIP_CHECK(hipStreamSynchronize(stream));
  if (rc != 0) { std::fprintf(stderr, "%s %s hip_gemm rc=%d\\n", kDtypeName, test.phase, rc); return 1; }
  HIP_CHECK(hipMemcpy(y.data(), dy, yCount * sizeof(Elem), hipMemcpyDeviceToHost));
  double worst = 0.0; int bad = 0;
  for (size_t i = 0; i < yCount; ++i) {
    const double error = std::abs(static_cast<double>(toFloat(y[i])) - reference[i]);
    const double relative = error / std::max(1.0, std::abs(static_cast<double>(reference[i])));
    worst = std::max(worst, relative);
    if (relative > kTolerance) ++bad;
  }
  hipEvent_t begin, end;
  HIP_CHECK(hipEventCreate(&begin)); HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipEventRecord(begin, stream));
  for (int iter = 0; iter < 3; ++iter) if (launch(stream, da, db, dc, dy, test) != 0) ++bad;
  HIP_CHECK(hipEventRecord(end, stream)); HIP_CHECK(hipEventSynchronize(end));
  float elapsed = 0.0f; HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
  std::printf("%s %s M=%d N=%d K=%d ta=%d tb=%d bias=%d dtype=%d max_rel=%.4g time_ms=%.4f fail=%d/%zu\\n",
              kDtypeName, test.phase, test.m, test.n, test.k, test.ta, test.tb,
              test.beta != 0.0f, kGemmDtype, worst, elapsed / 3.0f, bad, yCount);
  hipEventDestroy(begin); hipEventDestroy(end); hipFree(da); hipFree(db); hipFree(dy);
  if (dc) hipFree(dc); hipStreamDestroy(stream);
  return bad != 0;
}

int main(int argc, char** argv) {
  static const Case cases[] = {
      {1, 2048, 2048, 0, 1, 1, 2048, 1.0f, 0.0f, "GemvNt"},
      {1, 4096, 4096, 0, 0, 1, 4096, 1.0f, 0.25f, "GemvNn-bias"},
      {16, 4096, 11008, 0, 1, 1, 4096, 1.0f, 0.0f, "Wmma"},
      {64, 4096, 14336, 1, 1, 64, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 1024, 4096, 1, 0, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {512, 2048, 2048, 0, 0, 1, 2048, 1.0f, 0.25f, "fixed-layout-bias"},
      {1, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "GemvNt-bias"},
      {1, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.0f, "GemvNn"},
      {16, 1024, 4096, 0, 1, 1, 1024, 0.75f, 0.0f, "Wmma"},
      {16, 2048, 2048, 1, 0, 16, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {64, 4096, 4096, 0, 1, 1, 4096, 1.0f, 0.25f, "Wmma-bias"},
      {64, 11008, 4096, 1, 1, 64, 1, 1.0f, 0.0f, "fixed-layout"},
      {128, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 1024, 4096, 0, 1, 1, 1024, 1.0f, 0.0f, "Wmma"},
      {512, 4096, 4096, 1, 1, 512, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {512, 11008, 4096, 0, 0, 1, 11008, 1.0f, 0.0f, "fixed-layout"},
      {1024, 2048, 2048, 0, 1, 1, 2048, 1.0f, 0.25f, "Wmma-bias"},
      {1024, 4096, 4096, 1, 0, 1024, 1, 1.0f, 0.0f, "fixed-layout"},
      {16, 4096, 14336, 1, 1, 16, 1, 1.0f, 0.0f, "fixed-layout"},
      {64, 1024, 4096, 0, 0, 1, 1024, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 2048, 2048, 1, 0, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {512, 4096, 14336, 0, 1, 1, 4096, 1.0f, 0.25f, "Wmma-bias"},
      {1024, 1024, 4096, 0, 0, 1, 1024, 1.0f, 0.0f, "fixed-layout"},
      {16, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "Wmma-bias"},
      {64, 2048, 2048, 1, 1, 64, 1, 1.0f, 0.0f, "fixed-layout"},
      {128, 4096, 4096, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
      {512, 1024, 4096, 1, 0, 512, 1, 1.0f, 0.0f, "fixed-layout"},
      {1024, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "Wmma-bias"},
      {128, 11008, 4096, 1, 1, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {64, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
  };
  std::vector<int> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--coverage") == 0) { ++i; continue; }
    positional.push_back(std::atoi(argv[i]));
  }
  if (positional.size() >= 5) {
    const int ta = positional[3], tb = positional[4];
    const Case custom{positional[0], positional[1], positional[2], ta, tb,
                      ta ? positional[0] : 1, ta ? 1 : positional[1], 1.0f, 0.0f, "custom"};
    return runCase(custom);
  }
  const int tier = coverageTier(argc, argv);
  static const size_t kTier1[] = {0, 1, 2, 3, 4, 5};
  static const size_t kTier2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  const size_t total = sizeof(cases) / sizeof(cases[0]);
  const size_t* indexes = tier == 1 ? kTier1 : (tier == 2 ? kTier2 : nullptr);
  const size_t count = tier == 1 ? sizeof(kTier1) / sizeof(kTier1[0]) :
                       tier == 2 ? sizeof(kTier2) / sizeof(kTier2[0]) : total;
  std::printf("%s coverage=%d -> running %zu/%zu comprehensive cases (dtype=%d)\\n",
              kDtypeName, tier, count, total, kGemmDtype);
  int failures = 0;
  for (size_t i = 0; i < count; ++i) failures |= runCase(cases[indexes ? indexes[i] : i]);
  std::printf("%s %s (%d/%zu case(s) failing)\\n", kDtypeName,
              failures ? "SOME FAILED" : "ALL PASS", failures, count);
  return failures ? 1 : 0;
}
